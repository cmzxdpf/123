#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 6: DFM Facial Structure Restoration & Geometry-Constrained Final Merge."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

LOG = logging.getLogger("run_dfm_stage")


def frame_name(frame_id: int, suffix: str = ".png") -> str:
    return f"{frame_id:06d}{suffix}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def atomic_imwrite(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise IOError(f"Failed to write image: {tmp}")
    buf.tofile(str(tmp))
    os.replace(tmp, path)


def load_jsonl(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    if not path or not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                records[int(item["frame_id"])] = item
            except Exception as exc:
                LOG.warning("Malformed JSONL line %d in %s: %s", line_no, path, exc)
    return records


def read_mask(path: Path, width: int, height: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size > 0 else None
    except Exception:
        mask = None
    if mask is None:
        return None
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def read_soft_mask(path: Path, width: int, height: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size > 0 else None
    except Exception:
        mask = None
    if mask is None:
        return None
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return mask.astype(np.float32) / 255.0


def robust_median(values: np.ndarray, q0: float = 10.0, q1: float = 90.0) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    lo, hi = np.percentile(arr, [q0, q1])
    trimmed = arr[(arr >= lo) & (arr <= hi)]
    if trimmed.size == 0:
        trimmed = arr
    return float(np.median(trimmed))


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class FrameSource:
    def __init__(self, video_path: Optional[Path], frames_dir: Optional[Path]) -> None:
        if video_path is None and frames_dir is None:
            raise ValueError("Either video_path or frames_dir must be provided.")
        self.video_path = video_path
        self.frames_dir = frames_dir
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_pos: int = -1
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 30.0
        self.total_frames: int = 0

        if self.frames_dir is not None and self.frames_dir.is_dir():
            files = sorted(p for p in self.frames_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
            if not files:
                raise FileNotFoundError(f"No frame images found in {self.frames_dir}")
            sample = cv2.imdecode(np.fromfile(str(files[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
            if sample is None:
                raise RuntimeError(f"Cannot read sample frame: {files[0]}")
            self.height, self.width = sample.shape[:2]
            self.total_frames = len(files)
        elif self.video_path is not None:
            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            raise FileNotFoundError("FrameSource input does not exist.")

    def read_frame(self, frame_id: int) -> Optional[np.ndarray]:
        if self.frames_dir is not None and self.frames_dir.is_dir():
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = self.frames_dir / frame_name(frame_id, ext)
                if candidate.exists():
                    return cv2.imdecode(np.fromfile(str(candidate), dtype=np.uint8), cv2.IMREAD_COLOR)
            return None
        assert self.cap is not None
        if self.current_pos != frame_id:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.current_pos = -1
            return None
        self.current_pos = frame_id + 1
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class DFMOnnxRunner:
    """Direct ONNX Runtime wrapper for .dfm models with 5-point Umeyama similarity alignment."""

    # Canonical 5-point template in normalized [0, 1] DFM face coordinates:
    # left_eye, right_eye, nose_tip, left_mouth, right_mouth
    CANONICAL_5PTS = np.array([
        [0.315, 0.365],
        [0.685, 0.365],
        [0.500, 0.540],
        [0.345, 0.745],
        [0.655, 0.745],
    ], dtype=np.float32)

    def __init__(self, model_path: Path, device: str = "cuda:0") -> None:
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            device_id = int(device.split(":")[1]) if ":" in device else 0
            providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.in_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape

        self.nchw = False
        self.res = 256
        if len(in_shape) == 4:
            if in_shape[1] in (3, 4):
                self.nchw = True
                if isinstance(in_shape[2], int) and in_shape[2] > 0:
                    self.res = int(in_shape[2])
            elif in_shape[3] in (3, 4):
                self.nchw = False
                if isinstance(in_shape[1], int) and in_shape[1] > 0:
                    self.res = int(in_shape[1])

    def _compute_affine(self, theo_mask: np.ndarray, landmarks_path: Optional[Path]) -> Optional[np.ndarray]:
        if landmarks_path is not None and landmarks_path.exists():
            try:
                pts = np.load(str(landmarks_path))
                if pts.ndim == 2 and pts.shape[0] >= 68 and pts.shape[1] >= 2:
                    src_5 = np.array([
                        pts[36:42, :2].mean(axis=0),
                        pts[42:48, :2].mean(axis=0),
                        pts[30, :2],
                        pts[48, :2],
                        pts[54, :2],
                    ], dtype=np.float32)
                    dst_5 = self.CANONICAL_5PTS * float(self.res)
                    m_mat, _ = cv2.estimateAffinePartial2D(src_5, dst_5, method=cv2.LMEDS)
                    if m_mat is not None and np.all(np.isfinite(m_mat)):
                        return m_mat.astype(np.float32)
            except Exception:
                pass

        bbox = mask_bbox(theo_mask)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        half = max(max(x2 - x1, y2 - y1) * 0.625, 32.0)
        src_tri = np.array([[cx - half, cy - half], [cx + half, cy - half], [cx - half, cy + half]], dtype=np.float32)
        dst_tri = np.array([[0.0, 0.0], [float(self.res - 1), 0.0], [0.0, float(self.res - 1)]], dtype=np.float32)
        return cv2.getAffineTransform(src_tri, dst_tri).astype(np.float32)

    def infer_aligned(
        self, frame_bgr: np.ndarray, theo_mask: np.ndarray, landmarks_path: Optional[Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        m_mat = self._compute_affine(theo_mask, landmarks_path)
        if m_mat is None:
            return frame_bgr.copy(), np.zeros((h, w), dtype=np.float32)
        m_inv = cv2.invertAffineTransform(m_mat)

        aligned = cv2.warpAffine(frame_bgr, m_mat, (self.res, self.res), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blob = aligned.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...] if self.nchw else blob[None, ...]

        ort_outs = self.session.run(None, {self.in_name: blob})
        face_out: Optional[np.ndarray] = None
        mask_out: Optional[np.ndarray] = None

        for arr in ort_outs:
            a = np.asarray(arr)
            if a.ndim == 4:
                a = np.transpose(a[0], (1, 2, 0)) if self.nchw else a[0]
            if a.ndim == 3 and a.shape[2] >= 3 and face_out is None:
                face_out = np.clip(a[:, :, :3] * 255.0, 0.0, 255.0).astype(np.uint8)
                if a.shape[2] == 4:
                    mask_out = np.clip(a[:, :, 3], 0.0, 1.0).astype(np.float32)
            elif a.ndim == 3 and a.shape[2] == 1 and mask_out is None:
                mask_out = np.clip(a[:, :, 0], 0.0, 1.0).astype(np.float32)
            elif a.ndim == 2 and mask_out is None:
                mask_out = np.clip(a, 0.0, 1.0).astype(np.float32)

        if face_out is None:
            return frame_bgr.copy(), np.zeros((h, w), dtype=np.float32)
        if mask_out is None:
            mask_out = np.ones((self.res, self.res), dtype=np.float32)

        warped_face = cv2.warpAffine(face_out, m_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped_mask = cv2.warpAffine(mask_out, m_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        return warped_face, np.clip(warped_mask, 0.0, 1.0)


@dataclass
class TemporalColorState:
    dl: float = 0.0
    da: float = 0.0
    db: float = 0.0
    initialized: bool = False

    def update(self, measured: Sequence[float], alpha: float, max_step_l: float, max_step_ab: float) -> None:
        tl, ta, tb = map(float, measured[:3])
        if not self.initialized:
            self.dl, self.da, self.db = tl, ta, tb
            self.initialized = True
            return
        self.dl += alpha * clamp(tl - self.dl, -max_step_l, max_step_l)
        self.da += alpha * clamp(ta - self.da, -max_step_ab, max_step_ab)
        self.db += alpha * clamp(tb - self.db, -max_step_ab, max_step_ab)

    def restore(self, rec: Mapping[str, Any]) -> None:
        if "dfm_dl" in rec:
            self.dl = float(rec.get("dfm_dl", 0.0))
            self.da = float(rec.get("dfm_da", 0.0))
            self.db = float(rec.get("dfm_db", 0.0))
            self.initialized = True


class DFMGeometryComposer:
    def __init__(
        self,
        merge_scope: str = "mask_only",
        expand_mask_px: int = 18,
        feather_radius: int = 10,
        temporal_alpha: float = 0.20,
        max_l_shift: float = 24.0,
        max_ab_shift: float = 10.0,
        max_step_l: float = 5.0,
        max_step_ab: float = 2.0,
        min_visible_pixels: int = 400,
        color_match_strength: float = 0.85,
    ) -> None:
        self.merge_scope = merge_scope
        self.expand_mask_px = max(0, int(expand_mask_px))
        self.feather_radius = max(1, int(feather_radius))
        self.temporal_alpha = clamp(temporal_alpha, 0.01, 1.0)
        self.max_l_shift = float(max_l_shift)
        self.max_ab_shift = float(max_ab_shift)
        self.max_step_l = float(max_step_l)
        self.max_step_ab = float(max_step_ab)
        self.min_visible_pixels = max(50, int(min_visible_pixels))
        self.color_match_strength = clamp(color_match_strength, 0.0, 1.0)
        self.state = TemporalColorState()

    def _build_alpha(
        self, sec_mask: np.ndarray, theo_mask: np.ndarray, occ_mask: np.ndarray, dfm_soft_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        theo_b = theo_mask > 0
        occ_b = occ_mask > 0
        if self.merge_scope == "mask_only":
            base_mask = (sec_mask > 0).astype(np.uint8)
            if self.expand_mask_px > 0:
                k = self.expand_mask_px * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                base_mask = cv2.dilate(base_mask, kernel, iterations=1)
            allowed = (base_mask > 0) & theo_b & (~occ_b)
        else:
            allowed = theo_b & (~occ_b)

        if not np.any(allowed):
            return np.zeros(theo_mask.shape, dtype=np.float32)

        dist = cv2.distanceTransform(allowed.astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(dist / float(self.feather_radius), 0.0, 1.0).astype(np.float32)
        if dfm_soft_mask is not None:
            alpha = alpha * np.clip(dfm_soft_mask, 0.0, 1.0)
        alpha[~theo_b] = 0.0
        alpha[occ_b] = 0.0
        return alpha

    def _match_color_in_bbox(
        self, pre_dfm_bgr: np.ndarray, dfm_bgr: np.ndarray, visible_face: np.ndarray,
        occ_mask: np.ndarray, theo_mask: np.ndarray, bbox: Tuple[int, int, int, int], geometry_conf: float
    ) -> Tuple[np.ndarray, bool, int]:
        x1, y1, x2, y2 = bbox
        vis_valid = (visible_face > 0) & (theo_mask > 0) & ~(occ_mask > 0)
        vis_local = vis_valid[y1:y2, x1:x2]
        valid_count = int(np.count_nonzero(vis_local))

        pre_lab = cv2.cvtColor(pre_dfm_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.float32)
        dfm_lab = cv2.cvtColor(dfm_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.float32)

        updated = False
        if valid_count >= self.min_visible_pixels and geometry_conf >= 0.40:
            ys, xs = np.nonzero(vis_local)
            diff = pre_lab[ys, xs] - dfm_lab[ys, xs]
            dl = clamp(robust_median(diff[:, 0]), -self.max_l_shift, self.max_l_shift)
            da = clamp(robust_median(diff[:, 1]), -self.max_ab_shift, self.max_ab_shift)
            db = clamp(robust_median(diff[:, 2]), -self.max_ab_shift, self.max_ab_shift)
            self.state.update((dl, da, db), self.temporal_alpha, self.max_step_l, self.max_step_ab)
            updated = True

        s = self.color_match_strength
        dfm_lab[:, :, 0] += self.state.dl * s
        dfm_lab[:, :, 1] += self.state.da * s
        dfm_lab[:, :, 2] += self.state.db * s
        dfm_lab = np.clip(dfm_lab, 0.0, 255.0).astype(np.uint8)
        corrected_local = cv2.cvtColor(dfm_lab, cv2.COLOR_LAB2BGR)

        out_dfm = dfm_bgr.copy()
        out_dfm[y1:y2, x1:x2] = corrected_local
        return out_dfm, updated, valid_count

    def compose(
        self, pre_dfm_bgr: np.ndarray, dfm_bgr: np.ndarray, sec_mask: np.ndarray, theo_mask: np.ndarray,
        visible_face: np.ndarray, occ_mask: np.ndarray, dfm_soft_mask: Optional[np.ndarray], geometry: Mapping[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        bbox = mask_bbox(theo_mask)
        geom_conf = float(geometry.get("confidence", 1.0) or 0.0)
        if bbox is None:
            diag = {
                "dfm_active_pixels": 0, "visible_skin_pixels": 0,
                "dfm_dl": self.state.dl, "dfm_da": self.state.da, "dfm_db": self.state.db,
                "color_state_updated": False, "geometry_confidence": geom_conf, "fallback_used": True,
            }
            return pre_dfm_bgr.copy(), np.zeros(theo_mask.shape, dtype=np.uint8), diag

        alpha = self._build_alpha(sec_mask, theo_mask, occ_mask, dfm_soft_mask)
        corrected_dfm, updated, vis_count = self._match_color_in_bbox(
            pre_dfm_bgr, dfm_bgr, visible_face, occ_mask, theo_mask, bbox, geom_conf
        )

        x1, y1, x2, y2 = bbox
        out = pre_dfm_bgr.copy()
        a3 = alpha[y1:y2, x1:x2, None]
        blended = (
            corrected_dfm[y1:y2, x1:x2].astype(np.float32) * a3
            + pre_dfm_bgr[y1:y2, x1:x2].astype(np.float32) * (1.0 - a3)
        )
        out[y1:y2, x1:x2] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        alpha_u8 = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        diag = {
            "dfm_active_pixels": int(np.count_nonzero(alpha_u8 > 12)),
            "visible_skin_pixels": vis_count,
            "dfm_dl": round(self.state.dl, 3),
            "dfm_da": round(self.state.da, 3),
            "dfm_db": round(self.state.db, 3),
            "color_state_updated": updated,
            "geometry_confidence": geom_conf,
            "fallback_used": not updated,
        }
        return out, alpha_u8, diag


def make_debug_panel(
    pre_dfm: np.ndarray, raw_dfm: np.ndarray, sec_mask: np.ndarray, theo_mask: np.ndarray,
    occ_mask: np.ndarray, alpha_u8: np.ndarray, final_bgr: np.ndarray, diag: Mapping[str, Any], max_width: int = 640
) -> np.ndarray:
    def tag(img: np.ndarray, title: str) -> np.ndarray:
        canvas = img.copy()
        cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 480), 32), (0, 0, 0), -1)
        cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    mask_vis = pre_dfm.copy()
    mask_vis[sec_mask > 0] = (0.55 * mask_vis[sec_mask > 0] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
    mask_vis[theo_mask > 0] = (0.70 * mask_vis[theo_mask > 0] + 0.30 * np.array([0, 255, 0])).astype(np.uint8)
    mask_vis[occ_mask > 0] = (0.50 * mask_vis[occ_mask > 0] + 0.50 * np.array([255, 0, 0])).astype(np.uint8)

    alpha_vis = cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2BGR)
    diff_vis = np.clip(cv2.absdiff(final_bgr, pre_dfm).astype(np.float32) * 3.0, 0, 255).astype(np.uint8)

    panels = [
        tag(pre_dfm, "1. pre-DFM input"), tag(raw_dfm, "2. raw DFM output"),
        tag(mask_vis, "3. R=SeC G=3DDFA B=Occlusion"), tag(alpha_vis, "4. Clamped & Feathered Alpha"),
        tag(diff_vis, "5. Injected DFM Detail (x3)"), tag(final_bgr, "6. Final Composite"),
    ]
    h, w = pre_dfm.shape[:2]
    scale = min(1.0, max_width / max(w, 1))
    pw, ph = max(1, int(w * scale)), max(1, int(h * scale))
    resized = [cv2.resize(p, (pw, ph), interpolation=cv2.INTER_AREA) for p in panels]
    grid = np.vstack([np.hstack(resized[:3]), np.hstack(resized[3:])])

    banner = (
        f"frame={diag.get('frame_id')} | dL={diag.get('dfm_dl', 0):.1f} "
        f"da={diag.get('dfm_da', 0):.1f} db={diag.get('dfm_db', 0):.1f} | "
        f"active_px={diag.get('dfm_active_pixels', 0)} | conf={diag.get('geometry_confidence', 0):.2f}"
    )
    cv2.rectangle(grid, (0, 0), (min(grid.shape[1], 1100), 36), (0, 0, 0), -1)
    cv2.putText(grid, banner, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
    return grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 6: DFM facial restoration and geometry-constrained merge.")
    p.add_argument("--video", type=Path, default=None, help="Original video path (optional, used for audio muxing & FPS).")
    p.add_argument("--pre-dfm-video", type=Path, default=None)
    p.add_argument("--pre-dfm-dir", type=Path, default=None)

    p.add_argument("--dfm-video", type=Path, default=None)
    p.add_argument("--dfm-frames-dir", type=Path, default=None)
    p.add_argument("--dfm-mask-dir", type=Path, default=None)
    p.add_argument("--dfm-model", type=Path, default=None)

    p.add_argument("--sec4b-dir", type=Path, default=None, help="Root SeC-4B directory (auto-resolves mask/visible_face/occlusion).")
    p.add_argument("--3ddfa-dir", dest="tddfa_dir", type=Path, default=None, help="Root 3DDFA directory (auto-resolves face_mask/landmarks/params.jsonl).")
    p.add_argument("--sec-mask-dir", type=Path, default=None)
    p.add_argument("--visible-face-dir", type=Path, default=None)
    p.add_argument("--occlusion-dir", type=Path, default=None)
    p.add_argument("--theoretical-face-dir", type=Path, default=None)
    p.add_argument("--landmarks-dir", type=Path, default=None)
    p.add_argument("--geometry-params", type=Path, default=None)
    p.add_argument("--skin-profile", type=Path, default=None)
    p.add_argument("--output", "--output-dir", dest="output_dir", type=Path, required=True)
    p.add_argument("--export-video", type=Path, default=None)

    p.add_argument("--merge-scope", choices=("mask_only", "full_face"), default="mask_only")
    p.add_argument("--expand-mask-px", type=int, default=18)
    p.add_argument("--feather-radius", type=int, default=10)
    p.add_argument("--temporal-alpha", type=float, default=0.20)
    p.add_argument("--color-match-strength", type=float, default=0.85)

    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-every", type=int, default=30)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def export_video_ffmpeg(frames_dir: Path, output_video: Path, fps: float, audio_source: Optional[Path] = None) -> None:
    ensure_dir(output_video.parent)
    cmd = ["ffmpeg", "-y", "-framerate", f"{fps:.6f}", "-i", str(frames_dir / "%06d.png")]
    if audio_source is not None and audio_source.is_file():
        cmd.extend(["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy", "-shortest"])
    cmd.extend(["-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(output_video)])
    LOG.info("Exporting final video via FFmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    if args.sec4b_dir is not None:
        args.sec_mask_dir = args.sec_mask_dir or (args.sec4b_dir / "mask")
        args.visible_face_dir = args.visible_face_dir or (args.sec4b_dir / "visible_face")
        args.occlusion_dir = args.occlusion_dir or (args.sec4b_dir / "occlusion")
    if args.tddfa_dir is not None:
        args.theoretical_face_dir = args.theoretical_face_dir or (args.tddfa_dir / "face_mask")
        args.landmarks_dir = args.landmarks_dir or (args.tddfa_dir / "landmarks")
        args.geometry_params = args.geometry_params or (args.tddfa_dir / "params.jsonl")

    if not (args.sec_mask_dir and args.visible_face_dir and args.occlusion_dir and args.theoretical_face_dir):
        raise ValueError("Provide --sec4b-dir and --3ddfa-dir (or explicit mask directory flags).")

    pre_source = FrameSource(args.pre_dfm_video, args.pre_dfm_dir)
    ext_dfm_source: Optional[FrameSource] = None
    onnx_runner: Optional[DFMOnnxRunner] = None

    if args.dfm_video is not None or args.dfm_frames_dir is not None:
        ext_dfm_source = FrameSource(args.dfm_video, args.dfm_frames_dir)
    elif args.dfm_model is not None:
        onnx_runner = DFMOnnxRunner(args.dfm_model, device=args.device)
    else:
        raise ValueError("Must provide either (--dfm-video / --dfm-frames-dir) or --dfm-model.")

    frames_out_dir = args.output_dir / "frames"
    alphas_out_dir = args.output_dir / "alphas"
    debug_out_dir = args.output_dir / "debug"
    meta_path = args.output_dir / "metadata.jsonl"
    ensure_dir(frames_out_dir)
    ensure_dir(alphas_out_dir)
    if args.debug:
        ensure_dir(debug_out_dir)

    geometry_records = load_jsonl(args.geometry_params)
    existing_meta = load_jsonl(meta_path)

    composer = DFMGeometryComposer(
        merge_scope=args.merge_scope,
        expand_mask_px=args.expand_mask_px,
        feather_radius=args.feather_radius,
        temporal_alpha=args.temporal_alpha,
        color_match_strength=args.color_match_strength,
    )

    if args.start > 0 and existing_meta:
        prev_keys = [k for k in existing_meta.keys() if k < args.start]
        if prev_keys:
            composer.state.restore(existing_meta[max(prev_keys)])

    end_frame = pre_source.total_frames if args.end is None else min(args.end, pre_source.total_frames)
    w, h = pre_source.width, pre_source.height

    try:
        with meta_path.open("a", encoding="utf-8") as meta_fh:
            for fid in range(args.start, end_frame):
                out_frame_path = frames_out_dir / frame_name(fid)
                out_alpha_path = alphas_out_dir / frame_name(fid)

                if args.resume and not args.overwrite and out_frame_path.exists() and out_alpha_path.exists() and fid in existing_meta:
                    composer.state.restore(existing_meta[fid])
                    continue

                pre_frame = pre_source.read_frame(fid)
                if pre_frame is None:
                    LOG.warning("Missing pre-DFM frame %d; skipping", fid)
                    continue

                sec_mask = read_mask(args.sec_mask_dir / frame_name(fid), w, h)
                theo_mask = read_mask(args.theoretical_face_dir / frame_name(fid), w, h)
                vis_mask = read_mask(args.visible_face_dir / frame_name(fid), w, h)
                occ_mask = read_mask(args.occlusion_dir / frame_name(fid), w, h)

                if sec_mask is None:
                    sec_mask = np.zeros((h, w), dtype=np.uint8)
                if theo_mask is None:
                    theo_mask = np.zeros((h, w), dtype=np.uint8)
                if vis_mask is None:
                    vis_mask = np.zeros((h, w), dtype=np.uint8)
                if occ_mask is None:
                    occ_mask = np.zeros((h, w), dtype=np.uint8)

                dfm_soft: Optional[np.ndarray] = None
                if ext_dfm_source is not None:
                    raw_dfm = ext_dfm_source.read_frame(fid)
                    if raw_dfm is None:
                        raw_dfm = pre_frame.copy()
                    elif raw_dfm.shape[:2] != (h, w):
                        raw_dfm = cv2.resize(raw_dfm, (w, h), interpolation=cv2.INTER_LINEAR)
                    if args.dfm_mask_dir is not None:
                        dfm_soft = read_soft_mask(args.dfm_mask_dir / frame_name(fid), w, h)
                else:
                    assert onnx_runner is not None
                    lmk_path = (args.landmarks_dir / frame_name(fid, ".npy")) if args.landmarks_dir else None
                    raw_dfm, dfm_soft = onnx_runner.infer_aligned(pre_frame, theo_mask, lmk_path)

                geom = geometry_records.get(fid, {})
                final_bgr, alpha_u8, diag = composer.compose(
                    pre_frame, raw_dfm, sec_mask, theo_mask, vis_mask, occ_mask, dfm_soft, geom
                )
                diag["frame_id"] = fid

                atomic_imwrite(out_frame_path, final_bgr)
                atomic_imwrite(out_alpha_path, alpha_u8)
                meta_fh.write(json.dumps(diag, ensure_ascii=False) + "\n")
                meta_fh.flush()

                if args.debug and fid % args.debug_every == 0:
                    dbg = make_debug_panel(pre_frame, raw_dfm, sec_mask, theo_mask, occ_mask, alpha_u8, final_bgr, diag)
                    atomic_imwrite(debug_out_dir / frame_name(fid, ".jpg"), dbg)

                if (fid - args.start + 1) % 60 == 0:
                    LOG.info("[%d/%d] dL=%.1f da=%.1f db=%.1f active_px=%d",
                             fid + 1, end_frame, diag["dfm_dl"], diag["dfm_da"], diag["dfm_db"], diag["dfm_active_pixels"])
    finally:
        pre_source.close()
        if ext_dfm_source is not None:
            ext_dfm_source.close()

    if args.export_video is not None:
        export_video_ffmpeg(frames_out_dir, args.export_video, pre_source.fps, audio_source=args.video or args.pre_dfm_video)


if __name__ == "__main__":
    main()