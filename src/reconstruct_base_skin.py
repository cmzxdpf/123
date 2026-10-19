#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 3: rebuild a clean, low-frequency base-skin layer before DFM."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

LOG = logging.getLogger("reconstruct_base_skin")


def frame_name(frame_id: int) -> str:
    return f"{frame_id:06d}.png"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_imwrite(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise IOError(f"Failed to encode image: {tmp}")
    buf.tofile(str(tmp))
    os.replace(tmp, path)


def load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                out[int(item["frame_id"])] = item
            except Exception as exc:
                LOG.warning("Skip malformed %s line %d: %s", path, n, exc)
    return out


def read_binary_mask(path: Path, width: int, height: int) -> Optional[np.ndarray]:
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


def fill_mask_holes(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Fills internal holes (3D nostril cavities, mouth cavity) in the face mask."""
    if mask is None or not np.any(mask):
        return mask
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, contours, -1, 255 if mask.dtype == np.uint8 else 1, -1)
    return solid


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def robust_median(values: np.ndarray, q0: float = 10.0, q1: float = 90.0) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty robust_median input")
    lo, hi = np.percentile(arr, [q0, q1])
    trimmed = arr[(arr >= lo) & (arr <= hi)]
    if trimmed.size == 0:
        trimmed = arr
    return float(np.median(trimmed))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class SkinProfile:
    global_lab: np.ndarray
    regions: Dict[str, np.ndarray]
    lab_format: str

    @staticmethod
    def _extract_lab(obj: Any) -> Optional[Sequence[float]]:
        if isinstance(obj, (list, tuple)) and len(obj) >= 3:
            return [float(obj[0]), float(obj[1]), float(obj[2])]
        if not isinstance(obj, Mapping):
            return None
        for key in ("lab_mean", "mean_lab", "lab", "mean"):
            if key in obj:
                v = obj[key]
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    return [float(v[0]), float(v[1]), float(v[2])]
                if isinstance(v, Mapping):
                    if all(k in v for k in ("L", "a", "b")):
                        return [float(v["L"]), float(v["a"]), float(v["b"])]
                    if all(k in v for k in ("l", "a", "b")):
                        return [float(v["l"]), float(v["a"]), float(v["b"])]
        if all(k in obj for k in ("L", "a", "b")):
            return [float(obj["L"]), float(obj["a"]), float(obj["b"])]
        return None

    @staticmethod
    def _detect_format(raw: Mapping[str, Any], sample: Sequence[float]) -> str:
        text = " ".join(str(raw.get(k, "")) for k in ("lab_format", "color_space", "format")).lower()
        if "opencv" in text or "cv2" in text:
            return "opencv"
        if "cie" in text or "standard" in text:
            return "cielab"
        l, a, b = map(float, sample[:3])
        return "cielab" if l <= 100 and abs(a) < 100 and abs(b) < 100 else "opencv"

    @staticmethod
    def _to_cv_lab(v: Sequence[float], fmt: str) -> np.ndarray:
        l, a, b = map(float, v[:3])
        if fmt == "cielab":
            l = clamp(l, 0, 100) * 255.0 / 100.0
            a = clamp(a, -128, 127) + 128.0
            b = clamp(b, -128, 127) + 128.0
        return np.array([clamp(l, 0, 255), clamp(a, 0, 255), clamp(b, 0, 255)], np.float32)

    @classmethod
    def load(cls, path: Path) -> "SkinProfile":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, Mapping):
            raise ValueError("skin_profile.json must contain a JSON object")

        regions_src = raw.get("regions", {}) if isinstance(raw.get("regions"), Mapping) else {}
        aliases = {
            "forehead": ("forehead", "upper_face"),
            "left_cheek": ("left_cheek", "cheek_left", "l_cheek"),
            "right_cheek": ("right_cheek", "cheek_right", "r_cheek"),
            "chin": ("chin",),
            "nose_lower": ("nose_lower", "lower_nose", "under_nose", "nose"),
        }

        global_v = None
        if "global" in raw:
            global_v = cls._extract_lab(raw["global"])
        if global_v is None:
            for key in ("global_lab", "skin_lab", "reference_lab", "lab_mean"):
                if key in raw:
                    global_v = cls._extract_lab(raw[key])
                    if global_v is not None:
                        break

        region_raw: Dict[str, Sequence[float]] = {}
        for canonical, names in aliases.items():
            for name in names:
                if name in regions_src:
                    v = cls._extract_lab(regions_src[name])
                    if v is not None:
                        region_raw[canonical] = v
                        break
                if name in raw:
                    v = cls._extract_lab(raw[name])
                    if v is not None:
                        region_raw[canonical] = v
                        break

        if global_v is None:
            if not region_raw:
                raise ValueError(f"No usable LAB data in {path}")
            global_v = np.median(np.asarray(list(region_raw.values()), np.float32), axis=0).tolist()

        fmt = cls._detect_format(raw, global_v)
        return cls(
            global_lab=cls._to_cv_lab(global_v, fmt),
            regions={k: cls._to_cv_lab(v, fmt) for k, v in region_raw.items()},
            lab_format=fmt,
        )

    def get(self, name: str) -> np.ndarray:
        return self.regions.get(name, self.global_lab)

    def field(self, width: int, height: int, lowfreq_kernel: Optional[int] = None) -> np.ndarray:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        xn = xx / max(width - 1, 1)
        yn = yy / max(height - 1, 1)
        anchors = {
            "forehead": (0.50, 0.18),
            "left_cheek": (0.30, 0.48),
            "right_cheek": (0.70, 0.48),
            "nose_lower": (0.50, 0.56),
            "chin": (0.50, 0.82),
        }
        acc = np.zeros((height, width, 3), np.float32)
        wsum = np.zeros((height, width), np.float32)
        for name, (ax, ay) in anchors.items():
            d2 = (xn - ax) ** 2 + (yn - ay) ** 2
            w = 1.0 / (d2 + 0.035)
            acc += w[..., None] * self.get(name)
            wsum += w
        acc += 1.5 * self.global_lab
        wsum += 1.5
        out = acc / np.maximum(wsum[..., None], 1e-6)
        if lowfreq_kernel is not None and lowfreq_kernel > 0:
            k = int(lowfreq_kernel)
        else:
            k = int(max(3, min(height, width) // 3))
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(out, (k, k), 0)


@dataclass
class TemporalState:
    l_offset: float = 0.0
    a_offset: float = 0.0
    b_offset: float = 0.0
    light_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5, np.float32))
    initialized: bool = False

    def update(
        self,
        offsets: np.ndarray,
        coeffs: Optional[np.ndarray],
        alpha: float,
        coeff_alpha: float,
        max_step_l: float,
        max_step_ab: float,
    ) -> None:
        target = offsets.astype(np.float32)
        cur = np.array([self.l_offset, self.a_offset, self.b_offset], np.float32)
        if not self.initialized:
            cur = target
            self.initialized = True
        else:
            delta = target - cur
            delta[0] = np.clip(delta[0], -max_step_l, max_step_l)
            delta[1:] = np.clip(delta[1:], -max_step_ab, max_step_ab)
            cur += float(alpha) * delta
        self.l_offset, self.a_offset, self.b_offset = map(float, cur)
        if coeffs is not None and np.all(np.isfinite(coeffs)):
            self.light_coeffs = (1 - coeff_alpha) * self.light_coeffs + coeff_alpha * coeffs.astype(np.float32)

    def restore(self, record: Mapping[str, Any]) -> None:
        """Fixed: compatible with both new keys (exposure_gain/lab_shift/illumination_coeffs) and legacy keys."""
        try:
            if "exposure_gain" in record:
                self.l_offset = float(record["exposure_gain"])
            elif "exposure_l_offset" in record:
                self.l_offset = float(record["exposure_l_offset"])

            if "lab_shift" in record and isinstance(record["lab_shift"], (list, tuple)) and len(record["lab_shift"]) >= 2:
                self.a_offset = float(record["lab_shift"][0])
                self.b_offset = float(record["lab_shift"][1])
            else:
                self.a_offset = float(record.get("a_offset", self.a_offset))
                self.b_offset = float(record.get("b_offset", self.b_offset))

            coeffs = record.get("illumination_coeffs") if "illumination_coeffs" in record else record.get("light_coeffs")
            if isinstance(coeffs, list) and len(coeffs) == 5:
                self.light_coeffs = np.asarray(coeffs, np.float32)
            self.initialized = True
        except Exception:
            pass


class BaseSkinRenderer:
    def __init__(self, profile: SkinProfile, args: argparse.Namespace) -> None:
        self.profile = profile
        self.args = args
        self.state = TemporalState()
        self._prev_theoretical: Optional[np.ndarray] = None

    def _fit_light(
        self,
        current_l: np.ndarray,
        base_l: np.ndarray,
        valid: np.ndarray,
        global_l_offset: float,
    ) -> Optional[np.ndarray]:
        ys, xs = np.nonzero(valid)
        if xs.size < 256:
            return None
        if xs.size > self.args.max_fit_samples:
            idx = np.linspace(0, xs.size - 1, self.args.max_fit_samples, dtype=np.int32)
            xs, ys = xs[idx], ys[idx]

        h, w = current_l.shape
        xn = (xs.astype(np.float32) / max(w - 1, 1)) * 2 - 1
        yn = (ys.astype(np.float32) / max(h - 1, 1)) * 2 - 1
        target = current_l[ys, xs].astype(np.float32) - base_l[ys, xs].astype(np.float32) - global_l_offset
        target = np.clip(target, -self.args.max_light_residual, self.args.max_light_residual)
        A = np.stack([xn, yn, xn * yn, xn * xn - 1 / 3, yn * yn - 1 / 3], axis=1)
        ridge = 0.15
        try:
            return np.linalg.solve(A.T @ A + ridge * np.eye(5, dtype=np.float32), A.T @ target).astype(np.float32)
        except np.linalg.LinAlgError:
            return None

    def _lighting_surface(self, width: int, height: int) -> np.ndarray:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        x = (xx / max(width - 1, 1)) * 2 - 1
        y = (yy / max(height - 1, 1)) * 2 - 1
        c = self.state.light_coeffs
        out = c[0] * x + c[1] * y + c[2] * x * y + c[3] * (x * x - 1 / 3) + c[4] * (y * y - 1 / 3)
        return np.clip(out, -self.args.max_light_residual, self.args.max_light_residual)

    def _alpha(self, rebuild_local: np.ndarray, theo_local: np.ndarray) -> np.ndarray:
        """Feathers only against real visible skin inside theoretical_face; keeps full 1.0 alpha at jawline."""
        if self.args.feather_radius <= 0:
            return rebuild_local.astype(np.float32)
        interior_mask = np.where(theo_local, rebuild_local.astype(np.uint8), 1)
        dist = cv2.distanceTransform(interior_mask, cv2.DIST_L2, 5)
        ramp = np.clip(dist / max(float(self.args.feather_radius), 1.0), 0.0, 1.0)
        alpha = self.args.edge_min_alpha + (1.0 - self.args.edge_min_alpha) * ramp
        return np.where(rebuild_local, alpha, 0.0).astype(np.float32)

    def _safe_visible(
        self,
        frame: np.ndarray,
        visible: np.ndarray,
        theoretical: np.ndarray,
        sec_mask: np.ndarray,
        occlusion: np.ndarray,
    ) -> np.ndarray:
        base = (visible > 0) & (theoretical > 0) & ~(sec_mask > 0) & ~(occlusion > 0)
        if self.args.visible_erode > 0:
            r = int(self.args.visible_erode)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            base = cv2.erode(base.astype(np.uint8), kernel) > 0
        if int(np.count_nonzero(base)) < 128:
            return base.astype(np.uint8)

        bgr = frame.astype(np.float32)
        ys, xs = np.nonzero(base)
        g = bgr[..., 1][ys, xs]
        lo, hi = np.percentile(g, [20.0, 90.0])
        keep = (g >= lo) & (g <= hi)
        mx = bgr.max(axis=2)[ys, xs]
        mn = bgr.min(axis=2)[ys, xs]
        keep &= (mx - mn) < self.args.max_saturation_spread
        keep &= mx > self.args.min_valid_lum
        keep &= mn < self.args.max_valid_lum

        out = np.zeros((base.shape[0], base.shape[1]), np.uint8)
        out[ys[keep], xs[keep]] = 255
        return out

    def render(
        self,
        frame: np.ndarray,
        sec_mask: np.ndarray,
        visible: np.ndarray,
        occlusion: np.ndarray,
        theoretical: np.ndarray,
        geometry: Mapping[str, Any],
        landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        # 严格保持脸型宽度与下颌长度，仅垂直单向向上延伸 8px 彻底吞没鼻梁与面颊口罩暗色织物边，零横向与下向外扩
        k_up = np.ones((9, 1), dtype=np.uint8)
        sec_vert = cv2.dilate((sec_mask > 0).astype(np.uint8), k_up, anchor=(0, 0)) > 0
        face_region = sec_vert & (theoretical > 0)
        rebuild = face_region
        face_area = int(np.count_nonzero(theoretical))
        rebuild_area = int(np.count_nonzero(rebuild))
        bbox = bbox_from_mask(theoretical)
        geom_conf = float(geometry.get("confidence", geometry.get("tracking_quality", 1.0)) or 0.0)
        geom_fallback = bool(geometry.get("fallback_used", False))

        diag: Dict[str, Any] = {
            "face_region_pixels": int(np.count_nonzero(face_region)),
            "rebuild_pixels": rebuild_area,
            "geometry_confidence": geom_conf,
            "geometry_fallback": geom_fallback,
            "fallback": False,
            "fallback_reason": None,
        }

        if bbox is None or rebuild_area == 0 or not np.any(theoretical):
            diag.update({
                "visible_skin_pixels": 0,
                "visible_skin_ratio": 0.0,
                "exposure_gain": self.state.l_offset,
                "lab_shift": [self.state.a_offset, self.state.b_offset],
                "illumination_coeffs": self.state.light_coeffs.tolist(),
                "confidence": geom_conf,
                "fallback": True,
                "fallback_reason": "no_rebuild_region",
            })
            return frame.copy(), rebuild.astype(np.uint8) * 255, np.zeros_like(frame), diag

        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        base_field = self.profile.field(bw, bh, getattr(self.args, "lowfreq_kernel", 31))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        valid = self._safe_visible(frame, visible, theoretical, sec_mask, occlusion)
        valid_local = valid[y1:y2, x1:x2]
        current_local = lab[y1:y2, x1:x2]
        ys, xs = np.nonzero(valid_local)
        visible_pixels = int(xs.size)
        visible_ratio = visible_pixels / max(face_area, 1)

        can_update = False
        if visible_pixels > 0:
            residual = current_local[ys, xs] - base_field[ys, xs]
            offsets = np.array([
                robust_median(residual[:, 0]),
                robust_median(residual[:, 1]),
                robust_median(residual[:, 2]),
            ], np.float32)
            current_weight = clamp(1.0 - self.args.reference_weight, 0.0, 1.0)
            offsets *= current_weight
            offsets[0] = np.clip(offsets[0], -self.args.max_l_offset, self.args.max_l_offset)
            offsets[1:] = np.clip(offsets[1:], -self.args.max_ab_offset, self.args.max_ab_offset)
            coeffs = self._fit_light(current_local[:, :, 0], base_field[:, :, 0], valid_local, float(offsets[0]))

            can_update = (
                visible_ratio >= self.args.min_visible_ratio
                and geom_conf >= self.args.confidence_threshold
                and not geom_fallback
            )
            if can_update:
                self.state.update(
                    offsets,
                    coeffs,
                    self.args.temporal_alpha,
                    self.args.coeff_alpha,
                    self.args.max_step_l,
                    self.args.max_step_ab,
                )
            else:
                diag["fallback"] = True
                diag["fallback_reason"] = (
                    f"low_confidence(conf={geom_conf:.2f},vis={visible_ratio:.3f})"
                    if not geom_fallback else "geometry_fallback"
                )
        else:
            diag["fallback"] = True
            diag["fallback_reason"] = "no_safe_visible_skin"

        synthetic_lab = base_field.copy()
        synthetic_lab[:, :, 0] += self.state.l_offset + self._lighting_surface(bw, bh)
        synthetic_lab[:, :, 1] += self.state.a_offset
        synthetic_lab[:, :, 2] += self.state.b_offset
        synthetic_lab = np.clip(synthetic_lab, 0, 255).astype(np.uint8)
        skin_local = cv2.cvtColor(synthetic_lab, cv2.COLOR_LAB2BGR)

        # 3D-guided Micro-Texture Anchor Cues (subtle nose shadow & lip slit)
        if landmarks is not None and len(landmarks) >= 68:
            nb = landmarks[33]
            ml, mr = landmarks[48], landmarks[54]
            mm = (landmarks[62] + landmarks[66]) / 2.0
            nb_x, nb_y = int(nb[0] - x1), int(nb[1] - y1)
            ml_x, ml_y = int(ml[0] - x1), int(ml[1] - y1)
            mr_x, mr_y = int(mr[0] - x1), int(mr[1] - y1)
            mm_x, mm_y = int(mm[0] - x1), int(mm[1] - y1)

            shadow_local = np.zeros((bh, bw), dtype=np.float32)
            nw = max(4, int(bw * 0.08))
            nh = max(2, int(bh * 0.03))
            cv2.ellipse(shadow_local, (nb_x, nb_y + 2), (nw, nh), 0, 0, 360, 0.22, -1)
            curve = np.array([[ml_x, ml_y], [mm_x, mm_y], [mr_x, mr_y]], dtype=np.int32)
            cv2.polylines(shadow_local, [curve], False, 0.22, 2, cv2.LINE_AA)
            k_s = max(3, int(bw * 0.04) | 1)
            shadow_local = cv2.GaussianBlur(shadow_local, (k_s, k_s), 1.6)
            skin_local = np.clip(skin_local.astype(np.float32) * (1.0 - shadow_local[..., None]), 0, 255).astype(np.uint8)

        synthetic_full = np.zeros_like(frame)
        synthetic_full[y1:y2, x1:x2] = skin_local

        rebuild_local = rebuild[y1:y2, x1:x2]
        theo_local = theoretical[y1:y2, x1:x2] > 0
        alpha = self._alpha(rebuild_local, theo_local)[:, :, None]
        out = frame.copy()
        orig_local = out[y1:y2, x1:x2].astype(np.float32)
        blend = skin_local.astype(np.float32) * alpha + orig_local * (1.0 - alpha)
        out[y1:y2, x1:x2] = np.clip(blend, 0, 255).astype(np.uint8)

        diag.update({
            "visible_skin_pixels": visible_pixels,
            "visible_skin_ratio": visible_ratio,
            "exposure_gain": self.state.l_offset,
            "lab_shift": [round(self.state.a_offset, 2), round(self.state.b_offset, 2)],
            "illumination_coeffs": self.state.light_coeffs.astype(float).tolist(),
            "confidence": geom_conf,
            "state_updated": bool(can_update),
        })
        return out, rebuild.astype(np.uint8) * 255, synthetic_full, diag


def make_debug(
    frame: np.ndarray,
    sec: np.ndarray,
    theoretical: np.ndarray,
    visible: np.ndarray,
    occ: np.ndarray,
    rebuild: np.ndarray,
    synthetic: np.ndarray,
    output: np.ndarray,
    diag: Mapping[str, Any],
) -> np.ndarray:
    overlay = frame.copy()
    overlay[sec > 0] = (0.55 * overlay[sec > 0] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
    overlay[theoretical > 0] = (0.72 * overlay[theoretical > 0] + 0.28 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[occ > 0] = (0.60 * overlay[occ > 0] + 0.40 * np.array([255, 0, 0])).astype(np.uint8)

    def color_mask(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        x = np.zeros((*mask.shape, 3), np.uint8)
        x[mask > 0] = color
        return x

    panels = [frame, overlay, color_mask(visible, (0, 255, 0)), color_mask(rebuild, (0, 255, 255)), synthetic, output]
    max_w = 480
    scale = min(1.0, max_w / max(frame.shape[1], 1))
    size = (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale)))
    panels = [cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in panels]
    canvas = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    text = (
        f"frame={diag.get('frame_id')}  L={diag.get('exposure_gain', 0):.1f} "
        f"a={diag.get('lab_shift', [0, 0])[0] if isinstance(diag.get('lab_shift'), (list, tuple)) else 0:.1f} "
        f"vis={diag.get('visible_skin_ratio', 0):.3f} conf={diag.get('geometry_confidence', 0):.2f} "
        f"fallback={diag.get('fallback', False)}"
    )
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 1400), 34), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Task 3 pre-DFM base skin reconstruction")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--sec4b-dir", type=Path, required=True)
    p.add_argument("--3ddfa-dir", type=Path, required=True, dest="tddfa_dir")
    p.add_argument("--skin-profile", type=Path, required=True)
    p.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=120)

    p.add_argument("--reference-weight", type=float, default=0.78)
    p.add_argument("--feather-radius", type=int, default=7)
    p.add_argument("--edge-min-alpha", type=float, default=0.75)
    p.add_argument("--temporal-alpha", type=float, default=0.22)
    p.add_argument("--coeff-alpha", type=float, default=0.16)
    p.add_argument("--min-visible-ratio", type=float, default=0.03)
    p.add_argument("--max-l-offset", type=float, default=42.0)
    p.add_argument("--max-ab-offset", type=float, default=12.0)
    p.add_argument("--max-step-l", type=float, default=8.0)
    p.add_argument("--max-step-ab", type=float, default=3.0)
    p.add_argument("--visible-erode", type=int, default=3)
    p.add_argument("--confidence-threshold", type=float, default=0.45)
    p.add_argument("--lowfreq-kernel", type=int, default=31)
    p.add_argument("--max-light-residual", type=float, default=18.0)
    p.add_argument("--max-fit-samples", type=int, default=5000)
    p.add_argument("--min-valid-lum", type=float, default=40.0)
    p.add_argument("--max-valid-lum", type=float, default=250.0)
    p.add_argument("--max-saturation-spread", type=float, default=120.0)
    return p


def run(args: argparse.Namespace) -> None:
    profile = SkinProfile.load(args.skin_profile)
    sec_mask_dir = args.sec4b_dir / "mask"
    visible_dir = args.sec4b_dir / "visible_face"
    occlusion_dir = args.sec4b_dir / "occlusion"
    theoretical_dir = args.tddfa_dir / "face_mask"
    landmarks_dir = args.tddfa_dir / "landmarks"
    geometry = load_jsonl(args.tddfa_dir / "params.jsonl")

    frames_dir = args.output / "frames"
    masks_dir = args.output / "masks"
    debug_dir = args.output / "debug"
    ensure_dir(frames_dir)
    ensure_dir(masks_dir)

    metadata_path = args.output / "metadata.jsonl"
    old_meta = load_jsonl(metadata_path)

    renderer = BaseSkinRenderer(profile, args)
    if args.start > 0:
        prev_ids = [i for i in old_meta if i < args.start]
        if prev_ids:
            renderer.state.restore(old_meta[max(prev_ids)])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end = (total if args.end is None else min(args.end, total)) if total > 0 else args.end
    if end is None or end < args.start:
        raise ValueError("Invalid [start, end) range")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    mf = metadata_path.open("a", encoding="utf-8")
    processed = skipped = fallback_count = 0
    t0 = cv2.getTickCount()

    try:
        for frame_id in range(args.start, end):
            ok, frame = cap.read()
            if not ok or frame is None:
                LOG.warning("Decode failed at frame %d", frame_id)
                continue

            out_path = frames_dir / frame_name(frame_id)
            mask_out_path = masks_dir / frame_name(frame_id)
            if args.resume and not args.overwrite and frame_id in old_meta and out_path.exists() and mask_out_path.exists():
                renderer.state.restore(old_meta[frame_id])
                skipped += 1
                continue

            sec = read_binary_mask(sec_mask_dir / frame_name(frame_id), width, height)
            visible = read_binary_mask(visible_dir / frame_name(frame_id), width, height)
            occ = read_binary_mask(occlusion_dir / frame_name(frame_id), width, height)
            theoretical = read_binary_mask(theoretical_dir / frame_name(frame_id), width, height)
            if theoretical is not None and np.any(theoretical):
                theoretical = fill_mask_holes(theoretical)

            if visible is None:
                visible = np.zeros((height, width), np.uint8)
            if occ is None:
                occ = np.zeros((height, width), np.uint8)

            theo_fallback = False
            if theoretical is None:
                theo_fallback = True
                theoretical = renderer._prev_theoretical.copy() if renderer._prev_theoretical is not None else np.zeros((height, width), np.uint8)
            else:
                renderer._prev_theoretical = theoretical.copy()

            g = dict(geometry.get(frame_id, {}))
            if theo_fallback:
                g["fallback_used"] = True
                g["confidence"] = min(float(g.get("confidence", 0.35) or 0.35), 0.35)

            if sec is None or not np.any(sec):
                output = frame.copy()
                rebuild = np.zeros((height, width), np.uint8)
                synthetic = np.zeros_like(frame)
                diag = {
                    "frame_id": frame_id,
                    "face_region_pixels": 0,
                    "rebuild_pixels": 0,
                    "occlusion_ratio": float(np.count_nonzero(occ)) / max(np.count_nonzero(theoretical), 1),
                    "visible_skin_pixels": int(np.count_nonzero(visible)),
                    "visible_skin_ratio": 0.0,
                    "exposure_gain": renderer.state.l_offset,
                    "lab_shift": [renderer.state.a_offset, renderer.state.b_offset],
                    "illumination_coeffs": renderer.state.light_coeffs.tolist(),
                    "geometry_confidence": float(g.get("confidence", 0.0) or 0.0),
                    "confidence": float(g.get("confidence", 0.0) or 0.0),
                    "fallback": True,
                    "fallback_reason": "missing_sec_mask",
                    "output_valid": False,
                }
            else:
                lmk = None
                if landmarks_dir.exists():
                    lmk_p = landmarks_dir / frame_name(frame_id).replace(".png", ".npy")
                    if lmk_p.exists():
                        try:
                            lmk = np.load(str(lmk_p))
                        except Exception:
                            lmk = None
                output, rebuild, synthetic, diag = renderer.render(frame, sec, visible, occ, theoretical, g, landmarks=lmk)
                diag.update({
                    "frame_id": frame_id,
                    "output_valid": True,
                    "theoretical_fallback": theo_fallback,
                    "occlusion_ratio": float(np.count_nonzero(occ)) / max(np.count_nonzero(theoretical), 1),
                })

            if diag.get("fallback"):
                fallback_count += 1

            atomic_imwrite(out_path, output)
            atomic_imwrite(mask_out_path, rebuild)
            mf.write(json.dumps(diag, ensure_ascii=False, separators=(",", ":")) + "\n")
            mf.flush()
            old_meta[frame_id] = diag

            if args.debug_every and args.debug_every > 0 and frame_id % args.debug_every == 0:
                ensure_dir(debug_dir)
                dbg = make_debug(
                    frame,
                    sec if sec is not None else np.zeros((height, width), np.uint8),
                    theoretical, visible, occ, rebuild, synthetic, output, diag,
                )
                atomic_imwrite(debug_dir / f"{frame_id:06d}.jpg", dbg)

            processed += 1
            if processed % args.log_every == 0:
                elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
                LOG.info("[%d/%d] processed=%d skipped=%d speed=%.2f fps fallback=%d",
                         frame_id + 1, end, processed, skipped, processed / max(elapsed, 1e-6), fallback_count)
    finally:
        mf.close()
        cap.release()

    tmp = metadata_path.with_name(metadata_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for fid in sorted(old_meta):
            f.write(json.dumps(old_meta[fid], ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, metadata_path)
    gc.collect()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()