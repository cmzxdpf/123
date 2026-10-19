#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 4: Outside-Face Region Deterministic Restoration (DIS flow + Masked Telea Inpaint)."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gc
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("restore_outside_region")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def natural_sort_key(s: str) -> list:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


@dataclass
class RestoreConfig:
    video_path: str
    sec4b_dir: str
    tddfa_dir: str
    output_dir: str
    start_frame: int = 0
    end_frame: Optional[int] = None
    resume: bool = True
    overwrite: bool = False
    temporal_window: int = 7
    disable_temporal: bool = False
    debug_every: int = 100
    inpaint_method: str = "telea"
    inpaint_radius: Optional[int] = None
    inpaint_radius_ratio: float = 0.02
    dilate_pixels: Optional[int] = None
    dilate_ratio: float = 0.015
    min_dilate_px: int = 2
    max_dilate_px: int = 12
    thin_radius_ratio: float = 0.025
    thin_radius_px: float = 4.0
    small_area_ratio: float = 0.015
    small_area_px: int = 350
    photo_thresh: float = 30.0
    grad_thresh: float = 40.0
    flow_thresh: float = 3.5
    min_donor_confidence: float = 0.30
    log_every: int = 50


@dataclass
class FrameRecord:
    idx: int
    frame: np.ndarray
    sec_mask: np.ndarray
    face_mask: np.ndarray
    occlusion: np.ndarray
    face_width: float = 0.0
    face_height: float = 0.0
    face_area: int = 0
    face_bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class ComponentInfo:
    label_id: int
    area: int
    bbox: Tuple[int, int, int, int]
    max_radius: float
    mean_radius: float
    is_thin: bool
    is_small: bool
    classification: str


class VideoReader:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Input video/directory does not exist: {self.path}")

        self.is_video_file = os.path.isfile(self.path) and self.path.lower().endswith(VIDEO_EXTS)
        self.is_image_dir = os.path.isdir(self.path)
        if not (self.is_video_file or self.is_image_dir):
            raise ValueError(f"Unsupported input: {self.path}")

        self.frame_files: List[str] = []
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_pos = -1

        if self.is_image_dir:
            all_files = [f for f in os.listdir(self.path) if f.lower().endswith(IMAGE_EXTS)]
            all_files.sort(key=natural_sort_key)
            self.frame_files = [os.path.join(self.path, f) for f in all_files]
            first_frame = cv2.imdecode(np.fromfile(self.frame_files[0], dtype=np.uint8), cv2.IMREAD_COLOR)
            self.height, self.width = first_frame.shape[:2]
            self.total_frames = len(self.frame_files)
            self.fps = 30.0
        else:
            self._cap = cv2.VideoCapture(self.path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video file: {self.path}")
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)

    def seek(self, frame_idx: int) -> None:
        if self.is_video_file and self._cap is not None and self._current_pos != frame_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self._current_pos = frame_idx

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        if self.is_image_dir:
            return cv2.imdecode(np.fromfile(self.frame_files[frame_idx], dtype=np.uint8), cv2.IMREAD_COLOR)
        if self._current_pos != frame_idx:
            self.seek(frame_idx)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        self._current_pos = frame_idx + 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def resolve_subdirectories(sec4b_dir: str, tddfa_dir: str) -> Tuple[str, str, str]:
    sec_mask_dir = sec4b_dir
    for sub in ("mask", "masks"):
        cand = os.path.join(sec4b_dir, sub)
        if os.path.isdir(cand):
            sec_mask_dir = cand
            break
    occlusion_dir = sec4b_dir
    for sub in ("occlusion", "occlusions"):
        cand = os.path.join(sec4b_dir, sub)
        if os.path.isdir(cand):
            occlusion_dir = cand
            break
    face_mask_dir = tddfa_dir
    for sub in ("face_mask", "face_masks", "mask", "masks"):
        cand = os.path.join(tddfa_dir, sub)
        if os.path.isdir(cand):
            face_mask_dir = cand
            break
    return sec_mask_dir, occlusion_dir, face_mask_dir


def load_binary_mask(directory: str, frame_idx: int, target_shape: Tuple[int, int]) -> np.ndarray:
    h_tgt, w_tgt = target_shape
    for name in (f"{frame_idx:06d}.png", f"{frame_idx:06d}.jpg", f"{frame_idx:05d}.png", f"{frame_idx}.png"):
        cand_p = os.path.join(directory, name)
        if os.path.isfile(cand_p):
            try:
                mask = cv2.imdecode(np.fromfile(cand_p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    break
                if mask.shape[:2] != (h_tgt, w_tgt):
                    mask = cv2.resize(mask, (w_tgt, h_tgt), interpolation=cv2.INTER_NEAREST)
                return np.where(mask > 127, np.uint8(255), np.uint8(0))
            except Exception:
                break
    return np.zeros((h_tgt, w_tgt), dtype=np.uint8)


def save_image_unicode(path: str, img: np.ndarray, is_png: bool = True) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = ".png" if is_png else ".jpg"
    tmp_path = path + ".tmp" + ext
    params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if is_png else [cv2.IMWRITE_JPEG_QUALITY, 90]
    success, enc = cv2.imencode(ext, img, params)
    if success:
        enc.tofile(tmp_path)
        os.replace(tmp_path, path)
        return True
    return False


def compute_regions(
    sec_mask: np.ndarray, face_mask: np.ndarray, occlusion: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_bin = sec_mask > 0
    f_bin = face_mask > 0
    o_bin = occlusion > 0
    outside_region = s_bin & (~f_bin)
    restore_region = outside_region & (~o_bin)
    face_region = s_bin & f_bin
    return outside_region, restore_region, face_region


def analyze_components(
    restore_region: np.ndarray, face_width: float, face_area: int, cfg: RestoreConfig
) -> Tuple[List[ComponentInfo], np.ndarray]:
    h, w = restore_region.shape[:2]
    restore_u8 = (restore_region * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(restore_u8, connectivity=8)

    effective_face_w = face_width if face_width > 10.0 else float(w * 0.20)
    effective_face_area = face_area if face_area > 100 else int(effective_face_w * effective_face_w)
    thin_max_r_thresh = max(cfg.thin_radius_px, effective_face_w * cfg.thin_radius_ratio)
    small_area_thresh = max(cfg.small_area_px, int(effective_face_area * cfg.small_area_ratio))

    components: List[ComponentInfo] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        comp_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        bbox = (x, y, x + comp_w, y + comp_h)

        comp_mask_u8 = (labels[y:y + comp_h, x:x + comp_w] == label_id).astype(np.uint8)
        dist = cv2.distanceTransform(comp_mask_u8, cv2.DIST_L2, 5)
        max_r = float(dist.max())
        mean_r = float(dist[comp_mask_u8 > 0].mean()) if area > 0 else 0.0
        is_thin = max_r <= thin_max_r_thresh
        is_small = area <= small_area_thresh
        classification = "spatial" if (is_thin or is_small or cfg.disable_temporal) else "temporal"
        components.append(ComponentInfo(label_id, area, bbox, max_r, mean_r, is_thin, is_small, classification))
    return components, labels


def compute_adaptive_dilation(face_width: float, frame_w: int, cfg: RestoreConfig) -> int:
    if cfg.dilate_pixels is not None:
        return max(0, cfg.dilate_pixels)
    effective_face_w = face_width if face_width > 10.0 else float(frame_w * 0.20)
    return max(cfg.min_dilate_px, min(cfg.max_dilate_px, int(round(effective_face_w * cfg.dilate_ratio))))


def compute_adaptive_inpaint_radius(face_width: float, frame_w: int, cfg: RestoreConfig) -> int:
    if cfg.inpaint_radius is not None:
        return max(1, cfg.inpaint_radius)
    effective_face_w = face_width if face_width > 10.0 else float(frame_w * 0.20)
    return max(1, min(15, int(round(effective_face_w * cfg.inpaint_radius_ratio))))


def build_safe_component_mask(
    comp_mask: np.ndarray, face_mask_bin: np.ndarray, occlusion_bin: np.ndarray, dilate_px: int
) -> np.ndarray:
    if dilate_px > 0:
        k_size = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(comp_mask.astype(np.uint8), kernel) > 0
    else:
        dilated = comp_mask
    return dilated & (~face_mask_bin) & (~occlusion_bin)


def run_spatial_inpaint_patch(
    frame: np.ndarray,
    target_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    inpaint_radius: int,
    method_name: str = "telea",
    forbid_donor_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fixed: masks out forbid_donor_mask (sec_mask) during cv2.inpaint so mask colors never bleed into neck/bg."""
    if np.count_nonzero(target_mask) == 0:
        return frame

    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    margin = max(20, inpaint_radius * 3)
    rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
    rx2, ry2 = min(W, x2 + margin), min(H, y2 + margin)

    roi_img = frame[ry1:ry2, rx1:rx2].copy()
    mask_roi = target_mask[ry1:ry2, rx1:rx2]

    if forbid_donor_mask is not None:
        inpaint_zone = mask_roi | forbid_donor_mask[ry1:ry2, rx1:rx2]
    else:
        inpaint_zone = mask_roi

    roi_mask = (inpaint_zone * 255).astype(np.uint8)
    method = cv2.INPAINT_TELEA if method_name.lower() == "telea" else cv2.INPAINT_NS
    inpainted_roi = cv2.inpaint(roi_img, roi_mask, inpaint_radius, method)

    out = frame.copy()
    roi_current = out[ry1:ry2, rx1:rx2]
    roi_current[mask_roi] = inpainted_roi[mask_roi]
    out[ry1:ry2, rx1:rx2] = roi_current
    return out


class TemporalDonorEngine:
    def __init__(self, cfg: RestoreConfig):
        self.cfg = cfg
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.dis.setUseSpatialPropagation(True)

    def restore_component(
        self,
        curr_frame: np.ndarray,
        safe_comp_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        curr_rec: FrameRecord,
        buffer_records: Dict[int, FrameRecord],
        inpaint_radius: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], float, bool, Optional[str]]:
        H, W = curr_frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad = max(48, int(curr_rec.face_width * 0.35)) if curr_rec.face_width > 10 else 64
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(W, x2 + pad), min(H, y2 + pad)

        sec_full = curr_rec.sec_mask > 0
        # Pre-clean current frame ROI so DIS optical flow tracks background rather than the mask
        curr_clean_full = run_spatial_inpaint_patch(
            curr_frame, safe_comp_mask, bbox, inpaint_radius, self.cfg.inpaint_method, forbid_donor_mask=sec_full
        )
        curr_roi = curr_clean_full[ry1:ry2, rx1:rx2]
        curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
        comp_roi = safe_comp_mask[ry1:ry2, rx1:rx2]
        sec_roi = sec_full[ry1:ry2, rx1:rx2]
        occl_roi = curr_rec.occlusion[ry1:ry2, rx1:rx2] > 0

        H_roi, W_roi = curr_roi.shape[:2]
        grid_y, grid_x = np.indices((H_roi, W_roi), dtype=np.float32)

        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        ring = (cv2.dilate(comp_roi.astype(np.uint8), ring_kernel) > 0) & (~comp_roi)
        sobel_curr_x = cv2.Sobel(curr_gray, cv2.CV_32F, 1, 0)
        sobel_curr_y = cv2.Sobel(curr_gray, cv2.CV_32F, 0, 1)

        candidate_indices = [
            k for k in sorted(buffer_records.keys(), key=lambda idx: abs(idx - curr_rec.idx))
            if k != curr_rec.idx
        ]
        valid_donors: List[Tuple[int, np.ndarray, np.ndarray, float]] = []

        for cand_idx in candidate_indices:
            cand_rec = buffer_records[cand_idx]
            cand_roi = cand_rec.frame[ry1:ry2, rx1:rx2]
            cand_gray = cv2.cvtColor(cand_roi, cv2.COLOR_BGR2GRAY)
            cand_sec_roi = cand_rec.sec_mask[ry1:ry2, rx1:rx2] > 0
            cand_occl_roi = cand_rec.occlusion[ry1:ry2, rx1:rx2] > 0

            flow = self.dis.calc(curr_gray, cand_gray, None)
            map_x = grid_x + flow[..., 0]
            map_y = grid_y + flow[..., 1]
            in_bounds = (map_x >= 0.0) & (map_x <= float(W_roi - 1)) & (map_y >= 0.0) & (map_y <= float(H_roi - 1))

            rev_flow = self.dis.calc(cand_gray, curr_gray, None)
            remap_rev_x = cv2.remap(rev_flow[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            remap_rev_y = cv2.remap(rev_flow[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            fb_err = np.sqrt((flow[..., 0] + remap_rev_x) ** 2 + (flow[..., 1] + remap_rev_y) ** 2)

            warped_donor_roi = cv2.remap(cand_roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            warped_sec_roi = cv2.remap(cand_sec_roi.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST) > 0
            warped_occl_roi = cv2.remap(cand_occl_roi.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST) > 0

            valid_ring = ring & in_bounds & (~sec_roi) & (~occl_roi) & (~warped_sec_roi) & (~warped_occl_roi)
            if int(np.count_nonzero(valid_ring)) < 15:
                continue

            photo_diff = np.abs(curr_roi.astype(np.float32) - warped_donor_roi.astype(np.float32))
            photo_err = float(photo_diff[valid_ring].mean())

            warped_gray = cv2.cvtColor(warped_donor_roi, cv2.COLOR_BGR2GRAY)
            sobel_cand_x = cv2.Sobel(warped_gray, cv2.CV_32F, 1, 0)
            sobel_cand_y = cv2.Sobel(warped_gray, cv2.CV_32F, 0, 1)
            grad_err = float((np.abs(sobel_curr_x - sobel_cand_x) + np.abs(sobel_curr_y - sobel_cand_y))[valid_ring].mean())
            flow_err = float(fb_err[valid_ring].mean())

            conf = float(np.exp(-(photo_err / self.cfg.photo_thresh + grad_err / self.cfg.grad_thresh + flow_err / self.cfg.flow_thresh)))
            if photo_err <= self.cfg.photo_thresh and flow_err <= self.cfg.flow_thresh and conf >= self.cfg.min_donor_confidence:
                donor_px_roi = comp_roi & in_bounds & (~warped_sec_roi) & (~warped_occl_roi) & (fb_err <= self.cfg.flow_thresh * 1.5)
                if np.count_nonzero(donor_px_roi) > 0:
                    valid_donors.append((cand_idx, warped_donor_roi, donor_px_roi, conf))

        temporal_mask = np.zeros((H, W), dtype=bool)
        if not valid_donors:
            return curr_clean_full, temporal_mask, [], 1.0, True, "no_valid_temporal_donor"

        used_indices = [d[0] for d in valid_donors]
        avg_conf = float(np.mean([d[3] for d in valid_donors]))
        donor_imgs = [d[1] for d in valid_donors]
        donor_masks = [d[2] for d in valid_donors]

        stacked_masks = np.stack(donor_masks, axis=0)
        stacked_imgs = np.stack(donor_imgs, axis=0)
        temporal_covered_roi = np.sum(stacked_masks, axis=0) > 0

        blended_roi = curr_roi.copy()
        if len(donor_imgs) == 1:
            blended_roi[temporal_covered_roi] = donor_imgs[0][temporal_covered_roi]
        else:
            float_stack = stacked_imgs.astype(np.float32)
            float_stack[~stacked_masks] = np.nan
            median_val = np.nanmedian(float_stack, axis=0)
            valid_med = ~np.isnan(median_val)
            blended_roi[valid_med] = np.clip(median_val[valid_med], 0, 255).astype(np.uint8)

        out_frame = curr_clean_full.copy()
        patch = out_frame[ry1:ry2, rx1:rx2].copy()
        patch[temporal_covered_roi] = blended_roi[temporal_covered_roi]
        out_frame[ry1:ry2, rx1:rx2] = patch

        temporal_mask_roi = np.zeros((H_roi, W_roi), dtype=bool)
        temporal_mask_roi[temporal_covered_roi] = True
        temporal_mask[ry1:ry2, rx1:rx2] = temporal_mask_roi
        return out_frame, temporal_mask, used_indices, avg_conf, False, None


def create_task4_debug_panel(
    orig_frame: np.ndarray,
    restored_frame: np.ndarray,
    face_region: np.ndarray,
    outside_region: np.ndarray,
    occlusion: np.ndarray,
    modified_mask: np.ndarray,
    meta_info: dict,
) -> np.ndarray:
    H, W = orig_frame.shape[:2]
    scale = min(1.0, 960.0 / max(W, 1))
    disp_w, disp_h = int(round(W * scale)), int(round(H * scale))

    def rsz(im: np.ndarray):
        return cv2.resize(im, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    left = orig_frame.astype(np.float32)
    alpha = 0.45
    if np.any(outside_region):
        left[outside_region] = (1.0 - alpha) * left[outside_region] + alpha * np.array([255, 255, 0], np.float32)
    if np.any(face_region):
        left[face_region] = (1.0 - alpha) * left[face_region] + alpha * np.array([0, 0, 255], np.float32)
    if np.any(occlusion > 0):
        left[occlusion > 0] = (1.0 - alpha) * left[occlusion > 0] + alpha * np.array([255, 0, 0], np.float32)

    diff = np.abs(restored_frame.astype(np.float32) - orig_frame.astype(np.float32))
    diff_mag = np.clip(diff.mean(axis=-1) * 4.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_mag, cv2.COLORMAP_JET)
    right = (orig_frame * 0.3 + heatmap * 0.7).astype(np.uint8)
    if np.any(modified_mask):
        contours, _ = cv2.findContours(modified_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(right, contours, -1, (0, 255, 255), 2)

    canvas = np.hstack([rsz(np.clip(left, 0, 255).astype(np.uint8)), rsz(restored_frame), rsz(right)])
    banner = np.zeros((40, canvas.shape[1], 3), dtype=np.uint8)
    text = (
        f"Frame {meta_info.get('frame_id', 0):06d} | Outside: {meta_info.get('outside_pixels', 0)} px | "
        f"Restored: {meta_info.get('restore_pixels', 0)} px | Donors: {meta_info.get('temporal_donor_frames', [])}"
    )
    cv2.putText(banner, text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, canvas])


class OutsideRestorationPipeline:
    def __init__(self, cfg: RestoreConfig):
        self.cfg = cfg
        self.reader = VideoReader(cfg.video_path)
        self.sec_mask_dir, self.occl_dir, self.face_mask_dir = resolve_subdirectories(cfg.sec4b_dir, cfg.tddfa_dir)

        self.frames_dir = os.path.join(cfg.output_dir, "frames")
        self.masks_dir = os.path.join(cfg.output_dir, "masks")
        self.debug_dir = os.path.join(cfg.output_dir, "debug")
        self.meta_path = os.path.join(cfg.output_dir, "metadata.jsonl")

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        if cfg.debug_every > 0:
            os.makedirs(self.debug_dir, exist_ok=True)

        self.temporal_engine = None if cfg.disable_temporal else TemporalDonorEngine(cfg)
        self.K = 0 if cfg.disable_temporal else max(1, cfg.temporal_window // 2)
        self.buffer: Dict[int, FrameRecord] = collections.OrderedDict()
        self.completed_frames: Set[int] = set()
        if cfg.resume and not cfg.overwrite:
            self._scan_existing_outputs()

    def _scan_existing_outputs(self) -> None:
        if os.path.isdir(self.frames_dir) and os.path.isdir(self.masks_dir):
            common = set(os.listdir(self.frames_dir)).intersection(set(os.listdir(self.masks_dir)))
            for fname in common:
                if fname.lower().endswith(".png") and os.path.splitext(fname)[0].isdigit():
                    self.completed_frames.add(int(os.path.splitext(fname)[0]))

    def _load_record(self, idx: int) -> Optional[FrameRecord]:
        frame = self.reader.read_frame(idx)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        sec_mask = load_binary_mask(self.sec_mask_dir, idx, (h, w))
        face_mask = load_binary_mask(self.face_mask_dir, idx, (h, w))
        if face_mask is not None and np.any(face_mask):
            contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(face_mask, contours, -1, 255, -1)
        occl_mask = load_binary_mask(self.occl_dir, idx, (h, w))

        face_area = int(np.count_nonzero(face_mask))
        if face_area > 0:
            fx, fy, fw, fh = cv2.boundingRect(face_mask)
            return FrameRecord(idx, frame, sec_mask, face_mask, occl_mask, float(fw), float(fh), face_area, (fx, fy, fw, fh))
        return FrameRecord(idx, frame, sec_mask, face_mask, occl_mask, float(w * 0.20), float(h * 0.25), 0, None)

    def _ensure_buffer(self, curr_idx: int) -> None:
        if self.cfg.disable_temporal:
            self.buffer.clear()
            rec = self._load_record(curr_idx)
            if rec is not None:
                self.buffer[curr_idx] = rec
            return
        target_start = max(0, curr_idx - self.K)
        target_end = min(self.reader.total_frames - 1, curr_idx + self.K)
        for k in [k for k in self.buffer.keys() if k < target_start or k > target_end]:
            del self.buffer[k]
        for f_idx in range(target_start, target_end + 1):
            if f_idx not in self.buffer:
                rec = self._load_record(f_idx)
                if rec is not None:
                    self.buffer[f_idx] = rec

    def process_frame(self, curr_idx: int) -> dict:
        self._ensure_buffer(curr_idx)
        curr_rec = self.buffer.get(curr_idx)
        if curr_rec is None:
            raise RuntimeError(f"Could not read frame {curr_idx}")

        raw_frame = curr_rec.frame
        sec_mask, face_mask, occlusion = curr_rec.sec_mask, curr_rec.face_mask, curr_rec.occlusion
        outside_region, restore_region, face_region = compute_regions(sec_mask, face_mask, occlusion)

        outside_px_count = int(np.count_nonzero(outside_region))
        restore_px_count = int(np.count_nonzero(restore_region))

        if outside_px_count == 0 or restore_px_count == 0:
            out_frame = raw_frame.copy()
            out_mask = np.zeros(raw_frame.shape[:2], dtype=np.uint8)
            meta = {
                "frame_id": curr_idx, "status": "no_outside", "outside_pixels": 0, "restore_pixels": 0,
                "component_count": 0, "spatial_pixels": 0, "temporal_pixels": 0,
                "temporal_donor_frames": [], "confidence": 1.0, "fallback": False, "fallback_reason": None,
            }
            self._save_results(curr_idx, out_frame, out_mask, meta, curr_rec, outside_region, face_region)
            return meta

        components, labels = analyze_components(restore_region, curr_rec.face_width, curr_rec.face_area, self.cfg)
        adaptive_dilate = compute_adaptive_dilation(curr_rec.face_width, raw_frame.shape[1], self.cfg)
        adaptive_inpaint_r = compute_adaptive_inpaint_radius(curr_rec.face_width, raw_frame.shape[1], self.cfg)

        working_frame = raw_frame.copy()
        total_spatial_pixels = total_temporal_pixels = 0
        all_donors: Set[int] = set()
        confidences: List[float] = []
        had_any_fallback = False
        fallback_reasons: List[str] = []

        if self.cfg.disable_temporal:
            safe_comp_mask = build_safe_component_mask(restore_region, face_mask > 0, occlusion > 0, adaptive_dilate)
            if np.count_nonzero(safe_comp_mask) > 0:
                comp_ys, comp_xs = np.where(safe_comp_mask)
                comp_bbox = (int(comp_xs.min()), int(comp_ys.min()), int(comp_xs.max() + 1), int(comp_ys.max() + 1))
                working_frame = run_spatial_inpaint_patch(
                    working_frame, safe_comp_mask, comp_bbox, adaptive_inpaint_r,
                    self.cfg.inpaint_method, forbid_donor_mask=(sec_mask > 0),
                )
                total_spatial_pixels = int(np.count_nonzero(safe_comp_mask))
                confidences.append(1.0)
        else:
            for comp in components:
                comp_mask = labels == comp.label_id
                safe_comp_mask = build_safe_component_mask(comp_mask, face_mask > 0, occlusion > 0, adaptive_dilate)
                if np.count_nonzero(safe_comp_mask) == 0:
                    continue

                comp_ys, comp_xs = np.where(safe_comp_mask)
                comp_bbox = (int(comp_xs.min()), int(comp_ys.min()), int(comp_xs.max() + 1), int(comp_ys.max() + 1))

                if comp.classification == "spatial":
                    working_frame = run_spatial_inpaint_patch(
                        working_frame, safe_comp_mask, comp_bbox, adaptive_inpaint_r,
                        self.cfg.inpaint_method, forbid_donor_mask=(sec_mask > 0),
                    )
                    total_spatial_pixels += int(np.count_nonzero(safe_comp_mask))
                    confidences.append(1.0)
                else:
                    working_frame, temp_mask, used_donors, conf, fallback, fb_reason = self.temporal_engine.restore_component(
                        working_frame, safe_comp_mask, comp_bbox, curr_rec, self.buffer, adaptive_inpaint_r
                    )
                    total_temporal_pixels += int(np.count_nonzero(temp_mask))
                    total_spatial_pixels += int(np.count_nonzero(safe_comp_mask & (~temp_mask)))
                    all_donors.update(used_donors)
                    confidences.append(conf)
                    if fallback:
                        had_any_fallback = True
                        if fb_reason:
                            fallback_reasons.append(fb_reason)

        working_frame[face_region] = raw_frame[face_region]
        working_frame[occlusion > 0] = raw_frame[occlusion > 0]

        diff_mask = np.any(working_frame != raw_frame, axis=-1)
        out_mask = np.where(diff_mask, np.uint8(255), np.uint8(0))

        assert int(np.count_nonzero(diff_mask & face_region)) == 0
        assert int(np.count_nonzero(diff_mask & (occlusion > 0))) == 0

        meta = {
            "frame_id": curr_idx, "status": "ok", "outside_pixels": outside_px_count,
            "restore_pixels": int(np.count_nonzero(diff_mask)), "component_count": len(components),
            "spatial_pixels": total_spatial_pixels, "temporal_pixels": total_temporal_pixels,
            "temporal_donor_frames": sorted(list(all_donors)),
            "confidence": float(round(np.mean(confidences) if confidences else 1.0, 4)),
            "fallback": had_any_fallback,
            "fallback_reason": "; ".join(fallback_reasons) if fallback_reasons else None,
        }
        self._save_results(curr_idx, working_frame, out_mask, meta, curr_rec, outside_region, face_region)
        return meta

    def _save_results(self, frame_idx: int, restored_frame: np.ndarray, out_mask: np.ndarray,
                      meta: dict, rec: FrameRecord, outside_region: np.ndarray, face_region: np.ndarray,
                      meta_lock: Optional[threading.Lock] = None) -> None:
        save_image_unicode(os.path.join(self.frames_dir, f"{frame_idx:06d}.png"), restored_frame, is_png=True)
        save_image_unicode(os.path.join(self.masks_dir, f"{frame_idx:06d}.png"), out_mask, is_png=True)
        if self.cfg.debug_every > 0 and (frame_idx % self.cfg.debug_every == 0):
            panel = create_task4_debug_panel(rec.frame, restored_frame, face_region, outside_region, rec.occlusion, out_mask > 0, meta)
            save_image_unicode(os.path.join(self.debug_dir, f"{frame_idx:06d}.jpg"), panel, is_png=False)
        line = json.dumps(meta, ensure_ascii=False) + "\n"
        if meta_lock is not None:
            with meta_lock:
                with open(self.meta_path, "a", encoding="utf-8") as f:
                    f.write(line)
        else:
            with open(self.meta_path, "a", encoding="utf-8") as f:
                f.write(line)

    def _worker_process_frame(self, curr_idx: int, raw_frame: np.ndarray, meta_lock: threading.Lock) -> dict:
        H, W = raw_frame.shape[:2]
        sec_mask = load_binary_mask(self.sec_mask_dir, curr_idx, (H, W))
        face_mask = load_binary_mask(self.face_mask_dir, curr_idx, (H, W))
        if face_mask is not None and np.any(face_mask):
            contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(face_mask, contours, -1, 255, -1)
        occl_mask = load_binary_mask(self.occl_dir, curr_idx, (H, W))

        face_area = int(np.count_nonzero(face_mask))
        if face_area > 0:
            fx, fy, fw, fh = cv2.boundingRect(face_mask)
            curr_rec = FrameRecord(curr_idx, raw_frame, sec_mask, face_mask, occl_mask, float(fw), float(fh), face_area, (fx, fy, fw, fh))
        else:
            curr_rec = FrameRecord(curr_idx, raw_frame, sec_mask, face_mask, occl_mask, float(W * 0.20), float(H * 0.25), 0, None)

        outside_region, restore_region, face_region = compute_regions(sec_mask, face_mask, occl_mask)
        outside_px_count = int(np.count_nonzero(outside_region))
        restore_px_count = int(np.count_nonzero(restore_region))

        if outside_px_count == 0 or restore_px_count == 0:
            out_frame = raw_frame.copy()
            out_mask = np.zeros((H, W), dtype=np.uint8)
            meta = {
                "frame_id": curr_idx, "status": "no_outside", "outside_pixels": 0, "restore_pixels": 0,
                "component_count": 0, "spatial_pixels": 0, "temporal_pixels": 0,
                "temporal_donor_frames": [], "confidence": 1.0, "fallback": False, "fallback_reason": None,
            }
            self._save_results(curr_idx, out_frame, out_mask, meta, curr_rec, outside_region, face_region, meta_lock)
            return meta

        adaptive_dilate = compute_adaptive_dilation(curr_rec.face_width, W, self.cfg)
        adaptive_inpaint_r = compute_adaptive_inpaint_radius(curr_rec.face_width, W, self.cfg)

        working_frame = raw_frame.copy()
        safe_comp_mask = build_safe_component_mask(restore_region, face_mask > 0, occl_mask > 0, adaptive_dilate)
        total_spatial_pixels = 0
        if np.count_nonzero(safe_comp_mask) > 0:
            comp_ys, comp_xs = np.where(safe_comp_mask)
            comp_bbox = (int(comp_xs.min()), int(comp_ys.min()), int(comp_xs.max() + 1), int(comp_ys.max() + 1))
            working_frame = run_spatial_inpaint_patch(
                working_frame, safe_comp_mask, comp_bbox, adaptive_inpaint_r,
                self.cfg.inpaint_method, forbid_donor_mask=(sec_mask > 0),
            )
            total_spatial_pixels = int(np.count_nonzero(safe_comp_mask))

        working_frame[face_region] = raw_frame[face_region]
        working_frame[occl_mask > 0] = raw_frame[occl_mask > 0]

        diff_mask = np.any(working_frame != raw_frame, axis=-1)
        out_mask = np.where(diff_mask, np.uint8(255), np.uint8(0))

        meta = {
            "frame_id": curr_idx, "status": "ok", "outside_pixels": outside_px_count,
            "restore_pixels": int(np.count_nonzero(diff_mask)), "component_count": 1,
            "spatial_pixels": total_spatial_pixels, "temporal_pixels": 0,
            "temporal_donor_frames": [], "confidence": 1.0, "fallback": False, "fallback_reason": None,
        }
        self._save_results(curr_idx, working_frame, out_mask, meta, curr_rec, outside_region, face_region, meta_lock)
        return meta

    def run(self) -> None:
        total_frames = self.reader.total_frames
        start_f = max(0, self.cfg.start_frame)
        end_f = min(total_frames, self.cfg.end_frame) if self.cfg.end_frame is not None else total_frames
        t_start = time.time()
        processed_count = skipped_count = 0

        if self.cfg.disable_temporal:
            meta_lock = threading.Lock()
            max_workers = min(16, max(4, os.cpu_count() or 8))
            logger.info(f"[*] Starting 13900K High-Performance Multi-Threaded Engine with {max_workers} worker threads on {os.cpu_count()} CPU threads...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for curr_idx in range(start_f, end_f):
                    if self.cfg.resume and not self.cfg.overwrite and curr_idx in self.completed_frames:
                        skipped_count += 1
                        continue

                    raw_frame = self.reader.read_frame(curr_idx)
                    if raw_frame is None:
                        continue

                    fut = executor.submit(self._worker_process_frame, curr_idx, raw_frame, meta_lock)
                    futures[fut] = curr_idx

                    while len(futures) >= max_workers * 4:
                        done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                        for d in done:
                            fid = futures.pop(d)
                            meta = d.result()
                            processed_count += 1
                            if processed_count % self.cfg.log_every == 0 or processed_count == (end_f - start_f - skipped_count):
                                fps = processed_count / max(time.time() - t_start, 0.001)
                                logger.info(f"Frame [{fid:06d}/{end_f:06d}] | Speed: {fps:.2f} fps | Restored: {meta.get('restore_pixels', 0)} px")

                for fut in concurrent.futures.as_completed(futures.keys()):
                    fid = futures[fut]
                    meta = fut.result()
                    processed_count += 1
                    if processed_count % self.cfg.log_every == 0 or processed_count == (end_f - start_f - skipped_count):
                        fps = processed_count / max(time.time() - t_start, 0.001)
                        logger.info(f"Frame [{fid:06d}/{end_f:06d}] | Speed: {fps:.2f} fps | Restored: {meta.get('restore_pixels', 0)} px")
        else:
            for curr_idx in range(start_f, end_f):
                if self.cfg.resume and not self.cfg.overwrite and curr_idx in self.completed_frames:
                    skipped_count += 1
                    continue
                meta = self.process_frame(curr_idx)
                processed_count += 1
                if processed_count % self.cfg.log_every == 0 or curr_idx == end_f - 1:
                    fps = processed_count / max(time.time() - t_start, 0.001)
                    logger.info(f"Frame [{curr_idx:06d}/{end_f:06d}] | Speed: {fps:.2f} fps | Restored: {meta['restore_pixels']} px")
                if processed_count % 100 == 0:
                    gc.collect()

        self.reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 4: Restore outside-face region")
    parser.add_argument("--video", required=True)
    parser.add_argument("--sec4b-dir", dest="sec4b_dir", required=True)
    parser.add_argument("--3ddfa-dir", dest="tddfa_dir", required=True)
    parser.add_argument("--output", "--output-dir", dest="output", default="output/outside_restore")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--temporal-window", type=int, default=7)
    parser.add_argument("--disable-temporal", action="store_true", default=False)
    parser.add_argument("--inpaint-method", choices=["telea", "ns"], default="telea")
    parser.add_argument("--inpaint-radius", type=int, default=None)
    parser.add_argument("--dilate-pixels", type=int, default=None)
    parser.add_argument("--dilate-ratio", type=float, default=0.015)
    parser.add_argument("--debug-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = RestoreConfig(
        video_path=args.video, sec4b_dir=args.sec4b_dir, tddfa_dir=args.tddfa_dir,
        output_dir=args.output, start_frame=args.start, end_frame=args.end,
        resume=args.resume, overwrite=args.overwrite, temporal_window=args.temporal_window,
        disable_temporal=args.disable_temporal, debug_every=args.debug_every,
        inpaint_method=args.inpaint_method, inpaint_radius=args.inpaint_radius,
        dilate_pixels=args.dilate_pixels, dilate_ratio=args.dilate_ratio, log_every=args.log_every,
    )
    OutsideRestorationPipeline(cfg).run()


if __name__ == "__main__":
    main()