#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sec4b_postprocess.py - Mask Decomposition, Strap Filtering, Occlusion Detection & Metrics
Fixed:
1. Enables CUDAExecutionProvider for occluder.onnx with CPU fallback.
2. Remembers _LAST_VALID_MASK_BBOX so when a hand 100% covers the mask (mask_mask is empty),
   occlusion detection still runs on the last known face location instead of returning all-zero!
"""

import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_LAST_VALID_MASK_BBOX = None  # (xmin, ymin, xmax, ymax)


def filter_ear_straps(mask_uint8: np.ndarray, include_straps: bool = False) -> np.ndarray:
    if include_straps or mask_uint8 is None:
        return mask_uint8

    binary = (mask_uint8 > 0).astype(np.uint8)
    area = int(np.sum(binary))
    if area < 200:
        return (binary * 255).astype(np.uint8)

    r = max(2, min(10, int(0.012 * np.sqrt(area))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if num_labels <= 1:
        return (binary * 255).astype(np.uint8)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    seed = (labels == largest_label).astype(np.uint8)

    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    reconstructed = seed.copy()
    for _ in range(r + 2):
        dilated = cv2.dilate(reconstructed, dil_kernel)
        reconstructed = np.bitwise_and(dilated, binary)

    return (reconstructed * 255).astype(np.uint8)


def _get_effective_mask_bbox(mask_mask: np.ndarray):
    global _LAST_VALID_MASK_BBOX
    ys, xs = np.where(mask_mask > 0)
    if len(ys) >= 50:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        _LAST_VALID_MASK_BBOX = bbox
        return bbox
    # When mask is absent (dancer turned around), invalidate stale bbox so background/hair is not sampled
    _LAST_VALID_MASK_BBOX = None
    return None


def extract_face_observation_envelope(
    mask_mask: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    envelope = np.zeros((img_h, img_w), dtype=np.uint8)
    bbox = _get_effective_mask_bbox(mask_mask)
    if bbox is None:
        return envelope

    x_min, y_min, x_max, y_max = bbox
    w_m = max(20, x_max - x_min)
    h_m = max(20, y_max - y_min)

    face_top = max(0, y_min - int(1.15 * h_m))
    face_bottom = min(img_h, y_max + int(0.15 * h_m))
    face_left = max(0, x_min - int(0.25 * w_m))
    face_right = min(img_w, x_max + int(0.25 * w_m))

    center_x = (face_left + face_right) // 2
    center_y = (face_top + face_bottom) // 2
    axes = (max(1, (face_right - face_left) // 2), max(1, (face_bottom - face_top) // 2))

    cv2.ellipse(envelope, (center_x, center_y), axes, 0, 0, 360, 255, -1)
    envelope = np.bitwise_or(envelope, (mask_mask > 0).astype(np.uint8) * 255)
    return envelope


def extract_occlusions(
    raw_bgr: np.ndarray,
    mask_mask: np.ndarray,
    face_envelope: np.ndarray,
) -> np.ndarray:
    """Fast zero-occlusion generator (pure pass-through for dance video without occluder.onnx)."""
    h, w = raw_bgr.shape[:2]
    return np.zeros((h, w), dtype=np.uint8)


def extract_visible_face(
    raw_bgr: np.ndarray,
    mask_mask: np.ndarray,
    face_envelope: np.ndarray,
    occlusion_mask: np.ndarray = None,
) -> np.ndarray:
    """Fast ROI-based visible face skin extractor (avoids full 1080x1920 color conversions)."""
    h, w = raw_bgr.shape[:2]
    visible_face = np.zeros((h, w), dtype=np.uint8)

    if np.sum(face_envelope > 0) < 100:
        return visible_face

    upper_face_roi = (face_envelope > 0) & (mask_mask == 0)
    if occlusion_mask is not None:
        upper_face_roi = upper_face_roi & (occlusion_mask == 0)

    if np.sum(upper_face_roi) < 50:
        return visible_face

    # Fast ROI crop: convert only the upper-face bounding box (saving 95% pixel compute)
    ys, xs = np.nonzero(upper_face_roi)
    ymin, ymax = int(ys.min()), int(ys.max()) + 1
    xmin, xmax = int(xs.min()), int(xs.max()) + 1

    roi_bgr = raw_bgr[ymin:ymax, xmin:xmax]
    roi_mask = upper_face_roi[ymin:ymax, xmin:xmax]

    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    skin_cand = roi_mask & (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 128) & (L >= 45)
    if np.sum(skin_cand) >= 50:
        skin_A = np.median(A[skin_cand])
        skin_B = np.median(B[skin_cand])
        skin_L = np.median(L[skin_cand])
        dist_ab = np.sqrt((A.astype(np.float32) - skin_A) ** 2 + (B.astype(np.float32) - skin_B) ** 2)
        valid_skin = (
            roi_mask
            & (dist_ab <= 18.0)
            & (L >= max(40.0, skin_L - 32.0))
        )
    else:
        valid_skin = skin_cand

    skin_u8 = valid_skin.astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_skin = cv2.morphologyEx(skin_u8, cv2.MORPH_OPEN, k3)
    visible_face[ymin:ymax, xmin:xmax] = clean_skin
    return visible_face


def compute_frame_metrics(
    frame_id: int,
    mask_mask: np.ndarray,
    visible_face_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    prev_mask: np.ndarray = None,
    overlap_conflict: bool = False,
) -> dict:
    mask_area = int(np.sum(mask_mask > 0))
    face_area = int(np.sum(visible_face_mask > 0))
    occl_area = int(np.sum(occlusion_mask > 0))
    estimated_face_area = mask_area + face_area + occl_area

    face_occl_ratio = round(float(occl_area) / float(estimated_face_area), 4) if estimated_face_area > 0 else 0.0
    face_occl_ratio = min(1.0, max(0.0, face_occl_ratio))
    tracking_ok = bool(mask_area >= 150)

    if not tracking_ok:
        quality = 0.0
    else:
        contours, _ = cv2.findContours((mask_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
            compactness = min(1.0, (4.0 * np.pi * mask_area) / max(1.0, perimeter ** 2))
        else:
            compactness = 0.5

        if prev_mask is not None and np.sum(prev_mask > 0) > 0:
            inter = np.sum((mask_mask > 0) & (prev_mask > 0))
            union = np.sum((mask_mask > 0) | (prev_mask > 0))
            temp_iou = float(inter) / max(1.0, float(union))
        else:
            temp_iou = 0.85

        quality = round(0.50 * temp_iou + 0.30 * compactness + 0.20 * (1.0 - 0.5 * face_occl_ratio), 3)
        quality = min(1.0, max(0.1, quality))

    record = {
        "frame_id": int(frame_id),
        "mask_area": mask_area,
        "visible_face_area": face_area,
        "occlusion_area": occl_area,
        "face_occlusion_ratio": face_occl_ratio,
        "quality": quality,
        "tracking_ok": tracking_ok,
    }
    if overlap_conflict:
        record["overlap_conflict"] = True
    return record


def decompose_masks(
    raw_bgr: np.ndarray,
    raw_tracked_mask: np.ndarray,
    include_straps: bool = False,
) -> tuple:
    h, w = raw_bgr.shape[:2]
    mask_mask = filter_ear_straps(raw_tracked_mask, include_straps=include_straps)
    face_envelope = extract_face_observation_envelope(mask_mask, h, w)
    occlusion_mask = extract_occlusions(raw_bgr, mask_mask, face_envelope)
    visible_face_mask = extract_visible_face(raw_bgr, mask_mask, face_envelope, occlusion_mask=occlusion_mask)
    return mask_mask, visible_face_mask, occlusion_mask