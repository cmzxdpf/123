#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sec4b_postprocess.py - Mask Decomposition, Strap Filtering, Occlusion Detection & Metrics
========================================================================================
Pure video segmentation and mask decomposition:
1. mask_mask: True visible mask area, 3D bulges preserved, optional strap filtering.
2. visible_face_mask: Bare facial skin (forehead, temples, under-eyes, cheeks, nose bridge).
3. occlusion_mask: Foreground occluders over the face region (hands, arms, hair clumps, props).
4. Metrics: face_occlusion_ratio, quality, tracking_ok.
"""

import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_OCCLUDER_SESSION = None
_OCCLUDER_LOAD_ATTEMPTED = False


def get_occluder_session():
    """
    Load Face Occluder ONNX model session once.
    Checks project models/ directory first, then VisoMaster assets directory.
    Uses CPUExecutionProvider for maximum reliability across Windows environments.
    """
    global _OCCLUDER_SESSION, _OCCLUDER_LOAD_ATTEMPTED
    if _OCCLUDER_LOAD_ATTEMPTED:
        return _OCCLUDER_SESSION

    _OCCLUDER_LOAD_ATTEMPTED = True
    candidate_paths = [
        os.path.join(BASE_DIR, "models", "occluder.onnx"),
        r"J:\VisoMaster\VisoMaster\model_assets\occluder.onnx",
        os.path.join(os.path.dirname(BASE_DIR), "VisoMaster", "VisoMaster", "model_assets", "occluder.onnx"),
    ]

    for p in candidate_paths:
        if os.path.isfile(p):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.log_severity_level = 3
                _OCCLUDER_SESSION = ort.InferenceSession(p, sess_options=opts, providers=["CPUExecutionProvider"])
                break
            except Exception as e:
                print(f"[!] Warning: Could not initialize occluder session from {p}: {e}")

    return _OCCLUDER_SESSION


def filter_ear_straps(mask_uint8: np.ndarray, include_straps: bool = False) -> np.ndarray:
    """
    Filter ear straps from mask while preserving 3D bulges and natural contours.
    If include_straps is True, returns mask_uint8 unchanged.
    If False, uses morphological opening and geodesic reconstruction to remove thin ear straps.
    """
    if include_straps or mask_uint8 is None:
        return mask_uint8

    binary = (mask_uint8 > 0).astype(np.uint8)
    area = int(np.sum(binary))
    if area < 200:
        return (binary * 255).astype(np.uint8)

    # Kernel radius scaled to mask size: ear straps are narrow linear features (1~6 pixels)
    # while mask body is large (>30 pixels across)
    r = max(2, min(10, int(0.012 * np.sqrt(area))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    # Opening strips thin straps
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Keep largest connected component as the mask body
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if num_labels <= 1:
        return (binary * 255).astype(np.uint8)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    seed = (labels == largest_label).astype(np.uint8)

    # Morphological reconstruction: dilate seed iteratively constrained within original binary
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    reconstructed = seed.copy()
    for _ in range(r + 2):
        dilated = cv2.dilate(reconstructed, dil_kernel)
        reconstructed = np.bitwise_and(dilated, binary)

    return (reconstructed * 255).astype(np.uint8)


def extract_face_observation_envelope(
    mask_mask: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Estimate the face observation envelope (the expected face boundary)
    anchored to the detected mask.
    """
    envelope = np.zeros((img_h, img_w), dtype=np.uint8)
    ys, xs = np.where(mask_mask > 0)
    if len(ys) == 0:
        return envelope

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w_m = x_max - x_min
    h_m = y_max - y_min

    # The face extends upwards above the mask (forehead/temples/eyes)
    # and slightly to the sides (cheeks/jawline)
    face_top = max(0, y_min - int(1.15 * h_m))
    face_bottom = min(img_h, y_max + int(0.15 * h_m))
    face_left = max(0, x_min - int(0.25 * w_m))
    face_right = min(img_w, x_max + int(0.25 * w_m))

    # Draw rounded convex face oval
    center_x = (face_left + face_right) // 2
    center_y = (face_top + face_bottom) // 2
    axes = ((face_right - face_left) // 2, (face_bottom - face_top) // 2)

    cv2.ellipse(envelope, (center_x, center_y), axes, 0, 0, 360, 255, -1)
    # Combine with mask itself to ensure full coverage
    envelope = np.bitwise_or(envelope, (mask_mask > 0).astype(np.uint8) * 255)
    return envelope


def extract_occlusions(
    raw_bgr: np.ndarray,
    mask_mask: np.ndarray,
    face_envelope: np.ndarray,
) -> np.ndarray:
    """
    Extract temporary foreground occluders over the face observation region
    (hands, arms, hair clumps, foreign objects).
    Uses deep learning Face Occluder model when available, with a robust geometric fallback.
    Guarantees clean faces have ~0 occlusion (not classifying normal eyes/eyebrows/hair as occlusions).
    """
    h, w = raw_bgr.shape[:2]
    occlusion = np.zeros((h, w), dtype=np.uint8)

    ys, xs = np.where(mask_mask > 0)
    if len(ys) == 0:
        return occlusion

    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    hm = ymax - ymin
    xc = (xmin + xmax) // 2

    # Face box bounding the head/face anchored to mask
    fy1 = max(0, int(ymin - 1.15 * hm))
    fy2 = min(h, int(ymax + 0.12 * hm))
    fh = fy2 - fy1
    fw = int(fh * 0.85)
    fx1 = max(0, xc - fw // 2)
    fx2 = min(w, xc + fw // 2)

    if (fy2 - fy1) < 20 or (fx2 - fx1) < 20:
        return occlusion

    session = get_occluder_session()
    if session is not None:
        try:
            crop = raw_bgr[fy1:fy2, fx1:fx2]
            crop_256 = cv2.resize(crop, (256, 256))
            rgb = cv2.cvtColor(crop_256, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

            out = session.run(["output"], {"img": blob})[0][0, 0]
            pred = (out > 0.0).astype(np.uint8)

            # Connected component analysis in 256x256: remove tiny false positives (< 200 px)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred)
            clean_pred = np.zeros_like(pred)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 200:
                    clean_pred[labels == i] = 1

            if np.sum(clean_pred) > 0:
                crop_h, crop_w = fy2 - fy1, fx2 - fx1
                pred_res = cv2.resize(clean_pred * 255, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                occlusion[fy1:fy2, fx1:fx2] = pred_res

                # Constrain to face envelope and its immediate margin
                dil_env = cv2.dilate(face_envelope, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
                occlusion = np.bitwise_and(occlusion, dil_env)

            return occlusion
        except Exception as e:
            print(f"[!] Warning running occluder session: {e}")

    # Fallback when deep learning occluder is unavailable:
    # Only detect large intrusive non-face blobs that invade into the face region from outside,
    # avoiding normal eyes/eyebrows/hairline.
    non_accounted = (face_envelope > 0) & (mask_mask == 0)
    if np.sum(non_accounted) == 0:
        return occlusion

    # In fallback, filter out small facial features (eyes, eyebrows are < 3000 px)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_accounted.astype(np.uint8) * 255)
    min_occ_area = max(4000, int(0.08 * (hm * (xmax - xmin))))
    for i in range(1, num_labels):
        comp_area = stats[i, cv2.CC_STAT_AREA]
        # Only large intrusive blobs invading across face boundary
        if comp_area >= min_occ_area:
            occlusion[labels == i] = 255

    return occlusion


def extract_visible_face(
    raw_bgr: np.ndarray,
    mask_mask: np.ndarray,
    face_envelope: np.ndarray,
    occlusion_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Extract visible facial skin (forehead, temples, cheeks, under-eyes, nose bridge).
    Strictly excludes: mask, occluders (hands, arms), eyes, eyebrows, hair, and background.
    """
    h, w = raw_bgr.shape[:2]
    visible_face = np.zeros((h, w), dtype=np.uint8)

    if np.sum(face_envelope > 0) < 100:
        return visible_face

    # Upper face region: face envelope excluding mask and excluding any detected occlusion
    upper_face_roi = (face_envelope > 0) & (mask_mask == 0)
    if occlusion_mask is not None:
        upper_face_roi = upper_face_roi & (occlusion_mask == 0)

    if np.sum(upper_face_roi) < 50:
        return visible_face

    # Color space conversions: YCrCb and Lab
    ycrcb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2LAB)

    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    # Skin color chromaticity filter
    skin_cand = upper_face_roi & (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 128) & (L >= 45)

    if np.sum(skin_cand) >= 50:
        # Compute median skin tone in Lab space from the upper face region
        skin_A = np.median(A[skin_cand])
        skin_B = np.median(B[skin_cand])
        skin_L = np.median(L[skin_cand])

        # Distance from median chromaticity
        dist_ab = np.sqrt((A.astype(np.float32) - skin_A) ** 2 + (B.astype(np.float32) - skin_B) ** 2)

        # Exclude dark features: eyes, pupils, eyebrows, hair (L < skin_L - 30.0)
        # Exclude non-skin colors (dist_ab > 18.0)
        valid_skin = (
            upper_face_roi
            & (dist_ab <= 18.0)
            & (L >= max(40.0, skin_L - 32.0))
            & (mask_mask == 0)
        )
    else:
        # Fallback to standard YCrCb skin threshold if upper face sampling was sparse
        valid_skin = upper_face_roi & (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 128) & (L >= 45)

    if occlusion_mask is not None:
        valid_skin = valid_skin & (occlusion_mask == 0)

    # Clean up with morphology: 3x3 opening removes single-pixel noise, 3x3 closing fills tiny pores
    skin_u8 = valid_skin.astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_clean = cv2.morphologyEx(skin_u8, cv2.MORPH_OPEN, k3)

    return skin_clean


def compute_frame_metrics(
    frame_id: int,
    mask_mask: np.ndarray,
    visible_face_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    prev_mask: np.ndarray = None,
    overlap_conflict: bool = False,
) -> dict:
    """
    Compute all required standard metadata fields:
    - frame_id
    - mask_area
    - visible_face_area
    - occlusion_area
    - face_occlusion_ratio (occlusion_area / estimated_face_area)
    - quality (0.0 ~ 1.0)
    - tracking_ok (bool)
    """
    mask_area = int(np.sum(mask_mask > 0))
    face_area = int(np.sum(visible_face_mask > 0))
    occl_area = int(np.sum(occlusion_mask > 0))

    estimated_face_area = mask_area + face_area + occl_area

    if estimated_face_area > 0:
        face_occl_ratio = round(float(occl_area) / float(estimated_face_area), 4)
    else:
        face_occl_ratio = 0.0

    face_occl_ratio = min(1.0, max(0.0, face_occl_ratio))

    # tracking_ok: True if mask is detected with reasonable area
    tracking_ok = bool(mask_area >= 150)

    # Compute quality score:
    # 1. Base confidence from mask presence and shape
    if not tracking_ok:
        quality = 0.0
    else:
        # Perimeter and compactness
        contours, _ = cv2.findContours(mask_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
            compactness = min(1.0, (4.0 * np.pi * mask_area) / max(1.0, perimeter ** 2))
        else:
            compactness = 0.5

        # Temporal IoU with previous mask
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
    """
    Full pipeline to decompose raw tracking output into the 3 standardized masks:
    Returns (mask_mask, visible_face_mask, occlusion_mask).
    1. Filter straps from tracked mask.
    2. Extract face envelope.
    3. Detect foreground occlusions (hands, arms, props).
    4. Extract visible bare face skin (strictly excluding mask AND occluders).
    """
    h, w = raw_bgr.shape[:2]

    # 1. Extract and filter mask
    mask_mask = filter_ear_straps(raw_tracked_mask, include_straps=include_straps)

    # 2. Extract face envelope
    face_envelope = extract_face_observation_envelope(mask_mask, h, w)

    # 3. Extract occlusions first
    occlusion_mask = extract_occlusions(raw_bgr, mask_mask, face_envelope)

    # 4. Extract visible face skin (strictly excluding mask and occluders)
    visible_face_mask = extract_visible_face(raw_bgr, mask_mask, face_envelope, occlusion_mask=occlusion_mask)

    return mask_mask, visible_face_mask, occlusion_mask
