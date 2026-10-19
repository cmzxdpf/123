#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 2: Build regional & global OpenCV-LAB skin profile from unmasked reference images."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread_bgr(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size > 0 else None
    except Exception:
        return None


def detect_face_bbox(img_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> Tuple[int, int, int, int]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        return int(x), int(y), int(x + fw), int(y + fh)
    # Fallback to central portrait crop if reference image is already a tight face crop
    return int(w * 0.15), int(h * 0.12), int(w * 0.85), int(h * 0.88)


def robust_lab_stats(lab_pixels: np.ndarray) -> Optional[Dict[str, List[float]]]:
    if lab_pixels.shape[0] < 30:
        return None
    l_vals = lab_pixels[:, 0]
    lo, hi = np.percentile(l_vals, [15.0, 88.0])
    keep = (l_vals >= lo) & (l_vals <= hi)
    filtered = lab_pixels[keep] if np.any(keep) else lab_pixels
    med = np.median(filtered, axis=0)
    std = np.std(filtered, axis=0)
    return {
        "lab_mean": [round(float(med[0]), 3), round(float(med[1]), 3), round(float(med[2]), 3)],
        "lab_std": [round(float(std[0]), 3), round(float(std[1]), 3), round(float(std[2]), 3)],
    }


def extract_skin_mask_roi(roi_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    l_ch = lab[:, :, 0]
    skin = (cr >= 133) & (cr <= 178) & (cb >= 75) & (cb <= 130) & (l_ch >= 45) & (l_ch <= 245)
    return skin


def main() -> None:
    p = argparse.ArgumentParser(description="Task 2: Build skin_profile.json from reference faces")
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("output/skin_profile/skin_profile.json"))
    args = p.parse_args()

    files = sorted(f for f in args.reference_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No reference images found in {args.reference_dir}")

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)

    # Normalized sub-boxes (x1, y1, x2, y2) inside the detected face box
    region_boxes = {
        "forehead": (0.28, 0.08, 0.72, 0.26),
        "left_cheek": (0.14, 0.45, 0.38, 0.68),
        "right_cheek": (0.62, 0.45, 0.86, 0.68),
        "nose_lower": (0.40, 0.48, 0.60, 0.66),
        "chin": (0.35, 0.78, 0.65, 0.94),
    }

    collected: Dict[str, List[np.ndarray]] = {k: [] for k in region_boxes}
    global_samples: List[np.ndarray] = []

    for fp in files:
        img = imread_bgr(fp)
        if img is None:
            continue
        x1, y1, x2, y2 = detect_face_bbox(img, cascade)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        rh, rw = roi.shape[:2]
        skin_mask = extract_skin_mask_roi(roi)
        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

        if np.count_nonzero(skin_mask) >= 50:
            global_samples.append(roi_lab[skin_mask])

        for name, (rx1, ry1, rx2, ry2) in region_boxes.items():
            sx1, sy1 = int(rw * rx1), int(rh * ry1)
            sx2, sy2 = int(rw * rx2), int(rh * ry2)
            sub_mask = skin_mask[sy1:sy2, sx1:sx2]
            sub_lab = roi_lab[sy1:sy2, sx1:sx2]
            if np.count_nonzero(sub_mask) >= 20:
                collected[name].append(sub_lab[sub_mask])

    if not global_samples:
        raise RuntimeError("Could not extract valid skin pixels from reference images.")

    all_global = np.concatenate(global_samples, axis=0)
    global_stat = robust_lab_stats(all_global)
    assert global_stat is not None

    regions_out: Dict[str, Dict[str, List[float]]] = {}
    for name, chunks in collected.items():
        if chunks:
            st = robust_lab_stats(np.concatenate(chunks, axis=0))
            regions_out[name] = st if st is not None else global_stat
        else:
            regions_out[name] = global_stat

    profile_obj = {
        "lab_format": "opencv",
        "num_reference_images": len(files),
        "global": global_stat,
        "regions": regions_out,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profile_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"[*] Saved skin profile to {args.output}: global LAB={global_stat['lab_mean']}")


if __name__ == "__main__":
    main()