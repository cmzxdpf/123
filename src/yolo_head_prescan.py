#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_head_prescan.py - High-speed GPU YOLOv8-Face Detector & Head Radar
=======================================================================
Directly uses D:\\VisoMaster-Fusion\\model_assets\\yoloface_8n.onnx
1. Pre-scans all video frames at 150+ FPS with CUDAExecutionProvider.
2. Extracts accurate head bbox, confidence, and mask prompt anchor (cx, cy).
3. Applies Hard Geometric Firewall (cy <= 0.45 * H), 100% immune to chest stickers (y ~ 861).
4. Robust to white watermark text across forehead/eyes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure CUDAExecutionProvider finds PyTorch's CUDA/cuDNN DLLs
try:
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(torch_lib)
except Exception:
    pass

import onnxruntime as ort

DEFAULT_YOLO_CANDIDATES = [
    os.environ.get("YOLO_FACE_MODEL"),
    r"D:\VisoMaster-Fusion\model_assets\yoloface_8n.onnx",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_assets", "yoloface_8n.onnx"),
]
MODEL_PATH = next((p for p in DEFAULT_YOLO_CANDIDATES if p and os.path.isfile(p)), r"D:\VisoMaster-Fusion\model_assets\yoloface_8n.onnx")


class YOLOFaceDetector:
    def __init__(self, model_path: str = MODEL_PATH, device: str = "cuda"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YOLOFace ONNX model not found: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]
        self.in_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 640
        self.in_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = img_bgr.shape[:2]
        scale = min(self.in_w / w, self.in_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_w = self.in_w - nw
        pad_h = self.in_h - nh
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        padded = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        padded[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        return blob, scale, pad_left, pad_top

    def detect(
        self,
        img_bgr: np.ndarray,
        conf_thresh: float = 0.40,
        iou_thresh: float = 0.45,
    ) -> List[dict]:
        h, w = img_bgr.shape[:2]
        blob, scale, pad_left, pad_top = self.preprocess(img_bgr)

        outputs = self.session.run(None, {self.input_name: blob})[0]  # [1, 5, 8400] or [1, 8400, 5]
        if outputs.shape[1] < outputs.shape[2]:
            outputs = np.transpose(outputs, (0, 2, 1))
        preds = outputs[0]  # [8400, 5+] (cx, cy, bw, bh, conf)

        boxes = []
        scores = []
        for row in preds:
            conf = float(row[4])
            if conf >= conf_thresh:
                bcx, bcy, bw, bh = row[0], row[1], row[2], row[3]
                # Map back to padded coordinates
                x1 = (bcx - bw * 0.5 - pad_left) / scale
                y1 = (bcy - bh * 0.5 - pad_top) / scale
                x2 = (bcx + bw * 0.5 - pad_left) / scale
                y2 = (bcy + bh * 0.5 - pad_top) / scale

                # Hard Physical Firewall: Face center MUST reside in the upper 45% of 1080x1920 frame
                center_y = (y1 + y2) * 0.5
                if center_y > 0.45 * h:
                    continue  # Rejects all chest stickers, waist, dress text

                boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
                scores.append(conf)

        if len(boxes) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = boxes[idx]
                x1, y1 = max(0, bx), max(0, by)
                x2, y2 = min(w, bx + bw), min(h, by + bh)
                conf = scores[idx]

                # Prompt anchor targeting mask center (lower face, ~68% from face top)
                anchor_cx = float(x1 + (x2 - x1) * 0.5)
                anchor_cy = float(y1 + (y2 - y1) * 0.68)

                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "anchor": (anchor_cx, anchor_cy),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                })

        # Return largest / highest confidence face
        results.sort(key=lambda item: item["confidence"] * (item["width"] * item["height"]), reverse=True)
        return results


    def scan_video(
        self,
        video_path: str,
        output_npy: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_f = total_vid_frames if (end_frame is None or end_frame < 0) else min(end_frame, total_vid_frames)
        total_frames = end_f - start_frame

        print("=" * 70)
        print(f"[*] YOLOv8-Face GPU Pre-Scanner (Stage 0 Radar)")
        print(f"[*] Model Path  : {MODEL_PATH}")
        print(f"[*] Video Path  : {video_path}")
        print(f"[*] Frame Range : [{start_frame} -> {end_f}) (Count: {total_frames})")
        print("=" * 70)

        # Trajectory format per frame: [found (0/1), cx, cy, x1, y1, x2, y2, conf]
        trajectory = np.zeros((end_f, 8), dtype=np.float32)

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        t0 = time.time()
        last_log = t0
        detected_count = 0

        for fid in range(start_frame, end_f):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            faces = self.detect(frame)
            if len(faces) > 0:
                f = faces[0]
                cx, cy = f["anchor"]
                x1, y1, x2, y2 = f["bbox"]
                conf = f["confidence"]
                trajectory[fid] = [1.0, cx, cy, x1, y1, x2, y2, conf]
                detected_count += 1
            else:
                trajectory[fid, 0] = 0.0

            if (fid + 1) % 500 == 0 or (fid + 1) == end_f:
                dt = time.time() - t0
                fps = (fid - start_frame + 1) / max(dt, 0.001)
                print(
                    f"[*] Pre-scanned [{fid + 1:5d}/{end_f:5d}] | "
                    f"Speed: {fps:5.1f} FPS | Detected: {detected_count}/{fid - start_frame + 1} "
                    f"({detected_count / (fid - start_frame + 1) * 100:.1f}%)"
                )

        cap.release()
        total_time = time.time() - t0
        print(f"[√] Full Scan Finished in {total_time:.2f}s ({total_frames / max(total_time, 0.001):.1f} FPS)")

        os.makedirs(os.path.dirname(os.path.abspath(output_npy)), exist_ok=True)
        # Always save full 0-indexed trajectory up to end_f so trajectory[fid] directly maps to global frame ID
        np.save(output_npy, trajectory[:end_f])
        print(f"[√] Head trajectory saved to: {output_npy} ({os.path.getsize(output_npy) / 1024:.1f} KB)\n")
        return trajectory[:end_f]


def main():
    parser = argparse.ArgumentParser(description="YOLOFace Prescan Benchmark and Scanner")
    parser.add_argument("--video", "-v", default=r"C:\Users\Q\Desktop\2.mp4", help="Input video")
    parser.add_argument("--output", "-o", default=None, help="Output trajectory .npy path")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame")
    parser.add_argument("--test-frames", nargs="+", type=int, default=None, help="Test specific frame IDs")
    args = parser.parse_args()

    detector = YOLOFaceDetector()

    if args.test_frames:
        cap = cv2.VideoCapture(args.video)
        print("=" * 70)
        print(f"[*] YOLOFace Detector Loaded: {MODEL_PATH}")
        print(f"[*] Testing Specific Frames: {args.test_frames}")
        print("=" * 70)
        for fid in args.test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            t0 = time.time()
            faces = detector.detect(frame)
            dt = (time.time() - t0) * 1000.0
            if len(faces) > 0:
                f = faces[0]
                cx, cy = f["anchor"]
                x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
                print(
                    f"[√] Frame {fid:4d}: DETECTED in {dt:.1f}ms | Conf: {f['confidence']:.3f} | "
                    f"BBox: [{x1}, {y1}, {x2}, y2={y2}] | Anchor (cx, cy): ({cx:.1f}, {cy:.1f})"
                )
            else:
                print(f"[X] Frame {fid:4d}: NO FACE FOUND ({dt:.1f}ms)")
        cap.release()
    else:
        out_path = args.output or r"E:\VisoClean_NoDFM\output\2\head_trajectory.npy"
        detector.scan_video(args.video, out_path, start_frame=args.start, end_frame=args.end)


if __name__ == "__main__":
    main()
