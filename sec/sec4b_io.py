#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sec4b_io.py - I/O, Video Reading, Chunk Management, Mask Saving, Debug Overlay & Metadata
"""

import os
import sys
import re
import json
import shutil
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def _imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return cv2.imread(path, flags)


def _imwrite_unicode(path: str, img: np.ndarray, params=None) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    tmp_path = path + ".tmp" + ext
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(tmp_path)
    os.replace(tmp_path, path)
    return True


class VideoSource:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Input path does not exist: {self.path}")

        self.is_video_file = os.path.isfile(self.path) and self.path.lower().endswith(VIDEO_EXTENSIONS)
        self.is_image_dir = os.path.isdir(self.path)

        if not (self.is_video_file or self.is_image_dir):
            raise ValueError(f"Unsupported input path: {self.path}")

        self.frame_files = []
        self._cap = None
        self._current_pos = -1

        if self.is_image_dir:
            all_files = [f for f in os.listdir(self.path) if f.lower().endswith(IMAGE_EXTENSIONS)]
            all_files.sort(key=natural_sort_key)
            self.frame_files = [os.path.join(self.path, f) for f in all_files]
            if not self.frame_files:
                raise ValueError(f"No image files found in directory: {self.path}")
            first_frame = _imread_unicode(self.frame_files[0])
            if first_frame is None:
                raise RuntimeError(f"Failed to read first image: {self.frame_files[0]}")
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

    def get_total_frames(self) -> int:
        return self.total_frames

    def get_dimensions(self) -> tuple:
        return self.height, self.width

    def get_frame(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.total_frames - 1}]")

        if self.is_image_dir:
            img = _imread_unicode(self.frame_files[frame_idx])
            if img is None:
                raise RuntimeError(f"Failed to read frame file: {self.frame_files[frame_idx]}")
            return img
        else:
            if self._current_pos != frame_idx:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._current_pos = -1
                raise RuntimeError(f"Failed to read frame {frame_idx} from video {self.path}")
            self._current_pos = frame_idx + 1
            return frame

    def create_chunk_dir(self, frame_indices: list, chunk_dir: str) -> str:
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir, ignore_errors=True)
        os.makedirs(chunk_dir, exist_ok=True)

        if not self.is_image_dir and len(frame_indices) > 0:
            is_contiguous = (frame_indices[-1] - frame_indices[0] == len(frame_indices) - 1)
            if is_contiguous:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
                self._current_pos = frame_indices[0]
                for i, global_idx in enumerate(frame_indices):
                    target_path = os.path.join(chunk_dir, f"{i:06d}.jpg")
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        self._current_pos = global_idx + 1
                    else:
                        frame = self.get_frame(global_idx)
                    _imwrite_unicode(target_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return chunk_dir

        for i, global_idx in enumerate(frame_indices):
            target_path = os.path.join(chunk_dir, f"{i:06d}.jpg")
            if self.is_image_dir:
                src_path = self.frame_files[global_idx]
                try:
                    os.link(src_path, target_path)
                except (OSError, NotImplementedError):
                    img = _imread_unicode(src_path)
                    if img is not None:
                        _imwrite_unicode(target_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                img = self.get_frame(global_idx)
                _imwrite_unicode(target_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return chunk_dir

    def cleanup_chunk_dir(self, chunk_dir: str):
        if chunk_dir and os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir, ignore_errors=True)

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def save_binary_mask(mask: np.ndarray, out_path: str, orig_h: int = None, orig_w: int = None):
    if hasattr(mask, "cpu") and callable(mask.cpu):
        mask = mask.cpu().numpy()

    mask_u8 = np.where(mask > 0, np.uint8(255), np.uint8(0))
    if orig_h is not None and orig_w is not None:
        if mask_u8.shape[:2] != (orig_h, orig_w):
            mask_u8 = cv2.resize(mask_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    _imwrite_unicode(out_path, mask_u8)


def create_debug_overlay(
    raw_bgr: np.ndarray,
    mask_mask: np.ndarray,
    visible_face_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    alpha: float = 0.50,
    info_text: str = None,
) -> np.ndarray:
    h, w = raw_bgr.shape[:2]
    m_mask = (mask_mask > 0) if mask_mask.shape[:2] == (h, w) else (cv2.resize(mask_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
    m_face = (visible_face_mask > 0) if visible_face_mask.shape[:2] == (h, w) else (cv2.resize(visible_face_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
    m_occl = (occlusion_mask > 0) if occlusion_mask.shape[:2] == (h, w) else (cv2.resize(occlusion_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)

    combined = raw_bgr.astype(np.float32)
    if np.any(m_face):
        combined[m_face] = (1.0 - alpha) * combined[m_face] + alpha * np.array([0, 255, 0], dtype=np.float32)
    if np.any(m_mask):
        combined[m_mask] = (1.0 - alpha) * combined[m_mask] + alpha * np.array([0, 0, 255], dtype=np.float32)
    if np.any(m_occl):
        combined[m_occl] = (1.0 - alpha) * combined[m_occl] + alpha * np.array([255, 0, 0], dtype=np.float32)

    res_bgr = np.clip(combined, 0, 255).astype(np.uint8)
    if w >= 400 and h >= 300:
        cv2.rectangle(res_bgr, (10, 10), (min(w - 10, 520), 110), (20, 20, 20), -1)
        cv2.rectangle(res_bgr, (10, 10), (min(w - 10, 520), 110), (80, 80, 80), 1)
        cv2.circle(res_bgr, (25, 30), 7, (0, 0, 255), -1)
        cv2.putText(res_bgr, "Red: Mask", (38, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.circle(res_bgr, (175, 30), 7, (0, 255, 0), -1)
        cv2.putText(res_bgr, "Green: Visible Face", (188, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.circle(res_bgr, (365, 30), 7, (255, 0, 0), -1)
        cv2.putText(res_bgr, "Blue: Occlusion", (378, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        if info_text:
            y_pos = 60
            for line in info_text.strip().split("\n")[:2]:
                cv2.putText(res_bgr, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
                y_pos += 22
    return res_bgr


def save_debug_image(
    raw_bgr: np.ndarray,
    mask_mask: np.ndarray,
    visible_face_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    out_path: str,
    info_text: str = None,
):
    overlay = create_debug_overlay(raw_bgr, mask_mask, visible_face_mask, occlusion_mask, info_text=info_text)
    _imwrite_unicode(out_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])


class MetadataLogger:
    """O(1) append-mode logger with deferred deduplication to avoid O(N^2) disk rewrites on overlaps."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.jsonl_path = os.path.join(out_dir, "metadata.jsonl")
        os.makedirs(out_dir, exist_ok=True)
        self.records_by_frame = {}
        self._dirty_dupes = False
        self._writes_since_sync = 0
        self._load_existing()

    @property
    def existing_frame_ids(self) -> set:
        return set(self.records_by_frame.keys())

    def _load_existing(self):
        if os.path.exists(self.jsonl_path):
            try:
                with open(self.jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if "frame_id" in data:
                                    fid = int(data["frame_id"])
                                    if fid in self.records_by_frame:
                                        self._dirty_dupes = True
                                    self.records_by_frame[fid] = data
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                print(f"[!] Warning reading existing metadata.jsonl: {e}")

    def is_frame_recorded(self, frame_id: int) -> bool:
        return int(frame_id) in self.records_by_frame

    def write_record(self, record: dict):
        frame_id = int(record["frame_id"])
        if frame_id in self.records_by_frame:
            self._dirty_dupes = True
        self.records_by_frame[frame_id] = record
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        self._writes_since_sync += 1
        if self._dirty_dupes and self._writes_since_sync >= 500:
            self._sync_to_disk()

    def log_frame(self, frame_id: int, metrics: dict):
        rec = dict(metrics)
        rec["frame_id"] = int(frame_id)
        self.write_record(rec)

    def _sync_to_disk(self):
        sorted_ids = sorted(self.records_by_frame.keys())
        tmp_path = self.jsonl_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for fid in sorted_ids:
                f.write(json.dumps(self.records_by_frame[fid], ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.jsonl_path)
        self._dirty_dupes = False
        self._writes_since_sync = 0

    def close(self):
        if self._dirty_dupes:
            self._sync_to_disk()


def check_frame_completed(output_dir: str, frame_id: int, recorded_meta_ids: set) -> bool:
    fid_str = f"{frame_id:06d}"
    mask_file = os.path.join(output_dir, "mask", f"{fid_str}.png")
    face_file = os.path.join(output_dir, "visible_face", f"{fid_str}.png")
    occl_file = os.path.join(output_dir, "occlusion", f"{fid_str}.png")
    if not (os.path.exists(mask_file) and os.path.exists(face_file) and os.path.exists(occl_file)):
        return False
    return frame_id in recorded_meta_ids