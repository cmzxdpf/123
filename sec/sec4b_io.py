#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sec4b_io.py - I/O, Video Reading, Chunk Management, Mask Saving, Debug Overlay & Metadata
=========================================================================================
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import re
import json
import shutil
import cv2
import numpy as np

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')


def natural_sort_key(s):
    """Natural numerical sort key: frame_1, frame_2, ..., frame_10."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


class VideoSource:
    """
    Unified Video and Image Sequence Reader.
    Supports reading from a video file or an image sequence directory.
    Provides fast random or sequential access and temporary chunk projection.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Input path does not exist: {self.path}")

        self.is_video_file = os.path.isfile(self.path) and self.path.lower().endswith(VIDEO_EXTENSIONS)
        self.is_image_dir = os.path.isdir(self.path)

        if not (self.is_video_file or self.is_image_dir):
            raise ValueError(f"Path is neither a supported video file nor an image directory: {self.path}")

        self.frame_files = []
        self._cap = None

        if self.is_image_dir:
            all_files = [f for f in os.listdir(self.path) if f.lower().endswith(IMAGE_EXTENSIONS)]
            all_files.sort(key=natural_sort_key)
            self.frame_files = [os.path.join(self.path, f) for f in all_files]
            if not self.frame_files:
                raise ValueError(f"No image files found in directory: {self.path}")
            first_frame = cv2.imread(self.frame_files[0])
            if first_frame is None:
                raise RuntimeError(f"Failed to read first image: {self.frame_files[0]}")
            self.height, self.width = first_frame.shape[:2]
            self.total_frames = len(self.frame_files)
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
        """Fetch frame at frame_idx (0-indexed). Returns BGR uint8 image."""
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.total_frames - 1}]")

        if self.is_image_dir:
            img = cv2.imread(self.frame_files[frame_idx])
            if img is None:
                raise RuntimeError(f"Failed to read frame file: {self.frame_files[frame_idx]}")
            return img
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Failed to read frame {frame_idx} from video {self.path}")
            return frame

    def create_chunk_dir(self, frame_indices: list, chunk_dir: str) -> str:
        """
        Populate chunk_dir with 6-digit image files (000000.jpg, 000001.jpg, ...)
        corresponding to frame_indices.
        Uses NTFS hardlinks when possible (zero disk copy), or extracts frames directly.
        """
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir, ignore_errors=True)
        os.makedirs(chunk_dir, exist_ok=True)

        if not self.is_image_dir and len(frame_indices) > 0:
            is_contiguous = (frame_indices[-1] - frame_indices[0] == len(frame_indices) - 1)
            if is_contiguous:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
                for i, global_idx in enumerate(frame_indices):
                    target_name = f"{i:06d}.jpg"
                    target_path = os.path.join(chunk_dir, target_name)
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        frame = self.get_frame(global_idx)
                    cv2.imwrite(target_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return chunk_dir

        for i, global_idx in enumerate(frame_indices):
            target_name = f"{i:06d}.jpg"
            target_path = os.path.join(chunk_dir, target_name)

            if self.is_image_dir:
                src_path = self.frame_files[global_idx]
                try:
                    # Attempt zero-copy NTFS hardlink
                    os.link(src_path, target_path)
                except (OSError, NotImplementedError):
                    # Fallback to reading and writing
                    img = cv2.imread(src_path)
                    if img is not None:
                        cv2.imwrite(target_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                img = self.get_frame(global_idx)
                cv2.imwrite(target_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return chunk_dir

    def cleanup_chunk_dir(self, chunk_dir: str):
        if chunk_dir and os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir, ignore_errors=True)

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def save_binary_mask(mask: np.ndarray, out_path: str, orig_h: int = None, orig_w: int = None):
    """
    Save uint8 binary mask (values 0 or 255) at original video resolution.
    Creates parent directories if necessary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if hasattr(mask, "cpu") and callable(mask.cpu):
        mask = mask.cpu().numpy()

    if mask.dtype != np.uint8:
        if mask.dtype == bool:
            mask_u8 = (mask * 255).astype(np.uint8)
        else:
            mask_u8 = ((mask > 0) * 255).astype(np.uint8)
    else:
        # Normalize: any non-zero value becomes 255
        mask_u8 = np.where(mask > 0, np.uint8(255), np.uint8(0))

    if orig_h is not None and orig_w is not None:
        if mask_u8.shape[:2] != (orig_h, orig_w):
            mask_u8 = cv2.resize(mask_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(out_path, mask_u8)


def create_debug_overlay(
    raw_bgr: np.ndarray,
    mask_mask: np.ndarray,
    visible_face_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    alpha: float = 0.50,
    info_text: str = None,
) -> np.ndarray:
    """
    Generate standard RGB Debug Overlay:
    - Red (BGR: [0, 0, 255])   = mask_mask
    - Green (BGR: [0, 255, 0]) = visible_face_mask
    - Blue (BGR: [255, 0, 0])  = occlusion_mask
    Blended with raw_bgr at specified alpha.
    """
    h, w = raw_bgr.shape[:2]
    overlay = raw_bgr.copy()

    # Ensure masks match dimensions
    m_mask = (mask_mask > 0) if mask_mask.shape[:2] == (h, w) else (cv2.resize(mask_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
    m_face = (visible_face_mask > 0) if visible_face_mask.shape[:2] == (h, w) else (cv2.resize(visible_face_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
    m_occl = (occlusion_mask > 0) if occlusion_mask.shape[:2] == (h, w) else (cv2.resize(occlusion_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)

    # Color definitions (BGR)
    color_mask = np.array([0, 0, 255], dtype=np.float32)      # Red
    color_face = np.array([0, 255, 0], dtype=np.float32)      # Green
    color_occl = np.array([255, 0, 0], dtype=np.float32)      # Blue

    # Priority: occlusion > mask > visible_face where overlaps might occur in visualization
    combined = overlay.astype(np.float32)

    if np.any(m_face):
        combined[m_face] = (1.0 - alpha) * combined[m_face] + alpha * color_face
    if np.any(m_mask):
        combined[m_mask] = (1.0 - alpha) * combined[m_mask] + alpha * color_mask
    if np.any(m_occl):
        combined[m_occl] = (1.0 - alpha) * combined[m_occl] + alpha * color_occl

    res_bgr = np.clip(combined, 0, 255).astype(np.uint8)

    # Draw Legend and Info Header only on sufficiently large frames
    if w >= 400 and h >= 300:
        cv2.rectangle(res_bgr, (10, 10), (min(w - 10, 520), 110), (20, 20, 20), -1)
        cv2.rectangle(res_bgr, (10, 10), (min(w - 10, 520), 110), (80, 80, 80), 1)

        # Legend color badges
        cv2.circle(res_bgr, (25, 30), 7, (0, 0, 255), -1)
        cv2.putText(res_bgr, "Red: Mask", (38, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.circle(res_bgr, (175, 30), 7, (0, 255, 0), -1)
        cv2.putText(res_bgr, "Green: Visible Face", (188, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.circle(res_bgr, (365, 30), 7, (255, 0, 0), -1)
        cv2.putText(res_bgr, "Blue: Occlusion", (378, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        if info_text:
            lines = info_text.strip().split("\n")
            y_pos = 60
            for line in lines[:2]:
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
    """Generate and write debug overlay JPG image."""
    overlay = create_debug_overlay(raw_bgr, mask_mask, visible_face_mask, occlusion_mask, info_text=info_text)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])


class MetadataLogger:
    """
    Thread-safe / process-safe line-by-line metadata logger for metadata.jsonl.
    Maintains unique frame records, supports in-place updates for overlap frames,
    and prevents duplicate entries during resumes.
    """

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.jsonl_path = os.path.join(out_dir, "metadata.jsonl")
        os.makedirs(out_dir, exist_ok=True)
        self.records_by_frame = {}
        self._load_existing()

    @property
    def existing_frame_ids(self) -> set:
        return set(self.records_by_frame.keys())

    def _load_existing(self):
        """Load already committed frame IDs from metadata.jsonl for resume."""
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
                                    self.records_by_frame[fid] = data
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                print(f"[!] Warning reading existing metadata.jsonl: {e}")

    def is_frame_recorded(self, frame_id: int) -> bool:
        return int(frame_id) in self.records_by_frame

    def write_record(self, record: dict):
        """
        Record a frame's metadata. If frame_id is already present (e.g. overlap update),
        updates the record and synchronizes file with unique sorted entries.
        Otherwise appends directly.
        """
        frame_id = int(record["frame_id"])
        already_exists = frame_id in self.records_by_frame
        self.records_by_frame[frame_id] = record

        if already_exists:
            # Overlap update: sync the full file with unique sorted records
            self._sync_to_disk()
        else:
            # New frame: append directly
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    def _sync_to_disk(self):
        """Rewrite metadata.jsonl with strictly sorted, unique frame_ids."""
        sorted_ids = sorted(self.records_by_frame.keys())
        tmp_path = self.jsonl_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for fid in sorted_ids:
                f.write(json.dumps(self.records_by_frame[fid], ensure_ascii=False) + "\n")
        # Atomic replace
        os.replace(tmp_path, self.jsonl_path)


def check_frame_completed(output_dir: str, frame_id: int, recorded_meta_ids: set) -> bool:
    """
    Check if a frame is fully completed (all 3 masks present on disk + metadata entry exists).
    """
    fid_str = f"{frame_id:06d}"
    mask_file = os.path.join(output_dir, "mask", f"{fid_str}.png")
    face_file = os.path.join(output_dir, "visible_face", f"{fid_str}.png")
    occl_file = os.path.join(output_dir, "occlusion", f"{fid_str}.png")

    if not (os.path.exists(mask_file) and os.path.exists(face_file) and os.path.exists(occl_file)):
        return False

    return frame_id in recorded_meta_ids
