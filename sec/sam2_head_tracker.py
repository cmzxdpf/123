#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sam2_head_tracker.py - Pure SAM 2.1 Video Mask Tracker with YOLOv8-Face Radar & Zero-Disk Memory Stream
========================================================================================================
1803 Standard Pure SAM 2.1 Video Mask Segmentation (P0/P1 Hardened):
1. Ingests Stage 0 `head_trajectory.npy` from GPU YOLOv8-Face radar.
2. Continuous Re-Anchor State Machine: automatically re-anchors upon dancer turnaround or text watermark.
3. 100% In-Memory Chunking: zero JPEG writes to disk (saves 9,300 disk writes).
4. Packed Bitmask Memory Stream: outputs `sec_masks_packed.npz` (np.packbits, saves 18,621 PNG disk writes).
5. Head Safety Envelope & 25,000px Physical Firewall: 100% prevents dress/torso inpainting.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

BASE_DIR = Path(os.path.abspath(__file__)).parent.parent

# SAM2 repo path
SAM2_REPO = os.environ.get("SAM2_REPO", r"D:\portable-files\sam2_repo")
if SAM2_REPO not in sys.path:
    sys.path.insert(0, SAM2_REPO)

from sam2.build_sam import build_sam2_video_predictor

# Local helpers
sys.path.insert(0, str(BASE_DIR / "sec"))
from sec4b_io import (
    VideoSource,
    save_binary_mask,
    save_debug_image,
    MetadataLogger,
)
from sec4b_postprocess import decompose_masks, compute_frame_metrics


def compute_mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    b1 = (m1 > 0)
    b2 = (m2 > 0)
    inter = np.sum(b1 & b2)
    union = np.sum(b1 | b2)
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def get_yolo_envelope(
    traj_row: np.ndarray,
    h: int,
    w: int,
) -> Optional[np.ndarray]:
    """Build Head Safety Envelope around face from YOLO detection."""
    if traj_row[0] <= 0.5:
        return None
    x1, y1, x2, y2 = float(traj_row[3]), float(traj_row[4]), float(traj_row[5]), float(traj_row[6])
    fw = max(10.0, x2 - x1)
    fh = max(10.0, y2 - y1)

    env_x1 = max(0, int(x1 - 0.40 * fw))
    env_x2 = min(w, int(x2 + 0.40 * fw))
    env_y1 = max(0, int(y1 + 0.10 * fh))
    env_y2 = min(h, int(y2 + 0.20 * fh))

    envelope = np.zeros((h, w), dtype=bool)
    envelope[env_y1:env_y2, env_x1:env_x2] = True
    return envelope


def get_traj_row(traj: np.ndarray, fid: int, start_offset: int = 0) -> np.ndarray:
    """Safe trajectory lookup that handles both global-indexed and sliced trajectories."""
    if traj is None or len(traj) == 0:
        return np.zeros(8, dtype=np.float32)
    if 0 <= fid < len(traj):
        return traj[fid]
    local_f = fid - start_offset
    if 0 <= local_f < len(traj):
        return traj[local_f]
    return np.zeros(8, dtype=np.float32)


def make_head_prompt(
    traj_row: np.ndarray,
    h: int,
    w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds constrained box and positive/negative points for SAM 2."""
    x1, y1, x2, y2 = float(traj_row[3]), float(traj_row[4]), float(traj_row[5]), float(traj_row[6])
    fw = max(10.0, x2 - x1)
    fh = max(10.0, y2 - y1)
    cx, cy = float(traj_row[1]), float(traj_row[2])

    box = np.array([
        max(0.0, x1 - 0.20 * fw),
        max(0.0, y1 + 0.30 * fh),
        min(float(w), x2 + 0.20 * fw),
        min(float(h), y2 + 0.12 * fh),
    ], dtype=np.float32)

    pts = np.array([
        [cx, cy],
        [cx, max(0.0, y1 + 0.28 * fh)],
    ], dtype=np.float32)
    lbls = np.array([1, 0], dtype=np.int32)
    return box, pts, lbls


def build_in_memory_state(
    predictor,
    frames_bgr: List[np.ndarray],
    video_h: int,
    video_w: int,
    device: str = "cuda",
) -> dict:
    """Creates SAM 2.1 inference state directly in RAM without writing JPEGs to disk."""
    image_size = predictor.image_size
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

    imgs = []
    for f in frames_bgr:
        r = cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb.transpose(2, 0, 1))
        t = (t - img_mean) / img_std
        imgs.append(t)
    imgs_tensor = torch.stack(imgs, dim=0)

    state = {
        "images": imgs_tensor,
        "num_frames": len(frames_bgr),
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": True,
        "video_height": video_h,
        "video_width": video_w,
        "device": torch.device(device),
        "storage_device": torch.device("cpu"),
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
    return state


def track_and_process_segment(
    predictor,
    video_source: VideoSource,
    segment_indices: List[int],
    output_dir: str,
    meta_logger: MetadataLogger,
    trajectory: np.ndarray,
    packed_mask: np.ndarray,
    packed_vis: np.ndarray,
    packed_occ: np.ndarray,
    start_offset: int,
    pos_points: list = None,
    prev_carry_mask: np.ndarray = None,
    segment_id: int = 0,
    include_straps: bool = True,
    save_debug: bool = False,
    debug_every: int = 1,
    overlap_with_prev: int = 0,
    overlap_with_next: int = 0,
    next_start_frame_id: int = None,
    stored_overlap_results: dict = None,
    save_png: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Processes one segment purely in RAM using YOLO trajectory re-anchor state machine."""
    h, w = video_source.get_dimensions()
    seg_len = len(segment_indices)

    # 1. Fetch frames into RAM
    frames_bgr = [video_source.get_frame(fid) for fid in segment_indices]

    seg_last_mask = None
    next_anchor_mask = None
    current_overlap_results = {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        inference_state = build_in_memory_state(predictor, frames_bgr, h, w, device="cuda")
        predictor.reset_state(inference_state)

        ann_obj_id = 1
        valid_carry = (
            prev_carry_mask is not None
            and 0 < np.sum(prev_carry_mask > 0) <= 25000
        )

        first_global_fid = segment_indices[0]
        traj_first = get_traj_row(trajectory, first_global_fid, start_offset)
        first_head_env = get_yolo_envelope(traj_first, h, w)

        prompted_frames = set()

        if valid_carry:
            # Carry over mask from previous segment with YOLO envelope bounds
            carry_m = prev_carry_mask.astype(np.uint8)
            if first_head_env is not None:
                carry_m = (carry_m > 0) & first_head_env
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=ann_obj_id,
                mask=carry_m,
            )
            prompted_frames.add(0)
        else:
            # Find the first frame in this segment with valid YOLO head detection
            first_valid_local_idx = None
            for local_idx, g_fid in enumerate(segment_indices):
                if get_traj_row(trajectory, g_fid, start_offset)[0] > 0.5:
                    first_valid_local_idx = local_idx
                    break

            if first_valid_local_idx is not None:
                g_fid = segment_indices[first_valid_local_idx]
                box, pts, lbls = make_head_prompt(get_traj_row(trajectory, g_fid, start_offset), h, w)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=first_valid_local_idx,
                    obj_id=ann_obj_id,
                    points=pts,
                    labels=lbls,
                    box=box,
                )
                prompted_frames.add(first_valid_local_idx)
            else:
                # Entire segment has 0 face detections (dancer turned back)
                zero_m = np.zeros((h, w), dtype=np.uint8)
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=ann_obj_id,
                    mask=zero_m,
                )
                prompted_frames.add(0)

        # 2. Check for turnaround recovery points within the segment (0 -> 1 transition)
        for local_idx in range(1, seg_len):
            g_fid = segment_indices[local_idx]
            prev_g_fid = segment_indices[local_idx - 1]
            traj_curr = get_traj_row(trajectory, g_fid, start_offset)
            traj_prev = get_traj_row(trajectory, prev_g_fid, start_offset)
            # Dancer just turned forward
            if traj_curr[0] > 0.5 and traj_prev[0] <= 0.5:
                if local_idx not in prompted_frames:
                    box, pts, lbls = make_head_prompt(traj_curr, h, w)
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=local_idx,
                        obj_id=ann_obj_id,
                        points=pts,
                        labels=lbls,
                        box=box,
                    )
                    prompted_frames.add(local_idx)

        # 3. Propagation across in-memory frames
        yielded_frames = set()
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            yielded_frames.add(out_frame_idx)
            global_frame_id = segment_indices[out_frame_idx]
            tracked_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

            # Dynamic YOLO Head Safety Envelope Guard
            curr_traj = get_traj_row(trajectory, global_frame_id, start_offset)
            curr_env = get_yolo_envelope(curr_traj, h, w)
            if curr_env is not None:
                tracked_mask = tracked_mask & curr_env
            elif curr_traj[0] <= 0.5:
                # Back turned -> enforce 0 mask
                tracked_mask = np.zeros_like(tracked_mask)

            # P0 Hard Physical Firewall: Face mask CANNOT exceed 25,000 pixels
            mask_px_count = int(np.sum(tracked_mask > 0))
            if mask_px_count > 25000:
                if seg_last_mask is not None and np.sum(seg_last_mask > 0) > 0:
                    tracked_mask = seg_last_mask.copy()
                else:
                    tracked_mask = np.zeros_like(tracked_mask)

            seg_last_mask = tracked_mask

            if next_start_frame_id is not None and global_frame_id == next_start_frame_id:
                next_anchor_mask = tracked_mask.copy()

            # Overlap conflict resolution
            is_in_prev_overlap = (overlap_with_prev > 0 and out_frame_idx < overlap_with_prev)
            overlap_conflict = False
            if is_in_prev_overlap and stored_overlap_results and global_frame_id in stored_overlap_results:
                prev_res = stored_overlap_results[global_frame_id]
                prev_m = prev_res["mask_mask"]
                iou = compute_mask_iou(tracked_mask, prev_m)
                if iou < 0.55:
                    overlap_conflict = True
                    alpha = float(out_frame_idx) / float(max(1, overlap_with_prev))
                    if alpha < 0.5:
                        tracked_mask = prev_m

            is_in_next_overlap = (overlap_with_next > 0 and out_frame_idx >= (seg_len - overlap_with_next))

            raw_frame = frames_bgr[out_frame_idx]
            mask_mask, visible_face_mask, occlusion_mask = decompose_masks(
                raw_frame, tracked_mask, include_straps=include_straps
            )

            if is_in_next_overlap:
                current_overlap_results[global_frame_id] = {
                    "mask_mask": mask_mask.copy(),
                    "visible_face_mask": visible_face_mask.copy(),
                    "occlusion_mask": occlusion_mask.copy(),
                }

            # In-Memory Bitmask Storage (0 PNG disk writes!)
            arr_idx = global_frame_id - start_offset
            if 0 <= arr_idx < len(packed_mask):
                packed_mask[arr_idx] = np.packbits(mask_mask > 0)
                packed_vis[arr_idx] = np.packbits(visible_face_mask > 0)
                packed_occ[arr_idx] = np.packbits(occlusion_mask > 0)

            # Optional PNG saving (default: False for 0 disk I/O)
            if save_png:
                save_binary_mask(mask_mask, os.path.join(output_dir, "mask", f"{global_frame_id:06d}.png"))
                save_binary_mask(visible_face_mask, os.path.join(output_dir, "visible_face", f"{global_frame_id:06d}.png"))
                save_binary_mask(occlusion_mask, os.path.join(output_dir, "occlusion", f"{global_frame_id:06d}.png"))

            # Metrics logging
            metrics = compute_frame_metrics(global_frame_id, mask_mask, visible_face_mask, occlusion_mask)
            metrics["segment_id"] = segment_id
            metrics["overlap_conflict"] = overlap_conflict
            meta_logger.log_frame(global_frame_id, metrics)

            if save_debug and (global_frame_id % debug_every == 0):
                debug_path = os.path.join(output_dir, "debug", f"debug_{global_frame_id:06d}.jpg")
                face_cnt = metrics.get("visible_face_area", metrics.get("face_area", 0))
                info_text = (
                    f"Frame {global_frame_id:06d} (Seg {segment_id})\n"
                    f"Mask: {metrics['mask_area']} px | Face: {face_cnt} px | Occ: {metrics['occlusion_area']} px"
                )
                save_debug_image(raw_frame, mask_mask, visible_face_mask, occlusion_mask, debug_path, info_text=info_text)
        # Ensure any frames skipped prior to first detection are explicitly logged
        for local_idx in range(seg_len):
            if local_idx not in yielded_frames:
                g_fid = segment_indices[local_idx]
                zero_m = np.zeros((h, w), dtype=np.uint8)
                if save_png:
                    save_binary_mask(zero_m, os.path.join(output_dir, "mask", f"{g_fid:06d}.png"))
                    save_binary_mask(zero_m, os.path.join(output_dir, "visible_face", f"{g_fid:06d}.png"))
                    save_binary_mask(zero_m, os.path.join(output_dir, "occlusion", f"{g_fid:06d}.png"))
                metrics = compute_frame_metrics(g_fid, zero_m, zero_m, zero_m)
                metrics["segment_id"] = segment_id
                metrics["overlap_conflict"] = False
                meta_logger.log_frame(g_fid, metrics)

        del inference_state

    anchor_for_next = next_anchor_mask if next_anchor_mask is not None else seg_last_mask
    return anchor_for_next, current_overlap_results


def run_sam2_pipeline(
    video_path: str,
    output_dir: str,
    trajectory_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    segment_length: int = 150,
    overlap: int = 50,
    pos_points: list = None,
    include_straps: bool = True,
    save_debug: bool = False,
    debug_every: int = 1,
    save_png: bool = False,
):
    t0 = time.time()
    print("=" * 70)
    print("[*] SAM 2.1 Video Segmentation & Tracking (YOLO-Guided Zero-Disk Engine)")
    print(f"[*] Input Video     : {video_path}")
    print(f"[*] Output Directory : {output_dir}")
    print(f"[*] Trajectory Path : {trajectory_path}")
    print("=" * 70)

    # 1. Load YOLO trajectory
    if not os.path.isfile(trajectory_path):
        raise FileNotFoundError(f"Head trajectory not found at {trajectory_path}! Run yolo_head_prescan first.")
    trajectory = np.load(trajectory_path)

    # 2. Initialize Predictor
    ckpt_path = os.environ.get("SAM2_CHECKPOINT", r"D:\3\mask_skin_video\checkpoints\sam2.1_hiera_tiny.pt")
    cfg_name = "configs/sam2.1/sam2.1_hiera_t.yaml"
    print(f"[*] Loading SAM 2.1 Tiny: {ckpt_path}...")
    predictor = build_sam2_video_predictor(cfg_name, ckpt_path, device="cuda")
    print(f"[√] SAM 2.1 loaded. VRAM allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    os.makedirs(output_dir, exist_ok=True)
    if save_png:
        for sub in ["mask", "visible_face", "occlusion"]:
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    if save_debug:
        os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    video_source = VideoSource(video_path)
    total_frames = video_source.get_total_frames()
    start_f = max(0, start_frame)
    end_f = total_frames if (end_frame is None or end_frame < 0) else min(end_frame, total_frames)
    frame_count = end_f - start_f

    print(f"[*] Total video frames: {total_frames} | Processing range: [{start_f} -> {end_f}) (Count: {frame_count})")

    # 3. Pre-allocate in-memory packed bitmask arrays (6.3 GB RAM budget)
    h, w = video_source.get_dimensions()
    packed_len = (h * w + 7) // 8  # 259,200 bytes for 1080x1920
    print(f"[*] Pre-allocating In-Memory Bitmask Arrays: {frame_count} frames x {packed_len} bytes (~{frame_count * packed_len * 3 / 1024**2:.1f} MB RAM)...")
    packed_mask = np.zeros((frame_count, packed_len), dtype=np.uint8)
    packed_vis = np.zeros((frame_count, packed_len), dtype=np.uint8)
    packed_occ = np.zeros((frame_count, packed_len), dtype=np.uint8)

    meta_logger = MetadataLogger(output_dir)

    # 4. Segment planning
    stride = max(1, segment_length - overlap)
    segments = []
    curr = start_f
    while curr < end_f:
        seg_end = min(curr + segment_length, end_f)
        segments.append(list(range(curr, seg_end)))
        if seg_end >= end_f:
            break
        curr += stride

    print(f"[*] Segment Planner: {len(segments)} segments (length={segment_length}, overlap={overlap}, stride={stride})")

    prev_carry_mask = None
    stored_overlap = {}

    for seg_id, seg_indices in enumerate(segments):
        t_seg_start = time.time()
        ov_prev = overlap if seg_id > 0 else 0
        ov_next = overlap if (seg_id < len(segments) - 1) else 0
        next_start = segments[seg_id + 1][0] if (seg_id < len(segments) - 1) else None

        anchor_mask, next_overlap_res = track_and_process_segment(
            predictor=predictor,
            video_source=video_source,
            segment_indices=seg_indices,
            output_dir=output_dir,
            meta_logger=meta_logger,
            trajectory=trajectory,
            packed_mask=packed_mask,
            packed_vis=packed_vis,
            packed_occ=packed_occ,
            start_offset=start_f,
            pos_points=pos_points,
            prev_carry_mask=prev_carry_mask,
            segment_id=seg_id,
            include_straps=include_straps,
            save_debug=save_debug,
            debug_every=debug_every,
            overlap_with_prev=ov_prev,
            overlap_with_next=ov_next,
            next_start_frame_id=next_start,
            stored_overlap_results=stored_overlap,
            save_png=save_png,
        )

        prev_carry_mask = anchor_mask
        stored_overlap = next_overlap_res
        dt_seg = time.time() - t_seg_start
        fps_seg = len(seg_indices) / max(dt_seg, 0.001)
        if (seg_id + 1) % 5 == 0 or seg_id == 0 or (seg_id + 1) == len(segments):
            print(f"[*] Segment [{seg_id + 1:2d}/{len(segments):2d}] (Frames {seg_indices[0]:05d}..{seg_indices[-1]:05d}) | Speed: {fps_seg:4.1f} FPS | Elapsed: {dt_seg:.1f}s")

    total_time = time.time() - t0
    meta_logger.close()

    # 5. Save single packed bitmask npz (Instant 1-file save!)
    packed_npz_path = os.path.join(output_dir, "sec_masks_packed.npz")
    print(f"\n[*] Saving packed bitmask to: {packed_npz_path}...")
    t_save = time.time()
    np.savez_compressed(
        packed_npz_path,
        mask=packed_mask,
        visible_face=packed_vis,
        occlusion=packed_occ,
        start_frame=start_f,
        end_frame=end_f,
        shape=np.array([h, w]),
    )
    dt_save = time.time() - t_save
    file_size_mb = os.path.getsize(packed_npz_path) / 1024**2
    print(f"[√] Packed bitmask saved in {dt_save:.2f}s (Size: {file_size_mb:.1f} MB, ZERO PNGs written)")

    print("\n" + "=" * 70)
    print(f"[√] ALL {end_f - start_f} FRAMES TRACKED IN {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"[√] Average FPS: {(end_f - start_f) / max(total_time, 0.001):.1f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="SAM 2.1 YOLO-Guided Zero-Disk Video Tracker")
    parser.add_argument("--video", "-v", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--trajectory", "-t", default=None, help="YOLO head trajectory npy path")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end-frame", type=int, default=-1, help="End frame index (-1 = full)")
    parser.add_argument("--segment-length", type=int, default=150, help="Segment length (default: 150)")
    parser.add_argument("--overlap", type=int, default=20, help="Overlap length (default: 20)")
    parser.add_argument("--pos", default="", help="Positive prompt point x,y")
    parser.add_argument("--include-straps", default="true", help="Include straps")
    parser.add_argument("--save-debug", default="false", help="Save debug images")
    parser.add_argument("--save-png", action="store_true", help="Also save intermediate PNGs (legacy mode)")

    args = parser.parse_args()

    pos_pts = []
    if args.pos:
        nums = [float(t) for t in args.pos.replace(",", " ").split() if t]
        if len(nums) >= 2:
            pos_pts.append([nums[0], nums[1]])

    include_straps = str(args.include_straps).lower() in ("true", "1", "yes")
    save_debug = str(args.save_debug).lower() in ("true", "1", "yes")

    traj_path = args.trajectory or os.path.join(os.path.dirname(os.path.abspath(args.output)), "head_trajectory.npy")
    if not os.path.isfile(traj_path):
        traj_path = os.path.join(args.output, "head_trajectory.npy")

    run_sam2_pipeline(
        video_path=args.video,
        output_dir=args.output,
        trajectory_path=traj_path,
        start_frame=args.start_frame,
        end_frame=None if args.end_frame < 0 else args.end_frame,
        segment_length=args.segment_length,
        overlap=args.overlap,
        pos_points=pos_pts,
        include_straps=include_straps,
        save_debug=save_debug,
        save_png=args.save_png,
    )


if __name__ == "__main__":
    main()
