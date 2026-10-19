#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sec4b_tracker.py - Safe-Chunk Video Tracking, Mask Carry-Over & Overlap Blending
================================================================================
Handles SeC-4B / SAM2 continuous memory state, chunk segmentation,
overlap conflict logging, and per-frame immediate disk writes.
"""

import os
import sys
import time
import cv2
import numpy as np
import torch

from sec4b_model import get_memory_stats, cleanup_memory
from sec4b_io import (
    VideoSource,
    save_binary_mask,
    save_debug_image,
    MetadataLogger,
    check_frame_completed,
)
from sec4b_postprocess import (
    decompose_masks,
    compute_frame_metrics,
)


# Junction-safe base directory definitions
SEC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SEC_DIR)
REAL_BASE_DIR = os.path.dirname(os.path.realpath(SEC_DIR))


def find_tddfa_root() -> str | None:
    """Discovers 3DDFA_V2 across junction-safe roots."""
    candidates = [
        os.environ.get("TDDFA_ROOT"),
        os.path.join(BASE_DIR, "3DDFA_V2"),
        os.path.join(REAL_BASE_DIR, "3DDFA_V2"),
        r"E:\VisoClean_NoDFM\3DDFA_V2",
        r"J:\2\3DDFA_V2",
    ]
    for c in candidates:
        if c and os.path.isdir(c) and os.path.isdir(os.path.join(c, "FaceBoxes")):
            return os.path.abspath(c)
    return None


def compute_mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    b1 = (m1 > 0)
    b2 = (m2 > 0)
    inter = np.sum(b1 & b2)
    union = np.sum(b1 | b2)
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def track_and_process_segment(
    model,
    tokenizer,
    video_source: VideoSource,
    segment_indices: list,
    output_dir: str,
    meta_logger: MetadataLogger,
    pos_points: list = None,
    neg_points: list = None,
    prompt_box: list = None,
    prev_carry_mask: np.ndarray = None,
    segment_id: int = 0,
    include_straps: bool = False,
    save_debug: bool = True,
    debug_every: int = 1,
    resume: bool = False,
    overlap_with_prev: int = 0,
    overlap_with_next: int = 0,
    next_start_frame_id: int = None,
    stored_overlap_results: dict = None,
    chunk_base_dir: str = None,
) -> tuple:
    """
    Process a single segment of frames:
    1. Log memory before segment.
    2. Build isolated temp chunk.
    3. Init state in GPU (zero CPU offload).
    4. Inject carry-over mask for segment start frame, or initial prompt points/box.
    5. Propagate and immediately write masks to disk.
    6. Compare overlap with previous segment (IoU, conflict detection, quality blending).
    7. Save overlap frames and anchor mask for the next segment.
    8. Clean up state and log memory after cleanup.
    """
    total_in_seg = len(segment_indices)
    if total_in_seg == 0:
        return None, {}

    mem_before = get_memory_stats()
    print("=" * 70)
    print(f"[*] Starting Segment {segment_id} (Frames {segment_indices[0]} .. {segment_indices[-1]} | Total: {total_in_seg} frames)")
    print(f"[*] Memory Before Segment: RAM: {mem_before['ram_used_gb']}/{mem_before['ram_total_gb']} GB | VRAM: {mem_before['vram_alloc_gb']} GB")

    chunk_dir = os.path.join(chunk_base_dir or os.path.dirname(output_dir), f"_chunk_seg_{segment_id:04d}")
    video_source.create_chunk_dir(segment_indices, chunk_dir)

    seg_last_mask = None
    next_anchor_mask = None
    new_overlap_results = {}

    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            inference_state = model.grounding_encoder.init_state(
                video_path=chunk_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )
            model.grounding_encoder.reset_state(inference_state)

            mem_after_init = get_memory_stats()
            print(f"[*] Memory After State Init: RAM: {mem_after_init['ram_used_gb']} GB | VRAM: {mem_after_init['vram_alloc_gb']} GB")

            # Conditioning on first frame of this segment (frame_idx=0 in chunk)
            ann_frame_idx = 0
            ann_obj_id = 1

            # Check if previous carry mask is valid and within physical face bounds (<= 25000px)
            valid_carry = (
                prev_carry_mask is not None
                and 0 < np.sum(prev_carry_mask > 0) <= 25000
            )

            if valid_carry:
                print(f"[*] Segment {segment_id}: Injecting carry-over mask for frame {segment_indices[0]:06d} (pixels: {int(np.sum(prev_carry_mask > 0))})...")
                _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    mask=prev_carry_mask,
                )
            else:
                ann_frame_id = segment_indices[0]
                face_found = False
                cx, cy = 0.0, 0.0

                # Segment 0 initial prompt points/box if specified
                if segment_id == 0 and (pos_points or neg_points):
                    pts_list, lbls_list = [], []
                    for pt in (pos_points or []):
                        pts_list.append([float(pt[0]), float(pt[1])])
                        lbls_list.append(1)
                    for pt in (neg_points or []):
                        pts_list.append([float(pt[0]), float(pt[1])])
                        lbls_list.append(0)
                    points_arr = np.array(pts_list, dtype=np.float32)
                    labels_arr = np.array(lbls_list, dtype=np.int32)
                    print(f"[*] Segment {segment_id}: Injecting {len(pts_list)} prompt points...")
                    _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        points=points_arr,
                        labels=labels_arr,
                    )
                elif segment_id == 0 and prompt_box:
                    box_arr = np.array(prompt_box, dtype=np.float32)
                    print(f"[*] Segment {segment_id}: Injecting prompt box {prompt_box}...")
                    _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        box=box_arr,
                    )
                else:
                    # Smart auto anchor: 1. YOLO trajectory; 2. FaceBoxes; 3. Segment scan for face appearance
                    traj_candidates = [
                        os.path.join(os.path.dirname(os.path.abspath(output_dir)), "head_trajectory.npy"),
                        os.path.join(output_dir, "head_trajectory.npy"),
                        os.path.join(BASE_DIR, "output", "2", "head_trajectory.npy"),
                    ]
                    trajectory_data = None
                    for tc in traj_candidates:
                        if os.path.isfile(tc):
                            try:
                                trajectory_data = np.load(tc)
                                break
                            except Exception:
                                pass

                    if trajectory_data is not None:
                        if ann_frame_id < len(trajectory_data) and trajectory_data[ann_frame_id, 0] > 0.5:
                            cx = float(trajectory_data[ann_frame_id, 1])
                            cy = float(trajectory_data[ann_frame_id, 2])
                            face_found = True
                        else:
                            # Scan segment for first face appearance
                            for cand_idx, g_fid in enumerate(segment_indices):
                                if g_fid < len(trajectory_data) and trajectory_data[g_fid, 0] > 0.5:
                                    cx = float(trajectory_data[g_fid, 1])
                                    cy = float(trajectory_data[g_fid, 2])
                                    ann_frame_id = g_fid
                                    ann_frame_idx = cand_idx
                                    face_found = True
                                    break

                    if not face_found:
                        try:
                            ann_frame_img = video_source.get_frame(ann_frame_id)
                            if ann_frame_img is not None:
                                tddfa_dir = find_tddfa_root()
                                if tddfa_dir and os.path.isdir(tddfa_dir) and tddfa_dir not in sys.path:
                                    sys.path.insert(0, tddfa_dir)
                                try:
                                    from FaceBoxes import FaceBoxes
                                    fb = FaceBoxes()
                                    fb_boxes = fb(ann_frame_img)
                                    if len(fb_boxes) > 0:
                                        b = max(fb_boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
                                        if len(b) >= 5 and b[4] >= 0.45:
                                            cx = float((b[0] + b[2]) * 0.5)
                                            cy = float(b[1] + (b[3] - b[1]) * 0.70)
                                            face_found = True
                                except Exception as exc:
                                    print(f"[!] Warning: FaceBoxes detection failed: {exc}")
                        except Exception as exc:
                            print(f"[!] Warning: Frame fetch failed for frame {ann_frame_id}: {exc}")

                    if face_found:
                        default_pts = np.array([[cx, cy]], dtype=np.float32)
                        default_lbls = np.array([1], dtype=np.int32)
                        print(f"[*] Segment {segment_id}: Re-acquired front face at frame {ann_frame_id:06d} ({cx:.1f}, {cy:.1f}). Starting new epoch...")
                        _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            points=default_pts,
                            labels=default_lbls,
                        )
                    else:
                        print(f"[*] Segment {segment_id}: No face detected at frame {ann_frame_id:06d} (dancer turned back). Enforcing zero-mask epoch (0 edit permission).")
                        h_dim, w_dim = video_source.get_dimensions()
                        zero_m = np.zeros((h_dim, w_dim), dtype=np.uint8)
                        _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            mask=zero_m,
                        )

            init_mask = (out_mask_logits[0] > 0.0).cpu().numpy()

            # Propagation loop
            prev_frame_mask = None
            orig_h, orig_w = video_source.get_dimensions()

            # If face was detected at cand_idx > 0, emit zero masks for preceding frames [0 .. ann_frame_idx - 1]
            if ann_frame_idx > 0:
                print(f"[*] Segment {segment_id}: Emitting zero-mask prefix for {ann_frame_idx} frame(s) prior to first detection...")
                for pre_idx in range(ann_frame_idx):
                    g_fid = segment_indices[pre_idx]
                    is_in_prev_overlap = (overlap_with_prev > 0 and pre_idx < overlap_with_prev)
                    if is_in_prev_overlap and stored_overlap_results and g_fid in stored_overlap_results:
                        pre_tracked = stored_overlap_results[g_fid]["mask_mask"]
                    else:
                        pre_tracked = np.zeros((orig_h, orig_w), dtype=np.uint8)

                    if resume and check_frame_completed(output_dir, g_fid, meta_logger.existing_frame_ids) and not is_in_prev_overlap:
                        continue

                    raw_bgr = video_source.get_frame(g_fid)
                    m_mask, v_mask, o_mask = decompose_masks(raw_bgr, pre_tracked, include_straps=include_straps)
                    met = compute_frame_metrics(
                        frame_id=g_fid,
                        mask_mask=m_mask,
                        visible_face_mask=v_mask,
                        occlusion_mask=o_mask,
                        prev_mask=None,
                        overlap_conflict=False,
                    )
                    fid_str = f"{g_fid:06d}"
                    save_binary_mask(m_mask, os.path.join(output_dir, "mask", f"{fid_str}.png"), orig_h, orig_w)
                    save_binary_mask(v_mask, os.path.join(output_dir, "visible_face", f"{fid_str}.png"), orig_h, orig_w)
                    save_binary_mask(o_mask, os.path.join(output_dir, "occlusion", f"{fid_str}.png"), orig_h, orig_w)
                    if save_debug and (g_fid % debug_every == 0 or pre_idx == 0):
                        dbg_path = os.path.join(output_dir, "debug", f"{fid_str}.jpg")
                        info_text = f"Frame {fid_str} | Seg {segment_id} (No Face Detected Prefix)\nMask: 0 px | Face: 0 px | Occl: 0 px"
                        save_debug_image(raw_bgr, m_mask, v_mask, o_mask, dbg_path, info_text=info_text)
                    meta_logger.write_record(met)

            for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                inference_state,
                start_frame_idx=ann_frame_idx,
                reverse=False,
                init_mask=init_mask,
                tokenizer=tokenizer,
            ):
                global_frame_id = segment_indices[out_frame_idx]
                tracked_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

                # P0 Hard Physical Firewall: Face mask CANNOT exceed 25,000 pixels.
                # If area > 25,000px, it has leaked into dress/body -> immediately suppress to 0.
                mask_px_count = int(np.sum(tracked_mask > 0))
                if mask_px_count > 25000:
                    print(f"[!] Warning: Frame {global_frame_id:06d} mask leaked into body/dress ({mask_px_count} > 25,000px)! Suppressing to zero.")
                    tracked_mask = np.zeros_like(tracked_mask)

                seg_last_mask = tracked_mask

                # Save anchor mask for the next segment at its starting frame
                if next_start_frame_id is not None and global_frame_id == next_start_frame_id:
                    next_anchor_mask = tracked_mask.copy()

                # Check if this frame is in the overlap region with the PREVIOUS segment
                is_in_prev_overlap = (overlap_with_prev > 0 and out_frame_idx < overlap_with_prev)
                overlap_conflict = False

                # Handle overlap comparison if previous segment computed this frame
                if is_in_prev_overlap and stored_overlap_results and global_frame_id in stored_overlap_results:
                    prev_res = stored_overlap_results[global_frame_id]
                    prev_m = prev_res["mask_mask"]
                    iou = compute_mask_iou(tracked_mask, prev_m)
                    if iou < 0.55:
                        overlap_conflict = True
                        print(f"[!] Warning: Overlap conflict at frame {global_frame_id:06d}: IoU={iou:.3f} < 0.55 threshold")

                    # Quality-based selection: if previous segment had noticeably higher quality, retain it
                    if prev_res["metrics"]["quality"] > 0.80 and iou >= 0.55:
                        tracked_mask = prev_m

                # Resume check: if frame is already completed and not in active overlap blending, skip write
                if resume and check_frame_completed(output_dir, global_frame_id, meta_logger.existing_frame_ids) and not is_in_prev_overlap:
                    prev_frame_mask = tracked_mask
                    continue

                # Load raw frame
                raw_bgr = video_source.get_frame(global_frame_id)

                # Decompose into the 3 standardized masks
                mask_mask, visible_face_mask, occlusion_mask = decompose_masks(
                    raw_bgr, tracked_mask, include_straps=include_straps
                )

                # Compute frame metrics
                metrics = compute_frame_metrics(
                    frame_id=global_frame_id,
                    mask_mask=mask_mask,
                    visible_face_mask=visible_face_mask,
                    occlusion_mask=occlusion_mask,
                    prev_mask=prev_frame_mask,
                    overlap_conflict=overlap_conflict,
                )

                # Store for overlap handover if this frame is in the overlap region for the NEXT segment
                if overlap_with_next > 0 and out_frame_idx >= total_in_seg - overlap_with_next:
                    new_overlap_results[global_frame_id] = {
                        "mask_mask": mask_mask,
                        "visible_face_mask": visible_face_mask,
                        "occlusion_mask": occlusion_mask,
                        "metrics": metrics,
                    }

                # Save 3 masks immediately to disk
                fid_str = f"{global_frame_id:06d}"
                mask_out_path = os.path.join(output_dir, "mask", f"{fid_str}.png")
                face_out_path = os.path.join(output_dir, "visible_face", f"{fid_str}.png")
                occl_out_path = os.path.join(output_dir, "occlusion", f"{fid_str}.png")

                save_binary_mask(mask_mask, mask_out_path, orig_h, orig_w)
                save_binary_mask(visible_face_mask, face_out_path, orig_h, orig_w)
                save_binary_mask(occlusion_mask, occl_out_path, orig_h, orig_w)

                # Save debug overlay
                if save_debug and (global_frame_id % debug_every == 0 or out_frame_idx == 0 or out_frame_idx == total_in_seg - 1):
                    dbg_path = os.path.join(output_dir, "debug", f"{fid_str}.jpg")
                    info_text = (
                        f"Frame {fid_str} | Seg {segment_id} | Mask: {metrics['mask_area']} px | Face: {metrics['visible_face_area']} px\n"
                        f"Occl: {metrics['occlusion_area']} px | Ratio: {metrics['face_occlusion_ratio']:.2f} | Q: {metrics['quality']:.2f}"
                    )
                    save_debug_image(raw_bgr, mask_mask, visible_face_mask, occlusion_mask, dbg_path, info_text=info_text)

                # Append / update metadata.jsonl
                meta_logger.write_record(metrics)

                if out_frame_idx % 25 == 0 or out_frame_idx == total_in_seg - 1:
                    print(
                        f"[*] [Seg {segment_id}] Frame {global_frame_id:06d} ({out_frame_idx + 1}/{total_in_seg}) | "
                        f"Mask: {metrics['mask_area']} | Face: {metrics['visible_face_area']} | "
                        f"OcclRatio: {metrics['face_occlusion_ratio']:.2f} | Q: {metrics['quality']:.2f}"
                    )

                prev_frame_mask = mask_mask

            mem_after_prop = get_memory_stats()
            print(f"[*] Memory After Propagation: RAM: {mem_after_prop['ram_used_gb']} GB | VRAM: {mem_after_prop['vram_alloc_gb']} GB")

            del inference_state

    finally:
        video_source.cleanup_chunk_dir(chunk_dir)
        cleanup_memory()
        mem_after_cleanup = get_memory_stats()
        print(f"[*] Memory After Cleanup: RAM: {mem_after_cleanup['ram_used_gb']} GB | VRAM: {mem_after_cleanup['vram_alloc_gb']} GB")
        print("=" * 70)

    # Use exact frame mask corresponding to next segment's start frame if available, else last mask
    carry_over = next_anchor_mask if next_anchor_mask is not None else seg_last_mask
    return carry_over, new_overlap_results


def run_segmented_pipeline(
    model,
    tokenizer,
    video_source: VideoSource,
    output_dir: str,
    start_frame: int = 0,
    end_frame: int = None,
    segment_length: int = 150,
    overlap: int = 50,
    pos_points: list = None,
    neg_points: list = None,
    prompt_box: list = None,
    include_straps: bool = False,
    save_debug: bool = True,
    debug_every: int = 1,
    resume: bool = False,
):
    """
    Main segmented chunk pipeline orchestrator.
    Executes SeC-4B tracking chunk-by-chunk with seamless overlap carry-over.
    Guarantees constant RAM usage regardless of video length.
    """
    total_video_frames = video_source.get_total_frames()
    start_idx = max(0, min(start_frame, total_video_frames - 1))
    if end_frame is None or end_frame < 0 or end_frame >= total_video_frames:
        end_idx = total_video_frames - 1
    else:
        end_idx = end_frame

    frame_range = list(range(start_idx, end_idx + 1))
    total_frames_to_process = len(frame_range)

    print("=" * 70)
    print("[*] SEC-4B VIDEO SEGMENTATION & TRACKING PIPELINE")
    print(f"[*] Source: {video_source.path} ({video_source.width}x{video_source.height})")
    print(f"[*] Frame Range: {start_idx} .. {end_idx} (Total: {total_frames_to_process} frames)")
    print(f"[*] Segment Length: {segment_length} frames | Overlap: {overlap} frames")
    print(f"[*] Include Straps: {include_straps} | Save Debug: {save_debug} (every {debug_every})")
    print(f"[*] Resume Mode: {resume}")
    print(f"[*] Output Directory: {output_dir}")
    print("=" * 70)

    # Initialize metadata logger
    meta_logger = MetadataLogger(output_dir)

    # Build overlapping segment chunks
    segments = []
    curr = 0
    while curr < total_frames_to_process:
        seg_indices = frame_range[curr : curr + segment_length]
        if not seg_indices:
            break
        segments.append(seg_indices)
        if curr + segment_length >= total_frames_to_process:
            break
        curr += (segment_length - overlap)

    print(f"[*] Video split into {len(segments)} overlapping segments.")

    prev_carry_mask = None
    stored_overlap_results = {}
    chunk_base_dir = os.path.join(output_dir, "_temp_chunks")

    start_time = time.time()

    for seg_idx, seg_indices in enumerate(segments):
        has_prev = (seg_idx > 0)
        has_next = (seg_idx < len(segments) - 1)
        overlap_with_prev = overlap if has_prev else 0
        overlap_with_next = overlap if has_next else 0
        next_start_id = segments[seg_idx + 1][0] if has_next else None

        # Resume check: if all frames in this segment are already completed, skip GPU inference
        if resume and all(check_frame_completed(output_dir, fid, meta_logger.existing_frame_ids) for fid in seg_indices):
            print(f"[*] [Seg {seg_idx}] All {len(seg_indices)} frames already completed. Skipping segment (resume).")
            if next_start_id is not None:
                next_mask_path = os.path.join(output_dir, "mask", f"{next_start_id:06d}.png")
                if os.path.exists(next_mask_path):
                    prev_carry_mask = cv2.imread(next_mask_path, cv2.IMREAD_GRAYSCALE)

            # Populate overlap results for next segment if needed
            if overlap_with_next > 0:
                stored_overlap_results = {}
                for fid in seg_indices[-overlap_with_next:]:
                    m_p = os.path.join(output_dir, "mask", f"{fid:06d}.png")
                    f_p = os.path.join(output_dir, "visible_face", f"{fid:06d}.png")
                    o_p = os.path.join(output_dir, "occlusion", f"{fid:06d}.png")
                    if os.path.exists(m_p) and fid in meta_logger.records_by_frame:
                        stored_overlap_results[fid] = {
                            "mask_mask": cv2.imread(m_p, cv2.IMREAD_GRAYSCALE),
                            "visible_face_mask": cv2.imread(f_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(f_p) else None,
                            "occlusion_mask": cv2.imread(o_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(o_p) else None,
                            "metrics": meta_logger.records_by_frame[fid],
                        }
            continue

        carry_mask, new_overlap_dict = track_and_process_segment(
            model=model,
            tokenizer=tokenizer,
            video_source=video_source,
            segment_indices=seg_indices,
            output_dir=output_dir,
            meta_logger=meta_logger,
            pos_points=pos_points,
            neg_points=neg_points,
            prompt_box=prompt_box,
            prev_carry_mask=prev_carry_mask,
            segment_id=seg_idx,
            include_straps=include_straps,
            save_debug=save_debug,
            debug_every=debug_every,
            resume=resume,
            overlap_with_prev=overlap_with_prev,
            overlap_with_next=overlap_with_next,
            next_start_frame_id=next_start_id,
            stored_overlap_results=stored_overlap_results,
            chunk_base_dir=chunk_base_dir,
        )

        prev_carry_mask = carry_mask
        stored_overlap_results = new_overlap_dict

    # Final cleanup of temp chunk directory
    if os.path.exists(chunk_base_dir):
        import shutil
        shutil.rmtree(chunk_base_dir, ignore_errors=True)
    meta_logger.close()
    if hasattr(video_source, "close"):
        video_source.close()
    elif hasattr(video_source, "cap") and getattr(video_source, "cap") is not None:
        video_source.cap.release()

    elapsed = time.time() - start_time
    fps = total_frames_to_process / max(1e-4, elapsed)
    print("\n" + "=" * 70)
    print(f"[√] Processing completed successfully!")
    print(f"[*] Processed {total_frames_to_process} frames in {elapsed:.2f}s ({fps:.2f} FPS)")
    print(f"[*] Output masks saved to: {output_dir}")
    print(f"    - mask/           (Binary uint8, 0/255)")
    print(f"    - visible_face/   (Binary uint8, 0/255)")
    print(f"    - occlusion/      (Binary uint8, 0/255)")
    print(f"    - debug/          (RGB Overlay: Red=Mask, Green=Face, Blue=Occlusion)")
    print(f"    - metadata.jsonl  (Standardized metrics per frame)")
    print("=" * 70)
