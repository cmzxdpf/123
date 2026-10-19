#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sec4b_video_masks.py - Pure Video Segmentation & Tracking CLI Entry Point
========================================================================
Main CLI interface for SeC-4B video mask extraction:
Generates 3 standardized masks: mask_mask, visible_face_mask, occlusion_mask,
plus RGB debug overlay and metadata.jsonl with strict frame indexing.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import cv2
import numpy as np

# Force local environment configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7891")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7891")
os.environ.setdefault("ALL_PROXY", "http://127.0.0.1:7891")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("HF_HOME", os.path.join(BASE_DIR, "hf_cache"))
os.environ.setdefault("TORCH_HOME", os.path.join(BASE_DIR, "torch_cache"))
os.environ.setdefault("UV_CACHE_DIR", os.path.join(BASE_DIR, "uv_cache"))
os.environ.setdefault("PIP_CACHE_DIR", os.path.join(BASE_DIR, "pip_cache"))
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from sec4b_io import VideoSource, save_debug_image
from sec4b_model import load_sec4b, cleanup_memory
from sec4b_postprocess import decompose_masks
from sec4b_tracker import run_segmented_pipeline


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (true/false).')


def parse_points(text):
    """Parse 'x,y;x,y' into [[x, y], ...]. Empty string returns []."""
    if not text:
        return []
    pts = []
    for chunk in str(text).split(";"):
        nums = [t for t in chunk.replace(",", " ").split() if t]
        if len(nums) >= 2:
            pts.append([int(round(float(nums[0]))), int(round(float(nums[1])))])
    return pts


def parse_box(text):
    """Parse 'x1,y1,x2,y2' into [x1, y1, x2, y2]."""
    if not text:
        return None
    nums = [float(t) for t in text.replace(",", " ").split() if t]
    if len(nums) == 4:
        return nums
    return None


def run_preview(model, tokenizer, video_source, frame_idx, pos_points, neg_points, prompt_box, out_path, include_straps):
    """Single-frame preview mode."""
    total = video_source.get_total_frames()
    frame_idx = max(0, min(frame_idx, total - 1))
    raw_bgr = video_source.get_frame(frame_idx)
    h, w = raw_bgr.shape[:2]

    # Create 1-frame temp directory
    tmp_dir = os.path.join(BASE_DIR, "_preview_temp")
    video_source.create_chunk_dir([frame_idx], tmp_dir)

    try:
        import torch
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            inference_state = model.grounding_encoder.init_state(
                video_path=tmp_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )
            model.grounding_encoder.reset_state(inference_state)

            if pos_points or neg_points:
                pts_list, lbls_list = [], []
                for pt in (pos_points or []):
                    pts_list.append([float(pt[0]), float(pt[1])])
                    lbls_list.append(1)
                for pt in (neg_points or []):
                    pts_list.append([float(pt[0]), float(pt[1])])
                    lbls_list.append(0)
                points_arr = np.array(pts_list, dtype=np.float32)
                labels_arr = np.array(lbls_list, dtype=np.int32)
                _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=points_arr,
                    labels=labels_arr,
                )
            elif prompt_box:
                box_arr = np.array(prompt_box, dtype=np.float32)
                _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    box=box_arr,
                )
            else:
                cx, cy = w / 2.0, h * 0.45
                try:
                    tddfa_dir = os.path.join(os.path.dirname(BASE_DIR), "3DDFA_V2")
                    if os.path.isdir(tddfa_dir) and tddfa_dir not in sys.path:
                        sys.path.insert(0, tddfa_dir)
                    try:
                        from FaceBoxes import FaceBoxes
                        fb = FaceBoxes()
                        fb_boxes = fb(raw_bgr)
                        if len(fb_boxes) > 0:
                            b = max(fb_boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
                            cx = float((b[0] + b[2]) * 0.5)
                            cy = float(b[1] + (b[3] - b[1]) * 0.70)
                    except Exception:
                        gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
                        cascade_p = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                        if os.path.isfile(cascade_p):
                            cas = cv2.CascadeClassifier(cascade_p)
                            fcs = cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
                            if len(fcs) > 0:
                                fx, fy, fw, fh = max(fcs, key=lambda b: b[2] * b[3])
                                cx, cy = float(fx + fw * 0.5), float(fy + fh * 0.70)
                except Exception:
                    pass
                pts = np.array([[cx, cy]], dtype=np.float32)
                lbls = np.array([1], dtype=np.int32)
                _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=pts,
                    labels=lbls,
                )

            tracked_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            del inference_state
    finally:
        video_source.cleanup_chunk_dir(tmp_dir)
        cleanup_memory()

    # Decompose into 3 masks
    mask_mask, visible_face_mask, occlusion_mask = decompose_masks(
        raw_bgr, tracked_mask, include_straps=include_straps
    )

    info_text = (
        f"Preview Frame {frame_idx:06d}\n"
        f"Mask: {int(mask_mask.sum() / 255)} px | Face: {int(visible_face_mask.sum() / 255)} px | Occl: {int(occlusion_mask.sum() / 255)} px"
    )
    save_debug_image(raw_bgr, mask_mask, visible_face_mask, occlusion_mask, out_path, info_text=info_text)
    print(f"[√] Preview overlay saved to: {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="SeC-4B Pure Video Segmentation & Tracking (mask_mask, visible_face_mask, occlusion_mask)"
    )
    parser.add_argument("--video", "-v", required=True, help="Path to input video file or image sequence directory")
    parser.add_argument("--output", "-o", default=os.path.join(BASE_DIR, "output", "sec4b"), help="Output directory (default: output/sec4b)")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument("--end-frame", type=int, default=-1, help="End frame index (-1 = until end of video)")
    parser.add_argument("--segment-length", type=int, default=150, help="Segment length for safe chunking (default: 150)")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap frames between segments (default: 50)")
    parser.add_argument("--device", default="cuda", help="Computation device (default: cuda)")
    parser.add_argument("--include-straps", nargs="?", const=True, type=str2bool, default=False, help="Include ear straps in mask_mask (default: false)")
    parser.add_argument("--save-debug", nargs="?", const=True, type=str2bool, default=True, help="Save RGB debug overlay images (default: true)")
    parser.add_argument("--debug-every", type=int, default=1, help="Output debug image every N frames (default: 1)")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from previous breakpoint")

    # Optional prompt coordinates
    parser.add_argument("--pos", default="", help="Positive prompt points 'x,y;x,y'")
    parser.add_argument("--neg", default="", help="Negative prompt points 'x,y;x,y'")
    parser.add_argument("--prompt-box", default="", help="Prompt bounding box 'x1,y1,x2,y2'")
    parser.add_argument("--preview-frame", type=int, default=-1, help="Generate single-frame debug preview and exit")
    parser.add_argument("--preview-out", default="", help="Output path for preview image")

    args = parser.parse_args()

    pos_points = parse_points(args.pos)
    neg_points = parse_points(args.neg)
    prompt_box = parse_box(args.prompt_box)

    print("=" * 70)
    print("[*] SEC-4B Video Segmentation & Tracking System")
    print(f"[*] Input: {args.video}")
    print(f"[*] Output: {args.output}")
    print("=" * 70)

    video_source = VideoSource(args.video)
    model, tokenizer = load_sec4b(device=args.device)

    if args.preview_frame >= 0:
        out_path = args.preview_out or os.path.join(args.output, "preview", f"preview_{args.preview_frame:06d}.jpg")
        sys.exit(run_preview(
            model=model,
            tokenizer=tokenizer,
            video_source=video_source,
            frame_idx=args.preview_frame,
            pos_points=pos_points,
            neg_points=neg_points,
            prompt_box=prompt_box,
            out_path=out_path,
            include_straps=args.include_straps,
        ))

    end_frame = None if args.end_frame < 0 else args.end_frame

    run_segmented_pipeline(
        model=model,
        tokenizer=tokenizer,
        video_source=video_source,
        output_dir=args.output,
        start_frame=args.start_frame,
        end_frame=end_frame,
        segment_length=args.segment_length,
        overlap=args.overlap,
        pos_points=pos_points,
        neg_points=neg_points,
        prompt_box=prompt_box,
        include_straps=args.include_straps,
        save_debug=args.save_debug,
        debug_every=args.debug_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
