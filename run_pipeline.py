#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py - Complete End-to-End Orchestrator for Masked Face Restoration
==============================================================================
Orchestrates the entire multi-stage pipeline:
  Stage 1: SeC-4B Pure Video Mask Segmentation & Temporal Tracking
  Stage 2: Regional & Global Skin Profile Construction
  Stage 3: 3DDFA_V2 3D Geometry Tracking & Full-Face Baseline
  Stage 4: Low-Frequency Base Skin Reconstruction
  Stage 5: Outside-Face Region Deterministic Inpainting (DIS Flow + Telea)
  Stage 6: Pre-DFM Multi-Layer Composition
  Stage 7: DFM Facial Structure Restoration & Final Merge (Export MP4)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(os.path.abspath(__file__)).parent
PYTHON_EXE = sys.executable

# Ensure paths
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HOME", r"J:\SEC\hf_cache")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7891")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7891")


def log_step(title: str):
    print("\n" + "=" * 78)
    print(f"[*] {title}")
    print("=" * 78)


def run_command(cmd: list[str], desc: str, timeout: int | None = None) -> None:
    t0 = time.time()
    print(f"[*] Executing {desc}...")
    print(f"    Command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(BASE_DIR))
    try:
        ret = proc.wait(timeout=timeout)
        if ret != 0:
            raise RuntimeError(f"Step '{desc}' failed with exit code {ret}")
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as e:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise RuntimeError(f"Step '{desc}' aborted ({type(e).__name__}), process tree terminated.") from e
    dt = time.time() - t0
    print(f"[√] {desc} completed in {dt:.1f}s\n")


def count_jsonl_records(jsonl_path: Path) -> int:
    if not jsonl_path.is_file():
        return 0
    cnt = 0
    try:
        with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    cnt += 1
    except Exception:
        return 0
    return cnt


def parse_args():
    p = argparse.ArgumentParser(description="End-to-End Masked Face Inpainting Pipeline")
    p.add_argument("--video", "-v", default=None, help="Input video path or image sequence directory")
    p.add_argument("--reference-dir", "-r", default=r"D:\VisoMaster-ZH\image", help="Directory of reference face images")
    p.add_argument("--dfm-model", default=r"D:\VisoMaster-Fusion\model_assets\dfm_models\Ym.dfm", help="Path to .dfm model file")
    p.add_argument("--output-dir", "-o", default=str(BASE_DIR / "output"), help="Output root directory")
    p.add_argument("--start", type=int, default=0, help="Start frame index")
    p.add_argument("--end", type=int, default=None, help="End frame index (exclusive)")
    p.add_argument("--pos", default="", help="Optional positive prompt point x,y for mask (e.g. 576,570)")
    p.add_argument("--neg", default="", help="Optional negative prompt point x,y for mask (e.g. 500,800)")
    p.add_argument("--prompt-box", default="", help="Optional prompt bounding box x1,y1,x2,y2")
    p.add_argument("--include-straps", default="true", help="Include mask ear straps (true/false)")
    p.add_argument("--segment-length", type=int, default=200, help="SeC-4B segment length (default: 200)")
    p.add_argument("--overlap", type=int, default=50, help="SeC-4B overlap frames (default: 50)")
    p.add_argument("--device", default="cuda", help="Inference device (cuda or cpu)")
    p.add_argument("--tddfa-root", default=str(BASE_DIR / "3DDFA_V2"), help="3DDFA_V2 repository root")
    p.add_argument("--export-name", default="final_result.mp4", help="Filename of final output video")
    p.add_argument("--skip-sec", action="store_true", help="Skip Step 1 (SeC-4B)")
    p.add_argument("--skip-3ddfa", action="store_true", help="Skip Step 3 (3DDFA)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.video:
        print("=" * 78)
        print("      VisoClean (No DFM) - 视频选择")
        print("=" * 78)
        print("提示：1. 您可以直接将视频文件拖入此窗口并按回车")
        print("      2. 直接按回车将默认处理: C:\\Users\\Q\\Desktop\\2--.mp4")
        print("-" * 78)
        user_input = input("请输入或拖入视频文件路径: ").strip(' "\' \r\n\t\ufeff\u200b')
        args.video = user_input if user_input else r"C:\Users\Q\Desktop\2--.mp4"
        print(f"[*] 选定视频: {args.video}\n")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"找不到指定的视频文件: {args.video}")

    import cv2
    cap_probe = cv2.VideoCapture(str(video_path))
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap_probe.get(cv2.CAP_PROP_FPS)) or 30.0
    cap_probe.release()

    video_stem = video_path.stem
    if args.output_dir == str(BASE_DIR / "output"):
        out_root = BASE_DIR / "output" / video_stem
    else:
        out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.export_name == "final_result.mp4":
        args.export_name = f"{video_stem}_no_dfm.mp4"

    sec_out = out_root / "sec4b"
    skin_profile_out = out_root / "skin_profile" / "skin_profile.json"
    tddfa_out = out_root / "3ddfa"
    base_skin_out = out_root / "base_skin"
    outside_out = out_root / "outside"
    pre_dfm_video = out_root / "pre_dfm" / "pre_dfm.mp4"
    final_dfm_out = out_root / "final_dfm"
    final_video = out_root / args.export_name

    total_start = time.time()
    print("=" * 78)
    print("      END-TO-END MASKED FACE RESTORATION PIPELINE")
    print("=" * 78)
    print(f"[*] Input Video     : {args.video}")
    print(f"[*] Reference Faces : {args.reference_dir}")
    print(f"[*] DFM Model       : {args.dfm_model}")
    print(f"[*] Output Root     : {out_root}")
    print(f"[*] Device          : {args.device}")
    print(f"[*] Frame Range     : [{args.start}, {args.end or 'END'})")
    print("=" * 78)

    actual_end = total_frames if args.end is None else min(total_frames, args.end)
    expected_frames = max(0, actual_end - args.start)

    # -------------------------------------------------------------------------
    # Step 1: SeC-4B Segmentation & Tracking
    # -------------------------------------------------------------------------
    sec_meta = sec_out / "metadata.jsonl"
    if not args.skip_sec:
        cached_count = count_jsonl_records(sec_meta)
        if expected_frames > 0 and cached_count >= expected_frames:
            print(f"[*] Stage 1 (SeC-4B) results fully cached ({cached_count}/{expected_frames} frames) at {sec_out}, skipping.")
        else:
            log_step(f"STAGE 1: SeC-4B Mask Segmentation & Temporal Tracking (Cached: {cached_count}/{expected_frames})")
            sec_script = str(BASE_DIR / "sec" / "sec4b_video_masks.py")
            cmd1 = [
                PYTHON_EXE, sec_script,
                "--video", args.video,
                "--output", str(sec_out),
                "--start-frame", str(args.start),
                "--segment-length", str(args.segment_length),
                "--overlap", str(args.overlap),
                "--device", args.device,
                "--resume",
            ]
            if args.end is not None:
                # Note: sec4b_video_masks uses inclusive end-frame (e.g. 0..end-1)
                cmd1.extend(["--end-frame", str(max(args.start, args.end - 1))])
            if args.pos:
                cmd1.extend(["--pos", args.pos])
            if args.neg:
                cmd1.extend(["--neg", args.neg])
            if args.prompt_box:
                cmd1.extend(["--prompt-box", args.prompt_box])
            if args.include_straps:
                cmd1.extend(["--include-straps", args.include_straps])
            run_command(cmd1, "Stage 1 (SeC-4B Segmentation)")

    # -------------------------------------------------------------------------
    # Step 2: Reference Skin Profile
    # -------------------------------------------------------------------------
    log_step("STAGE 2: Build Reference Skin Profile")
    if not skin_profile_out.exists() or skin_profile_out.stat().st_size < 50:
        profile_script = str(BASE_DIR / "src" / "build_skin_profile.py")
        cmd2 = [
            PYTHON_EXE, profile_script,
            "--reference-dir", args.reference_dir,
            "--output", str(skin_profile_out),
        ]
        run_command(cmd2, "Stage 2 (Skin Profile)")
    else:
        print(f"[*] Skin profile already cached at {skin_profile_out}, skipping.")

    # -------------------------------------------------------------------------
    # Step 3: 3DDFA_V2 Geometry Baseline
    # -------------------------------------------------------------------------
    tddfa_params = tddfa_out / "params.jsonl"
    if not args.skip_3ddfa:
        cached_count = count_jsonl_records(tddfa_params)
        if expected_frames > 0 and cached_count >= expected_frames:
            print(f"[*] Stage 3 (3DDFA_V2) results fully cached ({cached_count}/{expected_frames} frames) at {tddfa_out}, skipping.")
        else:
            log_step(f"STAGE 3: 3DDFA_V2 3D Geometry Baseline (Cached: {cached_count}/{expected_frames})")
            tddfa_script = str(BASE_DIR / "src" / "run_3ddfa_geometry.py")
            cmd3 = [
                PYTHON_EXE, tddfa_script,
                "--video", args.video,
                "--sec-mask-dir", str(sec_out / "mask"),
                "--sec-occlusion-dir", str(sec_out / "occlusion"),
                "--output-dir", str(tddfa_out),
                "--tddfa-root", args.tddfa_root,
                "--reference-dir", args.reference_dir,
                "--start", str(args.start),
                "--device", f"{args.device}:0" if args.device.startswith("cuda") else "cpu",
            ]
            if args.end is not None:
                cmd3.extend(["--end", str(args.end)])
            run_command(cmd3, "Stage 3 (3DDFA Geometry)")

    # -------------------------------------------------------------------------
    # Step 4: Reconstruct Base Skin
    # -------------------------------------------------------------------------
    base_meta = base_skin_out / "metadata.jsonl"
    cached_count = count_jsonl_records(base_meta)
    if expected_frames > 0 and cached_count >= expected_frames:
        print(f"[*] Stage 4 (Base Skin) results fully cached ({cached_count}/{expected_frames} frames) at {base_skin_out}, skipping.")
    else:
        log_step(f"STAGE 4: Reconstruct Low-Frequency Base Skin (Cached: {cached_count}/{expected_frames})")
        base_script = str(BASE_DIR / "src" / "reconstruct_base_skin.py")
        cmd4 = [
            PYTHON_EXE, base_script,
            "--video", args.video,
            "--sec4b-dir", str(sec_out),
            "--3ddfa-dir", str(tddfa_out),
            "--skin-profile", str(skin_profile_out),
            "--output", str(base_skin_out),
            "--start", str(args.start),
            "--resume",
        ]
        if args.end is not None:
            cmd4.extend(["--end", str(args.end)])
        run_command(cmd4, "Stage 4 (Base Skin Reconstruction)")

    # -------------------------------------------------------------------------
    # Step 5: Restore Outside Region
    # -------------------------------------------------------------------------
    outside_meta = outside_out / "metadata.jsonl"
    cached_count = count_jsonl_records(outside_meta)
    if expected_frames > 0 and cached_count >= expected_frames:
        print(f"[*] Stage 5 (Outside Region Restoration) results fully cached ({cached_count}/{expected_frames} frames) at {outside_out}, skipping.")
    else:
        log_step(f"STAGE 5: Restore Outside-Face Region (DIS Flow + Telea) (Cached: {cached_count}/{expected_frames})")
        outside_script = str(BASE_DIR / "src" / "restore_outside_region.py")
        cmd5 = [
            PYTHON_EXE, outside_script,
            "--video", args.video,
            "--sec4b-dir", str(sec_out),
            "--3ddfa-dir", str(tddfa_out),
            "--output", str(outside_out),
            "--start", str(args.start),
            "--resume",
        ]
        if args.end is not None:
            cmd5.extend(["--end", str(args.end)])
        cmd5.append("--disable-temporal")
        run_command(cmd5, "Stage 5 (Outside Region Restoration)")

    # -------------------------------------------------------------------------
    # Step 6: Compose Pre-DFM
    # -------------------------------------------------------------------------
    stage6_manifest = Path(str(pre_dfm_video) + ".manifest.json")
    stage6_valid = False
    if pre_dfm_video.exists() and stage6_manifest.exists():
        try:
            import json
            m_data = json.loads(stage6_manifest.read_text(encoding="utf-8"))
            if m_data.get("status") == "complete" and m_data.get("frames_written", 0) >= expected_frames:
                stage6_valid = True
        except Exception:
            pass

    if stage6_valid:
        print(f"[*] Stage 6 (Compose Pre-DFM) video verified complete at {pre_dfm_video}, skipping.")
    else:
        log_step("STAGE 6: Compose Pre-DFM Multi-Layer Fusion")
        compose_script = str(BASE_DIR / "src" / "compose_pre_dfm.py")
        cmd6 = [
            PYTHON_EXE, compose_script,
            "--video", args.video,
            "--sec4b-dir", str(sec_out),
            "--3ddfa-dir", str(tddfa_out),
            "--base-skin-dir", str(base_skin_out),
            "--outside-dir", str(outside_out),
            "--output", str(pre_dfm_video),
            "--start", str(args.start),
            "--resume",
        ]
        if args.end is not None:
            cmd6.extend(["--end", str(args.end)])
        run_command(cmd6, "Stage 6 (Compose Pre-DFM)")

    # -------------------------------------------------------------------------
    # Step 7: Remux Audio into Final Export Video (DFM Bypassed)
    # -------------------------------------------------------------------------
    log_step("STAGE 7: Final Audio Mux (DFM Bypassed)")
    start_sec = args.start / fps
    actual_end = total_frames if args.end is None else min(total_frames, args.end)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(pre_dfm_video),
    ]
    if args.start > 0:
        ffmpeg_cmd.extend(["-ss", f"{start_sec:.4f}"])
    if args.end is not None:
        duration_sec = (actual_end - args.start) / fps
        ffmpeg_cmd.extend(["-t", f"{duration_sec:.4f}"])
    ffmpeg_cmd.extend([
        "-i", str(video_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy" if (args.start == 0 and args.end is None) else "aac",
        "-shortest",
        str(final_video),
    ])
    run_command(ffmpeg_cmd, "Final Audio Remux into export-name")

    total_time = time.time() - total_start
    print("=" * 78)
    print("   [√] PIPELINE (NO DFM) EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 78)
    print(f"[*] Total Time Taken : {total_time:.1f}s")
    print(f"[*] Final Video Saved: {final_video}")
    print("=" * 78)


if __name__ == "__main__":
    main()
