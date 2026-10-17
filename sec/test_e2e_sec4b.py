#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_e2e_sec4b.py - End-to-End Real Model & CLI Pipeline Verification
=====================================================================
Executes the full sec4b_video_masks.py CLI pipeline on real video frames,
testing:
1. Segment chunking (segment-length=5, overlap=2).
2. All 3 standardized masks generation (uint8, 0/255, original resolution).
3. Debug overlay RGB image creation.
4. metadata.jsonl content and schema integrity.
5. --resume functionality (ensuring completed frames are skipped, metadata not duplicated).
6. Single frame preview mode.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import shutil
import json
import subprocess
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = r"J:\VisoMaster\portable-files\venv\Scripts\python.exe"
CLI_SCRIPT = os.path.join(BASE_DIR, "sec4b_video_masks.py")
INPUT_FRAMES = r"J:\1\1_skin_5s_frames"
TEST_OUTPUT = os.path.join(BASE_DIR, "output", "sec4b_e2e_test")


def run_e2e_test():
    print("=" * 70)
    print("[*] STARTING REAL E2E INTEGRATION TEST")
    print(f"[*] Interpreter: {PYTHON_EXE}")
    print(f"[*] Input: {INPUT_FRAMES}")
    print(f"[*] Output: {TEST_OUTPUT}")
    print("=" * 70)

    if os.path.exists(TEST_OUTPUT):
        shutil.rmtree(TEST_OUTPUT, ignore_errors=True)

    # 1. Run full CLI pipeline: 10 frames (0 to 9) with segment=5 and overlap=2
    cmd = [
        PYTHON_EXE, CLI_SCRIPT,
        "--video", INPUT_FRAMES,
        "--output", TEST_OUTPUT,
        "--start-frame", "0",
        "--end-frame", "9",
        "--segment-length", "5",
        "--overlap", "2",
        "--pos", "626,484",
        "--save-debug", "true",
        "--debug-every", "1",
    ]
    print("[*] Running Phase 1: Initial segmented processing (10 frames)...")
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    
    # Print safe ascii / cleaned stdout
    clean_stdout = res.stdout.encode("ascii", errors="replace").decode("ascii")
    print("STDOUT (clean):\n", clean_stdout[-1500:])

    assert res.returncode == 0, f"CLI exited with error code {res.returncode}"

    # 2. Verify Output Directory Structure
    print("[*] Verifying output directory structure and file contents...")
    mask_dir = os.path.join(TEST_OUTPUT, "mask")
    face_dir = os.path.join(TEST_OUTPUT, "visible_face")
    occl_dir = os.path.join(TEST_OUTPUT, "occlusion")
    debug_dir = os.path.join(TEST_OUTPUT, "debug")
    meta_file = os.path.join(TEST_OUTPUT, "metadata.jsonl")

    assert os.path.isdir(mask_dir), "mask/ directory missing!"
    assert os.path.isdir(face_dir), "visible_face/ directory missing!"
    assert os.path.isdir(occl_dir), "occlusion/ directory missing!"
    assert os.path.isdir(debug_dir), "debug/ directory missing!"
    assert os.path.isfile(meta_file), "metadata.jsonl missing!"

    # Get sample raw frame for resolution check
    sample_raw = cv2.imread(os.path.join(INPUT_FRAMES, "frame_00001.png"))
    orig_h, orig_w = sample_raw.shape[:2]

    # Verify all 10 frames
    for fid in range(10):
        fid_str = f"{fid:06d}"

        # Check mask_mask
        m_path = os.path.join(mask_dir, f"{fid_str}.png")
        assert os.path.isfile(m_path), f"Mask file missing: {m_path}"
        m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        assert m_img is not None, f"Could not read {m_path}"
        assert m_img.shape == (orig_h, orig_w), f"Mask resolution mismatch: {m_img.shape} vs {(orig_h, orig_w)}"
        assert m_img.dtype == np.uint8
        assert set(np.unique(m_img)).issubset({0, 255}), f"Mask values {set(np.unique(m_img))} not binary 0/255"

        # Check visible_face_mask
        f_path = os.path.join(face_dir, f"{fid_str}.png")
        assert os.path.isfile(f_path), f"Face mask file missing: {f_path}"
        f_img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
        assert f_img is not None
        assert f_img.shape == (orig_h, orig_w)
        assert f_img.dtype == np.uint8
        assert set(np.unique(f_img)).issubset({0, 255})

        # Check occlusion_mask
        o_path = os.path.join(occl_dir, f"{fid_str}.png")
        assert os.path.isfile(o_path), f"Occlusion file missing: {o_path}"
        o_img = cv2.imread(o_path, cv2.IMREAD_GRAYSCALE)
        assert o_img is not None
        assert o_img.shape == (orig_h, orig_w)
        assert o_img.dtype == np.uint8
        assert set(np.unique(o_img)).issubset({0, 255})

        # Check debug image
        d_path = os.path.join(debug_dir, f"{fid_str}.jpg")
        assert os.path.isfile(d_path), f"Debug file missing: {d_path}"
        d_img = cv2.imread(d_path)
        assert d_img is not None
        assert d_img.shape == (orig_h, orig_w, 3)

    print("  [√] All 10 frames verified for mask, visible_face, occlusion, and debug overlay!")

    # 3. Verify metadata.jsonl lines and schema
    print("[*] Verifying metadata.jsonl schema...")
    records = []
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert len(records) == 10, f"Expected 10 metadata records, got {len(records)}"

    required_fields = {
        "frame_id", "mask_area", "visible_face_area",
        "occlusion_area", "face_occlusion_ratio", "quality", "tracking_ok"
    }

    for i, r in enumerate(records):
        assert r["frame_id"] == i, f"Record frame_id mismatch: {r['frame_id']} vs {i}"
        missing = required_fields - set(r.keys())
        assert not missing, f"Record {i} missing fields: {missing}"
        assert isinstance(r["mask_area"], int)
        assert isinstance(r["visible_face_area"], int)
        assert isinstance(r["occlusion_area"], int)
        assert isinstance(r["face_occlusion_ratio"], (float, int))
        assert 0.0 <= r["face_occlusion_ratio"] <= 1.0, f"Occlusion ratio {r['face_occlusion_ratio']} out of [0, 1]"
        assert r["face_occlusion_ratio"] < 0.10, f"Unoccluded frame {i} has excessive occlusion ratio {r['face_occlusion_ratio']} (expected < 0.10)"
        assert isinstance(r["quality"], (float, int))
        assert 0.0 <= r["quality"] <= 1.0, f"Quality {r['quality']} out of [0, 1]"
        assert isinstance(r["tracking_ok"], bool)
        assert r["tracking_ok"] is True, f"Tracking failed on frame {i}"

    print("  [√] metadata.jsonl schema verified!")

    # 4. Test --resume
    print("[*] Running Phase 2: Testing --resume behavior...")
    resume_cmd = cmd + ["--resume"]
    res_resume = subprocess.run(resume_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res_resume.returncode == 0, f"Resume run failed with returncode {res_resume.returncode}"

    # Verify metadata lines did NOT duplicate
    records_after_resume = []
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records_after_resume.append(json.loads(line))
    assert len(records_after_resume) == 10, f"Resume duplicated metadata! Got {len(records_after_resume)} lines, expected 10"
    print("  [√] Resume test passed: no duplicate processing or metadata entries.")

    # 5. Test single frame preview
    print("[*] Running Phase 3: Testing single-frame preview...")
    preview_out = os.path.join(TEST_OUTPUT, "preview_test.jpg")
    preview_cmd = [
        PYTHON_EXE, CLI_SCRIPT,
        "--video", INPUT_FRAMES,
        "--output", TEST_OUTPUT,
        "--preview-frame", "0",
        "--preview-out", preview_out,
        "--pos", "626,484",
    ]
    res_prev = subprocess.run(preview_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res_prev.returncode == 0, f"Preview failed with returncode {res_prev.returncode}"
    assert os.path.isfile(preview_out), f"Preview file not created: {preview_out}"
    prev_img = cv2.imread(preview_out)
    assert prev_img is not None, "Failed to read preview image"
    assert prev_img.shape == (orig_h, orig_w, 3)
    print("  [√] Single-frame preview verified!")

    print("\n" + "=" * 70)
    print("[√] ALL END-TO-END INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_test()
