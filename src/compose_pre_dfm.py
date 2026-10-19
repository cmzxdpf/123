#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6: Multi-Threaded High-Performance Pre-DFM Multi-Layer Fusion."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Optional, Tuple

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Compose pre-DFM intermediate frames and high-quality video")
    p.add_argument("--video", required=True)
    p.add_argument("--sec4b-dir", required=True)
    p.add_argument("--3ddfa-dir", dest="ddfa_dir", required=True)
    p.add_argument("--base-skin-dir", required=True)
    p.add_argument("--outside-dir", required=True)
    p.add_argument("--output", required=True, help="Output mp4 path or output directory")
    p.add_argument("--save-frames-dir", default=None, help="Optional directory to save lossless PNG frames")
    p.add_argument("--transition-px", type=int, default=6, help="Jawline feather transition width between Task3 and Task4")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="Exclusive")
    p.add_argument("--crf", type=int, default=10, help="FFmpeg libx264 CRF (default 10 near-lossless)")
    p.add_argument("--debug-every", type=int, default=0)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--workers", type=int, default=16, help="Worker threads for parallel frame processing")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def cv_imread_unicode(file_path: Path | str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(file_path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


def cv_imwrite_unicode(file_path: Path | str, img: np.ndarray, params=None) -> bool:
    try:
        ext = Path(file_path).suffix or ".png"
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            return False
        buf.tofile(str(file_path))
        return True
    except Exception:
        return False


def read_mask(path: Path, shape, strict=False) -> Optional[np.ndarray]:
    im = cv_imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        if strict:
            raise FileNotFoundError(path)
        return None
    if im.shape != shape:
        if strict:
            raise ValueError(f"mask size mismatch: {path}: {im.shape} != {shape}")
        im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return im > 127


def fill_mask_holes(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Fills internal holes (3D nostril cavities, mouth cavity) in the face mask."""
    if mask is None or not np.any(mask):
        return mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(solid, contours, -1, 1, -1)
    return solid > 0


def read_frame(path: Path, shape, strict=False) -> Optional[np.ndarray]:
    im = cv_imread_unicode(path, cv2.IMREAD_COLOR)
    if im is None:
        if strict:
            raise FileNotFoundError(path)
        return None
    if im.shape[:2] != shape:
        if strict:
            raise ValueError(f"frame size mismatch: {path}: {im.shape[:2]} != {shape}")
        im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return im


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def blend_task3_task4(
    original: np.ndarray,
    base: Optional[np.ndarray],
    outside: Optional[np.ndarray],
    t3eff: np.ndarray,
    t4eff: np.ndarray,
    occ: np.ndarray,
    transition_px: int,
) -> np.ndarray:
    """无缝合成：先铺设 Task 4 脸外修补(含外扩消边区)，再通过下颌羽化叠加 Task 3 伪皮肤，最后强制保护手部遮挡。"""
    result = original.copy()
    if outside is not None and np.any(t4eff):
        result[t4eff] = outside[t4eff]

    if base is not None and np.any(t3eff):
        # 基础：完全保留 Task 3 伪皮肤（鼻梁、脸颊上边缘保持 Stage 4 的干净羽化，严禁与 original 衰减）
        result[t3eff] = base[t3eff]

        # 仅在 t3eff 与 t4eff（脸外耳带修补区）的交界带进行羽化，在 base 与 outside 之间过渡
        if transition_px > 0 and outside is not None and np.any(t4eff):
            k_sz = transition_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
            near_t4 = cv2.dilate(t4eff.astype(np.uint8), kernel) > 0
            trans_zone = t3eff & near_t4
            if np.any(trans_zone):
                dist = cv2.distanceTransform(t3eff.astype(np.uint8), cv2.DIST_L2, 3)
                alpha = np.clip(dist / float(max(1, transition_px)), 0.15, 1.0)[:, :, np.newaxis]
                base_f = base.astype(np.float32)
                out_f = outside.astype(np.float32)
                blended = np.clip(base_f * alpha + out_f * (1.0 - alpha), 0, 255).astype(np.uint8)
                result[trans_zone] = blended[trans_zone]

    # 真实前景遮挡(手掌/头发)：仅保留在人脸修复区(t3eff)之外的前景，严禁假遮挡把口罩黑边贴回
    true_occ = occ & ~t3eff
    result[true_occ] = original[true_occ]
    return result


def make_debug(original, result, sec, face, occ, t3, t4):
    def vis(m):
        return cv2.cvtColor((m.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    top = np.hstack([original, result])
    masks = [vis(x) for x in (sec, face, occ, t3, t4)]
    target_w = top.shape[1]
    cell_w = max(1, target_w // len(masks))
    resized = [cv2.resize(x, (cell_w, max(1, int(x.shape[0] * cell_w / x.shape[1])))) for x in masks]
    h = min(x.shape[0] for x in resized)
    bottom = np.hstack([x[:h] for x in resized])
    bottom = cv2.resize(bottom, (target_w, bottom.shape[0]))
    return np.vstack([top, bottom])


def _process_single_frame(
    frame_id: int,
    original: np.ndarray,
    shape: Tuple[int, int],
    sec_dir: Path,
    ddfa_dir: Path,
    base_dir: Path,
    outside_dir: Path,
    transition_px: int,
    strict: bool,
):
    name = f"{frame_id:06d}.png"
    try:
        sec = read_mask(sec_dir / "mask" / name, shape, strict)
        face = fill_mask_holes(read_mask(ddfa_dir / "face_mask" / name, shape, strict))
        occ = read_mask(sec_dir / "occlusion" / name, shape, False)
        t3mask = read_mask(base_dir / "masks" / name, shape, strict)
        t4mask = read_mask(outside_dir / "masks" / name, shape, strict)
        base = read_frame(base_dir / "frames" / name, shape, strict)
        outside = read_frame(outside_dir / "frames" / name, shape, strict)

        if occ is None:
            occ = np.zeros(shape, dtype=bool)
        if sec is None or face is None:
            t3eff = np.zeros(shape, bool)
            t4eff = np.zeros(shape, bool)
            result = original.copy()
        else:
            t3eff = (t3mask if t3mask is not None else (sec & face)) & face
            t4_raw = t4mask if t4mask is not None else (sec & ~face)
            t4eff = (t4_raw | (sec & ~face)) & ~face & ~occ
            result = blend_task3_task4(original, base, outside, t3eff, t4eff, occ, transition_px)

        rec = {
            "frame_id": frame_id,
            "status": "ok",
            "task3_pixels": int(t3eff.sum()),
            "task4_pixels": int(t4eff.sum()),
            "occlusion_pixels": int(occ.sum()),
        }
        return frame_id, original, result, sec, face, occ, t3eff, t4eff, rec
    except Exception as e:
        if strict:
            raise
        rec = {
            "frame_id": frame_id,
            "status": f"fallback: {e}",
            "task3_pixels": 0,
            "task4_pixels": 0,
            "occlusion_pixels": 0,
        }
        return frame_id, original, original.copy(), None, None, None, None, None, rec


def main():
    a = parse_args()
    if a.start < 0 or (a.end is not None and a.end < a.start):
        raise SystemExit("invalid [start,end) range")

    video = Path(a.video)
    sec_dir = Path(a.sec4b_dir)
    ddfa_dir = Path(a.ddfa_dir)
    base_dir = Path(a.base_skin_dir)
    outside_dir = Path(a.outside_dir)
    out = Path(a.output)
    if out.suffix.lower() != ".mp4":
        out = out / "pre_dfm.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    frames_out_dir = Path(a.save_frames_dir) if a.save_frames_dir else None
    if frames_out_dir:
        frames_out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out.parent / "metadata.jsonl"
    manifest_path = Path(str(out) + ".manifest.json")
    debug_dir = out.parent / "debug"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end = total if a.end is None else min(a.end, total)

    if a.debug_every > 0:
        debug_dir.mkdir(parents=True, exist_ok=True)

    if a.overwrite:
        for p in (out, manifest_path, meta_path):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
    elif a.resume or not a.overwrite:
        if manifest_path.exists() and out.exists():
            try:
                m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if m_data.get("status") == "complete" and m_data.get("frames_written") == (end - a.start):
                    print(f"[√] Stage 6 output already complete ({end - a.start} frames). Resuming/skipping.")
                    return
            except Exception:
                pass

    tmp_out = Path(str(out) + ".part.mp4")
    tmp_meta = Path(str(meta_path) + ".part")
    for p in (tmp_out, tmp_meta):
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    print(f"[*] Starting Multi-Threaded Pre-DFM Composition Engine ({a.workers} worker threads on i9-13900K)...")
    print(f"[*] Total Frames to Compose: {end - a.start} [{a.start} -> {end})")
    print(f"[*] Lossless PNG Saving   : {'ENABLED (' + str(frames_out_dir) + ')' if frames_out_dir else 'DISABLED (Direct Streaming to FFmpeg for max speed)'}")

    ffmpeg_proc = None
    writer = None
    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "bgr24", "-r", f"{fps:.6f}",
            "-i", "-", "-c:v", "libx264", "-crf", str(a.crf), "-preset", "veryfast",
            "-pix_fmt", "yuv420p", str(tmp_out),
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception:
        ffmpeg_proc = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_out), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    count = 0
    failures = 0
    t_start = time.time()
    last_log_time = t_start
    last_log_count = 0

    in_flight = {}
    MAX_IN_FLIGHT = 64
    next_frame_to_submit = a.start

    with tmp_meta.open("w", encoding="utf-8") as mf:
        with ThreadPoolExecutor(max_workers=a.workers) as executor:
            # Prefill in-flight queue
            while next_frame_to_submit < end and len(in_flight) < MAX_IN_FLIGHT:
                ok, original = cap.read()
                if not ok or original is None:
                    break
                shape = original.shape[:2]
                fut = executor.submit(
                    _process_single_frame,
                    next_frame_to_submit,
                    original,
                    shape,
                    sec_dir,
                    ddfa_dir,
                    base_dir,
                    outside_dir,
                    a.transition_px,
                    a.strict,
                )
                in_flight[next_frame_to_submit] = fut
                next_frame_to_submit += 1

            for curr_frame_id in range(a.start, end):
                if curr_frame_id not in in_flight:
                    break

                fut = in_flight.pop(curr_frame_id)
                frame_id, orig, result, sec, face, occ, t3eff, t4eff, rec = fut.result()

                # Feed next frame into executor to keep pipeline full
                if next_frame_to_submit < end:
                    ok, original = cap.read()
                    if ok and original is not None:
                        f_shape = original.shape[:2]
                        new_fut = executor.submit(
                            _process_single_frame,
                            next_frame_to_submit,
                            original,
                            f_shape,
                            sec_dir,
                            ddfa_dir,
                            base_dir,
                            outside_dir,
                            a.transition_px,
                            a.strict,
                        )
                        in_flight[next_frame_to_submit] = new_fut
                        next_frame_to_submit += 1

                if rec["status"] != "ok":
                    failures += 1

                if ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
                    ffmpeg_proc.stdin.write(result.tobytes())
                elif writer is not None:
                    writer.write(result)

                if frames_out_dir is not None:
                    cv_imwrite_unicode(frames_out_dir / f"{frame_id:06d}.png", result, [cv2.IMWRITE_PNG_COMPRESSION, 2])

                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if a.debug_every > 0 and (frame_id - a.start) % a.debug_every == 0 and sec is not None and face is not None:
                    dbg = make_debug(orig, result, sec, face, occ, t3eff, t4eff)
                    cv_imwrite_unicode(debug_dir / f"{frame_id:06d}.jpg", dbg, [cv2.IMWRITE_JPEG_QUALITY, 90])

                count += 1

                if count % 100 == 0 or curr_frame_id == end - 1:
                    now = time.time()
                    curr_batch = count - last_log_count
                    fps_curr = curr_batch / max(0.001, now - last_log_time)
                    fps_overall = count / max(0.001, now - t_start)
                    rem = end - 1 - curr_frame_id
                    eta = rem / max(0.1, fps_curr)
                    pct = (curr_frame_id + 1 - a.start) / max(1, end - a.start) * 100.0
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"{ts} [INFO] Frame [{curr_frame_id+1:06d}/{end:06d}] ({pct:5.1f}%) | "
                        f"Speed: {fps_curr:5.2f} fps (Avg: {fps_overall:5.2f} fps) | ETA: {int(eta//60)}m {int(eta%60):02d}s",
                        flush=True,
                    )
                    last_log_time = now
                    last_log_count = count

    if ffmpeg_proc is not None:
        if ffmpeg_proc.stdin:
            ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    if writer is not None:
        writer.release()
    cap.release()

    import shutil
    time.sleep(0.2)
    for p_src, p_dst in ((tmp_out, out), (tmp_meta, meta_path)):
        if p_src.exists():
            for attempt in range(10):
                try:
                    if p_dst.exists():
                        p_dst.unlink()
                    shutil.move(str(p_src), str(p_dst))
                    break
                except Exception:
                    time.sleep(0.5)
    atomic_json(manifest_path, {"status": "complete", "frames_written": count, "failures": failures, "output": str(out.resolve())})
    total_sec = time.time() - t_start
    print(f"\n[√] Stage 6 Composition Complete in {total_sec:.1f}s ({count / max(0.001, total_sec):.1f} fps overall)!")
    print(f"[*] Pre-DFM Video: {out} ({count} frames, failures={failures})")


if __name__ == "__main__":
    main()