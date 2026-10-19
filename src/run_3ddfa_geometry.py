#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3DDFA_V2 geometry baseline / theoretical full-face mask generator (TASK 01).

This module processes a *single-face* video with 3DDFA_V2 and produces, per
frame, in the ORIGINAL video resolution:

    output/3ddfa/
    ├── face_mask/     000000.png ...      (uint8, binary 0/255 "theoretical full face")
    ├── landmarks/     000000.npy ...      (float32 (68, 2) landmarks, image pixels)
    │                  000000_contour.npy  (optional silhouette polygon, float32 (M, 2))
    ├── debug/         000000.jpg ...      (optional overlay, every --debug-every frames)
    ├── params.jsonl                       (one JSON record per frame)
    └── reference_baseline.npz             (cached shape baseline from reference faces)

Design goals (see task spec):
  * stable temporal tracking (never per-frame from-scratch detection/fitting);
  * identity/shape regularized to the reference baseline, pose/expression follow the video;
  * occlusion aware updates (optional SeC mask / occlusion mask inputs are never modified);
  * previous-stable / motion-predicted fallback, never a silently empty mask;
  * resume support (completed prefix is detected, processing restarts at the first
    incomplete frame without ever desyncing decoder position and frame ids),
    bounded RAM, models initialized exactly once;
  * theoretical face mask generated from the projected 3DMM mesh (back faces culled),
    NOT from the mask fabric outline and NOT a landmark convex hull.

Typical usage (run from the project root that contains src/):

    python src/run_3ddfa_geometry.py \
        --video input/dance_masked.mp4 \
        --reference-dir input/reference_faces \
        --sec-mask-dir output/sec4b/mask \
        --sec-occlusion-dir output/sec4b/occlusion \
        --output-dir output/3ddfa \
        --device cuda:0 \
        --debug --debug-every 30

Frame ranges use ``--start`` / ``--end`` with **end-exclusive** semantics,
i.e. ``--start 1200 --end 1800`` processes frames 1200..1799.

The 3DDFA_V2 repository must be available locally; point to it with
``--tddfa-root`` or the ``TDDFA_ROOT`` environment variable, or place it as
``./3DDFA_V2`` next to the working directory.  All fork specific integration is
encapsulated in :class:`TDDFAAdapter`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "opencv-python is required.  Install dependencies with:\n"
        "    pip install numpy opencv-python torch PyYAML tqdm scipy"
    ) from exc

LOG_NAME = "run_3ddfa_geometry"
LOG = logging.getLogger(LOG_NAME)

__version__ = "1.0.0"

FRAME_FMT = "{:06d}"  # 6-digit zero padded frame ids everywhere

# ---------------------------------------------------------------------------
# Tunables (all overridable from the CLI unless noted)
# ---------------------------------------------------------------------------

MIN_MASK_AREA = 64               # a smaller theoretical-face area is treated as invalid
REINIT_LOST_FRAMES_DEFAULT = 20  # lost frames after which a fresh detection re-initializes the track
FACEBOXES_MIN_SCORE = 0.30       # ignore weaker raw detections entirely
ASSOC_MIN_SCORE = 0.30           # minimum association score to accept a detection as the tracked face
DET_SCORE_DECAY = 0.995          # per-frame decay of the carried detector confidence
SHAPE_DEV_CLIP = 2.5             # clip shape-EMA deviation from the reference baseline (alpha units)
BLUR_FULL_CREDIT = 6.0           # default blur threshold (scaled by face size), CLI: --blur-threshold

# quality score weights (they are normalized to sum 1 in code)
QUALITY_WEIGHTS: Dict[str, float] = {
    "detector": 0.28,
    "iou": 0.14,
    "center": 0.10,
    "pose": 0.12,
    "scale": 0.08,
    "landmark": 0.12,
    "blur": 0.05,
    "out_of_frame": 0.06,
    "occlusion": 0.05,
}

# One Euro filter defaults per signal group (parameters are CLI adjustable via
# --smooth-min-cutoff and --smooth-beta which multiply these defaults).
ONEEURO_DEFAULTS: Dict[str, Tuple[float, float]] = {
    # group: (min_cutoff, beta)
    "pose12": (1.2, 2.0),   # first 12 3DMM params (s*R | t), crop space
    "exp": (1.2, 1.5),      # expression alphas
}

SHAPE_EMA_ALPHA = 0.12           # base EMA alpha for shape (further scaled by update weights)


class FatalError(RuntimeError):
    """Errors that must terminate the whole run (missing models, unreadable video...)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def clamp01(x: float) -> float:
    """Clamp a scalar into [0, 1]; NaN maps to 0."""
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def wrap180(deg: float) -> float:
    """Wrap an angle in degrees into (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def is_finite_array(arr: np.ndarray) -> bool:
    """Return True when every element is finite."""
    return bool(np.isfinite(np.asarray(arr, dtype=np.float64)).all())


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    """JSON fallback for numpy scalar types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def imread_bgr(path: Path) -> Optional[np.ndarray]:
    """Read an image as BGR or return None (Unicode/Windows path safe)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img if isinstance(img, np.ndarray) and img.size > 0 else None
    except OSError:
        return None


def imread_gray(path: Path) -> Optional[np.ndarray]:
    """Read an image as single channel gray or return None."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return img if isinstance(img, np.ndarray) and img.size > 0 else None
    except OSError:
        return None


def atomic_write_png(path: Path, image: np.ndarray) -> None:
    """Encode ``image`` as PNG and atomically replace ``path``."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"cv2.imencode failed for {path}")
    tmp = path.with_name(path.name + ".tmp")
    buf.tofile(str(tmp))
    os.replace(str(tmp), str(path))


def atomic_write_jpg(path: Path, image: np.ndarray, quality: int = 90) -> None:
    """Encode ``image`` as JPEG and atomically replace ``path``."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise OSError(f"cv2.imencode failed for {path}")
    tmp = path.with_name(path.name + ".tmp")
    buf.tofile(str(tmp))
    os.replace(str(tmp), str(path))


def atomic_write_npy(path: Path, array: np.ndarray) -> None:
    """Atomically save a numpy array (``path`` must end with ``.npy``)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.asarray(array), allow_pickle=False)
    os.replace(str(tmp), str(path))


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write utf-8 text."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(str(tmp), str(path))


def _windows_short_path(path: Path) -> Optional[str]:
    """Return an 8.3 short path on Windows (helps legacy camera/video backends)."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(4096)
        res = get_short(str(path), buf, 4096)
        if 0 < res < 4096:
            return buf.value or None
    except Exception:  # pragma: no cover - best effort only
        return None
    return None


def open_video(path: Path) -> "cv2.VideoCapture":
    """Open a video with cv2, transparently handling Unicode Windows paths."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        short = _windows_short_path(path)
        if short:
            cap.release()
            cap = cv2.VideoCapture(short)
    if not cap.isOpened():
        cap.release()
        raise FatalError(f"cannot open video: {path}")
    return cap


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in np.asarray(a, dtype=float)[:4])
    bx1, by1, bx2, by2 = (float(v) for v in np.asarray(b, dtype=float)[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def face_metrics(verts68: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float, float, float]]:
    """Return (centroid, face_size, lm_bbox) from ``(3, 68)`` projected landmarks.

    ``face_size`` is the geometric mean of the landmark bbox sides (pixel units),
    robust against pure in-plane rotation.
    """
    x = np.asarray(verts68[0], dtype=np.float64)
    y = np.asarray(verts68[1], dtype=np.float64)
    w = float(max(x.max() - x.min(), 1e-3))
    h = float(max(y.max() - y.min(), 1e-3))
    size = float(math.sqrt(w * h))
    centroid = np.array([x.mean(), y.mean()], dtype=np.float64)
    lm_bbox = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))
    return centroid, size, lm_bbox


def out_of_frame_ratio(lm_bbox: Tuple[float, float, float, float], width: int, height: int) -> float:
    """Fraction of the landmark bbox area that falls outside the frame."""
    x1, y1, x2, y2 = lm_bbox
    ix1, iy1 = max(x1, 0.0), max(y1, 0.0)
    ix2, iy2 = min(x2, float(width)), min(y2, float(height))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    total = max((x2 - x1) * (y2 - y1), 1e-6)
    return float(1.0 - (iw * ih) / total)


def blur_score(gray: np.ndarray, bbox: np.ndarray) -> float:
    """Variance-of-Laplacian blur metric normalized by the crop scale.

    Larger values mean sharper crops.  Returns 0.0 when the crop is invalid.
    """
    h, w = gray.shape[:2]
    x1 = int(max(0, min(w - 1, math.floor(float(bbox[0])))))
    y1 = int(max(0, min(h - 1, math.floor(float(bbox[1])))))
    x2 = int(max(x1 + 1, min(w, math.ceil(float(bbox[2])))))
    y2 = int(max(y1 + 1, min(h, math.ceil(float(bbox[3])))))
    crop = gray[y1:y2, x1:x2]
    if crop.size < 64:
        return 0.0
    lap = cv2.Laplacian(crop, cv2.CV_64F)
    var = float(lap.var())
    area_scale = max((crop.shape[0] * crop.shape[1]) / (300.0 * 300.0), 0.05)
    return var / area_scale


def mask_ratio(reference: np.ndarray, other: Optional[np.ndarray]) -> Optional[float]:
    """Fraction of ``other`` (>0) pixels inside ``reference`` binary mask region."""
    if reference is None or other is None:
        return None
    ref = reference > 0
    total = int(ref.sum())
    if total <= 0:
        return None
    other_bin = other > 127
    if other_bin.shape != ref.shape:
        other_bin = cv2.resize(
            other_bin.astype(np.uint8), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST
        ) > 0
    hit = int(np.logical_and(ref, other_bin).sum())
    return float(hit) / float(total)


# ---------------------------------------------------------------------------
# Geometry math (faithful ports of the official 3DDFA_V2 helpers so that the
# module stays fork independent; the official repo maths are tiny pure numpy)
# ---------------------------------------------------------------------------

def parse_roi_box_from_bbox(bbox: Sequence[float]) -> np.ndarray:
    """Port of ``utils.functions.parse_roi_box_from_bbox`` (official 3DDFA_V2)."""
    left, top, right, bottom = (float(v) for v in bbox[:4])
    old_size = (right - left + bottom - top) / 2.0
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    roi = np.empty(4, dtype=np.float32)
    roi[0] = center_x - size / 2.0
    roi[1] = center_y - size / 2.0
    roi[2] = roi[0] + size
    roi[3] = roi[1] + size
    return roi


def similar_transform(pts3d: np.ndarray, roi_box: Sequence[float], size: int) -> np.ndarray:
    """Port of ``utils.tddfa_util.similar_transform``: 3DMM crop space -> image px."""
    pts = np.array(pts3d, dtype=np.float32, copy=True)
    pts[0, :] -= 1.0
    pts[2, :] -= 1.0
    pts[1, :] = float(size) - pts[1, :]
    sx, sy, ex, ey = (float(v) for v in roi_box[:4])
    scale_x = (ex - sx) / float(size)
    scale_y = (ey - sy) / float(size)
    pts[0, :] = pts[0, :] * scale_x + sx
    pts[1, :] = pts[1, :] * scale_y + sy
    scale = (scale_x + scale_y) / 2.0
    pts[2, :] *= scale
    pts[2, :] -= float(np.min(pts[2, :]))
    return pts


def parse_param(param: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of ``utils.tddfa_util._parse_param`` (62 = 12 pose + 40 shape + 10 exp)."""
    param = np.asarray(param, dtype=np.float64).reshape(-1)
    n = param.shape[0]
    if n not in (62, 72):
        raise ValueError(f"unsupported 3DMM param length {n} (expected 62 or 72)")
    trans_dim, shape_dim = 12, 40
    p = param[:trans_dim].reshape(3, -1)
    rot = p[:, :3]
    offset = p[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)
    return rot, offset, alpha_shp, alpha_exp


def p2s_rt(affine: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Port of ``utils.pose.P2sRt``: decompose a (3, 4) affine camera matrix."""
    p = np.asarray(affine, dtype=np.float64)
    t3d = p[:, 3].copy()
    r1 = p[0:1, :3]
    r2 = p[1:2, :3]
    n1 = float(np.linalg.norm(r1))
    n2 = float(np.linalg.norm(r2))
    if n1 < 1e-9 or n2 < 1e-9:
        raise ValueError("degenerate 3DMM rotation")
    s = (n1 + n2) / 2.0
    r1 = r1 / n1
    r2 = r2 / n2
    r3 = np.cross(r1, r2)
    rot = np.concatenate((r1, r2, r3), axis=0)
    return float(s), rot, t3d


def matrix2angles_deg(rot: np.ndarray) -> Tuple[float, float, float]:
    """Port of ``utils.pose.matrix2angle``; returns (yaw, pitch, roll) degrees.

    The official code labels the Euler components ``x: yaw, y: pitch, z: roll``;
    this module follows exactly that convention.
    """
    r = np.asarray(rot, dtype=np.float64)
    if r[2, 0] > 0.998:
        z = 0.0
        x = np.pi / 2.0
        y = z + math.atan2(-r[0, 1], -r[0, 2])
    elif r[2, 0] < -0.998:
        z = 0.0
        x = -np.pi / 2.0
        y = -z + math.atan2(r[0, 1], r[0, 2])
    else:
        x = math.asin(float(r[2, 0]))
        y = math.atan2(float(r[2, 1]) / math.cos(x), float(r[2, 2]) / math.cos(x))
        z = math.atan2(float(r[1, 0]) / math.cos(x), float(r[0, 0]) / math.cos(x))
    return float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))


def pose_angles_deg(param: np.ndarray) -> Tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a 62-dim 3DMM parameter vector."""
    affine = np.asarray(param, dtype=np.float64)[:12].reshape(3, 4)
    _, rot, _ = p2s_rt(affine)
    return matrix2angles_deg(rot)


def param_scale_translation(param: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (scale, t3d) from a 3DMM parameter vector (crop space)."""
    affine = np.asarray(param, dtype=np.float64)[:12].reshape(3, 4)
    s, _, t3d = p2s_rt(affine)
    return s, t3d


def rebuild_param(param12: np.ndarray, shape40: np.ndarray, exp10: np.ndarray) -> np.ndarray:
    """Assemble a 62-dim parameter vector from its parts."""
    vec = np.concatenate(
        [
            np.asarray(param12, dtype=np.float64).reshape(-1),
            np.asarray(shape40, dtype=np.float64).reshape(-1),
            np.asarray(exp10, dtype=np.float64).reshape(-1),
        ]
    )
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal filters
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """One Euro filter (Casiez et al. 2012), vectorized over signal dimensions.

    Slow motion is smoothed strongly (low cutoff), fast motion is followed
    promptly because the cutoff is raised by the derivative estimate.
    """

    def __init__(self, dim: int, min_cutoff: float = 1.2, beta: float = 1.0, d_cutoff: float = 1.0,
                 gap_threshold: float = 0.5) -> None:
        self.dim = int(dim)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # ``t`` is REAL TIME IN SECONDS (see Runner._frame_time): mixing frame
        # indices in here made every dt == 1.0, which always exceeded this gap
        # threshold and therefore reset the filter every single frame, silently
        # disabling all temporal smoothing.
        self._gap_threshold = float(gap_threshold)
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
        return dt / (dt + tau)

    @property
    def initialized(self) -> bool:
        """Whether the filter holds a previous state."""
        return self._x_prev is not None

    def reset(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """Restart the filter at value ``x``."""
        self._x_prev = np.asarray(x, dtype=np.float64).reshape(self.dim).copy()
        self._dx_prev = np.zeros(self.dim, dtype=np.float64)
        self._t_prev = float(t) if t is not None else None
        return self._x_prev.copy()

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        """Feed measurement ``x`` sampled at time ``t`` and return the filtered value."""
        xt = np.asarray(x, dtype=np.float64).reshape(self.dim)
        if self._x_prev is None or self._t_prev is None:
            return self.reset(xt, t)
        dt = float(t) - float(self._t_prev)
        if dt <= 1e-6:
            return self._x_prev.copy()
        if dt > self._gap_threshold:  # large gap (resume / long fallback): re-snap
            return self.reset(xt, t)
        dx = (xt - self._x_prev) / dt
        a_d = self._alpha(np.array([self.d_cutoff]), dt)[0]
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * xt + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = float(t)
        return x_hat.copy()

    def value(self) -> Optional[np.ndarray]:
        """Current filtered value (None before the first sample)."""
        return None if self._x_prev is None else self._x_prev.copy()


class EMAFilter:
    """Simple exponential moving average with weight-scaled adaptation."""

    def __init__(self, dim: int, alpha: float = 0.12) -> None:
        self.dim = int(dim)
        self.alpha = float(alpha)
        self._value: Optional[np.ndarray] = None

    @property
    def initialized(self) -> bool:
        """Whether the filter holds a previous state."""
        return self._value is not None

    def reset(self, x: np.ndarray) -> np.ndarray:
        """Restart the filter at value ``x``."""
        self._value = np.asarray(x, dtype=np.float64).reshape(self.dim).copy()
        return self._value.copy()

    def update(self, x: np.ndarray, weight: float = 1.0) -> np.ndarray:
        """Blend ``x`` into the EMA using ``weight * alpha`` (weight in [0, 2])."""
        xt = np.asarray(x, dtype=np.float64).reshape(self.dim)
        if self._value is None:
            return self.reset(xt)
        a = float(np.clip(self.alpha * float(weight), 0.0, 1.0))
        self._value = self._value + a * (xt - self._value)
        return self._value.copy()

    def value(self) -> Optional[np.ndarray]:
        """Current EMA value (None before the first sample)."""
        return None if self._value is None else self._value.copy()


# ---------------------------------------------------------------------------
# Mesh rasterizer: projected 3DMM mesh -> binary theoretical-face mask
# ---------------------------------------------------------------------------

class MeshRasterizer:
    """Rasterizes the projected (back-face-culled) facial mesh into a binary mask.

    Two modes are supported:

    * ``chain`` (default): interior triangle edges cancel pairwise, the remaining
      silhouette/fold boundary edges are chained into polygons and filled with a
      single ``cv2.fillPoly`` call (about 50x faster than per-triangle filling,
      and produces the exact same region as the triangle union).
    * ``triangles``: fills every front-facing triangle (slow but trivially robust).

    ``auto`` starts in ``chain`` mode and falls back to ``triangles`` whenever the
    chaining becomes suspicious.  When the fork exposes no triangle topology a
    landmark-convex-hull fallback is used and a loud warning is logged (this is
    NOT the intended production path; the official repo provides ``tddfa.tri``).
    """

    def __init__(
        self,
        tri: Optional[np.ndarray],
        mean_template_flat: Optional[np.ndarray],
        logger: logging.Logger = LOG,
        mode: str = "auto",
    ) -> None:
        self.logger = logger
        self.mode = mode  # auto | chain | triangles | hull
        self.tri: Optional[np.ndarray] = None
        self._front_sign = 1.0
        self._sign_locked = False
        self._vote = 0.0
        self._hull_warned = False
        if tri is not None:
            t = np.asarray(tri)
            if t.ndim == 2 and t.shape[0] == 3:
                pass
            elif t.ndim == 2 and t.shape[1] == 3:
                t = t.T
            else:
                self.logger.warning("unexpected triangle topology shape %s; rasterizer runs in hull mode", t.shape)
                t = None
            if t is not None:
                self.tri = np.ascontiguousarray(t.astype(np.int64))
        if self.tri is not None and mean_template_flat is not None:
            template = np.asarray(mean_template_flat, dtype=np.float64)
            n_tpl = template.size // 3
            if template.size % 3 or n_tpl <= 0 or int(self.tri.max()) >= n_tpl:
                self.logger.warning(
                    "triangle topology (%d tris, max vertex id %d) does not match the mean "
                    "template (%d vertices); rasterizer runs in hull mode",
                    self.tri.shape[1], int(self.tri.max()), n_tpl,
                )
                self.tri = None
            else:
                self._front_sign = self._determine_front_sign(template)
                self._sign_locked = True
        if self.tri is None:
            self.logger.warning(
                "no triangle topology available from the 3DDFA fork; theoretical face mask "
                "falls back to the 68-landmark convex hull (degraded quality, task spec warns about this)"
            )

    # -- front/back face classification ------------------------------------

    def _determine_front_sign(self, template_flat: np.ndarray) -> float:
        """Sign such that ``signed_area * sign > 0`` selects camera-facing triangles.

        Computed once on the mean BFM template (frontal by construction) with the
        same y-flip that ``similar_transform`` applies, so the winding convention
        matches the projected image coordinates at run time.
        """
        assert self.tri is not None
        n_verts = template_flat.size // 3
        mean3 = template_flat.reshape(3, n_verts, order="F")
        x = mean3[0]
        y = -mean3[1]  # image y axis points down (similar_transform flips y)
        t0, t1, t2 = self.tri[0], self.tri[1], self.tri[2]
        area = (x[t1] - x[t0]) * (y[t2] - y[t0]) - (y[t1] - y[t0]) * (x[t2] - x[t0])
        total = float(area.sum())
        sign = 1.0 if total >= 0.0 else -1.0
        self.logger.info(
            "rasterizer: front-face winding sign=%+.0f (mean-template signed area=%.1f, %d triangles)",
            sign, total, self.tri.shape[1],
        )
        return sign

    @property
    def front_sign(self) -> float:
        """Current front-face winding sign."""
        return self._front_sign

    def _update_sign_vote(self, area: np.ndarray) -> None:
        """Fork without BFM template: vote on the winding sign over early frames."""
        if self._sign_locked:
            return
        self._vote += float(np.sum(area))
        if abs(self._vote) > 1e3:
            self._front_sign = 1.0 if self._vote >= 0.0 else -1.0
            self._sign_locked = True

    # -- public API ---------------------------------------------------------

    def rasterize(
        self,
        dense_verts: Optional[np.ndarray],
        verts68: np.ndarray,
        out_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
        """Render the theoretical face mask into a full-frame uint8 buffer.

        Parameters
        ----------
        dense_verts: (3, N) projected mesh in image pixel coordinates (``None``
            when the fork cannot produce dense vertices).
        verts68: (3, 68) projected landmarks, used for the hull fallback.
        out_hw: ``(height, width)`` of the original frame.

        Returns
        -------
        mask: (H, W) uint8 with values in {0, 255}.
        loops: silhouette polygons (full-image float coords) for contour output
            and debug drawing.
        info: dict with ``mode``, ``front_ratio``, ``loops`` counters.
        """
        height, width = int(out_hw[0]), int(out_hw[1])
        info: Dict[str, Any] = {"mode": "none", "front_ratio": 0.0, "loops": 0}
        full = np.zeros((height, width), dtype=np.uint8)

        use_tri = self.tri is not None and dense_verts is not None and self.mode != "hull"
        if use_tri:
            dense = np.asarray(dense_verts, dtype=np.float64)
            if dense.ndim != 2 or dense.shape[0] != 3 or not is_finite_array(dense):
                use_tri = False
            elif self.tri.shape[1] == 0 or int(self.tri.max()) >= dense.shape[1]:
                self.logger.warning("triangle topology does not match dense vertex count; hull fallback")
                use_tri = False

        if not use_tri:
            return self._rasterize_hull(verts68, out_hw, info)

        dense = np.asarray(dense_verts, dtype=np.float64)
        x, y = dense[0], dense[1]
        t0, t1, t2 = self.tri[0], self.tri[1], self.tri[2]
        area = (x[t1] - x[t0]) * (y[t2] - y[t0]) - (y[t1] - y[t0]) * (x[t2] - x[t0])
        if not self._sign_locked:
            self._update_sign_vote(area)
        front = area * self._front_sign > 0.0
        n_front = int(front.sum())
        info["front_ratio"] = float(n_front) / float(max(area.size, 1))
        if n_front < 4:
            self.logger.debug("rasterizer: too few front-facing triangles (%d)", n_front)
            return full, [], info

        front_tris = np.stack([t0[front], t1[front], t2[front]], axis=1)  # (F, 3)

        # Local window around the front-facing vertices (keep the frame memory bounded).
        fx = np.concatenate([x[front_tris[:, 0]], x[front_tris[:, 1]], x[front_tris[:, 2]]])
        fy = np.concatenate([y[front_tris[:, 0]], y[front_tris[:, 1]], y[front_tris[:, 2]]])
        margin = 3.0
        wx0 = int(max(0, math.floor(float(fx.min()) - margin)))
        wy0 = int(max(0, math.floor(float(fy.min()) - margin)))
        wx1 = int(min(width, math.ceil(float(fx.max()) + margin)))
        wy1 = int(min(height, math.ceil(float(fy.max()) + margin)))
        if wx1 - wx0 < 2 or wy1 - wy0 < 2:
            return full, [], info
        win_w, win_h = wx1 - wx0, wy1 - wy0

        mode = self.mode if self.mode in ("chain", "triangles") else "chain"
        canvas = None
        loops_local: List[np.ndarray] = []
        if mode == "chain":
            canvas, loops_local = self._fill_chain(front_tris, x, y, (wx0, wy0), (win_w, win_h), info)

        if canvas is None:  # fallback or explicit triangles mode
            mode = "triangles"
            canvas, loops_local = self._fill_triangles(front_tris, x, y, (wx0, wy0), (win_w, win_h))

        full[wy0:wy1, wx0:wx1] = canvas
        info["mode"] = mode
        info["loops"] = len(loops_local)
        loops_full = [loop + np.array([wx0, wy0], dtype=np.float32) for loop in loops_local]
        return full, loops_full, info

    # -- implementations ----------------------------------------------------

    def _fill_chain(
        self,
        front_tris: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        offset: Tuple[int, int],
        win: Tuple[int, int],
        info: Dict[str, Any],
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """Chain boundary edges of the front-facing patch into polygons and fill them."""
        try:
            loops = self._boundary_loops(front_tris)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("boundary chaining failed (%s); using triangle fill", exc)
            return None, []
        if loops is None or not loops or len(loops) > 64:
            return None, []
        ox, oy = float(offset[0]), float(offset[1])
        win_w, win_h = win
        polys: List[np.ndarray] = []
        loops_local: List[np.ndarray] = []
        for loop_verts in loops:
            if loop_verts.size < 3:
                continue
            pts = np.stack([x[loop_verts] - ox, y[loop_verts] - oy], axis=1)
            pts = np.clip(pts, 0.0, [win_w, win_h])
            poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            polys.append(poly)
            loops_local.append(pts.astype(np.float32))
        if not polys:
            return None, []
        canvas = np.zeros((win_h, win_w), dtype=np.uint8)
        cv2.fillPoly(canvas, polys, 255, lineType=cv2.LINE_8)
        info["chain_loops"] = len(polys)
        return canvas, loops_local

    def _boundary_loops(self, front_tris: np.ndarray) -> Optional[List[np.ndarray]]:
        """Return vertex-index loops of the front-facing patch boundary.

        Interior edges are shared by exactly two front-facing triangles and cancel;
        the remaining edges are the silhouette plus pose-dependent folds, which are
        chained into closed polygons.
        """
        n_tris = front_tris.shape[0]
        edges = np.concatenate(
            [front_tris[:, [0, 1]], front_tris[:, [1, 2]], front_tris[:, [2, 0]]], axis=0
        ).astype(np.int64)
        if float(front_tris.max()) < -1 or n_tris == 0:
            return None
        big = int(front_tris.max()) + 1
        if big <= 0:
            return None
        lo = np.minimum(edges[:, 0], edges[:, 1])
        hi = np.maximum(edges[:, 0], edges[:, 1])
        keys = lo * big + hi
        keys_sorted = np.sort(keys)
        if keys_sorted.size == 0:
            return None
        first = np.ones(keys_sorted.size, dtype=bool)
        first[1:] = keys_sorted[1:] != keys_sorted[:-1]
        last = np.ones(keys_sorted.size, dtype=bool)
        last[:-1] = keys_sorted[:-1] != keys_sorted[1:]
        boundary = first & last  # appears exactly once
        bkeys = keys_sorted[boundary]
        if bkeys.size == 0:
            return None
        if bkeys.size > 0.5 * keys.size + 16:  # pathological: refuse, let the caller fall back
            return None
        va = (bkeys // big).tolist()
        vb = (bkeys % big).tolist()
        adj: Dict[int, List[int]] = {}
        for a, b in zip(va, vb, strict=True):
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        used: set = set()
        loops: List[np.ndarray] = []
        max_steps = 4 * len(va) + 16
        steps = 0
        for start in list(adj.keys()):
            if not any((min(start, w), max(start, w)) not in used for w in adj[start]):
                continue
            path = [start]
            cur = start
            while True:
                nxt = -1
                for w in adj[cur]:
                    ek = (min(cur, w), max(cur, w))
                    if ek not in used:
                        nxt = w
                        break
                if nxt < 0:
                    break
                used.add((min(cur, nxt), max(cur, nxt)))
                path.append(nxt)
                cur = nxt
                steps += 1
                if steps > max_steps:
                    return None
                if cur == start:
                    break
            if len(path) >= 4:
                loops.append(np.asarray(path[:-1], dtype=np.int64))
        return loops or None

    def _fill_triangles(
        self,
        front_tris: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        offset: Tuple[int, int],
        win: Tuple[int, int],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Slow but bulletproof: fill every front-facing triangle."""
        ox, oy = float(offset[0]), float(offset[1])
        win_w, win_h = win
        pts = np.stack([x[front_tris] - ox, y[front_tris] - oy], axis=-1)  # (F, 3, 2)
        pts = np.clip(pts, 0.0, [win_w, win_h])
        q = np.ascontiguousarray(np.round(pts).astype(np.int32))
        canvas = np.zeros((win_h, win_w), dtype=np.uint8)
        cv2.fillPoly(canvas, q, 255, lineType=cv2.LINE_8)
        return canvas, []

    def _rasterize_hull(
        self, verts68: np.ndarray, out_hw: Tuple[int, int], info: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
        """Last-resort mask from the 68-landmark convex hull (degraded)."""
        height, width = int(out_hw[0]), int(out_hw[1])
        full = np.zeros((height, width), dtype=np.uint8)
        info["mode"] = "hull"
        v = np.asarray(verts68, dtype=np.float64)
        if v.ndim != 2 or v.shape[0] < 2 or v.shape[1] < 3 or not is_finite_array(v):
            return full, [], info
        pts = v[:2].T.astype(np.float32)
        if not self._hull_warned:
            self.logger.warning("theoretical mask uses the 68-landmark convex hull fallback (degraded mode)")
            self._hull_warned = True
        hull = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)
        q = np.clip(np.round(hull), 0, [width - 1, height - 1]).astype(np.int32)
        cv2.fillPoly(full, [q.reshape(-1, 1, 2)], 255, lineType=cv2.LINE_8)
        info["loops"] = 1
        return full, [hull], info


# ---------------------------------------------------------------------------
# 3DDFA_V2 adapter: every fork-specific integration detail lives here
# ---------------------------------------------------------------------------

def _mini_yaml(text: str) -> Dict[str, Any]:
    """Very small YAML subset parser (the official cfgs are flat ``key: value``)."""
    out: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        low = value.lower()
        if low in ("true", "false"):
            out[key] = low == "true"
            continue
        try:
            out[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            out[key] = float(value)
            continue
        except ValueError:
            pass
        out[key] = value.strip("'\"")
    return out


def load_config_file(path: Path) -> Dict[str, Any]:
    """Load a 3DDFA config (.yml/.yaml/.json) into a dict."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise FatalError(f"config file {path} must contain a mapping")
        return data
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise FatalError(f"config file {path} must contain a mapping")
        return dict(data)
    except ImportError:
        LOG.debug("PyYAML not installed; using the built-in mini parser for %s", path)
        return _mini_yaml(text)


class TDDFAAdapter:
    """Encapsulates all 3DDFA_V2 repository access.

    The adapter targets the official ``cleardusk/3DDFA_V2`` layout
    (``TDDFA.py`` + ``FaceBoxes`` + ``configs/*.yml``) and normalizes the small
    differences found in forks (detector return shapes, recon output containers,
    missing ``gpu_id`` keyword, missing triangle topology ...).  Swapping forks
    should not require touching any tracking/business logic.
    """

    CONFIG_CANDIDATES = ("configs/mb1_120x120.yml", "configs/mobile0.25_120x120.yml")

    def __init__(
        self,
        root: Path,
        config: Optional[Path],
        device: str,
        checkpoint: Optional[str] = None,
        onnx: bool = False,
        logger: logging.Logger = LOG,
    ) -> None:
        self.logger = logger
        self.root = Path(root).resolve()
        if not ((self.root / "TDDFA.py").is_file() or (self.root / "TDDFA_ONNX.py").is_file()):
            raise FatalError(
                f"{self.root} does not look like a 3DDFA_V2 repository (TDDFA.py not found). "
                "Pass --tddfa-root pointing at the cloned 3DDFA_V2 directory."
            )
        cfg_path = self._resolve_config(config)
        cfg = load_config_file(cfg_path)
        cfg = self._absolutize_cfg_paths(cfg)
        if checkpoint:
            cfg["checkpoint_fp"] = str(Path(checkpoint).expanduser().resolve())
        self._validate_weight_files(cfg)
        self.device = self._normalize_device(device)
        gpu_mode = self.device.startswith("cuda")
        gpu_id = 0
        if gpu_mode and ":" in self.device:
            try:
                gpu_id = int(self.device.split(":", 1)[1])
            except ValueError:
                gpu_id = 0

        root_str = str(self.root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        import torch  # deferred: allows --help and unit tests without torch

        self._torch = torch
        if gpu_mode and not torch.cuda.is_available():
            self.logger.warning("CUDA requested (%s) but not available; falling back to CPU", self.device)
            self.device = "cpu"
            gpu_mode = False
        torch.set_grad_enabled(False)

        if onnx:
            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX as FaceBoxesCls  # type: ignore
            from TDDFA_ONNX import TDDFA_ONNX as TDDFACls  # type: ignore
        else:
            from FaceBoxes import FaceBoxes as FaceBoxesCls  # type: ignore
            from TDDFA import TDDFA as TDDFACls  # type: ignore

        # Init the 3DDFA model exactly once (never per frame).
        try:
            self.tddfa = TDDFACls(gpu_mode=gpu_mode, gpu_id=gpu_id, **cfg)
        except TypeError:
            # some forks have no gpu_id kwarg
            self.tddfa = TDDFACls(gpu_mode=gpu_mode, **cfg)
        self.face_boxes = FaceBoxesCls()
        self.size = int(cfg.get("size", 120))
        tri = getattr(self.tddfa, "tri", None)
        self.tri = None if tri is None else np.asarray(tri)
        self.onnx = bool(onnx)
        self.logger.info(
            "3DDFA_V2 loaded once: root=%s config=%s device=%s size=%d dense_tri=%s",
            self.root, cfg_path.name, self.device, self.size,
            "yes" if self.tri is not None else "no",
        )

    # -- setup helpers -------------------------------------------------------

    @staticmethod
    def discover_root(hint: Optional[str] = None) -> Optional[Path]:
        """Find the 3DDFA_V2 repository from CLI hint, env vars or common locations."""
        candidates: List[Path] = []
        if hint:
            candidates.append(Path(hint).expanduser())
        for env_name in ("TDDFA_ROOT", "3DDFA_ROOT", "3DDFA_V2_ROOT"):
            env_val = os.environ.get(env_name)
            if env_val:
                candidates.append(Path(env_val).expanduser())
        here = Path(__file__).resolve().parent
        cwd = Path.cwd()
        for base in (cwd, here, here.parent, cwd.parent):
            candidates.extend(
                [base / "3DDFA_V2", base / "3ddfa_v2", base / "third_party" / "3DDFA_V2"]
            )
        seen = set()
        for cand in candidates:
            key = str(cand).lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                if (cand / "TDDFA.py").is_file() or (cand / "TDDFA_ONNX.py").is_file():
                    return cand.resolve()
            except OSError:
                continue
        return None

    def _resolve_config(self, config: Optional[Path]) -> Path:
        if config is not None:
            cfg = Path(config).expanduser().resolve()
            if not cfg.is_file():
                raise FatalError(f"3DDFA config not found: {cfg}")
            return cfg
        for rel in self.CONFIG_CANDIDATES:
            cand = self.root / rel
            if cand.is_file():
                return cand
        raise FatalError(
            f"no 3DDFA config found under {self.root / 'configs'}; pass --config explicitly"
        )

    def _absolutize_cfg_paths(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """The official cfg stores repo-relative paths; make them CWD independent."""
        out = dict(cfg)
        for key in ("checkpoint_fp", "bfm_fp", "param_mean_std_fp"):
            val = out.get(key)
            if isinstance(val, str) and val and not os.path.isabs(val):
                out[key] = str((self.root / val).resolve())
        return out

    def _validate_weight_files(self, cfg: Dict[str, Any]) -> None:
        missing = []
        for key in ("checkpoint_fp", "bfm_fp"):
            val = cfg.get(key)
            if isinstance(val, str) and val and not Path(val).is_file():
                missing.append(f"{key}={val}")
        if missing:
            raise FatalError(
                "3DDFA weight files are missing:\n  " + "\n  ".join(missing) +
                "\nDownload the official weights (see 3DDFA_V2 README: init.sh / weights/, "
                "configs/bfm_noneck_v3.pkl) or pass --checkpoint."
            )

    def _normalize_device(self, device: str) -> str:
        d = str(device).strip().lower()
        if d in ("cpu",):
            return "cpu"
        if d.startswith("cuda"):
            return d if ":" in d else "cuda:0"
        if d in ("gpu",):
            return "cuda:0"
        self.logger.warning("unsupported device %r; using cpu", device)
        return "cpu"

    # -- detector ------------------------------------------------------------

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run the face detector; returns an (N, 5) float array (x1, y1, x2, y2, score)."""
        raw = self.face_boxes(img_bgr)
        det = self._normalize_detections(raw)
        if det.shape[0] > 1:
            order = np.argsort(-det[:, 4], kind="stable")
            det = det[order]
        return det

    @staticmethod
    def _normalize_detections(raw: Any) -> np.ndarray:
        boxes: Any = raw
        scores: Optional[np.ndarray] = None
        if isinstance(raw, tuple) and len(raw) == 2:
            boxes, scores = raw[0], np.asarray(raw[1], dtype=np.float64).reshape(-1)
        arr = np.asarray(boxes, dtype=np.float64)
        if arr.size == 0:
            return np.zeros((0, 5), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 4:
            return np.zeros((0, 5), dtype=np.float64)
        if arr.shape[1] == 4:
            extra = scores if scores is not None else np.ones(arr.shape[0], dtype=np.float64)
            extra = np.asarray(extra).reshape(-1)
            if extra.shape[0] != arr.shape[0]:
                extra = np.ones(arr.shape[0], dtype=np.float64)
            arr = np.concatenate([arr, extra.reshape(-1, 1)], axis=1)
        return arr[:, :5].copy()

    # -- 3DMM fitting / reconstruction ---------------------------------------

    def fit(self, img_bgr: np.ndarray, bbox: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Fit 3DMM to the face defined by ``bbox``; returns (param62, roi_box4)."""
        box = [float(v) for v in np.asarray(bbox, dtype=np.float64).reshape(-1)[:4]]
        param_lst, roi_lst = self.tddfa(img_bgr, [box])
        if not param_lst or not roi_lst:
            raise RuntimeError("3DDFA returned no parameters")
        param = np.asarray(param_lst[0], dtype=np.float32).reshape(-1)
        roi = np.asarray(roi_lst[0], dtype=np.float32).reshape(4)
        if param.size < 62 or not is_finite_array(param) or not is_finite_array(roi):
            raise RuntimeError("3DDFA returned degenerate parameters")
        return param, roi

    def recon_dense(self, param: np.ndarray, roi_box: np.ndarray) -> np.ndarray:
        """Full 3DMM mesh projected to image pixels as (3, N) float32."""
        return self._recon(param, roi_box, dense=True)

    def recon_sparse(self, param: np.ndarray, roi_box: np.ndarray) -> np.ndarray:
        """68 projected landmarks as (3, 68) float32 in image pixels."""
        return self._recon(param, roi_box, dense=False)

    def _recon(self, param: np.ndarray, roi_box: np.ndarray, dense: bool) -> np.ndarray:
        try:
            ver_lst = self.tddfa.recon_vers([np.asarray(param, np.float32)], [np.asarray(roi_box, np.float32)],
                                            dense_flag=bool(dense))
            verts = np.asarray(ver_lst[0], dtype=np.float32)
            if verts.ndim == 2 and verts.shape[0] == 3 and verts.shape[1] > 0 and is_finite_array(verts):
                return verts
            raise ValueError(f"unexpected recon shape {verts.shape}")
        except Exception as exc:
            self.logger.debug("recon_vers failed (%s); trying the manual BFM path", exc)
            return self._manual_recon(param, roi_box, dense)

    def _manual_recon(self, param: np.ndarray, roi_box: np.ndarray, dense: bool) -> np.ndarray:
        """Fork-proof reconstruction directly from the BFM arrays (official maths)."""
        bfm = getattr(self.tddfa, "bfm", None)
        if bfm is None:
            raise FatalError("the 3DDFA fork exposes neither recon_vers nor bfm arrays")
        rot, offset, alpha_shp, alpha_exp = parse_param(param)

        def _mat(name: str) -> np.ndarray:
            arr = np.asarray(getattr(bfm, name), dtype=np.float64)
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr

        if dense:
            base = _mat("u") + _mat("w_shp") @ alpha_shp + _mat("w_exp") @ alpha_exp
        else:
            base = _mat("u_base") + _mat("w_shp_base") @ alpha_shp + _mat("w_exp_base") @ alpha_exp
        pts3d = rot @ base.reshape(3, -1, order="F") + offset
        return similar_transform(pts3d.astype(np.float32), roi_box, self.size)

    def mean_template_flat(self) -> Optional[np.ndarray]:
        """Flattened mean BFM template vertices (used once for the winding sign)."""
        bfm = getattr(self.tddfa, "bfm", None)
        u = getattr(bfm, "u", None) if bfm is not None else None
        if u is None:
            return None
        return np.asarray(u, dtype=np.float64).reshape(-1)

    def close(self) -> None:
        """Release the models and empty the CUDA cache."""
        try:
            del self.tddfa  # noqa: F821 - attribute set in __init__
            del self.face_boxes
        except Exception:
            pass
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


# ---------------------------------------------------------------------------
# Reference geometry baseline (identity / shape regularization)
# ---------------------------------------------------------------------------

@dataclass
class ReferenceBaseline:
    """Stable identity/shape statistics derived from the unmasked reference faces."""

    shape_median: Optional[np.ndarray] = None          # (40,)
    shape_samples: Optional[np.ndarray] = None         # (K, 40)
    exp_median: Optional[np.ndarray] = None            # (10,)
    params: Optional[np.ndarray] = None                # (K, 62) raw reference params
    landmarks: Optional[np.ndarray] = None             # (K, 68, 2)
    scores: Optional[np.ndarray] = None                # (K,)
    names: Optional[np.ndarray] = None                 # (K,)
    face_size_median: float = 0.0
    face_ratio_median: float = 0.0                     # face size / image diagonal
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """Whether a usable shape median exists."""
        return self.shape_median is not None and self.shape_median.size == 40


class ReferenceGeometryBuilder:
    """Builds / caches the reference face shape baseline (``reference_baseline.npz``)."""

    REF_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, adapter: TDDFAAdapter, logger: logging.Logger = LOG) -> None:
        self.adapter = adapter
        self.logger = logger

    @classmethod
    def _reference_files(cls, ref_dir: Path) -> List[Path]:
        """Reference images consumed by :meth:`build`, in deterministic order."""
        exts = set(cls.REF_IMAGE_EXTS)
        return sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in exts and p.is_file())

    # Reference dirs are small (a handful of images); hashing them is cheap
    # and catches content changes that preserve size *and* mtime (rsync/cp -p,
    # archive extraction, network sync), which a size+mtime fingerprint misses.
    REF_HASH_MAX_BYTES = 64 * 1024 * 1024

    @classmethod
    def _file_digest(cls, fp: Path) -> str:
        """sha256 of one reference file (hex), or '' when unreadable/too large."""
        try:
            st = fp.stat()
            if st.st_size > cls.REF_HASH_MAX_BYTES:
                return ""
            h = hashlib.sha256()
            with open(fp, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return ""

    @classmethod
    def reference_identity(cls, ref_dir: Path) -> str:
        """Fingerprint of the reference file set: name + size + mtime (ns) + sha256.

        Stored inside the cached baseline, so a changed reference directory (new,
        removed or modified images) triggers an automatic rebuild instead of
        silently reusing stale identity statistics.
        """
        entries: List[List[Any]] = []
        for fp in cls._reference_files(ref_dir):
            try:
                st = fp.stat()
                size, mtime = int(st.st_size), int(st.st_mtime_ns)
            except OSError:
                size, mtime = -1, -1
            entries.append([fp.name, size, mtime, cls._file_digest(fp)])
        return json.dumps(entries, ensure_ascii=False, separators=(",", ":"))

    def load_or_build(
        self,
        ref_dir: Optional[Path],
        cache_path: Path,
        rebuild: bool = False,
    ) -> Optional[ReferenceBaseline]:
        """Return the baseline from cache when possible, else build and cache it.

        The cache stores a fingerprint of the reference images; when the reference
        directory changed since the cache was written the baseline is rebuilt
        automatically (``--rebuild-reference`` forces this unconditionally).
        """
        if ref_dir is None:
            self.logger.warning("no --reference-dir given: shape runs in EMA-only mode (no identity baseline)")
            return None
        ref_dir = Path(ref_dir)
        if not ref_dir.is_dir():
            self.logger.warning("reference dir does not exist: %s (shape runs in EMA-only mode)", ref_dir)
            return None
        identity = self.reference_identity(ref_dir)
        if cache_path.is_file() and not rebuild:
            baseline = self.load(cache_path)
            if baseline is not None:
                cached_identity = baseline.meta.get("ref_identity")
                if cached_identity == identity:
                    self.logger.info("reference baseline loaded from cache: %s", cache_path)
                    return baseline
                if cached_identity is None:
                    self.logger.warning("reference baseline cache has no identity fingerprint; rebuilding")
                else:
                    self.logger.warning("reference images changed since the cache was built; rebuilding")
            else:
                self.logger.warning("reference baseline cache invalid; rebuilding")
        baseline = self.build(ref_dir)
        if baseline is not None:
            self.save(cache_path, baseline, ref_dir)
        return baseline

    def build(self, ref_dir: Path) -> Optional[ReferenceBaseline]:
        """Run the detector + 3DDFA on every reference image and robustly combine shapes."""
        files = self._reference_files(ref_dir)
        if not files:
            self.logger.warning("no reference images found in %s", ref_dir)
            return None
        names: List[str] = []
        params: List[np.ndarray] = []
        shapes: List[np.ndarray] = []
        exps: List[np.ndarray] = []
        lms: List[np.ndarray] = []
        scores: List[float] = []
        sizes: List[float] = []
        ratios: List[float] = []
        for fp in files:
            try:
                img = imread_bgr(fp)
                if img is None:
                    self.logger.warning("reference image unreadable: %s", fp.name)
                    continue
                h, w = img.shape[:2]
                det = self.adapter.detect(img)
                if det.shape[0] == 0:
                    self.logger.warning("no face detected in reference %s", fp.name)
                    continue
                areas = (det[:, 3] - det[:, 1]) * (det[:, 2] - det[:, 0])
                best = det[int(np.argmax(areas))]
                if float(best[4]) < 0.5:
                    self.logger.warning("weak detection (%.2f) in reference %s", float(best[4]), fp.name)
                    continue
                param, roi = self.adapter.fit(img, best[:4])
                verts68 = self.adapter.recon_sparse(param, roi)
                yaw, pitch, _ = pose_angles_deg(param)
                if abs(yaw) > 45.0 or abs(pitch) > 35.0:
                    self.logger.warning("reference %s too non-frontal (yaw=%.1f pitch=%.1f); skipped",
                                        fp.name, yaw, pitch)
                    continue
                centroid, face_size, _ = face_metrics(verts68)
                if face_size < 60.0:
                    self.logger.warning("reference %s face too small (%.0fpx); skipped", fp.name, face_size)
                    continue
                names.append(fp.name)
                params.append(np.asarray(param, np.float64))
                shapes.append(np.asarray(param[12:52], np.float64))
                exps.append(np.asarray(param[52:62], np.float64))
                lms.append(np.asarray(verts68[:2], np.float64).T)  # (68, 2)
                scores.append(float(best[4]))
                sizes.append(face_size)
                ratios.append(face_size / max(math.hypot(w, h), 1.0))
            except Exception as exc:
                self.logger.warning("reference %s failed: %s", fp.name, exc)
                continue
        if not shapes:
            self.logger.warning("no usable reference faces; shape runs in EMA-only mode")
            return None
        shape_mat = np.stack(shapes, axis=0)  # (K, 40)
        keep = self._trim_outliers(shape_mat)
        kept = int(keep.sum())
        if kept < 1:
            keep = np.ones(shape_mat.shape[0], dtype=bool)
            kept = shape_mat.shape[0]
        self.logger.info("reference baseline: %d/%d images kept", kept, len(files))
        shape_kept = shape_mat[keep]
        shape_median = np.median(shape_kept, axis=0)
        kept_idx = np.where(keep)[0].tolist()
        baseline = ReferenceBaseline(
            shape_median=shape_median.astype(np.float64),
            shape_samples=shape_kept.astype(np.float64),
            exp_median=np.median(np.stack(exps, 0)[keep], axis=0),
            params=np.stack(params, 0)[keep],
            landmarks=np.stack(lms, 0)[keep],
            scores=np.asarray(scores, np.float64)[keep],
            names=np.asarray(names, dtype=object)[keep],
            face_size_median=float(np.median(np.asarray(sizes)[keep])),
            face_ratio_median=float(np.median(np.asarray(ratios)[keep])),
            meta={
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "ref_dir": str(ref_dir),
                "ref_identity": self.reference_identity(ref_dir),
                "n_files": len(files),
                "n_kept": kept,
                "kept_indices": kept_idx,
                "all_names": list(names),
                "module_version": __version__,
            },
        )
        if kept < 3:
            self.logger.warning("only %d usable reference(s): the shape baseline is weak", kept)
        return baseline

    @staticmethod
    def _trim_outliers(shape_mat: np.ndarray, z: float = 3.5) -> np.ndarray:
        """Robust median/MAD based outlier rejection in shape space."""
        if shape_mat.shape[0] <= 2:
            return np.ones(shape_mat.shape[0], dtype=bool)
        med = np.median(shape_mat, axis=0)
        dist = np.linalg.norm(shape_mat - med, axis=1)
        mad = float(np.median(np.abs(dist - np.median(dist))))
        limit = float(np.median(dist)) + 1.4826 * mad * z
        keep = dist <= max(limit, np.median(dist))
        if keep.sum() < 2:
            order = np.argsort(dist)
            keep = np.zeros(shape_mat.shape[0], dtype=bool)
            keep[order[: max(2, min(3, shape_mat.shape[0]))]] = True
        return keep

    def save(self, cache_path: Path, baseline: ReferenceBaseline, ref_dir: Path) -> None:
        """Cache the baseline as a .npz next to the video outputs."""
        ensure_dir(cache_path.parent)
        tmp = cache_path.with_name(cache_path.name + ".tmp.npz")
        np.savez(
            tmp,
            shape_params=baseline.shape_median,
            median_shape=baseline.shape_median,
            shape_samples=baseline.shape_samples,
            exp_median=baseline.exp_median,
            params=baseline.params,
            reference_landmarks=baseline.landmarks,
            reference_face_ratio=np.asarray([baseline.face_ratio_median]),
            face_size_median=np.asarray([baseline.face_size_median]),
            scores=baseline.scores,
            names=np.asarray(baseline.names, dtype=object),
            metadata=np.asarray([json.dumps(baseline.meta, ensure_ascii=False)]),
        )
        os.replace(str(tmp), str(cache_path))
        self.logger.info("reference baseline cached: %s", cache_path)

    def load(self, cache_path: Path) -> Optional[ReferenceBaseline]:
        """Load a cached baseline; returns None when the file is unusable."""
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                shape = np.asarray(data["shape_params"], dtype=np.float64).reshape(-1)
                if shape.size != 40 or not is_finite_array(shape):
                    return None
                meta_raw = data["metadata"] if "metadata" in data else None
                meta = {}
                if meta_raw is not None and np.asarray(meta_raw).size:
                    try:
                        meta = json.loads(str(np.asarray(meta_raw).reshape(-1)[0]))
                    except (ValueError, TypeError):
                        meta = {}
                return ReferenceBaseline(
                    shape_median=shape,
                    shape_samples=self._opt(data, "shape_samples"),
                    exp_median=self._opt(data, "exp_median"),
                    params=self._opt(data, "params"),
                    landmarks=self._opt(data, "reference_landmarks"),
                    scores=self._opt(data, "scores"),
                    names=self._opt(data, "names"),
                    face_size_median=self._scalar(data, "face_size_median"),
                    face_ratio_median=self._scalar(data, "reference_face_ratio"),
                    meta=meta,
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.logger.warning("failed to load baseline cache %s: %s", cache_path, exc)
            return None

    @staticmethod
    def _opt(data: "np.lib.npyio.NpzFile", key: str) -> Optional[np.ndarray]:
        return np.asarray(data[key]) if key in data else None

    @staticmethod
    def _scalar(data: "np.lib.npyio.NpzFile", key: str) -> float:
        if key not in data:
            return 0.0
        arr = np.asarray(data[key]).reshape(-1)
        return float(arr[0]) if arr.size else 0.0


# ---------------------------------------------------------------------------
# Tracking: measurement selection, quality scoring, fallback state machine
# ---------------------------------------------------------------------------

@dataclass
class QualityConfig:
    """All acceptance thresholds (task spec: thresholds must be configurable)."""

    confidence_threshold: float = 0.45
    occ_mid: float = 0.20
    occ_severe: float = 0.45
    occ_hard: float = 0.85
    max_pose_jump_deg: float = 70.0
    min_scale_ratio: float = 0.65
    max_scale_ratio: float = 1.55
    max_shift_ratio: float = 0.40
    max_landmark_jump_ratio: float = 0.45
    blur_threshold: float = BLUR_FULL_CREDIT
    out_of_frame_tolerance: float = 0.35
    min_mask_area: int = MIN_MASK_AREA


@dataclass
class Measurement:
    """A face detection accepted as the tracked person for this frame."""

    bbox: np.ndarray              # (4,) x1,y1,x2,y2
    score: float
    iou: Optional[float]          # IoU with the previous tracked bbox (None if no history)
    assoc: float                  # association score used for selection


@dataclass
class QualityScores:
    """Composed confidence for one frame (spec section: quality score)."""

    confidence: float
    components: Dict[str, float]
    d_pose: float
    scale_ratio: float
    shift_ratio: float
    lm_jump_ratio: float
    hard_reject: Optional[str] = None


@dataclass
class AcceptedState:
    """Everything needed to keep tracking and to fall back after a failure."""

    frame_id: int
    param: np.ndarray             # (62,) smoothed 3DMM parameters (crop space)
    roi_box: np.ndarray           # (4,)
    verts68: np.ndarray           # (3, 68) projected landmarks (image px)
    pose_deg: Tuple[float, float, float]
    centroid: np.ndarray          # (2,)
    face_size: float
    lm_bbox: Tuple[float, float, float, float]
    mask_area: int
    track_bbox: np.ndarray        # (4,)
    det_score: float
    quality: float
    v_param: Optional[np.ndarray] = None  # per-frame velocity in param space
    scale_baseline_px: Optional[float] = None


class FaceTracker:
    """Maintains the primary face track: detection scheduling, bbox filtering and scoring."""

    def __init__(self, qcfg: QualityConfig, logger: logging.Logger = LOG) -> None:
        self.qcfg = qcfg
        self.logger = logger
        self.bbox_filter = OneEuroFilter(4, min_cutoff=1.5, beta=0.4)
        self.frames_since_detect = 10 ** 9
        self.lost_count = 0
        self.low_quality_streak = 0
        self.det_score_carry = 0.0
        self.detector_calls = 0
        self.track_bbox: Optional[np.ndarray] = None      # last filtered bbox
        self.bbox_velocity = np.zeros(4, dtype=np.float64)

    # -- detection scheduling ------------------------------------------------

    def needs_detection(self, detect_interval: int, trigger_streak: int) -> bool:
        """Whether the frame should run the face detector."""
        if self.track_bbox is None:
            return True
        if self.lost_count > 0:
            return True
        if self.frames_since_detect >= max(1, detect_interval):
            return True
        if self.low_quality_streak >= max(1, trigger_streak):
            return True
        # detection is skipped this frame: count it (the caller resets the
        # counter to 0 whenever a detection actually runs)
        self.frames_since_detect += 1
        return False

    # -- measurement selection (spec: multi-face guard) ------------------------

    def select_measurement(self, dets: np.ndarray) -> Optional[Measurement]:
        """Choose the candidate belonging to the tracked person (not simply top score)."""
        if dets is None or dets.shape[0] == 0:
            return None
        if self.track_bbox is None:
            best = int(np.argmax(dets[:, 4]))
            return Measurement(dets[best, :4].astype(np.float64), float(dets[best, 4]), None, float(dets[best, 4]))
        prev = self.track_bbox
        prev_size = max(0.5 * ((prev[2] - prev[0]) + (prev[3] - prev[1])), 1.0)
        prev_center = np.array([(prev[0] + prev[2]) / 2.0, (prev[1] + prev[3]) / 2.0])
        best_idx, best_assoc, best_iou = -1, -1.0, None
        for i in range(dets.shape[0]):
            box = dets[i, :4].astype(np.float64)
            score = float(dets[i, 4])
            iou = bbox_iou(box, prev)
            center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
            cdist = float(np.linalg.norm(center - prev_center)) / prev_size
            size = max(0.5 * ((box[2] - box[0]) + (box[3] - box[1])), 1.0)
            size_err = abs(math.log(size / prev_size))
            assoc = 1.5 * iou + 0.8 * math.exp(-cdist) + 0.4 * math.exp(-3.0 * size_err) + 0.3 * score
            if assoc > best_assoc:
                best_idx, best_assoc, best_iou = i, assoc, iou
        if best_idx < 0:
            return None
        if best_assoc < ASSOC_MIN_SCORE and self.lost_count <= 45:
            # detector latched onto something else entirely - keep the main track
            self.logger.debug("ignoring weak detection association (assoc=%.2f)", best_assoc)
            return None
        box = dets[best_idx, :4].astype(np.float64)
        return Measurement(box, float(dets[best_idx, 4]), float(best_iou or 0.0), float(best_assoc))

    # -- bbox filter ---------------------------------------------------------

    def update_bbox(self, meas: Measurement, t: float, reinit: bool = False) -> np.ndarray:
        """Blend a fresh detection into the tracked bbox (or re-initialize it)."""
        if reinit or self.track_bbox is None:
            filtered = self.bbox_filter.reset(meas.bbox, t)
            self.bbox_velocity = np.zeros(4, dtype=np.float64)
        else:
            previous = self.track_bbox
            filtered = self.bbox_filter.filter(meas.bbox, t)
            self.bbox_velocity = np.clip(filtered - previous, -40.0, 40.0)
        self.track_bbox = filtered.astype(np.float64)
        self.frames_since_detect = 0
        return self.track_bbox

    def filtered_bbox(self) -> Optional[np.ndarray]:
        """Current smoothed bbox used for fitting (None while no track exists)."""
        return None if self.track_bbox is None else self.track_bbox.copy()

    def predicted_bbox(self, max_extrapolation: float = 3.0) -> Optional[np.ndarray]:
        """Damped constant-velocity bbox extrapolation (for detection scoring while lost)."""
        if self.track_bbox is None:
            return None
        k = min(float(self.lost_count - 1), max_extrapolation * 5.0)
        if k <= 0:
            return self.track_bbox.copy()
        damp_sum = sum(0.9 ** i for i in range(1, int(k) + 1))
        return (self.track_bbox + self.bbox_velocity * damp_sum).astype(np.float64)

    def decay_score(self) -> None:
        """Carry the last detector score forward with a decay."""
        self.det_score_carry = max(self.det_score_carry * DET_SCORE_DECAY, 0.55)

    # -- quality scoring (spec: compose quality from many cues) ---------------

    def evaluate(
        self,
        param: np.ndarray,
        verts68: np.ndarray,
        meas_score: Optional[float],
        meas_iou: Optional[float],
        blur: float,
        occ_ratio: Optional[float],
        out_ratio: float,
        mask_area_est: float,
        prev: Optional[AcceptedState],
        reinit: bool,
    ) -> QualityScores:
        """Compose a 0..1 confidence plus individual components for one candidate frame."""
        qcfg = self.qcfg
        det_conf = float(meas_score) if meas_score is not None else float(self.det_score_carry)
        q_det = clamp01((det_conf - 0.5) / 0.45)
        q_blur = clamp01(0.35 + 0.65 * clamp01(blur / max(qcfg.blur_threshold, 1e-6)))
        q_oof = max(0.15, clamp01(1.0 - (out_ratio - qcfg.out_of_frame_tolerance) / 0.5))
        if occ_ratio is None:
            q_occ = 1.0
        else:
            q_occ = clamp01(1.0 - 0.85 * (occ_ratio - qcfg.occ_mid) / max(qcfg.occ_hard - qcfg.occ_mid, 1e-6))

        d_pose = 0.0
        scale_ratio = 1.0
        shift_ratio = 0.0
        lm_jump = 0.0
        if prev is None or reinit:
            q_iou = q_center = q_pose = q_scale = q_landmark = 1.0
        else:
            pose = pose_angles_deg(param)
            d_pose = max(
                abs(wrap180(pose[0] - prev.pose_deg[0])),
                abs(pose[1] - prev.pose_deg[1]),
                abs(pose[2] - prev.pose_deg[2]),
            )
            centroid, face_size, _ = face_metrics(verts68)
            scale_ratio = float(face_size / max(prev.face_size, 1e-6))
            shift_ratio = float(np.linalg.norm(centroid - prev.centroid) / max(prev.face_size, 1e-6))
            dx = np.asarray(verts68[0]) - np.asarray(prev.verts68[0])
            dy = np.asarray(verts68[1]) - np.asarray(prev.verts68[1])
            lm_jump = float(np.mean(np.hypot(dx, dy)) / max(prev.face_size, 1e-6))
            q_center = clamp01(1.0 - shift_ratio / 0.5)
            q_pose = 1.0 - clamp01((d_pose - 25.0) / 45.0)
            scale_dev = max(scale_ratio, 1.0 / max(scale_ratio, 1e-6))
            q_scale = 1.0 - clamp01((scale_dev - 1.15) / 0.40)
            q_landmark = 1.0 - clamp01((lm_jump - 0.15) / 0.30)
            if meas_iou is not None:
                q_iou = clamp01((meas_iou - 0.08) / 0.5)
            else:
                q_iou = clamp01(1.0 - shift_ratio / 0.35)

        components = {
            "detector": q_det,
            "iou": q_iou,
            "center": q_center,
            "pose": q_pose,
            "scale": q_scale,
            "landmark": q_landmark,
            "blur": q_blur,
            "out_of_frame": q_oof,
            "occlusion": q_occ,
        }
        total_w = sum(QUALITY_WEIGHTS.values())
        confidence = sum(components[k] * w for k, w in QUALITY_WEIGHTS.items()) / total_w

        hard_reject: Optional[str] = None
        if mask_area_est < qcfg.min_mask_area:
            hard_reject = "mask_too_small"
        elif not reinit and prev is not None:
            if d_pose > qcfg.max_pose_jump_deg:
                hard_reject = f"pose_jump({d_pose:.0f}deg)"
            elif scale_ratio > qcfg.max_scale_ratio or scale_ratio < qcfg.min_scale_ratio:
                hard_reject = f"scale_jump(x{scale_ratio:.2f})"
            elif shift_ratio > qcfg.max_shift_ratio:
                hard_reject = f"shift_jump({shift_ratio:.2f})"
            elif lm_jump > qcfg.max_landmark_jump_ratio:
                hard_reject = f"landmark_jump({lm_jump:.2f})"
        if occ_ratio is not None and occ_ratio > qcfg.occ_hard:
            hard_reject = hard_reject or f"occlusion_hard({occ_ratio:.2f})"
        if hard_reject is None and confidence < qcfg.confidence_threshold:
            hard_reject = f"low_confidence({confidence:.2f})"

        return QualityScores(
            confidence=float(confidence),
            components=components,
            d_pose=float(d_pose),
            scale_ratio=float(scale_ratio),
            shift_ratio=float(shift_ratio),
            lm_jump_ratio=float(lm_jump),
            hard_reject=hard_reject,
        )


# ---------------------------------------------------------------------------
# Geometry smoother: One Euro on pose/expression, regularized shape baseline
# ---------------------------------------------------------------------------

class GeometrySmoother:
    """Temporal smoothing of the 3DMM parameters.

    * pose (first 12 params) and expression (last 10) run through One Euro filters
      so that slow motion is stable and fast head turns are followed promptly;
    * shape (middle 40) runs through a slow EMA, is clipped against the reference
      baseline and finally blended with the per-frame measurement as
      ``baseline + W * (measurement - baseline)`` where ``W`` is
      ``--shape-measurement-weight``: 0 locks the shape to the reference baseline,
      1 fully adopts the measurement (identity must not jump because of the mask,
      hands or detector noise).
    """

    def __init__(
        self,
        baseline: Optional[ReferenceBaseline],
        measurement_weight: float,
        min_cutoff: float,
        beta_mult: float,
        enabled: bool = True,
        logger: logging.Logger = LOG,
    ) -> None:
        self.logger = logger
        self.enabled = bool(enabled)
        self.baseline = baseline if (baseline is not None and baseline.available) else None
        self.measurement_weight = float(np.clip(measurement_weight, 0.0, 1.0))
        pose_cfg = ONEEURO_DEFAULTS["pose12"]
        exp_cfg = ONEEURO_DEFAULTS["exp"]
        self.f_pose = OneEuroFilter(12, min_cutoff=min_cutoff, beta=pose_cfg[1] * beta_mult)
        self.f_exp = OneEuroFilter(10, min_cutoff=min_cutoff, beta=exp_cfg[1] * beta_mult)
        self.shp_ema = EMAFilter(40, alpha=SHAPE_EMA_ALPHA)
        self._last: Optional[np.ndarray] = None

    @property
    def baseline_used(self) -> bool:
        """Whether a reference shape baseline participates in the smoothing."""
        return self.baseline is not None

    def initialize(self, param: np.ndarray, t: float) -> np.ndarray:
        """(Re)start every filter at ``param``."""
        p = np.asarray(param, dtype=np.float64).reshape(-1)
        self.f_pose.reset(p[:12], t)
        self.f_exp.reset(p[52:62], t)
        shape0 = p[12:52]
        if self.baseline is not None:
            shape0 = 0.5 * (shape0 + self.baseline.shape_median)
        self.shp_ema.reset(shape0)
        self._last = self._assemble()
        assert self._last is not None
        return self._last.copy()

    def update(self, param: np.ndarray, weights: Dict[str, float], t: float) -> np.ndarray:
        """Feed a new measurement and return the smoothed parameter vector."""
        p = np.asarray(param, dtype=np.float64).reshape(-1)
        if not self.enabled:
            self._last = p.astype(np.float32)
            return self._last.copy()
        w_pose = float(np.clip(weights.get("pose", 1.0), 0.0, 1.0))
        w_exp = float(np.clip(weights.get("expression", 1.0), 0.0, 1.0))
        w_shp = float(np.clip(weights.get("shape", 1.0), 0.0, 1.0))

        pose_in = p[:12]
        if w_pose < 1.0 and self.f_pose.initialized:
            pose_in = self.f_pose.value() + w_pose * (pose_in - self.f_pose.value())
        if w_pose > 0.0:
            self.f_pose.filter(pose_in, t)

        exp_in = p[52:62]
        if w_exp < 1.0 and self.f_exp.initialized:
            exp_in = self.f_exp.value() + w_exp * (exp_in - self.f_exp.value())
        if w_exp > 0.0:
            self.f_exp.filter(exp_in, t)

        if w_shp > 0.0:
            self.shp_ema.update(p[12:52], weight=w_shp)

        self._last = self._assemble()
        assert self._last is not None
        return self._last.copy()

    def _blend_shape(self) -> np.ndarray:
        shape = self.shp_ema.value()
        if shape is None:
            shape = self.baseline.shape_median if self.baseline is not None else np.zeros(40)
        if self.baseline is not None:
            dev = np.clip(shape - self.baseline.shape_median, -SHAPE_DEV_CLIP, SHAPE_DEV_CLIP)
            shape = self.baseline.shape_median + dev
            return self.baseline.shape_median + self.measurement_weight * (shape - self.baseline.shape_median)
        return shape

    def _assemble(self) -> Optional[np.ndarray]:
        pose = self.f_pose.value()
        exp = self.f_exp.value()
        if pose is None or exp is None:
            return None
        return rebuild_param(pose, self._blend_shape(), exp)

    def current(self) -> Optional[np.ndarray]:
        """Latest smoothed parameter vector (None before initialization)."""
        return None if self._last is None else self._last.copy()


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """One processed frame, ready to be written to disk."""

    frame_id: int
    mask: np.ndarray                       # (H, W) uint8 {0, 255}
    verts68: np.ndarray                    # (3, 68) final projected landmarks
    contour: Optional[np.ndarray]          # (M, 2) float32 silhouette polygon or None
    record: Dict[str, Any]                 # params.jsonl record (JSON-safe)
    debug: Dict[str, Any] = field(default_factory=dict)  # misc for the debug overlay


class ParamsStore:
    """Append-friendly ``params.jsonl`` manager with dedupe + atomic rewrite.

    Every processed frame appends one JSON line (crash-granular).  Since reruns
    may append a second line for the same frame id, readers must treat the *last*
    line of a frame id as authoritative; the store atomically rewrites the file
    (sorted, deduplicated) at the end of the run and at checkpoints, so a cleanly
    finished run always contains exactly one line per frame.
    """

    def __init__(self, path: Path, logger: logging.Logger = LOG) -> None:
        self.path = path
        self.logger = logger
        self.records: Dict[int, Dict[str, Any]] = {}
        self.prerun_fids: set = set()
        self.invalid_lines = 0
        self._fh = None
        self._writes_since_rewrite = 0
        self._dirty_dupes = False

    def load(self) -> None:
        """Load existing records (last line per frame wins) as the resume index."""
        self.records.clear()
        self.invalid_lines = 0
        self._dirty_dupes = False
        if not self.path.is_file():
            self.prerun_fids = set()
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        self.invalid_lines += 1
                        continue
                    if not isinstance(rec, dict) or "frame_id" not in rec:
                        self.invalid_lines += 1
                        continue
                    try:
                        fid = int(rec["frame_id"])
                    except (TypeError, ValueError):
                        self.invalid_lines += 1
                        continue
                    if fid in self.records:
                        self._dirty_dupes = True
                    self.records[fid] = rec
        except OSError as exc:
            self.logger.warning("could not read %s: %s", self.path, exc)
        self.prerun_fids = set(self.records.keys())
        if self.records:
            self.logger.info("params.jsonl: %d existing frame records (%d invalid lines)",
                             len(self.records), self.invalid_lines)

    def has(self, frame_id: int) -> bool:
        """Whether a *pre-existing* valid record exists (used for resume skips)."""
        return frame_id in self.prerun_fids

    def get(self, frame_id: int) -> Optional[Dict[str, Any]]:
        """Return the current record for a frame id (or None)."""
        return self.records.get(frame_id)

    def _append_handle(self):
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8", newline="\n")
        return self._fh

    def put(self, record: Dict[str, Any]) -> None:
        """Store and append one frame record."""
        fid = int(record["frame_id"])
        self.records[fid] = record
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=_json_default)
        fh = self._append_handle()
        fh.write(line + "\n")
        fh.flush()
        self._writes_since_rewrite += 1

    def rewrite(self) -> None:
        """Atomically rewrite the file: sorted by frame_id, one record per frame."""
        try:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                for fid in sorted(self.records.keys()):
                    rec = self.records[fid]
                    fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=_json_default))
                    fh.write("\n")
            os.replace(str(tmp), str(self.path))
            self._writes_since_rewrite = 0
        except OSError as exc:
            self.logger.warning("params.jsonl rewrite failed: %s", exc)

    def maybe_rewrite(self, every: int, force: bool = False) -> None:
        """Rewrite when enough writes accumulated (or when forced)."""
        if force or (every > 0 and self._writes_since_rewrite >= every):
            self.rewrite()

    def close(self) -> None:
        """Flush, rewrite and close."""
        self.rewrite()


class OutputWriter:
    """Owns the output tree, per-frame atomic writes and debug rendering."""

    def __init__(
        self,
        output_dir: Path,
        debug: bool,
        debug_every: int,
        save_contour: bool,
        logger: logging.Logger = LOG,
    ) -> None:
        self.logger = logger
        self.root = Path(output_dir)
        self.face_mask_dir = ensure_dir(self.root / "face_mask")
        self.landmarks_dir = ensure_dir(self.root / "landmarks")
        self.debug_dir = ensure_dir(self.root / "debug") if debug else None
        self.debug = bool(debug)
        self.debug_every = max(1, int(debug_every))
        self.save_contour = bool(save_contour)
        self.params = ParamsStore(self.root / "params.jsonl", logger)
        self.frame_index = 0

    def load_params(self) -> None:
        """Load the params.jsonl resume index."""
        self.params.load()

    def frame_paths(self, frame_id: int) -> Tuple[Path, Path, Path]:
        """Return (mask, landmarks, contour) paths for a frame id."""
        name = FRAME_FMT.format(frame_id)
        return (
            self.face_mask_dir / f"{name}.png",
            self.landmarks_dir / f"{name}.npy",
            self.landmarks_dir / f"{name}_contour.npy",
        )

    def frame_complete(self, frame_id: int) -> bool:
        """Resume check: mask + landmarks exist and a valid params record exists."""
        mask_fp, lm_fp, _ = self.frame_paths(frame_id)
        try:
            if not (mask_fp.is_file() and mask_fp.stat().st_size > 0):
                return False
            if not (lm_fp.is_file() and lm_fp.stat().st_size > 0):
                return False
        except OSError:
            return False
        return self.params.has(frame_id)

    def wants_debug(self, frame_id: int, start: int) -> bool:
        """Whether a debug overlay should be written for this frame."""
        if not self.debug or self.debug_dir is None:
            return False
        return (frame_id - start) % self.debug_every == 0

    def write(
        self,
        result: FrameResult,
        frame_bgr: Optional[np.ndarray] = None,
        want_debug: bool = False,
        occ_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Write mask / landmarks / contour / params (and debug) for one frame."""
        mask_fp, lm_fp, contour_fp = self.frame_paths(result.frame_id)
        mask = np.asarray(result.mask, dtype=np.uint8)
        atomic_write_png(mask_fp, mask)
        lms = np.asarray(result.verts68[:2].T, dtype=np.float32)
        if not is_finite_array(lms):
            raise ValueError(f"non-finite landmarks on frame {result.frame_id}")
        atomic_write_npy(lm_fp, lms)
        if self.save_contour:
            if result.contour is not None and result.contour.shape[0] >= 3:
                atomic_write_npy(contour_fp, np.asarray(result.contour, dtype=np.float32))
            elif contour_fp.is_file():
                try:
                    contour_fp.unlink()  # stale contour from an older run
                except OSError:
                    pass
        if want_debug and frame_bgr is not None and self.debug_dir is not None:
            overlay = draw_debug_overlay(frame_bgr, result, occ_mask=occ_mask)
            atomic_write_jpg(self.debug_dir / f"{FRAME_FMT.format(result.frame_id)}.jpg", overlay)
        self.params.put(result.record)
        self.frame_index += 1

    def finalize(self) -> None:
        """Flush the params store."""
        self.params.close()


def draw_debug_overlay(
    frame_bgr: np.ndarray,
    result: FrameResult,
    occ_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render the debug overlay: mask boundary (green), landmarks (yellow),
    jaw contour (cyan), tracked bbox (blue), occlusion boundary (magenta),
    status text (red when fallback / low confidence)."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    rec = result.record

    # theoretical mask fill + boundary
    mask = result.mask > 0
    if mask.any():
        tint = np.zeros_like(img)
        tint[mask] = (80, 220, 80)
        img = cv2.addWeighted(img, 1.0, tint, 0.15, 0)
        if result.contour is not None and result.contour.shape[0] >= 3:
            poly = np.round(result.contour).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

    # landmarks (yellow) and jaw (cyan)
    lms = np.asarray(result.verts68[:2].T, dtype=np.float64)
    for i in range(min(lms.shape[0], 68)):
        px, py = int(round(lms[i, 0])), int(round(lms[i, 1]))
        px = min(max(px, 0), w - 1)
        py = min(max(py, 0), h - 1)
        cv2.circle(img, (px, py), 1, (0, 255, 255), -1, cv2.LINE_AA)
    jaw = np.round(lms[:17]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [jaw], False, (255, 255, 0), 1, cv2.LINE_AA)

    # tracked bbox (blue)
    tbox = result.debug.get("track_bbox")
    if tbox is not None:
        x1, y1, x2, y2 = (int(round(float(v))) for v in np.asarray(tbox)[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)

    # occlusion mask boundary (magenta)
    if occ_mask is not None and occ_mask.size:
        occ_bin = (occ_mask > 127).astype(np.uint8)
        if occ_bin.shape[:2] == (h, w) and occ_bin.any():
            contours, _ = cv2.findContours(occ_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 0, 255), 1, cv2.LINE_AA)

    # status text
    fallback = bool(rec.get("fallback_used"))
    invalid = bool(rec.get("invalid"))
    conf = float(rec.get("confidence", 0.0))
    color = (0, 0, 255) if (fallback or invalid) else (0, 255, 0)
    lines = [
        "f={:06d} yaw={:.1f} pitch={:.1f} roll={:.1f}".format(
            int(rec.get("frame_id", -1)), float(rec.get("yaw", 0.0)),
            float(rec.get("pitch", 0.0)), float(rec.get("roll", 0.0))),
        "conf={:.2f} occ={:.2f} mask={} mode={}".format(
            conf, float(rec.get("occlusion_ratio", 0.0)), int(rec.get("mask_area", 0)),
            str(result.debug.get("raster_mode", "-"))),
        "fallback={} invalid={} scale={:.2f} blur={:.1f}".format(
            int(fallback), int(invalid), float(rec.get("scale", 1.0)), float(rec.get("blur_score", 0.0))),
    ]
    y = 18
    for text in lines:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 18
    return img


# ---------------------------------------------------------------------------
# Optional SeC mask inputs
# ---------------------------------------------------------------------------

class MaskDirReader:
    """Reads per-frame binary PNG masks from an optional directory (never writes)."""

    def __init__(self, root: Optional[str], name: str, logger: logging.Logger = LOG) -> None:
        self.root = Path(root) if root else None
        self.name = name
        self.logger = logger
        self.missing = 0
        self.corrupt = 0
        self._warned = False
        self._resize_warned = False

    @property
    def available(self) -> bool:
        """Whether a directory was provided."""
        return self.root is not None

    def read(self, frame_id: int) -> Optional[np.ndarray]:
        """Return the frame mask as uint8 {0, 255}, or None when unavailable."""
        if self.root is None:
            return None
        path = self.root / f"{FRAME_FMT.format(frame_id)}.png"
        if not path.is_file():
            self.missing += 1
            if not self._warned:
                self.logger.warning("%s: first missing frame mask at %s (further warnings suppressed)",
                                    self.name, path.name)
                self._warned = True
            return None
        img = imread_gray(path)
        if img is None:
            self.corrupt += 1
            return None
        return np.where(img > 127, np.uint8(255), np.uint8(0))

    def auto_resized(self, expected_hw: Tuple[int, int], mask: np.ndarray) -> np.ndarray:
        """Nearest-resize a mask to the video resolution when it does not match."""
        if mask.shape[:2] == expected_hw:
            return mask
        if not self._resize_warned:
            self.logger.warning("%s mask resolution %s != video %s; resizing (nearest)",
                                self.name, mask.shape[:2], expected_hw)
            self._resize_warned = True
        return cv2.resize(mask, (expected_hw[1], expected_hw[0]), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Runner: the per-frame pipeline
# ---------------------------------------------------------------------------

class Runner:
    """Full run pipeline: init once, stream frames, write outputs, resume safely."""

    def __init__(self, args: argparse.Namespace, logger: logging.Logger = LOG) -> None:
        self.args = args
        self.logger = logger
        self.video = Path(args.video).expanduser()
        if not self.video.is_file():
            raise FatalError(f"video not found: {self.video}")
        self.output_dir = Path(args.output_dir).expanduser()
        self.start = max(0, int(args.start or 0))
        self.end: Optional[int] = int(args.end) if args.end is not None else None
        if self.end is not None and self.end <= self.start:
            raise FatalError(f"empty frame range: --start {self.start} --end {self.end} (end is exclusive)")
        self.overwrite = bool(args.overwrite)
        self.resume = bool(args.resume) and not self.overwrite

        self.qcfg = QualityConfig(
            confidence_threshold=float(args.confidence_threshold),
            occ_mid=float(args.mid_occlusion_threshold),
            occ_severe=float(args.occlusion_threshold),
            occ_hard=float(args.hard_occlusion_threshold),
            max_pose_jump_deg=float(args.max_pose_jump_deg),
            min_scale_ratio=float(args.min_scale_ratio),
            max_scale_ratio=float(args.max_scale_ratio),
            max_shift_ratio=float(args.max_shift_ratio),
            max_landmark_jump_ratio=float(args.max_landmark_jump_ratio),
            blur_threshold=float(args.blur_threshold),
            out_of_frame_tolerance=float(args.out_of_frame_tolerance),
            min_mask_area=int(args.min_mask_area),
        )

        self.writer = OutputWriter(
            self.output_dir,
            debug=bool(args.debug),
            debug_every=int(args.debug_every),
            save_contour=bool(args.save_contour),
            logger=logger,
        )
        self.writer.load_params()

        self.adapter: Optional[TDDFAAdapter] = None
        self.rasterizer: Optional[MeshRasterizer] = None
        self.smoother: Optional[GeometrySmoother] = None
        self.tracker: Optional[FaceTracker] = None
        self.baseline: Optional[ReferenceBaseline] = None

        self.prev_accepted: Optional[AcceptedState] = None
        self.prev_mask: Optional[np.ndarray] = None
        self.scale_baseline_px: Optional[float] = None
        self._scale_samples: List[float] = []
        self._raster_verified = False
        self._raw_lm_bbox = (0.0, 0.0, 1.0, 1.0)
        self._last_raw_oof = 0.0
        self.cap: Optional["cv2.VideoCapture"] = None
        # Decoder bookkeeping: frame id that the *next* ``cap.read()`` will
        # return, or None when unknown.  Kept next to ``self.cap`` so a resume
        # seek can be verified instead of trusted (a stale ``CAP_PROP_POS_FRAMES``
        # is exactly what desynchronises frame ids from decoded content).
        self._next_fid: Optional[int] = None
        self.width = 0
        self.height = 0
        self.fps = 30.0

        if getattr(args, "sec4b_dir", None):
            sec_root = Path(args.sec4b_dir)
            if not args.sec_mask_dir and (sec_root / "mask").is_dir():
                args.sec_mask_dir = str(sec_root / "mask")
            if not args.sec_occlusion_dir and (sec_root / "occlusion").is_dir():
                args.sec_occlusion_dir = str(sec_root / "occlusion")
        self.occ_reader = MaskDirReader(args.sec_occlusion_dir, "occlusion", logger)
        self.sec_reader = MaskDirReader(args.sec_mask_dir, "sec_mask", logger)

        self.trajectory = None
        if getattr(args, "trajectory", None) and os.path.isfile(args.trajectory):
            try:
                self.trajectory = np.load(args.trajectory)
                self.logger.info("loaded YOLO head trajectory: %s (shape: %s)", args.trajectory, self.trajectory.shape)
            except Exception as exc:
                self.logger.warning("failed to load trajectory %s: %s", args.trajectory, exc)

        self.stats: Dict[str, int] = {
            "processed": 0, "skipped": 0, "fallback": 0, "invalid": 0,
            "reinit": 0, "recovered": 0, "detections": 0, "fit_failures": 0,
            "decode_failures": 0,
        }
        self._log_t0 = time.time()
        self._log_n = 0
        self._window_t0 = time.time()
        self._window_n = 0

    # -- public entry ---------------------------------------------------------

    def run(self) -> int:
        """Execute the whole run; returns the process exit code."""
        frame_count, fps = self._probe_video()
        end = self.end if self.end is not None else frame_count
        if frame_count is not None and end > frame_count:
            self.logger.warning("--end %d exceeds video frame count %d; clamping", end, frame_count)
            end = frame_count
        if end is None:
            end = self.start + 10 ** 7  # unknown length: read until decode fails
        if frame_count is not None and self.start >= frame_count:
            raise FatalError(f"--start {self.start} is beyond the video ({frame_count} frames)")
        end = int(end)

        # Resume: scan the completed prefix of [start, end) and restart from the
        # first incomplete frame.  Frames are never skipped mid-range: the video
        # decoder, tracker filters and smoother must see every frame in order,
        # otherwise frame ids and video content drift apart.
        effective_start = self.start
        if self.resume:
            first_incomplete = self._first_incomplete(self.start, end)
            self.stats["skipped"] = first_incomplete - self.start
            if self.stats["skipped"]:
                self.logger.info("resume: %d completed frame(s) in [%d, %d)",
                                 self.stats["skipped"], self.start, first_incomplete)
            if first_incomplete >= end:
                self.logger.info("resume: [%d, %d) is already complete; nothing to do", self.start, end)
                try:
                    self.writer.finalize()
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.warning("params finalize failed: %s", exc)
                self._log_summary()
                return 0
            effective_start = first_incomplete

        root = TDDFAAdapter.discover_root(self.args.tddfa_root)
        if root is None:
            raise FatalError(
                "3DDFA_V2 repository not found. Clone it (git clone "
                "https://github.com/cleardusk/3DDFA_V2) and pass --tddfa-root PATH, "
                "or set TDDFA_ROOT, or place it at ./3DDFA_V2."
            )
        self.logger.info("3DDFA_V2 root: %s", root)
        self.adapter = TDDFAAdapter(
            root,
            config=Path(self.args.config).expanduser() if self.args.config else None,
            device=self.args.device,
            checkpoint=self.args.checkpoint,
            onnx=bool(self.args.onnx),
            logger=self.logger,
        )
        try:
            self._init_components()
            self.cap = open_video(self.video)
            self._next_fid = 0
            try:
                if effective_start > 0:
                    if not self._seek_to(effective_start):
                        raise FatalError(
                            f"cannot position the decoder at frame {effective_start}: refusing to run "
                            "because frame ids would no longer match the decoded video content"
                        )
                    self.logger.info("decoder verified at frame %d (resume point)", effective_start)
                self._try_warm_start(effective_start)
                self._main_loop(effective_start, end)
            finally:
                try:
                    if self.cap is not None:
                        self.cap.release()
                except Exception:
                    pass
                self.cap = None
                self._next_fid = None
        finally:
            try:
                self.writer.finalize()
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.warning("params finalize failed: %s", exc)
            if self.adapter is not None:
                self.adapter.close()
        self._log_summary()
        return 0

    def _init_components(self) -> None:
        assert self.adapter is not None
        builder = ReferenceGeometryBuilder(self.adapter, self.logger)
        self.baseline = builder.load_or_build(
            Path(self.args.reference_dir).expanduser() if self.args.reference_dir else None,
            self.output_dir / "reference_baseline.npz",
            rebuild=bool(self.args.rebuild_reference),
        )
        self.rasterizer = MeshRasterizer(
            self.adapter.tri,
            self.adapter.mean_template_flat(),
            logger=self.logger,
            mode=str(self.args.rasterizer),
        )
        self.smoother = GeometrySmoother(
            self.baseline,
            measurement_weight=float(self.args.shape_measurement_weight),
            min_cutoff=float(self.args.smooth_min_cutoff),
            beta_mult=float(self.args.smooth_beta),
            enabled=not bool(self.args.no_smooth),
            logger=self.logger,
        )
        self.tracker = FaceTracker(self.qcfg, self.logger)

    def _probe_video(self) -> Tuple[Optional[int], float]:
        cap = open_video(self.video)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if width <= 0 or height <= 0:
            raise FatalError(f"cannot read video size from {self.video}")
        self.width, self.height = width, height
        if not (1.0 <= fps <= 240.0):
            self.logger.warning("implausible fps=%.2f; assuming 30", fps)
            fps = 30.0
        self.fps = fps
        frame_count = count if count > 0 else None
        self.logger.info(
            "video: %dx%d fps=%.2f frames=%s range=[%d, %s)",
            width, height, fps, str(frame_count) if frame_count else "unknown",
            self.start, str(self.end) if self.end is not None else "EOF",
        )
        if self.occ_reader.available:
            self.logger.info("occlusion masks: %s", self.occ_reader.root)
        if self.sec_reader.available:
            self.logger.info("sec mask: %s", self.sec_reader.root)
        return frame_count, fps

    # -- resume scan / warm start ------------------------------------------------

    def _first_incomplete(self, start: int, end: int) -> int:
        """First frame id in [start, end) whose mask+landmarks+params are not all present.

        Returns ``end`` when the whole range is already complete.  The walk is
        bounded by the pre-existing records, so an open-ended range (unknown video
        length) terminates at the first gap instead of scanning billions of ids.
        """
        fid = start
        while fid < end and self.writer.frame_complete(fid):
            fid += 1
        return fid

    def _try_warm_start(self, first_fid: int) -> None:
        """Restore tracker/filter state from the record of ``first_fid - 1`` if possible."""
        if first_fid <= 0 or self.adapter is None or self.smoother is None or self.tracker is None:
            return
        rec = self.writer.params.get(first_fid - 1)
        if not rec or rec.get("param_vector") is None or rec.get("roi_box") is None:
            self.logger.info("cold start at frame %d (no usable warm-start record)", first_fid)
            return
        try:
            param = np.asarray(rec["param_vector"], dtype=np.float64).reshape(-1)
            roi = np.asarray(rec["roi_box"], dtype=np.float64).reshape(4)
            if param.size != 62:
                raise ValueError("bad param_vector")
            verts68 = self.adapter.recon_sparse(param.astype(np.float32), roi.astype(np.float32))
            pose = pose_angles_deg(param)
            centroid, face_size, lm_bbox = face_metrics(verts68)
            track_bbox = rec.get("detector_bbox") or rec.get("face_bbox")
            track_bbox = np.asarray(track_bbox if track_bbox else lm_bbox, dtype=np.float64).reshape(4)
            state = AcceptedState(
                frame_id=first_fid - 1,
                param=param.astype(np.float32),
                roi_box=roi.astype(np.float32),
                verts68=verts68,
                pose_deg=pose,
                centroid=centroid,
                face_size=face_size,
                lm_bbox=lm_bbox,
                mask_area=int(rec.get("mask_area", 0)),
                track_bbox=track_bbox,
                det_score=float(rec.get("detector_confidence", 0.85)),
                quality=float(rec.get("confidence", 0.85)),
                scale_baseline_px=rec.get("scale_baseline_px"),
            )
            self.prev_accepted = state
            self.tracker.track_bbox = track_bbox
            t_prev = self._frame_time(first_fid - 1)
            self.tracker.bbox_filter.reset(track_bbox, t_prev)
            self.tracker.bbox_velocity = np.zeros(4, dtype=np.float64)
            self.tracker.det_score_carry = float(rec.get("detector_confidence", 0.85))
            self.smoother.initialize(state.param, t_prev)
            # Recover the motion velocity from the previous record so a fallback
            # right after resume can extrapolate instead of freezing on "hold".
            prev_rec = self.writer.params.get(first_fid - 2)
            if prev_rec and prev_rec.get("param_vector") is not None:
                try:
                    p_prev = np.asarray(prev_rec["param_vector"], dtype=np.float64).reshape(-1)
                    if p_prev.size == 62:
                        state.v_param = np.asarray(state.param, np.float64) - p_prev
                except (TypeError, ValueError):
                    state.v_param = None
            if rec.get("scale_baseline_px"):
                self.scale_baseline_px = float(rec["scale_baseline_px"])
            mask_fp, _, _ = self.writer.frame_paths(first_fid - 1)
            if mask_fp.is_file():
                m = imread_gray(mask_fp)
                if m is not None:
                    self.prev_mask = np.where(m > 127, np.uint8(255), np.uint8(0))
            self.logger.info("warm start from frame %d (quality=%.2f)", first_fid - 1, state.quality)
        except Exception as exc:
            self.logger.warning("warm start failed (%s); cold start", exc)
            self.prev_accepted = None

    # -- main loop ---------------------------------------------------------------

    def _main_loop(self, start: int, end: int) -> None:
        """Process frames [start, end) in order.

        ``start`` must be the first incomplete frame (see :meth:`_first_incomplete`):
        every frame from there is read from the decoder and processed consecutively,
        so resume never skips frames mid-range and tracker/smoother state always
        matches the video content.
        """
        assert self.tracker is not None and self.rasterizer is not None
        consecutive_decode_fail = 0
        for fid in range(start, end):
            frame = self._read_frame(fid)
            if frame is None:
                consecutive_decode_fail += 1
                self.stats["decode_failures"] += 1
                self.logger.warning("frame %d could not be decoded (%d consecutive)", fid, consecutive_decode_fail)
                if consecutive_decode_fail > int(self.args.max_decode_failures):
                    raise FatalError(
                        f"{consecutive_decode_fail} consecutive decode failures around frame {fid}; aborting"
                    )
                result = self._fallback_result(fid, None, "decode_failed")
                self._emit(result, None)
                continue
            consecutive_decode_fail = 0

            try:
                result, occ_for_debug = self._process_frame(fid, frame)
            except FatalError:
                raise  # model/repo level failures must abort, not feed the fallback loop
            except Exception as exc:
                self.logger.exception("frame %d failed: %s", fid, exc)
                self.stats["fit_failures"] += 1
                result = self._fallback_result(fid, frame, "exception")
                occ_for_debug = None
            self._emit(result, frame, occ_mask=occ_for_debug)

            self.stats["processed"] += 1
            self._progress_log(fid, result)
            if int(self.args.empty_cache_every) > 0 and fid % int(self.args.empty_cache_every) == 0:
                self._empty_cache()
            if fid % 1000 == 0:
                gc.collect()
            self.writer.params.maybe_rewrite(int(self.args.params_rewrite_every))

    def _emit(self, result: FrameResult, frame: Optional[np.ndarray],
              occ_mask: Optional[np.ndarray] = None) -> None:
        want_debug = self.writer.wants_debug(result.frame_id, self.start)
        self.writer.write(result, frame_bgr=frame, want_debug=want_debug, occ_mask=occ_mask)

    def _seek_to(self, target: int) -> bool:
        """Position the decoder so that the next ``read()`` returns frame ``target``.

        Returns True only when the position is *verified*.  ``VideoCapture.set``
        is a best-effort request: some backends ignore it, and some report a
        stale ``CAP_PROP_POS_FRAMES``.  Trusting either one is exactly what made
        resume desynchronise frame ids from the decoded content, so the position
        is checked and, when the fast path fails, recovered by decoding forward
        from frame 0 (sequential reads are always exact).
        """
        cap = self.cap
        if cap is None:
            return False
        if target < 0:
            return False
        if self._next_fid == target:
            return True          # already positioned exactly where we need to be
        if target == 0:
            if not self._reopen():
                return False
            self._next_fid = 0   # a fresh capture always decodes frame 0 next
            return True

        for attempt in range(2):
            cap = self.cap
            assert cap is not None
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, float(target)):
                self.logger.warning("backend refused CAP_PROP_POS_FRAMES=%d (attempt %d)", target, attempt + 1)
            else:
                pos = self._reported_pos()
                if pos == target:
                    self._next_fid = target
                    return True
                self.logger.warning("seek to %d reported position %s (attempt %d)",
                                    target, str(pos), attempt + 1)
            # Re-open: a fresh decoder has no cached position to lie about.
            if not self._reopen():
                return False
            self._next_fid = 0
            cap = self.cap
        return self._seek_by_decoding(target)

    def _seek_by_decoding(self, target: int) -> bool:
        """Last-resort positioning: decode and discard frames 0..target-1.

        Sequential decoding is exact on every backend, so this always lands on
        the right frame; it costs ``target`` decodes and is only used when the
        container's random access is unusable.
        """
        cap = self.cap
        if cap is None:
            return False
        self.logger.warning("random-access seek unavailable; decoding forward to frame %d "
                            "(slow but exact)", target)
        if not self._reopen():
            return False
        cap = self.cap
        assert cap is not None
        self._next_fid = 0
        for fid in range(target):
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                self._next_fid = None
                self.logger.error("cannot decode forward to frame %d (stopped at %d)", target, fid)
                return False
            self._next_fid = fid + 1
        return self._next_fid == target

    def _reopen(self) -> bool:
        """Release and re-open the video; True when a usable capture resulted.

        Leaves ``self.cap = None`` and ``self._next_fid = None`` on failure, so a
        caller can never mistake an unusable handle for a positioned one.
        """
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self._next_fid = None
        try:
            self.cap = open_video(self.video)
        except FatalError as exc:
            self.logger.error("cannot reopen %s: %s", self.video, exc)
            return False
        return True

    def _reported_pos(self) -> Optional[int]:
        """Current ``CAP_PROP_POS_FRAMES`` as an int, or None when unusable."""
        cap = self.cap
        if cap is None:
            return None
        try:
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        except Exception:
            return None
        if pos is None or not np.isfinite(float(pos)):
            return None
        return int(round(float(pos)))

    def _read_frame(self, fid: int) -> Optional[np.ndarray]:
        """Read the next sequential frame; on failure reopen the video and re-seek.

        The decoder position is cross-checked against ``fid`` before every read:
        a mismatch means the stream moved under us, and reading anyway would
        silently attach the wrong video content to ``fid`` (the resume bug this
        guard exists to prevent).
        """
        cap = self.cap
        if cap is None:
            return None
        if self._next_fid != fid:
            # Either the stream moved under us, or the position is unknown (the
            # normal state right after a decode failure); in both cases reading
            # now could attach the wrong content to ``fid``.
            if self._next_fid is None:
                self.logger.warning("decoder position unknown for frame %d; re-seeking", fid)
            else:
                self.logger.error("decoder desync: expected frame %d, decoder at %d; re-seeking",
                                  fid, self._next_fid)
            if not self._seek_to(fid):
                return None
            cap = self.cap
            if cap is None:
                return None
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            self._next_fid = fid + 1
            return frame
        # Decode failure: the stream may be corrupt from here on.  Re-open and
        # re-position with verification so a single bad packet cannot shift
        # every subsequent frame id.
        if not self._reopen():
            return None
        cap = self.cap
        assert cap is not None
        self._next_fid = 0
        if not self._seek_to(fid):
            return None
        cap = self.cap
        assert cap is not None
        ok2, frame2 = cap.read()
        if ok2 and frame2 is not None and frame2.size > 0:
            self._next_fid = fid + 1
            self.logger.info("video reopened after a decode failure at frame %d", fid)
            return frame2
        self._next_fid = None
        return None

    def _empty_cache(self) -> None:
        if self.adapter is not None:
            try:
                if self.adapter._torch.cuda.is_available():
                    self.adapter._torch.cuda.empty_cache()
            except Exception:
                pass

    def _frame_time(self, fid: int) -> float:
        """Wall-clock timestamp (seconds) of frame ``fid``.

        The One Euro filters take *seconds* and their cutoffs are in Hz, so the
        filter time base must be the frame index divided by the frame rate.
        Passing the raw frame index made every ``dt`` equal to 1.0 s, which blew
        past the re-snap threshold and reset the filter on every frame — i.e. no
        smoothing at all.  Falls back to 30 fps when the container reports none.
        """
        fps = self.fps if self.fps and self.fps > 0 else 30.0
        return float(fid) / float(fps)

    # -- per-frame pipeline ------------------------------------------------------

    def _process_frame(self, fid: int, frame: np.ndarray) -> Tuple[FrameResult, Optional[np.ndarray]]:
        assert self.adapter is not None and self.rasterizer is not None
        assert self.smoother is not None and self.tracker is not None
        tracker = self.tracker
        h, w = frame.shape[:2]
        t = self._frame_time(fid)

        # 1) detection (only when needed: init / lost / periodic / low quality)
        meas: Optional[Measurement] = None
        if tracker.needs_detection(int(self.args.detect_interval), int(self.args.low_quality_trigger)):
            dets = None
            if self.trajectory is not None:
                traj_idx = fid if fid < len(self.trajectory) else (fid - self.start)
                if 0 <= traj_idx < len(self.trajectory):
                    row = self.trajectory[traj_idx]
                    if row[0] > 0.5:
                        x1, y1, x2, y2, conf = float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])
                        dets = np.array([[x1, y1, x2, y2, conf]], dtype=np.float32)
                    else:
                        dets = np.zeros((0, 5), dtype=np.float32)
            if dets is None:
                dets = self.adapter.detect(frame)
            self.stats["detections"] += 1
            tracker.frames_since_detect = 0
            meas = tracker.select_measurement(dets)
            if meas is not None:
                tracker.det_score_carry = max(float(meas.score), tracker.det_score_carry * 0.8)
        else:
            tracker.decay_score()

        # 2) decide the fit bbox
        reinit = False
        meas_iou: Optional[float] = None
        if meas is not None:
            reinit = tracker.track_bbox is None or tracker.lost_count >= int(self.args.reinit_after_lost)
            fit_bbox = tracker.update_bbox(meas, t, reinit=reinit)
            meas_score: Optional[float] = float(meas.score)
            meas_iou = float(meas.iou) if meas.iou is not None else None
        elif tracker.track_bbox is not None:
            fit_bbox = tracker.filtered_bbox()
            meas_score = None
        else:
            result = self._fallback_result(fid, frame, "no_face")
            return result, None
        if fit_bbox is None:
            result = self._fallback_result(fid, frame, "no_face")
            return result, None

        # 3) fit + projected landmarks
        try:
            param_raw, roi = self.adapter.fit(frame, fit_bbox)
            verts68_raw = self.adapter.recon_sparse(param_raw, roi)
        except Exception as exc:
            self.logger.debug("fit failed on frame %d: %s", fid, exc)
            raise
        _, face_size_raw, lm_bbox_raw = face_metrics(verts68_raw)
        self._raw_lm_bbox = lm_bbox_raw
        oof = out_of_frame_ratio(lm_bbox_raw, w, h)
        self._last_raw_oof = oof

        # 4) occlusion / sec mask ratios against the previous stable mask
        occ_mask = self.occ_reader.read(fid)
        if occ_mask is not None:
            occ_mask = self.occ_reader.auto_resized((h, w), occ_mask)
        sec_mask = self.sec_reader.read(fid)
        if sec_mask is not None:
            sec_mask = self.sec_reader.auto_resized((h, w), sec_mask)
        occ_ratio = mask_ratio(self.prev_mask, occ_mask) if self.prev_mask is not None else None
        sec_ratio = mask_ratio(self.prev_mask, sec_mask) if self.prev_mask is not None else None

        blur = blur_score(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), fit_bbox)
        mask_area_est = 0.55 * max((lm_bbox_raw[2] - lm_bbox_raw[0]) * (lm_bbox_raw[3] - lm_bbox_raw[1]), 0.0)

        # 5) quality & decision
        qs = tracker.evaluate(
            param_raw, verts68_raw, meas_score, meas_iou, blur, occ_ratio, oof,
            mask_area_est, self.prev_accepted, reinit,
        )
        accept = qs.hard_reject is None

        if accept:
            weights = {"pose": 1.0, "expression": 1.0, "shape": 1.0}
            reduced = False
            if occ_ratio is not None and occ_ratio > self.qcfg.occ_severe:
                reduced = True
                weights["expression"] = 0.0
                weights["shape"] = 0.0
                weights["pose"] = 0.25 if qs.confidence >= 0.75 else 0.10
            elif occ_ratio is not None and occ_ratio > self.qcfg.occ_mid:
                reduced = True
                weights["expression"] = 0.4
                weights["shape"] = 0.3
                weights["pose"] = 0.6
            if sec_ratio is not None and sec_ratio > 0.4:
                weights["expression"] *= 1.0 - 0.7 * clamp01((sec_ratio - 0.4) / 0.6)
                weights["shape"] *= 1.0 - 0.5 * clamp01((sec_ratio - 0.4) / 0.6)
            if reinit or not self.smoother.f_pose.initialized:
                smoothed = self.smoother.initialize(param_raw, t)
                self.stats["reinit"] += 1
            else:
                smoothed = self.smoother.update(param_raw, weights, t)

            verts68 = self.adapter.recon_sparse(smoothed, roi)
            dense = self.adapter.recon_dense(smoothed, roi)
            mask, loops, info = self.rasterizer.rasterize(dense, verts68, (h, w))
            if not self._raster_verified:
                self._raster_verified = self._verify_rasterizer(dense, verts68, (h, w))
            contour = self._mask_contour(mask, loops)

            prev_state = self.prev_accepted
            v_param = None
            if prev_state is not None and not reinit:
                v_param = np.asarray(smoothed, np.float64) - np.asarray(prev_state.param, np.float64)
            if self.tracker.lost_count > 0:
                self.stats["recovered"] += 1
            self._update_scale_baseline(face_size_raw)
            state = AcceptedState(
                frame_id=fid,
                param=np.asarray(smoothed, np.float32),
                roi_box=np.asarray(roi, np.float32),
                verts68=verts68,
                pose_deg=pose_angles_deg(smoothed),
                centroid=face_metrics(verts68)[0],
                face_size=face_metrics(verts68)[1],
                lm_bbox=face_metrics(verts68)[2],
                mask_area=int(np.count_nonzero(mask)),
                track_bbox=tracker.track_bbox.copy() if tracker.track_bbox is not None else fit_bbox,
                det_score=tracker.det_score_carry,
                quality=qs.confidence,
                v_param=v_param,
                scale_baseline_px=self.scale_baseline_px,
            )
            self.prev_accepted = state
            self.prev_mask = mask
            tracker.lost_count = 0
            tracker.low_quality_streak = 0
            record = self._make_record(
                fid, state, qs, occ_ratio, sec_ratio, blur,
                fallback=False, invalid=False, reduced=reduced, reinit=reinit,
                raster_info=info, source_frame=None,
            )
            result = FrameResult(
                frame_id=fid, mask=mask, verts68=verts68, contour=contour, record=record,
                debug={"track_bbox": state.track_bbox, "raster_mode": info.get("mode", "?")},
            )
            return result, occ_mask

        # fallback path
        self.stats["fallback"] += 1
        result = self._fallback_result(fid, frame, qs.hard_reject or "rejected", qs=qs,
                                       occ_ratio=occ_ratio, sec_ratio=sec_ratio, blur=blur)
        return result, occ_mask

    def _update_scale_baseline(self, face_size: float) -> None:
        if self.scale_baseline_px is not None or face_size <= 1.0:
            return
        if len(self._scale_samples) < 60:
            self._scale_samples.append(face_size)
        if len(self._scale_samples) >= 60:
            self.scale_baseline_px = float(np.median(np.asarray(self._scale_samples)))

    def _verify_rasterizer(self, dense: np.ndarray, verts68: np.ndarray, hw: Tuple[int, int],
                           ) -> bool:
        """One-time chain-vs-triangle fill check on the first accepted frame."""
        assert self.rasterizer is not None
        if self.rasterizer.tri is None or self.rasterizer.mode in ("triangles", "hull"):
            return True
        try:
            keep_mode = self.rasterizer.mode
            self.rasterizer.mode = "chain"
            mask_chain, _, _ = self.rasterizer.rasterize(dense, verts68, hw)
            self.rasterizer.mode = "triangles"
            mask_tri, _, _ = self.rasterizer.rasterize(dense, verts68, hw)
            self.rasterizer.mode = keep_mode
            diff = float(np.mean(np.logical_xor(mask_chain > 0, mask_tri > 0)))
            self.logger.info("rasterizer verification: chain-vs-triangles XOR=%.3f%%", diff * 100.0)
            if diff > 0.02:
                self.logger.warning("chain rasterizer mismatch; switching to per-triangle fill")
                self.rasterizer.mode = "triangles"
                return False
            return True
        except Exception as exc:
            self.logger.warning("rasterizer verification failed (%s); switching to per-triangle fill", exc)
            try:
                self.rasterizer.mode = "triangles"
            except Exception:
                pass
            return False

    @staticmethod
    def _mask_contour(mask: np.ndarray, loops: List[np.ndarray]) -> Optional[np.ndarray]:
        """Silhouette polygon of the final mask (largest external contour)."""
        if not mask.any():
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0.0
        for cnt in contours:
            area = abs(float(cv2.contourArea(cnt)))
            if area > best_area:
                best_area, best = area, cnt
        if best is None or best_area < 3.0:
            if loops:
                return np.asarray(loops[0], dtype=np.float32)
            return None
        return best.reshape(-1, 2).astype(np.float32)

    def _fallback_result(
        self,
        fid: int,
        frame: Optional[np.ndarray],
        reason: str,
        qs: Optional[QualityScores] = None,
        occ_ratio: Optional[float] = None,
        sec_ratio: Optional[float] = None,
        blur: float = 0.0,
    ) -> FrameResult:
        """Build the fallback output: previous stable -> damped prediction -> invalid."""
        assert self.rasterizer is not None
        tracker = self.tracker
        assert tracker is not None
        tracker.lost_count += 1
        tracker.low_quality_streak += 1
        prev = self.prev_accepted
        h = self.height if hasattr(self, "height") else (frame.shape[0] if frame is not None else 0)
        w = self.width if hasattr(self, "width") else (frame.shape[1] if frame is not None else 0)

        def base_record(source_frame: Optional[int], invalid: bool, conf: float) -> Dict[str, Any]:
            return {
                "frame_id": fid,
                "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                "scale": 1.0, "translation": [0.0, 0.0],
                "confidence": float(conf),
                "occlusion_ratio": float(occ_ratio if occ_ratio is not None else 0.0),
                "fallback_used": True,
                "fallback_reason": reason,
                "face_bbox": [0.0, 0.0, 0.0, 0.0],
                "detector_confidence": float(tracker.det_score_carry),
                "landmark_confidence": 0.0,
                "landmark_count": 68,
                "tracking_quality": 0.0,
                "mask_area": 0,
                "reference_baseline_used": bool(self.baseline is not None and self.baseline.available),
                "previous_frame_used": source_frame is not None,
                "interpolated": False,
                "invalid": bool(invalid),
                "reduced_update": False,
                "reinitialized": False,
                "sec_mask_ratio": (float(sec_ratio) if sec_ratio is not None else None),
                "out_of_frame_ratio": float(self._last_raw_oof),
                "blur_score": float(blur),
                "face_size_px": 0.0,
                "face_center": [0.0, 0.0],
                "rasterizer": "none",
                "fallback_source_frame": source_frame,
                "param_vector": None,
                "roi_box": None,
                "scale_baseline_px": self.scale_baseline_px,
            }

        if prev is None:
            self.stats["invalid"] += 1
            if frame is not None and h and w:
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                mask = np.zeros((max(h, 1), max(w, 1)), dtype=np.uint8)
            verts68 = np.zeros((3, 68), dtype=np.float32)
            record = base_record(None, True, 0.0)
            record["rasterizer"] = "none"
            return FrameResult(fid, mask, verts68, None, record,
                               {"track_bbox": None, "raster_mode": "none"})

        lost = tracker.lost_count
        hold = int(self.args.max_hold_frames)
        predict = int(self.args.max_predict_frames)
        invalidate = int(self.args.invalidate_after)
        if lost > hold + predict and lost > invalidate:
            self.stats["invalid"] += 1
            mask = np.zeros((h, w), dtype=np.uint8)
            verts68 = np.zeros((3, 68), dtype=np.float32)
            record = base_record(None, True, 0.0)
            return FrameResult(fid, mask, verts68, None, record,
                               {"track_bbox": None, "raster_mode": "none"})

        if lost <= hold or prev.v_param is None:
            param_used = np.asarray(prev.param, np.float64)
            mode = "hold"
        else:
            k = min(lost - hold, predict) if lost <= hold + predict else 0
            if k > 0:
                damp_sum = sum(0.9 ** i for i in range(1, k + 1))
                delta = np.clip(np.asarray(prev.v_param, np.float64) * damp_sum, -60.0, 60.0)
                param_used = np.asarray(prev.param, np.float64) + delta
                mode = "predict"
            else:
                param_used = np.asarray(prev.param, np.float64)
                mode = "hold"
        param_used = param_used.astype(np.float32)
        verts68 = self.adapter.recon_sparse(param_used, prev.roi_box) if self.adapter else np.zeros((3, 68), np.float32)
        dense = self.adapter.recon_dense(param_used, prev.roi_box) if self.adapter else None
        if dense is not None and self.rasterizer is not None:
            mask, loops, info = self.rasterizer.rasterize(dense, verts68, (h, w))
        else:
            mask, loops, info = np.zeros((h, w), np.uint8), [], {"mode": "none"}
        contour = self._mask_contour(mask, loops)
        conf = 0.0 if mode == "hold" else float(max(0.0, (qs.confidence if qs else prev.quality) * 0.5))
        record = base_record(prev.frame_id, False, conf)
        pose = pose_angles_deg(param_used)
        centroid, face_size, lm_bbox = face_metrics(verts68)
        scale = 1.0 if not self.scale_baseline_px else float(face_size / max(self.scale_baseline_px, 1e-6))
        record.update({
            "yaw": float(pose[0]), "pitch": float(pose[1]), "roll": float(pose[2]),
            "scale": float(scale),
            "translation": [float(centroid[0]), float(centroid[1])],
            "face_bbox": [float(v) for v in lm_bbox],
            "landmark_confidence": float((qs.components.get("landmark", 0.0) if qs else 0.0)),
            "tracking_quality": float(qs.confidence if qs else 0.0),
            "mask_area": int(np.count_nonzero(mask)),
            "face_size_px": float(face_size),
            "face_center": [float(centroid[0]), float(centroid[1])],
            "rasterizer": str(info.get("mode", "none")),
            "fallback_source_frame": prev.frame_id if mode != "hold" else prev.frame_id,
            "param_vector": [float(v) for v in np.asarray(param_used, np.float64)],
            "roi_box": [float(v) for v in np.asarray(prev.roi_box, np.float64)],
        })
        debug = {"track_bbox": tracker.track_bbox, "raster_mode": f"{info.get('mode', '?')}:{mode}"}
        return FrameResult(fid, mask, verts68, contour, record, debug)

    # -- record assembly ----------------------------------------------------------

    def _make_record(
        self,
        fid: int,
        state: AcceptedState,
        qs: QualityScores,
        occ_ratio: Optional[float],
        sec_ratio: Optional[float],
        blur: float,
        fallback: bool,
        invalid: bool,
        reduced: bool,
        reinit: bool,
        raster_info: Dict[str, Any],
        source_frame: Optional[int],
    ) -> Dict[str, Any]:
        pose = state.pose_deg
        scale = 1.0 if not self.scale_baseline_px else float(state.face_size / max(self.scale_baseline_px, 1e-6))
        return {
            "frame_id": fid,
            "yaw": float(pose[0]), "pitch": float(pose[1]), "roll": float(pose[2]),
            "scale": float(scale),
            "translation": [float(state.centroid[0]), float(state.centroid[1])],
            "confidence": float(qs.confidence),
            "occlusion_ratio": float(occ_ratio if occ_ratio is not None else 0.0),
            "fallback_used": bool(fallback),
            "fallback_reason": None,
            "face_bbox": [float(v) for v in state.lm_bbox],
            "detector_confidence": float(state.det_score),
            "landmark_confidence": float(qs.components.get("landmark", 0.0)),
            "landmark_count": int(np.asarray(state.verts68).shape[-1]),
            "tracking_quality": float(qs.confidence),
            "mask_area": int(state.mask_area),
            "reference_baseline_used": bool(self.smoother is not None and self.smoother.baseline_used),
            "previous_frame_used": False,
            "interpolated": False,
            "invalid": bool(invalid),
            "reduced_update": bool(reduced),
            "reinitialized": bool(reinit),
            "sec_mask_ratio": (float(sec_ratio) if sec_ratio is not None else None),
            "out_of_frame_ratio": float(self._last_raw_oof),
            "blur_score": float(blur),
            "face_size_px": float(state.face_size),
            "face_center": [float(state.centroid[0]), float(state.centroid[1])],
            "detector_bbox": [float(v) for v in np.asarray(state.track_bbox, np.float64).reshape(-1)[:4]],
            "rasterizer": str(raster_info.get("mode", "none")),
            "fallback_source_frame": source_frame,
            "param_vector": [round(float(v), 6) for v in np.asarray(state.param, np.float64)],
            "roi_box": [round(float(v), 4) for v in np.asarray(state.roi_box, np.float64)],
            "scale_baseline_px": self.scale_baseline_px,
        }

    # -- logging -------------------------------------------------------------------

    def _progress_log(self, fid: int, result: FrameResult) -> None:
        self._log_n += 1
        self._window_n += 1
        every = max(1, int(self.args.log_every))
        if self._log_n % every != 0:
            return
        now = time.time()
        win_dt = max(now - self._window_t0, 1e-6)
        fps = self._window_n / win_dt
        self._window_t0 = now
        self._window_n = 0
        rec = result.record
        total = (self.end if self.end is not None else 0)
        self.logger.info(
            "[%d/%s] fps=%.1f conf=%.2f occ=%.2f yaw=%.1f pitch=%.1f fallback=%d skipped=%d",
            fid, str(total) if total else "?", fps,
            float(rec.get("confidence", 0.0)), float(rec.get("occlusion_ratio", 0.0)),
            float(rec.get("yaw", 0.0)), float(rec.get("pitch", 0.0)),
            self.stats["fallback"] + self.stats["invalid"], self.stats["skipped"],
        )

    def _log_summary(self) -> None:
        elapsed = time.time() - self._log_t0
        self.logger.info(
            "done: processed=%d skipped=%d fallback=%d invalid=%d recovered=%d reinit=%d "
            "detector_calls=%d fit_failures=%d decode_failures=%d elapsed=%.1fs avg_fps=%.1f",
            self.stats["processed"], self.stats["skipped"], self.stats["fallback"],
            self.stats["invalid"], self.stats["recovered"], self.stats["reinit"],
            self.stats["detections"], self.stats["fit_failures"], self.stats["decode_failures"],
            elapsed, self.stats["processed"] / max(elapsed, 1e-6),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command line parser (``--help`` documents every knob)."""
    p = argparse.ArgumentParser(
        prog="run_3ddfa_geometry.py",
        description=(
            "3DDFA_V2 geometry baseline: per-frame 3D face tracking and theoretical full-face "
            "binary mask generation for a single-face video (TASK 01)."
        ),
        epilog=(
            "Frame ranges are END-EXCLUSIVE: --start 1200 --end 1800 processes frames 1200..1799.\n"
            "Resume is ON by default: the completed prefix of [start, end) is detected, the decoder\n"
            "seeks straight to the first incomplete frame and every frame from there is processed in\n"
            "order (no mid-range skipping, so tracking state never loses sync with the video).\n"
            "A fully complete range exits immediately without loading the model (use --overwrite to\n"
            "force a full recomputation).\n\n"
            "examples:\n"
            "  python src/run_3ddfa_geometry.py --video input/dance_masked.mp4 "
            "--reference-dir input/reference_faces --output-dir output/3ddfa --device cuda:0\n"
            "  python src/run_3ddfa_geometry.py --video input/dance_masked.mp4 --start 1200 "
            "--end 1800 --debug --debug-every 30\n"
            "  python src/run_3ddfa_geometry.py --video input/dance_masked.mp4 "
            "--sec-mask-dir output/sec4b/mask --sec-occlusion-dir output/sec4b/occlusion\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- inputs / outputs -----------------------------------------------------
    g = p.add_argument_group("inputs and outputs")
    g.add_argument("--video", required=True,
                   help="path to the input video (single face assumed)")
    g.add_argument("--output-dir", "--output", dest="output_dir", default=os.path.join("output", "3ddfa"),
                   help="output root; face_mask/ landmarks/ debug/ params.jsonl go here")
    g.add_argument("--reference-dir",
                   help="directory of reference face images used to build the identity/shape "
                        "baseline (median/trimmed-mean of the 3DMM shape); when omitted or empty, "
                        "the video's own first frames are used instead")
    g.add_argument("--rebuild-reference", action="store_true",
                   help="ignore a cached reference_baseline.npz and rebuild it from --reference-dir")
    g.add_argument("--sec4b-dir", default=None,
                   help="optional SeC-4B root dir; auto-populates --sec-mask-dir and --sec-occlusion-dir")
    g.add_argument("--sec-mask-dir", default=None,
                   help="optional directory of per-frame SeC-4B face masks (6-digit ids). These "
                        "are read-only inputs used to damp updates; they never define the face shape")
    g.add_argument("--sec-occlusion-dir", default=None,
                   help="optional directory of per-frame SeC-4B occlusion masks (hand/microphone/...). "
                        "Raising occlusion_ratio lowers update weights and can trigger fallback")
    g.add_argument("--trajectory", default=None,
                   help="optional YOLO head trajectory (.npy) from Stage 0 to bypass internal detector calls")

    # -- 3DDFA_V2 -------------------------------------------------------------
    g = p.add_argument_group("3DDFA_V2 integration")
    g.add_argument("--tddfa-root", default=None,
                   help="path to the 3DDFA_V2 repository clone; when omitted, the TDDFA_ROOT "
                        "environment variable and ./3DDFA_V2 are tried")
    g.add_argument("--config", default=None,
                   help="path to the 3DDFA config yml (default: <tddfa-root>/configs/mb1_120x120.yml)")
    g.add_argument("--checkpoint", default=None,
                   help="path to the model checkpoint (default: checkpoint_fp from the config, "
                        "e.g. <tddfa-root>/weights/mb1_120x120.pth)")
    g.add_argument("--device", default="cuda:0",
                   help="torch device for 3DDFA_V2 (e.g. cuda:0, cpu)")
    g.add_argument("--onnx", action="store_true",
                   help="prefer the ONNX runtime path when the fork supports it")

    # -- frame range / resume --------------------------------------------------
    g = p.add_argument_group("frame range and resume")
    g.add_argument("--start", type=int, default=0,
                   help="first frame id to process (inclusive)")
    g.add_argument("--end", type=int, default=None,
                   help="last frame id to process, EXCLUSIVE: [start, end); default: end of the video")
    g.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="skip frames that are already complete (mask + landmarks + params record); "
                        "use --no-resume to recompute everything")
    g.add_argument("--overwrite", action="store_true",
                   help="force a full recomputation of the requested range (implies --no-resume)")

    # -- tracking quality ------------------------------------------------------
    g = p.add_argument_group("tracking quality thresholds")
    g.add_argument("--confidence-threshold", type=float, default=0.45,
                   help="minimum composed quality for a frame to be accepted")
    g.add_argument("--occlusion-threshold", type=float, default=0.45,
                   help="occlusion ratio above which identity/expression updates are heavily "
                        "damped (occ_severe)")
    g.add_argument("--mid-occlusion-threshold", type=float, default=0.20,
                   help="occlusion ratio above which updates are moderately damped (occ_mid)")
    g.add_argument("--hard-occlusion-threshold", type=float, default=0.85,
                   help="occlusion ratio above which the frame is hard-rejected (fallback)")
    g.add_argument("--max-pose-jump-deg", type=float, default=70.0,
                   help="maximum per-frame pose change; beyond it the fit is rejected")
    g.add_argument("--min-scale-ratio", type=float, default=0.65,
                   help="minimum accepted face-size ratio against the previous frame")
    g.add_argument("--max-scale-ratio", type=float, default=1.55,
                   help="maximum accepted face-size ratio against the previous frame")
    g.add_argument("--max-shift-ratio", type=float, default=0.40,
                   help="maximum accepted centroid shift (in face-size units)")
    g.add_argument("--max-landmark-jump-ratio", type=float, default=0.45,
                   help="maximum accepted mean landmark jump (in face-size units)")
    g.add_argument("--blur-threshold", type=float, default=BLUR_FULL_CREDIT,
                   help="blur score at which the blur quality bottoms out")
    g.add_argument("--out-of-frame-tolerance", type=float, default=0.35,
                   help="landmark-bbox out-of-frame ratio tolerated before the score drops")
    g.add_argument("--min-mask-area", type=int, default=MIN_MASK_AREA,
                   help="theoretical-face area (px) below which a frame is rejected outright")
    g.add_argument("--detect-interval", type=int, default=15,
                   help="run the face detector at most once every N frames (plus on loss/low quality)")
    g.add_argument("--low-quality-trigger", type=int, default=5,
                   help="consecutive low-quality frames that force an extra detection")
    g.add_argument("--reinit-after-lost", type=int, default=REINIT_LOST_FRAMES_DEFAULT,
                   help="lost frames after which a fresh detector box re-initializes the tracker")

    # -- smoothing ---------------------------------------------------------------
    g = p.add_argument_group("temporal smoothing")
    g.add_argument("--shape-measurement-weight", "--shape-blend", dest="shape_measurement_weight",
                   type=float, default=0.30, metavar="W",
                   help="identity/shape update weight: shape = baseline + W*(measurement - baseline); "
                        "0 locks the shape to the reference baseline, 1 fully adopts the per-frame "
                        "measurement (--shape-blend is kept as a compatibility alias)")
    g.add_argument("--smooth-min-cutoff", type=float, default=1.2,
                   help="One Euro min cutoff (Hz) for pose/expression; higher follows faster")
    g.add_argument("--smooth-beta", type=float, default=1.0,
                   help="One Euro speed coefficient multiplier; higher reduces lag on fast motion")
    g.add_argument("--no-smooth", action="store_true",
                   help="disable temporal smoothing entirely (debugging only; the mask becomes jittery)")

    # -- fallback ------------------------------------------------------------------
    g = p.add_argument_group("fallback / occlusion losses")
    g.add_argument("--max-hold-frames", type=int, default=12,
                   help="frames to hold the previous stable geometry after a failure")
    g.add_argument("--max-predict-frames", type=int, default=24,
                   help="frames to extrapolate motion (damped constant velocity) after the hold period")
    g.add_argument("--invalidate-after", type=int, default=90,
                   help="loss length after which frames are written as explicit invalid (empty mask)")

    # -- performance / diagnostics ---------------------------------------------------
    g = p.add_argument_group("performance and diagnostics")
    g.add_argument("--rasterizer", choices=("auto", "chain", "triangles", "hull"), default="auto",
                   help="mesh rasterization strategy: chain=boundary-edge fill (fast), "
                        "triangles=per-triangle fill (slow reference), auto=chain with runtime "
                        "verification, hull=68-point convex hull (degraded, no mesh fill)")
    g.add_argument("--save-contour", action=argparse.BooleanOptionalAction, default=True,
                   help="additionally write the {frame}_contour.npy silhouette polygon per frame")
    g.add_argument("--debug", action="store_true",
                   help="write debug overlay images (mask / landmarks / bbox / occlusion) every "
                        "--debug-every frames into debug/")
    g.add_argument("--debug-every", type=int, default=30,
                   help="write one debug overlay every N processed frames")
    g.add_argument("--log-every", type=int, default=100,
                   help="progress log line every N processed frames")
    g.add_argument("--empty-cache-every", type=int, default=200,
                   help="call torch.cuda.empty_cache() every N frames (0 disables)")
    g.add_argument("--params-rewrite-every", type=int, default=2000,
                   help="atomically deduplicate params.jsonl every N processed frames (0 disables)")
    g.add_argument("--max-decode-failures", type=int, default=10,
                   help="consecutive undecodable frames that abort the run (hardware/codec problem)")
    g.add_argument("--verbose", action="store_true",
                   help="enable DEBUG logging")

    return p


def setup_logging(verbose: bool) -> None:
    """Configure the root logger (stderr, with a compact per-frame timestamp)."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(threadName)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    LOG.setLevel(level)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns the process exit code (0 ok, 2 fatal, 130 interrupted)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(bool(args.verbose))
    LOG.info("%s v%s starting (python %s, numpy %s, cv2 %s)",
             LOG_NAME, __version__, sys.version.split()[0], np.__version__, cv2.__version__)
    try:
        runner = Runner(args, logger=LOG)
        return runner.run()
    except KeyboardInterrupt:
        LOG.warning("interrupted by user; progress is saved — rerun the same command to resume "
                    "(--resume is on by default)")
        return 130
    except FatalError as exc:
        LOG.error("fatal: %s", exc)
        return 2
    except Exception:  # pragma: no cover - unexpected bugs
        LOG.exception("unhandled error; this is a bug — please report it with the log above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
