#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sec4b_model.py - SeC-4B Model Loading, Environment Isolation, and Memory Diagnostics
====================================================================================
"""

import os
import sys
import gc
import psutil

# 1. Environment & C-Drive Protection (All caches redirected to J: drive)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7891")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7891")
os.environ.setdefault("ALL_PROXY", "http://127.0.0.1:7891")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# Locate existing SeC-4B model weights on J: drive without duplicating 14GB files
candidate_hf_caches = [
    os.environ.get("HF_HOME", ""),
    os.path.join(BASE_DIR, "hf_cache"),
    r"J:\SEC_VideoMasks\hf_cache",
    r"J:\SEC\hf_cache",
    r"J:\SEC_Cutout\hf_cache",
]
resolved_hf_home = None
for cand in candidate_hf_caches:
    if cand and os.path.isdir(os.path.join(cand, "hub", "models--OpenIXCLab--SeC-4B")):
        resolved_hf_home = cand
        break
if not resolved_hf_home:
    resolved_hf_home = os.path.join(BASE_DIR, "hf_cache")

os.environ["HF_HOME"] = resolved_hf_home
os.environ.setdefault("TORCH_HOME", os.path.join(BASE_DIR, "torch_cache"))
os.environ.setdefault("UV_CACHE_DIR", os.path.join(BASE_DIR, "uv_cache"))
os.environ.setdefault("PIP_CACHE_DIR", os.path.join(BASE_DIR, "pip_cache"))
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Add SeC codebase to Python sys.path
sec_repo = os.path.join(BASE_DIR, "SeC")
if os.path.exists(sec_repo) and sec_repo not in sys.path:
    sys.path.insert(0, sec_repo)

import torch
from transformers import AutoTokenizer

try:
    from inference.configuration_sec import SeCConfig
    from inference.modeling_sec import SeCModel
except ImportError as e:
    raise ImportError(
        f"Failed to import SeC model from {sec_repo}. Please ensure the SeC repository is linked or present: {e}"
    )


def get_memory_stats() -> dict:
    """Returns current RAM and VRAM usage in GB."""
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024 ** 3)
    ram_total_gb = ram.total / (1024 ** 3)

    vram_alloc_gb = 0.0
    vram_res_gb = 0.0
    if torch.cuda.is_available():
        vram_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_res_gb = torch.cuda.memory_reserved() / (1024 ** 3)

    return {
        "ram_used_gb": round(ram_used_gb, 2),
        "ram_total_gb": round(ram_total_gb, 2),
        "vram_alloc_gb": round(vram_alloc_gb, 2),
        "vram_reserved_gb": round(vram_res_gb, 2),
    }


def cleanup_memory():
    """Forces CPU and GPU garbage collection and empty CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_sec4b(model_path: str = "OpenIXCLab/SeC-4B", device: str = "cuda"):
    """
    Load SeC-4B model and tokenizer with local cache priority and zero duplicate disk usage.
    """
    hf_home = os.environ.get("HF_HOME", os.path.join(BASE_DIR, "hf_cache"))
    hub_cache_dir = os.path.join(hf_home, "hub", "models--OpenIXCLab--SeC-4B")
    if os.path.isdir(hub_cache_dir):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("=" * 70)
    print(f"[*] Loading SeC-4B Model: {model_path}")
    print(f"[*] Cache location (J: drive): {hf_home}")
    print(f"[*] Offline mode: {'ACTIVE (local zero-download loading)' if os.environ.get('HF_HUB_OFFLINE') == '1' else 'ONLINE'}")
    print(f"[*] Target device: {device}")
    print("=" * 70)

    config = None
    try:
        config = SeCConfig.from_pretrained(model_path, local_files_only=True)
        config.hydra_overrides_extra = ["++model.non_overlap_masks=false"]
    except Exception:
        try:
            config = SeCConfig.from_pretrained(model_path)
            config.hydra_overrides_extra = ["++model.non_overlap_masks=false"]
        except Exception as e:
            print(f"[!] Warning loading SeCConfig: {e}")

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[*] Inference precision: {dtype} (~7.4GB VRAM footprint)")

    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
    }
    if config is not None:
        kwargs["config"] = config

    try:
        model = SeCModel.from_pretrained(model_path, local_files_only=True, **kwargs)
    except Exception:
        model = SeCModel.from_pretrained(model_path, **kwargs)

    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    elif device != "cpu":
        model = model.to(device)

    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    stats = get_memory_stats()
    print(f"[*] SeC-4B Model ready. RAM used: {stats['ram_used_gb']} GB, VRAM: {stats['vram_alloc_gb']} GB")
    print("=" * 70)
    return model, tokenizer
