# VisoClean (No DFM) - Masked Face Restoration & Video Pre-DFM Fusion Pipeline

VisoClean (No DFM) is a high-performance, deterministic multi-stage computer vision pipeline designed to restore masked faces and produce seamless, artifact-free intermediate video for downstream facial restoration (such as VisoMaster / DeepFaceMerge).

## Key Features
- **Zero-Disk Streaming Architecture (`run_production_pipeline.py`)**: Eliminates tens of thousands of intermediate PNG frame writes by streaming packed bitmasks directly through RAM into multi-core synthesis workers.
- **Dynamic Head Tracking & Auto-Prompting (YOLOv8-Face + SeC-4B / SAM 2.1)**: Automatic person radar prescan that auto-infers mask positive anchors, seamlessly handling turnaround frames and camera cuts.
- **3DMM Geometric Face Mesh (`3DDFA_V2`)**: Tracks 3D facial topology and full theoretical face envelopes regularized against reference facial profiles.
- **2nd-Order Polynomial Lighting Adaptation**: Eliminates facial brightness flicker across frames with chronological exponential moving average (EMA) smoothing and Lambertian shading.
- **Anti-Lace Ear Strap Inpainting**: Telea-based deterministic outside-face ear strap removal with lace mask exclusion to avoid dirty donor pixel contamination.
- **Seamless Jawline Feathering (`transition_px`)**: Precision distance-transform blending strictly along the interface between face skin and neck repair zones, preventing mask outline leakage at cheekbones and nose bridge.

## Repository Structure
```
VisoClean_NoDFM/
├── run_pipeline.py               # Standard multi-stage orchestrator
├── run_production_pipeline.py    # 6207-frame high-performance zero-disk orchestrator
├── run.bat                       # Interactive Windows launcher
├── src/                          # Core synthesis, inpainting, and geometry algorithms
│   ├── build_skin_profile.py     # Reference face skin profile generator
│   ├── compose_pre_dfm.py        # Multi-layer pre-DFM compositor with jawline feathering
│   ├── reconstruct_base_skin.py  # Low-frequency base skin reconstructor
│   ├── restore_outside_region.py # DIS optical flow + Telea ear strap restoration
│   ├── run_3ddfa_geometry.py     # 3DDFA_V2 3D mesh tracker & landmark generator
│   ├── stream_synthesis_fusion.py# High-performance multi-threaded in-memory fusion engine
│   └── yolo_head_prescan.py      # Stage 0 YOLOv8-face high-speed head radar
├── sec/                          # SeC-4B and SAM 2.1 video segmentation & tracking
│   ├── sam2_head_tracker.py      # SAM 2.1 video segmentation tracker
│   ├── sec4b_tracker.py          # SeC-4B multi-segment temporal tracker
│   ├── sec4b_video_masks.py      # SeC-4B CLI entry point
│   ├── sec4b_postprocess.py      # Mask decomposition & skin thresholding
│   ├── sec4b_model.py            # SeC model loader & memory manager
│   └── sec4b_io.py               # Video source decoding & debug overlays
└── 3DDFA_V2/                     # 3DDFA_V2 3D face alignment submodule
```

## Quick Start

### 1. Requirements
- Windows 10/11 or Linux
- Python >= 3.10
- NVIDIA GPU with >= 8GB VRAM (CUDA 11.8+ / 12.1+)
- FFmpeg installed in system PATH

### 2. Model Weights Setup
Place your model weights into the respective directories (or configure environment variables):
- **YOLOv8-Face**: Set `YOLO_FACE_MODEL=path/to/yoloface_8n.onnx`
- **SAM 2.1**: Set `SAM2_CHECKPOINT=path/to/sam2.1_hiera_tiny.pt`
- **3DDFA_V2**: Download weights as described in `3DDFA_V2/weights/readme.md`
- **SeC-4B**: Download via Hugging Face (`OpenIXCLab/SeC-4B`)

### 3. Run Pipeline
```bash
# Production zero-disk pipeline (Recommended for full-length videos):
python run_production_pipeline.py --video "path/to/video.mp4" --output-dir "output"

# Standard multi-stage pipeline:
python run_pipeline.py --video "path/to/video.mp4" --reference-dir "path/to/reference_faces"
```

## License
MIT / Apache 2.0 (Check submodules for respective licenses).
