**EdgeScope Studio** is a **local-first** computer vision lab for prototyping **image and video** pipelines on your own machine (offline).

It has two modes:

- **Image mode (RTMDet + SAM)**: general object detection + promptable segmentation for still images.
- **Video mode (RVM)**: real-time portrait matting + background blur with **temporal stability** (recurrent states).

The core idea:

> Load your images -> run a permissive detector (RTMDet Tiny on COCO) + SAM -> inspect boxes & masks -> iterate on thresholds, models, and logic without touching the cloud.

This is designed as a **general CV tool**, but with a strong focus on **on-device and privacy-preserving use cases** (e.g. ergonomics / digital wellbeing, industrial inspection, etc.).

For real-time portrait effects we use **Robust Video Matting (RVM)** (video-native, recurrent temporal states).
For general object segmentation in still images we use **detect -> segment (RTMDet + SAM)**.

RVM is a video matting model that keeps **recurrent state** across frames, which stabilizes edges and reduces flicker compared to per-frame-only inference. Those temporal states let the model "remember" motion and fine hair detail so the mask stays coherent over time.

![EdgeScope Studio UI](docs/assets/ui-snapshot-image-based-analysis.png)

## Quick start: Video mode (RVM background blur)

**Requirements:** Windows + NVIDIA GPU recommended (ONNX Runtime CUDA).  
**Model:** Robust Video Matting (RVM) MobileNetV3 ONNX (downloaded locally; not committed).

### 1) Setup (download/verify model, write IO metadata)
```bash
python scripts/setup_video.py
```

### 2) Run webcam demo (OpenCV window)
```bash
python scripts/run_video.py --device cuda --input-size 512 --downsample 0.25
# CPU fallback (slower)
python scripts/run_video.py --device cpu --input-size 512 --downsample 0.25
```

Controls: `q` quit, `b` toggle blur/debug, `r` reset temporal state

### 3) Windows capture backend (auto-select + caching)
Default backend is `auto` (probe + cache). Auto mode will:
- run a short blur-off probe (`msmf` vs `dshow`)
- select the best stable backend and cache the decision
- define stable as: passed health check (frames flowing / not stuck) + `warning_count == 0`

Override / reprobe:
```bash
python scripts/run_video.py --backend dshow ...
python scripts/run_video.py --backend msmf ...
python scripts/run_video.py --backend auto --reprobe ...
```

Probe directly:
```bash
python scripts/probe_backends.py --device cuda --input-size 512 --downsample 0.25 --width 1280 --height 720 --duration 20
```

### 4) Benchmark (headless)
```bash
# Blur ON (pin backend for reproducibility)
python scripts/benchmark_video.py --device cuda --backend dshow --input-size 512 --downsample 0.25 \
  --width 1280 --height 720 --duration 30 --blur \
  --out benchmarks/rvm_512_ds025_720p_blur.json

# Blur OFF
python scripts/benchmark_video.py --device cuda --backend dshow --input-size 512 --downsample 0.25 \
  --width 1280 --height 720 --duration 30 \
  --out benchmarks/rvm_512_ds025_720p_no_blur.json
```

Note: Results can vary by camera/driver/virtual-cam; run `scripts/probe_backends.py` to pick the best backend on your system.

Related scripts: `scripts/run_video.py`, `scripts/benchmark_video.py`, `scripts/compare_compositing_precision.py`.

## What's implemented

- Image demo with **RTMDet Tiny (COCO)** for boxes + labels.
- **Segment Anything (SAM ViT-B)** turns those boxes into masks; toggleable in the UI.
- Class whitelist + aliases in `config/classes.yaml` (single source of truth).
- Gradio UI (`scripts/run_image_app.py`) with confidence slider and "Show SAM masks".

## Setup

> Use Python 3.10 and the provided requirements. CUDA builds are pinned; adjust if needed.
> (Video mode uses ONNX Runtime; see the Video quick start above.)

1) Install deps (in your env, e.g. `conda activate edgescope-cuda`):
```bash
pip install -r requirements.txt
```

2) Download checkpoints:
- RTMDet: already in `rtmdet/` (`rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth`)
- SAM ViT-B: place at `sam/sam_vit_b_01ec64.pth` (fallback name `sam_vit_b.pth`)

3) Run the app:
```bash
python scripts/run_image_app.py
```
Open `http://127.0.0.1:7860`, upload an image, set confidence, and toggle "Show SAM masks".

## Notes

- Detector is COCO-trained; class filtering/aliasing is controlled by `config/classes.yaml`.
- SAM is class-agnostic; we prompt it with RTMDet boxes so we only segment detected objects (faster than running SAM across the whole image and it carries the detector's class labels).
- Why detection first: without detector boxes you'd have to run SAM's auto-segmentation over the whole image (more masks, higher latency) and then classify each mask with another model to know the class--slower and less reliable than detect -> segment.
- If the default port is busy, change `server_port` in `scripts/run_image_app.py`.
- Performance snapshot on RTX 4060 Ti (1024x1536 image): first-run init ~33.4s (detector) + ~3.1s (SAM); per-image after init ~1.5s detector + ~1.1s SAM.

## Benchmark snapshots (steady state, RTX 4060 Ti)

| Resolution        | Detections | Masks | Detector (s) | SAM (s) | Total (s) |
|-------------------|------------|-------|--------------|---------|-----------|
| 1024x1536 (orig)  | 39         | 39    | 0.746        | 0.764   | 1.511     |
| 640x426 (downscale) | 38        | 38    | 0.125        | 0.594   | 0.718     |

Notes: models are already loaded; numbers exclude one-time init.

## Video benchmarks (RTX 4060 Ti, Windows, 1280x720 capture)

Collected with `scripts/benchmark_video.py` (30s, `--input-size 512 --downsample 0.25`). Backend pinned to `dshow` (cached winner on this machine).

| Blur | FPS (mean)        | Total mean (ms)     | Total p95 (ms)      | Infer mean (ms)     | Infer p95 (ms)      | Comp mean (ms)     | Comp p95 (ms)     |
|------|-------------------|---------------------|---------------------|---------------------|---------------------|--------------------|-------------------|
| ON   | 21.285356455308616| 46.92536544799805   | 54.26680908203125   | 20.598535537719727  | 31.321360778808593  | 10.083098411560059 | 12.234370231628418|
| OFF  | 21.26155004604029 | 46.984901428222656  | 54.732181549072266  | 23.900632858276367  | 33.9217212677002    | 0.0                | 0.0               |

### How to reproduce
```bash
# Blur ON (backend pinned for reproducibility)
python scripts/benchmark_video.py --device cuda --backend dshow --input-size 512 --downsample 0.25 \
  --width 1280 --height 720 --duration 30 --blur \
  --out benchmarks/rvm_512_ds025_720p_blur.json

# Blur OFF
python scripts/benchmark_video.py --device cuda --backend dshow --input-size 512 --downsample 0.25 \
  --width 1280 --height 720 --duration 30 \
  --out benchmarks/rvm_512_ds025_720p_no_blur.json
```

**Known issues**
- Windows capture backend variability (camera/driver/virtual-cam). Use `scripts/probe_backends.py` and pin `--backend` when benchmarking.
- First-run warmup effects; the benchmark includes a warmup phase to reduce first-frame skew.

## Video roadmap (optional)

- Add a simple quality knob for blur (downscale/sigma) and document tradeoffs.
- Add a temporal-stability comparison mode (reset recurrent states) + jitter metric.
- Optional: integrate video into a UI (Gradio) once the core pipeline is rock-solid.
