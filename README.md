**EdgeScope Studio** is a local-first computer vision lab for rapidly prototyping **detection + segmentation** pipelines on images (and later videos), with everything running **on your own machine, offline**.

The core idea:

> Load your images → run a permissive detector (RTMDet Tiny on COCO) + SAM → inspect boxes & masks → iterate on thresholds, models, and logic without touching the cloud.

This is designed as a **general CV tool**, but with a strong focus on **on-device and privacy-preserving use cases** (e.g. ergonomics / digital wellbeing, industrial inspection, etc.).

## What’s implemented

- Image demo with **RTMDet Tiny (COCO)** for boxes + labels.
- **Segment Anything (SAM ViT-B)** turns those boxes into masks; toggleable in the UI.
- Class whitelist + aliases in `config/classes.yaml` (single source of truth).
- Gradio UI (`scripts/run_image_app.py`) with confidence slider and “Show SAM masks”.

## Setup

> Use Python 3.10 and the provided requirements. CUDA builds are pinned; adjust if needed.

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
Open `http://127.0.0.1:7860`, upload an image, set confidence, and toggle “Show SAM masks”.

## Notes

- Detector is COCO-trained; class filtering/aliasing is controlled by `config/classes.yaml`.
- SAM is class-agnostic; we prompt it with RTMDet boxes (optional target labels in code).
- If the default port is busy, change `server_port` in `scripts/run_image_app.py`.
