**EdgeScope Studio** is a local-first computer vision lab for rapidly prototyping **detection + segmentation** pipelines on images (and later videos), with everything running **on your own machine, offline**.

The core idea:

> Load your images → run a permissive detector (e.g. YOLOX) + SAM → inspect boxes & masks → iterate on thresholds, models, and logic without touching the cloud.

This is designed as a **general CV tool**, but with a strong focus on **on-device and privacy-preserving use cases** (e.g. ergonomics / digital wellbeing, industrial inspection, etc.).

## Planned features

### v0 (MVP – images only)

- Load images from a local folder
- Run a **permissively licensed detector** (e.g. YOLOX) to get bounding boxes + classes
- Run **SAM** to obtain precise masks for selected detections
- Visualize boxes & masks in a simple UI (Gradio/Streamlit)
- Optionally export annotations (YOLO / simple JSON) for later training

### Next steps

- Video frame explorer (run the same pipeline on video frames)
- Live camera mode (low-FPS debug view)
- Model/config switching (different YOLOX variants, custom models)
- Per-project class & model configs

## Tech stack

- Python
- Detector: YOLO-style model with a **permissive license** (e.g. YOLOX – Apache-2.0)
- Segmentor: **Segment Anything (SAM)** – Apache-2.0
- UI: Gradio or Streamlit
- Image handling: PyTorch, OpenCV, NumPy

## Dev quickstart

> Note: this is early-stage, interfaces may change.

```bash
# create a virtual environment as you like, then:
pip install -r requirements.txt

python scripts/run_image_app.py
