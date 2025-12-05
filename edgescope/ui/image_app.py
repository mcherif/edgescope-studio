from __future__ import annotations
from pathlib import Path
from typing import Tuple

import gradio as gr
import cv2
import numpy as np

from edgescope.engine.detector import RTMDetDetector
from edgescope.engine.segmentor import DummySegmentor

# Locate project root (edgescope-studio/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# RTMDet config + checkpoint
RTMDET_CONFIG = PROJECT_ROOT / "rtmdet" / "rtmdet_tiny_8xb32-300e_coco.py"
RTMDET_CHECKPOINT = PROJECT_ROOT / "rtmdet" / \
    "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"


detector = RTMDetDetector(
    config_path=str(RTMDET_CONFIG),
    checkpoint_path=str(RTMDET_CHECKPOINT),
    device=None,          # auto cuda/cpu
    score_threshold=0.3,  # default, will be overridden by slider
)
segmentor = DummySegmentor()


def run_pipeline(image: np.ndarray, conf: float) -> np.ndarray:
    """
    Gradio callback: takes an RGB image, runs RTMDet with given confidence,
    and returns an annotated BGR->RGB image for display.
    """
    if image is None:
        return None

    # Gradio gives RGB uint8
    detector.score_threshold = conf
    detections = detector.detect(image)

    # Draw boxes on a BGR copy (OpenCV drawing)
    vis_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for det in detections:
        pt1 = int(det.x1), int(det.y1)
        pt2 = int(det.x2), int(det.y2)
        cv2.rectangle(vis_bgr, pt1, pt2, (0, 255, 0), 2)
        label_text = f"{det.label} {det.score:.2f}"
        cv2.putText(
            vis_bgr,
            label_text,
            (pt1[0], max(0, pt1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Convert back to RGB for Gradio
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return vis_rgb


def create_app() -> gr.Blocks:
    with gr.Blocks(title="EdgeScope Studio - Image") as demo:
        gr.Markdown("# EdgeScope Studio\nImage demo with RTMDet (COCO).")

        with gr.Row():
            inp = gr.Image(type="numpy", label="Input image")
            out = gr.Image(type="numpy", label="Output (detections)")

        conf_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.3,
            step=0.01,
            label="Confidence threshold",
        )

        run_btn = gr.Button("Run pipeline")

        run_btn.click(
            fn=run_pipeline,
            inputs=[inp, conf_slider],
            outputs=out,
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()
