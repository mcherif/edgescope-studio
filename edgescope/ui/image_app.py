from __future__ import annotations

from typing import Tuple

import gradio as gr
import cv2
import numpy as np

from edgescope.engine.detector import DummyDetector
from edgescope.engine.segmentor import DummySegmentor


detector = DummyDetector()
segmentor = DummySegmentor()


def run_pipeline(image: np.ndarray) -> np.ndarray:
    # Gradio gives image as RGB uint8 already
    detections = detector.detect(image)
    masks = segmentor.segment(image, detections)

    # Draw boxes on a copy
    vis = image.copy()
    for det in detections:
        pt1 = int(det.x1), int(det.y1)
        pt2 = int(det.x2), int(det.y2)
        cv2.rectangle(vis, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{det.label} {det.score:.2f}",
            (pt1[0], max(0, pt1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return vis


def create_app() -> gr.Blocks:
    with gr.Blocks(title="EdgeScope Studio - Image") as demo:
        gr.Markdown("# EdgeScope Studio\nImage demo (dummy detector for now).")

        with gr.Row():
            inp = gr.Image(type="numpy", label="Input image")
            out = gr.Image(type="numpy", label="Output (detections)")

        run_btn = gr.Button("Run pipeline")

        run_btn.click(fn=run_pipeline, inputs=inp, outputs=out)

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()
