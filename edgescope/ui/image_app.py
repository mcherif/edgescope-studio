from __future__ import annotations

import gradio as gr
import cv2
import numpy as np

from edgescope.engine.detector import RTMDetDetector, Detection
from edgescope.engine.segmentor import DummySegmentor
from edgescope.engine.segmenter import SamSegmenter, MaskResult
from edgescope.config import PROJECT_ROOT, load_classes_config

# RTMDet config + checkpoint
RTMDET_CONFIG = PROJECT_ROOT / "rtmdet" / "rtmdet_tiny_8xb32-300e_coco.py"
RTMDET_CHECKPOINT = PROJECT_ROOT / "rtmdet" / "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"

# SAM checkpoint (prefer the official filename if present)
_sam_dir = PROJECT_ROOT / "sam"
_sam_candidates = [
    _sam_dir / "sam_vit_b_01ec64.pth",  # official ViT-B
    _sam_dir / "sam_vit_b.pth",
]
SAM_CHECKPOINT = next((p for p in _sam_candidates if p.exists()), _sam_candidates[0])

KEEP_CLASSES, CLASS_ALIASES = load_classes_config()

detector = RTMDetDetector(
    config_path=str(RTMDET_CONFIG),
    checkpoint_path=str(RTMDET_CHECKPOINT),
    device=None,          # auto cuda/cpu
    score_threshold=0.3,  # default, will be overridden by slider
)
segmentor = DummySegmentor()
sam_segmenter = SamSegmenter(
    checkpoint_path=str(SAM_CHECKPOINT),
    model_type="vit_b",
    device=None,  # auto cuda/cpu
)


def run_pipeline(image: np.ndarray, conf: float, show_masks: bool) -> np.ndarray:
    """
    Gradio callback: takes an RGB image, runs RTMDet with given confidence,
    optionally overlays SAM masks, and returns an annotated RGB image.
    """
    if image is None:
        return None

    # Gradio gives RGB uint8
    detector.score_threshold = conf
    raw_detections = detector.detect(image)

    # remap & filter using config/classes.yaml
    filtered: list[Detection] = []
    for d in raw_detections:
        label = CLASS_ALIASES.get(d.label, d.label)  # apply alias if any
        if label not in KEEP_CLASSES:
            continue

        filtered.append(
            Detection(
                x1=d.x1,
                y1=d.y1,
                x2=d.x2,
                y2=d.y2,
                score=d.score,
                label_id=d.label_id,
                label=label,
            )
        )

    # Filter tiny boxes (3% of short side)
    H, W, _ = image.shape
    min_side = 0.03 * min(H, W)
    detections: list[Detection] = []
    for d in filtered:
        if (d.x2 - d.x1) < min_side or (d.y2 - d.y1) < min_side:
            continue
        detections.append(d)

    # Draw on BGR copy (OpenCV drawing)
    vis_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Optional SAM masks overlay
    if show_masks and detections:
        target_labels = ["person", "chair", "monitor", "laptop"]
        mask_results: list[MaskResult] = sam_segmenter.segment(image, detections, target_labels)

        overlay = vis_bgr.copy()
        palette = [
            (0, 255, 0),
            (0, 200, 255),
            (255, 200, 0),
            (200, 0, 255),
            (255, 128, 0),
        ]
        for idx, mr in enumerate(mask_results):
            color = palette[idx % len(palette)]
            overlay[mr.mask] = color
        alpha = 0.4
        vis_bgr = cv2.addWeighted(overlay, alpha, vis_bgr, 1 - alpha, 0)

    # Boxes + labels
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

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return vis_rgb


def create_app() -> gr.Blocks:
    with gr.Blocks(title="EdgeScope Studio - Image") as demo:
        gr.Markdown("# EdgeScope Studio\nImage demo with RTMDet + SAM.")

        with gr.Row():
            inp = gr.Image(type="numpy", label="Input image")
            out = gr.Image(type="numpy", label="Output (detections + masks)")

        conf_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.3,
            step=0.01,
            label="Confidence threshold",
        )
        show_masks = gr.Checkbox(label="Show SAM masks", value=True)

        run_btn = gr.Button("Run pipeline")

        run_btn.click(
            fn=run_pipeline,
            inputs=[inp, conf_slider, show_masks],
            outputs=out,
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()
