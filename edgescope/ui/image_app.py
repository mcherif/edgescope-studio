from __future__ import annotations

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

KEEP_CLASSES, CLASS_ALIASES, SAM_TARGETS_DEFAULT = load_classes_config()


ICON_MAP = {
    "person": "🧍",
    "chair": "💺",
    "monitor": "🖥️",
    "laptop": "💻",
    "keyboard": "⌨️",
    "mouse": "🖱️",
    "cup": "☕",
    "bottle": "🥤",
    "backpack": "🎒",
    "book": "📚",
    "clock": "⏰",
    "tv": "📺",
    "screen": "🖥️",
}

_PALETTE = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 200, 0),
    (200, 0, 255),
    (255, 128, 0),
    (120, 180, 255),
    (255, 105, 180),
    (180, 255, 120),
    (50, 50, 255),
    (255, 50, 50),
    (50, 255, 50),
    (255, 0, 125),
    (0, 125, 255),
    (125, 0, 255),
    (0, 255, 125),
    (125, 255, 0),
]
CLASS_COLORS = {label: _PALETTE[idx % len(_PALETTE)] for idx, label in enumerate(sorted(KEEP_CLASSES))}


def _label_with_icon(label: str) -> str:
    icon = ICON_MAP.get(label, "")
    return f"{icon} {label}" if icon else label


def _strip_icon(label_with_icon: str) -> str:
    # Drop the first token (emoji) if present; otherwise return as-is.
    parts = label_with_icon.strip().split(" ", 1)
    return parts[1] if len(parts) == 2 else parts[0]


def _color_for_label(label: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(label, (0, 255, 0))


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


def run_pipeline(
    image: np.ndarray,
    conf: float,
    show_boxes: bool,
    show_masks: bool,
    sam_targets_display: list[str],
) -> np.ndarray:
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
        targets = [_strip_icon(t) for t in (sam_targets_display or [])]
        targets = [t for t in targets if t in KEEP_CLASSES]

        if targets:
            mask_results: list[MaskResult] = sam_segmenter.segment(image, detections, targets)

            overlay = vis_bgr.copy()
            for mr in mask_results:
                color = _color_for_label(mr.detection.label)
                overlay[mr.mask] = color
            alpha = 0.4
            vis_bgr = cv2.addWeighted(overlay, alpha, vis_bgr, 1 - alpha, 0)

    # Boxes + labels
    if show_boxes:
        for det in detections:
            pt1 = int(det.x1), int(det.y1)
            pt2 = int(det.x2), int(det.y2)
            color = _color_for_label(det.label)
            cv2.rectangle(vis_bgr, pt1, pt2, color, 2)
            label_text = f"{det.label} {det.score:.2f}"
            cv2.putText(
                vis_bgr,
                label_text,
                (pt1[0], max(0, pt1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return vis_rgb


def create_app() -> gr.Blocks:
    def apply_add_all(add_all: bool, manual_selection: list[str]):
        # When enabled, fill the selection with every class; otherwise restore last manual picks.
        new_value = sam_choices if add_all else manual_selection
        return gr.update(value=new_value)

    def remember_manual_selection(current: list[str], add_all: bool):
        # Only store manual selections when "add all" is off so we can restore them later.
        if add_all:
            return gr.update()  # keep previous manual selection
        return current

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
        show_boxes = gr.Checkbox(label="Show boxes + labels", value=True)
        show_masks = gr.Checkbox(label="Show SAM masks", value=True)
        sam_choices = [_label_with_icon(c) for c in sorted(KEEP_CLASSES)]
        sam_default = [_label_with_icon(c) for c in SAM_TARGETS_DEFAULT]
        manual_sam_targets = gr.State(value=sam_default)
        sam_add_all = gr.Checkbox(
            label="Add all SAM classes",
            value=False,
            info="Temporarily select every class for segmentation masks.",
        )
        sam_targets = gr.CheckboxGroup(
            choices=sam_choices,
            value=sam_default,
            label="SAM target classes",
            info="Labels used to prompt SAM for masks.",
        )

        run_btn = gr.Button("Run pipeline")

        inp.change(
            fn=run_pipeline,
            inputs=[inp, conf_slider, show_boxes, show_masks, sam_targets],
            outputs=out,
        )
        conf_slider.change(
            fn=run_pipeline,
            inputs=[inp, conf_slider, show_boxes, show_masks, sam_targets],
            outputs=out,
        )
        sam_targets.change(
            fn=remember_manual_selection,
            inputs=[sam_targets, sam_add_all],
            outputs=manual_sam_targets,
        ).then(
            fn=run_pipeline,
            inputs=[inp, conf_slider, show_boxes, show_masks, sam_targets],
            outputs=out,
        )
        sam_add_all.change(
            fn=apply_add_all,
            inputs=[sam_add_all, manual_sam_targets],
            outputs=sam_targets,
        ).then(
            fn=run_pipeline,
            inputs=[inp, conf_slider, show_boxes, show_masks, sam_targets],
            outputs=out,
        )
        show_boxes.change(
            fn=run_pipeline,
            inputs=[inp, conf_slider, show_boxes, show_masks, sam_targets],
            outputs=out,
        )
        show_masks.change(
            fn=run_pipeline,
            inputs=[inp, conf_slider, show_boxes, show_masks, sam_targets],
            outputs=out,
        )
        run_btn.click(
            fn=run_pipeline,
            inputs=[inp, conf_slider, show_boxes, show_masks, sam_targets],
            outputs=out,
        )
        demo.load(
            fn=run_pipeline,
            inputs=[inp, conf_slider, show_boxes, show_masks, sam_targets],
            outputs=out,
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()
