from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .detector import Detection


@dataclass
class MaskResult:
    detection: Detection
    mask: np.ndarray  # bool, shape (H, W)


class SamSegmenter:
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str | None = "cuda"):
        from segment_anything import sam_model_registry, SamPredictor
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.device = device

    def segment(
        self,
        image_rgb: np.ndarray,
        detections: Sequence[Detection],
        target_labels: Sequence[str] | None = None,
    ) -> List[MaskResult]:
        """
        Run SAM with RTMDet boxes as prompts. Returns masks in full image coords.
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)

        self.predictor.set_image(image_rgb)

        results: List[MaskResult] = []

        for det in detections:
            if target_labels is not None and det.label not in target_labels:
                continue

            box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)

            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=False,  # single best mask
            )
            mask = masks[0].astype(bool)  # (H, W)

            results.append(MaskResult(detection=det, mask=mask))

        return results
