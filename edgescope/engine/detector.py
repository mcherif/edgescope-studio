from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol
import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label_id: int
    label: str


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run object detection on a single RGB image.

        Args:
            image: numpy array of shape (H, W, 3), dtype uint8, RGB.

        Returns:
            List of Detection instances (bbox + score + class info).
        """
        ...


class DummyDetector:
    """
    Temporary implementation so the rest of the app can run
    before we plug in a real detector.
    """

    def detect(self, image: np.ndarray) -> List[Detection]:
        h, w, _ = image.shape
        # single fake detection in the center
        return [
            Detection(
                x1=w * 0.3,
                y1=h * 0.3,
                x2=w * 0.7,
                y2=h * 0.7,
                score=0.9,
                label_id=0,
                label="dummy",
            )
        ]


class RTMDetDetector:
    """
    Wrapper around an RTMDet checkpoint from MMDetection.
    Converts the model outputs (list of per-class arrays or DetDataSample)
    into Detection objects.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str | None = None,
        score_threshold: float = 0.0,
    ):
        from mmdet.apis import init_detector
        import torch

        # Auto-select device if not given
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.class_names = self._extract_class_names()
        self.score_threshold = score_threshold

    def _extract_class_names(self) -> List[str]:
        # MMDet v3 prefers dataset_meta["classes"], but also support model.CLASSES
        meta = getattr(self.model, "dataset_meta", None) or {}
        classes = meta.get("classes", None)
        if classes is None and hasattr(self.model, "CLASSES"):
            classes = self.model.CLASSES
        return list(classes) if classes is not None else []

    def detect(self, image: np.ndarray) -> List[Detection]:
        from mmdet.apis import inference_detector
        import cv2

        # Expect RGB uint8, enforce
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected image of shape (H, W, 3), got {image.shape}")

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # MMDetection expects BGR ndarray when passing images directly
        image_bgr = np.ascontiguousarray(image[..., ::-1])

        result = inference_detector(self.model, image_bgr)
        detections: List[Detection] = []

        # Case 1: MMDet v3 DetDataSample with pred_instances
        if hasattr(result, "pred_instances"):
            pred = result.pred_instances
            bboxes = getattr(pred, "bboxes", [])
            scores = getattr(pred, "scores", [])
            labels = getattr(pred, "labels", [])

            for bbox, score, label_id in zip(bboxes, scores, labels):
                score_val = float(score)
                if score_val < self.score_threshold:
                    continue
                x1, y1, x2, y2 = (float(coord) for coord in bbox)
                label_int = int(label_id)
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        score=score_val,
                        label_id=label_int,
                        label=self._label_name(label_int),
                    )
                )
            return detections

        # Case 2: old-style list/tuple of per-class arrays
        if isinstance(result, (list, tuple)):
            for label_id, class_dets in enumerate(result):
                if class_dets is None:
                    continue
                for bbox in class_dets:
                    x1, y1, x2, y2, score = (float(v) for v in bbox)
                    if score < self.score_threshold:
                        continue
                    detections.append(
                        Detection(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            score=score,
                            label_id=label_id,
                            label=self._label_name(label_id),
                        )
                    )
            return detections

        raise TypeError(
            f"Unexpected output type from inference_detector: {type(result)}")

    def _label_name(self, label_id: int) -> str:
        if self.class_names and 0 <= label_id < len(self.class_names):
            return str(self.class_names[label_id])
        return str(label_id)
