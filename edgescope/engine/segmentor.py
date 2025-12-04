from __future__ import annotations

from typing import Protocol, List
import numpy as np

from .detector import Detection


class Segmentor(Protocol):
    def segment(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Compute segmentation masks for detections.

        Args:
            image: (H, W, 3) RGB uint8.
            detections: list of Detection.

        Returns:
            masks: numpy array of shape (N, H, W), bool or uint8.
        """
        ...


class DummySegmentor:
    """
    Placeholder implementation: returns empty masks with no segmentation.
    """

    def segment(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        h, w, _ = image.shape
        n = len(detections)
        return np.zeros((n, h, w), dtype=bool)
