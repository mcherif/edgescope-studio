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
    label: str


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run object detection on a single RGB image.

        Args:
            image: numpy array of shape (H, W, 3), dtype uint8, RGB.

        Returns:
            List of Detection instances.
        """
        ...


class DummyDetector:
    """
    Temporary implementation so the rest of the app can run
    before we plug in YOLOX.
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
                label="dummy",
            )
        ]
