from __future__ import annotations

import cv2
import numpy as np


class BackgroundBlur:
    """Composite sharp foreground over blurred background using alpha mask."""

    def __init__(self, blur_kernel: int = 51):
        # must be odd
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    def apply(self, frame_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Args:
            frame_bgr: (H,W,3) uint8
            alpha: (H,W) float32 in [0,1], where 1=foreground
        Returns:
            composited BGR uint8
        """
        blurred = cv2.GaussianBlur(
            frame_bgr, (self.blur_kernel, self.blur_kernel), 0)

        a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        a3 = np.repeat(a[..., None], 3, axis=2)

        out = frame_bgr.astype(np.float32) * a3 + \
            blurred.astype(np.float32) * (1.0 - a3)
        return np.clip(out, 0, 255).astype(np.uint8)
