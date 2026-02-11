from __future__ import annotations
import cv2
import numpy as np


class BackgroundBlur:
    def __init__(self, downscale: float = 0.25, sigma: float = 8.0):
        self.downscale = float(downscale)
        self.sigma = float(sigma)

    def apply(self, frame_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        # Fast blur: downscale -> blur -> upscale
        s = self.downscale
        sw, sh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        small_blur = cv2.GaussianBlur(
            small, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        blurred = cv2.resize(small_blur, (w, h),
                             interpolation=cv2.INTER_LINEAR)

        # Alpha to uint8 mask
        a = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
        a3 = cv2.merge([a, a, a])  # (H,W,3) uint8

        # out = frame*alpha + blurred*(1-alpha) using integer ops
        fg = cv2.multiply(frame_bgr, a3, scale=1.0 / 255.0)
        bg = cv2.multiply(blurred, cv2.bitwise_not(a3), scale=1.0 / 255.0)
        out = cv2.add(fg, bg)
        return out
