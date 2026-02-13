from __future__ import annotations
import time
import cv2
import numpy as np


class BackgroundBlur:
    def __init__(
        self,
        blur_scale: float = 0.5,
        sigma: float = 8.0,
        downscale: float | None = None,
        comp_mode: str = "soft",
        alpha_path: str = "optimized",
        alpha_thresh: int = 128,
        alpha_lo: int = 32,
        alpha_hi: int = 224,
        alpha_feather: int = 0,
    ):
        # `downscale` is kept for backward compatibility.
        if downscale is not None:
            blur_scale = downscale
        self.blur_scale = float(blur_scale)
        self.downscale = self.blur_scale
        self.sigma = float(sigma)
        self.comp_mode = comp_mode.lower()
        self.alpha_path = alpha_path.lower()
        self.alpha_thresh = int(alpha_thresh)
        self.alpha_lo = int(alpha_lo)
        self.alpha_hi = int(alpha_hi)
        self.alpha_feather = int(alpha_feather)
        self._buf_shape: tuple[int, int] | None = None
        self._a: np.ndarray | None = None
        self._a_blur: np.ndarray | None = None
        self._a3: np.ndarray | None = None
        self._inv_a3: np.ndarray | None = None
        self._tmp_fg: np.ndarray | None = None
        self._tmp_bg: np.ndarray | None = None
        self._out: np.ndarray | None = None

    def _ensure_buffers(self, h: int, w: int) -> None:
        if self._buf_shape == (h, w):
            return
        self._buf_shape = (h, w)
        self._a = np.empty((h, w), dtype=np.uint8)
        self._a_blur = np.empty((h, w), dtype=np.uint8)
        self._a3 = np.empty((h, w, 3), dtype=np.uint8)
        self._inv_a3 = np.empty((h, w, 3), dtype=np.uint8)
        self._tmp_fg = np.empty((h, w, 3), dtype=np.uint8)
        self._tmp_bg = np.empty((h, w, 3), dtype=np.uint8)
        self._out = np.empty((h, w, 3), dtype=np.uint8)

    def apply(
        self,
        frame_bgr: np.ndarray,
        alpha: np.ndarray,
        profile: dict | None = None,
    ) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self.alpha_path == "optimized":
            self._ensure_buffers(h, w)

        # Fast blur: downscale -> blur -> upscale
        s = self.blur_scale
        sw, sh = max(1, int(w * s)), max(1, int(h * s))
        t0 = time.perf_counter()
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        t1 = time.perf_counter()
        small_blur = cv2.GaussianBlur(
            small, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        t2 = time.perf_counter()
        blurred = cv2.resize(small_blur, (w, h),
                             interpolation=cv2.INTER_LINEAR)
        t3 = time.perf_counter()

        # Alpha to uint8 mask
        t4 = time.perf_counter()
        feather_ms = 0.0
        alpha_expand_ms = 0.0
        alpha_invert_ms = 0.0
        alpha_convert_ms = 0.0

        a3 = None
        mask = None
        fg_mask = None
        edge_mask = None
        roi = None

        if self.alpha_path == "legacy":
            a_src = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
            if self.alpha_feather > 0:
                k = self.alpha_feather
                if k % 2 == 0:
                    k += 1
                a_src = cv2.GaussianBlur(a_src, (k, k), sigmaX=0)
            if self.comp_mode == "soft":
                a3 = cv2.merge([a_src, a_src, a_src])
        else:
            cv2.convertScaleAbs(alpha, alpha=255.0, dst=self._a)
            t4b = time.perf_counter()
            alpha_convert_ms = (t4b - t4) * 1000.0

            a_src = self._a
            if self.alpha_feather > 0:
                k = self.alpha_feather
                if k % 2 == 0:
                    k += 1
                cv2.GaussianBlur(self._a, (k, k), sigmaX=0, dst=self._a_blur)
                a_src = self._a_blur
                feather_ms = (time.perf_counter() - t4b) * 1000.0

            if self.comp_mode == "soft":
                t_exp0 = time.perf_counter()
                cv2.cvtColor(a_src, cv2.COLOR_GRAY2BGR, dst=self._a3)
                t_exp1 = time.perf_counter()
                alpha_expand_ms = (t_exp1 - t_exp0) * 1000.0
                t_inv0 = time.perf_counter()
                cv2.bitwise_not(self._a3, dst=self._inv_a3)
                t_inv1 = time.perf_counter()
                alpha_invert_ms = (t_inv1 - t_inv0) * 1000.0
                a3 = self._a3

        if self.comp_mode == "hard":
            mask = (a_src > self.alpha_thresh).astype(np.uint8) * 255
        elif self.comp_mode == "trimap":
            lo = self.alpha_lo
            hi = self.alpha_hi
            if hi <= lo:
                hi = min(255, lo + 1)
            bg_mask = (a_src <= lo).astype(np.uint8) * 255
            fg_mask = (a_src >= hi).astype(np.uint8) * 255
            edge_mask = cv2.bitwise_not(cv2.bitwise_or(bg_mask, fg_mask))
            if cv2.countNonZero(edge_mask) > 0:
                x, y, rw, rh = cv2.boundingRect(edge_mask)
                roi = (x, y, rw, rh)
        t5 = time.perf_counter()

        t6 = time.perf_counter()
        if self.comp_mode == "hard":
            out = blurred.copy()
            cv2.copyTo(frame_bgr, mask, out)
        elif self.comp_mode == "trimap":
            out = blurred.copy()
            if fg_mask is not None:
                cv2.copyTo(frame_bgr, fg_mask, out)
            if edge_mask is not None and roi is not None:
                x, y, rw, rh = roi
                roi_frame = frame_bgr[y:y + rh, x:x + rw]
                roi_blur = blurred[y:y + rh, x:x + rw]
                roi_alpha = a_src[y:y + rh, x:x + rw]
                roi_edge = edge_mask[y:y + rh, x:x + rw]

                a3 = cv2.merge([roi_alpha, roi_alpha, roi_alpha])
                fg = cv2.multiply(roi_frame, a3, scale=1.0 / 255.0)
                bg = cv2.multiply(roi_blur, cv2.bitwise_not(a3), scale=1.0 / 255.0)
                blended = cv2.add(fg, bg)
                cv2.copyTo(blended, roi_edge, out[y:y + rh, x:x + rw])
        else:
            # out = frame*alpha + blurred*(1-alpha) using integer ops
            if self.alpha_path == "legacy":
                fg = cv2.multiply(frame_bgr, a3, scale=1.0 / 255.0)
                bg = cv2.multiply(blurred, cv2.bitwise_not(a3), scale=1.0 / 255.0)
                out = cv2.add(fg, bg)
            else:
                cv2.multiply(frame_bgr, a3, scale=1.0 / 255.0, dst=self._tmp_fg)
                cv2.multiply(blurred, self._inv_a3, scale=1.0 / 255.0, dst=self._tmp_bg)
                cv2.add(self._tmp_fg, self._tmp_bg, dst=self._out)
                out = self._out
        t7 = time.perf_counter()

        if profile is not None:
            profile["resize_down_ms"] = (t1 - t0) * 1000.0
            profile["blur_ms"] = (t2 - t1) * 1000.0
            profile["resize_up_ms"] = (t3 - t2) * 1000.0
            profile["mask_ms"] = (t5 - t4) * 1000.0
            profile["alpha_convert_ms"] = alpha_convert_ms
            profile["alpha_feather_ms"] = feather_ms
            profile["alpha_expand_ms"] = alpha_expand_ms
            profile["alpha_invert_ms"] = alpha_invert_ms
            profile["blend_ms"] = (t7 - t6) * 1000.0
        return out
