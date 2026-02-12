from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class RVMResult:
    alpha: np.ndarray          # (H,W) float32 [0,1]
    latency_ms: float


class RVMPipeline:
    """
    RVM inference pipeline:
      BGR frame -> resize -> RGB -> normalize -> ORT -> alpha -> resize back
    Keeps recurrent states across frames.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 512,
        downsample_ratio: float = 0.25,
        device: str = "cuda",
    ):
        self.model_path = str(model_path)
        self.input_size = int(input_size)
        self.downsample_ratio = float(downsample_ratio)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else [
            "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            self.model_path, providers=providers)

        # Initial recurrent states per RVM ONNX spec
        self.rec: List[np.ndarray] = [
            np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(4)]
        self.downsample = np.asarray([self.downsample_ratio], dtype=np.float32)

        # Names (avoid hardcoding)
        ins = {i.name for i in self.session.get_inputs()}
        required = {"src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio"}
        missing = required - ins
        if missing:
            raise RuntimeError(f"Model missing required inputs: {missing}")

        print(
            f"OK: RVMPipeline ready: provider={self.session.get_providers()[0]}, input={self.input_size}x{self.input_size}")
        # Burn-in to avoid first-frame latency spike skewing metrics
        dummy = np.zeros(
            (1, 3, self.input_size, self.input_size), dtype=np.float32)
        for _ in range(3):
            _, _, r1o, r2o, r3o, r4o = self.session.run(
                None,
                {
                    "src": dummy,
                    "r1i": self.rec[0],
                    "r2i": self.rec[1],
                    "r3i": self.rec[2],
                    "r4i": self.rec[3],
                    "downsample_ratio": self.downsample,
                },
            )
            self.rec = [r1o, r2o, r3o, r4o]

    def reset(self, verbose: bool = True) -> None:
        self.rec = [np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(4)]
        if verbose:
            print("RVM recurrent states reset")

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Resize to model input
        resized = cv2.resize(
            frame_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0,1], NCHW float32
        src = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        return np.ascontiguousarray(src)

    def process_frame(self, frame_bgr: np.ndarray) -> RVMResult:
        h, w = frame_bgr.shape[:2]
        src = self._preprocess(frame_bgr)

        t0 = time.perf_counter()
        fgr, pha, r1o, r2o, r3o, r4o = self.session.run(
            None,
            {
                "src": src,
                "r1i": self.rec[0],
                "r2i": self.rec[1],
                "r3i": self.rec[2],
                "r4i": self.rec[3],
                "downsample_ratio": self.downsample,
            },
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Update recurrent states
        self.rec = [r1o, r2o, r3o, r4o]

        # pha: (1,1,512,512) -> (512,512)
        alpha_small = pha[0, 0].astype(np.float32)

        # Resize back to original frame size
        alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0.0, 1.0)

        return RVMResult(alpha=alpha, latency_ms=latency_ms)
