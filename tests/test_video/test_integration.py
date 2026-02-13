from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


MODEL_PATH = Path("models/rvm_mobilenetv3_fp32.onnx")


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="RVM model not found; run scripts/setup_video.py to download.",
)
def test_rvm_pipeline_cpu_smoke() -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("cv2")

    from edgescope.video.pipeline import RVMPipeline

    # CPU-only path should run locally even without CUDA.
    pipeline = RVMPipeline(
        model_path=str(MODEL_PATH),
        device="cpu",
        input_size=512,
        downsample_ratio=0.25,
    )
    # Keep frame small (240p) so CPU inference is quick.
    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    res = pipeline.process_frame(frame)
    assert res.alpha.shape == (240, 320)
    assert res.alpha.dtype == np.float32
    assert res.alpha.min() >= 0.0
    assert res.alpha.max() <= 1.0
