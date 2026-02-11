"""
Warm up RVM ONNX and prove recurrent-state loop works.
Run:
  python scripts/warmup_rvm.py --device cuda --iters 20 --size 512 --downsample 0.25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--downsample", type=float, default=0.25)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "models" / "rvm_mobilenetv3_fp32.onnx"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: python scripts/setup_video.py")
        return 2

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else [
        "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), providers=providers)

    used = sess.get_providers()[0]
    print("Using provider:", used)

    # src: [B,C,H,W] RGB normalized 0..1
    H = W = args.size
    src = np.random.rand(1, 3, H, W).astype(np.float32)

    # RVM spec: initial recurrent states are zeros [1,1,1,1], dtype matches model precision. :contentReference[oaicite:3]{index=3}
    rec = [np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(4)]

    # downsample_ratio must be FP32. :contentReference[oaicite:4]{index=4}
    downsample_ratio = np.asarray([args.downsample], dtype=np.float32)

    # Warm-up loop
    lat_ms = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        fgr, pha, r1o, r2o, r3o, r4o = sess.run(
            None,
            {
                "src": src,
                "r1i": rec[0],
                "r2i": rec[1],
                "r3i": rec[2],
                "r4i": rec[3],
                "downsample_ratio": downsample_ratio,
            },
        )
        dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt)

        rec = [r1o, r2o, r3o, r4o]
        if i in (0, args.iters - 1):
            print(
                f"iter {i}: pha={pha.shape} r1o={r1o.shape} r4o={r4o.shape} latency={dt:.2f}ms")

    lat_ms = np.asarray(lat_ms, dtype=np.float32)
    print("\nLatency ms:")
    print(f"  mean: {lat_ms.mean():.2f}")
    print(f"  p50 : {np.percentile(lat_ms, 50):.2f}")
    print(f"  p95 : {np.percentile(lat_ms, 95):.2f}")
    print(f"  p99 : {np.percentile(lat_ms, 99):.2f}")

    burn_in = min(3, len(lat_ms))
    steady = lat_ms[burn_in:]
    print("\nSteady-state (excluding first %d):" % burn_in)
    print(f"  mean: {steady.mean():.2f}")
    print(f"  p50 : {np.percentile(steady, 50):.2f}")
    print(f"  p95 : {np.percentile(steady, 95):.2f}")
    print(f"  p99 : {np.percentile(steady, 99):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
