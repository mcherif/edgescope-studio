#!/usr/bin/env python
"""
Headless benchmark for EdgeScope Video Mode (RVM + compositing).

Writes a JSON summary with:
- fps_mean
- latency_mean/p50/p95/p99 (end-to-end per-frame)
- stage timings: capture/infer/comp
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    from edgescope.video.compositing import BackgroundBlur
    from edgescope.video.pipeline import RVMPipeline
except ModuleNotFoundError as e:
    if e.name != "edgescope":
        raise
    # Add project root (parent of scripts/) to sys.path and retry.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from edgescope.video.compositing import BackgroundBlur
    from edgescope.video.pipeline import RVMPipeline


@dataclass
class Percentiles:
    mean: float
    p50: float
    p95: float
    p99: float


def pct(x: List[float]) -> Percentiles:
    if not x:
        return Percentiles(0.0, 0.0, 0.0, 0.0)
    a = np.array(x, dtype=np.float32)
    return Percentiles(
        mean=float(np.mean(a)),
        p50=float(np.percentile(a, 50)),
        p95=float(np.percentile(a, 95)),
        p99=float(np.percentile(a, 99)),
    )


def open_camera(index: int, width: int, height: int, fps: int, backend: str) -> cv2.VideoCapture:
    backend_map = {
        "any": 0,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }
    api = backend_map.get(backend.lower(), 0)
    cap = cv2.VideoCapture(index, api)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, int(fps))
    # Reduce buffering latency if supported
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Headless benchmark for RVM video mode")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--backend", type=str, default="dshow",
                    choices=["any", "dshow", "msmf"])

    ap.add_argument("--model", type=str,
                    default="models/rvm_mobilenetv3_fp32.onnx")
    ap.add_argument("--device", type=str, default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--downsample", type=float, default=0.25)

    ap.add_argument("--blur", action="store_true",
                    help="Enable background blur compositing")
    ap.add_argument("--blur-downscale", type=float, default=0.25,
                    help="Downscale used for fast blur (e.g. 0.25)")
    ap.add_argument("--blur-sigma", type=float, default=8.0,
                    help="Gaussian sigma used for blur")

    ap.add_argument("--warmup-frames", type=int, default=30)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--out", type=str,
                    default="benchmarks/rvm_512_ds025_720p_blur.json")
    ap.add_argument("--print-every", type=int, default=60)

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.camera, args.width,
                      args.height, args.fps, args.backend)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open camera {args.camera}")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_w}x{actual_h} backend={args.backend}")

    pipeline = RVMPipeline(
        model_path=args.model,
        device=args.device,
        input_size=int(args.input_size),
        downsample_ratio=float(args.downsample),
    )

    compositor: Optional[BackgroundBlur] = None
    if args.blur:
        compositor = BackgroundBlur(downscale=float(
            args.blur_downscale), sigma=float(args.blur_sigma))

    # Timings
    cap_ms: List[float] = []
    infer_ms: List[float] = []
    comp_ms: List[float] = []
    total_ms: List[float] = []

    # Warmup
    print(f"Warmup: {args.warmup_frames} frames...")
    for _ in range(args.warmup_frames):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            continue
        _ = (t1 - t0) * 1000.0  # capture only
        res = pipeline.process_frame(frame)
        if compositor is not None:
            _ = compositor.apply(frame, res.alpha)

    print(
        f"Benchmark: {args.duration}s (blur={'on' if args.blur else 'off'})...")
    start = time.perf_counter()
    frames = 0

    while True:
        now = time.perf_counter()
        if now - start >= args.duration:
            break

        t_cap0 = time.perf_counter()
        ok, frame = cap.read()
        t_cap1 = time.perf_counter()
        if not ok:
            continue

        t_inf0 = time.perf_counter()
        # includes preprocess+ORT+postprocess
        res = pipeline.process_frame(frame)
        t_inf1 = time.perf_counter()

        if compositor is not None:
            t_cmp0 = time.perf_counter()
            _ = compositor.apply(frame, res.alpha)
            t_cmp1 = time.perf_counter()
            comp_ms.append((t_cmp1 - t_cmp0) * 1000.0)

        cap_ms.append((t_cap1 - t_cap0) * 1000.0)
        infer_ms.append((t_inf1 - t_inf0) * 1000.0)

        # Total = capture + infer + comp (if enabled)
        if comp_ms:
            total_ms.append(cap_ms[-1] + infer_ms[-1] + comp_ms[-1])
        else:
            total_ms.append(cap_ms[-1] + infer_ms[-1])

        frames += 1
        if args.print_every > 0 and frames % args.print_every == 0:
            elapsed = time.perf_counter() - start
            fps = frames / elapsed if elapsed > 0 else 0.0
            print(
                f"frames={frames} fps={fps:.1f} "
                f"cap={np.mean(cap_ms[-args.print_every:]):.1f}ms "
                f"infer={np.mean(infer_ms[-args.print_every:]):.1f}ms "
                f"comp={(np.mean(comp_ms[-args.print_every:]) if comp_ms else 0.0):.1f}ms"
            )

    cap.release()

    elapsed = time.perf_counter() - start
    fps_mean = frames / elapsed if elapsed > 0 else 0.0

    summary: Dict = {
        "config": {
            "camera": args.camera,
            "backend": args.backend,
            "capture_resolution": [actual_w, actual_h],
            "target_fps": args.fps,
            "device": args.device,
            "model": args.model,
            "input_size": args.input_size,
            "downsample_ratio": args.downsample,
            "blur": bool(args.blur),
            "blur_downscale": args.blur_downscale,
            "blur_sigma": args.blur_sigma,
            "warmup_frames": args.warmup_frames,
            "duration_s": args.duration,
        },
        "results": {
            "frames": frames,
            "elapsed_s": float(elapsed),
            "fps_mean": float(fps_mean),
            "capture_ms": asdict(pct(cap_ms)),
            "infer_ms": asdict(pct(infer_ms)),
            "comp_ms": asdict(pct(comp_ms)) if comp_ms else asdict(Percentiles(0.0, 0.0, 0.0, 0.0)),
            "total_ms": asdict(pct(total_ms)),
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote: {out_path.resolve()}")
    print(json.dumps(summary["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
