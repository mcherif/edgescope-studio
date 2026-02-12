from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    from edgescope.video.pipeline import RVMPipeline
except ModuleNotFoundError as e:
    if e.name != "edgescope":
        raise
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from edgescope.video.pipeline import RVMPipeline


def open_camera(
    index: int, width: int, height: int, fps: int, backend: str
) -> cv2.VideoCapture:
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
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def jitter_stats(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(np.percentile(arr, 95))


def run_trial(
    label: str,
    args: argparse.Namespace,
    pipeline: RVMPipeline,
    reset_each_frame: bool,
) -> Dict:
    cap = open_camera(args.camera, args.width, args.height, args.fps, args.backend)
    if not cap.isOpened():
        raise RuntimeError(f"ERROR: cannot open camera {args.camera}")

    # Warmup (not counted)
    for _ in range(args.warmup_frames):
        ok, frame = cap.read()
        if not ok:
            continue
        if reset_each_frame:
            pipeline.reset(verbose=False)
        _ = pipeline.process_frame(frame)

    pipeline.reset(verbose=False)

    jitters: List[float] = []
    frames = 0
    prev_alpha: np.ndarray | None = None
    start = time.perf_counter()

    while True:
        now = time.perf_counter()
        if now - start >= args.duration:
            break

        ok, frame = cap.read()
        if not ok:
            continue

        if reset_each_frame:
            pipeline.reset(verbose=False)

        res = pipeline.process_frame(frame)

        if prev_alpha is not None:
            jit = float(np.mean(np.abs(res.alpha - prev_alpha)))
            jitters.append(jit)
        prev_alpha = res.alpha
        frames += 1

    cap.release()

    elapsed = time.perf_counter() - start
    fps = frames / elapsed if elapsed > 0 else 0.0
    jitter_mean, jitter_p95 = jitter_stats(jitters)

    return {
        "label": label,
        "frames": frames,
        "elapsed_s": float(elapsed),
        "fps_mean": float(fps),
        "jitter_mean": jitter_mean,
        "jitter_p95": jitter_p95,
        "reset_each_frame": reset_each_frame,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare temporal stability with/without recurrent states."
    )
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

    ap.add_argument("--warmup-frames", type=int, default=30)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--out-on", type=str, default="benchmarks/temporal_on.json")
    ap.add_argument("--out-off", type=str, default="benchmarks/temporal_off.json")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: python scripts/setup_video.py")
        return 2

    pipeline = RVMPipeline(
        model_path=str(model_path),
        device=args.device,
        input_size=int(args.input_size),
        downsample_ratio=float(args.downsample),
    )

    config = {
        "camera": args.camera,
        "backend": args.backend,
        "capture_resolution": [args.width, args.height],
        "fps": args.fps,
        "device": args.device,
        "model": args.model,
        "input_size": args.input_size,
        "downsample_ratio": args.downsample,
        "warmup_frames": args.warmup_frames,
        "duration_s": args.duration,
    }

    print("Temporal ON (recurrent states enabled)")
    on = run_trial("temporal_on", args, pipeline, reset_each_frame=False)
    print(
        f"  fps={on['fps_mean']:.2f} "
        f"jitter_mean={on['jitter_mean']:.6f} "
        f"jitter_p95={on['jitter_p95']:.6f}"
    )

    print("Temporal OFF (reset states every frame)")
    off = run_trial("temporal_off", args, pipeline, reset_each_frame=True)
    print(
        f"  fps={off['fps_mean']:.2f} "
        f"jitter_mean={off['jitter_mean']:.6f} "
        f"jitter_p95={off['jitter_p95']:.6f}"
    )

    Path(args.out_on).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_off).parent.mkdir(parents=True, exist_ok=True)

    Path(args.out_on).write_text(
        json.dumps({"config": config, "results": on}, indent=2), encoding="utf-8"
    )
    Path(args.out_off).write_text(
        json.dumps({"config": config, "results": off}, indent=2), encoding="utf-8"
    )

    # Improvement (lower is better)
    if off["jitter_mean"] > 0:
        improvement = (off["jitter_mean"] - on["jitter_mean"]) / off["jitter_mean"] * 100.0
    else:
        improvement = 0.0
    print(f"Improvement (mean jitter): {improvement:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
