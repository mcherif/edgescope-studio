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


def open_video(path: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(path)


def read_frame(cap: cv2.VideoCapture, loop: bool) -> tuple[bool, np.ndarray | None]:
    ok, frame = cap.read()
    if ok:
        return True, frame
    if not loop:
        return False, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    return ok, frame if ok else None


def jitter_stats(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(np.percentile(arr, 95))


def edge_mask(alpha: np.ndarray, mode: str, grad_thresh: float = 0.02) -> np.ndarray:
    band = (alpha >= 0.1) & (alpha <= 0.9)
    a32 = alpha.astype(np.float32)
    gx = cv2.Sobel(a32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a32, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    grad = mag > float(grad_thresh)
    if mode == "band":
        return band
    if mode == "grad":
        return grad
    return band | grad


def run_trial(
    label: str,
    args: argparse.Namespace,
    pipeline: RVMPipeline,
    reset_each_frame: bool,
) -> Dict:
    use_video = args.video is not None
    cap = open_video(args.video) if use_video else open_camera(
        args.camera, args.width, args.height, args.fps, args.backend
    )
    if not cap.isOpened():
        source = args.video if use_video else f"camera {args.camera}"
        raise RuntimeError(f"ERROR: cannot open {source}")

    # Camera health check: verify we can read frames and they're not all identical.
    start = time.perf_counter()
    got_frame = False
    last = None
    same_count = 0
    while time.perf_counter() - start < 1.0:
        ok, frame = read_frame(cap, loop=use_video)
        if not ok:
            continue
        got_frame = True
        if not use_video:
            if last is not None and frame.shape == last.shape and np.array_equal(frame, last):
                same_count += 1
                if same_count >= 5:
                    cap.release()
                    raise RuntimeError("Camera appears inactive/off; temporal results invalid.")
            else:
                same_count = 0
        last = frame
    if not got_frame:
        cap.release()
        if use_video:
            raise RuntimeError("Video appears unreadable; temporal results invalid.")
        raise RuntimeError("Camera appears inactive/off; temporal results invalid.")

    # Warmup (not counted)
    for _ in range(args.warmup_frames):
        ok, frame = read_frame(cap, loop=use_video)
        if not ok:
            continue
        if reset_each_frame:
            pipeline.reset(verbose=False)
        _ = pipeline.process_frame(frame)

    pipeline.reset(verbose=False)

    jitters_all: List[float] = []
    jitters_edge: List[float] = []
    edge_fractions: List[float] = []
    frames = 0
    prev_alpha: np.ndarray | None = None
    prev_edge: np.ndarray | None = None
    start = time.perf_counter()

    while True:
        now = time.perf_counter()
        if now - start >= args.duration:
            break

        ok, frame = read_frame(cap, loop=use_video)
        if not ok:
            continue

        if reset_each_frame:
            pipeline.reset(verbose=False)

        res = pipeline.process_frame(frame)

        if prev_alpha is not None:
            diff = np.abs(res.alpha - prev_alpha)
            jitters_all.append(float(diff.mean()))

            edge = edge_mask(res.alpha, args.edge_mode, args.edge_grad_thresh)
            if prev_edge is None:
                prev_edge = edge_mask(prev_alpha, args.edge_mode, args.edge_grad_thresh)
            mask = edge | prev_edge
            if np.any(mask):
                jitters_edge.append(float(diff[mask].mean()))
            else:
                jitters_edge.append(float("nan"))

            prev_edge = edge
        prev_alpha = res.alpha
        edge_fractions.append(float(edge_mask(res.alpha, args.edge_mode, args.edge_grad_thresh).mean()))
        frames += 1

    cap.release()

    elapsed = time.perf_counter() - start
    fps = frames / elapsed if elapsed > 0 else 0.0
    jitter_all_mean, jitter_all_p95 = jitter_stats(jitters_all)
    jitter_edge_mean, jitter_edge_p95 = jitter_stats(jitters_edge)
    edge_frac_mean, edge_frac_p95 = jitter_stats(edge_fractions)
    edge_jitter_per_edgepixel_mean = (
        jitter_edge_mean / edge_frac_mean if edge_frac_mean > 0 else 0.0
    )

    return {
        "label": label,
        "frames": frames,
        "elapsed_s": float(elapsed),
        "fps_mean": float(fps),
        "jitter_all_mean": jitter_all_mean,
        "jitter_all_p95": jitter_all_p95,
        "jitter_edge_mean": jitter_edge_mean,
        "jitter_edge_p95": jitter_edge_p95,
        "edge_fraction_mean": edge_frac_mean,
        "edge_fraction_p95": edge_frac_p95,
        "edge_jitter_per_edgepixel_mean": edge_jitter_per_edgepixel_mean,
        "reset_each_frame": reset_each_frame,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare temporal stability with/without recurrent states."
    )
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=str, default=None,
                    help="Optional path to a video file for benchmarking")
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
    ap.add_argument("--edge-mode", type=str, default="band_or_grad",
                    choices=["band", "grad", "band_or_grad"],
                    help="Edge definition used for jitter (band, grad, or both)")
    ap.add_argument("--edge-grad-thresh", type=float, default=0.02,
                    help="Gradient magnitude threshold for edge detection")

    ap.add_argument("--warmup-frames", type=int, default=30)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--out-on", type=str, default="benchmarks/temporal_on.json")
    ap.add_argument("--out-off", type=str, default="benchmarks/temporal_off.json")
    args = ap.parse_args()

    if args.video is not None:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return 2

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
    if args.device == "cuda" and "CUDAExecutionProvider" not in pipeline.session.get_providers():
        print(
            "ERROR: CUDAExecutionProvider not available; refusing to fall back to CPU. "
            "Install CUDA-enabled onnxruntime or use --device cpu.",
            file=sys.stderr,
        )
        return 3

    config = {
        "camera": args.camera,
        "backend": args.backend,
        "video": args.video,
        "capture_resolution": [args.width, args.height],
        "fps": args.fps,
        "device": args.device,
        "model": args.model,
        "input_size": args.input_size,
        "downsample_ratio": args.downsample,
        "warmup_frames": args.warmup_frames,
        "duration_s": args.duration,
        "jitter_edge_band": [0.1, 0.9],
        "jitter_edge_mode": args.edge_mode,
        "jitter_grad_thresh": args.edge_grad_thresh,
    }

    try:
        print("Temporal ON (recurrent states enabled)")
        on = run_trial("temporal_on", args, pipeline, reset_each_frame=False)
        print(
            f"  fps={on['fps_mean']:.2f} "
            f"jitter_all_mean={on['jitter_all_mean']:.6f} "
            f"jitter_all_p95={on['jitter_all_p95']:.6f} "
            f"jitter_edge_mean={on['jitter_edge_mean']:.6f} "
            f"jitter_edge_p95={on['jitter_edge_p95']:.6f} "
            f"edge_fraction_mean={on['edge_fraction_mean']*100:.2f}% "
            f"edge_fraction_p95={on['edge_fraction_p95']*100:.2f}% "
            f"edge_jitter_per_edgepixel_mean={on['edge_jitter_per_edgepixel_mean']:.6f}"
        )

        print("Temporal OFF (reset states every frame)")
        off = run_trial("temporal_off", args, pipeline, reset_each_frame=True)
        print(
            f"  fps={off['fps_mean']:.2f} "
            f"jitter_all_mean={off['jitter_all_mean']:.6f} "
            f"jitter_all_p95={off['jitter_all_p95']:.6f} "
            f"jitter_edge_mean={off['jitter_edge_mean']:.6f} "
            f"jitter_edge_p95={off['jitter_edge_p95']:.6f} "
            f"edge_fraction_mean={off['edge_fraction_mean']*100:.2f}% "
            f"edge_fraction_p95={off['edge_fraction_p95']*100:.2f}% "
            f"edge_jitter_per_edgepixel_mean={off['edge_jitter_per_edgepixel_mean']:.6f}"
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    Path(args.out_on).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_off).parent.mkdir(parents=True, exist_ok=True)

    Path(args.out_on).write_text(
        json.dumps({"config": config, "results": on}, indent=2), encoding="utf-8"
    )
    Path(args.out_off).write_text(
        json.dumps({"config": config, "results": off}, indent=2), encoding="utf-8"
    )

    # Improvement (lower is better)
    if off["jitter_edge_mean"] > 0:
        improvement = (
            (off["jitter_edge_mean"] - on["jitter_edge_mean"])
            / off["jitter_edge_mean"]
            * 100.0
        )
    else:
        improvement = 0.0
    print(f"Improvement (edge jitter mean): {improvement:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
