from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

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


def run_probe(
    backend: str,
    args: argparse.Namespace,
    pipeline: RVMPipeline,
) -> Optional[Dict[str, float]]:
    cap = open_camera(args.camera, args.width, args.height, args.fps, backend)
    if not cap.isOpened():
        print(f"[{backend}] ERROR: cannot open camera {args.camera}")
        return None

    # Camera health check: verify we can read frames and they're not all identical.
    start = time.perf_counter()
    got_frame = False
    last = None
    same_count = 0
    warning_count = 0
    while time.perf_counter() - start < 1.0:
        ok, frame = cap.read()
        if not ok:
            warning_count += 1
            continue
        got_frame = True
        if last is not None and frame.shape == last.shape and np.array_equal(frame, last):
            same_count += 1
            if same_count >= 5:
                cap.release()
                print("Camera appears inactive/off; probe results invalid.")
                return None
        else:
            same_count = 0
        last = frame
    if not got_frame:
        cap.release()
        print("Camera appears inactive/off; probe results invalid.")
        return None

    # Warmup
    for _ in range(args.warmup_frames):
        ok, frame = cap.read()
        if not ok:
            warning_count += 1
            continue
        _ = pipeline.process_frame(frame)

    pipeline.reset()

    total_ms = []
    frames = 0
    start = time.perf_counter()
    while True:
        now = time.perf_counter()
        if now - start >= args.duration:
            break

        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            warning_count += 1
            continue
        _ = pipeline.process_frame(frame)
        t1 = time.perf_counter()

        total_ms.append((t1 - t0) * 1000.0)
        frames += 1

    cap.release()

    elapsed = time.perf_counter() - start
    fps = frames / elapsed if elapsed > 0 else 0.0
    p95 = float(np.percentile(total_ms, 95)) if total_ms else 0.0
    status = "OK" if warning_count == 0 else "UNSTABLE"
    reason = "" if warning_count == 0 else f"read_failures={warning_count}"
    return {
        "frames": float(frames),
        "elapsed_s": float(elapsed),
        "fps_mean": float(fps),
        "total_p95_ms": float(p95),
        "warning_count": int(warning_count),
        "status": status,
        "reason": reason,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Probe camera backends (msmf vs dshow) with a short benchmark."
    )
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--backend", type=str, default="both",
                    choices=["both", "msmf", "dshow"])

    ap.add_argument("--model", type=str,
                    default="models/rvm_mobilenetv3_fp32.onnx")
    ap.add_argument("--device", type=str, default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--downsample", type=float, default=0.25)

    ap.add_argument("--warmup-frames", type=int, default=30)
    ap.add_argument("--duration", type=int, default=10)
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
    if args.device == "cuda" and "CUDAExecutionProvider" not in pipeline.session.get_providers():
        print(
            "ERROR: CUDAExecutionProvider not available; refusing to fall back to CPU. "
            "Install CUDA-enabled onnxruntime or use --device cpu.",
            file=sys.stderr,
        )
        return 3

    cache_path = Path(__file__).resolve().parents[1] / "data" / "cache" / "backend_probe.json"
    cache_key = f"{args.camera}:{args.width}x{args.height}:{args.fps}"

    def write_cache(winner: str, winner_reason: str, results: Dict[str, Dict[str, float]]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
        cache[cache_key] = {
            "winner": winner,
            "winner_reason": winner_reason,
            "timestamp": time.time(),
            "camera": args.camera,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "backends": results,
        }
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    backends = ["msmf", "dshow"] if args.backend == "both" else [args.backend]
    results: Dict[str, Dict[str, float]] = {}

    for backend in backends:
        print(f"\n[{backend}] Probing for {args.duration}s...")
        res = run_probe(backend, args, pipeline)
        if res is None:
            return 1
        results[backend] = res
        print(
            f"[{backend}] fps={res['fps_mean']:.2f} "
            f"p95_total={res['total_p95_ms']:.2f}ms "
            f"frames={int(res['frames'])} "
            f"status={res['status']} "
            f"(warning_count={int(res['warning_count'])})"
        )

    if not results:
        print("No usable backend found.")
        return 1

    stable = {k: v for k, v in results.items() if v.get("status") == "OK"}
    pool = stable if stable else results
    winner_reason = "stable" if stable else "all_unstable"

    # Pick winner:
    # If fps is within 5%, choose lower p95_total; otherwise choose higher fps.
    if "msmf" in pool and "dshow" in pool:
        fps_m = pool["msmf"]["fps_mean"]
        fps_d = pool["dshow"]["fps_mean"]
        max_fps = max(fps_m, fps_d)
        if max_fps > 0 and abs(fps_m - fps_d) / max_fps < 0.05:
            winner = "msmf" if pool["msmf"]["total_p95_ms"] < pool["dshow"]["total_p95_ms"] else "dshow"
        else:
            winner = "msmf" if fps_m > fps_d else "dshow"
    else:
        winner = sorted(
            pool.items(),
            key=lambda kv: (-kv[1]["fps_mean"], kv[1]["total_p95_ms"]),
        )[0][0]

    if stable:
        unstable = [k for k, v in results.items() if v.get("status") != "OK"]
        if unstable:
            winner_reason = f"{','.join(unstable)}_unstable"

    write_cache(winner, winner_reason, results)
    print(f"\nWinner: {winner} ({winner_reason})")
    print("stable = passed health check + warning_count == 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
