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


def camera_health_check(cap: cv2.VideoCapture) -> None:
    # Verify frames are flowing and not identical for ~1s.
    start = time.perf_counter()
    got_frame = False
    last = None
    same_count = 0
    while time.perf_counter() - start < 1.0:
        ok, frame = cap.read()
        if not ok:
            continue
        got_frame = True
        if last is not None and frame.shape == last.shape and np.array_equal(frame, last):
            same_count += 1
            if same_count >= 5:
                raise SystemExit("ERROR: camera appears inactive/off; benchmark invalid.")
        else:
            same_count = 0
        last = frame
    if not got_frame:
        raise SystemExit("ERROR: camera appears inactive/off; benchmark invalid.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Headless benchmark for RVM video mode")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional video path. If set, uses file input (loops by default).",
    )
    ap.add_argument(
        "--video-frame-index",
        type=int,
        default=0,
        help="0-based frame index to start from when --video is set.",
    )
    ap.add_argument(
        "--video-frame-count",
        type=int,
        default=0,
        help="Number of consecutive frames to loop when --video is set (0 = full clip).",
    )
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
    ap.add_argument("--blur-scale", type=float, default=0.5,
                    help="Downscale used for fast blur (e.g. 0.5 or 0.4)")
    ap.add_argument("--blur-downscale", type=float, default=None,
                    help="(deprecated) Same as --blur-scale")
    ap.add_argument("--comp-mode", type=str, default="soft",
                    choices=["soft", "hard", "trimap"],
                    help="Compositing mode: soft alpha blend or hard mask copy")
    ap.add_argument("--alpha-thresh", type=int, default=128,
                    help="Threshold for hard mask compositing (0-255)")
    ap.add_argument("--alpha-lo", type=int, default=32,
                    help="Lower threshold for trimap (0-255)")
    ap.add_argument("--alpha-hi", type=int, default=224,
                    help="Upper threshold for trimap (0-255)")
    ap.add_argument("--alpha-feather", type=int, default=0,
                    help="Feather kernel size for alpha (odd, 0 disables)")
    ap.add_argument("--blur-sigma", type=float, default=8.0,
                    help="Gaussian sigma used for blur")
    ap.add_argument("--alpha-path", type=str, default="optimized",
                    choices=["optimized", "legacy"],
                    help="Alpha prep path: optimized (OpenCV + reuse) or legacy (numpy + allocs)")

    ap.add_argument("--warmup-frames", type=int, default=30)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--out", type=str,
                    default="benchmarks/rvm_512_ds025_720p_blur.json")
    ap.add_argument("--print-every", type=int, default=60)

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_video = args.video is not None
    cap = None
    video_frames = None
    video_stream = False
    actual_w = args.width
    actual_h = args.height
    if use_video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise SystemExit(f"ERROR: video not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise SystemExit(f"ERROR: cannot open video {video_path}")
        if args.video_frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(args.video_frame_index))
        frame_count = int(args.video_frame_count)
        if frame_count <= 0:
            ok, first = cap.read()
            if not ok or first is None:
                cap.release()
                raise SystemExit(
                    f"ERROR: cannot read frame {args.video_frame_index} from {video_path}"
                )
            if first.shape[1] != args.width or first.shape[0] != args.height:
                first = cv2.resize(
                    first,
                    (args.width, args.height),
                    interpolation=cv2.INTER_AREA,
                )
            actual_w, actual_h = first.shape[1], first.shape[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(args.video_frame_index))
            video_stream = True
            print(
                f"Video opened: {actual_w}x{actual_h} src={video_path} "
                f"(frames {args.video_frame_index}..end, loop)"
            )
        else:
            frames = []
            for _ in range(max(1, frame_count)):
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    raise SystemExit(
                        f"ERROR: cannot read frame {args.video_frame_index + len(frames)} "
                        f"from {video_path}"
                    )
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(
                        frame,
                        (args.width, args.height),
                        interpolation=cv2.INTER_AREA,
                    )
                frames.append(frame)
            cap.release()
            video_frames = frames
            actual_w, actual_h = frames[0].shape[1], frames[0].shape[0]
            print(
                f"Video opened: {actual_w}x{actual_h} src={video_path} "
                f"(frames {args.video_frame_index}.."
                f"{args.video_frame_index + len(frames) - 1})"
            )
    else:
        cap = open_camera(args.camera, args.width,
                          args.height, args.fps, args.backend)
        if not cap.isOpened():
            raise SystemExit(f"ERROR: cannot open camera {args.camera}")
        camera_health_check(cap)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {actual_w}x{actual_h} backend={args.backend}")

    pipeline = RVMPipeline(
        model_path=args.model,
        device=args.device,
        input_size=int(args.input_size),
        downsample_ratio=float(args.downsample),
    )
    if args.device == "cuda" and "CUDAExecutionProvider" not in pipeline.session.get_providers():
        if cap is not None and not use_video:
            cap.release()
        print(
            "ERROR: CUDAExecutionProvider not available; refusing to fall back to CPU. "
            "Install CUDA-enabled onnxruntime or use --device cpu.",
            file=sys.stderr,
        )
        return 3

    if args.blur_downscale is not None:
        args.blur_scale = args.blur_downscale

    compositor: Optional[BackgroundBlur] = None
    if args.blur:
        compositor = BackgroundBlur(
            blur_scale=float(args.blur_scale),
            sigma=float(args.blur_sigma),
            comp_mode=args.comp_mode,
            alpha_path=args.alpha_path,
            alpha_thresh=args.alpha_thresh,
            alpha_lo=args.alpha_lo,
            alpha_hi=args.alpha_hi,
            alpha_feather=args.alpha_feather,
        )

    # Timings
    cap_ms: List[float] = []
    infer_ms: List[float] = []
    comp_ms: List[float] = []
    comp_resize_down_ms: List[float] = []
    comp_blur_ms: List[float] = []
    comp_resize_up_ms: List[float] = []
    comp_mask_ms: List[float] = []
    comp_alpha_convert_ms: List[float] = []
    comp_alpha_feather_ms: List[float] = []
    comp_alpha_expand_ms: List[float] = []
    comp_alpha_invert_ms: List[float] = []
    comp_blend_ms: List[float] = []
    total_ms: List[float] = []

    # Warmup
    print(f"Warmup: {args.warmup_frames} frames...")
    video_idx = 0
    for _ in range(args.warmup_frames):
        t0 = time.perf_counter()
        if use_video:
            if video_stream:
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(args.video_frame_index))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise SystemExit("ERROR: cannot read frame from video stream")
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(
                        frame,
                        (args.width, args.height),
                        interpolation=cv2.INTER_AREA,
                    )
                t1 = time.perf_counter()
            else:
                frame = video_frames[video_idx]
                video_idx = (video_idx + 1) % len(video_frames)
                t1 = time.perf_counter()
        else:
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
        if use_video:
            if video_stream:
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(args.video_frame_index))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise SystemExit("ERROR: cannot read frame from video stream")
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(
                        frame,
                        (args.width, args.height),
                        interpolation=cv2.INTER_AREA,
                    )
                t_cap1 = time.perf_counter()
            else:
                frame = video_frames[video_idx]
                video_idx = (video_idx + 1) % len(video_frames)
                t_cap1 = time.perf_counter()
        else:
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
            profile: Dict[str, float] = {}
            _ = compositor.apply(frame, res.alpha, profile=profile)
            t_cmp1 = time.perf_counter()
            comp_ms.append((t_cmp1 - t_cmp0) * 1000.0)
            comp_resize_down_ms.append(profile.get("resize_down_ms", 0.0))
            comp_blur_ms.append(profile.get("blur_ms", 0.0))
            comp_resize_up_ms.append(profile.get("resize_up_ms", 0.0))
            comp_mask_ms.append(profile.get("mask_ms", 0.0))
            comp_alpha_convert_ms.append(profile.get("alpha_convert_ms", 0.0))
            comp_alpha_feather_ms.append(profile.get("alpha_feather_ms", 0.0))
            comp_alpha_expand_ms.append(profile.get("alpha_expand_ms", 0.0))
            comp_alpha_invert_ms.append(profile.get("alpha_invert_ms", 0.0))
            comp_blend_ms.append(profile.get("blend_ms", 0.0))

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
            comp_window = np.mean(comp_ms[-args.print_every:]) if comp_ms else 0.0
            msg = (
                f"frames={frames} fps={fps:.1f} "
                f"cap={np.mean(cap_ms[-args.print_every:]):.1f}ms "
                f"infer={np.mean(infer_ms[-args.print_every:]):.1f}ms "
                f"comp={comp_window:.1f}ms"
            )
            if comp_ms:
                msg += (
                    f" (down={np.mean(comp_resize_down_ms[-args.print_every:]):.1f}"
                    f" blur={np.mean(comp_blur_ms[-args.print_every:]):.1f}"
                    f" up={np.mean(comp_resize_up_ms[-args.print_every:]):.1f}"
                    f" mask={np.mean(comp_mask_ms[-args.print_every:]):.1f}"
                    f" blend={np.mean(comp_blend_ms[-args.print_every:]):.1f})"
                )
            print(msg)

    if cap is not None:
        cap.release()

    elapsed = time.perf_counter() - start
    fps_mean = frames / elapsed if elapsed > 0 else 0.0

    summary: Dict = {
        "config": {
            "camera": args.camera,
            "video": args.video,
            "video_frame_index": args.video_frame_index,
            "video_frame_count": args.video_frame_count,
            "backend": args.backend,
            "capture_resolution": [actual_w, actual_h],
            "target_fps": args.fps,
            "device": args.device,
            "model": args.model,
            "input_size": args.input_size,
            "downsample_ratio": args.downsample,
            "blur": bool(args.blur),
            "blur_scale": args.blur_scale,
            "blur_sigma": args.blur_sigma,
            "comp_mode": args.comp_mode,
            "alpha_path": args.alpha_path,
            "alpha_thresh": args.alpha_thresh,
            "alpha_lo": args.alpha_lo,
            "alpha_hi": args.alpha_hi,
            "alpha_feather": args.alpha_feather,
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
            "comp_breakdown_ms": {
                "resize_down": asdict(pct(comp_resize_down_ms)),
                "blur": asdict(pct(comp_blur_ms)),
                "resize_up": asdict(pct(comp_resize_up_ms)),
                "mask": asdict(pct(comp_mask_ms)),
                "alpha_convert": asdict(pct(comp_alpha_convert_ms)),
                "alpha_feather": asdict(pct(comp_alpha_feather_ms)),
                "alpha_expand": asdict(pct(comp_alpha_expand_ms)),
                "alpha_invert": asdict(pct(comp_alpha_invert_ms)),
                "blend": asdict(pct(comp_blend_ms)),
            },
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
