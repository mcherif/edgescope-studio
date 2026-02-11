from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
import time
from pathlib import Path

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", type=str,
                    default="models/rvm_mobilenetv3_fp32.onnx")
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--downsample", type=float, default=0.25)
    ap.add_argument("--blur", type=int, default=51)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: python scripts/setup_video.py")
        return 2

    print("=" * 60)
    print("EdgeScope Video Mode (RVM background blur)")
    print("=" * 60)
    print("Controls: q=quit, b=toggle blur/debug, r=reset temporal states")
    print("=" * 60)

    pipeline = RVMPipeline(
        model_path=str(model_path),
        input_size=args.input_size,
        downsample_ratio=args.downsample,
        device=args.device,
    )

    # blur = BackgroundBlur(blur_kernel=args.blur)
    blur = BackgroundBlur(downscale=0.25, sigma=8.0)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}")
        return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_w}x{actual_h}")

    # FPS smoothing
    fps_ema = None
    debug = False
    t_prev = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed")
                break

            # Inference
            t0 = time.perf_counter()
            res = pipeline.process_frame(frame)
            t1 = time.perf_counter()

            # Composite or debug
            if not debug:
                out = blur.apply(frame, res.alpha)
            else:
                # show alpha heatmap overlay
                a8 = (np.clip(res.alpha, 0, 1) * 255).astype(np.uint8)
                cm = cv2.applyColorMap(a8, cv2.COLORMAP_JET)
                out = cv2.addWeighted(frame, 0.6, cm, 0.4, 0)
            t2 = time.perf_counter()

            # FPS (EMA)
            t_now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, (t_now - t_prev))
            t_prev = t_now
            fps_ema = inst_fps if fps_ema is None else (
                0.9 * fps_ema + 0.1 * inst_fps)

            # Overlay
            cv2.rectangle(out, (10, 10), (360, 90), (0, 0, 0), -1)
            cv2.putText(out, f"FPS: {fps_ema:5.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(out, f"RVM latency: {res.latency_ms:5.1f} ms", (
                20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("EdgeScope Video", out)
            t3 = time.perf_counter()

            infer_ms = (t1 - t0) * 1000
            comp_ms = (t2 - t1) * 1000
            show_ms = (t3 - t2) * 1000

            frame_idx += 1
            if frame_idx % 60 == 0:
                print(
                    f"infer={infer_ms:5.1f}ms comp={comp_ms:5.1f}ms show={show_ms:5.1f}ms",
                    flush=True,
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                debug = not debug
                print(f"Debug view: {'ON' if debug else 'OFF'}")
            elif key == ord("r"):
                pipeline.reset()

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
