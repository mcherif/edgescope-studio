from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
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
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--backend", type=str, default="auto",
                    choices=["auto", "any", "dshow", "msmf"])
    ap.add_argument("--reprobe", action="store_true",
                    help="Ignore cached backend probe and re-test.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Process N frames then exit (0 = run forever).")
    ap.add_argument("--headless", action="store_true",
                    help="Disable window display (useful for testing).")
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

    backend_map = {
        "any": 0,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }

    def open_camera(backend: str) -> cv2.VideoCapture:
        api = backend_map.get(backend.lower(), 0)
        cam = cv2.VideoCapture(args.camera, api)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cam.set(cv2.CAP_PROP_FPS, args.fps)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cam

    cache_path = Path(__file__).resolve().parents[1] / "data" / "cache" / "backend_probe.json"
    cache_key = f"{args.camera}:{args.width}x{args.height}:{args.fps}"

    def read_cache() -> dict | None:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        entry = cache.get(cache_key)
        if not isinstance(entry, dict):
            return None
        winner = entry.get("winner")
        if winner not in ("msmf", "dshow", "any"):
            return None
        return entry

    def write_cache(winner: str, winner_reason: str, results: dict) -> None:
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

    def probe_backend(duration_s: float = 3.0) -> str:
        candidates = ["msmf", "dshow"]
        results = {}
        for backend in candidates:
            cam = open_camera(backend)
            if not cam.isOpened():
                cam.release()
                continue

            # Camera health check: verify we can read frames and they're not all identical.
            start = time.perf_counter()
            got_frame = False
            last = None
            same_count = 0
            warning_count = 0
            while time.perf_counter() - start < 1.0:
                ok, frame = cam.read()
                if not ok:
                    warning_count += 1
                    continue
                got_frame = True
                if last is not None and frame.shape == last.shape and np.array_equal(frame, last):
                    same_count += 1
                    if same_count >= 5:
                        cam.release()
                        results[backend] = {
                            "fps_mean": 0.0,
                            "total_p95_ms": float("inf"),
                            "status": "UNSTABLE",
                            "warning_count": warning_count,
                            "reason": "inactive/identical",
                        }
                        break
                else:
                    same_count = 0
                last = frame
            if backend in results:
                continue
            if not got_frame:
                cam.release()
                results[backend] = {
                    "fps_mean": 0.0,
                    "total_p95_ms": float("inf"),
                    "status": "UNSTABLE",
                    "warning_count": warning_count,
                    "reason": "inactive/no_frames",
                }
                continue

            # quick warmup
            for _ in range(10):
                ok, frame = cam.read()
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
                if now - start >= duration_s:
                    break
                t0 = time.perf_counter()
                ok, frame = cam.read()
                if not ok:
                    warning_count += 1
                    continue
                _ = pipeline.process_frame(frame)
                t1 = time.perf_counter()
                total_ms.append((t1 - t0) * 1000.0)
                frames += 1

            cam.release()

            elapsed = time.perf_counter() - start
            fps = frames / elapsed if elapsed > 0 else 0.0
            p95 = np.percentile(
                total_ms, 95) if total_ms else float("inf")
            status = "OK" if warning_count == 0 else "UNSTABLE"
            reason = "" if warning_count == 0 else f"read_failures={warning_count}"
            results[backend] = {
                "fps_mean": float(fps),
                "total_p95_ms": float(p95),
                "status": status,
                "warning_count": warning_count,
                "reason": reason,
            }

        for backend, res in results.items():
            print(
                f"{backend}: fps={res['fps_mean']:.2f} "
                f"p95_total={res['total_p95_ms']:.2f}ms "
                f"status={res['status']} "
                f"(warning_count={int(res['warning_count'])})"
            )

        if not results:
            return "msmf"

        stable = {k: v for k, v in results.items() if v.get("status") == "OK"}
        pool = stable if stable else results
        winner_reason = "stable" if stable else "all_unstable"

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
                pool.items(), key=lambda kv: (-kv[1]["fps_mean"], kv[1]["total_p95_ms"])
            )[0][0]

        if stable:
            unstable = [k for k, v in results.items() if v.get("status") != "OK"]
            if unstable:
                winner_reason = f"{','.join(unstable)}_unstable"

        print(f"winner: {winner} ({winner_reason})")
        write_cache(winner, winner_reason, results)
        return winner

    backend = args.backend
    if backend == "auto":
        if not args.reprobe:
            cached = read_cache()
        else:
            cached = None

        if cached is not None:
            backend = cached["winner"]
            reason = cached.get("winner_reason", "cached")
            print(
                f"Recommended backend on this machine (cached): {backend} "
                f"({reason})"
            )
            print("Override: --backend msmf|dshow, Reprobe: --reprobe")
        else:
            backend = probe_backend()
            print(f"Recommended backend on this machine: {backend}")
            print("Override: --backend msmf|dshow, Reprobe: --reprobe")
        pipeline.reset()

    cap = open_camera(backend)

    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}")
        return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"Camera opened: {actual_w}x{actual_h} backend={backend} "
        f"(try scripts/probe_backends.py to compare)"
    )

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

            if not args.headless:
                cv2.imshow("EdgeScope Video", out)
                # If user closes the window, stop the loop cleanly.
                if cv2.getWindowProperty("EdgeScope Video", cv2.WND_PROP_VISIBLE) < 1:
                    break
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

            if not args.headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("b"):
                    debug = not debug
                    print(f"Debug view: {'ON' if debug else 'OFF'}")
                elif key == ord("r"):
                    pipeline.reset()

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
