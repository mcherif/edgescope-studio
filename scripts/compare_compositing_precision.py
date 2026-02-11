import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# current uint8 version
try:
    from edgescope.video.compositing import BackgroundBlur as FastBlur
except ModuleNotFoundError as e:
    if e.name != "edgescope":
        raise
    # Add project root (parent of scripts/) to sys.path and retry.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from edgescope.video.compositing import BackgroundBlur as FastBlur


def slow_float_blend(frame_bgr, blurred_bgr, alpha):
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    a3 = np.repeat(a[..., None], 3, axis=2)
    out = frame_bgr.astype(np.float32) * a3 + \
        blurred_bgr.astype(np.float32) * (1.0 - a3)
    return np.clip(out, 0, 255).astype(np.uint8)


def psnr(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 99.0
    return 10.0 * np.log10((255.0 ** 2) / mse)


def main():
    # Use one real image if available; otherwise random but prefer real.
    # Prefer demos/webcam_frame.png, then sample_image.png, else fallback to random.
    webcam_path = REPO_ROOT / "demos" / "webcam_frame.png"
    sample_path = REPO_ROOT / "sample_image.png"

    img = None
    img_src = "random"
    if webcam_path.exists():
        img = cv2.imread(str(webcam_path))
        if img is None:
            print(f"Failed to read {webcam_path}; falling back.")
        else:
            img_src = str(webcam_path)

    if img is None:
        img = cv2.imread(str(sample_path))
        if img is None:
            print("No sample image found, using random image (less meaningful).")
            img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        else:
            img_src = str(sample_path)

    print(f"Comparing compositing precision on: {img_src}")

    h, w = img.shape[:2]

    # Create a synthetic alpha with smooth transitions (worst-case for quantization)
    x = np.linspace(0, 1, w, dtype=np.float32)
    alpha = np.tile(x[None, :], (h, 1))  # smooth ramp

    # Build blurred background once (match current fast blur idea)
    small = cv2.resize(img, (w//4, h//4), interpolation=cv2.INTER_AREA)
    small_blur = cv2.GaussianBlur(small, (0, 0), sigmaX=8.0, sigmaY=8.0)
    blurred = cv2.resize(small_blur, (w, h), interpolation=cv2.INTER_LINEAR)

    # Old vs new outputs
    out_slow = slow_float_blend(img, blurred, alpha)
    fast = FastBlur(downscale=0.25, sigma=8.0)
    out_fast = fast.apply(img, alpha)

    diff = out_slow.astype(np.int16) - out_fast.astype(np.int16)
    mae = np.mean(np.abs(diff))
    p = psnr(out_slow, out_fast)

    # Edge-only region: alpha in (0.1, 0.9)
    edge_mask = (alpha > 0.1) & (alpha < 0.9)
    if edge_mask.any():
        edge_mae = np.mean(np.abs(diff[edge_mask.repeat(3).reshape(h, w, 3)]))
    else:
        edge_mae = float("nan")

    print(f"MAE (all pixels): {mae:.4f} intensity levels (0-255)")
    print(f"MAE (edge alpha 0.1..0.9): {edge_mae:.4f}")
    print(f"PSNR: {p:.2f} dB")

    cv2.imwrite("demos/comp_slow.png", out_slow)
    cv2.imwrite("demos/comp_fast.png", out_fast)
    cv2.imwrite("demos/comp_diff.png",
                (np.clip(np.abs(diff) * 10, 0, 255)).astype(np.uint8))
    print("Wrote demos/comp_slow.png, demos/comp_fast.png, demos/comp_diff.png")


if __name__ == "__main__":
    main()
