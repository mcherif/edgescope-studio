from pathlib import Path
import sys

# Ensure project root (parent of scripts/) is on sys.path so we can import edgescope.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edgescope.engine.detector import RTMDetDetector
import numpy as np
import cv2


def main() -> None:
    project_root = PROJECT_ROOT

    img_path = project_root / "data" / "images" / "test.jpg"
    config_path = project_root / "rtmdet" / "rtmdet_tiny_8xb32-300e_coco.py"
    checkpoint_path = project_root / "rtmdet" / \
        "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"

    if not img_path.exists():
        raise FileNotFoundError(
            f"{img_path} not found. Drop an image named 'test.jpg' into data/images."
        )

    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            "RTMDet config or checkpoint not found in rtmdet/. "
            "Check filenames in scripts/test_detector_rtm.py."
        )

    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image at {img_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detector = RTMDetDetector(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=None,          # auto: cuda if available, else cpu
        score_threshold=0.3,  # filter low scores
    )

    detections = detector.detect(image_rgb)
    print(f"Found {len(detections)} detections")
    for det in detections:
        print(det)

    vis = image_bgr.copy()
    for det in detections:
        pt1 = int(det.x1), int(det.y1)
        pt2 = int(det.x2), int(det.y2)
        cv2.rectangle(vis, pt1, pt2, (0, 255, 0), 2)
        label_text = f"{det.label} {det.score:.2f}"
        cv2.putText(
            vis,
            label_text,
            (pt1[0], max(0, pt1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    out_dir = project_root / "data" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_rtm_output.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
