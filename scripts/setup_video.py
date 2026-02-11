"""
Download and verify RVM ONNX model for EdgeScope Video Mode.
Run: python scripts/setup_video.py
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str  # lowercase hex
    filename: str


RVM_MOBILENETV3_FP32 = ModelSpec(
    name="RVM MobileNetV3 fp32 (ONNX)",
    url="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
    # Published SHA256 for this file (used by Git/LFS pointers on HF).
    sha256="88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828",
    filename="rvm_mobilenetv3_fp32.onnx",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path, retries: int = 5) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading: {url}")
            print(f"To: {dst}")
            with urllib.request.urlopen(url, timeout=60) as r, dst.open("wb") as f:
                # Stream copy
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** (attempt - 1)
            print(f"Download failed (attempt {attempt}/{retries}): {e}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)


def ensure_model(spec: ModelSpec, models_dir: Path) -> Path:
    path = models_dir / spec.filename

    if path.exists():
        got = sha256_file(path)
        if got.lower() == spec.sha256.lower():
            print(f"✓ Model already present and verified: {path}")
            return path
        print("✗ Existing file checksum mismatch. Re-downloading.")
        path.unlink()

    download(spec.url, path)
    got = sha256_file(path)

    if got.lower() != spec.sha256.lower():
        path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded file failed SHA256 verification.\n"
            f"Expected: {spec.sha256}\n"
            f"Got:      {got}\n"
            "This usually means a partial/corrupted download."
        )

    print(f"✓ Downloaded and verified: {path}")
    return path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"

    # Allow override for experimentation
    model_choice = os.environ.get("EDGESCOPE_RVM_MODEL", "fp32").lower()
    if model_choice not in ("fp32",):
        print("Only fp32 is wired for now. Set EDGESCOPE_RVM_MODEL=fp32")
        return 2

    ensure_model(RVM_MOBILENETV3_FP32, models_dir)
    print("\nNext:")
    print("  python scripts/inspect_rvm.py")
    print("  python scripts/warmup_rvm.py --device cuda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
