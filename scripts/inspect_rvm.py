"""
Inspect RVM ONNX model inputs/outputs and persist metadata.
Run: python scripts/inspect_rvm.py
"""

from __future__ import annotations

import json
from pathlib import Path

import onnxruntime as ort


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "models" / "rvm_mobilenetv3_fp32.onnx"
    out_path = repo_root / "models" / "rvm_io.json"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: python scripts/setup_video.py")
        return 2

    sess = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    info = {
        "providers": sess.get_providers(),
        "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()],
        "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in sess.get_outputs()],
    }

    print("PROVIDERS:", info["providers"])
    print("\nINPUTS:")
    for i in info["inputs"]:
        print(" ", i["name"], i["shape"], i["type"])
    print("\nOUTPUTS:")
    for o in info["outputs"]:
        print(" ", o["name"], o["shape"], o["type"])

    out_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
