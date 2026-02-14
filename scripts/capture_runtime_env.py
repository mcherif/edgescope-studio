from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import onnxruntime as ort


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def pip_freeze() -> dict[str, Any]:
    return run_cmd([sys.executable, "-m", "pip", "freeze"])


def conda_list() -> dict[str, Any] | None:
    if not os.environ.get("CONDA_PREFIX"):
        return None
    return run_cmd(["conda", "list"])


def find_in_path(name: str) -> str | None:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if not p:
            continue
        candidate = Path(p) / name
        if candidate.exists():
            return str(candidate.resolve())
    return None


def get_ort_cuda_provider_dll() -> str | None:
    capi_dir = Path(ort.__file__).resolve().parent / "capi"
    dll = capi_dir / "onnxruntime_providers_cuda.dll"
    return str(dll) if dll.exists() else None


def _win_loaded_module_path(dll_name: str) -> str | None:
    if platform.system().lower() != "windows":
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    k32.GetModuleHandleW.restype = ctypes.c_void_p
    h = k32.GetModuleHandleW(dll_name)
    if not h:
        return None
    buf = ctypes.create_unicode_buffer(4096)
    k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    k32.GetModuleFileNameW.restype = ctypes.c_uint
    n = k32.GetModuleFileNameW(h, buf, len(buf))
    if n == 0:
        return None
    return buf.value


def probe_cuda_session(model_path: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"attempted": False, "ok": False, "error": None}
    if not model_path:
        return out
    mp = Path(model_path)
    if not mp.exists():
        out["error"] = f"model not found: {mp}"
        return out
    out["attempted"] = True
    try:
        sess = ort.InferenceSession(str(mp), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        out["ok"] = True
        out["providers_active"] = sess.get_providers()
    except Exception as exc:
        out["ok"] = False
        out["error"] = str(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture reproducibility/runtime environment metadata")
    ap.add_argument("--out", type=str, required=True, help="Output JSON path")
    ap.add_argument(
        "--model",
        type=str,
        default="models/rvm_mobilenetv3_fp32.onnx",
        help="Optional model path for CUDA provider probing",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Attempt to load CUDA EP in this process to surface which DLLs are truly loaded.
    session_probe = probe_cuda_session(args.model)

    target_dlls = [
        "onnxruntime_providers_cuda.dll",
        "cudnn64_9.dll",
        "cublas64_12.dll",
        "cudart64_12.dll",
    ]

    dlls: dict[str, Any] = {}
    for dll in target_dlls:
        if dll == "onnxruntime_providers_cuda.dll":
            path_resolved = get_ort_cuda_provider_dll()
        else:
            path_resolved = find_in_path(dll)
        dlls[dll] = {
            "resolved_path": path_resolved,
            "loaded_module_path": _win_loaded_module_path(dll),
        }

    report = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "environment": {
            "cwd": str(Path.cwd()),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "path": os.environ.get("PATH", ""),
        },
        "onnxruntime": {
            "version": getattr(ort, "__version__", None),
            "device": ort.get_device(),
            "providers_available": ort.get_available_providers(),
        },
        "opencv": {
            "version": cv2.__version__,
            "build_information": cv2.getBuildInformation(),
        },
        "provenance": {
            "pip_freeze": pip_freeze(),
            "conda_list": conda_list(),
        },
        "session_probe": session_probe,
        "dlls": dlls,
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
