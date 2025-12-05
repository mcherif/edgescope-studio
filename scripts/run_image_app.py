from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root (parent of scripts/) is on sys.path for edgescope imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Gradio / gradio_client schema guard patch (API docs only) ---
import gradio_client.utils as gc_utils  # type: ignore[attr-defined]

_orig_get_type = getattr(gc_utils, "get_type", None)
_orig_json_to_py = getattr(gc_utils, "_json_schema_to_python_type", None)


def _safe_get_type(schema: Any) -> str:
    if not isinstance(schema, dict):
        # Non-dict schemas (e.g., bool) only affect API docs.
        return "any"
    return _orig_get_type(schema)  # type: ignore[misc]


def _safe_json_schema_to_python_type(schema: Any, defs: Any) -> str:
    if isinstance(schema, bool):
        return "any"
    return _orig_json_to_py(schema, defs)  # type: ignore[misc]


if _orig_get_type is not None:
    gc_utils.get_type = _safe_get_type  # type: ignore[assignment]
if _orig_json_to_py is not None:
    gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type  # type: ignore[assignment]
# ---------------------------------------------------------------

from edgescope.ui.image_app import create_app


if __name__ == "__main__":
    app = create_app()
    # Bind to localhost with a fixed port; adjust if 7860 is occupied.
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_api=False,
        inbrowser=False,
        prevent_thread_lock=False,  # keep server alive until Ctrl+C
    )
