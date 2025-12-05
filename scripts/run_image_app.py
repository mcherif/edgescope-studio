from __future__ import annotations

from typing import Any
from pathlib import Path
import sys
import socket

# Ensure project root is on sys.path so we can import edgescope.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edgescope.ui.image_app import create_app

# --- Work around Gradio / gradio_client JSON-schema bug ---
import gradio_client.utils as gc_utils  # type: ignore[attr-defined]

_orig_get_type = getattr(gc_utils, "get_type", None)
_orig_json_to_py = getattr(gc_utils, "_json_schema_to_python_type", None)


def _safe_get_type(schema: Any) -> str:
    # Some Gradio versions pass a bool instead of a dict into get_type(),
    # which causes: TypeError: argument of type 'bool' is not iterable
    if not isinstance(schema, dict):
        # Degrade to a generic type; this only affects API docs,
        # not actual app behavior.
        return "any"
    return _orig_get_type(schema)  # type: ignore[misc]


def _safe_json_schema_to_python_type(schema: Any, defs: Any) -> str:
    # Guard against boolean schemas (e.g., additionalProperties: True)
    # which break gradio_client.utils._json_schema_to_python_type
    if isinstance(schema, bool):
        return "any"
    return _orig_json_to_py(schema, defs)  # type: ignore[misc]


if _orig_get_type is not None:
    gc_utils.get_type = _safe_get_type  # type: ignore[assignment]
if _orig_json_to_py is not None:
    gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type  # type: ignore[assignment]
# ----------------------------------------------------------


if __name__ == "__main__":
    # Prefer a fixed port for easy access; change if 7860 is busy.
    port = 7860

    app = create_app()
    # Bind to localhost and disable API docs to sidestep gradio_client schema parsing issues.
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_api=False,
        inbrowser=False,
        # Keep the thread alive so the server stays up until you Ctrl+C.
        prevent_thread_lock=False,
    )
