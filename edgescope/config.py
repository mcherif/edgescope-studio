from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Set, Tuple
import yaml

# Project root: parent directory of the top-level `edgescope/` package
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_classes_config() -> Tuple[Set[str], Dict[str, str]]:
    """
    Load class whitelist and aliases from config/classes.yaml.

    Returns:
        keep: set of class names to keep.
        aliases: mapping of alternate names -> canonical name.
    """
    cfg_path = PROJECT_ROOT / "config" / "classes.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    keep = set(data.get("classes", []))
    aliases = data.get("aliases", {}) or {}
    return keep, aliases
