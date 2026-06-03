"""Config loader: reads code/config.yaml and validates that every path exists."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "code" / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Load config.yaml as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def validate_paths(cfg: dict[str, Any], require_data: bool = True) -> list[str]:
    """Return a list of missing paths. If require_data is False, skips data-side paths.
    Used to enable env-only smoke tests on machines without the data drive mounted."""
    missing: list[str] = []
    for section, items in cfg.items():
        if not isinstance(items, dict):
            continue
        if section == "results":
            continue  # results dirs are created on demand
        if not require_data and section != "repo_root":
            continue
        for key, value in items.items():
            if "in_zip" in key:
                continue  # internal zip member, not a filesystem path
            p = Path(value)
            if not p.exists():
                missing.append(f"{section}.{key}: {value}")
    return missing
