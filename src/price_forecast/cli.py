"""Lightweight CLI wrapper.

Dynamically imports the monolithic script from the repository root and
delegates to its ``main()`` entry-point.  No modifications are made to
the original script — this file merely resolves the path and forwards
the call so that the package can be invoked via::

    price-forecast train --history data.csv ...
    python -m price_forecast train --history data.csv ...
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the original monolithic script.
# Priority:
#   1. Environment variable  PRICE_FORECAST_SCRIPT  (absolute path)
#   2. Sibling file in the repo root  (../../rt_forecast_b_route_v2_6_9_annotated.py)
# ---------------------------------------------------------------------------

_SCRIPT_NAME = "rt_forecast_b_route_v2_6_9_annotated.py"

# cli.py -> price_forecast/ -> src/ -> repo root (worktree root)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_script_path() -> Path:
    """Return the absolute path to the monolithic forecast script."""
    env = os.environ.get("PRICE_FORECAST_SCRIPT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"PRICE_FORECAST_SCRIPT points to non-existent file: {p}")

    candidate = _REPO_ROOT / _SCRIPT_NAME
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Cannot find {_SCRIPT_NAME!r} in repo root {_REPO_ROOT}. "
        f"Set PRICE_FORECAST_SCRIPT to the full path."
    )


def _load_script():
    """Load the monolithic script as a Python module (preserving its top-level state)."""
    script_path = _resolve_script_path()
    spec = importlib.util.spec_from_file_location(
        "rt_forecast_main", str(script_path),
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    # Make the script's directory the first entry in sys.path so that any
    # relative imports / data lookups inside it still work.
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    """Entry-point that delegates to the monolithic script's ``main()``."""
    mod = _load_script()
    mod.main()


if __name__ == "__main__":
    main()
