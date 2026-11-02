"""Resolve vendored source packages inside this repository."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]


def enable_sources():
    for path in (
        ROOT / "external/imitation_deps",
        ROOT / "external/nflows",
        ROOT / "external/imitation/src",
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"Required AIRL source/dependency directory: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
