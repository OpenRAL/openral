"""Extension-less ROS entry-point scripts import only names that exist.

``packages/*/scripts/*`` are Python files without a ``.py`` suffix, so ``--include=*.py``
greps, ruff's default file discovery and mypy all skip them. A symbol moved between
modules (``merge_deploy_sensors`` left ``openral_rskill_ros.sensor_leg`` for
``openral_core``) then leaves a stale ``from ... import`` that only fails at node start —
on ``runtime_node`` that was every real deploy with ``--config``. This walks each script's
``from openral_* import ...`` statements (deferred ones included) and checks every name
resolves in the real module.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _python_scripts() -> list[Path]:
    scripts = []
    for path in sorted(REPO_ROOT.glob("packages/*/scripts/*")):
        if not (path.is_file() and path.suffix == ""):
            continue
        shebang = path.read_bytes().split(b"\n", 1)[0]
        if shebang.startswith(b"#!") and b"python" in shebang:
            scripts.append(path)
    return scripts


@pytest.mark.parametrize("script", _python_scripts(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_script_imports_resolve(script: Path) -> None:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.module and node.level == 0):
            continue
        if not node.module.startswith("openral_"):
            continue
        try:
            module = importlib.import_module(node.module)
        except ImportError as exc:  # a module that needs the ROS overlay
            pytest.skip(f"{node.module} not importable here: {exc}")
        for alias in node.names:
            assert hasattr(module, alias.name), (
                f"{script.name}:{node.lineno} imports {alias.name!r} from {node.module}, "
                "which no longer defines it"
            )
            checked += 1
    assert checked or "openral_" not in script.read_text(encoding="utf-8")


def test_the_scan_reaches_runtime_node() -> None:
    names = [p.name for p in _python_scripts()]
    assert "runtime_node" in names
