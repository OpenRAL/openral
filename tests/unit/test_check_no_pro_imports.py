"""Tests for the open-repo ``openral_pro*`` import guard (``tools/check_no_pro_imports.py``).

``test_repo_has_no_pro_imports`` runs the guard over every tracked file — the
regression it exists for (8 direct ``openral_pro_trt`` imports had leaked into
``act.py``, ``smolvla.py``, ``reader.py`` and ``detector_runner.py``). The rest
re-seed each import form the guard must catch, and the near-miss names it must
not (``openral_prompt_router``, the package name inside an error string).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "check_no_pro_imports", REPO_ROOT / "tools" / "check_no_pro_imports.py"
)
assert _spec is not None and _spec.loader is not None
check_no_pro_imports = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check_no_pro_imports
_spec.loader.exec_module(check_no_pro_imports)
find = check_no_pro_imports.find_pro_imports


def test_repo_has_no_pro_imports() -> None:
    assert check_no_pro_imports.main([]) == 0


@pytest.mark.parametrize(
    ("source", "module"),
    [
        ("import openral_pro_trt", "openral_pro_trt"),
        ("import openral_pro", "openral_pro"),
        ("import os, openral_pro.fleet as f", "openral_pro.fleet"),
        ("from openral_pro_trt.nvbufsurface import load", "openral_pro_trt.nvbufsurface"),
        ("def f():\n    from openral_pro_trt import act_trt", "openral_pro_trt"),
        (
            "from importlib import import_module\nimport_module('openral_pro_trt.act_nvmm')",
            "openral_pro_trt.act_nvmm",
        ),
        ("import importlib\nimportlib.import_module('openral_pro_trt')", "openral_pro_trt"),
        ("__import__('openral_pro_trt.smolvla_trt')", "openral_pro_trt.smolvla_trt"),
    ],
)
def test_catches_each_import_form(source: str, module: str) -> None:
    assert [m for _, m in find(source)] == [module]


@pytest.mark.parametrize(
    "source",
    [
        "import openral_prompt_router",
        "from openral_prompt_router import PromptRouterNode",
        "msg = 'install the private openral-pro-trt package'",
        "x = 'openral_pro_trt.smolvla_trt'  # a string, not an import",
        "from . import openral_pro_local",
        "import_module(name)",
    ],
)
def test_ignores_near_misses(source: str) -> None:
    assert find(source) == []


def test_main_reports_and_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\nfrom openral_pro_trt.nvbufsurface import load\n")
    assert check_no_pro_imports.main([str(bad)]) == 1
    assert f"{bad}:2:" in capsys.readouterr().out
