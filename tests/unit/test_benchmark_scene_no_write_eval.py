"""``openral benchmark scene --no-write-eval`` non-mutating smoke-run mode.

``benchmark scene`` defaults its eval JSON to ``rskills/<dir>/eval/
scene_<scene_id>.json`` — a *tracked* path. A quick ``--n-episodes 1``
smoke run therefore overwrote committed eval results (and ``--no-update-
manifest`` only suppressed the ``rskill.yaml`` edit, not the JSON write).
``--no-write-eval`` makes the run fully non-mutating: the rollout still
executes, but ``_persist_scene_eval`` writes nothing.

CLAUDE.md §1.11 — exercised with a real ``RSkillEvalResult`` loaded from a
shipped ``rskills/.../eval/*.json`` fixture, no mocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openral_cli.main import _persist_scene_eval
from openral_core import RSkillEvalResult

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _a_real_result() -> RSkillEvalResult:
    for p in sorted((_REPO_ROOT / "rskills").glob("*/eval/*.json")):
        return RSkillEvalResult.model_validate(json.loads(p.read_text()))
    pytest.skip("no shipped eval JSON fixture under rskills/*/eval/")


def test_no_write_eval_writes_nothing(tmp_path: Path) -> None:
    """``write_eval=False`` persists nothing and reports it via the return value."""
    out = tmp_path / "eval" / "scene_x.json"
    wrote = _persist_scene_eval(_a_real_result(), out, write_eval=False)
    assert wrote is False
    assert not out.exists()
    assert not out.parent.exists()  # did not even create the (tracked) eval/ dir


def test_write_eval_persists_a_valid_result(tmp_path: Path) -> None:
    """``write_eval=True`` writes a JSON that round-trips as RSkillEvalResult."""
    result = _a_real_result()
    out = tmp_path / "eval" / "scene_x.json"
    wrote = _persist_scene_eval(result, out, write_eval=True)
    assert wrote is True
    assert out.exists()
    RSkillEvalResult.model_validate(json.loads(out.read_text()))
