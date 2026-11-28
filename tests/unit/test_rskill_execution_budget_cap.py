"""A manifest's ``latency_budget.max_execution_s`` caps every goal's ``deadline_s``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def _manifest(budget: float | None) -> SimpleNamespace:
    return SimpleNamespace(latency_budget=SimpleNamespace(max_execution_s=budget))


def test_manifest_budget_caps_a_longer_goal_deadline_and_fills_the_sentinel() -> None:
    from openral_rskill_ros.rskill_runner_node import _resolved_deadline_s

    capped = _manifest(60.0)
    assert _resolved_deadline_s(600.0, capped) == (60.0, True)  # the goal can't buy more time
    assert _resolved_deadline_s(20.0, capped) == (20.0, False)  # less is the goal's call
    assert _resolved_deadline_s(0.0, capped) == (60.0, True)  # sentinel -> manifest
    assert _resolved_deadline_s(600.0, _manifest(None)) == (600.0, False)  # no ceiling
    assert _resolved_deadline_s(0.0, None, _manifest(30.0)) == (30.0, True)  # first that has one


def test_a_real_manifest_budget_caps_the_goal() -> None:
    from openral_rskill.loader import parse_rskill_manifest
    from openral_rskill_ros.rskill_runner_node import _resolved_deadline_s

    root = Path(__file__).resolve().parents[2]
    manifest = parse_rskill_manifest(root / "rskills" / "smolvla-libero" / "rskill.yaml")
    assert _resolved_deadline_s(600.0, manifest) == (60.0, True)
