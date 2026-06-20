"""Regression test: the RLBench scene sidecar must NOT crash a benchmark run
when RLBench's sampling-based motion planner fails to reach a predicted keypose.

ADR-0062. ``EndEffectorPoseViaPlanning`` is stochastic — on some seeds it raises
``InvalidActionError`` / ``IKError`` / ``ConfigurationPathError`` because the
keypose is unreachable (no path / IK failure / collision). The reference 3D
Diffuser Actor evaluator counts that as a *failed episode* and keeps going; if
the sidecar instead re-raises, the openral backend turns it into a fatal
``ROSRuntimeError`` and the whole multi-episode run aborts (observed live
2026-06-20 on ``open_drawer`` — one attempt crashed, the rerun passed 2/2).

The sidecar's top-level imports are py3.12-safe (rlbench/pyrep are imported
lazily inside ``_RLBenchScene.build``), so this runs as a plain unit test with
no CoppeliaSim provisioning. Planner exceptions are matched by class name so the
assertion needs no rlbench install — a stand-in exception named
``InvalidActionError`` reproduces the real failure mode (CLAUDE.md §1.11: real
control flow, no mock of the code under test).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "rlbench_sidecar", REPO_ROOT / "tools" / "rlbench_sidecar.py"
)
assert _spec is not None and _spec.loader is not None
rlbench_sidecar = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rlbench_sidecar
_spec.loader.exec_module(rlbench_sidecar)


# A stand-in for rlbench.backend.exceptions.InvalidActionError — the sidecar
# matches planner failures by class name, so the name is what matters.
class InvalidActionError(Exception):
    """No path could be found (mirrors the RLBench peract-fork exception)."""


class _UnplannableTask:
    """A task whose ``step`` always raises a planner path-failure."""

    def __init__(self) -> None:
        self.calls = 0

    def step(self, action: Any) -> Any:
        self.calls += 1
        raise InvalidActionError("A path could not be found.")


class _RealFaultTask:
    """A task whose ``step`` raises a genuine (non-planner) bug."""

    def step(self, action: Any) -> Any:
        raise ValueError("genuine programming error")


def _scene() -> Any:
    args = argparse.Namespace(max_tries=10, success_key="is_success", variation=0)
    scene = rlbench_sidecar._RLBenchScene(args)
    # Stand in for the wrapped obs cached by reset()/a prior successful step.
    scene._last_wrapped_obs = {
        "state": np.zeros(8, dtype=np.float32),
        "task": "open the bottom drawer",
    }
    scene._arm_action_mode = None
    return scene


def test_planner_path_failure_ends_episode_as_failure_not_a_crash() -> None:
    scene = _scene()
    scene._task = _UnplannableTask()

    reply = scene.step(np.zeros(8, dtype=np.float32))

    assert reply["terminated"] is True
    assert reply["truncated"] is False
    assert reply["reward"] == 0.0
    assert reply["info"]["is_success"] is False
    # The cached observation is returned (no fresh obs exists on a planner miss).
    assert reply["observation"] is scene._last_wrapped_obs


def test_genuine_error_still_propagates() -> None:
    scene = _scene()
    scene._task = _RealFaultTask()

    with pytest.raises(ValueError, match="genuine programming error"):
        scene.step(np.zeros(8, dtype=np.float32))


@pytest.mark.parametrize(
    ("exc_name", "expected"),
    [
        ("InvalidActionError", True),
        ("IKError", True),
        ("ConfigurationPathError", True),
        ("ValueError", False),
        ("RuntimeError", False),
    ],
)
def test_is_planner_path_failure_matches_by_name(exc_name: str, expected: bool) -> None:
    exc = type(exc_name, (Exception,), {})("boom")
    assert rlbench_sidecar._is_planner_path_failure(exc) is expected
