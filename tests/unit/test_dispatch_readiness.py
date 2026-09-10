# SPDX-License-Identifier: Apache-2.0
"""A dispatch that failed because the graph was not up yet is not a policy result.

Issue #262. The harness waits for `/openral/execute_rskill` to appear, sleeps a
fixed 5 s, and dispatches. The action server answering does not mean the graph
is assembled: the TF tree can still be two disjoint trees, and a camera the
rSkill declares can still have published nothing.

On an idle host 5 s covers it. Under contention it does not — 6
`ConnectivityException` plus 17 camera failures across the 78 goal logs of the
2026-09-10 ceiling battery, and 10 of 18 runs in the first resolution A/B. Each
returns in ~0.4 s having delivered zero action chunks, and each was scored as
the policy failing, because the graph survives and prints
`sim.task_success_final` at teardown like any ordinary non-completion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import validation_matrix as vm  # type: ignore[import-not-found]  # reason: sibling script, no stub


def _goal(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "run_goal.log"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_disjoint_tf_tree_is_a_readiness_failure(tmp_path: Path) -> None:
    """The dominant one: verbatim from `0.025/baguette-on/r08`."""
    reason = vm.dispatch_not_ready_reason(
        _goal(
            tmp_path,
            '{"failure_reason": "ConnectivityException: Could not find a connection '
            "between 'base_link' and 'panda_hand_tcp' because they are not part of the "
            'same tree.Tf has two or more unconnected trees.", "first_chunk_s": null, '
            '"latest_chunk": 0, "status": 6, "success": false, "wall_s": 0.4}',
        )
    )
    assert "ConnectivityException" in reason


def test_a_camera_that_never_published_is_a_readiness_failure(tmp_path: Path) -> None:
    """17 of the ceiling battery's 78 goal logs."""
    reason = vm.dispatch_not_ready_reason(
        _goal(
            tmp_path,
            '{"failure_reason": "ROSConfigError: XR-1 expected camera '
            '\'camera1\'; observation had none", "latest_chunk": 0, "status": 6}',
        )
    )
    assert "expected camera" in reason


@pytest.mark.parametrize(
    "body",
    [
        # A real deadline: the policy ran for its whole budget.
        '{"status": -1, "latest_chunk": 400, "success": false}',
        # A real E-stop, which must never be retried away.
        '{"status": 6, "failure_reason": "safety_estop:ROSPublishingHAL: safety stop '
        'latched", "latest_chunk": 161}',
        # A real capability mismatch — a genuine outcome, not a slow start.
        '{"status": 6, "failure_reason": "ROSCapabilityMismatch: no lidar", "latest_chunk": 0}',
    ],
)
def test_a_real_outcome_is_never_a_readiness_failure(tmp_path: Path, body: str) -> None:
    """Only the readiness class retries. Everything else is the run's own result."""
    assert vm.dispatch_not_ready_reason(_goal(tmp_path, body)) == ""


def test_delivered_chunks_outrank_the_reason(tmp_path: Path) -> None:
    """A run that produced action chunks had an assembled graph, whatever it says.

    Guards the ordering: a late TF failure in a run that had already been
    driving the arm is that run's outcome, not a reason to dispatch again.
    """
    body = (
        '{"status": 6, "failure_reason": "ConnectivityException: lost the tree", '
        '"latest_chunk": 240}'
    )
    assert vm.dispatch_not_ready_reason(_goal(tmp_path, body)) == ""


def test_a_missing_goal_log_is_not_a_readiness_failure(tmp_path: Path) -> None:
    """Absence is a harness error the existing detectors own, not a retry trigger."""
    assert vm.dispatch_not_ready_reason(tmp_path / "absent.log") == ""
