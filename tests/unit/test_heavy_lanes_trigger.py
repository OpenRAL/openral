"""Tests for the heavy-lanes auto-trigger (``tools/heavy_lanes_trigger.py``).

The readiness verdict is a pure function over PR state, so it is exercised
directly. The check names it keys on are cross-checked against the REAL
workflow files, so renaming a job cannot silently make the trigger wait for a
check that no longer exists (never ready) or treat its own lanes as a
precondition (never ready either).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.heavy_lanes_trigger import (  # noqa: E402
    DEFAULT_REQUIRED_CHECKS,
    OWN_CHECK_NAMES,
    CheckRunState,
    LaneRunState,
    PullRequestState,
    readiness,
)

REQUIRED = list(DEFAULT_REQUIRED_CHECKS)


def _workflow(name: str) -> dict[Any, Any]:
    data: dict[Any, Any] = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return data


def _job_names(workflow: dict[Any, Any]) -> set[str]:
    return {job.get("name", job_id) for job_id, job in workflow["jobs"].items()}


def _green(name: str) -> CheckRunState:
    return CheckRunState(name=name, status="completed", conclusion="success")


def _ready_pr(**overrides: Any) -> PullRequestState:
    fields: dict[str, Any] = {
        "number": 302,
        "draft": False,
        "head_ref": "claude/heavy-lanes-ci-config-ot37h4",
        "head_sha": "2442ff62e3d0d80c68e8338e40880ebd49defe74",
        "base_ref": "master",
        "same_repo": True,
        "behind_by": 0,
        # The check runs PR #302's head actually carried (test-selective,
        # quality, DCO), incl. skipped matrix legs.
        "checks": [
            _green("select"),
            _green("core_full (0)"),
            CheckRunState(name="core_selected", status="completed", conclusion="skipped"),
            _green("select-and-test"),
            _green("quality"),
            _green("Verify Signed-off-by"),
        ],
        "statuses": [],
        "unresolved_threads": 0,
        "lane_runs": [],
    }
    fields.update(overrides)
    return PullRequestState(**fields)


def test_own_check_names_match_heavy_lanes_workflows() -> None:
    """The lanes' own checks, and the trigger's (posted on the PR head by a
    workflow_run-triggered sweep), are exactly what the verdict ignores."""
    own = _job_names(_workflow("heavy-lanes.yml")) | _job_names(
        _workflow("heavy-lanes-trigger.yml")
    )
    assert own == set(OWN_CHECK_NAMES)


def test_required_checks_are_real_pr_job_names() -> None:
    pr_jobs = set()
    for name in ("test-selective.yml", "quality.yml", "dco.yml"):
        pr_jobs |= _job_names(_workflow(name))
    missing = set(DEFAULT_REQUIRED_CHECKS) - pr_jobs
    assert not missing, f"trigger waits on checks no PR workflow posts: {missing}"


def test_trigger_listens_to_every_pull_request_workflow() -> None:
    pr_workflows = {
        _workflow(path.name)["name"]
        for path in WORKFLOWS.glob("*.yml")
        if "pull_request" in (_workflow(path.name).get(True) or {})
    }
    trigger = _workflow("heavy-lanes-trigger.yml")
    assert set(trigger[True]["workflow_run"]["workflows"]) == pr_workflows


def test_ready_pr_is_ready() -> None:
    verdict = readiness(_ready_pr(), REQUIRED)
    assert verdict.ready, verdict.reasons


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"draft": True}, "draft PR"),
        ({"same_repo": False}, "fork PR"),
        ({"behind_by": 3}, "3 commit(s) behind master"),
        ({"unresolved_threads": 2}, "2 unresolved review thread(s)"),
        ({"statuses": ["success", "pending"]}, "a commit status is pending"),
        ({"lane_runs": [LaneRunState(status="in_progress")]}, "already started"),
        (
            {"lane_runs": [LaneRunState(status="completed", conclusion="failure")]},
            "already started",
        ),
    ],
)
def test_each_unmet_condition_blocks(overrides: dict[str, Any], reason: str) -> None:
    verdict = readiness(_ready_pr(**overrides), REQUIRED)
    assert not verdict.ready
    assert any(reason in r for r in verdict.reasons), verdict.reasons


def test_red_or_running_check_blocks() -> None:
    base = _ready_pr().checks
    red = [*base, CheckRunState(name="core_full (1)", status="completed", conclusion="failure")]
    running = [*base, CheckRunState(name="core_full (2)", status="in_progress")]
    assert "concluded failure" in readiness(_ready_pr(checks=red), REQUIRED).reasons[0]
    assert "is in_progress" in readiness(_ready_pr(checks=running), REQUIRED).reasons[0]


def test_required_check_not_yet_reported_blocks() -> None:
    checks = [c for c in _ready_pr().checks if c.name != "quality"]
    verdict = readiness(_ready_pr(checks=checks), REQUIRED)
    assert verdict.reasons == ["check 'quality' has not reported yet"]


def test_own_checks_and_cancelled_runs_do_not_block() -> None:
    """A cancelled earlier dispatch leaves heavy-lanes' own checks behind; re-dispatch."""
    checks = [
        *_ready_pr().checks,
        CheckRunState(name="lanes-select", status="completed", conclusion="cancelled"),
        CheckRunState(name="lane (sim, tests/sim)", status="completed", conclusion="cancelled"),
        CheckRunState(name="heavy-lanes", status="completed", conclusion="cancelled"),
        CheckRunState(name="heavy-lanes-trigger", status="in_progress"),
    ]
    runs = [LaneRunState(status="completed", conclusion="cancelled")]
    verdict = readiness(_ready_pr(checks=checks, lane_runs=runs), REQUIRED)
    assert verdict.ready, verdict.reasons
