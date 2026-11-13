"""Guard: every live-ROS-gated integration test is listed in ros_live_tests.sh.

``scripts/ros_live_tests.sh`` is the ONLY runner of the
``OPENRAL_TEST_ROS_LIVE=1`` suite (the docker-build workflow execs it;
test-selective has no rclpy, so every gated test importorskips there). A gated
file missing from its TARGETS list therefore runs on NO CI surface — which
happened to the first new live test added after the list was written
(``test_reasoner_palette_primes_vram_gate.py``). This test makes that drift a
unit-test failure instead of a silent coverage hole.

Run with:
    uv run pytest tests/unit/test_ros_live_targets.py -v
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.unit.conftest import _script_targets

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "ros_live_tests.sh"
_INTEGRATION = _REPO_ROOT / "tests" / "integration"
_DURATIONS = _REPO_ROOT / "scripts" / "ros_live_durations.txt"


def test_every_live_gated_integration_test_is_in_targets() -> None:
    gated = {
        f"tests/integration/{p.name}"
        for p in _INTEGRATION.glob("test_*.py")
        if "OPENRAL_TEST_ROS_LIVE" in p.read_text(encoding="utf-8")
    }
    assert gated, "no live-gated integration tests found — glob or gate string moved?"
    missing = gated - _script_targets(_SCRIPT)
    assert not missing, (
        f"live-ROS-gated tests missing from {_SCRIPT.relative_to(_REPO_ROOT)} TARGETS "
        f"(they will run on NO CI surface): {sorted(missing)}"
    )


def test_targets_only_lists_files_that_exist() -> None:
    stale = {t for t in _script_targets(_SCRIPT) if not (_REPO_ROOT / t).exists()}
    assert not stale, f"TARGETS lists files that do not exist: {sorted(stale)}"


def _selection(shard: str | None) -> list[str]:
    """The files the script runs under ``OPENRAL_LIVE_SHARD=shard`` (unset for None).

    ``OPENRAL_LIVE_LIST=1`` prints the selection the script would hand to pytest and
    stops, through the same selection code a real run takes. The unit here is that
    selection: the tests themselves run in docker-build's live lane, so this is no
    substitute for running them (CLAUDE.md §1.11).
    """
    env = {k: v for k, v in os.environ.items() if k != "OPENRAL_LIVE_SHARD"}
    env["OPENRAL_LIVE_LIST"] = "1"
    if shard is not None:
        env["OPENRAL_LIVE_SHARD"] = shard
    done = subprocess.run(
        ["bash", str(_SCRIPT)], env=env, capture_output=True, text=True, check=True, timeout=30
    )
    return done.stdout.split()


def _durations() -> dict[str, int]:
    """``scripts/ros_live_durations.txt``: recorded wall seconds per isolated file."""
    rows = (line.split(maxsplit=1) for line in _DURATIONS.read_text(encoding="utf-8").splitlines())
    return {f.strip(): int(c) for c, f in (r for r in rows if r and r[0].isdigit())}


@pytest.mark.parametrize("shards", [1, 3, 4, 7])
def test_the_shards_run_every_target_exactly_once(shards: int) -> None:
    """docker-build runs I/K for every I: together, the unsharded run, nothing twice."""
    run = [t for i in range(1, shards + 1) for t in _selection(f"{i}/{shards}")]
    assert len(run) == len(set(run)), sorted({t for t in run if run.count(t) > 1})
    assert sorted(run) == sorted(_selection(None))
    assert set(run) == _script_targets(_SCRIPT)


def test_the_quiet_tail_is_every_target_with_a_wall_clock_test() -> None:
    marked = {
        t
        for t in _script_targets(_SCRIPT)
        if "pytest.mark.wall_clock" in (_REPO_ROOT / t).read_text(encoding="utf-8")
    }
    assert "tests/sim/safety/test_kernel_latency_soak.py" in marked
    assert sorted(_selection("quiet")) == sorted(marked)


def test_the_shards_split_the_recorded_isolated_time_evenly() -> None:
    """Longest first onto the least-loaded shard: within the longest file of even."""
    cost = _durations()
    loads = [sum(cost.get(t, 0) for t in _selection(f"{i}/4")) for i in range(1, 5)]
    assert max(loads) - min(loads) <= max(cost.values()), loads


@pytest.mark.parametrize("shard", ["0/4", "5/4", "4", "a/b", "1/0", "010/012", "08/10"])
def test_a_malformed_shard_is_refused(shard: str) -> None:
    with pytest.raises(subprocess.CalledProcessError) as raised:
        _selection(shard)
    assert raised.value.returncode == 2
