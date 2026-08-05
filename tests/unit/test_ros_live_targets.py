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

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "ros_live_tests.sh"
_INTEGRATION = _REPO_ROOT / "tests" / "integration"


def _script_targets() -> set[str]:
    """The repo-relative paths inside the TARGETS=( … ) block."""
    body = _SCRIPT.read_text(encoding="utf-8")
    block = re.search(r"TARGETS=\((.*?)\)", body, flags=re.DOTALL)
    assert block is not None, f"no TARGETS=() block in {_SCRIPT}"
    return {
        line.strip() for line in block.group(1).splitlines() if line.strip().startswith("tests/")
    }


def test_every_live_gated_integration_test_is_in_targets() -> None:
    gated = {
        f"tests/integration/{p.name}"
        for p in _INTEGRATION.glob("test_*.py")
        if "OPENRAL_TEST_ROS_LIVE" in p.read_text(encoding="utf-8")
    }
    assert gated, "no live-gated integration tests found — glob or gate string moved?"
    missing = gated - _script_targets()
    assert not missing, (
        f"live-ROS-gated tests missing from {_SCRIPT.relative_to(_REPO_ROOT)} TARGETS "
        f"(they will run on NO CI surface): {sorted(missing)}"
    )


def test_targets_only_lists_files_that_exist() -> None:
    stale = {t for t in _script_targets() if not (_REPO_ROOT / t).exists()}
    assert not stale, f"TARGETS lists files that do not exist: {sorted(stale)}"
