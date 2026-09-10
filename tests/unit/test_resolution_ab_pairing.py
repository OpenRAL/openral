# SPDX-License-Identifier: Apache-2.0
"""The A/B's two arms have to meet the same host, and only one graph fits on it.

Two ways to get this wrong, both paid for. The first version fed a
scene-interleaved worklist to `xargs -P`, which does not interleave anything —
a worker holds its slot for all ROUNDS rounds, so only the first pair
overlapped. On 2026-09-10 `sink_cup` ran its arms 29 minutes apart and its
7/10-vs-0/10 unreadable-run gap read as a resolution effect; `baguette`, the
one lane that stayed paired, was 3/10 against 3/10.

Running the lanes genuinely concurrently fixed the pairing and broke the host:
baseline occupancy on q-laptop is ~9.1 GB of 15.4 and one deploy graph adds
~4.9 GB, so two exhausted all 4 GB of swap, drove load averages to 225, and
returned 9 of 20 rounds unreadable.

So the arms alternate round by round with one graph live at a time, which is
tighter pairing than concurrent lanes gave — adjacent rounds share an instant,
not merely a window. Exercises the real script through its `RUN_ROUND_CMD`
seam; only the round body is substituted, never the scheduling under test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "resolution_ab.sh"

_OVERLAY = REPO / "install" / "setup.bash"
_ROS = Path("/opt/ros/jazzy/setup.bash")
pytestmark = pytest.mark.skipif(
    not (_OVERLAY.exists() and _ROS.exists() and shutil.which("bash")),
    reason=(
        "needs /opt/ros/jazzy + this worktree's install/ overlay; the script sources both "
        "before scheduling. Run `just ros2-build` to exercise it here."
    ),
)


@pytest.fixture
def round_recorder(tmp_path: Path) -> Path:
    """A stand-in round: appends its identity and whether a peer was live."""
    script = tmp_path / "round.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        # A lock that a concurrent round would find already taken.
        'if ! mkdir "$LIVE_DIR" 2>/dev/null; then echo "CONCURRENT" >> "$ROUND_LOG"; fi\n'
        'printf "%s %s %s %s\\n" "$1" "$2" "$3" "$ROS_DOMAIN_ID" >> "$ROUND_LOG"\n'
        'rmdir "$LIVE_DIR" 2>/dev/null\n'
    )
    script.chmod(0o755)
    return script


def _run(tmp_path: Path, recorder: Path, rounds: str = "2") -> list[list[str]]:
    round_log = tmp_path / "rounds.log"
    env = {
        **os.environ,
        "ROUNDS": rounds,
        "SIDECAR_BOOT_S": "0",
        "OUT": str(tmp_path / "ab"),
        "RUN_ROUND_CMD": str(recorder),
        "ROUND_LOG": str(round_log),
        "LIVE_DIR": str(tmp_path / "live.lock"),
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "RESOLUTION_AB_DONE (0 round(s) non-zero)" in proc.stdout
    return [line.split() for line in round_log.read_text().splitlines()]


def test_arms_alternate_round_by_round(tmp_path: Path, round_recorder: Path) -> None:
    rows = _run(tmp_path, round_recorder)

    assert "CONCURRENT" not in {row[0] for row in rows}, (
        "two rounds were live at once; one deploy graph is all this host holds"
    )
    # Within a scene, the arms alternate and round numbers ascend in pairs.
    by_scene: dict[str, list[tuple[str, str]]] = {}
    for scene, res, rnd, _domain in rows:
        by_scene.setdefault(scene, []).append((res, rnd))
    assert len(by_scene) >= 2, rows
    for scene, seq in by_scene.items():
        assert seq == [(res, str(n)) for n in (1, 2) for res in ("0.025", "0.015")], (
            f"{scene}: {seq}"
        )


def test_each_arm_keeps_its_own_domain(tmp_path: Path, round_recorder: Path) -> None:
    """The arms must not share a ROS domain, or one round's leftovers reach the next."""
    rows = _run(tmp_path, round_recorder)
    domains = {res: {row[3] for row in rows if row[1] == res} for res in ("0.025", "0.015")}
    assert domains["0.025"] == {"80"}, domains
    assert domains["0.015"] == {"81"}, domains


def test_a_failing_round_is_counted_not_swallowed(tmp_path: Path) -> None:
    """A non-zero round must reach the summary line, not be lost to `set -e`."""
    recorder = tmp_path / "boom.sh"
    recorder.write_text("#!/usr/bin/env bash\nexit 3\n")
    recorder.chmod(0o755)
    env = {
        **os.environ,
        "ROUNDS": "1",
        "SIDECAR_BOOT_S": "0",
        "OUT": str(tmp_path / "ab"),
        "RUN_ROUND_CMD": str(recorder),
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "RESOLUTION_AB_DONE (8 round(s) non-zero) ===" in proc.stdout, proc.stdout[-500:]
