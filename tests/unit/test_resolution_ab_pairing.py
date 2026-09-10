# SPDX-License-Identifier: Apache-2.0
"""The A/B's two arms have to meet the same host, so they have to run together.

The primary endpoint is the *paired* over-approximation per stop, which means
nothing unless a scene's 25 mm and 15 mm lanes saw the same machine. The first
version of `tools/resolution_ab.sh` wrote a scene-interleaved worklist and fed
it to `xargs -P`: each worker holds its slot for all ROUNDS rounds, so only the
first pair ever overlapped and every later lane drifted by about one lane's
duration.

In the 2026-09-10 run that produced `baguette` paired (3/10 unreadable runs
against 3/10, Fisher p = 1.0) and `sink_cup` not — its 25 mm arm ran alone
18:50-19:19 and its 15 mm arm alone 19:19-19:37, after ~30 rounds of host wear
(7/10 against 0/10, p = 0.003). That read as a resolution effect until the
order was checked, so the scheduling is asserted here rather than described in
a comment.

Exercises the real script through its `RUN_WORKER_CMD` seam — no stub of the
scheduler itself, only of the 10-hour lane body it drives.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "resolution_ab.sh"

# The script sources the ROS distro and this worktree's own overlay before it
# schedules anything, so a host without a built workspace cannot run it at all.
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
def lane_recorder(tmp_path: Path) -> Path:
    """A stand-in lane that records when it started and stopped, then exits."""
    script = tmp_path / "lane.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s %s %s start %s\\n" "$1" "$2" "$3" "$(date +%s.%N)" >> "$LANE_LOG"\n'
        "sleep 1\n"
        'printf "%s %s %s end %s\\n" "$1" "$2" "$3" "$(date +%s.%N)" >> "$LANE_LOG"\n'
    )
    script.chmod(0o755)
    return script


def _windows(lane_log: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """`(scene, resolution) -> (start, end)` from the recorder's log."""
    out: dict[tuple[str, str], list[float]] = {}
    for line in lane_log.read_text().splitlines():
        _idx, scene, res, event, stamp = line.split()
        out.setdefault((scene, res), [0.0, 0.0])[0 if event == "start" else 1] = float(stamp)
    return {k: (v[0], v[1]) for k, v in out.items()}


def test_both_arms_of_a_scene_overlap_and_scenes_do_not(
    tmp_path: Path, lane_recorder: Path
) -> None:
    lane_log = tmp_path / "lanes.log"
    out = tmp_path / "ab"
    env = {
        **os.environ,
        "ROUNDS": "1",
        "SIDECAR_BOOT_S": "0",
        "OUT": str(out),
        "RUN_WORKER_CMD": str(lane_recorder),
        "LANE_LOG": str(lane_log),
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "RESOLUTION_AB_DONE (0 lane(s) non-zero)" in proc.stdout

    windows = _windows(lane_log)
    scenes = sorted({scene for scene, _ in windows})
    assert len(scenes) >= 2, windows

    for scene in scenes:
        coarse = windows[(scene, "0.025")]
        fine = windows[(scene, "0.015")]
        # Overlap: each arm starts before the other has finished.
        assert coarse[0] < fine[1] and fine[0] < coarse[1], (
            f"{scene}: arms did not overlap — 0.025 {coarse}, 0.015 {fine}. "
            "This is the sink_cup failure; the paired endpoint is void without it."
        )

    # And the barrier holds: no scene starts before the previous one is done.
    by_scene = {
        scene: (
            min(windows[(scene, r)][0] for r in ("0.025", "0.015")),
            max(windows[(scene, r)][1] for r in ("0.025", "0.015")),
        )
        for scene in scenes
    }
    ordered = sorted(by_scene.values())
    for earlier, later in pairwise(ordered):
        assert earlier[1] <= later[0], (
            f"scenes overlapped: {earlier} then {later}. Three live graphs is more "
            "than this host carries, and it breaks the pairing of both."
        )


def test_every_lane_gets_its_own_ros_domain(tmp_path: Path, lane_recorder: Path) -> None:
    """Two concurrent lanes on one domain would see each other's nodes."""
    lane_log = tmp_path / "lanes.log"
    out = tmp_path / "ab"
    domain_log = tmp_path / "domains.log"
    recorder = tmp_path / "lane_domain.sh"
    recorder.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s %s %s %s\\n" "$2" "$3" "$ROS_DOMAIN_ID" '
        '"$OPENRAL_OCTOMAP_RESOLUTION_M" >> "$DOMAIN_LOG"\n'
        'printf "%s %s %s start 0\\n" "$1" "$2" "$3" >> "$LANE_LOG"\n'
        'printf "%s %s %s end 1\\n" "$1" "$2" "$3" >> "$LANE_LOG"\n'
    )
    recorder.chmod(0o755)
    env = {
        **os.environ,
        "ROUNDS": "1",
        "SIDECAR_BOOT_S": "0",
        "OUT": str(out),
        "RUN_WORKER_CMD": str(recorder),
        "LANE_LOG": str(lane_log),
        "DOMAIN_LOG": str(domain_log),
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    rows = [line.split() for line in domain_log.read_text().splitlines()]
    domains = [row[2] for row in rows]
    assert len(domains) == len(set(domains)), f"domain reused across lanes: {rows}"
    # And the arm's resolution reaches the lane, which is the only difference
    # between the two arms of a pair.
    for scene, res, _domain, exported in rows:
        assert exported == res, f"{scene}: lane got {exported}, lane is {res}"
