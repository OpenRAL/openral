# SPDX-License-Identifier: Apache-2.0
"""A leaked graph node poisons every later round on its domain.

`ros2 launch` starts the octomap nodes in their own session, so the probe's
`os.killpg` never reached them: each round leaked an `octomap_server_node` +
`octomap_voxel_bridge` pair. By 2026-09-10, 46 such orphans had accumulated
over 23 h — 30 % of a core, 826 MB, and 253 Fast-DDS `/dev/shm` segments. The
damage is not the memory but the `fastrtps_port<N>_el` lock file an orphan
keeps open: the next run on that domain fails `open_and_lock_file`, the policy
is handed 0 chunks, and the round buckets `harness-error`.

That mechanism, not voxel resolution, produced the apparent 15 mm mortality in
the first resolution A/B — the one lane that was interleaved between arms
(`baguette`) shows 3 failures against 3, while the lane that ran the 15 mm arm
strictly last (`sink_cup`) shows 7 against 0.

`ROS_DOMAIN_ID` is the ownership key. `--ros-args` is what separates a node
from the probe and its shell, which share the domain and must survive the
sweep.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import _ceiling_probe as probe  # type: ignore[import-not-found]  # reason: tools/ is not a package

DOMAIN = "231"  # reason: outside the 60-90 band the batteries use, so a live round is never swept


def _spawn(*extra: str) -> subprocess.Popen[bytes]:
    """A real process on DOMAIN, in its own session — as `ros2 launch` leaves them."""
    env = {**os.environ, "ROS_DOMAIN_ID": DOMAIN}
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", *extra],
        env=env,
        start_new_session=True,
    )
    time.sleep(0.3)  # reason: /proc/<pid>/environ is not readable until execve completes
    return proc


@pytest.fixture
def survivors() -> Iterator[list[subprocess.Popen[bytes]]]:
    spawned: list[subprocess.Popen[bytes]] = []
    yield spawned
    for proc in spawned:
        proc.kill()
        proc.wait(timeout=10)


def test_reaps_a_node_the_process_group_kill_cannot_reach(survivors: list) -> None:
    node = _spawn("--ros-args", "-r", "__node:=octomap_voxel_bridge")
    survivors.append(node)

    assert probe._reap_domain(DOMAIN, signal.SIGKILL) >= 1
    assert node.wait(timeout=10) is not None
    assert node.poll() is not None


def test_spares_the_probe_and_its_shell_on_the_same_domain(survivors: list) -> None:
    """The probe exports ROS_DOMAIN_ID too; only `--ros-args` marks a node."""
    not_a_node = _spawn()
    survivors.append(not_a_node)

    probe._reap_domain(DOMAIN, signal.SIGKILL)
    time.sleep(0.5)
    assert not_a_node.poll() is None


def test_leaves_other_domains_alone(survivors: list) -> None:
    """Workers share a host; reaping domain N must not touch worker N+1."""
    env = {**os.environ, "ROS_DOMAIN_ID": "232"}
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", "--ros-args"],
        env=env,
        start_new_session=True,
    )
    survivors.append(other)
    time.sleep(0.3)

    probe._reap_domain(DOMAIN, signal.SIGKILL)
    time.sleep(0.5)
    assert other.poll() is None
