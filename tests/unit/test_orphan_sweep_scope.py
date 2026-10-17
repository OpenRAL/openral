"""The orphan sweep only ever reaps this CLI's own graph: same domain, same checkout (#418).

On Spark (2026-10-17) two sessions ran deploy sims from different worktrees on the same
host. The argv sweep (``_kill_orphan_openral_graph_processes``) matched every deploy graph
of the user, so one session's deploy start or exit could SIGKILL the other's live graph.
It now requires the process's own environment to carry this graph's ``ROS_DOMAIN_ID`` and
checkout (``OPENRAL_DEPLOY_GRAPH_ROOT``, set on every launch env). Real processes: three
fake graph members differing only in domain and checkout.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the sweep reads /proc")


def _fake_member(domain: str, root: str) -> subprocess.Popen[bytes]:
    env = {**os.environ, "ROS_DOMAIN_ID": domain, "OPENRAL_DEPLOY_GRAPH_ROOT": root}
    # The needle sits in argv, as a real `ros2 launch ... deploy_e2e.launch.py` has it.
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "deploy_e2e.launch.py"], env=env
    )


def test_the_sweep_spares_other_domains_and_other_checkouts(tmp_path: Path) -> None:
    from openral_cli.deploy_sim import _kill_orphan_openral_graph_processes

    mine_root, other_root = str(tmp_path / "wt-a"), str(tmp_path / "wt-b")
    mine = _fake_member("91", mine_root)
    other_domain = _fake_member("92", mine_root)
    other_checkout = _fake_member("91", other_root)
    members = (mine, other_domain, other_checkout)
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not all(
            b"deploy_e2e" in Path(f"/proc/{p.pid}/cmdline").read_bytes() for p in members
        ):
            time.sleep(0.01)
        env = {"ROS_DOMAIN_ID": "91", "OPENRAL_DEPLOY_GRAPH_ROOT": mine_root}
        assert _kill_orphan_openral_graph_processes(env) == 1
        assert mine.wait(timeout=5) == -9
        for p in (other_domain, other_checkout):
            with pytest.raises(subprocess.TimeoutExpired):
                p.wait(timeout=0.5)
    finally:
        for p in members:
            p.kill()
            p.wait()
