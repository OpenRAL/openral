"""A SIGKILLed ``openral deploy sim`` CLI leaves no launch node or sidecar behind (#418).

r4dr_shadow2 (Spark, 2026-10-17): the deploy CLI died without running its teardown and
the launch's nodes (each in its own process group) and the Isaac sidecar (its own
session) kept running on the ROS domain, feeding the next trials. ``_run_launch`` now
starts ``openral_cli._launch_guard``, a stdlib-only process in its own session that
watches the CLI and reaps every process it ever saw under the launch once the CLI is
gone, however it died.

Real processes only: a stand-in CLI runs the real ``_run_launch`` over a stand-in launch
tree (a child in its own process group, a grandchild in its own session, like the
sidecar), and is then SIGKILLed.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the guard reads /proc")

_TREE = textwrap.dedent(
    """
    import os, subprocess, sys, time
    out = sys.argv[1]
    sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]
    node = subprocess.Popen(sleeper, process_group=0)          # a launch node: own group
    sidecar = subprocess.Popen(sleeper, start_new_session=True)  # the sidecar: own session
    with open(out, "w") as f:
        f.write(f"{os.getpid()} {node.pid} {sidecar.pid}")
    time.sleep(120)
    """
)

_CLI = textwrap.dedent(
    """
    import sys
    from openral_cli.deploy_sim import _run_launch
    tree, out = sys.argv[1], sys.argv[2]
    _run_launch([sys.executable, tree, out], dict(__import__("os").environ), orphan_guard=True)
    """
)


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def test_a_sigkilled_cli_leaves_no_launch_node_or_sidecar(tmp_path: Path) -> None:
    tree = tmp_path / "tree.py"
    tree.write_text(_TREE)
    out = tmp_path / "pids"
    env = {**os.environ, "OPENRAL_SKIP_ORPHAN_REAP": "1"}  # never sweep this host's graphs
    cli = subprocess.Popen([sys.executable, "-c", _CLI, str(tree), str(out)], env=env)
    pids: list[int] = []
    try:
        deadline = time.monotonic() + 30.0
        while not (out.exists() and out.read_text()) and time.monotonic() < deadline:
            time.sleep(0.1)
        pids = [int(p) for p in out.read_text().split()]
        time.sleep(2.0)  # the guard snapshots the tree every half second
        assert all(_alive(p) for p in pids)
        cli.send_signal(signal.SIGKILL)
        cli.wait(timeout=5)
        deadline = time.monotonic() + 20.0
        while any(_alive(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not [p for p in pids if _alive(p)], "the launch tree outlived the CLI"
    finally:
        for p in pids:
            with contextlib.suppress(OSError):
                os.kill(p, signal.SIGKILL)
        cli.kill()


def test_without_the_guard_the_tree_outlives_a_sigkilled_cli(tmp_path: Path) -> None:
    """The failure the guard removes, reproduced (and cleaned up) for contrast."""
    tree = tmp_path / "tree.py"
    tree.write_text(_TREE)
    out = tmp_path / "pids"
    env = {**os.environ, "OPENRAL_SKIP_ORPHAN_REAP": "1"}
    cli = subprocess.Popen(
        [sys.executable, "-c", _CLI.replace("orphan_guard=True", "orphan_guard=False"),
         str(tree), str(out)],
        env=env,
    )  # fmt: skip
    pids: list[int] = []
    try:
        deadline = time.monotonic() + 30.0
        while not (out.exists() and out.read_text()) and time.monotonic() < deadline:
            time.sleep(0.1)
        pids = [int(p) for p in out.read_text().split()]
        cli.send_signal(signal.SIGKILL)
        cli.wait(timeout=5)
        time.sleep(3.0)
        assert all(_alive(p) for p in pids)
    finally:
        for p in pids:
            with contextlib.suppress(OSError):
                os.kill(p, signal.SIGKILL)


def test_a_sidecar_marked_kept_alive_survives_the_cli_that_spawned_it(tmp_path: Path) -> None:
    """`--keep-sim-open` on a deploy: the sidecar outlives its CLI for the next run to
    adopt, while the rest of the launch tree is still reaped."""
    from openral_sim.sidecar import KEPT_ALIVE_ENV

    tree = tmp_path / "tree.py"
    tree.write_text(
        _TREE.replace(
            "start_new_session=True)",
            f'start_new_session=True, env={{**os.environ, "{KEPT_ALIVE_ENV}": "1"}})',
        )
    )
    out = tmp_path / "pids"
    env = {**os.environ, "OPENRAL_SKIP_ORPHAN_REAP": "1"}
    cli = subprocess.Popen([sys.executable, "-c", _CLI, str(tree), str(out)], env=env)
    pids: list[int] = []
    try:
        deadline = time.monotonic() + 30.0
        while not (out.exists() and out.read_text()) and time.monotonic() < deadline:
            time.sleep(0.1)
        pids = [int(p) for p in out.read_text().split()]
        time.sleep(2.0)
        cli.send_signal(signal.SIGKILL)
        cli.wait(timeout=5)
        node, sidecar = pids[1], pids[2]
        deadline = time.monotonic() + 20.0
        while _alive(node) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _alive(node), "the launch node outlived the CLI"
        assert _alive(sidecar), "the kept-alive sidecar was reaped with the launch"
    finally:
        for p in pids:
            with contextlib.suppress(OSError):
                os.kill(p, signal.SIGKILL)
        cli.kill()


_REPARENTED_TREE = textwrap.dedent(
    """
    import os, subprocess, sys, time
    out = sys.argv[1]
    sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]
    spawn = (
        "import subprocess, sys; "
        f"p = subprocess.Popen({sleeper!r}, start_new_session=True, "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print(p.pid)"
    )
    # A launch child that starts a long-running process and exits at once: the
    # grandchild is reparented out of the launch tree before any snapshot sees it.
    mid = subprocess.run([sys.executable, "-c", spawn], capture_output=True, text=True)
    with open(out, "w") as f:
        f.write(f"{os.getpid()} {mid.stdout.strip()}")
    time.sleep(120)
    """
)


def test_a_process_reparented_out_of_the_tree_is_still_reaped(tmp_path: Path) -> None:
    """#421 review: a descendant whose parent exited between two snapshots lost its link
    to the launch; the launch's environment tag still names it."""
    tree = tmp_path / "tree.py"
    tree.write_text(_REPARENTED_TREE)
    out = tmp_path / "pids"
    env = {**os.environ, "OPENRAL_SKIP_ORPHAN_REAP": "1"}
    cli = subprocess.Popen([sys.executable, "-c", _CLI, str(tree), str(out)], env=env)
    pids: list[int] = []
    try:
        deadline = time.monotonic() + 30.0
        while not (out.exists() and len(out.read_text().split()) == 2) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.1)
        pids = [int(p) for p in out.read_text().split()]
        time.sleep(2.0)
        assert all(_alive(p) for p in pids)
        cli.send_signal(signal.SIGKILL)
        cli.wait(timeout=5)
        deadline = time.monotonic() + 20.0
        while any(_alive(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not [p for p in pids if _alive(p)], "the reparented process outlived the CLI"
    finally:
        for p in pids:
            with contextlib.suppress(OSError):
                os.kill(p, signal.SIGKILL)
        cli.kill()


def test_a_cli_gone_before_the_guard_started_still_reaps_the_launch() -> None:
    """#421 review: the guard returned at once when the CLI was already dead, leaving the
    launch it was started to clean up."""
    guard = Path(__file__).resolve().parents[2] / "python/cli/src/openral_cli/_launch_guard.py"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    launch = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
    )
    try:
        subprocess.run(
            [sys.executable, str(guard), str(dead.pid), str(launch.pid)], timeout=30, check=True
        )
        launch.wait(timeout=10)
    finally:
        launch.kill()
    assert launch.returncode == -signal.SIGTERM
