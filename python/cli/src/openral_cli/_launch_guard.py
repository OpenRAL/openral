"""Reap a deploy's launch tree once the deploy CLI is gone, however it died.

Usage (spawned by ``openral_cli.deploy_sim._run_launch``, never by hand)::

    python _launch_guard.py <cli_pid> <launch_pid>

The CLI's own teardown (``_run_launch``'s ``finally``) cannot run when the CLI is
SIGKILLed or crashes hard, and ``ros2 launch`` puts every node in its own process group
(a sim sidecar even in its own session), so nothing signals them: they keep running on
the ROS domain (#418 r4dr_shadow2). This process, in its own session so a signal to the
CLI's group does not reach it, snapshots every descendant of the launch every
``POLL_S`` and, once the CLI's pid is gone (or reused), SIGTERMs every process it ever
saw that is still the same process (pid + start time), then SIGKILLs what is left after
``GRACE_S``. A sidecar marked kept alive (``KEPT_ALIVE_ENV``) is left running.
Stdlib only: it must start fast and never import the CLI.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
from pathlib import Path

POLL_S = 0.5
GRACE_S = 5.0
#: ``openral_sim.sidecar.KEPT_ALIVE_ENV`` (this module is stdlib-only, so it repeats the name).
KEPT_ALIVE_ENV = b"OPENRAL_SIDECAR_KEPT_ALIVE=1"


def _stat(pid: int) -> tuple[int, int] | None:
    """``(ppid, start time in clock ticks)`` of a live, non-zombie ``pid``, else ``None``."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    if fields[0] == "Z":
        return None
    return int(fields[1]), int(fields[19])


def _same(pid: int, start: int) -> bool:
    st = _stat(pid)
    return st is not None and st[1] == start


def _children() -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            st = _stat(int(entry))
            if st is not None:
                out.setdefault(st[0], []).append(int(entry))
    return out


def _snapshot(seen: dict[int, int]) -> None:
    """Add every live descendant of every process already seen."""
    children = _children()
    todo = [p for p, s in seen.items() if _same(p, s)]
    while todo:
        for child in children.get(todo.pop(), []):
            st = _stat(child)
            if st is not None and child not in seen:
                seen[child] = st[1]
                todo.append(child)


def _kept_alive(pid: int) -> bool:
    """Whether ``pid`` is a sidecar spawned to outlive its deploy (``--keep-sim-open``)."""
    try:
        return KEPT_ALIVE_ENV in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False


def _reap(seen: dict[int, int]) -> None:
    for pid in [p for p in seen if _kept_alive(p)]:
        del seen[pid]  # the next run adopts it; ending it is the run that is not kept alive
    for sig, wait_s in ((signal.SIGTERM, GRACE_S), (signal.SIGKILL, 0.0)):
        for pid, start in seen.items():
            if _same(pid, start):
                with contextlib.suppress(OSError):
                    os.kill(pid, sig)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and any(_same(p, s) for p, s in seen.items()):
            time.sleep(0.1)


def main(argv: list[str]) -> int:
    cli_pid, launch_pid = int(argv[0]), int(argv[1])
    cli, launch = _stat(cli_pid), _stat(launch_pid)
    if cli is None or launch is None:
        return 0
    seen = {launch_pid: launch[1]}
    while _same(cli_pid, cli[1]):
        _snapshot(seen)
        time.sleep(POLL_S)
    _snapshot(seen)  # what the dying launch left behind
    _reap(seen)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
