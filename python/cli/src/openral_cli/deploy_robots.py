"""``openral deploy sim`` of a scene with several robots (``DeployScene.robots``, openral-pro#99).

One Isaac sidecar holds every robot of the scene; each robot gets its own unchanged
``openral deploy sim`` graph on its own ROS domain and ports, reading its own one-robot view
of the scene (``$OPENRAL_SCENE_ROBOT``, ``openral_core.scene_paths``). Each graph's HAL
leaves its sidecar entry in a shared run directory instead of starting a sidecar
(``openral_sim.backends.isaac_sim._join_shared_sidecar``); this process starts the one
``--robots`` sidecar once every entry is there, and ends everything together.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

from openral_core import DeployScene
from openral_core.exceptions import ROSConfigError
from openral_core.scene_paths import SCENE_ROBOT_ENV
from rich.console import Console

__all__ = ["robot_argv", "robot_slots", "run_scene_robots", "shared_sidecar_argv"]

_console = Console()

#: Each robot's graph: ROS domain ``_DOMAIN_BASE + _DOMAIN_STEP * i`` (below the per-PID
#: test domains 50-101 and clear of 42, ``_dds_scope.RESERVED_DOMAINS``; a step of 4 so a
#: wider discovery range never reaches a neighbour), dashboard and Foxglove ports.
_DOMAIN_BASE, _DOMAIN_STEP = 30, 4
_DASHBOARD_BASE, _FOXGLOVE_BASE = 4330, 8830
#: ponytail: five robots fit below domain 50; more need another domain plan.
_MAX_ROBOTS = 5
#: How long every robot's graph gets to leave its sidecar entry (its HAL's spec build).
_ENTRY_TIMEOUT_S = 600.0
#: How long a robot's ``deploy sim``, then the sidecar, get to end before they are killed.
_CHILD_GRACE_S = 30.0
#: Options this command sets per robot: dropped from the operator's command line.
_PER_ROBOT_OPTIONS = ("--config", "-c", "--dashboard-port", "--foxglove-port")


def robot_slots(scene: DeployScene) -> list[dict[str, int | str]]:
    """Each robot's name, ROS domain, dashboard and Foxglove port, in scene order.

    Raises:
        ROSConfigError: More robots than the domain plan holds.

    Example:
        >>> from openral_core import Pose6D, SceneRobot, SceneSpec
        >>> at = Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id="world")
        >>> scene = DeployScene(
        ...     scene=SceneSpec(id="aisle", backend="isaacsim"),
        ...     robots=[SceneRobot(name=n, robot_id="openamrobot_v2", base_pose=at) for n in "ab"],
        ... )
        >>> [(s["name"], s["domain"], s["dashboard_port"]) for s in robot_slots(scene)]
        [('a', 30, 4330), ('b', 34, 4331)]
    """
    if len(scene.robots) > _MAX_ROBOTS:
        raise ROSConfigError(
            f"scene {scene.scene.id!r} lists {len(scene.robots)} robots; `deploy sim` runs "
            f"at most {_MAX_ROBOTS} in one scene."
        )
    return [
        {
            "name": r.name,
            "domain": _DOMAIN_BASE + _DOMAIN_STEP * i,
            "dashboard_port": _DASHBOARD_BASE + i,
            "foxglove_port": _FOXGLOVE_BASE + i,
        }
        for i, r in enumerate(scene.robots)
    ]


def robot_argv(argv: list[str], config: Path, slot: dict[str, int | str]) -> list[str]:
    """One robot's ``openral deploy sim`` command line: the operator's, with its own ports.

    ``argv`` is the operator's arguments after ``openral`` (``deploy sim --config ...``).

    Example:
        >>> robot_argv(
        ...     ["deploy", "sim", "--config", "s.yaml", "--dashboard-port=1", "--no-foxglove"],
        ...     Path("/w/s.yaml"),
        ...     {"name": "a", "domain": 30, "dashboard_port": 4330, "foxglove_port": 8830},
        ... )[3:]  # doctest: +NORMALIZE_WHITESPACE
        ['deploy', 'sim', '--no-foxglove', '--config', '/w/s.yaml', '--dashboard-port', '4330',
         '--foxglove-port', '8830']
    """
    kept: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in _PER_ROBOT_OPTIONS:
            skip = True  # its value follows
            continue
        if arg.split("=", 1)[0] in _PER_ROBOT_OPTIONS:
            continue
        kept.append(arg)
    return [
        sys.executable,
        "-c",
        "from openral_cli.main import app; app()",  # `openral`, on this interpreter
        *kept,
        "--config",
        str(config),
        "--dashboard-port",
        str(slot["dashboard_port"]),
        "--foxglove-port",
        str(slot["foxglove_port"]),
    ]


def shared_sidecar_argv(entries: list[dict[str, object]], scene_id: str) -> list[str]:
    """The one sidecar's command line from every robot's entry (the first robot's launch).

    Each entry's ``launch_argv`` is its own one-robot launch; the first one's, without its
    ``--robot``, plus ``--objects-json`` and every robot in ``--robots``, is the shared one.

    Example:
        >>> e = {
        ...     "launch_argv": ["py", "sidecar.py", "--robot", "amr", "--port", "5"],
        ...     "objects_json": "",
        ...     "name": "a",
        ...     "robot_spec": "a.json",
        ...     "spawn_pose": [0, 0, 0, 0],
        ... }
        >>> shared_sidecar_argv([e, {**e, "name": "b", "robot_spec": "b.json"}], "aisle")[:4]
        ['py', 'sidecar.py', '--port', '5']
    """
    first = [str(a) for a in entries[0]["launch_argv"]]  # type: ignore[attr-defined]
    i = first.index("--robot")
    argv = first[:i] + first[i + 2 :]
    objects = str(entries[0].get("objects_json") or "")
    if objects:
        argv += ["--objects-json", objects]
    robots = [
        {"name": e["name"], "robot_spec": e["robot_spec"], "spawn_pose": e["spawn_pose"]}
        for e in entries
    ]
    return [*argv, "--robots", json.dumps(robots), "--launch-id", f"shared-{scene_id}"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _stop_with_parent() -> None:
    """In a child, before exec: SIGINT it if this launcher dies (Linux ``PR_SET_PDEATHSIG``)."""
    if _LIBC is not None:
        _LIBC.prctl(1, signal.SIGINT)


#: libc on Linux (loaded here so nothing is imported between fork and exec).
_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None


def _stop(children: Iterable[subprocess.Popen[bytes]]) -> None:
    """SIGINT every child still running; kill what outlives the grace."""
    children = list(children)
    for child in children:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)  # `deploy sim` reaps its own graph on SIGINT
    deadline = time.monotonic() + _CHILD_GRACE_S
    for child in children:
        try:
            child.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def _spawn(argv: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen[bytes]:
    with open(log_path, "wb") as log_file:
        return subprocess.Popen(  # reason: our own interpreter and sidecar
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Ctrl-C reaches it once, from _stop
            preexec_fn=_stop_with_parent if _LIBC is not None else None,  # noqa: PLW1509  # reason: one prctl call, libc preloaded
        )


def run_scene_robots(scene: DeployScene, config: Path, argv: list[str]) -> int:
    """Run every robot of ``scene`` in one Isaac world: a graph each, one sidecar.

    Returns the exit code: 0 after Ctrl-C, 1 when a robot's graph or the sidecar ended on
    its own (one robot gone ends the world: the others would command a world it left).

    Raises:
        ROSConfigError: A robot never left its sidecar entry, or too many robots.
    """
    slots = robot_slots(scene)
    run_dir = Path(tempfile.mkdtemp(prefix=f"openral-robots-{scene.scene.id[:24]}-"))
    port = _free_port()
    _console.print(f"[bold]{scene.scene.id}[/bold]: {len(slots)} robots, logs in {run_dir}")
    children: dict[str, subprocess.Popen[bytes]] = {}
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    if previous[signal.SIGHUP] is signal.SIG_IGN:
        del previous[signal.SIGHUP]
    for signum in previous:
        signal.signal(signum, signal.default_int_handler)
    code = 0
    try:
        for slot in slots:
            name = str(slot["name"])
            env = {
                **os.environ,
                "ROS_DOMAIN_ID": str(slot["domain"]),
                SCENE_ROBOT_ENV: name,
                "OPENRAL_ISAAC_SHARED_DIR": str(run_dir),
                "OPENRAL_ISAAC_SHARED_PORT": str(port),
                # Each robot is its own openral.com robot (`openral_core.robot_ui.enrolment_path`).
                "OPENRAL_ROBOT_ENROLMENT": str(
                    Path.home() / ".config/openral/robots" / scene.scene.id / name / "robot.json"
                ),
                "PYTHONUNBUFFERED": "1",
            }
            children[name] = _spawn(robot_argv(argv, config, slot), env, run_dir / f"{name}.log")
            _console.print(
                f"  robot [bold]{name}[/bold]: ROS_DOMAIN_ID={slot['domain']} dashboard "
                f"http://127.0.0.1:{slot['dashboard_port']}/ log {run_dir / f'{name}.log'}"
            )
        entries = _wait_for_entries(run_dir, [str(s["name"]) for s in slots], children)
        sidecar_argv = shared_sidecar_argv(entries, scene.scene.id)
        children["(sidecar)"] = _spawn(sidecar_argv, dict(os.environ), run_dir / "sidecar.log")
        _console.print(
            f"  Isaac sidecar for every robot: port {port}, log {run_dir / 'sidecar.log'}"
        )
        while all(child.poll() is None for child in children.values()):
            time.sleep(0.5)
        ended = [n for n, c in children.items() if c.poll() is not None]
        _console.print(f"{ended} ended: stopping every robot", style="yellow")
        code = 1
    except KeyboardInterrupt:
        _console.print("stopping every robot...")
    finally:
        sidecar = children.pop("(sidecar)", None)
        _stop(children.values())  # the graphs first: their HALs leave the world
        if sidecar is not None:
            _stop([sidecar])
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return code


def _wait_for_entries(
    run_dir: Path, names: list[str], children: dict[str, subprocess.Popen[bytes]]
) -> list[dict[str, object]]:
    """Every robot's sidecar entry, in scene order, once its HAL has written it."""
    deadline = time.monotonic() + _ENTRY_TIMEOUT_S
    while True:
        missing = [n for n in names if not (run_dir / f"{n}.json").is_file()]
        if not missing:
            return [json.loads((run_dir / f"{n}.json").read_text(encoding="utf-8")) for n in names]
        dead = [n for n in missing if children[n].poll() is not None]
        if dead:
            raise ROSConfigError(f"robots {dead} ended before reaching the sidecar: see their logs")
        if time.monotonic() > deadline:
            raise ROSConfigError(f"robots {missing} never reached the sidecar: see their logs")
        time.sleep(1.0)
