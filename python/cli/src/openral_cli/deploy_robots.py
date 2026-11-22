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
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

from openral_core import DeployScene, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_core.scene_paths import (
    SCENE_PEERS_ENV,
    SCENE_ROBOT_ENV,
    SCENE_ROBOT_MAP_FRAME_ENV,
    format_scene_peers,
)
from openral_observability.dashboard.remote_auth import load_enrolment
from openral_observability.dashboard.remote_router import route_host
from openral_observability.dashboard.tunnel import REMOTE_PORT_ENV, TUNNEL_PORT
from openral_sim.backends.isaac_sim import shared_launch_id
from openral_sim.policies.robots import resolve_robot_manifest
from rich.console import Console

from openral_cli._dds_scope import RESERVED_DOMAINS
from openral_cli.deploy_sim import _repo_root_from

__all__ = [
    "robot_argv",
    "robot_enrolment_path",
    "robot_memory_dirs",
    "robot_slots",
    "router_argv",
    "run_scene_robots",
    "shared_sidecar_argv",
    "tunnel_routes",
    "wait_for_entries",
]

_console = Console()

#: Each robot's graph's ROS domain, in scene order: every 4th from 30 below the per-PID test
#: domains 50-101, skipping ``_dds_scope.RESERVED_DOMAINS`` (42); a step of 4 so a wider
#: discovery range never reaches a neighbour. Its dashboard and Foxglove ports count up.
_DOMAINS = tuple(d for d in range(30, 50, 4) if d not in RESERVED_DOMAINS)
_DASHBOARD_BASE, _FOXGLOVE_BASE = 4330, 8830
#: Each linked robot's remote listener; the scene's router holds the tunnel port (4319) for all.
_REMOTE_BASE = 4350
#: ponytail: four robots fit below domain 50; more need another domain plan.
_MAX_ROBOTS = len(_DOMAINS)
#: How long every robot's graph gets to leave its sidecar entry (its HAL's spec build).
_ENTRY_TIMEOUT_S = 600.0
#: How long a robot's ``deploy sim``, then the sidecar, get to end before they are killed.
_CHILD_GRACE_S = 30.0
#: Options this command sets per robot: dropped from the operator's command line.
_PER_ROBOT_OPTIONS = ("--config", "-c", "--dashboard-port", "--foxglove-port", "--memory-dir")


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
            "domain": _DOMAINS[i],
            "dashboard_port": _DASHBOARD_BASE + i,
            "foxglove_port": _FOXGLOVE_BASE + i,
            "remote_port": _REMOTE_BASE + i,
        }
        for i, r in enumerate(scene.robots)
    ]


def robot_argv(
    argv: list[str], config: Path, slot: dict[str, int | str], memory_dir: Path | None = None
) -> list[str]:
    """One robot's ``openral deploy sim`` command line: the operator's, with its own ports.

    ``argv`` is the operator's arguments after ``openral`` (``deploy sim --config ...``).
    ``memory_dir``: the robot's own memory bundle (``<--memory-dir>/<name>``), when one is set.

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
        *(["--memory-dir", str(memory_dir)] if memory_dir is not None else []),
    ]


def router_argv(routes: dict[str, int]) -> list[str]:
    """The scene's tunnel router's command line (``remote_router``), on this interpreter.

    Example:
        >>> router_argv({"r-a1.openral.com": 4350})[1:]
        ['-m', 'openral_observability.dashboard.remote_router', '--route', 'r-a1.openral.com=4350']
    """
    route_args = [a for host, port in routes.items() for a in ("--route", f"{host}={port}")]
    return [sys.executable, "-m", "openral_observability.dashboard.remote_router", *route_args]


def shared_sidecar_argv(entries: list[dict[str, object]], launch_id: str) -> list[str]:
    """The one sidecar's command line from every robot's entry (the first robot's launch).

    Each entry's ``launch_argv`` is its own one-robot launch; the first one's, without its
    ``--robot``, plus ``--objects-json`` and every robot in ``--robots``, is the shared one.
    ``launch_id`` is this run's (``isaac_sim.shared_launch_id``): each graph refuses a sidecar
    that does not echo it in ``ping``.

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
    return [*argv, "--robots", json.dumps(robots), "--launch-id", launch_id]


def robot_enrolment_path(scene_id: str, name: str) -> Path:
    """Where a scene robot's openral.com enrolment is: each robot is its own openral.com robot.

    Example:
        >>> robot_enrolment_path("isaac_sim", "a").parts[-4:]
        ('robots', 'isaac_sim', 'a', 'robot.json')
    """
    return Path.home() / ".config/openral/robots" / scene_id / name / "robot.json"


def tunnel_routes(scene_id: str, slots: list[dict[str, int | str]]) -> dict[str, int]:
    """Each linked robot's tunnel host to its remote listener port: the scene router's table.

    Robots not linked (no enrolment, or one that doesn't load or can't be read) are left out,
    and so is a robot whose enrolment names a host another robot of the scene already has (one
    link code used twice): nothing reaches them from openral.com, and each is said why.
    """
    routes: dict[str, int] = {}
    for slot in slots:
        name = str(slot["name"])
        path = robot_enrolment_path(scene_id, name)
        try:
            enrolment = load_enrolment(path)
        except (ValueError, OSError) as exc:
            _console.print(f"  robot {name}: enrolment {path} refused: {exc}", style="yellow")
            continue
        if enrolment is None:
            continue
        host = route_host(enrolment.aud)
        if host in routes:
            _console.print(
                f"  robot {name}: {host} is another robot's of this scene (one link code for "
                "two robots?): not reachable from openral.com",
                style="yellow",
            )
            continue
        routes[host] = int(slot["remote_port"])
    return routes


#: How long the router gets to bind 4319 (or exit, it taken) before the scene goes on.
_ROUTER_START_S = 5.0


def _router_up(router: subprocess.Popen[bytes], timeout_s: float = _ROUTER_START_S) -> bool:
    """Whether the router holds 4319: it accepts a connection before it exits or time runs out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and router.poll() is None:
        with (
            contextlib.suppress(OSError),
            socket.create_connection(("127.0.0.1", TUNNEL_PORT), timeout=0.5),
        ):
            return router.poll() is None  # this router's, not a dashboard's that held it first
        time.sleep(0.1)
    return False


def _start_router(
    scene_id: str, slots: list[dict[str, int | str]], run_dir: Path
) -> subprocess.Popen[bytes] | None:
    """The scene's tunnel router, when any robot is linked; ``None`` when none is.

    Every linked robot's tunnel reaches 4319 (openral.com sets the ingress): the router there
    sends each to its robot by host. It is not a robot: its exit (4319 taken) stops nothing,
    but it is said, since then no linked robot of the scene is reachable from openral.com.
    """
    routes = tunnel_routes(scene_id, slots)
    if not routes:
        return None
    log = run_dir / "router.log"
    router = _spawn(router_argv(routes), dict(os.environ), log)
    if _router_up(router):
        _console.print(
            f"  tunnel router on 127.0.0.1:{TUNNEL_PORT} for {sorted(routes)}: log {log}"
        )
    else:
        _console.print(
            f"  tunnel router did not start (127.0.0.1:{TUNNEL_PORT} taken? see {log}): "
            f"{sorted(routes)} are not reachable from openral.com",
            style="yellow",
        )
    return router


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


def _option_value(argv: list[str], option: str) -> str | None:
    """The value of ``option`` on a command line (``--opt v`` or ``--opt=v``), else ``None``.

    Example:
        >>> _option_value(["deploy", "sim", "--memory-dir=/m"], "--memory-dir")
        '/m'
        >>> _option_value(["--memory-dir", "/m", "-c", "s.yaml"], "-c")
        's.yaml'
    """
    for i, arg in enumerate(argv):
        if arg == option and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(f"{option}="):
            return arg.split("=", 1)[1]
    return None


def robot_memory_dirs(root: str | None, names: list[str]) -> dict[str, Path]:
    """Each robot's own memory bundle, ``<root>/<name>``; none without a ``root``.

    One bundle per robot: its ``object_memory.json`` holds poses in its own map, and each
    graph rewrites its whole file, so robots sharing one would restore and overwrite each
    other's objects. The directories are created when missing.

    Raises:
        ROSConfigError: ``root`` is not an existing directory.

    Example:
        >>> import tempfile
        >>> root = tempfile.mkdtemp()
        >>> {n: p.name for n, p in robot_memory_dirs(root, ["a", "b"]).items()}
        {'a': 'a', 'b': 'b'}
        >>> robot_memory_dirs(None, ["a"])
        {}
    """
    if not root:
        return {}
    base = Path(root).expanduser()
    if not base.is_dir():
        raise ROSConfigError(
            f"--memory-dir {root!r} is not an existing directory: a scene with several robots "
            "keeps each robot's bundle in <memory-dir>/<robot name>."
        )
    dirs = {name: base / name for name in names}
    for d in dirs.values():
        d.mkdir(exist_ok=True)
    return dirs


def scene_peers(scene: DeployScene, slots: list[dict[str, int | str]]) -> dict[str, str]:
    """Per robot, ``SCENE_PEERS_ENV``: every other robot's name, domain and manifest path."""
    root = _repo_root_from(Path(__file__))
    manifests = {
        r.name: str(resolve_robot_manifest(r.robot_id, repo_root=root)) for r in scene.robots
    }
    entries = [(str(s["name"]), int(s["domain"]), manifests[str(s["name"])]) for s in slots]
    return {
        name: format_scene_peers([e for e in entries if e[0] != name]) for name, _, _ in entries
    }


def _map_frames(scene: DeployScene) -> dict[str, str]:
    """Each robot's ``RobotDescription.map_frame``: the frame its one-robot scene's goals are in."""
    root = _repo_root_from(Path(__file__))
    return {
        r.name: RobotDescription.from_yaml(
            str(resolve_robot_manifest(r.robot_id, repo_root=root))
        ).map_frame
        for r in scene.robots
    }


def _print_dry_run(
    scene: DeployScene,
    config: Path,
    argv: list[str],
    slots: list[dict[str, int | str]],
    map_frames: dict[str, str],
    memory_root: str | None,
    peers: dict[str, str],
) -> None:
    """What ``--dry-run`` shows: each robot's environment and command line, and the router's."""
    for slot in slots:
        name = str(slot["name"])
        memory = Path(memory_root).expanduser() / name if memory_root else None
        shown = f"ROS_DOMAIN_ID={slot['domain']} {SCENE_ROBOT_ENV}={name} "
        shown += f"{SCENE_ROBOT_MAP_FRAME_ENV}={map_frames[name]} "
        shown += f"{SCENE_PEERS_ENV}={shlex.quote(peers[name])} "
        shown += f"OPENRAL_ROBOT_ENROLMENT={robot_enrolment_path(scene.scene.id, name)} "
        shown += f"{REMOTE_PORT_ENV}={slot['remote_port']} "
        shown += shlex.join(robot_argv(argv, config, slot, memory))
        _console.print(f"  robot {name}: {shown}", soft_wrap=True, markup=False)
    routes = tunnel_routes(scene.scene.id, slots)
    if routes:
        shown = shlex.join(router_argv(routes))
        _console.print(f"  tunnel router: {shown}", soft_wrap=True, markup=False)


def run_scene_robots(
    scene: DeployScene, config: Path, argv: list[str], *, dry_run: bool = False
) -> int:
    """Run every robot of ``scene`` in one Isaac world: a graph each, one sidecar.

    Returns the exit code: 0 after Ctrl-C, 1 when a robot's graph or the sidecar ended on
    its own (one robot gone ends the world: the others would command a world it left).
    ``dry_run`` prints each robot's command line and environment and starts nothing.

    Raises:
        ROSConfigError: A robot never left its sidecar entry, or too many robots.
    """
    slots = robot_slots(scene)
    names = [str(s["name"]) for s in slots]
    map_frames = _map_frames(scene)
    peers = scene_peers(scene, slots)
    memory_root = _option_value(argv, "--memory-dir") or scene.memory_dir
    if dry_run:
        _print_dry_run(scene, config, argv, slots, map_frames, memory_root, peers)
        return 0
    memory_dirs = robot_memory_dirs(memory_root, names)
    run_dir = Path(tempfile.mkdtemp(prefix=f"openral-robots-{scene.scene.id[:24]}-"))
    port = _free_port()
    _console.print(f"[bold]{scene.scene.id}[/bold]: {len(slots)} robots, logs in {run_dir}")
    children: dict[str, subprocess.Popen[bytes]] = {}
    router: subprocess.Popen[bytes] | None = None
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
                SCENE_ROBOT_MAP_FRAME_ENV: map_frames[name],
                SCENE_PEERS_ENV: peers[name],
                "OPENRAL_ISAAC_SHARED_DIR": str(run_dir),
                "OPENRAL_ISAAC_SHARED_PORT": str(port),
                # Each robot is its own openral.com robot (`openral_core.robot_ui.enrolment_path`).
                "OPENRAL_ROBOT_ENROLMENT": str(robot_enrolment_path(scene.scene.id, name)),
                # Its remote listener's own port: the scene's router holds the tunnel port.
                REMOTE_PORT_ENV: str(slot["remote_port"]),
                "PYTHONUNBUFFERED": "1",
            }
            children[name] = _spawn(
                robot_argv(argv, config, slot, memory_dirs.get(name)),
                env,
                run_dir / f"{name}.log",
            )
            _console.print(
                f"  robot [bold]{name}[/bold]: ROS_DOMAIN_ID={slot['domain']} dashboard "
                f"http://127.0.0.1:{slot['dashboard_port']}/ log {run_dir / f'{name}.log'}"
            )
        router = _start_router(scene.scene.id, slots, run_dir)
        entries = wait_for_entries(run_dir, names, children)
        sidecar_argv = shared_sidecar_argv(entries, shared_launch_id(run_dir))
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
        if router is not None:
            _stop([router])
        if sidecar is not None:
            _stop([sidecar])
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return code


def wait_for_entries(
    run_dir: Path,
    names: list[str],
    children: dict[str, subprocess.Popen[bytes]],
    timeout_s: float = _ENTRY_TIMEOUT_S,
) -> list[dict[str, object]]:
    """Every robot's sidecar entry, in scene order, once its HAL has written it.

    Raises:
        ROSConfigError: Any robot's graph ended (with its entry written or not: the world
            would start without it), or an entry is still missing after ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        dead = [n for n in names if children[n].poll() is not None]
        if dead:
            raise ROSConfigError(f"robots {dead} ended before the sidecar started: see their logs")
        missing = [n for n in names if not (run_dir / f"{n}.json").is_file()]
        if not missing:
            return [json.loads((run_dir / f"{n}.json").read_text(encoding="utf-8")) for n in names]
        if time.monotonic() > deadline:
            raise ROSConfigError(f"robots {missing} never reached the sidecar: see their logs")
        time.sleep(1.0)
