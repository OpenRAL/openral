"""``openral deploy world`` — several robots in one MuJoCo world (issue #666).

Starts the world's one physics process (``openral_hal.shared_world``) and one
unchanged ``openral deploy sim`` per robot, each on its own ``ROS_DOMAIN_ID``
and ports, attached to that world through two ``--hal`` keys.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

import typer
import yaml
from openral_core import DeployScene, SharedWorldRobot, SharedWorldScene, load_scene_strict
from openral_core.exceptions import ROSConfigError
from pydantic import ValidationError
from rich.console import Console

__all__ = ["deploy_world_command", "robot_deploy_argv", "world_robot_yamls"]

_console = Console()

#: How long a robot's ``deploy sim`` gets to shut its graph down before it is killed.
_CHILD_GRACE_S = 30.0


def world_robot_yamls(world: SharedWorldScene, world_path: Path) -> dict[str, str]:
    """Each robot's ``robot.yaml``, by robot name, from its ``DeployScene``.

    Raises:
        ROSConfigError: A robot's scene does not load as a ``DeployScene``, names
            no ``robot_id``, or would not build a bare MuJoCo twin (it composes
            its own arena or scene-attaches; the world owns the arena here).

    Example:
        >>> from pathlib import Path
        >>> path = Path("scenes/worlds/so101_handover_table.yaml")
        >>> yamls = world_robot_yamls(SharedWorldScene.from_yaml(str(path)), path)
        >>> sorted(yamls), Path(yamls["a"]).parent.name
        (['a', 'b'], 'so101_follower')
    """
    from openral_sim.policies.robots import resolve_robot_manifest

    from openral_cli.deploy_sim import _repo_root_from, _scene_builds_bare_twin

    yamls: dict[str, str] = {}
    for robot in world.robots:
        scene_path = (world_path.parent / robot.scene).resolve()
        try:
            scene = load_scene_strict(str(scene_path), DeployScene)
        except FileNotFoundError as exc:
            raise ROSConfigError(f"{world_path}: robot {robot.name!r}: {exc}") from exc
        if scene.robot_id is None or scene.composition is not None:
            raise ROSConfigError(
                f"{scene_path}: a shared-world robot's scene must name `robot_id` and "
                "declare no `composition` (the world's `arena_mjcf` is the arena)."
            )
        if not _scene_builds_bare_twin(scene):
            raise ROSConfigError(
                f"{scene_path}: scene id {scene.scene.id!r} is a registered sim scene "
                "(it would scene-attach); a shared-world robot is a bare MuJoCo twin."
            )
        yamls[robot.name] = str(
            resolve_robot_manifest(scene.robot_id, repo_root=_repo_root_from(Path(__file__)))
        )
    return yamls


def robot_deploy_argv(
    robot: SharedWorldRobot, world_path: Path, address: str, extra: list[str]
) -> list[str]:
    """The ``openral deploy sim`` command line of one robot of the world.

    Example:
        >>> from pathlib import Path
        >>> robot = SharedWorldRobot(
        ...     name="a",
        ...     scene="../deploy/so101_shared_table.yaml",
        ...     ros_domain_id=36,
        ...     dashboard_port=4336,
        ...     foxglove_port=8836,
        ... )
        >>> robot_deploy_argv(robot, Path("/w/worlds/x.yaml"), "/tmp/w.sock", ["--no-foxglove"])[3:]
        ... # doctest: +NORMALIZE_WHITESPACE
        ['deploy', 'sim', '--config', '/w/deploy/so101_shared_table.yaml',
         '--dashboard-port', '4336', '--foxglove-port', '8836',
         '--hal', 'shared_world="/tmp/w.sock"', '--hal', 'shared_world_robot="a"',
         '--no-foxglove']
    """
    return [
        sys.executable,
        "-c",
        "from openral_cli.main import app; app()",  # `openral`, on this interpreter
        "deploy",
        "sim",
        "--config",
        str((world_path.parent / robot.scene).resolve()),
        "--dashboard-port",
        str(robot.dashboard_port),
        "--foxglove-port",
        str(robot.foxglove_port),
        # `--hal` values are JSON-parsed: quoted, a robot named `true` stays a string.
        "--hal",
        f"shared_world={json.dumps(address)}",
        "--hal",
        f"shared_world_robot={json.dumps(robot.name)}",
        *extra,
    ]


def _stop_with_parent() -> None:
    """In a robot's ``deploy sim``, before exec: SIGINT it if this launcher dies.

    The launcher's own teardown covers Ctrl-C, SIGTERM and SIGHUP; this covers a
    launcher that is killed outright, whose robots would otherwise keep running
    against a world that is gone. Linux ``prctl(PR_SET_PDEATHSIG)``.
    """
    _LIBC.prctl(1, signal.SIGINT)  # PR_SET_PDEATHSIG = 1


#: libc on Linux (loaded here so nothing is imported between fork and exec); ``None``
#: elsewhere, where there is no ``prctl`` and the robots are not tied to the launcher.
_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None


def _stop_robots(children: Iterable[subprocess.Popen[bytes]]) -> None:
    """SIGINT every robot's ``deploy sim`` still running; kill what outlives the grace."""
    children = list(children)
    for child in children:
        if child.poll() is None:
            # `deploy sim` forwards SIGINT to its launch and reaps its own graph.
            child.send_signal(signal.SIGINT)
    deadline = time.monotonic() + _CHILD_GRACE_S
    for child in children:
        try:
            child.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def deploy_world_command(
    ctx: typer.Context,
    config: Path = typer.Option(  # reason: typer Option idiom
        ...,
        "--config",
        "-c",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Path to a SharedWorldScene YAML (scenes/worlds/).",
    ),
    log_dir: Path | None = typer.Option(
        None,
        "--log-dir",
        help="Where each robot's `deploy sim` output goes (<name>.log). Default: a temp dir.",
    ),
) -> None:
    """Run several robots in ONE MuJoCo world, each with its own deploy graph (sim only).

    Arguments after ``--`` are passed to every robot's ``openral deploy sim``
    (e.g. ``-- --no-foxglove --no-enable-reasoner``).
    """
    from openral_hal.shared_world import SharedWorldServer

    config = config.resolve()
    try:
        world = SharedWorldScene.from_yaml(str(config))
        robot_yamls = world_robot_yamls(world, config)
        # A Unix socket path is at most ~107 bytes: a short private dir under /tmp,
        # whatever $TMPDIR says.
        run_dir = Path(tempfile.mkdtemp(prefix=f"openral-world-{world.id[:24]}-", dir="/tmp"))
        address = str(run_dir / "world.sock")
        server = SharedWorldServer(world, config.parent / world.arena_mjcf, robot_yamls, address)
        server.start()
    except (ROSConfigError, ValidationError, OSError, yaml.YAMLError) as exc:
        _console.print(f"{config}: {exc}", style="red", markup=False)
        raise typer.Exit(code=2) from exc
    logs = (log_dir or run_dir).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    _console.print(f"[bold]shared world[/bold] {world.id}: physics at {address}")

    children: dict[str, subprocess.Popen[bytes]] = {}
    interrupted = False
    # SIGTERM, SIGHUP (the terminal closed), and a SIGINT inherited as ignored (a shell's
    # background job) must still stop every robot's graph: all end the loop below as
    # Ctrl-C does.
    # A SIGHUP inherited as ignored (`nohup`) stays ignored. The previous handlers are
    # put back at the end: the `openral>` REPL runs this command in its own process.
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    if previous[signal.SIGHUP] is signal.SIG_IGN:
        del previous[signal.SIGHUP]
    for signum in previous:
        signal.signal(signum, signal.default_int_handler)
    try:
        for robot in world.robots:
            log_path = logs / f"{robot.name}.log"
            with open(log_path, "wb") as log_file:
                children[robot.name] = subprocess.Popen(
                    robot_deploy_argv(robot, config, address, list(ctx.args)),
                    env={
                        **os.environ,
                        "ROS_DOMAIN_ID": str(robot.ros_domain_id),
                        "PYTHONUNBUFFERED": "1",  # the log file follows the run
                    },
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # Ctrl-C reaches it once, from the loop below
                    preexec_fn=_stop_with_parent if _LIBC is not None else None,  # noqa: PLW1509  # reason: one prctl call, libc preloaded
                )
            _console.print(
                f"  robot [bold]{robot.name}[/bold]: ROS_DOMAIN_ID={robot.ros_domain_id} "
                f"dashboard http://127.0.0.1:{robot.dashboard_port}/ log {log_path}"
            )
        # One robot's graph ending (or Ctrl-C) ends the world: the others would be
        # left commanding a world their peer has silently left.
        while all(child.poll() is None for child in children.values()):
            time.sleep(0.5)
    except KeyboardInterrupt:
        interrupted = True
        _console.print("stopping the world...")
    finally:
        for signum in previous:  # a second Ctrl-C must not cut the teardown short
            signal.signal(signum, signal.SIG_IGN)
        _stop_robots(children.values())
        server.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    codes = {name: child.returncode for name, child in children.items()}
    _console.print(f"shared world {world.id} ended at sim t={server.sim_time_s():.2f} s: {codes}")
    # Ctrl-C is the normal way out; otherwise a robot's graph ended by itself.
    raise typer.Exit(code=0 if interrupted else 1)
