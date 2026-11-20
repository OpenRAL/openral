"""Several robots in ONE MuJoCo world, each driven by its own deploy graph (issue #666).

``SharedWorldServer`` owns the physics of a ``SharedWorldScene``: every robot's
MJCF and one shared arena, composed into one model. Each robot's deploy graph
(its own ROS domain, unchanged) runs a ``SharedWorldArmHAL``: a ``MujocoArmHAL``
that keeps a *mirror* of the world for its cameras and joint reads, and asks the
server to advance instead of stepping the mirror itself.

A robot's mirror is the world **in that robot's base frame, with its own MJCF
elements unprefixed and first** (``compose_shared_world(view=<robot>)``), so
everything its graph reads (joint indices, camera and body names, sensor
extrinsics, the safety workspace) is exactly a single-robot twin's; the other
robots appear under ``<name>_`` prefixes. The server maps its state into each
view (a block permutation, plus the planar base transform on free bodies).

Time (the single-robot rule, kept per robot): one ``send_action`` is one control
period. A robot is *commanding* from a ``send_action`` until its next idle step;
the world advances one period once every commanding robot has sent its action
(a lockstep barrier, bounded by ``SharedWorldScene.barrier_timeout_s``). With no
robot commanding, idle steps advance the world on wall time.

The wire is JSON over a Unix socket in a private directory: local, sim only.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import threading
import time
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Any

import structlog
from openral_core import RobotDescription, SharedWorldScene
from openral_core.exceptions import ROSConfigError, ROSError, ROSRuntimeError
from openral_core.geometry import yaw_to_quat_wxyz

from openral_hal._camera_rig import rig_cameras_into_mjcf
from openral_hal._mujoco_arm import _IDLE_STEP_CAP, MujocoArmHAL, _resolve_mjcf_path

__all__ = [
    "SharedWorldArmHAL",
    "SharedWorldServer",
    "compose_shared_world",
    "open_view",
    "robot_root_bodies",
]

log = structlog.get_logger(__name__)

# (name, joints, actuators) of each robot, in the order its block sits in a model.
_Blocks = list[tuple[str, int, int]]


def _robot_spec(mj: Any, description: RobotDescription) -> Any:
    """The robot's own MJCF as ``MujocoArmHAL`` loads it: manifest cameras rigged in.

    A rigged MJCF is parsed from a string (no rigged file is written beside the
    source, which several processes would race on), so a robot whose cameras
    must be rigged in and whose MJCF ``<include>``s other files is not supported.
    """
    path = _resolve_mjcf_path(description)
    try:
        xml, rigged = rig_cameras_into_mjcf(Path(path).read_text(), list(description.sensors))
        spec = mj.MjSpec.from_string(xml) if rigged else mj.MjSpec.from_file(path)
    except (OSError, ValueError) as exc:
        raise ROSConfigError(f"shared world: could not load MJCF {path!r}: {exc}") from exc
    if rigged:  # parsed from a string: anchor the relative asset dirs by hand
        spec.meshdir = os.path.join(os.path.dirname(path), spec.meshdir)
        spec.texturedir = os.path.join(os.path.dirname(path), spec.texturedir)
    # The rig's visual-only staging plane: the arena is the floor here.
    floor = next((g for g in spec.geoms if g.name == "camrig_floor"), None)
    if floor is not None:
        spec.delete(floor)
    if any(not b.name for b in spec.worldbody.bodies):
        # A peer's links are found under its named top-level bodies (robot_root_bodies).
        raise ROSConfigError(
            f"shared world: robot {description.name!r} has an unnamed top-level MJCF body."
        )
    if len(spec.worldbody.geoms):
        # Attached geoms of a robot's <worldbody> land on the world body, where nothing
        # tells them from the arena: its peers could not see them as part of it.
        raise ROSConfigError(
            f"shared world: robot {description.name!r} has geoms directly in its MJCF "
            "<worldbody> (a pedestal, a plate); put them in a body of the robot."
        )
    if any(j.type not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE) for j in spec.joints):
        raise ROSConfigError(
            f"shared world: robot {description.name!r} has a free or ball joint; only "
            "fixed-base robots with hinge / slide joints are supported."
        )
    return spec


def compose_shared_world(
    world: SharedWorldScene,
    arena_mjcf: Path,
    descriptions: dict[str, RobotDescription],
    view: str | None,
) -> tuple[Any, _Blocks]:
    """Compose the world's ``MjSpec`` as one robot sees it, and its robot blocks.

    Args:
        world: The shared world.
        arena_mjcf: The arena MJCF file (``world.arena_mjcf``, resolved).
        descriptions: Each robot's manifest, by ``SharedWorldRobot.name``.
        view: The robot whose base frame is the model's world frame and whose
            elements are unprefixed and first; ``None`` is the server's model
            (world frame, every robot prefixed, in ``world.robots`` order).

    Returns:
        ``(spec, blocks)``: the uncompiled spec, and ``(name, joints, actuators)``
        per robot in the model's joint order (each joint is one qpos and one dof).

    Raises:
        ROSConfigError: A robot or the arena cannot be loaded, or ``view`` /
            ``descriptions`` do not match ``world.robots``.

    Example:
        >>> spec, blocks = compose_shared_world(world, arena, descriptions, "b")  # doctest: +SKIP
        >>> blocks[0][0]  # doctest: +SKIP
        'b'
    """
    import mujoco as mj  # reason: optional sim-only dep

    robots = {r.name: r for r in world.robots}
    if set(descriptions) != set(robots) or (view is not None and view not in robots):
        raise ROSConfigError(
            f"shared world {world.id!r}: robots are {sorted(robots)}, got manifests for "
            f"{sorted(descriptions)} and view {view!r}."
        )
    vx, vy, vz = robots[view].base_xyz if view else (0.0, 0.0, 0.0)
    vyaw = math.radians(robots[view].base_yaw_deg) if view else 0.0

    def frame(xyz: tuple[float, float, float], yaw_rad: float) -> Any:
        """A worldbody frame at a WORLD pose, expressed in the view's frame."""
        dx, dy = xyz[0] - vx, xyz[1] - vy
        c, s = math.cos(-vyaw), math.sin(-vyaw)
        return spec.worldbody.add_frame(
            pos=[c * dx - s * dy, s * dx + c * dy, xyz[2] - vz],
            quat=list(yaw_to_quat_wxyz(yaw_rad - vyaw)),
        )

    spec = mj.MjSpec()
    children = {name: _robot_spec(mj, descriptions[name]) for name in robots}
    # Attaching drops a child's <option> and <visual> (the rig's fill light): every
    # view takes the first robot's, so all models step and render alike.
    spec.option = children[world.robots[0].name].option
    spec.visual = children[world.robots[0].name].visual
    blocks: _Blocks = []
    for name in ([view] if view else []) + [n for n in robots if n != view]:
        child = children[name]
        blocks.append((name, len(child.joints), len(child.actuators)))
        spec.attach(
            child,
            frame=frame(robots[name].base_xyz, math.radians(robots[name].base_yaw_deg)),
            prefix="" if name == view else f"{name}_",
        )
    try:
        arena = mj.MjSpec.from_file(str(arena_mjcf))
    except (OSError, ValueError) as exc:
        raise ROSConfigError(f"shared world: could not load arena {arena_mjcf}: {exc}") from exc
    spec.attach(arena, frame=frame((0.0, 0.0, 0.0), 0.0), prefix="")
    return spec, blocks


def robot_root_bodies(robot: str, description: RobotDescription, *, view: str | None) -> set[str]:
    """The names, in a composed model, of ``robot``'s top-level bodies.

    Unprefixed in that robot's own view, ``<robot>_…`` everywhere else; a peer's
    links are exactly the bodies under these (``body_rootid``).

    Example:
        >>> robot_root_bodies("b", so101, view="a")  # doctest: +SKIP
        {'b_base'}
    """
    import mujoco as mj  # reason: optional sim-only dep

    prefix = "" if robot == view else f"{robot}_"
    return {f"{prefix}{b.name}" for b in _robot_spec(mj, description).worldbody.bodies}


def _ask(conn: Connection, msg: dict[str, object]) -> dict[str, Any]:
    """One request / reply on a world connection; a refusal is a ``ROSConfigError``."""
    conn.send_bytes(json.dumps(msg).encode())
    reply: dict[str, Any] = json.loads(conn.recv_bytes())
    if not reply["ok"]:
        raise ROSConfigError(str(reply["error"]))
    return reply


def open_view(
    address: str, robot: str
) -> tuple[Connection, SharedWorldScene, dict[str, RobotDescription], Any]:
    """Connect to a shared world and compile ``robot``'s view of it.

    Returns:
        ``(connection, world, manifests by robot name, compiled view model)``.

    Raises:
        ROSConfigError: No world at ``address``, or the view does not compose.
    """
    try:
        conn = Client(address, family="AF_UNIX")
    except OSError as exc:
        raise ROSConfigError(
            f"no shared world at {address!r} ({exc}); `openral deploy world` starts one."
        ) from exc
    try:
        hello = _ask(conn, {"op": "hello"})
        world = SharedWorldScene.model_validate(hello["world"])
        descriptions = {n: RobotDescription.from_yaml(p) for n, p in hello["robot_yamls"].items()}
        spec, _ = compose_shared_world(world, Path(hello["arena_mjcf"]), descriptions, robot)
        return conn, world, descriptions, _compile(spec, world.id)
    except (EOFError, OSError) as exc:
        conn.close()
        raise ROSRuntimeError(f"shared world at {address!r} went away: {exc}") from exc
    except BaseException:
        conn.close()
        raise


def _compile(spec: Any, world_id: str) -> Any:
    try:
        return spec.compile()
    except ValueError as exc:  # e.g. an arena name clashing with a robot's
        raise ROSConfigError(f"shared world {world_id!r} does not compile: {exc}") from exc


class SharedWorldServer:
    """The physics of a ``SharedWorldScene``, served to its robots' HALs.

    Args:
        world: The shared world.
        arena_mjcf: The arena MJCF file (``world.arena_mjcf``, resolved).
        robot_yamls: Each robot's ``robot.yaml`` path, by ``SharedWorldRobot.name``.
        address: The Unix socket path to listen on (keep it in a private directory).

    Raises:
        ROSConfigError: The world does not compose (see ``compose_shared_world``),
            or it has actuator activation states (not mirrored).

    Example:
        >>> server = SharedWorldServer(world, arena, robot_yamls, address)  # doctest: +SKIP
        >>> server.start()  # doctest: +SKIP
    """

    def __init__(
        self,
        world: SharedWorldScene,
        arena_mjcf: Path,
        robot_yamls: dict[str, str],
        address: str,
    ) -> None:
        """Compose and compile the world; nothing is served until ``start()``."""
        import mujoco as mj  # reason: optional sim-only dep
        import numpy as np  # reason: with mujoco

        self._mj = mj
        self._world = world
        self._address = address
        self._hello = {
            "ok": True,
            "world": world.model_dump(),
            "arena_mjcf": str(arena_mjcf),
            "robot_yamls": dict(robot_yamls),
        }
        descriptions = {n: RobotDescription.from_yaml(p) for n, p in robot_yamls.items()}
        rates = {n: d.control_rate_hz for n, d in descriptions.items()}
        twins = all(
            d.hal.sim is None and d.sim is not None and d.sim.keyframe_index is None
            for d in descriptions.values()
        )
        if not twins or None in rates.values() or len(set(rates.values())) != 1:
            raise ROSConfigError(
                f"shared world {world.id!r}: every robot must be a manifest-derived MuJoCo "
                "twin (`hal.sim: null`, a `sim:` block with no `keyframe_index`) and all "
                f"must declare the same action_spec.control_freq_hz (got {rates})."
            )
        spec, blocks = compose_shared_world(world, arena_mjcf, descriptions, None)
        self._model = m = _compile(spec, world.id)
        if m.na:
            raise ROSConfigError(
                f"shared world {world.id!r}: actuator activation states are not supported."
            )
        # Every robot's view must compile too (an arena name equal to a robot's own
        # unprefixed element clashes only there): fail here, not in each robot's HAL.
        for name in robot_yamls:
            view = _compile(
                compose_shared_world(world, arena_mjcf, descriptions, name)[0], world.id
            )
            if (view.nq, view.nv, view.nu) != (m.nq, m.nv, m.nu):
                raise ROSConfigError(f"shared world {world.id!r}: robot {name!r}'s view differs.")
        self._data = mj.MjData(self._model)
        mj.mj_forward(self._model, self._data)
        # Robot blocks come first in every model: joints [j0, j1), actuators [u0, u1).
        self._joints: dict[str, tuple[int, int]] = {}
        self._actuators: dict[str, tuple[int, int]] = {}
        j = u = 0
        for name, n_joint, n_act in blocks:
            self._joints[name] = (j, j + n_joint)
            self._actuators[name] = (u, u + n_act)
            j, u = j + n_joint, u + n_act
        self._seed_ctrl(descriptions)
        # World index of each slot of a robot's view: its own block, then the rest in order.
        self._view_qpos: dict[str, Any] = {}
        self._view_qvel: dict[str, Any] = {}
        self._view_ctrl: dict[str, Any] = {}
        for name, (j0, j1) in self._joints.items():
            head = np.concatenate([np.arange(j0, j1), np.arange(0, j0), np.arange(j1, j)])
            self._view_qpos[name] = np.concatenate([head, np.arange(j, self._model.nq)])
            self._view_qvel[name] = np.concatenate([head, np.arange(j, self._model.nv)])
            u0, u1 = self._actuators[name]
            self._view_ctrl[name] = np.concatenate(
                [np.arange(u0, u1), np.arange(0, u0), np.arange(u1, self._model.nu)]
            )
        # Free bodies (arena objects): their pose and linear velocity are frame-dependent.
        self._free = [
            (int(self._model.jnt_qposadr[i]), int(self._model.jnt_dofadr[i]))
            for i in range(self._model.njnt)
            if self._model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE
        ]
        self._cond = threading.Condition()
        self._attached: set[str] = set()
        self._joined: set[str] = set()  # robots that have joined at least once
        # Robots inside the lockstep barrier, and when each last sent an action.
        self._commanding: dict[str, float] = {}
        self._pending: set[str] = set()  # commanding robots whose action awaits the advance
        self._tick_steps = 0  # physics steps per control period; 0 until the first join
        self._last_advance = time.monotonic()
        self._listener: Listener | None = None
        self._acceptor: threading.Thread | None = None

    def _seed_ctrl(self, descriptions: dict[str, RobotDescription]) -> None:
        """Hold the initial pose of every robot that seeds its targets from it.

        A robot with ``sim.seed_ctrl_from_qpos`` holds its initial pose from the start:
        the world may advance on another robot's idle steps before this one joins.
        """
        mj, m = self._mj, self._model
        for name, (u0, u1) in self._actuators.items():
            sim = descriptions[name].sim
            if sim is None or not sim.seed_ctrl_from_qpos:
                continue
            for u in range(u0, u1):
                if m.actuator_trntype[u] == mj.mjtTrn.mjTRN_JOINT:
                    joint = int(m.actuator_trnid[u, 0])
                    self._data.ctrl[u] = self._data.qpos[m.jnt_qposadr[joint]]

    # ── Serving ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Listen on ``address``; each robot's HAL is served on its own thread."""
        self._listener = Listener(self._address, family="AF_UNIX")
        self._acceptor = threading.Thread(
            target=self._accept, name="shared-world-accept", daemon=True
        )
        self._acceptor.start()
        log.info("shared_world.serving", world=self._world.id, address=self._address)

    def close(self) -> None:
        """Stop listening.  Idempotent; a robot still attached is served until it leaves."""
        listener, self._listener = self._listener, None
        if listener is None:
            return
        # Closing a listening socket from another thread does not reliably wake a
        # blocked accept(): connect once, so the acceptor sees the listener is gone.
        with contextlib.suppress(OSError):
            Client(self._address, family="AF_UNIX").close()
        if self._acceptor is not None:  # set by start(), with the listener
            self._acceptor.join(timeout=5.0)
        listener.close()

    def sim_time_s(self) -> float:
        """The world's elapsed simulation time, seconds."""
        with self._cond:
            return float(self._data.time)

    def _accept(self) -> None:
        while (listener := self._listener) is not None:
            try:
                conn = listener.accept()
            except OSError:  # closed
                return
            if self._listener is None:  # close()'s wake-up connection
                conn.close()
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: Connection) -> None:
        robot: str | None = None
        try:
            while True:
                raw = conn.recv_bytes()  # EOF / socket errors: the connection ends below
                try:
                    msg = json.loads(raw)
                    if msg["op"] == "hello":
                        reply: dict[str, object] = self._hello
                    elif msg["op"] == "observe":  # a reader (peer_voxels), never attached
                        reply = self._observe(msg)
                    elif msg["op"] == "join" and robot is None:
                        reply = self._join(msg)
                        robot = str(msg["robot"])
                    elif robot is not None and msg["op"] in ("step", "idle", "poke", "hold"):
                        reply = getattr(self, f"_{msg['op']}")(robot, msg)
                    else:
                        raise ROSConfigError(f"unexpected {msg['op']!r} from {robot or 'a client'}")
                except (ROSConfigError, KeyError, ValueError, TypeError) as exc:
                    # A malformed request is refused, never fatal to the robot's thread.
                    reply = {"ok": False, "error": f"shared world {self._world.id!r}: {exc}"}
                conn.send_bytes(json.dumps(reply).encode())
        except (EOFError, OSError):
            pass  # the HAL went away (shutdown, e-stop, crash): leave below
        finally:
            conn.close()
            if robot is not None:
                self._leave(robot)

    # ── Operations (one robot's request each) ──────────────────────────────────

    def _join(self, msg: dict[str, Any]) -> dict[str, object]:
        robot = str(msg["robot"])
        m = self._model
        with self._cond:
            if robot not in self._joints:
                raise ROSConfigError(f"no robot {robot!r} (robots: {sorted(self._joints)})")
            if robot in self._attached:
                raise ROSConfigError(f"robot {robot!r} is already attached")
            if list(msg["shape"]) != [m.nq, m.nv, m.nu, m.nbody, m.ngeom]:
                raise ROSConfigError(
                    f"robot {robot!r} composed a different model {msg['shape']} than the "
                    f"world's {[m.nq, m.nv, m.nu, m.nbody, m.ngeom]} (different checkout?)"
                )
            steps, timestep = int(msg["tick_steps"]), float(msg["timestep"])
            if not self._tick_steps:
                self._tick_steps = steps
                m.opt.timestep = timestep
            elif steps != self._tick_steps or not math.isclose(timestep, m.opt.timestep):
                raise ROSConfigError(
                    f"robot {robot!r} runs {steps} x {timestep} s per action, the world "
                    f"{self._tick_steps} x {m.opt.timestep} s: every robot of a shared "
                    "world must have the same control period."
                )
            if robot not in self._joined:
                # Its initial targets. A robot joining AGAIN (a restarted HAL) takes the
                # targets held for it since it left instead: its arm must not move.
                self._set_ctrl(robot, msg["ctrl"])
                self._joined.add(robot)
            if not self._attached:
                self._last_advance = time.monotonic()  # wall time starts with the first robot
            self._attached.add(robot)
            log.info("shared_world.joined", world=self._world.id, robot=robot)
            return self._state(robot)

    def _observe(self, msg: dict[str, Any]) -> dict[str, object]:
        """The world's current state in ``msg["robot"]``'s view, for a reader that never steps."""
        robot = str(msg["robot"])
        with self._cond:
            if robot not in self._joints:
                raise ROSConfigError(f"no robot {robot!r} (robots: {sorted(self._joints)})")
            return self._state(robot)

    def _step(self, robot: str, msg: dict[str, Any]) -> dict[str, object]:
        """One ``send_action``: advance one control period with every commanding robot.

        The barrier does not check the released targets against each other: each was
        checked by its own robot's kernel against the others' last peer grid
        (``peer_voxels``). That is the stated limit for simultaneous moves
        (``docs/reference/shared-world-sim.md``).
        """
        with self._cond:
            self._set_ctrl(robot, msg["ctrl"])
            self._commanding[robot] = time.monotonic()
            self._pending.add(robot)
            deadline = time.monotonic() + self._world.barrier_timeout_s
            while robot in self._pending:
                late = self._commanding.keys() - self._pending
                remaining = deadline - time.monotonic()
                if not late:
                    self._advance(self._tick_steps)
                    self._pending.clear()
                    self._cond.notify_all()
                elif remaining > 0:
                    self._cond.wait(remaining)  # then re-read the barrier from the top
                else:
                    self._drop_late(late, waiting=robot)
            return self._state(robot)

    def _idle(self, robot: str, _msg: dict[str, Any]) -> dict[str, object]:
        """One idle step: the robot stops commanding; a world nobody commands runs on wall time."""
        with self._cond:
            self._stop_commanding(robot)
            # A commanding robot that went silent must not freeze an idle one either.
            stale = time.monotonic() - self._world.barrier_timeout_s
            late = {r for r, t in self._commanding.items() if t < stale and r not in self._pending}
            if late:
                self._drop_late(late, waiting=robot)
            if not self._commanding:
                dt = float(self._model.opt.timestep)
                steps = int((time.monotonic() - self._last_advance) / dt)
                if steps > _IDLE_STEP_CAP:  # a long stall must not fast-forward the world
                    self._advance(_IDLE_STEP_CAP)
                elif steps:
                    carried = self._last_advance + steps * dt
                    self._advance(steps)
                    self._last_advance = carried  # keep the sub-step remainder
            return self._state(robot)

    def _poke(self, robot: str, msg: dict[str, Any]) -> dict[str, object]:
        """``reset_to_pose``: set the joints the pose names (view indices), and the targets.

        Only those entries: a passive joint of the robot keeps its world state.
        """
        with self._cond:
            j0, j1 = self._joints[robot]
            for key, view_index in (
                ("qpos", self._view_qpos[robot]),
                ("qvel", self._view_qvel[robot]),
            ):
                for index, value in msg[key]:
                    if not 0 <= int(index) < j1 - j0:  # the robot's own block, first in its view
                        raise ValueError(f"{key} index {index} is not one of {robot!r}'s joints")
                    getattr(self._data, key)[view_index[int(index)]] = float(value)
            self._set_ctrl(robot, msg["ctrl"])
            self._mj.mj_forward(self._model, self._data)
            return self._state(robot)

    def _hold(self, robot: str, msg: dict[str, Any]) -> dict[str, object]:
        """A stopping robot's last word: the targets that hold its arm where it is."""
        with self._cond:
            self._set_ctrl(robot, msg["ctrl"])
            self._stop_commanding(robot)
            return self._state(robot)

    def _leave(self, robot: str) -> None:
        with self._cond:
            self._attached.discard(robot)
            self._stop_commanding(robot)
        log.info("shared_world.left", world=self._world.id, robot=robot, note="targets held")

    # ── Internals (call with ``_cond`` held) ───────────────────────────────────

    def _stop_commanding(self, robot: str) -> None:
        if self._commanding.pop(robot, None) is not None and not self._commanding:
            # Wall-paced idling starts now, not at the last commanded advance: the
            # first idle step must not fast-forward the pause before it.
            self._last_advance = time.monotonic()
        self._pending.discard(robot)
        self._cond.notify_all()  # a robot waiting on this one re-checks the barrier

    def _drop_late(self, late: set[str], *, waiting: str) -> None:
        """Take silent robots out of the barrier. Explicit, never silent: logged, targets held."""
        log.warning(
            "shared_world.barrier_timeout",
            world=self._world.id,
            waiting=waiting,
            late=sorted(late),
            timeout_s=self._world.barrier_timeout_s,
        )
        for robot in late:
            self._stop_commanding(robot)

    def _set_ctrl(self, robot: str, view_ctrl: list[float]) -> None:
        u0, u1 = self._actuators[robot]
        self._data.ctrl[u0:u1] = view_ctrl[: u1 - u0]  # the robot's block is first in its view

    def _advance(self, steps: int) -> None:
        for _ in range(steps):
            self._mj.mj_step(self._model, self._data)
        self._last_advance = time.monotonic()

    def _state(self, robot: str) -> dict[str, object]:
        """The world's state in ``robot``'s view (its base frame, its block first)."""
        qpos = self._data.qpos[self._view_qpos[robot]]
        qvel = self._data.qvel[self._view_qvel[robot]]
        base = next(r for r in self._world.robots if r.name == robot)
        yaw = -math.radians(base.base_yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        hc, hs = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        for qa, da in self._free:  # tail addresses: the same in every view
            x, y = qpos[qa] - base.base_xyz[0], qpos[qa + 1] - base.base_xyz[1]
            qpos[qa : qa + 3] = (c * x - s * y, s * x + c * y, qpos[qa + 2] - base.base_xyz[2])
            qw, qx, qy, qz = qpos[qa + 3 : qa + 7]  # yaw quaternion (hc, 0, 0, hs) x q
            qpos[qa + 3 : qa + 7] = (
                hc * qw - hs * qz,
                hc * qx - hs * qy,
                hc * qy + hs * qx,
                hc * qz + hs * qw,
            )
            vx, vy = qvel[da], qvel[da + 1]  # linear velocity is in the world frame;
            qvel[da : da + 2] = (c * vx - s * vy, s * vx + c * vy)  # angular is body-local
        return {
            "ok": True,
            "time": float(self._data.time),
            "qpos": qpos.tolist(),
            "qvel": qvel.tolist(),
            # Every robot's targets: the mirror's actuator and contact forces then
            # match the world's, the peers' included.
            "ctrl": self._data.ctrl[self._view_ctrl[robot]].tolist(),
        }


class SharedWorldArmHAL(MujocoArmHAL):
    """A ``MujocoArmHAL`` whose physics is a shared world's (see the module docstring).

    Built by ``build_hal`` when the HAL transport names a ``shared_world``
    (``openral deploy world`` passes it as ``--hal shared_world=...``).

    Args:
        description: The robot manifest (``sim:`` block required).
        shared_world: The ``SharedWorldServer``'s Unix socket path.
        shared_world_robot: This robot's ``SharedWorldRobot.name``.
        staleness_limit_s: As ``MujocoArmHAL``.

    Note:
        ``estop`` and ``disconnect`` leave the world, whose other robots keep
        running: the arm is first told to hold the pose it has at that moment
        (not its last commanded target, and not zeroed targets).

    Example:
        >>> hal = SharedWorldArmHAL(
        ...     desc, shared_world=addr, shared_world_robot="a"
        ... )  # doctest: +SKIP
        >>> hal.connect()  # doctest: +SKIP
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        shared_world: str,
        shared_world_robot: str,
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Initialise the adapter; the world is not joined until ``connect()``."""
        self._init_from_description(description, staleness_limit_s=staleness_limit_s)
        self._world_address = shared_world
        self._world_robot = shared_world_robot
        self._conn: Connection | None = None
        self._conn_lock = threading.Lock()

    def connect(self) -> None:
        """Compose this robot's view of the world and join it.

        Raises:
            ROSConfigError: No world at the address, the world refuses the robot
                (unknown name, already attached, another control period), or the
                manifest declares no control rate.
            ROSRuntimeError: If already connected.
        """
        if self._conn is not None:
            raise ROSRuntimeError(f"SharedWorldArmHAL({self._world_robot!r}) is already connected.")
        try:
            super().connect()
            assert self._model is not None and self._data is not None
            if self._sim_dt_per_tick_s is None:
                raise ROSConfigError(
                    f"shared world: robot {self.description.name!r} needs "
                    "action_spec.control_freq_hz (one action = one control period)."
                )
            m = self._model
            self._sync(
                self._request(
                    {
                        "op": "join",
                        "robot": self._world_robot,
                        "tick_steps": self._settle_steps,
                        "timestep": float(m.opt.timestep),
                        "shape": [m.nq, m.nv, m.nu, m.nbody, m.ngeom],
                        "ctrl": self._data.ctrl.tolist(),
                    }
                )
            )
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        """Hold the arm where it is, leave the world, and release the mirror."""
        self._hold_and_leave()
        super().disconnect()

    def estop(self) -> None:
        """Hold the arm where it is, leave the world, then ``MujocoArmHAL.estop`` (raises)."""
        self._hold_and_leave()
        super().estop()

    def _hold_and_leave(self) -> None:
        """Retarget this arm to its current pose in the world, then close the socket.

        Without it the world would keep servoing a stopped arm to its last commanded
        target. A world that is already gone is not an error here: stopping goes on.
        """
        if self._connected and self._conn is not None and self._data is not None:
            try:
                self._sync(self._request({"op": "idle"}))  # the arm's pose now
                pose = self.read_state().position
                self._apply_arm_targets(pose)
                if self._grippers_by_joint:
                    self._apply_gripper_targets(pose)
                self._request({"op": "hold", "ctrl": self._data.ctrl.tolist()})
            except ROSError as exc:
                log.warning("shared_world.hold_failed", robot=self._world_robot, error=str(exc))
        self._close_conn()

    def reset_to_pose(self, pose: list[float]) -> None:
        """Snap this arm to ``pose`` in the shared world (as ``MujocoArmHAL.reset_to_pose``)."""
        super().reset_to_pose(pose)
        assert self._data is not None
        # Only what the base reset set: the named joints (a gripper's every qpos slot).
        qpos_idx = sorted(
            {a for n, a in self._joint_qpos_addr.items() if n not in self._grippers_by_joint}
            | {a for g in self._grippers_by_joint.values() for a in g.qpos_addrs}
        )
        qvel_idx = sorted(self._joint_qvel_addr[n] for n in self._joint_names)
        self._sync(
            self._request(
                {
                    "op": "poke",
                    "qpos": [[i, float(self._data.qpos[i])] for i in qpos_idx],
                    "qvel": [[i, float(self._data.qvel[i])] for i in qvel_idx],
                    "ctrl": self._data.ctrl.tolist(),
                }
            )
        )

    def _load_model(self, mj: Any) -> Any:
        del mj  # the view is composed by open_view
        self._conn, _, _, model = open_view(self._world_address, self._world_robot)
        return model

    def _step_physics(self, steps: int, targets: list[float] | None = None) -> None:
        """Ask the world to advance: a commanded period (``targets``) or an idle step."""
        del steps  # the world advances the joined control period, or wall time when idle
        assert self._data is not None
        if targets is None:
            self._sync(self._request({"op": "idle"}))
        else:
            self._sync(self._request({"op": "step", "ctrl": self._data.ctrl.tolist()}))

    def _request(self, msg: dict[str, object]) -> dict[str, Any]:
        with self._conn_lock:  # one request / reply at a time
            conn = self._conn
            if conn is None:
                raise ROSRuntimeError("SharedWorldArmHAL: not attached to a shared world.")
            try:
                return _ask(conn, msg)
            except (EOFError, OSError) as exc:
                raise ROSRuntimeError(
                    f"shared world at {self._world_address!r} went away: {exc}"
                ) from exc

    def _sync(self, state: dict[str, Any]) -> None:
        """Copy the world's state into the mirror."""
        import mujoco as mj  # reason: optional sim-only dep

        assert self._model is not None and self._data is not None
        self._data.time = state["time"]
        self._data.qpos[:] = state["qpos"]
        self._data.qvel[:] = state["qvel"]
        self._data.ctrl[:] = state["ctrl"]
        mj.mj_forward(self._model, self._data)

    def _close_conn(self) -> None:
        # Under the lock: never close the socket beneath a request in flight (it waits
        # at most ``barrier_timeout_s``, with this arm's targets held).
        with self._conn_lock:
            conn, self._conn = self._conn, None
            if conn is not None:
                conn.close()
