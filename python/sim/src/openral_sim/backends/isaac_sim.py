r"""Isaac Sim scene adapter — drives an Isaac Lab env through an out-of-process sidecar.

ADR-0045. NVIDIA Isaac Sim (Omniverse Kit + PhysX + RTX) ships per-interpreter
wheels: 4.x→py3.10, 5.x→py3.11, 6.x→py3.12. The openral workspace pins
``>=3.12,<3.13``, and Isaac Sim's stack (its own torch / CUDA build, the rigid
``SimulationApp``-before-``omni.*`` import order, a libgomp/OpenMP ``LD_PRELOAD``
clash with the VLA torch stack) makes an in-process load impractical inside the
3.12 venv. So — exactly like the RLDX-1 policy sidecar
(:mod:`openral_sim.policies.rldx`) — we run Isaac Lab in its own py3.11 venv and
talk to it over ZMQ REQ/REP framed by msgpack.

This module is the **openral side**: a thin :class:`SimRollout` that marshals
``reset`` / ``step`` / ``render`` / ``close`` to the sidecar
(``tools/isaac_sidecar.py``) and unwraps the responses. The sidecar owns the
Omniverse app, the Franka manipulation env, PhysX stepping, and RTX camera
rendering.

Lifecycle (mirrors the RLDX adapter's auto-spawn block)
-------------------------------------------------------
* On build we ping the sidecar at ``host:port``.
* If the ping fails and ``auto_spawn`` is on (default; ``vla``-free scene path,
  toggle via ``OPENRAL_ISAAC_AUTO_SPAWN=0``), we ``Popen`` the launcher with the
  resolved scene config (task id, layout, obs size, instruction) and poll
  ``ping`` until it answers or ``boot_timeout_s`` elapses. First boot pays the
  tens-of-seconds Omniverse Kit start; ``boot_timeout_s`` defaults large.
* :meth:`close` terminates only a child we spawned ourselves; a pre-existing
  operator-launched sidecar is left running.

Sidecar python resolution
-------------------------
The launcher runs under the **isaac** venv, not this one. We resolve its
interpreter from ``OPENRAL_ISAAC_SIDECAR_PYTHON`` (absolute path to the py3.11
venv's ``python``). That venv (Isaac Sim is a ~50 GB, RTX-only, license-gated
install) is provisioned out of band by the user — there is no auto-install plan
for it, unlike the lightweight openral-side ``isaac_client`` wire deps. Without
the env var set we fall back to the cache default
(``~/.cache/openral/isaac-sidecar/.venv/bin/python``) and raise a typed
:class:`ROSConfigError` carrying the exact provisioning commands if absent.

Scene category: **free-axis** (``fixed_robot=None``). The sidecar's env is
Franka-based today but the scene is robot-flagged for forward compatibility with
other Isaac Lab embodiments; ``robot_id`` from the YAML is forwarded to the
launcher.

Licensing (ADR-0045 §Decision-2 / CLAUDE.md §1.9, §3): Isaac Sim's Omniverse Kit
components are proprietary and **never vendored** — the sidecar venv is an
externally-provisioned dependency the user installs (and, by running the
launcher, accepts the NVIDIA Omniverse EULA via ``OMNI_KIT_ACCEPT_EULA=YES``).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult
from openral_sim.sidecar import SidecarClient

if TYPE_CHECKING:
    from openral_core import RobotDescription, SceneSpec, SensorSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


_ISAAC_SCENE_ID = "isaac_sim"
_AUTO_SPAWN_ENV = "OPENRAL_ISAAC_AUTO_SPAWN"
_SIDECAR_PYTHON_ENV = "OPENRAL_ISAAC_SIDECAR_PYTHON"
_SIDECAR_SCRIPT_ENV = "OPENRAL_ISAAC_SIDECAR_SCRIPT"
# Default ZMQ endpoint. Distinct port from the RLDX sidecar (5555-ish) so the
# two can coexist on one host.
_DEFAULT_HOST = "127.0.0.1"
# Per-scene default ports live in 20000–39999 (clear of well-known ports and the
# usual ephemeral range) — the SAME band the RLDX sidecar uses for the same
# reason (``policies.rldx._derive_sidecar_port``). One sidecar serves one scene,
# so two DIFFERENT scenes must NOT share a port: a lingering sidecar from scene A
# would otherwise be silently adopted by scene B (same host:port) and serve it
# the wrong layout. ``_scene_default_port`` derives a stable per-scene port so
# that never happens; an explicit ``backend_options.port`` still wins.
_SIDECAR_PORT_MIN = 20_000
_SIDECAR_PORT_MAX = 40_000


def _scene_default_port(task_id: str, robot_id: str, layout: str) -> int:
    """Deterministic per-scene ZMQ port, stable across processes.

    Mirrors ``policies.rldx._derive_sidecar_port`` (policy identity) for the
    scene-identity case. Uses a ``hashlib`` digest (NOT the builtin ``hash``,
    which is salted per process via ``PYTHONHASHSEED``) so the port the sidecar
    binds in its spawn process matches the one a later client process probes for
    the same scene. Distinct scenes map to distinct ports with overwhelming
    probability; any residual collision is caught loudly by the identity-checked
    ping handshake (``SidecarClient.expected_identity``), never served as wrong
    data. SHA-256 is used only to spread identities, never as a security boundary.
    """
    import hashlib

    key = f"{task_id}|{robot_id}|{layout}".encode()
    digest = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return _SIDECAR_PORT_MIN + (digest % (_SIDECAR_PORT_MAX - _SIDECAR_PORT_MIN))


# REQ recv timeout for a steady-state step (Omniverse PhysX + RTX render of one
# frame is far slower than MuJoCo — generous so a slow frame is not read as a
# dead sidecar).
_DEFAULT_TIMEOUT_MS = 120_000
_DEFAULT_BOOT_TIMEOUT_S = 900.0
# Truncation cap when the scene comes from a taskless DeployScene (deploy sim).
_DEFAULT_MAX_STEPS = 1_000_000
# Above this (m), a manifest gripper position-limit reads as a NORMALISED [0, 1]
# width rather than physical Panda finger travel — don't apply it to the Isaac
# finger DOF (it would tear the joint); fall back to the Panda default.
_MAX_PHYSICAL_GRIPPER_TRAVEL_M = 0.1


# ── SimRollout adapter ────────────────────────────────────────────────────────


def _coerce_sim_time_ns(value: object) -> int | None:
    """Coerce an optional wire ``sim_time_ns`` (int / float / None) to ``int | None``.

    The sidecar's msgpack reply carries sim time as a plain number (or omits it
    on an older protocol); anything non-numeric degrades to ``None`` so the HAL
    simply publishes no ``/clock`` rather than crashing.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


@dataclass
class _IsaacSimSidecar:
    """:class:`SimRollout` that proxies an Isaac Lab env over the sidecar.

    Observations come back from the sidecar already in the eval-layer shape
    (``images`` dict of HWC uint8, ``state`` 1-D float32, ``task`` str); we only
    re-wrap into a plain dict and cache the last RGB frame for ``render``.
    """

    scene: SceneSpec
    task: TaskSpec
    _client: SidecarClient
    _last_image: NDArray[np.uint8] | None = None
    _action_dim: int | None = None
    _last_sim_time_ns: int | None = None

    @property
    def action_dim(self) -> int:
        """Flat action width ``step`` accepts — queried from the sidecar ping.

        ``openral deploy sim`` wraps this rollout in ``SimAttachedHAL``, whose
        ``_probe_env_action_dim`` reads ``env.action_dim`` to size the HAL's
        action packing (ADR-0036). The sidecar's ``ping`` reply already carries
        the scene's action width (8 for ``lift_cube``, 7 for ``bowl_plate``); we
        cache it on first access.
        """
        if self._action_dim is None:
            reply = self._client.call("ping")
            self._action_dim = int(self._client.require(reply, "action_dim"))
        return self._action_dim

    def reset(self, seed: int | None = None) -> Observation:
        reply = self._client.call("reset", {"seed": seed})
        self._last_sim_time_ns = _coerce_sim_time_ns(reply.get("sim_time_ns"))
        return self._wrap_obs(self._client.require(reply, "observation"))

    def step(self, action: NDArray[np.float32]) -> StepResult:
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        reply = self._client.call("step", {"action": action_np})
        # ADR-0048 Phase 2 — cache the sidecar's elapsed sim time so the
        # deploy-sim HAL can publish /clock with an Isaac backend. Optional in
        # the wire protocol (older sidecars omit it) → stays None, /clock off.
        self._last_sim_time_ns = _coerce_sim_time_ns(reply.get("sim_time_ns"))
        return StepResult(
            observation=self._wrap_obs(self._client.require(reply, "observation")),
            reward=float(self._client.require(reply, "reward")),
            terminated=bool(self._client.require(reply, "terminated")),
            truncated=bool(self._client.require(reply, "truncated")),
            info=dict(reply.get("info", {})),
        )

    def sim_time_ns(self) -> int | None:
        """Elapsed simulation time in ns from the last sidecar reply, or ``None``.

        ADR-0048 Phase 2 — the value the deploy-sim HAL reads (through
        ``SimAttachedHAL.sim_time_ns``, which adds the cross-reset offset) to
        publish ``/clock``. ``None`` when the sidecar does not report sim time
        (older protocol), so the graph stays on wall-clock.
        """
        return self._last_sim_time_ns

    def render(self) -> NDArray[np.uint8] | None:
        return None if self._last_image is None else self._last_image.copy()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.call("close")
        self._client.close()

    def _wrap_obs(self, raw: dict[str, Any]) -> Observation:
        images_raw = raw.get("images", {})
        images: dict[str, NDArray[np.uint8]] = {
            k: np.asarray(v, dtype=np.uint8) for k, v in images_raw.items()
        }
        if images:
            self._last_image = next(iter(images.values()))
        else:
            h = self.scene.observation_height
            w = self.scene.observation_width
            images = {"camera1": np.zeros((h, w, 3), dtype=np.uint8)}
        state = np.asarray(raw.get("state", []), dtype=np.float32).reshape(-1)
        obs: Observation = {
            "images": images,
            "state": state,
            "task": raw.get("task", self.task.instruction),
        }
        # Real robot joint angles (manifest order), when the sidecar provides
        # them — `openral deploy sim`'s SimAttachedHAL.read_state reads this for
        # a non-MuJoCo backend's /joint_states (ADR-0045 / ADR-0034 amendment).
        joints = raw.get("joint_positions")
        if joints is not None:
            obs["joint_positions"] = np.asarray(joints, dtype=np.float32).reshape(-1)
        joint_vel = raw.get("joint_velocities")
        if joint_vel is not None:
            obs["joint_velocities"] = np.asarray(joint_vel, dtype=np.float32).reshape(-1)
        # Kinematic planar-base pose (x, y, yaw), when the manifest scene drives a
        # mobile base — the deploy-sim odom path reads this (ADR-0045 amendment M3).
        base_pose = raw.get("base_pose")
        if base_pose is not None:
            obs["base_pose"] = np.asarray(base_pose, dtype=np.float32).reshape(-1)
        # Per-depth-sensor point clouds ((N,3) base_link), when the manifest scene
        # renders a depth camera — SimSensorBridge publishes them as PointCloud2
        # (ADR-0045 amendment, perception leg).
        clouds = raw.get("depth_points")
        if isinstance(clouds, dict):
            obs["depth_points"] = {
                k: np.asarray(v, dtype=np.float32).reshape(-1, 3) for k, v in clouds.items()
            }
        # 2-D LaserScan range fan (base_link), when the manifest scene has a lidar
        # — SimSensorBridge publishes it as /scan (ADR-0045 perception leg).
        scan = raw.get("scan")
        if scan is not None:
            obs["scan"] = np.asarray(scan, dtype=np.float32).reshape(-1)
        return obs


# ── factory ───────────────────────────────────────────────────────────────────


_Num = TypeVar("_Num", int, float)


def _opt_num(
    opts: dict[str, object], key: str, default: _Num, cast: Callable[[int | float | str], _Num]
) -> _Num:
    """Coerce a ``backend_options`` value (typed ``object``) via ``cast``, else default.

    Returns ``default`` for a missing key, a ``bool`` (an ``int`` subclass we do
    not want silently accepted), a non-scalar type, OR an unparseable scalar
    (e.g. ``port: "auto"`` → ``ValueError`` → ``default``) — never raises.
    """
    value = opts.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return cast(value)
    except (ValueError, TypeError):
        return default


def _sidecar_python() -> Path:
    """Resolve the isaac sidecar venv interpreter, or raise with the install hint."""
    override = os.environ.get(_SIDECAR_PYTHON_ENV)
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_PYTHON_ENV}={override!r} is not a file.")
        return p
    default = Path.home() / ".cache" / "openral" / "isaac-sidecar" / ".venv" / "bin" / "python"
    if default.is_file():
        return default
    raise ROSConfigError(
        "Isaac Sim sidecar venv not found. It is an externally-provisioned "
        "dependency (NVIDIA Isaac Sim / Isaac Lab, separate license, RTX GPU; "
        "ADR-0045). Provision it, then point "
        f"{_SIDECAR_PYTHON_ENV} at its py3.11 python, e.g.:\n"
        "  uv venv --python 3.11 ~/.cache/openral/isaac-sidecar/.venv\n"
        "  UV_HTTP_TIMEOUT=900 uv pip install --python "
        "~/.cache/openral/isaac-sidecar/.venv/bin/python \\\n"
        "    --extra-index-url https://pypi.nvidia.com --index-strategy "
        "unsafe-best-match --no-build-isolation-package flatdict \\\n"
        "    'isaacsim[all]==5.1.0.0' 'isaaclab==2.3.2' pyzmq msgpack"
    )


def _locate_sidecar_script() -> Path:
    """Find ``tools/isaac_sidecar.py`` (env override, else walk up from here)."""
    override = os.environ.get(_SIDECAR_SCRIPT_ENV)
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_SCRIPT_ENV}={override!r} is not a file.")
        return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "isaac_sidecar.py"
        if candidate.is_file():
            return candidate
    raise ROSConfigError(
        f"Could not locate tools/isaac_sidecar.py upwards from {here}. Set "
        f"{_SIDECAR_SCRIPT_ENV} to its absolute path."
    )


# ── robot-spec marshalling (ADR-0045 amendment: robot-agnostic manifest scene) ──


def _sensor_dict(sensor: SensorSpec) -> dict[str, Any]:
    """Serialize one ``SensorSpec`` to the plain-JSON shape the sidecar reads.

    The Isaac sidecar runs py3.11 and cannot import ``openral_core``, so each
    sensor crosses the venv boundary as a dict of primitives — only the fields
    the URDF-driven scene needs to attach a camera / depth / lidar.
    """
    intr = sensor.intrinsics
    return {
        "name": sensor.name,
        "modality": getattr(sensor.modality, "value", str(sensor.modality)),
        "frame_id": sensor.frame_id,
        "parent_frame": sensor.parent_frame,
        "vla_feature_key": sensor.vla_feature_key,
        "intrinsics": (
            None
            if intr is None
            else {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "cx": intr.cx,
                "cy": intr.cy,
            }
        ),
        "range_min_m": sensor.range_min_m,
        "range_max_m": sensor.range_max_m,
        "n_channels": sensor.n_channels,
    }


def _build_robot_spec(desc: RobotDescription, robot_id: str) -> dict[str, Any]:
    """Marshal a ``RobotDescription`` to the JSON isaac robot spec (ADR-0045).

    Resolves the manifest ``urdf_path`` to an on-disk file (the
    ``python:<module>:<attr>`` form resolves where ``robot_descriptions`` lives,
    on this py3.12 side), then carries the URDF path + actuated-joint order/role +
    action contract + sensors. The planar base joints (``base_joints``) are
    excluded from ``joints`` — they are not URDF DOFs; the sidecar adds the base
    programmatically (M3).
    """
    from openral_core.urdf_resolve import resolve_urdf_path

    if not desc.urdf_path:
        raise ROSConfigError(
            f"robot {robot_id!r} has no urdf_path; the Isaac manifest scene "
            "(--layout manifest) imports the robot from its URDF."
        )
    urdf = resolve_urdf_path(desc.urdf_path)
    if not urdf or not Path(urdf).is_file():
        raise ROSConfigError(
            f"could not resolve urdf_path {desc.urdf_path!r} for {robot_id!r} to a file."
        )

    base_joint_names = set(desc.base_joints or [])

    def _joint_type(j: Any) -> str:
        return getattr(j.joint_type, "value", str(j.joint_type))

    def _eff_role(j: Any) -> str:
        """Normalise the joint role for the sidecar's joint wiring.

        ``base`` = a planar-base joint (driven kinematically, not a URDF DOF);
        ``gripper`` = the manifest gripper (collapsed two-finger width DoF);
        otherwise ``arm`` — the URDF DOFs the articulation controller drives. The
        manifest's own ``role`` is advisory and often left ``unknown`` on arm
        joints (e.g. ``franka_panda`` tags only the gripper).
        """
        if j.name in base_joint_names:
            return "base"
        if j.role == "gripper":
            return "gripper"
        return "arm"

    # Keep ALL non-fixed manifest joints in order (URDF fixed joints carry no DOF):
    # base joints stay so the sidecar can fill /joint_states from the kinematic
    # base pose; arm/gripper map to URDF DOFs.
    joints = [j for j in desc.joints if _joint_type(j) != "fixed"]
    arm_n = sum(1 for j in joints if _eff_role(j) == "arm")
    has_gripper = any(_eff_role(j) == "gripper" for j in joints)
    # A planar base (base_joints) is driven kinematically, not as URDF DOFs — its
    # 3 twist channels (vx, vy, wyaw, base frame) extend the action vector.
    has_base = bool(desc.base_joints)

    gripper_open, gripper_closed = 0.04, 0.0
    for j in joints:
        if _eff_role(j) == "gripper" and j.position_limits is not None:
            lo, hi = float(j.position_limits[0]), float(j.position_limits[1])
            # Use the manifest limits only when they read as physical finger
            # travel (metres). Some manifests (e.g. panda_mobile) declare a
            # NORMALISED [0, 1] gripper width — applying 1.0 m to the Isaac
            # finger DOF would tear the joint, so fall back to the Panda default.
            if 0.0 <= lo < hi <= _MAX_PHYSICAL_GRIPPER_TRAVEL_M:
                gripper_closed, gripper_open = lo, hi
            break

    return {
        "robot_id": robot_id,
        "urdf_path": urdf,
        "base_frame": desc.base_frame,
        # The arm is always pinned to its (possibly moving) root: a fixed arm is
        # pinned to the world; a mobile base teleports that pinned root each step
        # (kinematic base, ADR-0045 amendment M3). Either way fix_base=True keeps
        # the arm from falling.
        "fix_base": True,
        "joints": [
            {
                "name": j.name,
                "role": _eff_role(j),
                "joint_type": _joint_type(j),
            }
            for j in joints
        ],
        "base_joints": desc.base_joints,
        "base_kinematics": desc.base_kinematics,
        "action": {
            "dim": arm_n + (1 if has_gripper else 0) + (3 if has_base else 0),
            "control_mode": "joint_position",
            "arm_delta_scale": 0.05,
            "gripper_open_m": gripper_open,
            "gripper_closed_m": gripper_closed,
            "has_base": has_base,
        },
        "sensors": [_sensor_dict(s) for s in desc.sensors],
    }


def _write_robot_spec(env_cfg: SimEnvironment) -> str:
    """Build the robot spec for ``env_cfg.robot_id`` and write it to a temp JSON.

    Returns the temp file path passed to the sidecar via ``--robot-spec``; the
    caller unlinks it on ``close()``.
    """
    from openral_sim.registry import ROBOTS

    robot_id = env_cfg.robot_id or "franka_panda"
    try:
        desc = ROBOTS.get(robot_id)()
    except KeyError as exc:
        raise ROSConfigError(
            f"unknown robot_id {robot_id!r} for the Isaac manifest scene; "
            "expected a robots/<id>/robot.yaml manifest."
        ) from exc
    spec = _build_robot_spec(desc, robot_id)
    fd, path = tempfile.mkstemp(prefix=f"isaac_robot_spec_{robot_id}_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(spec, fh)
    return path


@SCENES.register(_ISAAC_SCENE_ID, fixed_robot=None)
def _build_isaac_sim_scene(env_cfg: SimEnvironment) -> _IsaacSimSidecar:
    """Build an Isaac Lab scene behind the out-of-process sidecar.

    Lazy-imports pyzmq/msgpack (the openral-side wire) via the ``isaac_client``
    install plan, resolves the sidecar interpreter + script, and connects (auto-
    spawning the Isaac Lab process on first use).
    """
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("isaac_client")
    try:
        import msgpack  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: opt-in isaacsim group
        import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: opt-in isaacsim group
    except ImportError as exc:  # pragma: no cover — runtime-error path
        raise ROSConfigError(
            "isaac_sim backend needs pyzmq + msgpack on the openral venv: "
            "uv sync --all-packages --group isaacsim --inexact"
        ) from exc

    opts = env_cfg.scene.backend_options
    host = str(opts.get("host", _DEFAULT_HOST))
    timeout_ms = _opt_num(opts, "timeout_ms", _DEFAULT_TIMEOUT_MS, int)
    boot_timeout_s = _opt_num(opts, "boot_timeout_s", _DEFAULT_BOOT_TIMEOUT_S, float)
    layout = str(opts.get("layout", "lift_cube"))
    # Default to a per-scene port (no cross-scene sidecar reuse); an explicit
    # ``port`` in backend_options still wins.
    robot_id = env_cfg.robot_id or "franka_panda"
    port = _opt_num(opts, "port", _scene_default_port(env_cfg.task.id, robot_id, layout), int)
    headless = bool(opts.get("headless", True))
    auto_spawn = os.environ.get(_AUTO_SPAWN_ENV, "1") != "0"

    launch_argv = [
        str(_sidecar_python()),
        str(_locate_sidecar_script()),
        "--task",
        env_cfg.task.id,
        "--robot",
        env_cfg.robot_id or "franka_panda",
        "--instruction",
        env_cfg.task.instruction,
        "--layout",
        layout,
        "--obs-height",
        str(env_cfg.scene.observation_height),
        "--obs-width",
        str(env_cfg.scene.observation_width),
        "--max-steps",
        # A DeployScene (openral deploy sim) has no task, so build_sim_env_from_yaml
        # synthesises a noop task whose max_steps is None — fall back to a large
        # cap so the continuously-driven deploy env never truncates mid-run.
        str(env_cfg.task.max_steps if env_cfg.task.max_steps is not None else _DEFAULT_MAX_STEPS),
        "--success-key",
        env_cfg.task.success_key or "is_success",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if headless:
        launch_argv.append("--headless")

    # ADR-0045 amendment — the robot-agnostic layout imports the manifest robot's
    # URDF. Marshal the RobotDescription to a temp JSON the py3.11 sidecar reads
    # (it cannot import openral_core) and pass it via --robot-spec.
    robot_spec_path: str | None = None
    if layout == "manifest":
        robot_spec_path = _write_robot_spec(env_cfg)
        launch_argv += ["--robot-spec", robot_spec_path]

    client = SidecarClient(
        name="isaac",
        host=host,
        port=port,
        timeout_ms=timeout_ms,
        boot_timeout_s=boot_timeout_s,
        launch_argv=launch_argv,
        auto_spawn=auto_spawn,
        # Reject (loudly) an already-running sidecar on this port that serves a
        # different scene, instead of silently adopting its wrong layout.
        expected_identity={"task": env_cfg.task.id, "layout": layout},
    )
    try:
        client.connect()
    finally:
        # The sidecar reads the spec once at boot (before it answers ping), so by
        # the time connect() returns or fails the temp file is consumed — unlink
        # it here rather than leaking it past process exit.
        if robot_spec_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(robot_spec_path)
    return _IsaacSimSidecar(scene=env_cfg.scene, task=env_cfg.task, _client=client)
