"""MujocoTwinRollout — the bare MuJoCo twin's physics behind the sim HAL shell.

The one MuJoCo driver of the twin hierarchy: it loads the robot's MJCF (camera rig
spliced, keyframe, gravity, ctrl seeded from qpos), resolves the time base (one
action is one control period of ``description.control_rate_hz``), composes and
validates the commands it is given, writes arm and gripper targets into ``ctrl``,
drives the base, and owns the ONE stepping loop (``advance``: the per-step hook,
then ``mj_step``) every action and idle slice runs through. It reads the joint state
in the manifest's public units and renders the manifest's RGB cameras on demand.

``MujocoArmHAL`` is a ``SimRolloutHAL`` whose env is one of these, so the twin and
every scene-attached backend share one HAL shell, which drives it through the env
hooks it reads: ``step_action_group`` (every command: a slot group, or one ungrouped
action as a group of one — ``steps_every_command_as_group``), ``idle_step``,
``read_joint_state``, ``read_images``, ``mujoco_handles``, ``sim_time_ns`` and
``sim_dt_per_tick_s``. A robot whose twin behaves differently is a subclass
overriding the per-robot hooks (``_per_step``, ``_apply_targets``, ``_drive_base``,
``_on_load``, ``_on_idle``, ``_compose``), chosen by its HAL subclass; the building
blocks for the humanoids live here (``_GlideTwin``: the ADR-0087 pinned glide base;
``_SoftwarePDTwin``: a position loop over torque actuators; ``_FreeBaseTwin``: a base
pose read off the free joint). The rollout never holds a reference to its HAL.

Only ``mujoco`` is needed (imported lazily, on ``load``): ``openral_sim`` is never
imported here, so a bare twin runs on a host without it.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, NotRequired, Required, TypedDict, Unpack

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    Action,
    ControlMode,
    GripperReadMode,
    GripperWriteMode,
    JointState,
    RobotDescription,
    SimGripperDescription,
)
from openral_core.time_base import TIME_BASE_TOL, physics_steps_per_tick
from pydantic import BaseModel, ConfigDict, Field

from openral_hal._base import (
    require_connected,
    require_control_mode,
    require_supported_control_mode,
    validate_action_dims,
)
from openral_hal._floating_base import (
    BasePin,
    capture_base_pin,
    integrate_planar_twist,
    validate_planar_body_twist,
    write_base_pin,
)
from openral_hal._slot_group import compose_slot_group_action

if TYPE_CHECKING:
    import mujoco

__all__ = ["MujocoTwinRollout", "TwinKwargs", "TwinStepResult", "joint_actuator_effort"]

# Upper bound on how much sim time one idle tick may advance, in physics
# steps. Guards the wall-time catch-up in ``idle_step``: if the executor stalls
# (a long render, a GC pause) the next tick must not fast-forward the world by seconds.
_IDLE_STEP_CAP = 200

# The twin's log lines keep the logger they were always emitted under.
log = structlog.get_logger("openral_hal._mujoco_arm")


def joint_actuator_effort(data: Any, dof: int) -> float:  # noqa: ANN401  # reason: mujoco.MjData
    """The sim's ``/joint_states.effort`` for one joint: MuJoCo ``qfrc_actuator`` at its DOF.

    The generalized actuator force on the joint (gear and transmission applied, every
    actuator on the DOF summed), gravity load included — what a current-sensing drive
    reports. The one definition every MuJoCo-backed HAL publishes (#373).
    """
    return float(data.qfrc_actuator[dof])


def _gripper_command_to_raw(gripper: SimGripperDescription, command: float) -> float:
    """Map the public gripper command to the actuator's native range."""
    if gripper.write_mode is GripperWriteMode.PASSTHROUGH:
        return command
    low, high = gripper.ctrl_range
    return low + max(0.0, min(1.0, command)) * (high - low)


class TwinKwargs(TypedDict, total=False):
    """``MujocoTwinRollout``'s keyword arguments: the one typed shape of the twin's wiring.

    ``MujocoArmHAL.__init__`` takes them, ``MujocoArmHAL._sim_kwargs_for`` builds them
    from ``description.sim`` and a robot's rollout subclass forwards them.
    """

    mjcf_path: Required[str]
    joint_qpos_addr: Required[dict[str, int]]
    actuator_index: Required[dict[str, int]]
    joint_qvel_addr: NotRequired[dict[str, int] | None]
    grippers: NotRequired[Sequence[SimGripperDescription]]
    keyframe_index: NotRequired[int | None]
    seed_ctrl_from_qpos: NotRequired[bool]
    settle_steps: NotRequired[int | None]
    gravity_enabled: NotRequired[bool]
    backend: NotRequired[str]
    staleness_limit_s: NotRequired[float]
    body_twist_dt_s: NotRequired[float | None]


class TwinStepResult(BaseModel):
    """One twin transition, as ``SimRolloutHAL`` reads it (frozen).

    ``info["applied_action"]`` is the command the step applied (``step_action_group``):
    the composed full-dof action for a slot group, the action itself otherwise. No
    ``observation``: the twin renders lazily (``read_images``) and reads its state live
    (``read_joint_state``), so nothing is cached per step.
    """

    model_config = ConfigDict(frozen=True)

    info: dict[str, Any] = Field(default_factory=dict)


class MujocoTwinRollout:
    """A robot's own MJCF as a ``SimRollout``: the physics of ``MujocoArmHAL``.

    Args:
        description: Normative ``RobotDescription``; ``description.joints`` is the
            order of every target row and joint-state vector.
        mjcf_path: Filesystem path to the MJCF XML.
        joint_qpos_addr: ``joint_name -> qpos index``.
        actuator_index: ``joint_name -> ctrl column``.
        joint_qvel_addr: ``joint_name -> qvel index``; defaults to ``joint_qpos_addr``.
        grippers: ``SimGripperDescription`` entries, each on a ``description.joints`` joint.
        keyframe_index: Keyframe ``load`` resets the data to, when set.
        seed_ctrl_from_qpos: Seed each position actuator's ctrl from its joint's qpos.
        settle_steps: Pinned physics steps per ``step``; ``None`` derives one control
            period at ``load`` (``_resolve_time_base``).
        gravity_enabled: ``False`` zeroes gravity at ``load``.
        backend: The HAL class name its messages and time-base log lines carry.
        staleness_limit_s: Gap (seconds) since the twin last stepped or was read above
            which ``read_joint_state`` logs a one-shot starvation WARNING.
        body_twist_dt_s: Seconds one ``BODY_TWIST`` command integrates a base by.
            ``None`` (default) is the twin's control period (``sim_dt_per_tick_s``, or
            ``settle_steps`` physics steps while wall-paced), resolved at ``load``; an
            explicit value that disagrees with a timing-faithful period is refused
            there. Only a twin with a base reads it.

    Raises:
        ROSConfigError: ``description.joints`` is empty, the robot name is not a file
            name, or ``grippers`` repeats a joint or names one outside the manifest.
    """

    #: The clock id the shell's ``clock_authority`` reports for this env.
    clock_id = "mujoco"
    #: The shell hands every command to ``step_action_group``, an ungrouped action as a
    #: group of one (``SimRolloutHAL``'s explicit route for this env; see ``SimRollout``).
    steps_every_command_as_group: ClassVar[bool] = True

    def __init__(self, description: RobotDescription, **kwargs: Unpack[TwinKwargs]) -> None:
        """Bind the configuration; the MJCF is not loaded until ``load``."""
        mjcf_path = kwargs["mjcf_path"]
        joint_qpos_addr = kwargs["joint_qpos_addr"]
        joint_qvel_addr = kwargs.get("joint_qvel_addr")
        grippers = kwargs.get("grippers", ())
        settle_steps = kwargs.get("settle_steps")
        backend = kwargs.get("backend", "MujocoArmHAL")
        if not description.joints:
            raise ROSConfigError(
                f"RobotDescription '{description.name}' has no joints; cannot initialise {backend}."
            )
        # load() writes ``<name>_camrig.xml`` beside the source MJCF; a
        # separator in the name would pick a different directory for that write.
        if os.path.basename(description.name) != description.name or (
            os.altsep and os.altsep in description.name
        ):
            raise ROSConfigError(
                f"RobotDescription name {description.name!r} contains a path separator; "
                f"{backend} uses it as a file name."
            )
        self.description = description
        self.mjcf_path = mjcf_path
        self.joint_qpos_addr = dict(joint_qpos_addr)
        self.joint_qvel_addr = (
            dict(joint_qvel_addr) if joint_qvel_addr is not None else dict(joint_qpos_addr)
        )
        self.actuator_index = dict(kwargs["actuator_index"])
        self.grippers_by_joint: dict[str, SimGripperDescription] = {g.joint: g for g in grippers}
        if len(self.grippers_by_joint) != len(list(grippers)):
            duplicates = [g.joint for g in grippers]
            raise ROSConfigError(f"grippers contains duplicate joint names: {duplicates}")
        joint_name_set = {j.name for j in description.joints}
        for g_joint in self.grippers_by_joint:
            if g_joint not in joint_name_set:
                raise ROSConfigError(
                    f"grippers[].joint={g_joint!r} is not present in "
                    f"description.joints (have: {sorted(joint_name_set)})"
                )
        self.joint_names: list[str] = [j.name for j in description.joints]
        self.keyframe_index = kwargs.get("keyframe_index")
        self.seed_ctrl_from_qpos = kwargs.get("seed_ctrl_from_qpos", False)
        self.settle_steps_pinned = settle_steps
        # Resolved at ``load`` (``_resolve_time_base``); the pinned count or the
        # legacy single step until then.
        self.settle_steps: int = 1 if settle_steps is None else int(settle_steps)
        # Sim seconds one ``step`` advances once held to the control rate;
        # ``None`` while pinned / rate-less (not timing-faithful).
        self.sim_dt_per_tick_s: float | None = None
        # The explicit twist interval, and the one resolved at ``load`` (the period).
        self.body_twist_dt_s_pinned = kwargs.get("body_twist_dt_s")
        self.body_twist_dt_s: float | None = None
        self.gravity_enabled = kwargs.get("gravity_enabled", True)
        self.backend = backend
        # Joints a command may leave uncommanded: they keep their current ctrl
        # target (``compose_slot_group(hold=...)``). Empty = every joint must be
        # commanded. The OpenAMRobot twin lists its base and lift.
        self.hold_if_uncommanded: frozenset[str] = frozenset()
        self.staleness_limit_s = kwargs.get("staleness_limit_s", 0.5)
        # Monotonic time the twin last stepped, loaded, snapped or was read; a read
        # long after it means the HAL's publish loop was starved (``read_joint_state``).
        self.last_state_time: float = 0.0
        # One-shot guard so a sustained executor stall logs a single starvation
        # WARNING instead of one per tick.
        self._starvation_warned = False
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        # Lazily-created offscreen renderers keyed by (height, width): every camera
        # renders at ITS OWN declared intrinsics, not a shared max. Created on the
        # first ``read_images`` call so the EGL context binds on the caller's thread
        # (EGL contexts are thread-affine).
        self.renderers: dict[tuple[int, int], Any] = {}  # (h, w) -> mujoco.Renderer
        # Serialises a render (the bridge's camera timer) against the renderers'
        # teardown (``close``, from a lifecycle transition on another thread):
        # closing a GL context mid ``render()`` is a use-after-free.
        self.render_lock = threading.RLock()
        self._render_failed: bool = False
        self._mjcf_cameras: set[str] = set()
        self._render_missing_warned: set[str] = set()

    # ── SimRollout face ─────────────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        """Width of a ``step`` row: one target per ``description.joints`` joint."""
        return len(self.joint_names)

    def load(self) -> None:
        """Load the MJCF into a fresh ``MjData`` and resolve the time base.

        Raises:
            ROSConfigError: ``mujoco`` is not installed, the MJCF cannot be read or
                loaded, or ``keyframe_index`` is out of range.
        """
        try:
            import mujoco as mj  # reason: optional sim-only dep
        except ModuleNotFoundError as exc:
            raise ROSConfigError(
                "mujoco is not installed. Install the sim extras with: "
                "just sync --all-packages --group sim"
            ) from exc

        # Generic camera rig: splice the manifest's declared RGB cameras into the
        # MJCF when they are absent (a bare-arm twin ships none). Idempotent: an
        # MJCF that already has them loads from the original path unchanged.
        mjcf_path = self.mjcf_path
        try:
            from openral_hal._camera_rig import rig_cameras_into_mjcf

            with open(self.mjcf_path) as fh:
                rigged_xml, changed = rig_cameras_into_mjcf(
                    fh.read(), list(self.description.sensors)
                )
            if changed:
                # Write next to the source MJCF so its relative meshdir resolves.
                rig_path = os.path.join(
                    os.path.dirname(self.mjcf_path),
                    f"{self.description.name}_camrig.xml",
                )
                with open(rig_path, "w") as fh:
                    fh.write(rigged_xml)
                mjcf_path = rig_path
        except OSError as exc:
            raise ROSConfigError(f"Could not read MJCF '{self.mjcf_path}': {exc}") from exc

        try:
            model = mj.MjModel.from_xml_path(mjcf_path)
        except (OSError, ValueError) as exc:
            raise ROSConfigError(f"Could not load MJCF '{mjcf_path}': {exc}") from exc

        if not self.gravity_enabled:
            model.opt.gravity[:] = 0.0
        data = mj.MjData(model)
        if self.keyframe_index is not None:
            if self.keyframe_index >= model.nkey:
                raise ROSConfigError(
                    f"keyframe_index={self.keyframe_index} but the MJCF has "
                    f"only {model.nkey} keyframes ({self.mjcf_path})"
                )
            mj.mj_resetDataKeyframe(model, data, self.keyframe_index)
        mj.mj_forward(model, data)
        # Nothing is assigned before the MJCF has loaded: a refused load leaves the
        # rollout closed (no clock, no handles, no renders), and so does a refused
        # time base or robot setup below.
        self.model, self.data = model, data
        try:
            if self.seed_ctrl_from_qpos:
                self._hold_qpos_in_ctrl(self.joint_names)
            self._resolve_time_base()
            self._resolve_twist_interval()
            self._on_load()
        except BaseException:
            self.close()
            raise
        self.last_state_time = time.monotonic()
        log.info(
            "hal.connect",
            robot=self.description.name,
            mjcf=self.mjcf_path,
            n_joints=len(self.joint_names),
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """(Re)load the twin into its initial state: ``close``, then ``load``.

        Deterministic: ``seed`` is unused.

        Raises:
            ROSConfigError: See ``load``.
        """
        del seed
        self.close()
        self.load()
        return {}

    def step(self, action: Sequence[float]) -> TwinStepResult:
        """The plain ``SimRollout`` face: a full-dof target row as one ungrouped command.

        Delegates to ``step_action_group`` (a ``JOINT_POSITION`` action of the row), so
        there is one command path; the shell never calls it.

        Args:
            action: Full-dof targets in ``description.joints`` order, public units.
        """
        return self.step_action_group(
            [
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[[float(v) for v in action]],
                )
            ]
        )

    def step_action_group(self, actions: list[Action]) -> TwinStepResult:
        """Apply one command for one control period: a slot group, or one ungrouped action.

        ``_compose`` validates the command and turns it into a full-dof target row and/or
        a planar twist; the row goes into ``ctrl`` (``_apply_targets``), the twist drives
        the base (``_drive_base``), and ONE ``advance`` runs the period, so a tick that
        carries joint and twist slots moves both inside the same control period.

        Raises:
            ROSConfigError: An unsupported control mode, a malformed or unplaceable
                command (``_compose``).
        """
        targets, twist, applied = self._compose(actions)
        assert self.model is not None and self.data is not None
        self._command(targets, twist)
        self.advance(self._action_steps(twist))
        return TwinStepResult(info={"applied_action": applied})

    def advance(self, steps: int) -> None:
        """The one stepping loop: ``steps`` times the per-step hook, then ``mj_step``.

        Every action and idle slice of the twin hierarchy steps physics here.
        """
        import mujoco as mj  # reason: optional sim-only dep

        model, data = self.model, self.data
        assert model is not None and data is not None
        for _ in range(steps):
            self._per_step()
            mj.mj_step(model, data)
        # A long settle is not a starved publish loop.
        self.last_state_time = time.monotonic()

    def close(self) -> None:
        """Close the renderers and release the model and data. Idempotent."""
        with self.render_lock:
            for renderer in self.renderers.values():
                try:
                    renderer.close()
                except (
                    Exception
                ) as exc:  # reason: a GL teardown failure must not keep the model alive
                    log.warning(
                        "hal.renderer_close_failed", robot=self.description.name, error=str(exc)
                    )
            self.renderers.clear()
            self.model = None
            self.data = None

    # ── SimRolloutHAL env hooks ─────────────────────────────────────────────

    def mujoco_handles(self) -> tuple[Any, Any] | None:
        """The live ``(model, data)``, or ``None`` before ``load`` / after ``close``."""
        if self.model is None or self.data is None:
            return None
        return self.model, self.data

    @property
    def physics_dt_s(self) -> float | None:
        """The physics timestep, ``None`` before ``load`` / after ``close``.

        The shell's ``clock_authority`` reports it while the twin is wall-paced (no
        ``sim_dt_per_tick_s``).
        """
        return None if self.model is None else float(self.model.opt.timestep)

    @property
    def time_base_fields(self) -> dict[str, str | float | int]:
        """The shell's ``sim.time_base`` fields: the HAL class (``backend``) and physics.

        Empty until the time base is held to the rate (``sim_dt_per_tick_s``).
        """
        if self.model is None or self.sim_dt_per_tick_s is None:
            return {}
        return {
            "backend": self.backend,
            "physics_dt_s": round(float(self.model.opt.timestep), 9),
            "physics_steps": self.settle_steps,
        }

    def sim_time_ns(self) -> int | None:
        """``MjData.time`` in ns, or ``None`` before ``load`` / after ``close``."""
        if self.data is None:
            return None
        return round(float(self.data.time) * 1_000_000_000)

    def idle_step(self, wall_dt_s: float | None = None) -> bool:
        """Advance a wall-time slice of sim time holding the last command.

        ``ctrl`` already holds the last commanded / seeded pose, so this HOLD slice
        does not move the arm; the per-step hook keeps running (a PD loop tracks its
        last targets, a pinned base stays pinned), after ``_on_idle``. One ``mj_step``
        per idle tick would make sim time crawl at ``timestep * tick_rate`` and starve
        every sim-clock timer; stepping the tick's worth keeps the clocks in step,
        capped at ``_IDLE_STEP_CAP`` steps so a stall cannot fast-forward the world.

        Args:
            wall_dt_s: Wall seconds the tick represents; ``None`` is one step.

        Returns:
            ``True`` if the sim advanced, ``False`` before ``load`` / after ``close``.
        """
        if self.data is None or self.model is None:
            return False
        steps = 1
        if wall_dt_s is not None and wall_dt_s > 0:
            timestep = float(self.model.opt.timestep)
            if timestep > 0:
                steps = max(1, min(_IDLE_STEP_CAP, round(wall_dt_s / timestep)))
        self._on_idle()
        self.advance(steps)
        return True

    def read_joint_state(self) -> JointState:
        """The live joint state in ``description.joints`` order.

        Grippers report their public position (``read_mode``); a PASSTHROUGH
        gripper's qvel is meaningful, a normalised one reports 0. Efforts are
        ``joint_actuator_effort`` at each joint's DOF.

        The read is the live in-process ``MjData`` — always the *current* simulator
        state — so a large gap since the twin last stepped or was read is never bad
        data: it means the HAL's publish loop was starved (a slow camera render
        hogging the single-threaded executor). That is a one-shot WARNING
        (``hal.read_state.starved``), re-armed by a healthy read, never a latched
        ``ROSPerceptionStale`` (which bricked the HAL on one transient stall).
        """
        data = self.data
        assert data is not None
        age = time.monotonic() - self.last_state_time
        if age > self.staleness_limit_s:
            if not self._starvation_warned:
                self._starvation_warned = True
                log.warning(
                    "hal.read_state.starved",
                    robot=self.description.name,
                    age_s=round(age, 3),
                    limit_s=self.staleness_limit_s,
                    note=(
                        "publish loop starved > staleness_limit_s (likely the "
                        "single-threaded executor blocked on rendering); serving "
                        "live MjData and refreshing the clock — not latching stale"
                    ),
                )
        else:
            self._starvation_warned = False
        self.last_state_time = time.monotonic()
        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []
        for name in self.joint_names:
            gripper = self.grippers_by_joint.get(name)
            if gripper is not None:
                positions.append(self._read_gripper_value(gripper))
                if gripper.read_mode is GripperReadMode.PASSTHROUGH:
                    velocities.append(float(data.qvel[gripper.qpos_addrs[0]]))
                else:
                    velocities.append(0.0)
                efforts.append(joint_actuator_effort(data, gripper.qpos_addrs[0]))
            else:
                qpos_addr = self.joint_qpos_addr[name]
                qvel_addr = self.joint_qvel_addr[name]
                positions.append(float(data.qpos[qpos_addr]))
                velocities.append(float(data.qvel[qvel_addr]))
                efforts.append(joint_actuator_effort(data, qvel_addr))
        return JointState(
            name=list(self.joint_names),
            position=positions,
            velocity=velocities,
            effort=efforts,
            stamp_ns=time.time_ns(),
        )

    def read_images(self) -> dict[str, Any]:
        """Render the manifest's RGB cameras off the live MJCF (issue #191 Phase 3b).

        ``{sensor.name: HxWx3 uint8}``, each from the MJCF camera
        ``sensor.sim_camera_name or sensor.name`` at its own ``intrinsics``
        resolution (one ``mujoco.Renderer`` per distinct ``(height, width)``). A
        camera absent from the MJCF, or a render error, is skipped with a one-shot
        warning, never raising. ``{}`` before ``load``, after ``close``, without RGB
        sensors, or after a renderer failure. Holds ``render_lock`` throughout, so
        a concurrent ``close`` waits for it.
        """
        with self.render_lock:
            return self._read_images_locked()

    def _read_images_locked(self) -> dict[str, Any]:
        if self.model is None or self.data is None:
            return {}
        rgb = [s for s in self.description.sensors if s.modality == "rgb"]
        if not rgb or self._render_failed:
            return {}
        try:
            import mujoco as mj  # reason: optional sim-only dep
        except ModuleNotFoundError:
            self._render_failed = True
            return {}

        if not self._mjcf_cameras:
            self._mjcf_cameras = {
                mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_CAMERA, i)
                for i in range(int(self.model.ncam))
            }

        frames: dict[str, Any] = {}
        for s in rgb:
            cam = s.sim_camera_name or s.name
            if cam not in self._mjcf_cameras:
                if cam not in self._render_missing_warned:
                    self._render_missing_warned.add(cam)
                    log.warning(
                        "hal.read_images.camera_absent",
                        robot=self.description.name,
                        camera=cam,
                        available=sorted(c for c in self._mjcf_cameras if c),
                    )
                continue
            height = s.intrinsics.height if s.intrinsics is not None else 480
            width = s.intrinsics.width if s.intrinsics is not None else 640
            renderer = self.renderers.get((height, width))
            if renderer is None:
                # The MJCF's offscreen framebuffer (default 640x480) must fit the
                # sensor raster, e.g. the OpenArm ZED's 672x376 — grow it first.
                vis = self.model.vis.global_
                vis.offwidth = max(int(vis.offwidth), width)
                vis.offheight = max(int(vis.offheight), height)
                try:
                    renderer = mj.Renderer(self.model, height=height, width=width)
                except Exception as exc:  # reason: GL/MJCF render setup is non-fatal
                    log.warning(
                        "hal.read_images.renderer_failed",
                        robot=self.description.name,
                        error=str(exc),
                    )
                    self._render_failed = True
                    self.renderers.clear()
                    return {}
                self.renderers[(height, width)] = renderer
            try:
                renderer.update_scene(self.data, camera=cam)
                frames[s.name] = renderer.render()
            except Exception as exc:  # reason: a per-tick render hiccup skips one frame
                log.warning(
                    "hal.read_images.render_failed",
                    robot=self.description.name,
                    camera=cam,
                    error=str(exc),
                )
        return frames

    # ── Twin operations ─────────────────────────────────────────────────────

    def reset_to_pose(self, pose: list[float]) -> None:
        """Snap ``qpos`` to ``pose`` (velocities zeroed) and re-seed ``ctrl`` to hold it.

        Args:
            pose: Positions in ``description.joints`` order; grippers in public units.

        Raises:
            ROSConfigError: ``len(pose)`` is not the joint count, or a gripper has no
                actuator index.
        """
        model, data = self.model, self.data
        assert model is not None and data is not None
        if len(pose) != len(self.joint_names):
            raise ROSConfigError(
                f"reset_to_pose: {len(pose)} entries, expected "
                f"{len(self.joint_names)} (description.joints order).",
            )
        import mujoco as mj  # reason: optional sim-only dep

        for name, value in zip(self.joint_names, pose, strict=True):
            qpos_addr = self.joint_qpos_addr.get(name)
            if qpos_addr is None:
                continue
            gripper = self.grippers_by_joint.get(name)
            if gripper is None:
                data.qpos[qpos_addr] = float(value)
            else:
                self._reset_gripper_qpos(gripper, float(value))
            qvel_addr = self.joint_qvel_addr.get(name)
            if qvel_addr is not None:
                # Zero qvel too so the snap doesn't carry momentum.
                data.qvel[qvel_addr] = 0.0
        mj.mj_forward(model, data)
        # Re-seed ctrl so the actuators hold the new pose on the next ``mj_step``:
        # position actuators from qpos, grippers through their write mode.
        self._hold_qpos_in_ctrl([n for n in self.joint_names if n not in self.grippers_by_joint])
        if self.grippers_by_joint:
            self.apply_gripper_targets([float(v) for v in pose])
        self.last_state_time = time.monotonic()

    def apply_arm_targets(self, targets: list[float]) -> None:
        """Write each non-gripper joint's target into its position actuator's ``ctrl``."""
        assert self.data is not None
        for name, value in zip(self.joint_names, targets, strict=True):
            if name in self.grippers_by_joint:
                continue
            act_idx = self.actuator_index.get(name)
            if act_idx is None:
                continue
            self.data.ctrl[act_idx] = float(value)

    def apply_gripper_targets(self, targets: list[float]) -> None:
        """Write the per-gripper command to ``ctrl`` (and mirror, if any).

        ``NORMALISED`` write mode clips the input to ``[0, 1]`` and maps it affinely
        to ``gripper.ctrl_range``; ``PASSTHROUGH`` writes it verbatim (MuJoCo's
        actuator ``ctrlrange`` clips at simulation time).

        Raises:
            ROSConfigError: A gripper has no actuator index.
        """
        assert self.data is not None
        for joint_name, gripper in self.grippers_by_joint.items():
            command = float(targets[self.joint_names.index(joint_name)])
            raw = _gripper_command_to_raw(gripper, command)
            primary_idx = self._gripper_actuator_index(gripper, joint_name)
            if primary_idx is None:
                raise ROSConfigError(
                    f"Gripper {joint_name!r} has no actuator index — set "
                    "sim.grippers[].actuator_index or include the joint in "
                    "sim.actuator_index."
                )
            self.data.ctrl[primary_idx] = raw
            if gripper.mirror_actuator_index is not None:
                self.data.ctrl[gripper.mirror_actuator_index] = -raw

    # ── Commands (per-robot hooks; a subclass overrides what its robot does) ──

    def _compose(
        self, actions: list[Action]
    ) -> tuple[list[float] | None, tuple[float, float, float] | None, Action]:
        """Validate one command; its full-dof target row, its ``(vx, vy, wz)``, what it applied.

        A slot group (or an ungrouped row naming a subset of the joints, a ramp of a
        slot-subset starting pose) is composed by name (``compose_slot_group_action``),
        joints in ``hold_if_uncommanded`` keeping their current ctrl target; at most
        one ``BODY_TWIST`` slot drives the base (``_split_twist``).

        Raises:
            ROSConfigError: See ``_validate``, ``_split_twist``, ``compose_slot_group``.
        """
        twist, joints = self._split_twist(actions)
        if not joints:
            return None, twist, actions[0]
        command = joints[0]
        if len(actions) > 1 or (
            command.joint_names and set(command.joint_names) != set(self.joint_names)
        ):
            command = compose_slot_group_action(
                joints, self.joint_names, hold=self._held_ctrl_targets()
            )
        self._validate(command)
        assert command.joint_targets is not None
        return list(command.joint_targets[-1]), twist, command

    def _split_twist(
        self, actions: list[Action]
    ) -> tuple[tuple[float, float, float] | None, list[Action]]:
        """The command's planar twist (at most one ``BODY_TWIST`` slot) and its other slots.

        Raises:
            ROSConfigError: Two twist slots, or a twist ``_validate`` refuses.
        """
        twists = [a for a in actions if a.control_mode is ControlMode.BODY_TWIST]
        if len(twists) > 1:
            raise ROSConfigError(f"{self.backend}: {len(twists)} BODY_TWIST slots in one tick.")
        for slot in twists:
            self._validate(slot)
        joints = [a for a in actions if a.control_mode is not ControlMode.BODY_TWIST]
        return (self._planar_twist(twists[0]) if twists else None), joints

    def _validate(self, action: Action) -> None:
        """Refuse an unsupported mode or a malformed command before anything moves.

        Raises:
            ROSConfigError: A mode outside ``supported_control_modes``, a twist that
                ``_planar_twist`` refuses, a non-``JOINT_POSITION`` joint command, or a
                row that is not one target per manifest joint.
        """
        require_supported_control_mode(self.description, action)
        if action.control_mode is ControlMode.BODY_TWIST:
            self._planar_twist(action)
            return
        require_control_mode(action, ControlMode.JOINT_POSITION, self.backend)
        if not action.joint_targets or not action.joint_targets[0]:
            raise ROSConfigError("action.joint_targets[0] is required.")
        validate_action_dims(action, len(self.joint_names), self.description.name)

    def _planar_twist(self, action: Action) -> tuple[float, float, float]:
        """``(vx, vy, wz)`` of a ``BODY_TWIST`` (``validate_planar_body_twist``).

        A manifest that declares ``base_kinematics: differential`` refuses a lateral vy.
        """
        return validate_planar_body_twist(
            action,
            self.backend,
            differential=self.description.base_kinematics == "differential",
        )

    def _command(
        self, targets: list[float] | None, twist: tuple[float, float, float] | None
    ) -> None:
        """Write one command before its period: the target row, then the base twist."""
        if targets is not None:
            self._apply_targets(targets)
        if twist is not None:
            self._drive_base(*twist)

    def _apply_targets(self, targets: list[float]) -> None:
        """Write a full-dof row into ``ctrl``: arm joints verbatim, grippers by write mode."""
        self.apply_arm_targets(targets)
        if self.grippers_by_joint:
            self.apply_gripper_targets(targets)

    def _drive_base(self, vx: float, vy: float, wz: float) -> None:
        """Move the base by one twist command; a fixed-base twin has none.

        Raises:
            ROSConfigError: Always, here (a robot whose manifest declares no twist).
        """
        raise ROSConfigError(f"{self.backend}: robot '{self.description.name}' has no base.")

    def _action_steps(self, twist: tuple[float, float, float] | None) -> int:
        """Physics steps one command advances: one control period."""
        del twist
        return self.settle_steps

    def _per_step(self) -> None:
        """Before every ``mj_step`` of ``advance``; position actuators need nothing."""

    def _on_idle(self) -> None:
        """Before an idle slice; nothing to change by default."""

    def _on_load(self) -> None:
        """After the MJCF is loaded and the time base resolved (robot-specific setup)."""

    def _held_ctrl_targets(self) -> dict[str, float]:
        """Current ctrl target of each ``hold_if_uncommanded`` joint."""
        assert self.data is not None
        return {
            name: float(self.data.ctrl[self.actuator_index[name]])
            for name in self.hold_if_uncommanded
            if name in self.actuator_index
        }

    def _require_loaded(self, operation: str) -> tuple[Any, Any]:
        """The live ``(model, data)``.

        Raises:
            ROSRuntimeError: Before ``load`` / after ``close`` (``_base.require_connected``:
                the HAL is not connected).
        """
        model, data = self.model, self.data
        require_connected(model is not None and data is not None, self.backend, operation)
        return model, data

    # ── Internals ───────────────────────────────────────────────────────────

    def _hold_qpos_in_ctrl(self, names: list[str]) -> None:
        """Each named joint's actuator holds its current qpos (``seed_ctrl_from_qpos``, snaps)."""
        assert self.data is not None
        for name in names:
            act_idx = self.actuator_index.get(name)
            qpos_addr = self.joint_qpos_addr.get(name)
            if act_idx is None or qpos_addr is None:
                continue
            self.data.ctrl[act_idx] = float(self.data.qpos[qpos_addr])

    def _resolve_time_base(self) -> None:
        """Make one action one control period of the robot (issue #358).

        With no pinned ``settle_steps`` and a declared
        ``description.control_rate_hz``, derives the physics steps per action via
        ``physics_steps_per_tick`` — shrinking ``opt.timestep`` (never growing it) so
        the period is a whole number of steps — and declares the period as
        ``sim_dt_per_tick_s``, which the shell holds to the rate (``check_time_base``)
        and logs once per connect (``sim.time_base``). A pinned count is honoured and
        logged as an override (``hal.time_base.pinned``); a manifest without a rate
        keeps the legacy single step (``hal.time_base.unresolved``). Neither of those
        is timing-faithful: ``sim_dt_per_tick_s`` stays ``None``.
        """
        assert self.model is not None
        rate = self.description.control_rate_hz
        physics_dt = float(self.model.opt.timestep)
        self.sim_dt_per_tick_s = None
        if self.settle_steps_pinned is not None:
            self.settle_steps = int(self.settle_steps_pinned)
            log.warning(
                "hal.time_base.pinned",
                robot=self.description.name,
                settle_steps=self.settle_steps,
                sim_dt_per_action_s=round(self.settle_steps * physics_dt, 6),
                control_freq_hz=rate,
                hint="pinned settle_steps: wall-paced, not one control period per action",
            )
            return
        if rate is None:
            self.settle_steps = 1
            log.warning(
                "hal.time_base.unresolved",
                robot=self.description.name,
                hint="declare action_spec.control_freq_hz so one action is one control period",
            )
            return
        steps, dt = physics_steps_per_tick(rate, physics_dt)
        if dt != physics_dt:
            self.model.opt.timestep = dt
            log.info(
                "hal.time_base.timestep_shrunk",
                robot=self.description.name,
                mjcf_timestep_s=physics_dt,
                timestep_s=round(dt, 9),
                control_freq_hz=rate,
            )
        self.settle_steps = steps
        self.sim_dt_per_tick_s = steps * dt

    def _resolve_twist_interval(self) -> None:
        """Make one ``BODY_TWIST`` command integrate the twin's own control period.

        The period is ``sim_dt_per_tick_s``, or ``settle_steps`` physics steps while the
        twin is wall-paced: one command advances that much physics, so a base driven by
        any other interval would move at another speed than commanded on the sim clock.
        ``body_twist_dt_s=None`` takes the period; an explicit value is honoured on a
        wall-paced twin only.

        Raises:
            ROSConfigError: An explicit ``body_twist_dt_s`` that is not positive, or that
                differs from a timing-faithful period by more than ``TIME_BASE_TOL``.
        """
        assert self.model is not None
        period = self.sim_dt_per_tick_s or self.settle_steps * float(self.model.opt.timestep)
        pinned = self.body_twist_dt_s_pinned
        if pinned is None:
            self.body_twist_dt_s = period
            return
        if pinned <= 0.0 or (
            self.sim_dt_per_tick_s is not None and abs(pinned - period) > TIME_BASE_TOL * period
        ):
            raise ROSConfigError(
                f"{self.backend}: body_twist_dt_s={pinned} s but one command advances "
                f"{period:.6f} s of physics (one control period); a twist would drive the "
                "base at another speed than commanded. Leave body_twist_dt_s unset."
            )
        self.body_twist_dt_s = float(pinned)

    def _gripper_actuator_index(
        self, gripper: SimGripperDescription, joint_name: str
    ) -> int | None:
        """``gripper.actuator_index`` when set, else the joint-wide ``actuator_index``."""
        if gripper.actuator_index is not None:
            return gripper.actuator_index
        return self.actuator_index.get(joint_name)

    def _reset_gripper_qpos(self, gripper: SimGripperDescription, value: float) -> None:
        """Write a public gripper value into its native qpos representation."""
        assert self.data is not None
        if gripper.read_mode is GripperReadMode.AFFINE_LOW_HIGH:
            self.data.qpos[gripper.qpos_addrs[0]] = _gripper_command_to_raw(gripper, value)
            return
        if gripper.read_mode is GripperReadMode.PASSTHROUGH:
            self.data.qpos[gripper.qpos_addrs[0]] = value
            return
        each = max(0.0, min(1.0, value)) * gripper.qpos_scale / len(gripper.qpos_addrs)
        for addr in gripper.qpos_addrs:
            self.data.qpos[addr] = each

    def _read_gripper_value(self, gripper: SimGripperDescription) -> float:
        """The gripper's public position according to ``gripper.read_mode``."""
        assert self.data is not None
        if gripper.read_mode is GripperReadMode.PASSTHROUGH:
            return float(self.data.qpos[gripper.qpos_addrs[0]])
        if gripper.read_mode is GripperReadMode.AFFINE_LOW_HIGH:
            low, high = gripper.ctrl_range
            span = high - low
            if span == 0.0:
                return 0.0
            qpos = float(self.data.qpos[gripper.qpos_addrs[0]])
            return max(0.0, min(1.0, (qpos - low) / span))
        # SUM_OVER_SCALE (default)
        total = sum(float(self.data.qpos[i]) for i in gripper.qpos_addrs)
        if gripper.qpos_scale == 0.0:
            return 0.0
        return max(0.0, min(1.0, total / gripper.qpos_scale))


# ── Building blocks of the humanoid twins (G1, H1, H2) ───────────────────────


class _GlideTwin(MujocoTwinRollout):
    """ADR-0087's upright kinematic glide: the free joint re-pinned before every step.

    The pin ``(x, y, yaw, z0)`` is captured from qpos on first use after a load and
    written into the free joint (roll = pitch = 0, velocity zeroed) before every physics
    step, actions and idle slices alike, so a gravity-on twin stands without an S0
    controller. A ``BODY_TWIST`` slides the pin by one ``body_twist_dt_s`` of
    ``(vx, vy, wz)`` (``integrate_planar_twist``). No balance, no gait.

    A ``reset_to_pose`` keeps the pin: the snap moves no free joint, and recapturing it
    from qpos would read the free joint one physics step after the pin was written —
    lower by that step's fall, so every snap would sink the base.
    """

    _base_pin: BasePin | None = None

    def _on_load(self) -> None:
        super()._on_load()
        self._base_pin = None

    def read_base_pose(self) -> tuple[float, float, float]:
        """The pinned base ``(x, y, yaw)``.

        Raises:
            ROSRuntimeError: Not loaded (the HAL is not connected).
        """
        x, y, yaw, _z0 = self._pin()
        return (x, y, yaw)

    def _drive_base(self, vx: float, vy: float, wz: float) -> None:
        assert self.body_twist_dt_s is not None
        self._base_pin = integrate_planar_twist(self._pin(), vx, vy, wz, self.body_twist_dt_s)

    def _per_step(self) -> None:
        """Write the pin into the free joint, then the next block's hook."""
        assert self.data is not None
        write_base_pin(self.data, self._pin())
        super()._per_step()

    def _pin(self) -> BasePin:
        """The glide pin ``(x, y, yaw, z0)``, captured from qpos on first use."""
        _model, data = self._require_loaded("base_pose")
        if self._base_pin is None:
            self._base_pin = capture_base_pin(data.qpos)
        return self._base_pin


class _FreeBaseTwin(MujocoTwinRollout):
    """A twin whose base pose is the free joint's own (it walks or balances, unpinned)."""

    def read_base_pose(self) -> tuple[float, float, float]:
        """The free base's current ``(x, y, yaw)``.

        Raises:
            ROSRuntimeError: Not loaded (the HAL is not connected).
        """
        _model, data = self._require_loaded("base_pose")
        x, y, yaw, _z = capture_base_pin(data.qpos)
        return (x, y, yaw)


class _SoftwarePDTwin(MujocoTwinRollout):
    """A position loop in software over an MJCF's torque (``motor``) actuators.

    ``ctrl`` is N·m on such an actuator, so a command's target row never reaches it:
    it becomes the PD target, and every physics step (actions and idle slices alike)
    writes ``tau = kp (target - q) - kd dq``, clamped to the actuator's ``ctrlrange`` —
    the public ``Action.joint_targets`` contract stays "position targets", as on every
    twin, and the split is the real ``unitree_sdk2`` driver's (torque at the bus,
    position + Kp/Kd above it). Index arrays and gains are built once per load.

    Before the first command there is no target and no torque, unless the subclass
    holds its loaded pose (``hold_pose_on_load``). A ``reset_to_pose`` zeroes ``ctrl``
    (the base class re-seeds it from qpos, a position-actuator idiom that would apply
    joint angles as N·m) and makes the snapped pose the target; a reload clears it.
    """

    #: Hold the loaded pose from ``load`` on (``True``), or apply no torque before the
    #: first command (``False``).
    hold_pose_on_load: ClassVar[bool] = False

    def __init__(
        self,
        description: RobotDescription,
        *,
        pd_gains: dict[str, tuple[float, float]],
        **kwargs: Unpack[TwinKwargs],
    ) -> None:
        """Bind the twin; ``pd_gains`` maps a joint to ``(kp, kd)``; ``kwargs`` are the base's."""
        super().__init__(description, **kwargs)
        self._pd_gains = pd_gains
        self._pd_targets: NDArray[np.float64] | None = None

    def _on_load(self) -> None:
        super()._on_load()
        model = self.model
        assert model is not None
        driven = [n for n in self.joint_names if n in self.actuator_index]
        self._pd_rows = np.array([self.joint_names.index(n) for n in driven], dtype=np.intp)
        self._pd_act = np.array([self.actuator_index[n] for n in driven], dtype=np.intp)
        self._pd_qpos = np.array([self.joint_qpos_addr[n] for n in driven], dtype=np.intp)
        self._pd_qvel = np.array([self.joint_qvel_addr[n] for n in driven], dtype=np.intp)
        self._pd_kp = np.array([self._pd_gains[n][0] for n in driven], dtype=np.float64)
        self._pd_kd = np.array([self._pd_gains[n][1] for n in driven], dtype=np.float64)
        ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64)[self._pd_act]
        self._pd_low, self._pd_high = ctrlrange[:, 0], ctrlrange[:, 1]
        self._pd_targets = self._driven_qpos() if self.hold_pose_on_load else None

    def reset_to_pose(self, pose: list[float]) -> None:
        """Snap the joints to ``pose``; it becomes the PD target, ``ctrl`` starts at zero."""
        super().reset_to_pose(pose)
        assert self.data is not None
        self.data.ctrl[:] = 0.0
        self._pd_targets = self._driven_qpos()

    def _apply_targets(self, targets: list[float]) -> None:
        """Make ``targets`` the PD target (``ctrl`` is torque, written every step)."""
        self._pd_targets = np.asarray(targets, dtype=np.float64)[self._pd_rows]

    def _per_step(self) -> None:
        """The next block's hook, then the PD torques from the current ``qpos`` / ``qvel``."""
        super()._per_step()
        if self._pd_targets is None:
            return
        data = self.data
        assert data is not None
        tau = (
            self._pd_kp * (self._pd_targets - data.qpos[self._pd_qpos])
            - self._pd_kd * (data.qvel[self._pd_qvel])
        )
        data.ctrl[self._pd_act] = np.clip(tau, self._pd_low, self._pd_high)

    def _driven_qpos(self) -> NDArray[np.float64]:
        assert self.data is not None
        return np.asarray(self.data.qpos[self._pd_qpos], dtype=np.float64).copy()
