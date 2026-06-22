"""Internal MuJoCo-backed HAL implementation shared by UR/Franka adapters.

This module provides :class:`MujocoArmHAL`, a HAL Protocol implementation that
drives a MuJoCo ``MjModel`` / ``MjData`` pair through position-controlled
actuators.  It is the simulation backbone behind :class:`UR5eHAL`,
:class:`UR10eHAL` and :class:`FrankaPandaHAL`.

The class is **not** part of the public ``openral_hal`` surface (leading
underscore module name).  Adapters subclass or wrap it and expose a typed
constructor specific to their robot.

Design notes
------------
* ``mujoco`` is imported lazily inside :meth:`MujocoArmHAL.connect` so the
  module loads cleanly on hosts without it (matches the ``SO100FollowerHAL``
  convention for ``lerobot``).
* ``send_action`` writes the **last** waypoint of an action chunk into
  ``data.ctrl`` and steps the simulator ``settle_steps`` times.  The HAL is a
  thin adapter — trajectory interpolation is the job of the controller above
  it (a ``ros2_control`` joint trajectory controller in production, or the
  test directly here).
* The gripper joint, when present, is a synthetic 1-DoF channel reported in
  ``[0, 1]`` regardless of the underlying actuator's native range.  The
  conversion is done by the per-robot adapter via ``gripper_ctrl_range``.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import structlog
from openral_core.assets import AssetRefError, resolve_asset
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSRuntimeError,
)
from openral_core.schemas import (
    Action,
    ControlMode,
    GripperReadMode,
    GripperWriteMode,
    JointState,
    RobotDescription,
    SimGripperDescription,
)

from openral_hal._base import HALBase

if TYPE_CHECKING:
    import mujoco

__all__ = ["MujocoArmHAL"]


class _MujocoArmInitKwargs(TypedDict):
    """Typed shape of the kwargs accepted by :meth:`MujocoArmHAL.__init__`.

    Lets :meth:`MujocoArmHAL._sim_kwargs_for` return a value that unpacks
    cleanly into the constructor under ``mypy --strict`` without the
    ``# type: ignore[arg-type]`` hatch every thin subclass used to need.
    """

    mjcf_path: str
    joint_qpos_addr: dict[str, int]
    joint_qvel_addr: dict[str, int]
    actuator_index: dict[str, int]
    grippers: Sequence[SimGripperDescription]
    keyframe_index: int | None
    seed_ctrl_from_qpos: bool
    settle_steps: int
    gravity_enabled: bool
    staleness_limit_s: float


def _resolve_mjcf_path(desc: RobotDescription) -> str:
    """Resolve ``desc.assets.mjcf`` to an absolute MJCF path via :func:`resolve_asset`.

    The MujocoArmHAL is built from a :class:`RobotDescription` alone (the
    manifest path is not threaded through ``from_description``), so no
    ``manifest_dir`` is supplied — every in-tree MJCF ref is a ``rd:`` /
    ``gym_aloha:`` / ``openarm:`` / ``menagerie:`` scheme that resolves without
    one (no robot declares a ``file:`` MJCF). ``AssetRefError`` is translated to
    :class:`ROSConfigError` at this HAL boundary, matching the contract
    ``_sim_kwargs_for`` documented for the old resolver.

    Raises:
        ROSConfigError: If ``desc.assets.mjcf`` is unset or cannot be resolved.
    """
    if not desc.assets.mjcf:
        raise ROSConfigError(
            f"RobotDescription '{desc.name}' has no `assets.mjcf` ref; "
            "cannot resolve a MuJoCo MJCF for MujocoArmHAL."
        )
    try:
        path = resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetRefError as exc:
        raise ROSConfigError(str(exc)) from exc
    if path is None:  # ros2:// is urdf-only; resolve_asset never returns None for mjcf
        raise ROSConfigError(
            f"assets.mjcf={desc.assets.mjcf!r} for '{desc.name}' did not resolve to a file."
        )
    return str(path)


log = structlog.get_logger(__name__)


class MujocoArmHAL(HALBase):
    """Generic MuJoCo-backed HAL adapter for position-controlled arms.

    Supports single-arm robots, floating-base humanoids (via the
    ``joint_qvel_addr`` offset), and bimanual robots (via multiple
    :class:`SimGripperDescription` entries with optional
    ``mirror_actuator_index`` for parallel-jaw configurations like Aloha).

    Args:
        description: Normative :class:`RobotDescription`.  ``description.joints``
            must enumerate every controllable joint exposed to higher layers in
            the same order as the robot's MuJoCo actuators.
        mjcf_path: Filesystem path to the MJCF XML.
        joint_qpos_addr: Mapping ``joint_name -> qpos index`` (mjcf coordinate
            in ``data.qpos``).
        actuator_index: Mapping ``joint_name -> actuator index`` (column of
            ``data.ctrl``).  ``None`` for read-only joints (e.g. the mirrored
            second finger of a parallel gripper).
        joint_qvel_addr: Optional mapping ``joint_name -> qvel index``.
            Defaults to ``joint_qpos_addr`` for arms without a floating
            base; for floating-base humanoids the qvel indices are offset
            by 6 (vs qpos 7) and must be passed explicitly.
        grippers: Zero or more :class:`SimGripperDescription` entries.
            Each gripper's ``joint`` must also appear in
            ``description.joints``.  Single-arm robots ship one entry (or
            none); bimanual robots (Aloha, OpenArm) ship two.
        keyframe_index: When set, :meth:`connect` calls
            ``mj_resetDataKeyframe(model, data, keyframe_index)`` before
            ``mj_forward``.  Required for MJCFs whose default
            ``MjData.qpos`` sits outside the actuator ``ctrlrange``
            (gym-aloha).
        seed_ctrl_from_qpos: When True, :meth:`connect` seeds
            ``data.ctrl[actuator] = data.qpos[joint_qpos_addr]`` for every
            controllable joint so position actuators hold the initial
            pose (OpenArm v2).
        settle_steps: Number of ``mj_step`` calls performed in
            :meth:`send_action` to advance the simulation toward the new
            target.  Defaults to ``1``.
        gravity_enabled: When ``False``, gravity is zeroed at ``connect()``
            time.  Useful for closed-loop tests where exact convergence is
            asserted.
        staleness_limit_s: Age (seconds) of the cached state above which
            ``read_state`` emits a one-shot starvation WARNING (the
            in-process read still returns live ``MjData`` — see
            :meth:`read_state`).

    Raises:
        ROSConfigError: If ``description.joints`` is empty.
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        mjcf_path: str,
        joint_qpos_addr: dict[str, int],
        actuator_index: dict[str, int],
        joint_qvel_addr: dict[str, int] | None = None,
        grippers: Sequence[SimGripperDescription] = (),
        keyframe_index: int | None = None,
        seed_ctrl_from_qpos: bool = False,
        settle_steps: int = 1,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Initialise the adapter; the MJCF is not loaded until ``connect()``."""
        if not description.joints:
            raise ROSConfigError(
                f"RobotDescription '{description.name}' has no joints; "
                "cannot initialise MujocoArmHAL."
            )

        self.description = description
        self._mjcf_path = mjcf_path
        self._joint_qpos_addr = dict(joint_qpos_addr)
        self._joint_qvel_addr = (
            dict(joint_qvel_addr) if joint_qvel_addr is not None else dict(joint_qpos_addr)
        )
        self._actuator_index = dict(actuator_index)
        self._grippers_by_joint: dict[str, SimGripperDescription] = {g.joint: g for g in grippers}
        if len(self._grippers_by_joint) != len(list(grippers)):
            duplicates = [g.joint for g in grippers]
            raise ROSConfigError(f"grippers contains duplicate joint names: {duplicates}")
        joint_name_set = {j.name for j in description.joints}
        for g_joint in self._grippers_by_joint:
            if g_joint not in joint_name_set:
                raise ROSConfigError(
                    f"grippers[].joint={g_joint!r} is not present in "
                    f"description.joints (have: {sorted(joint_name_set)})"
                )
        self._keyframe_index = keyframe_index
        self._seed_ctrl_from_qpos = seed_ctrl_from_qpos
        self._settle_steps = settle_steps
        self._gravity_enabled = gravity_enabled
        self._staleness_limit_s = staleness_limit_s

        self._joint_names: list[str] = [j.name for j in description.joints]

        self._connected: bool = False
        self._last_state_time: float = 0.0
        # ADR-0049 / ADR-0034 — the idle-step hand-off timestamp the
        # SimSensorBridge reads to decide when an idle scene may auto-step
        # (``should_idle_step``). ``0`` = never actuated → idle-stepping starts
        # immediately so a parked arm keeps its cameras + joint_state snapshot
        # live. Stamped by every ``send_action``.
        self._last_action_ns: int = 0
        # One-shot guard so a sustained executor stall logs a single starvation
        # WARNING from ``read_state`` instead of spamming one per tick.
        self._starvation_warned: bool = False
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        # Lazily-created offscreen renderers for read_images() (issue #191
        # Phase 3b), keyed by (height, width). One renderer per distinct camera
        # resolution so every camera renders at ITS OWN declared intrinsics
        # rather than a shared max (a 256x256 wrist camera alongside a 640x480
        # overhead must publish 256x256, not 640x480). Created on the first
        # read_images() call so the EGL context binds on the caller's thread
        # (SimSensorBridge's camera timer on the node's single-threaded
        # executor) — EGL contexts are thread-affine.
        self._renderers: dict[tuple[int, int], Any] = {}  # (h, w) -> mujoco.Renderer
        self._render_failed: bool = False
        self._mjcf_cameras: set[str] = set()
        self._render_missing_warned: set[str] = set()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Load the MJCF and prepare an ``MjData`` buffer.

        Raises:
            ROSConfigError: If ``mujoco`` is not installed or the MJCF path is
                invalid.
            ROSRuntimeError: If already connected.
        """
        if self._connected:
            raise ROSRuntimeError(f"MujocoArmHAL('{self.description.name}') is already connected.")
        try:
            import mujoco as mj  # reason: optional sim-only dep
        except ModuleNotFoundError as exc:
            raise ROSConfigError(
                "mujoco is not installed. Install the sim extras with: "
                "just sync --all-packages --group sim"
            ) from exc

        # ADR-0065 — generic camera rig: splice the manifest's declared RGB
        # cameras into the MJCF when they are absent (a bare-arm twin ships
        # none), so deploy sim renders them without a per-robot scene composer.
        # Idempotent: a scene-attached / already-composed MJCF that already has
        # the cameras passes through and we load the original path unchanged.
        mjcf_path = self._mjcf_path
        try:
            from openral_hal._camera_rig import rig_cameras_into_mjcf

            with open(self._mjcf_path) as fh:
                rigged_xml, changed = rig_cameras_into_mjcf(
                    fh.read(), list(self.description.sensors)
                )
            if changed:
                # Write next to the source MJCF so its relative meshdir resolves.
                rig_path = os.path.join(
                    os.path.dirname(self._mjcf_path),
                    f"{self.description.name}_camrig.xml",
                )
                with open(rig_path, "w") as fh:
                    fh.write(rigged_xml)
                mjcf_path = rig_path
        except OSError as exc:
            raise ROSConfigError(f"Could not read MJCF '{self._mjcf_path}': {exc}") from exc

        try:
            self._model = mj.MjModel.from_xml_path(mjcf_path)
        except (OSError, ValueError) as exc:
            raise ROSConfigError(f"Could not load MJCF '{mjcf_path}': {exc}") from exc

        if not self._gravity_enabled:
            self._model.opt.gravity[:] = 0.0
        self._data = mj.MjData(self._model)
        if self._keyframe_index is not None:
            if self._keyframe_index >= self._model.nkey:
                raise ROSConfigError(
                    f"keyframe_index={self._keyframe_index} but the MJCF has "
                    f"only {self._model.nkey} keyframes ({self._mjcf_path})"
                )
            mj.mj_resetDataKeyframe(self._model, self._data, self._keyframe_index)
        mj.mj_forward(self._model, self._data)
        if self._seed_ctrl_from_qpos:
            for name in self._joint_names:
                act_idx = self._actuator_index.get(name)
                qpos_addr = self._joint_qpos_addr.get(name)
                if act_idx is None or qpos_addr is None:
                    continue
                self._data.ctrl[act_idx] = float(self._data.qpos[qpos_addr])

        self._connected = True
        self._last_state_time = time.monotonic()
        log.info(
            "hal.connect",
            robot=self.description.name,
            mjcf=self._mjcf_path,
            n_joints=len(self._joint_names),
        )

    def disconnect(self) -> None:
        """Release the MuJoCo model + renderer.  Idempotent."""
        if not self._connected:
            return
        log.info("hal.disconnect", robot=self.description.name)
        for renderer in self._renderers.values():
            with contextlib.suppress(Exception):
                renderer.close()
        self._renderers.clear()
        self._model = None
        self._data = None
        self._connected = False

    def mujoco_handles(self) -> tuple[Any, Any] | None:
        """Expose the live MuJoCo ``(model, data)`` for the bare-twin arm.

        Mirrors :meth:`SimAttachedHAL.mujoco_handles` so
        :class:`~openral_hal.sim_sensor_bridge.SimSensorBridge`'s offscreen
        cinecam can render a 3rd-person view of a composed-scene arm (openarm /
        so101 / franka bare twins). Returns ``None`` until connected.
        """
        if self._model is None or self._data is None:
            return None
        return self._model, self._data

    def read_images(self) -> dict[str, object]:
        """Render the manifest's RGB cameras off the live MJCF (issue #191 Phase 3b).

        Returns a ``{sensor.name: HxWx3 uint8 ndarray}`` dict — the same contract
        :meth:`SimAttachedHAL.read_images` exposes, which
        :class:`~openral_hal.sim_sensor_bridge.SimSensorBridge` consumes — so a
        composed-scene arm (openarm) publishes ``/openral/cameras/<name>/image``
        through the shared bridge instead of a bespoke node renderer.

        Each RGB :class:`~openral_core.schemas.SensorSpec` is rendered from the
        MJCF camera ``sensor.sim_camera_name or sensor.name`` at **its own**
        ``intrinsics`` resolution (one ``mujoco.Renderer`` is cached per distinct
        ``(height, width)``), so the published frame size always matches the
        sensor's camera model — a 256x256 wrist camera alongside a 640x480
        overhead publishes 256x256, not the larger of the two. A camera absent
        from the MJCF (or a render error) is skipped with a one-shot warning,
        never raising — a missing camera must not crash the actuation path.
        Returns ``{}`` when not connected, when no RGB sensors are declared, or
        after a renderer failure.

        Renderers are created lazily on the first call that needs each resolution
        so their EGL context binds on the caller's thread (the bridge's camera
        timer runs on the node's single-threaded executor; EGL contexts are
        thread-affine).
        """
        if not self._connected or self._model is None or self._data is None:
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
                mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_CAMERA, i)
                for i in range(int(self._model.ncam))
            }

        frames: dict[str, object] = {}
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
            # One renderer per distinct resolution so each camera renders at ITS
            # declared intrinsics (not a shared max) — the published frame size
            # then matches the sensor's camera model for every robot.
            height = s.intrinsics.height if s.intrinsics is not None else 480
            width = s.intrinsics.width if s.intrinsics is not None else 640
            renderer = self._renderers.get((height, width))
            if renderer is None:
                try:
                    renderer = mj.Renderer(self._model, height=height, width=width)
                except Exception as exc:  # reason: GL/MJCF render setup is non-fatal
                    log.warning(
                        "hal.read_images.renderer_failed",
                        robot=self.description.name,
                        error=str(exc),
                    )
                    self._render_failed = True
                    self._renderers.clear()
                    return {}
                self._renderers[(height, width)] = renderer
            try:
                renderer.update_scene(self._data, camera=cam)
                frames[s.name] = renderer.render()
            except Exception as exc:  # reason: a per-tick render hiccup skips one frame
                log.warning(
                    "hal.read_images.render_failed",
                    robot=self.description.name,
                    camera=cam,
                    error=str(exc),
                )
        return frames

    # ── Hot path ───────────────────────────────────────────────────────────────

    def read_state(self) -> JointState:
        """Return the latest joint state in the order of ``description.joints``.

        Raises:
            ROSRuntimeError: If not connected.

        Returns:
            ``JointState`` with positions / velocities for every controllable
            joint in ``description.joints``.  The gripper position (if any) is
            reported in ``[0, 1]``.

        Note:
            The read below returns the live in-process ``MjData`` — always the
            *current* simulator state — so a large gap since the last service is
            never bad data. It means this HAL's publish loop was starved (e.g. a
            slow camera render hogging the single-threaded executor). We surface
            that as a one-shot diagnostic WARNING and refresh the clock, rather
            than latching a fatal ``ROSPerceptionStale``: the previous behaviour
            raised *before* refreshing the clock, so a single transient stall
            bricked the HAL permanently (every subsequent read raised too). Live
            joint feedback from an async source (a real robot) is policed by the
            subscription-based HALs (``ros_control``/``aloha``), not here.
        """
        self._require_connected("read_state")
        age = time.monotonic() - self._last_state_time
        if age > self._staleness_limit_s:
            if not self._starvation_warned:
                self._starvation_warned = True
                log.warning(
                    "hal.read_state.starved",
                    robot=self.description.name,
                    age_s=round(age, 3),
                    limit_s=self._staleness_limit_s,
                    note=(
                        "publish loop starved > staleness_limit_s (likely the "
                        "single-threaded executor blocked on rendering); serving "
                        "live MjData and refreshing the clock — not latching stale"
                    ),
                )
        else:
            # Healthy read — re-arm the one-shot so a *later* starvation episode
            # warns again rather than staying silent after the first.
            self._starvation_warned = False

        assert self._data is not None  # guaranteed by _require_connected
        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []
        for name in self._joint_names:
            gripper = self._grippers_by_joint.get(name)
            if gripper is not None:
                positions.append(self._read_gripper_value(gripper))
                # qvel reporting for a gripper joint — for PASSTHROUGH grippers
                # the qvel index is meaningful (single revolute / prismatic),
                # for normalised grippers it isn't (the public surface is unit-
                # less); use 0.0 there to avoid surprising clients.
                if gripper.read_mode is GripperReadMode.PASSTHROUGH:
                    velocities.append(float(self._data.qvel[gripper.qpos_addrs[0]]))
                else:
                    velocities.append(0.0)
                eff_idx = self._effective_actuator_index_for(gripper, name)
                if eff_idx is not None:
                    efforts.append(float(self._data.actuator_force[eff_idx]))
                else:
                    efforts.append(0.0)
            else:
                qpos_addr = self._joint_qpos_addr[name]
                qvel_addr = self._joint_qvel_addr[name]
                positions.append(float(self._data.qpos[qpos_addr]))
                velocities.append(float(self._data.qvel[qvel_addr]))
                act_idx = self._actuator_index.get(name)
                if act_idx is not None:
                    efforts.append(float(self._data.actuator_force[act_idx]))
                else:
                    efforts.append(0.0)

        self._last_state_time = time.monotonic()
        return JointState(
            name=list(self._joint_names),
            position=positions,
            velocity=velocities,
            effort=efforts,
            stamp_ns=time.time_ns(),
        )

    def send_action(self, action: Action) -> None:
        """Forward the **last** waypoint of *action* to MuJoCo and step.

        Args:
            action: ``Action`` produced by a Skill.  Must declare a
                ``control_mode`` in ``description.capabilities.supported_control_modes``.

        Raises:
            ROSRuntimeError: If not connected.
            ROSConfigError: If the action's control mode is unsupported, or if
                the joint dimensions do not match the robot.
        """
        self._require_connected("send_action")
        self._validate_action(action)

        assert self._data is not None and self._model is not None
        import mujoco as mj  # reason: optional sim-only dep

        last_step = self._last_arm_targets(action)
        self._apply_arm_targets(last_step)
        if self._grippers_by_joint:
            self._apply_gripper_targets(last_step)

        for _ in range(self._settle_steps):
            self._per_step_update(last_step)
            mj.mj_step(self._model, self._data)
        # Refresh the staleness clock so multi-thousand-step settles don't trip
        # a starvation warning on the next ``read_state``, and stamp the
        # idle-step hand-off so the SimSensorBridge yields to this real action.
        self._last_state_time = time.monotonic()
        self._last_action_ns = time.monotonic_ns()

        log.debug(
            "hal.send_action",
            robot=self.description.name,
            control_mode=action.control_mode,
            horizon=action.horizon,
            settle_steps=self._settle_steps,
        )

    # ── Pose reset ─────────────────────────────────────────────────────────────

    def reset_to_pose(self, pose: list[float]) -> None:
        """Snap the MuJoCo ``qpos`` to a specific pose and re-seed ``ctrl``.

        Used by the ``rskill_runner_node`` before the first inference tick
        when the active rSkill's manifest declares a ``starting_pose:``
        teleop home.  Without this, a VLA checkpoint can see an
        out-of-distribution joint state on step 1 and drift joints into
        their mechanical stops within a few seconds.

        Args:
            pose: Joint positions in the order of ``description.joints``
                (units are whatever the underlying MJCF expects —
                radians for revolute joints, metres for prismatic).
                ``len(pose)`` must equal ``len(description.joints)``.

        Raises:
            ROSRuntimeError: If not connected.
            ROSConfigError: If ``pose`` length doesn't match the joint
                count.

        Example:
            >>> # hal.connect()
            >>> # hal.reset_to_pose([0.0] * len(hal.description.joints))
        """
        self._require_connected("reset_to_pose")
        assert self._data is not None and self._model is not None
        if len(pose) != len(self._joint_names):
            raise ROSConfigError(
                f"reset_to_pose: {len(pose)} entries, expected "
                f"{len(self._joint_names)} (description.joints order).",
            )
        import mujoco as mj  # reason: optional sim-only dep

        for name, value in zip(self._joint_names, pose, strict=True):
            qpos_addr = self._joint_qpos_addr.get(name)
            if qpos_addr is None:
                continue
            self._data.qpos[qpos_addr] = float(value)
            qvel_addr = self._joint_qvel_addr.get(name)
            if qvel_addr is not None:
                # Zero qvel too so the snap doesn't carry momentum.
                self._data.qvel[qvel_addr] = 0.0
        mj.mj_forward(self._model, self._data)
        # Re-seed ctrl from qpos so position actuators hold the new pose
        # on the next ``mj_step``.
        for name in self._joint_names:
            act_idx = self._actuator_index.get(name)
            qpos_addr = self._joint_qpos_addr.get(name)
            if act_idx is None or qpos_addr is None:
                continue
            self._data.ctrl[act_idx] = float(self._data.qpos[qpos_addr])
        self._last_state_time = time.monotonic()
        log.info(
            "hal.reset_to_pose",
            robot=self.description.name,
            n_joints=len(self._joint_names),
        )

    # ── Safety ─────────────────────────────────────────────────────────────────

    def estop(self) -> None:
        """Trigger an emergency stop: zero ``ctrl`` and raise.

        Raises:
            ROSEStopRequested: Always.
        """
        log.critical("hal.estop", robot=self.description.name)
        if self._data is not None:
            self._data.ctrl[:] = 0.0
        self._model = None
        self._data = None
        self._connected = False
        raise ROSEStopRequested(f"Emergency stop triggered on robot '{self.description.name}'.")

    @property
    def last_action_ns(self) -> int:
        """``time.monotonic_ns()`` of the last :meth:`send_action` (``0`` if never).

        The SimSensorBridge reads this (``should_idle_step``) to yield the idle
        stepper to a recently-commanded skill. Mirrors the same surface on
        :class:`~openral_hal.sim_attached.SimAttachedHAL`.
        """
        return self._last_action_ns

    def idle_step(self) -> bool:
        """Advance the sim one HOLD tick so cameras + state stay live while idle.

        Gives a bare ``MujocoArmHAL`` the ADR-0034 (idle cameras) + ADR-0049
        (off-executor joint_state via ``ProprioSnapshot``) treatment that the
        lifecycle node gates on a *callable* ``idle_step``. ``ctrl`` is left
        untouched — it already holds the last commanded / seeded pose, so a HOLD
        step does not move the arm; it only re-integrates physics and refreshes
        the staleness clock.

        SAFETY: this is defined only on the sim-only ``MujocoArmHAL`` hierarchy
        (no real HAL inherits it), and it is a no-op returning ``False`` once the
        HAL has e-stopped (``estop()`` disconnects), so it can never autonomously
        drive an e-stopped robot. ``False`` also signals the bridge to leave the
        cached frame frozen.

        Returns:
            ``True`` if the sim advanced, ``False`` if the HAL is not connected
            (e.g. after :meth:`estop`).
        """
        if not self._connected or self._data is None or self._model is None:
            return False
        import mujoco as mj  # reason: optional sim-only dep

        mj.mj_step(self._model, self._data)
        self._last_state_time = time.monotonic()
        return True

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _require_connected(self, operation: str) -> None:
        if not self._connected or self._data is None or self._model is None:
            raise ROSRuntimeError(
                f"{type(self).__name__}.{operation}() called while not connected. "
                "Call connect() first."
            )

    def _validate_action(self, action: Action) -> None:
        supported = self.description.capabilities.supported_control_modes
        if supported and action.control_mode not in supported:
            raise ROSConfigError(
                f"Action control_mode '{action.control_mode}' is not in "
                f"supported_control_modes {supported} for robot "
                f"'{self.description.name}'."
            )
        self._require_control_mode(action, ControlMode.JOINT_POSITION)
        if not action.joint_targets or not action.joint_targets[0]:
            raise ROSConfigError("action.joint_targets[0] is required.")
        self._validate_action_dims(action, len(self._joint_names))

    def _last_arm_targets(self, action: Action) -> list[float]:
        assert action.joint_targets is not None
        return list(action.joint_targets[-1])

    def _apply_arm_targets(self, targets: list[float]) -> None:
        assert self._data is not None
        for name, value in zip(self._joint_names, targets, strict=True):
            if name in self._grippers_by_joint:
                continue
            act_idx = self._actuator_index.get(name)
            if act_idx is None:
                continue
            self._data.ctrl[act_idx] = float(value)

    def _per_step_update(self, targets: list[float]) -> None:
        """Hook invoked before every ``mj_step`` inside the settle loop.

        Default: no-op — the base class assumes position-controlled
        actuators (with an internal PD law) where ``ctrl`` set once
        outside the loop is enough.  Subclasses that drive torque-mode
        actuators (e.g. :class:`openral_hal.H1MujocoHAL` for the
        Unitree H1's ``motor`` actuators) override this to recompute
        the torque each step from current ``qpos`` / ``qvel``.

        Args:
            targets: The last-waypoint joint targets handed to
                :meth:`_apply_arm_targets`, in the same order as
                ``self._joint_names``.
        """
        del targets  # base no-op; param kept so the override signature lines up

    def _apply_gripper_targets(self, last_step: list[float]) -> None:
        """Write the per-gripper command to ``ctrl`` (and mirror, if any).

        For ``NORMALISED`` write mode the input is clipped to ``[0, 1]``
        and mapped affinely to ``gripper.ctrl_range``.  For
        ``PASSTHROUGH`` mode the input is written verbatim; MuJoCo's
        actuator ``ctrlrange`` does the clipping at simulation time.
        """
        assert self._data is not None
        for joint_name, gripper in self._grippers_by_joint.items():
            joint_idx = self._joint_names.index(joint_name)
            command = float(last_step[joint_idx])
            if gripper.write_mode is GripperWriteMode.NORMALISED:
                normalised = max(0.0, min(1.0, command))
                low, high = gripper.ctrl_range
                raw = low + normalised * (high - low)
            else:  # PASSTHROUGH
                raw = command
            primary_idx = self._effective_actuator_index_for(gripper, joint_name)
            if primary_idx is None:
                raise ROSConfigError(
                    f"Gripper {joint_name!r} has no actuator index — set "
                    "sim.grippers[].actuator_index or include the joint in "
                    "sim.actuator_index."
                )
            self._data.ctrl[primary_idx] = raw
            if gripper.mirror_actuator_index is not None:
                self._data.ctrl[gripper.mirror_actuator_index] = -raw

    def _read_gripper_value(self, gripper: SimGripperDescription) -> float:
        """Report the gripper's public position according to ``gripper.read_mode``."""
        assert self._data is not None
        if gripper.read_mode is GripperReadMode.PASSTHROUGH:
            return float(self._data.qpos[gripper.qpos_addrs[0]])
        if gripper.read_mode is GripperReadMode.AFFINE_LOW_HIGH:
            low, high = gripper.ctrl_range
            span = high - low
            if span == 0.0:
                return 0.0
            qpos = float(self._data.qpos[gripper.qpos_addrs[0]])
            return max(0.0, min(1.0, (qpos - low) / span))
        # SUM_OVER_SCALE (default)
        total = sum(float(self._data.qpos[i]) for i in gripper.qpos_addrs)
        if gripper.qpos_scale == 0.0:
            return 0.0
        return max(0.0, min(1.0, total / gripper.qpos_scale))

    def _effective_actuator_index_for(
        self, gripper: SimGripperDescription, joint_name: str
    ) -> int | None:
        """Resolve the MJCF actuator index that receives this gripper's command.

        ``gripper.actuator_index`` (if set) wins; otherwise fall back to
        the joint-wide ``actuator_index`` map.
        """
        if gripper.actuator_index is not None:
            return gripper.actuator_index
        return self._actuator_index.get(joint_name)

    # ── Manifest-driven constructor ────────────────────────────────────────────

    @staticmethod
    def _sim_kwargs_for(
        description: RobotDescription,
        *,
        settle_steps: int | None = None,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
        mjcf_path_override: str | None = None,
    ) -> _MujocoArmInitKwargs:
        """Translate ``description.sim`` to the kwarg dict accepted by ``__init__``.

        This is the single seam where the declarative manifest is mapped to
        the imperative constructor.  Both :meth:`from_description` /
        :meth:`_init_from_description` and any caller that wants to
        post-process the kwargs delegate here.  The
        :class:`_MujocoArmInitKwargs` return type lets ``**kwargs``
        unpack into the constructor under ``mypy --strict`` without an
        ``arg-type`` ignore.

        Raises:
            ROSConfigError: If ``description.sim`` is None, or if
                ``sim.joint_qpos_addr`` references an unknown joint, or if
                ``floating_base=True`` is paired with only one of
                ``joint_qpos_addr`` / ``joint_qvel_addr`` set.
        """
        if description.sim is None:
            raise ROSConfigError(
                f"RobotDescription '{description.name}' has no `sim` block; "
                "cannot construct a MujocoArmHAL from it. Add a `sim:` section "
                "to the manifest (see docs/adr/0023-data-driven-mujoco-hal.md)."
            )
        sim = description.sim
        joint_names = [j.name for j in description.joints]

        qpos_offset = 7 if sim.floating_base else 0
        qvel_offset = 6 if sim.floating_base else 0

        default_qpos = {name: qpos_offset + i for i, name in enumerate(joint_names)}
        default_qvel = {name: qvel_offset + i for i, name in enumerate(joint_names)}
        default_actuator = {name: i for i, name in enumerate(joint_names)}

        if sim.floating_base and ((sim.joint_qpos_addr is None) ^ (sim.joint_qvel_addr is None)):
            raise ROSConfigError(
                f"RobotDescription '{description.name}': floating_base=True "
                "requires sim.joint_qpos_addr and sim.joint_qvel_addr to be "
                "set together (their offsets differ — 7 vs 6 — so omitting one "
                "is almost always a bug)."
            )

        joint_qpos_addr = sim.joint_qpos_addr or default_qpos
        joint_qvel_addr = sim.joint_qvel_addr or (
            joint_qpos_addr
            if sim.joint_qvel_addr is None and not sim.floating_base
            else default_qvel
        )
        actuator_index = sim.actuator_index or default_actuator

        for name in joint_qpos_addr:
            if name not in joint_names:
                raise ROSConfigError(
                    f"sim.joint_qpos_addr references unknown joint {name!r} (known: {joint_names})"
                )

        return _MujocoArmInitKwargs(
            mjcf_path=mjcf_path_override or _resolve_mjcf_path(description),
            joint_qpos_addr=joint_qpos_addr,
            joint_qvel_addr=joint_qvel_addr,
            actuator_index=actuator_index,
            grippers=list(sim.grippers),
            keyframe_index=sim.keyframe_index,
            seed_ctrl_from_qpos=sim.seed_ctrl_from_qpos,
            settle_steps=(sim.settle_steps_default if settle_steps is None else settle_steps),
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )

    @classmethod
    def from_description(
        cls,
        description: RobotDescription,
        *,
        settle_steps: int | None = None,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
        mjcf_path_override: str | None = None,
    ) -> MujocoArmHAL:
        """Build a :class:`MujocoArmHAL` purely from ``description.sim``.

        This is the **recommended** entry point for any code that selects a
        robot at runtime — it removes every hardcoded constant from the HAL
        Python files. The legacy keyword-argument constructor stays
        available for tests and for one-off overrides.

        The default joint→qpos/qvel/actuator mappings are 1:1 with
        ``description.joints`` order, offset by 7/6 if
        ``description.sim.floating_base`` is True (humanoids).  Explicit
        per-joint overrides in ``description.sim`` win when present.

        Args:
            description: A :class:`RobotDescription` whose ``sim`` field is
                populated.
            settle_steps: Optional override for the number of MuJoCo
                physics steps per :meth:`send_action`.  Defaults to
                ``description.sim.settle_steps_default``.
            gravity_enabled: Forwarded to :meth:`__init__`.
            staleness_limit_s: Forwarded to :meth:`__init__`.
            mjcf_path_override: Optional absolute path that wins over
                ``description.assets.mjcf`` — useful in tests that ship a
                stripped MJCF.

        Returns:
            An un-connected :class:`MujocoArmHAL`.

        Raises:
            ROSConfigError: If ``description.sim`` is None, or if its
                joint-index overrides reference unknown joint names, or if
                the MJCF URI cannot be resolved.

        Example:
            >>> from openral_core import RobotDescription
            >>> # desc = RobotDescription.from_yaml("robots/ur5e/robot.yaml")
            >>> # hal = MujocoArmHAL.from_description(desc)
            >>> # hal.connect()
        """
        kwargs = cls._sim_kwargs_for(
            description,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
            mjcf_path_override=mjcf_path_override,
        )
        return cls(description, **kwargs)

    def _init_from_description(
        self,
        description: RobotDescription,
        *,
        mjcf_path: str | None = None,
        settle_steps: int | None = None,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Initialise *self* from a fully-populated ``description.sim`` block.

        This is the seam every thin per-robot subclass (UR5e/UR10e, Franka,
        ALOHA, OpenArm, Rizon4, G1, H1, SO-100) uses to drop the
        boilerplate ``super().__init__(DESC, **MujocoArmHAL._sim_kwargs_for(DESC, ...))``
        dance.  Subclasses keep their typed ``__init__`` signature (so IDEs
        still surface ``mjcf_path``/``settle_steps``/``gravity_enabled``/
        ``staleness_limit_s`` as the four user-tunable knobs) and forward
        straight to here.

        ``settle_steps=None`` (the default) lets
        :meth:`_sim_kwargs_for` substitute
        ``description.sim.settle_steps_default``.

        Args:
            description: A :class:`RobotDescription` whose ``sim`` field is
                populated.
            mjcf_path: Optional override for the MJCF file path; wins over
                ``description.assets.mjcf`` when set.
            settle_steps: Optional override for the number of MuJoCo
                physics steps per :meth:`send_action`.
            gravity_enabled: When ``False``, gravity is zeroed at
                :meth:`connect` time.
            staleness_limit_s: Age above which :meth:`read_state` emits a
                one-shot starvation WARNING (it still returns live state).

        Raises:
            ROSConfigError: Propagated from :meth:`_sim_kwargs_for` /
                :meth:`__init__` for malformed ``description.sim``.
        """
        kwargs = MujocoArmHAL._sim_kwargs_for(
            description,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
            mjcf_path_override=mjcf_path,
        )
        MujocoArmHAL.__init__(self, description, **kwargs)
