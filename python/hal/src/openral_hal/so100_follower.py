"""SO100FollowerHAL — HAL adapter wrapping lerobot's SO-100 follower arm.

This adapter bridges ``lerobot.robots.so_follower.SO100Follower`` to the
``openral_hal.HAL`` Protocol.  ``lerobot`` is an **optional** dependency;
the module imports cleanly without it and only raises ``ROSConfigError`` at
``connect()`` time when the package is missing.

Unit angles
-----------
lerobot's ``SOFollower`` (with ``use_degrees=True``, its default) exposes joint
positions in **degrees**.  openral ``JointState`` and ``Action`` use
**radians** for revolute joints per the ``JointSpec`` contract.  This adapter
converts transparently:

- ``get_observation()`` degrees → ``read_state()`` radians
- ``send_action(Action)`` radians → ``send_action(dict)`` degrees

The gripper joint is kept in [0, 100] (lerobot's normalised range) on the
lerobot side and mapped to [0, 1] in the ``Action`` gripper field.

Example:
    >>> from openral_hal.so100_follower import SO100FollowerHAL, SO100_DESCRIPTION
    >>> hal = SO100FollowerHAL(port="/dev/ttyUSB0")
    >>> hal.description.name
    'so100_follower'
    >>> hal.description.embodiment_kind.value
    'manipulator'
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import structlog
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSRuntimeError,
)
from openral_core.schemas import (
    Action,
    AssetRefs,
    ControlMode,
    EmbodimentKind,
    EndEffectorSpec,
    GripperReadMode,
    HalEntrypoints,
    JointSpec,
    JointState,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SimDescription,
    SimGripperDescription,
    UrdfAsset,
)

from openral_hal._base import HALBase
from openral_hal._sensor_wiring import with_sensors

if TYPE_CHECKING:
    # Import only for type checking — lerobot is optional at runtime.
    from lerobot.robots import Robot as _LeRobotRobot
    from lerobot.robots.so_follower import SOFollowerRobotConfig as _SOFollowerRobotConfig

__all__ = ["SO100_DESCRIPTION", "SO100FollowerHAL", "so100_with_sensors"]

log = structlog.get_logger(__name__)

# ── Canonical joint order (matches lerobot's bus motor dict order) ─────────────

_SO100_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# ── reset_to_pose ramp tuning ─────────────────────────────────────────────────
# The real-arm starting-pose ramp (see ``reset_to_pose``). 0.5 rad/s is well
# under teleop speeds on these servos; the [min, max] duration clamp keeps
# micro-moves from finishing instantly and giant moves from taking forever.
# ponytail: module constants, promote to HAL params if a robot ever needs
# per-deploy tuning.
_RESET_MAX_RAD_S = 0.5
_RESET_MIN_S = 1.0
_RESET_MAX_S = 6.0
_RESET_STEP_HZ = 30.0
# Below this max joint delta (rad) the arm counts as already at the target
# and the ramp is skipped entirely.
_RESET_DEADBAND_RAD = 1e-3

# ── Canonical RobotDescription for the SO-100 follower arm ────────────────────

SO100_DESCRIPTION = RobotDescription(
    name="so100_follower",
    embodiment_kind=EmbodimentKind.MANIPULATOR,
    base_frame="so100_base_link",
    joints=[
        JointSpec(
            name="shoulder_pan",
            joint_type=JointType.REVOLUTE,
            parent_link="so100_base_link",
            child_link="so100_shoulder_pan_link",
            position_limits=(-math.pi, math.pi),
            velocity_limit=math.pi,
            effort_limit=1.5,
            has_torque_sensor=False,
            actuator_kind="servo",
        ),
        JointSpec(
            name="shoulder_lift",
            joint_type=JointType.REVOLUTE,
            parent_link="so100_shoulder_pan_link",
            child_link="so100_shoulder_lift_link",
            position_limits=(-math.pi, math.pi),
            velocity_limit=math.pi,
            effort_limit=1.5,
            has_torque_sensor=False,
            actuator_kind="servo",
        ),
        JointSpec(
            name="elbow_flex",
            joint_type=JointType.REVOLUTE,
            parent_link="so100_shoulder_lift_link",
            child_link="so100_elbow_link",
            position_limits=(-math.pi, math.pi),
            velocity_limit=math.pi,
            effort_limit=1.5,
            has_torque_sensor=False,
            actuator_kind="servo",
        ),
        JointSpec(
            name="wrist_flex",
            joint_type=JointType.REVOLUTE,
            parent_link="so100_elbow_link",
            child_link="so100_wrist_flex_link",
            position_limits=(-math.pi, math.pi),
            velocity_limit=math.pi,
            effort_limit=0.5,
            has_torque_sensor=False,
            actuator_kind="servo",
        ),
        JointSpec(
            name="wrist_roll",
            joint_type=JointType.REVOLUTE,
            parent_link="so100_wrist_flex_link",
            child_link="so100_wrist_roll_link",
            position_limits=(-math.pi, math.pi),
            velocity_limit=math.pi,
            effort_limit=0.5,
            has_torque_sensor=False,
            actuator_kind="servo",
        ),
        JointSpec(
            name="gripper",
            joint_type=JointType.PRISMATIC,
            parent_link="so100_wrist_roll_link",
            child_link="so100_gripper_link",
            position_limits=(0.0, 1.0),  # [0=closed, 1=open] normalised
            velocity_limit=1.0,
            effort_limit=5.0,
            has_torque_sensor=False,
            actuator_kind="servo",
        ),
    ],
    end_effectors=[
        EndEffectorSpec(
            name="gripper",
            kind="parallel_gripper",
            n_dof=1,
            max_grip_force_n=5.0,
        )
    ],
    capabilities=RobotCapabilities(
        supported_control_modes=[ControlMode.JOINT_POSITION],
        embodiment_tags=["so100_follower"],
    ),
    safety=SafetyEnvelope(
        max_ee_speed_m_s=0.3,
        max_joint_speed_factor=0.5,
        deadman_required=False,  # tabletop arm; deadman handled by USB watchdog
    ),
    sdk_kind="open",
    hal=HalEntrypoints(sim=None, real="openral_hal.so100_follower:SO100FollowerHAL"),
    assets=AssetRefs(
        urdf=UrdfAsset(ref="rd:so_arm100_description"),
        mjcf="rd:so_arm100_mj_description",
    ),
    sim=SimDescription(
        grippers=[
            SimGripperDescription(
                joint="gripper",
                ctrl_range=(-0.174, 1.75),
                qpos_addrs=(5,),
                # qpos_scale unused by AFFINE_LOW_HIGH, kept to satisfy
                # the manifest invariant.
                qpos_scale=1.924,
                read_mode=GripperReadMode.AFFINE_LOW_HIGH,
            ),
        ],
    ),
)


# ── Sensor-wired description factory (issue #23) ──────────────────────────────


def so100_with_sensors(
    catalog_ids: list[str] | None = None,
) -> RobotDescription:
    """Return a copy of :data:`SO100_DESCRIPTION` with catalog sensors attached.

    The reference LeRobot SO-100 setup uses a single Logitech C920 scene
    camera; pass ``None`` to get that default, or pass an explicit list of
    catalog ids (``"logitech/c920"``, ``"intel/realsense_d405"``, …) to
    override.  Pass ``[]`` to get an empty sensor loadout.

    Args:
        catalog_ids: Catalog ids to resolve, or ``None`` for the LeRobot
            reference loadout (``["logitech/c920"]``).

    Returns:
        A new :class:`RobotDescription` with ``sensors`` / ``sensor_bundles``
        populated.

    Example:
        >>> desc = so100_with_sensors()
        >>> desc.sensors[0].vendor
        'Logitech'
    """
    if catalog_ids is None:
        catalog_ids = ["logitech/c920"]
    return with_sensors(SO100_DESCRIPTION, catalog_ids)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * math.pi / 180.0


def _joint_values_to_lerobot(step: Sequence[float]) -> dict[str, float]:
    """Manifest-order joint values → lerobot's ``{"<joint>.pos": float}`` dict.

    THE single unit conversion on the SO-100/101 actuation path — radians →
    degrees for the five arm joints, ``[0, 1]`` → ``[0, 100]`` for the gripper
    channel. Both ``send_action`` (via ``_action_to_lerobot``) and the
    ``reset_to_pose`` ramp route through here so a calibration offset or
    range change can never apply to one path and not the other. Callers
    validate ``len(step) == len(_SO100_JOINT_NAMES)``.
    """
    out: dict[str, float] = {}
    for i, name in enumerate(_SO100_JOINT_NAMES):
        val = step[i]
        if name == "gripper":
            out[f"{name}.pos"] = val * 100.0  # [0, 1] → [0, 100]
        else:
            out[f"{name}.pos"] = _rad_to_deg(val)
    return out


def _rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * 180.0 / math.pi


# ── SO100FollowerHAL ────────────────────────────────────────────────────────────────


class SO100FollowerHAL(HALBase):
    """HAL adapter wrapping lerobot's SO-100 follower arm.

    The adapter is instantiated without a live robot connection; call
    ``connect()`` to open the USB serial port.  No ``lerobot`` import happens
    at module load time so the class can be used in environments where lerobot
    is not installed (e.g., CI type-checking runs).

    For testing without hardware, pass an already-constructed lerobot
    ``Robot`` instance via the ``robot`` keyword argument.  The injected robot
    is used directly and no serial port is opened:

        >>> from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
        >>> twin = SO100DigitalTwin(SO100DigitalTwinConfig())
        >>> hal = SO100FollowerHAL(robot=twin)
        >>> hal.connect()
        >>> hal.description.name
        'so100_follower'

    Args:
        port: USB serial port, e.g. ``"/dev/ttyUSB0"``.  Ignored when
            ``robot`` is provided.  Defaults to ``"/dev/ttyUSB0"`` on
            Linux / ``"/dev/cu.usbserial-0001"`` on macOS.
        calibrate_on_connect: If ``True`` (default ``False``), run lerobot's
            interactive calibration wizard at ``connect()`` time.  Set to
            ``False`` for automated / HIL use; the stored calibration file is
            used instead.
        max_relative_target: Per-joint or scalar cap on goal-position deltas
            forwarded to the motor bus.  ``None`` means no capping.  Ignored
            when ``robot`` is provided.
        staleness_limit_s: How old (seconds) a ``read_state()`` timestamp may
            be before ``ROSPerceptionStale`` is raised.  Defaults to ``0.5``.
        robot: Optional pre-constructed lerobot ``Robot`` instance.  When
            provided, ``connect()`` calls ``robot.connect()`` directly instead
            of constructing a ``SO100Follower`` and opening the serial port.
            Intended for testing via ``SO100DigitalTwin``.

    Raises:
        ROSConfigError: At ``connect()`` time if ``lerobot`` is not installed
            and no ``robot`` was injected.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        *,
        calibrate_on_connect: bool = False,
        id: str | None = None,  # reason: mirrors lerobot RobotConfig.id verbatim
        calibration_dir: str | None = None,  # reason: mirrors lerobot RobotConfig.calibration_dir
        max_relative_target: float | dict[str, float] | None = None,
        staleness_limit_s: float = 0.5,
        robot: _LeRobotRobot | None = None,
    ) -> None:
        """Initialise the adapter; does not open any connection yet.

        ``id`` is lerobot's calibration identity: the stored calibration file
        resolves to ``<calibration_dir>/<id>.json``. ``calibration_dir``
        defaults (in lerobot) to ``~/.cache/huggingface/lerobot/calibration/
        robots/so_follower/``; pass it to load a calibration committed
        alongside the deploy config instead — so the deploy owns its own
        calibration and never silently picks up a stale/other cached file
        (two ``so_follower/*.json`` for one arm is a real footgun). Without an
        ``id`` (and with ``calibrate_on_connect=False``) the bus connects but
        every ``get_observation()`` raises ``has no calibration registered`` —
        pass the same ``id`` the arm was calibrated with
        (``lerobot-calibrate --robot.id=<id>``), e.g. via the deploy config's
        ``hal.params.id`` + ``hal.params.calibration_dir``.
        """
        self.description: RobotDescription = SO100_DESCRIPTION
        self._port = port
        self._calibrate_on_connect = calibrate_on_connect
        self._id = id
        self._calibration_dir = calibration_dir
        self._max_relative_target = max_relative_target
        self._staleness_limit_s = staleness_limit_s
        self._injected_robot: _LeRobotRobot | None = robot

        self._robot: _LeRobotRobot | None = None
        self._connected: bool = False
        self._last_obs_time: float = 0.0
        self._last_obs: dict[str, float] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the USB serial connection to the SO-100 arm.

        If a ``robot`` was passed at construction time, that instance is used
        directly (no serial port is opened).  Otherwise, a ``SO100Follower``
        is constructed from ``port`` and the lerobot package is imported lazily.

        Raises:
            ROSConfigError: If ``lerobot`` is not installed and no robot was
                injected, or if already connected.
            ROSRuntimeError: If the underlying serial open fails.
        """
        if self._connected:
            raise ROSRuntimeError(f"SO100FollowerHAL(port={self._port!r}) is already connected.")

        if self._injected_robot is not None:
            # Use the pre-constructed robot (e.g. SO100DigitalTwin for testing).
            try:
                self._injected_robot.connect(calibrate=self._calibrate_on_connect)
            except Exception as exc:
                raise ROSRuntimeError(f"Injected robot connect() failed: {exc}") from exc
            self._robot = self._injected_robot
        else:
            try:
                from lerobot.robots.so_follower import (  # noqa: PLC0415
                    SO100Follower,
                    SOFollowerRobotConfig,
                )
            except ModuleNotFoundError as exc:
                raise ROSConfigError(
                    "lerobot is not installed. Install it with: "
                    "uv add lerobot --package openral-hal"
                ) from exc

            from pathlib import Path  # noqa: PLC0415

            cfg: _SOFollowerRobotConfig = SOFollowerRobotConfig(
                port=self._port,
                id=self._id,  # calibration identity → <calibration_dir>/<id>.json
                # None → lerobot's default HF cache dir; a path → the deploy's
                # committed calibration (see __init__).
                calibration_dir=Path(self._calibration_dir) if self._calibration_dir else None,
                max_relative_target=self._max_relative_target,
                use_degrees=True,  # adapter converts degrees ↔ radians
            )
            try:
                robot = SO100Follower(cfg)
            except ModuleNotFoundError as exc:
                # lerobot's FeetechMotorsBus pulls in `scservo_sdk` at __init__
                # time; surface a typed config error instead of leaking the raw
                # ImportError traceback.
                raise ROSConfigError(
                    f"SO-100 driver dependency missing ({exc.name!r}). "
                    "Install with: uv add scservo_sdk --package openral-hal"
                ) from exc
            if not os.path.exists(self._port):
                raise ROSConfigError(
                    f"SO-100 serial port {self._port!r} does not exist. "
                    "Connect the arm via USB, or pass --port /dev/ttyUSBn."
                )
            try:
                robot.connect(calibrate=self._calibrate_on_connect)
            except Exception as exc:
                raise ROSRuntimeError(
                    f"Failed to connect to SO-100 on {self._port!r}: {exc}"
                ) from exc
            self._robot = robot

        self._preflight_servo_ping(self._robot)
        self._connected = True
        self._last_obs_time = time.monotonic()
        log.info("hal.connect", robot="so100_follower", port=self._port)

    def _preflight_servo_ping(self, robot: _LeRobotRobot) -> None:
        """Ping every servo so a dark/half-powered bus fails LOUD at connect.

        lerobot's ``send_action`` is fire-and-forget — the write expects no
        status packet — so a bus that drops every *read* still ``connect()``s
        cleanly and then silently freezes mid-run: the policy commands big
        moves while ``read_state`` returns ``"There is no status packet!"`` on
        every id and the observed joint state never changes (exactly the SO-101
        symptom we hit — 1500+ read failures, frozen ``state_to_policy``, no
        motion). Ping each motor once here and raise a typed
        :class:`ROSConfigError` naming the unresponsive ids, so a comms/power
        fault surfaces at bring-up instead of as a mystery freeze under load.

        No-op when the robot exposes no pingable ``bus`` (an injected
        ``SO100DigitalTwin`` in tests) — there is no real servo to verify.
        """
        bus = getattr(robot, "bus", None)
        ping = getattr(bus, "ping", None)
        if not callable(ping):
            return
        dark: list[str] = []
        for name in _SO100_JOINT_NAMES:
            try:
                if ping(name, num_retry=2) is None:
                    dark.append(name)
            except Exception:  # reason: any comms error = that servo is dark
                dark.append(name)
        if dark:
            raise ROSConfigError(
                f"SO-100 pre-flight servo ping failed for {dark} on {self._port!r}: "
                "the motor bus is not answering. Check the arm's 12 V power (a "
                "USB-only ~5 V feed browns out the servos), the USB-serial "
                "adapter/cable, and that no other process holds the port. "
                "send_action is fire-and-forget, so without this pre-flight the "
                "arm would silently freeze mid-run instead of failing here."
            )

    def disconnect(self) -> None:
        """Close the USB connection, disabling motor torque.  Idempotent."""
        if not self._connected:
            return
        if self._robot is not None:
            with contextlib.suppress(Exception):
                self._robot.disconnect()
        self._robot = None
        self._connected = False
        log.info("hal.disconnect", robot="so100_follower", port=self._port)

    # ── Hot path ───────────────────────────────────────────────────────────────

    def read_state(self) -> JointState:
        """Return the latest joint state in radians.

        Raises:
            ROSRuntimeError: If not connected.
            ROSPerceptionStale: If the last observation is too old.

        Returns:
            ``JointState`` with positions in radians (gripper in [0, 1]).
        """
        self._require_connected("read_state")

        assert self._robot is not None  # guaranteed by _require_connected
        try:
            obs: dict[str, object] = self._robot.get_observation()
        except Exception as exc:
            raise ROSRuntimeError(f"get_observation() failed: {exc}") from exc

        self._last_obs_time = time.monotonic()
        self._last_obs = {k: float(v) for k, v in obs.items() if isinstance(v, (int, float))}

        positions = self._obs_to_positions(self._last_obs)
        return JointState(
            name=_SO100_JOINT_NAMES,
            position=positions,
            stamp_ns=time.time_ns(),
        )

    def send_action(self, action: Action) -> None:
        """Forward one action step to the SO-100 motor bus.

        Only the **first step** of the action chunk is sent; lerobot handles
        single-step commands.  Action chunks are executed by calling
        ``send_action`` repeatedly in the skill executor's async loop.

        Args:
            action: The ``Action`` produced by a Skill.  Must use
                ``ControlMode.JOINT_POSITION`` with ``joint_targets`` set.

        Raises:
            ROSRuntimeError: If not connected.
            ROSConfigError: If the action cannot be converted.
        """
        self._require_connected("send_action")
        lerobot_action = self._action_to_lerobot(action)
        assert self._robot is not None
        try:
            self._robot.send_action(lerobot_action)
        except Exception as exc:
            raise ROSRuntimeError(f"send_action() failed: {exc}") from exc

    def reset_to_pose(self, pose: list[float]) -> None:
        """Ramp the real arm to ``pose`` with a slow linear interpolation.

        The real-hardware counterpart of the sim arms' ``qpos`` snap
        (``_mujoco_arm.reset_to_pose``): the ``rskill_runner_node`` calls the
        HAL's ``/openral/<robot>/reset_to_pose`` service before the first
        inference tick so the VLA starts from its manifest ``starting_pose``
        instead of wherever the arm was left. A snap is fine for a simulator;
        on real Feetech servos an instantaneous position command is a violent
        full-speed move, so this interpolates current → target at
        ``_RESET_STEP_HZ`` with the per-joint speed capped by
        ``_RESET_MAX_RAD_S`` (duration clamped to
        [``_RESET_MIN_S``, ``_RESET_MAX_S``]).

        Args:
            pose: Target joint positions in manifest order — radians for the
                five arm joints, ``[0, 1]`` for the gripper channel (the same
                contract as ``read_state`` / ``send_action``).

        Raises:
            ROSRuntimeError: If not connected (or a bus write fails mid-ramp).
            ROSConfigError: If ``pose`` length doesn't match the joint count.

        Example:
            >>> # hal.connect()
            >>> # hal.reset_to_pose([-0.115, -1.79, 1.57, 1.03, 0.10, 0.02])
        """
        self._require_connected("reset_to_pose")
        n = len(_SO100_JOINT_NAMES)
        if len(pose) != n:
            raise ROSConfigError(
                f"reset_to_pose expects {n} values (SO-100 joint order); got {len(pose)}."
            )

        current = self.read_state().position
        deltas = [t - c for t, c in zip(pose, current, strict=True)]
        max_delta = max(abs(d) for d in deltas)
        if max_delta < _RESET_DEADBAND_RAD:
            return  # already there — no motion

        # ponytail: fixed-rate linear ramp; per-joint trapezoids if anyone
        # ever needs smoother-than-gentle. The gripper channel is [0, 1] not
        # radians, so the shared speed cap treats a full gripper stroke like
        # a 1-rad joint move — conservative, never fast.
        duration_s = min(max(max_delta / _RESET_MAX_RAD_S, _RESET_MIN_S), _RESET_MAX_S)
        steps = max(int(duration_s * _RESET_STEP_HZ), 1)
        log.info(
            "hal.reset_to_pose.ramp",
            robot="so100_follower",
            max_delta_rad=round(max_delta, 3),
            duration_s=round(duration_s, 2),
            steps=steps,
        )
        assert self._robot is not None  # guaranteed by _require_connected
        for i in range(1, steps + 1):
            alpha = i / steps
            waypoint = [c + d * alpha for c, d in zip(current, deltas, strict=True)]
            try:
                self._robot.send_action(_joint_values_to_lerobot(waypoint))
            except Exception as exc:
                raise ROSRuntimeError(f"reset_to_pose send_action failed: {exc}") from exc
            time.sleep(1.0 / _RESET_STEP_HZ)

    def estop(self) -> None:
        """Trigger an emergency stop: disconnect motors then raise.

        Raises:
            ROSEStopRequested: Always.
        """
        log.critical("hal.estop", robot="so100_follower", port=self._port)
        if self._robot is not None and self._connected:
            with contextlib.suppress(Exception):
                self._robot.disconnect()
        self._robot = None
        self._connected = False
        raise ROSEStopRequested(f"Emergency stop triggered on SO-100 at port '{self._port}'.")

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _obs_to_positions(obs: dict[str, float]) -> list[float]:
        """Convert a lerobot observation dict to an ordered position list.

        Non-gripper joints are converted from degrees to radians.
        The gripper value (already 0-100) is normalised to [0, 1].
        """
        result: list[float] = []
        for name in _SO100_JOINT_NAMES:
            val = obs.get(f"{name}.pos", 0.0)
            if name == "gripper":
                result.append(val / 100.0)
            else:
                result.append(_deg_to_rad(val))
        return result

    def _action_to_lerobot(self, action: Action) -> dict[str, float]:
        """Convert an ``Action`` to lerobot's ``{"<joint>.pos": float}`` dict.

        Only ``JOINT_POSITION`` mode is supported.  Uses the first step of the
        chunk (``joint_targets[0]``).

        Raises:
            ROSConfigError: If mode is unsupported or targets are missing.
        """
        if action.control_mode != ControlMode.JOINT_POSITION:
            raise ROSConfigError(
                f"SO100FollowerHAL only supports JOINT_POSITION; got {action.control_mode!r}."
            )
        if not action.joint_targets or not action.joint_targets[0]:
            raise ROSConfigError("action.joint_targets[0] is required for SO100FollowerHAL.")

        step = action.joint_targets[0]
        n = len(_SO100_JOINT_NAMES)
        if len(step) != n:
            raise ROSConfigError(
                f"action.joint_targets[0] has {len(step)} values; SO-100 has {n} joints."
            )
        return _joint_values_to_lerobot(step)
