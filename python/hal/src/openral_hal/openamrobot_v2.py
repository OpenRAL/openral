"""OpenAMRobot 2.0 MuJoCo digital twin: differential base + lift + two OpenArm v2 arms.

The twin is a manifest-driven ``MujocoArmHAL``: the lift and the 2 x (7 arm + gripper)
manifest joints map onto the MJCF's position actuators (``sim.actuator_index``), and the
three planar base joints are unactuated and measured
(``openral_hal._openamrobot_v2_assets``), so ``JOINT_POSITION`` and slot groups work
unchanged. Its rollout (``_OpenAMRobotV2Twin``) turns ``BODY_TWIST`` into wheel speeds on
the differential base and reports wheel odometry as ``base_pose`` and the measured
``base_twist``; ``MobileBaseBridge`` reads both through the shell.

Subclassing ``OpenArmMujocoHAL`` would buy nothing: that class only binds the
OpenArm manifest. The arms here differ only by their joint names in the
manifest. The real robot is ``openral_hal.composed.ComposedHAL`` (the manifest's
``hal.real``), which reuses ``OpenArmRealHAL`` unchanged beside a platform base and
a held lift; it has not run on hardware.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Unpack

import numpy as np
from openral_core.exceptions import ROSConfigError

from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal._openamrobot_v2_assets import WHEEL_JOINTS, diff_drive_accel_limits
from openral_hal.mujoco_twin import MujocoTwinRollout, TwinKwargs

if TYPE_CHECKING:
    from openral_core import Action, RobotDescription

__all__ = ["OpenAMRobotV2MujocoHAL"]

_PLANAR_BASE_DOF = 3  # x, y, yaw
# diff_drive_controller's cmd_vel_timeout (and SimAttachedHAL's held setpoint): a wheel
# command older than this, in sim time, is dropped and the wheels are told to stop.
_CMD_VEL_TIMEOUT_S = 0.5
# A wheel axis must lie along the body's y (cosine): a left and a right drive wheel.
_AXIS_ALONG_Y = 0.99
# How far (m, rad) a joint row's base entry may sit from the measured base: a row read
# back from the state a tick or two earlier while the base rolls. Anything further is a
# base move the row cannot make (the base has no actuator; only BODY_TWIST moves it).
_BASE_ROW_TOLERANCE = 0.05


def _toward(value: float, target: float, step: float) -> float:
    """``value`` moved towards ``target`` by at most ``step``."""
    return value + max(-step, min(step, target - value))


class _OpenAMRobotV2Twin(MujocoTwinRollout):
    """The OpenAMRobot 2.0 twin: a differential base driven through its two wheels.

    A ``BODY_TWIST`` (the manifest's ``base_kinematics: differential`` refuses a
    lateral ``vy``) becomes the two wheels' speed targets, ``(vx -/+ wz * h) / r`` for
    the left / right wheel, with the wheel radius ``r`` and half-track ``h`` read from
    the URDF the twin was built from (upstream's wheel cylinders and joint positions),
    reached under upstream's DiffDrive acceleration limits (``diff_drive_accel_limits``).
    Contact with the floor moves the base; its three planar joints are unactuated and
    measure where it went. Like ``diff_drive_controller``, a target is held until the
    next twist and ramped to zero after ``_CMD_VEL_TIMEOUT_S`` of sim time without one;
    an e-stop's ``hold_present_pose`` stops the wheels at once.

    ``read_base_pose`` is wheel odometry (the wheel angles integrated with the same
    model, so wheel slip shows up as drift, as on the real base) and ``read_base_twist``
    the speed the wheels measure. The base joints' qpos stays the simulated ground truth.
    Arms-only commands leave the wheels and hold the lift (``role: torso``) at its
    current ctrl target.
    """

    def __init__(self, description: RobotDescription, **kwargs: Unpack[TwinKwargs]) -> None:
        """Bind the twin; ``kwargs`` are ``MujocoTwinRollout``'s."""
        super().__init__(description, **kwargs)
        self._base_names: list[str] = list(description.base_joints or [])
        self.hold_if_uncommanded = frozenset(
            j.name for j in description.joints if j.role in ("base", "torso")
        )
        self._wheel_qpos: tuple[int, int] = (0, 0)
        self._wheel_dof: tuple[int, int] = (0, 0)
        self._wheel_act: tuple[int, int] = (0, 0)
        self._wheel_r: tuple[float, float] = (0.0, 0.0)
        self._half_track = 0.0
        self._wheel_sign: tuple[float, float] = (1.0, 1.0)
        self._odom = [0.0, 0.0, 0.0]
        self._last_wheel_angle: tuple[float, float] = (0.0, 0.0)
        self._last_cmd_time: float | None = None
        self._accel = (0.0, 0.0)  # (m/s^2, rad/s^2): upstream's DiffDrive, read at load
        self._target_twist = (0.0, 0.0)  # (vx, wz) commanded
        self._twist = (0.0, 0.0)  # (vx, wz) on the wheels, ramped towards the target

    def _on_load(self) -> None:
        """Read the wheel geometry off the compiled model and zero odometry.

        Raises:
            ROSConfigError: Gravity is off (the wheels need the base's weight to grip, so
                the base would never move while its wheels report motion), the wheels are
                not a left and a right one on axes along y, or upstream's DiffDrive limits
                cannot be read (``diff_drive_accel_limits``).
        """
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        model, data = self._require_loaded("wheel geometry")
        if not np.any(model.opt.gravity):
            raise ROSConfigError(
                "openamrobot_v2 twin: gravity_enabled=False, but the base rolls on its wheels "
                "and they grip the floor only under the base's weight. Leave gravity on "
                "(everything on the lift carriage is gravity-compensated)."
            )
        self._accel = diff_drive_accel_limits()
        base = model.body("base_footprint").id
        base_rot = data.xmat[base].reshape(3, 3)
        qpos, dof, act, sign, offset, radius = [], [], [], [], [], []
        for joint in WHEEL_JOINTS:
            jid = model.joint(joint).id
            qpos.append(int(model.jnt_qposadr[jid]))
            dof.append(int(model.jnt_dofadr[jid]))
            act.append(int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{joint}_vel")))
            # Body frame: +y is left. A positive wheel turn rolls the base forward when
            # the wheel axis points left (the contact point moves backwards).
            sign.append(float(base_rot.T @ data.xaxis[jid] @ np.array([0.0, 1.0, 0.0])))
            offset.append(float((base_rot.T @ (data.xanchor[jid] - data.xpos[base]))[1]))
            body = int(model.jnt_bodyid[jid])
            cylinders = [
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] == body
                and model.geom_type[g] == mj.mjtGeom.mjGEOM_CYLINDER
                and (model.geom_contype[g] or model.geom_conaffinity[g])
            ]
            if len(cylinders) != 1:
                raise ROSConfigError(f"{joint}: expected one collision cylinder, got {cylinders}")
            radius.append(float(model.geom_size[cylinders[0]][0]))
        self._wheel_qpos = (qpos[0], qpos[1])
        self._wheel_dof = (dof[0], dof[1])
        self._wheel_act = (act[0], act[1])
        self._wheel_sign = (sign[0], sign[1])
        self._wheel_r = (radius[0], radius[1])
        self._half_track = (offset[0] - offset[1]) / 2.0
        aligned = all(abs(s) > _AXIS_ALONG_Y for s in sign)
        if not (min(radius) > 0.0 and self._half_track > 0.0 and aligned):
            raise ROSConfigError(
                f"openamrobot_v2 wheels: radii {radius}, half-track {self._half_track}, "
                f"axis signs {sign}; expected a left and a right wheel on axes along y."
            )
        self._odom = [0.0, 0.0, 0.0]
        self._last_wheel_angle = self._wheel_angles()
        self._last_cmd_time = None
        self._stop_wheels()

    def _wheel_angles(self) -> tuple[float, float]:
        assert self.data is not None
        return (
            float(self.data.qpos[self._wheel_qpos[0]]),
            float(self.data.qpos[self._wheel_qpos[1]]),
        )

    def _stop_wheels(self) -> None:
        """Zero the speed command at once; the wheels brake where they are."""
        assert self.model is not None and self.data is not None
        self._target_twist = self._twist = (0.0, 0.0)
        for act, qpos in zip(self._wheel_act, self._wheel_qpos, strict=True):
            self.data.ctrl[act] = 0.0
            # The integrated target restarts at the wheel's angle: no snap back to where
            # a slipping wheel's target had run ahead.
            self.data.act[self.model.actuator_actadr[act]] = self.data.qpos[qpos]

    def _per_step(self) -> None:
        """Odometry over the last step, the command timeout, then the ramped wheel speeds."""
        assert self.model is not None and self.data is not None
        left, right = self._wheel_angles()
        last_l, last_r = self._last_wheel_angle
        self._last_wheel_angle = (left, right)
        ds_l = (left - last_l) * self._wheel_r[0] * self._wheel_sign[0]
        ds_r = (right - last_r) * self._wheel_r[1] * self._wheel_sign[1]
        ds, dyaw = (ds_l + ds_r) / 2.0, (ds_r - ds_l) / (2.0 * self._half_track)
        x, y, yaw = self._odom
        mid = yaw + dyaw / 2.0
        self._odom = [
            x + ds * math.cos(mid),
            y + ds * math.sin(mid),
            (yaw + dyaw + math.pi) % (2.0 * math.pi) - math.pi,
        ]
        if (
            self._last_cmd_time is not None
            and float(self.data.time) - self._last_cmd_time > _CMD_VEL_TIMEOUT_S
        ):
            self._target_twist = (0.0, 0.0)  # ramped down, as diff_drive_controller does
            self._last_cmd_time = None
        # Move the wheels' twist towards the command by upstream's acceleration limits.
        dt = float(self.model.opt.timestep)
        self._twist = (
            _toward(self._twist[0], self._target_twist[0], self._accel[0] * dt),
            _toward(self._twist[1], self._target_twist[1], self._accel[1] * dt),
        )
        vx, wz = self._twist
        self.data.ctrl[self._wheel_act[0]] = (vx - wz * self._half_track) / (
            self._wheel_r[0] * self._wheel_sign[0]
        )
        self.data.ctrl[self._wheel_act[1]] = (vx + wz * self._half_track) / (
            self._wheel_r[1] * self._wheel_sign[1]
        )

    def read_base_pose(self) -> tuple[float, float, float]:
        """The base ``(x, y, yaw)`` by wheel odometry, from where the twin loaded.

        Raises:
            ROSRuntimeError: Not loaded (the HAL is not connected).
        """
        self._require_loaded("base_pose")
        x, y, yaw = self._odom
        return (x, y, yaw)

    def read_urdf_extra_joint_positions(self) -> dict[str, float]:
        """The drive wheels' angles by URDF joint name (``robot_state_publisher``'s TF).

        Empty before load: the HAL then has no wheels to report.
        """
        if self.data is None:
            return {}
        return dict(zip(WHEEL_JOINTS, self._wheel_angles(), strict=True))

    def read_base_twist(self) -> tuple[float, float, float, float, float, float]:
        """The body twist the wheels measure: ``(vx, 0, 0, 0, 0, wz)``."""
        _model, data = self._require_loaded("base_twist")
        v_l = float(data.qvel[self._wheel_dof[0]]) * self._wheel_r[0] * self._wheel_sign[0]
        v_r = float(data.qvel[self._wheel_dof[1]]) * self._wheel_r[1] * self._wheel_sign[1]
        return ((v_l + v_r) / 2.0, 0.0, 0.0, 0.0, 0.0, (v_r - v_l) / (2.0 * self._half_track))

    def _held_ctrl_targets(self) -> dict[str, float]:
        """The lift's ctrl target, and each unactuated base joint at its current qpos."""
        held = super()._held_ctrl_targets()
        assert self.data is not None
        for name in self._base_names:
            held.setdefault(name, float(self.data.qpos[self.joint_qpos_addr[name]]))
        return held

    def _compose(
        self, actions: list[Action]
    ) -> tuple[list[float] | None, tuple[float, float, float] | None, Action]:
        """``MujocoTwinRollout._compose``, refusing a joint row that would move the base.

        The base joints have no actuator, so a row's base entries cannot be applied; one
        within ``_BASE_ROW_TOLERANCE`` of the measured base (a row read back from the state)
        is the base staying put, anything further is refused rather than dropped.

        Raises:
            ROSConfigError: A base entry beyond the tolerance, or see the base class.
        """
        targets, twist, applied = super()._compose(actions)
        if targets is not None:
            assert self.data is not None
            for name in self._base_names:
                now = float(self.data.qpos[self.joint_qpos_addr[name]])
                off = targets[self.joint_names.index(name)] - now
                if name == self._base_names[-1]:  # yaw: the short way round
                    off = (off + math.pi) % (2.0 * math.pi) - math.pi
                if abs(off) > _BASE_ROW_TOLERANCE:
                    raise ROSConfigError(
                        f"openamrobot_v2: a joint row puts {name} at "
                        f"{targets[self.joint_names.index(name)]:.3f}, {off:+.3f} from the "
                        "base's measured pose; the base moves only on its wheels (BODY_TWIST)."
                    )
        return targets, twist, applied

    def reset_to_pose(self, pose: list[float]) -> None:
        """Snap the joints to ``pose``; the wheels stop and odometry restarts at the new base.

        Raises:
            ROSConfigError: See ``MujocoTwinRollout.reset_to_pose``.
        """
        super().reset_to_pose(pose)
        assert self.data is not None
        self._stop_wheels()
        self._last_cmd_time = None
        x, y, yaw = (float(self.data.qpos[self.joint_qpos_addr[n]]) for n in self._base_names)
        self._odom = [x, y, yaw]
        self._last_wheel_angle = self._wheel_angles()

    def hold_present_pose(self) -> None:
        """Hold every joint where it is and stop the wheels (an e-stop latch or clear)."""
        super().hold_present_pose()
        if self.data is not None:
            self._stop_wheels()
            self._last_cmd_time = None

    def _drive_base(self, vx: float, vy: float, wz: float) -> None:
        """Command ``(vx, wz)``; the wheels ramp to it and hold it until the next twist."""
        del vy  # 0.0: refused beyond tolerance on a differential base (_planar_twist)
        assert self.data is not None
        self._target_twist = (vx, wz)
        self._last_cmd_time = float(self.data.time)


class OpenAMRobotV2MujocoHAL(MujocoArmHAL):
    """HAL adapter for the OpenAMRobot 2.0 MuJoCo twin.

    ``BODY_TWIST`` sets the two drive wheels' speeds for ``(vx, wz)``, and the base
    rolls on them (``MobileBaseBridge`` sends one action per ``/cmd_vel``). The speeds
    are held until the next twist, as ``diff_drive_controller`` holds them, and dropped
    after 0.5 s of sim time without one. The arms and lift keep their last targets. A
    non-zero ``vy`` is refused, because the base is differential
    (``base_kinematics: differential``). ``base_pose`` is wheel odometry.

    A slot group that leaves the base and lift (``role: base`` / ``torso``)
    uncommanded leaves the wheels alone and holds the lift, so an OpenArm rSkill's
    arms-and-grippers tick runs unchanged; an uncommanded arm or gripper joint is
    still refused. A tick that carries a twist slot beside its joint slots moves the
    base and the arms inside the same control period (one step).

    Args:
        description: The loaded ``robots/openamrobot_v2/robot.yaml``; ``build_hal``
            threads it.
        mjcf_path: Optional MJCF override; default builds ``openarm:openamrobot_v2``.
        settle_steps: MuJoCo steps per action; ``None`` derives one control period.
        gravity_enabled: When ``False``, gravity is zeroed at ``connect``.
        staleness_limit_s: Maximum age of a cached state.
        body_twist_dt_s: Must be one control period (``MujocoTwinRollout`` refuses any
            other at ``connect``); the wheel-driven base holds a twist until the next
            one or its timeout rather than integrating it over this interval.

    Example:
        >>> from openral_core import RobotDescription  # doctest: +SKIP
        >>> desc = RobotDescription.from_yaml("robots/openamrobot_v2/robot.yaml")  # doctest: +SKIP
        >>> hal = OpenAMRobotV2MujocoHAL(description=desc)  # doctest: +SKIP
        >>> hal.connect()  # doctest: +SKIP
        >>> hal.base_pose  # doctest: +SKIP
        (0.0, 0.0, 0.0)
    """

    def __init__(
        self,
        *,
        description: RobotDescription,
        mjcf_path: str | None = None,
        settle_steps: int | None = None,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
        body_twist_dt_s: float | None = None,
    ) -> None:
        """Initialise from the manifest; the MJCF is built/loaded at ``connect``."""
        if not description.base_joints or len(description.base_joints) != _PLANAR_BASE_DOF:
            raise ROSConfigError(
                "OpenAMRobotV2MujocoHAL needs base_joints [x, y, yaw]; "
                f"got {description.base_joints}"
            )
        if body_twist_dt_s is not None and not body_twist_dt_s > 0.0:  # NaN too
            raise ROSConfigError("OpenAMRobotV2MujocoHAL body_twist_dt_s must be > 0.")
        self._init_from_description(
            description,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
            body_twist_dt_s=body_twist_dt_s,
            twin=_OpenAMRobotV2Twin,
        )
