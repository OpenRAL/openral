"""OpenAMRobot 2.0 MuJoCo digital twin: differential base + lift + two OpenArm v2 arms.

The twin is a manifest-driven ``MujocoArmHAL``: the 20 manifest joints (3 planar
base, lift, 2 x (7 arm + gripper)) map 1:1 onto the MJCF's position actuators
(``openral_hal._openamrobot_v2_assets``), so ``JOINT_POSITION`` and slot groups work
unchanged. Its rollout (``_OpenAMRobotV2Twin``) adds ``BODY_TWIST`` on the
differential base and reads its own ``base_pose`` (the base joints' qpos);
``MobileBaseBridge`` reads it and the shell's ``base_twist`` through the shell.

Subclassing ``OpenArmMujocoHAL`` would buy nothing: that class only binds the
OpenArm manifest. The arms here differ only by their joint names in the
manifest. The real robot is ``openral_hal.composed.ComposedHAL`` (the manifest's
``hal.real``), which reuses ``OpenArmRealHAL`` unchanged beside a platform base and
a held lift; it has not run on hardware.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Unpack

from openral_core.exceptions import ROSConfigError

from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal.mujoco_twin import MujocoTwinRollout, TwinKwargs

if TYPE_CHECKING:
    from openral_core import RobotDescription

__all__ = ["OpenAMRobotV2MujocoHAL"]

# Below this yaw rate the arc integration degenerates to a straight line.
_STRAIGHT_LINE_WZ = 1e-9
_PLANAR_BASE_DOF = 3  # x, y, yaw


class _OpenAMRobotV2Twin(MujocoTwinRollout):
    """The OpenAMRobot 2.0 twin: a differential base placed kinematically per twist.

    A ``BODY_TWIST`` (the manifest's ``base_kinematics: differential`` refuses a
    lateral ``vy``) integrates ``(vx, wz)`` exactly (an arc; a straight line when
    ``wz == 0``) over ``body_twist_dt_s`` — one control period — from the current base
    qpos, writes the new pose into the base joints' qpos and ctrl (their position
    actuators hold it) and zeroes their qvel; the arms and lift keep their targets.
    Arms-only commands hold the base and lift (``role: base`` / ``torso``) at their
    current ctrl targets. ``read_base_pose`` reads the three base joints' qpos.
    """

    def __init__(self, description: RobotDescription, **kwargs: Unpack[TwinKwargs]) -> None:
        """Bind the twin; ``kwargs`` are ``MujocoTwinRollout``'s."""
        super().__init__(description, **kwargs)
        self._base_names: list[str] = list(description.base_joints or [])
        self.hold_if_uncommanded = frozenset(
            j.name for j in description.joints if j.role in ("base", "torso")
        )

    def read_base_pose(self) -> tuple[float, float, float]:
        """The base ``(x, y, yaw)``: its three planar joints' qpos.

        Raises:
            ROSRuntimeError: Not loaded (the HAL is not connected).
        """
        _model, data = self._require_loaded("base_pose")
        x, y, yaw = (float(data.qpos[self.joint_qpos_addr[n]]) for n in self._base_names)
        return (x, y, yaw)

    def _drive_base(self, vx: float, vy: float, wz: float) -> None:
        """Place the base where one ``body_twist_dt_s`` of ``(vx, wz)`` takes it."""
        del vy  # 0.0: refused beyond tolerance on a differential base (_planar_twist)
        data = self.data
        assert data is not None and self.body_twist_dt_s is not None
        x, y, yaw = self.read_base_pose()
        dt = self.body_twist_dt_s
        # Exact arc integration (straight line when wz == 0) keeps odom honest.
        if abs(wz) > _STRAIGHT_LINE_WZ:
            x += vx / wz * (math.sin(yaw + wz * dt) - math.sin(yaw))
            y -= vx / wz * (math.cos(yaw + wz * dt) - math.cos(yaw))
        else:
            x += vx * math.cos(yaw) * dt
            y += vx * math.sin(yaw) * dt
        yaw = (yaw + wz * dt + math.pi) % (2.0 * math.pi) - math.pi
        for name, value in zip(self._base_names, (x, y, yaw), strict=True):
            data.qpos[self.joint_qpos_addr[name]] = value
            data.qvel[self.joint_qvel_addr[name]] = 0.0
            data.ctrl[self.actuator_index[name]] = value


class OpenAMRobotV2MujocoHAL(MujocoArmHAL):
    """HAL adapter for the OpenAMRobot 2.0 MuJoCo twin.

    ``BODY_TWIST`` integrates ``(vx, wz)`` for one ``body_twist_dt_s`` — one control
    period of the twin, so the base moves at the commanded speed on the sim clock
    (``MobileBaseBridge`` sends one action per ``/cmd_vel``). It places the base there
    kinematically, since a diff-drive base has no wheel model here, and its position actuators
    hold it. The arms and lift keep their last targets. A non-zero ``vy`` is refused,
    because the base is differential (``base_kinematics: differential``).

    A slot group that leaves the base and lift (``role: base`` / ``torso``)
    uncommanded holds them at their current ctrl targets, so an OpenArm rSkill's
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
        body_twist_dt_s: Seconds of motion one ``BODY_TWIST`` action integrates;
            ``None`` (default) is one control period (``MujocoTwinRollout``).

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
        if body_twist_dt_s is not None and body_twist_dt_s <= 0.0:
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
