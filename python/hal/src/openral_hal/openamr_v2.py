"""OpenAMRobot 2.0 MuJoCo digital twin: differential base + lift + two OpenArm v2 arms.

The twin is a manifest-driven ``MujocoArmHAL``: the 20 manifest joints (3 planar
base, lift, 2 x (7 arm + gripper)) map 1:1 onto the MJCF's position actuators
(``openral_hal._openamr_v2_assets``), so ``JOINT_POSITION`` and slot groups work
unchanged. This class adds the mobile base surface that ``MobileBaseBridge``
reads (``base_pose`` / ``base_twist``) and ``BODY_TWIST``.

Subclassing ``OpenArmMujocoHAL`` would buy nothing: that class only binds the
OpenArm manifest. The arms here differ only by their joint names in the
manifest. The real robot is ``openral_hal.composed.ComposedHAL`` (the manifest's
``hal.real``), which reuses ``OpenArmRealHAL`` unchanged beside a platform base and
a held lift; it has not run on hardware (units ship 2027-04).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import structlog
from openral_core import BODY_TWIST_DIM
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import Action, ControlMode

from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal._slot_group import compose_slot_group_action

if TYPE_CHECKING:
    from openral_core import RobotDescription

__all__ = ["OpenAMRv2MujocoHAL"]

log = structlog.get_logger(__name__)

# Tolerance for a twist component the base cannot actuate (vy, vz, wx, wy).
_UNACTUATED_TWIST_EPS = 1e-6
# Below this yaw rate the arc integration degenerates to a straight line.
_STRAIGHT_LINE_WZ = 1e-9
_PLANAR_BASE_DOF = 3  # x, y, yaw


class OpenAMRv2MujocoHAL(MujocoArmHAL):
    """HAL adapter for the OpenAMRobot 2.0 MuJoCo twin.

    ``BODY_TWIST`` Euler-integrates ``(vx, wz)`` for one ``body_twist_dt_s``
    (the ``G1MujocoHAL`` / ``PandaMobileHAL`` convention; ``MobileBaseBridge``
    sends one action per ``/cmd_vel``). It places the base there kinematically,
    since a diff-drive base has no wheel model here, and its position actuators
    hold it. The arms and lift keep their last targets. A non-zero ``vy`` is refused,
    because the base is differential (``base_kinematics: differential``).

    A slot group that leaves the base and lift (``role: base`` / ``torso``)
    uncommanded holds them at their current ctrl targets, so an OpenArm rSkill's
    arms-and-grippers tick runs unchanged; an uncommanded arm or gripper joint is
    still refused.

    Args:
        description: The loaded ``robots/openamr_v2/robot.yaml``; ``build_hal``
            threads it.
        mjcf_path: Optional MJCF override; default builds ``openarm:openamr_v2``.
        settle_steps: MuJoCo steps per action; ``None`` derives one control period.
        gravity_enabled: When ``False``, gravity is zeroed at ``connect``.
        staleness_limit_s: Maximum age of a cached state.
        body_twist_dt_s: Seconds of motion one ``BODY_TWIST`` action integrates.

    Example:
        >>> from openral_core import RobotDescription  # doctest: +SKIP
        >>> desc = RobotDescription.from_yaml("robots/openamr_v2/robot.yaml")  # doctest: +SKIP
        >>> hal = OpenAMRv2MujocoHAL(description=desc)  # doctest: +SKIP
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
        body_twist_dt_s: float = 0.05,
    ) -> None:
        """Initialise from the manifest; the MJCF is built/loaded at ``connect``."""
        if not description.base_joints or len(description.base_joints) != _PLANAR_BASE_DOF:
            raise ROSConfigError(
                f"OpenAMRv2MujocoHAL needs base_joints [x, y, yaw]; got {description.base_joints}"
            )
        if body_twist_dt_s <= 0.0:
            raise ROSConfigError("OpenAMRv2MujocoHAL body_twist_dt_s must be > 0.")
        self._init_from_description(
            description,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )
        self._body_twist_dt_s = body_twist_dt_s
        self._base_names: list[str] = list(description.base_joints)
        # An arms-only slot group (a 16-D OpenArm rSkill) holds the base and lift.
        self._hold_if_uncommanded = frozenset(
            j.name for j in description.joints if j.role in ("base", "torso")
        )
        self._last_body_twist: tuple[float, float, float, float, float, float] = (0.0,) * 6

    @property
    def base_pose(self) -> tuple[float, float, float]:
        """Current base ``(x, y, yaw)`` in the odom/world frame."""
        self._require_connected("base_pose")
        assert self._data is not None
        x, y, yaw = (float(self._data.qpos[self._joint_qpos_addr[n]]) for n in self._base_names)
        return (x, y, yaw)

    @property
    def base_twist(self) -> tuple[float, float, float, float, float, float]:
        """Last commanded body twist (``base_footprint`` frame); zeroed by any other action."""
        return self._last_body_twist

    def send_action(self, action: Action) -> None:
        """Route ``BODY_TWIST`` to the planar base; everything else to the joint path.

        A slot group (``tick_group_size > 1``) is staged whole, like the joint path:
        its twist slot drives the base and its other slots compose into one joint
        step, applied together and committed once against the tick watermark.
        """
        if int(action.tick_group_size) > 1:
            self._send_group(action)
            return
        if action.control_mode != ControlMode.BODY_TWIST.value:
            self._last_body_twist = (0.0,) * 6
            super().send_action(action)
            return
        self._require_connected("send_action")
        self._validate_action(action)
        self._slot_group.admit(action)
        self._drive_base(action)
        self._slot_group.commit_tick(action)

    def _send_group(self, action: Action) -> None:
        self._require_connected("send_action")
        group = self._slot_group.stage(action)
        if group is None:
            return
        twist = ControlMode.BODY_TWIST.value
        twists = [a for a in group if a.control_mode == twist]
        joints = [a for a in group if a.control_mode != twist]
        if len(twists) > 1:
            raise ROSConfigError(f"OpenAMRv2MujocoHAL: {len(twists)} BODY_TWIST slots in one tick.")
        # Everything is validated before anything moves.
        for t in twists:
            self._validate_action(t)
        composed = (
            compose_slot_group_action(joints, self._joint_names, hold=self._held_ctrl_targets())
            if joints
            else None
        )
        if composed is not None:
            self._validate_action(composed)
            super().send_action(composed)  # unticked: the group commits below
        if twists:
            self._drive_base(twists[0])
        else:
            self._last_body_twist = (0.0,) * 6
        self._slot_group.commit(group, applied=composed if composed is not None else twists[0])

    def _drive_base(self, action: Action) -> None:
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        assert self._data is not None and self._model is not None
        vx, _vy, _vz, _wx, _wy, wz = (float(v) for v in action.body_twist[0])  # type: ignore[index]  # reason: _validate_action guarantees body_twist
        self._last_body_twist = (vx, 0.0, 0.0, 0.0, 0.0, wz)
        x, y, yaw = self.base_pose
        dt = self._body_twist_dt_s
        # Exact arc integration (straight line when wz == 0) keeps odom honest.
        if abs(wz) > _STRAIGHT_LINE_WZ:
            x += vx / wz * (math.sin(yaw + wz * dt) - math.sin(yaw))
            y -= vx / wz * (math.cos(yaw + wz * dt) - math.cos(yaw))
        else:
            x += vx * math.cos(yaw) * dt
            y += vx * math.sin(yaw) * dt
        yaw = (yaw + wz * dt + math.pi) % (2.0 * math.pi) - math.pi
        for name, value in zip(self._base_names, (x, y, yaw), strict=True):
            self._data.qpos[self._joint_qpos_addr[name]] = value
            self._data.qvel[self._joint_qvel_addr[name]] = 0.0
            self._data.ctrl[self._actuator_index[name]] = value
        for _ in range(self._settle_steps):
            mj.mj_step(self._model, self._data)
        self._last_state_time = time.monotonic()
        self._last_action_ns = time.monotonic_ns()
        log.debug("hal.send_action", robot=self.description.name, body_twist=(vx, wz))

    def idle_step(self, wall_dt_s: float | None = None) -> bool:
        """HOLD-step; the base is still, so its reported twist is zero."""
        self._last_body_twist = (0.0,) * 6
        return super().idle_step(wall_dt_s=wall_dt_s)

    def _validate_action(self, action: Action) -> None:
        if action.control_mode != ControlMode.BODY_TWIST.value:
            super()._validate_action(action)
            return
        if not action.body_twist or len(action.body_twist[0]) != BODY_TWIST_DIM:
            raise ROSConfigError(
                f"OpenAMRv2MujocoHAL: BODY_TWIST needs one {BODY_TWIST_DIM}-float row "
                "(vx, vy, vz, wx, wy, wz)."
            )
        row = action.body_twist[0]
        if any(abs(row[i]) > _UNACTUATED_TWIST_EPS for i in (1, 2, 3, 4)):
            raise ROSConfigError(
                "OpenAMRv2MujocoHAL: the base is differential: only vx and wz are "
                f"actuated; got (vx, vy, vz, wx, wy, wz) = {tuple(row)}."
            )
