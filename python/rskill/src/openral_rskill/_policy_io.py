"""The one policy <-> robot I/O codec (joint order, deg<->rad, gripper scale, clamp).

openral's ``JointState`` / ``Action`` contract is radians in ``RobotDescription.joints``
order with a normalized ``[0, 1]`` gripper. A checkpoint can differ on all three axes:
its joint order (OpenArm right-first vs ``robot.yaml`` left-first), its angular unit
(LeRobot SO-ARM datasets record degrees) and its gripper unit (SO-ARM ``[0, 100]``).
``PolicyIOCodec`` is built once from the manifest + robot description and applied on
EVERY dispatch path — the whole-vector joint path and the ``action_contract.slots``
path — so a degrees value can never reach the safety kernel as radians.

Conversion order:

* state (robot -> policy): permute robot order -> policy order, then per channel
  ``gripper * gripper_scale`` or ``degrees(joint)`` (degrees checkpoints only).
* action, joint path (policy -> robot): per channel ``gripper / gripper_scale`` or
  ``radians(joint)``, written back in robot order; ``clamp`` then pulls each joint
  strictly inside ``RobotDescription`` ``position_limits``.
* action, slot path: the SAME per-channel unit conversion, in policy order, per slot
  (JOINT_POSITION / JOINT_VELOCITY channels deg->rad, gripper channels and
  GRIPPER_POSITION slots descaled). No permutation: slots route by their own
  ``joint_names``, and permuting first would shift every slot range.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import structlog
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import ActionRepresentation, ControlMode, JointUnits
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from openral_core.schemas import (
        ActionContract,
        ActionSlot,
        JointSpec,
        RobotDescription,
        RSkillManifest,
    )

__all__ = ["PolicyIOCodec"]

_log = structlog.get_logger(__name__)

# Slot control modes whose channels are joint angles (or angular rates).
_ANGULAR_SLOT_MODES = frozenset({ControlMode.JOINT_POSITION, ControlMode.JOINT_VELOCITY})
# Strictly inside the envelope: the safety kernel rejects ``value > limit`` on open
# intervals, so clamping to the exact limit would still trip it after float round-trips.
_CLAMP_EPS = 1e-3


def _effective_perm(robot_to_policy: list[int] | None, n: int) -> list[int]:
    """The joint permutation to use, identity when there is no (valid) reorder.

    Returning ``range(n)`` instead of skipping the conversion block is what keeps the
    deg<->rad conversion running on the no-reorder path (the bug that once sent a
    degrees checkpoint's actions out raw, ~57x too large).
    """
    if robot_to_policy is not None and len(robot_to_policy) == n:
        return robot_to_policy
    return list(range(n))


def _is_gripper_joint(joint: JointSpec) -> bool:
    """An explicit ``role`` wins; the name heuristic only covers untagged joints."""
    role = getattr(joint, "role", None)
    role = getattr(role, "value", role)
    if role not in (None, "unknown"):
        return role == "gripper"
    return "gripper" in joint.name.lower()


def _normalize_feature_name(name: str) -> str:
    """LeRobot feature key -> robot.yaml joint name (``right_joint_1.pos`` -> ``right_joint1``)."""
    return name.replace(".pos", "").replace("_joint_", "_joint")


def _adapter_feature_names(adapter: object) -> tuple[list[str], int | None]:
    """The checkpoint's action feature names and action dim, when the adapter exposes them."""
    policy = getattr(adapter, "_policy", None)
    config = getattr(policy, "config", None)
    try:
        names = [str(n) for n in getattr(config, "action_feature_names", None) or []]
    except TypeError:
        names = []
    try:
        dim: int | None = int(config.output_features["action"].shape[0])  # type: ignore[union-attr]  # reason: duck-typed lerobot config; AttributeError is caught
    except (AttributeError, KeyError, TypeError, IndexError):
        dim = None
    return names, dim


class PolicyIOCodec(BaseModel):
    """Typed policy <-> robot conversion, built by :meth:`from_manifest`.

    Attributes:
        joint_units_are_degrees: The checkpoint's joint channels are degrees.
        gripper_scale: Policy gripper units per robot (normalized ``[0, 1]``) unit.
        robot_to_policy: ``robot_to_policy[i] = j``: robot joint ``i`` is policy channel
            ``j``. ``None`` = identity.
        policy_is_gripper: Per policy channel, whether it is a gripper (never
            deg<->rad converted; scaled by ``gripper_scale``).
        policy_joint_names: The policy's joint order in robot joint names, when known.
        robot_joint_names: ``RobotDescription.joints`` names (slot-path gripper lookup).
        robot_is_gripper: Per robot joint, whether it is a gripper.
        joint_limits: Per robot joint ``(lo, hi)`` position limits, ``None`` if undeclared.

    Example:
        >>> import numpy as np
        >>> codec = PolicyIOCodec(
        ...     joint_units_are_degrees=True, policy_is_gripper=[False, True], gripper_scale=100.0
        ... )
        >>> codec.to_robot_action(np.array([180.0, 50.0])).round(4).tolist()
        [3.1416, 0.5]
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    joint_units_are_degrees: bool = False
    gripper_scale: float = Field(default=1.0, gt=0.0)
    robot_to_policy: list[int] | None = None
    policy_is_gripper: list[bool] = Field(default_factory=list)
    policy_joint_names: list[str] | None = None
    robot_joint_names: list[str] = Field(default_factory=list)
    robot_is_gripper: list[bool] = Field(default_factory=list)
    joint_limits: list[tuple[float, float] | None] = Field(default_factory=list)

    @classmethod
    def from_manifest(
        cls,
        manifest: RSkillManifest | None,
        description: RobotDescription | None,
        *,
        adapter: object | None = None,
    ) -> PolicyIOCodec:
        """Build the codec from the manifest's ``action_contract`` + the robot description.

        Joint order precedence: ``action_contract.joint_names`` (robot joint names in
        policy order — a name not on the robot is a ``ROSConfigError``) > the adapter's
        ``policy.config.action_feature_names`` (``.pos`` / ``_joint_`` normalised; a
        mismatch logs ``policy_io.feature_names_unmatched`` and falls back to identity)
        > identity.

        Raises:
            ROSConfigError: joint-position contract without ``joint_units``; a
                ``joint_names`` entry not on the robot; a legacy
                ``policy_extras.gripper_scale`` contradicting ``action_contract``.
        """
        ac = manifest.action_contract if manifest is not None else None
        name = manifest.name if manifest is not None else "?"
        degrees = cls._resolve_degrees(ac, name)
        gripper_scale = cls._resolve_gripper_scale(manifest, name)
        if description is None:
            return cls(joint_units_are_degrees=degrees, gripper_scale=gripper_scale)

        robot_names = [j.name for j in description.joints]
        robot_grip = [_is_gripper_joint(j) for j in description.joints]
        limits: list[tuple[float, float] | None] = [
            (float(j.position_limits[0]), float(j.position_limits[1]))
            if j.position_limits is not None
            else None
            for j in description.joints
        ]
        declared = ac.joint_names if ac is not None else None
        if declared is not None:
            unknown = [n for n in declared if n not in robot_names]
            if unknown or len(declared) != len(robot_names):
                raise ROSConfigError(
                    f"rSkill {name!r}: action_contract.joint_names must be a permutation "
                    f"of the robot's joints {robot_names}; unknown={unknown}, "
                    f"got {len(declared)} names for {len(robot_names)} joints"
                )
            policy_names = list(declared)
        else:
            feature_names, action_dim = (
                _adapter_feature_names(adapter) if adapter is not None else ([], None)
            )
            normalized = [_normalize_feature_name(n) for n in feature_names]
            if action_dim is None and ac is not None:
                action_dim = ac.dim  # the manifest's own action width
            if normalized and sorted(normalized) != sorted(robot_names):
                _log.warning(
                    "policy_io.feature_names_unmatched",
                    skill=name,
                    policy_names=normalized,
                    robot_names=robot_names,
                    hint="declare action_contract.joint_names to set the policy joint order",
                )
                action_dim = len(normalized)
                normalized = []
            if not normalized:
                return cls(
                    joint_units_are_degrees=degrees,
                    gripper_scale=gripper_scale,
                    policy_is_gripper=robot_grip if action_dim == len(robot_names) else [],
                    robot_joint_names=robot_names,
                    robot_is_gripper=robot_grip,
                    joint_limits=limits,
                )
            policy_names = normalized
        perm = [policy_names.index(n) for n in robot_names]
        policy_is_gripper = [False] * len(perm)
        for i, j in enumerate(perm):
            policy_is_gripper[j] = robot_grip[i]
        return cls(
            joint_units_are_degrees=degrees,
            gripper_scale=gripper_scale,
            robot_to_policy=perm,
            policy_is_gripper=policy_is_gripper,
            policy_joint_names=policy_names,
            robot_joint_names=robot_names,
            robot_is_gripper=robot_grip,
            joint_limits=limits,
        )

    @staticmethod
    def _resolve_degrees(ac: ActionContract | None, name: str) -> bool:
        if ac is None or ac.joint_units is None:
            if ac is not None and ac.representation is ActionRepresentation.JOINT_POSITIONS:
                raise ROSConfigError(
                    f"rSkill {name!r} has action_contract.representation='joint_positions' "
                    "but no action_contract.joint_units. Declare 'degrees' or 'radians' "
                    "(verified against the checkpoint's normalizer stats; issue #135)."
                )
            return False
        return bool(ac.joint_units is JointUnits.DEGREES)

    @staticmethod
    def _resolve_gripper_scale(manifest: RSkillManifest | None, name: str) -> float:
        if manifest is None:
            return 1.0
        declared = manifest.action_contract.gripper_scale if manifest.action_contract else 1.0
        if "gripper_scale" not in manifest.policy_extras:
            return declared
        raw = manifest.policy_extras["gripper_scale"]
        try:
            legacy = float(raw)  # type: ignore[arg-type]  # reason: untyped policy_extras value, validated below
        except (TypeError, ValueError) as exc:
            raise ROSConfigError(
                f"rSkill {name!r}: policy_extras.gripper_scale must be a number, got {raw!r}"
            ) from exc
        if not math.isfinite(legacy) or legacy <= 0.0:
            raise ROSConfigError(
                f"rSkill {name!r}: policy_extras.gripper_scale must be > 0, got {legacy}"
            )
        if declared not in (1.0, legacy):
            raise ROSConfigError(
                f"rSkill {name!r}: policy_extras.gripper_scale={legacy} contradicts "
                f"action_contract.gripper_scale={declared}; keep only the latter"
            )
        _log.warning(
            "policy_io.deprecated_policy_extras_gripper_scale",
            skill=name,
            gripper_scale=legacy,
            hint="move policy_extras.gripper_scale to action_contract.gripper_scale",
        )
        return legacy

    def _is_policy_gripper(self, j: int) -> bool:
        return j < len(self.policy_is_gripper) and self.policy_is_gripper[j]

    def to_policy_state(self, robot_state: NDArray[Any]) -> NDArray[Any]:
        """Robot-order radians state -> policy-order, policy-unit state."""
        n = robot_state.shape[0]
        out = robot_state.copy()
        for i, j in enumerate(_effective_perm(self.robot_to_policy, n)):
            val = float(robot_state[i])
            if self._is_policy_gripper(j):
                val *= self.gripper_scale
            elif self.joint_units_are_degrees:
                val = math.degrees(val)
            out[j] = val
        return out

    def to_robot_action(
        self, policy_action: NDArray[Any], *, slots: Sequence[ActionSlot] | None = None
    ) -> NDArray[Any]:
        """Policy action -> robot units (and robot order on the whole-vector path).

        With ``slots`` the vector stays in policy order (slots index it) and each
        slot's channels get the unit conversion its control mode implies.
        """
        if slots:
            return self._slots_to_robot_units(policy_action, slots)
        n = policy_action.shape[0]
        out = policy_action.copy()
        for i, j in enumerate(_effective_perm(self.robot_to_policy, n)):
            val = float(policy_action[j])
            if self._is_policy_gripper(j):
                val /= self.gripper_scale
            elif self.joint_units_are_degrees:
                val = math.radians(val)
            out[i] = val
        return out

    def _slots_to_robot_units(
        self, policy_action: NDArray[Any], slots: Sequence[ActionSlot]
    ) -> NDArray[Any]:
        out = policy_action.copy()
        grip_by_name = dict(zip(self.robot_joint_names, self.robot_is_gripper, strict=True))
        for slot in slots:
            if slot.discard:
                continue
            lo, hi = slot.range
            if slot.control_mode is ControlMode.GRIPPER_POSITION:
                out[lo : hi + 1] = policy_action[lo : hi + 1] / self.gripper_scale
                continue
            if slot.control_mode not in _ANGULAR_SLOT_MODES:
                continue
            for k in range(hi - lo + 1):
                if slot.joint_names:
                    is_grip = grip_by_name.get(slot.joint_names[k], False)
                else:  # whole-robot slot in RobotDescription.joints order
                    is_grip = k < len(self.robot_is_gripper) and self.robot_is_gripper[k]
                val = float(policy_action[lo + k])
                if is_grip:
                    val /= self.gripper_scale
                elif self.joint_units_are_degrees:
                    val = math.radians(val)
                out[lo + k] = val
        return out

    def clamp(self, robot_action: NDArray[Any]) -> NDArray[Any]:
        """Pull each joint strictly inside its declared position limits.

        A no-op when the vector width differs from the robot's joint count (the safety
        kernel still enforces the envelope downstream).
        """
        out = robot_action.copy()
        if not self.joint_limits or out.shape[0] != len(self.joint_limits):
            return out
        for i, lims in enumerate(self.joint_limits):
            if lims is not None:
                out[i] = min(max(float(out[i]), lims[0] + _CLAMP_EPS), lims[1] - _CLAMP_EPS)
        return out
