"""Joint-units conversion at the policy boundary (issue #135).

The runner no longer guesses checkpoint joint units — every joint-position
rSkill declares ``action_contract.joint_units``, converted deg<->rad only
when ``degrees``. Pins the two actuation-critical helpers:

* ``_robot_state_to_policy`` — robot radians → policy order, rad->deg
  for a degrees checkpoint (else ~57x too small → OOD).
* ``_policy_action_to_robot`` — policy action → robot order, deg->rad
  for a degrees checkpoint (else ~57x too large → arm slams its limits).

Gripper channels carry a motor unit, not an angle; ``gripper_scale`` maps the
HAL's normalized [0, 1] to a checkpoint's [0, 100] dataset surface when declared.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import RobotDescription, RSkillManifest
from openral_core.exceptions import ROSConfigError
from openral_rskill._policy_io import PolicyIOCodec
from pydantic import ValidationError

_SO101_SMOLVLA = "rskills/rskill-smolvla-so101-eraser_place-bf16/rskill.yaml"
_SO101_MOLMOACT2 = "rskills/molmoact2-so101-nf4/rskill.yaml"


def _codec(
    robot_to_policy: list[int] | None,
    joint_units_are_degrees: bool,
    policy_is_gripper: list[bool],
    policy_gripper_scale: float = 1.0,
) -> PolicyIOCodec:
    return PolicyIOCodec(
        robot_to_policy=robot_to_policy,
        joint_units_are_degrees=joint_units_are_degrees,
        policy_is_gripper=policy_is_gripper,
        gripper_scale=policy_gripper_scale,
    )


def _robot_state_to_policy(state: NDArray[Any], *args: Any, **kwargs: Any) -> NDArray[Any]:
    return _codec(*args, **kwargs).to_policy_state(state)


def _policy_action_to_robot(action: NDArray[Any], *args: Any, **kwargs: Any) -> NDArray[Any]:
    return _codec(*args, **kwargs).to_robot_action(action)


def _build_joint_permutation(
    *, adapter: object, description: RobotDescription
) -> tuple[list[int] | None, list[bool]]:
    codec = PolicyIOCodec.from_manifest(None, description, adapter=adapter)
    return codec.robot_to_policy, codec.policy_is_gripper


def test_degrees_state_converts_rad_to_deg() -> None:
    """A degrees policy is fed rad->deg-converted state; the gripper is untouched."""
    robot_state = np.array([math.pi / 2, -math.pi, 0.5])  # radians (arm, arm, gripper)
    policy_is_gripper = [False, False, True]
    out = _robot_state_to_policy(
        robot_state, None, joint_units_are_degrees=True, policy_is_gripper=policy_is_gripper
    )
    assert out[0] == 90.0
    assert out[1] == -180.0
    assert out[2] == 0.5  # gripper left as-is


def test_degrees_action_converts_deg_to_rad() -> None:
    """A degrees policy's action is deg->rad-converted before the HAL; gripper untouched."""
    policy_action = np.array([90.0, -180.0, 42.0])  # degrees (arm, arm, gripper)
    policy_is_gripper = [False, False, True]
    out = _policy_action_to_robot(
        policy_action, None, joint_units_are_degrees=True, policy_is_gripper=policy_is_gripper
    )
    assert math.isclose(out[0], math.pi / 2)
    assert math.isclose(out[1], -math.pi)
    assert out[2] == 42.0  # gripper left as-is


def test_radians_is_identity() -> None:
    """A radians checkpoint (joint_units_are_degrees=False) → no conversion."""
    state = np.array([1.0, -2.0, 0.3])
    action = np.array([0.7, -1.1, 0.9])
    gripper = [False, False, True]
    np.testing.assert_array_equal(
        _robot_state_to_policy(
            state, None, joint_units_are_degrees=False, policy_is_gripper=gripper
        ),
        state,
    )
    np.testing.assert_array_equal(
        _policy_action_to_robot(
            action, None, joint_units_are_degrees=False, policy_is_gripper=gripper
        ),
        action,
    )


def test_conversion_round_trips_through_reorder() -> None:
    """deg->rad->deg round-trips even with a robot<->policy permutation."""
    robot_state = np.array([0.1, 0.2, 0.3])
    perm = [2, 0, 1]  # robot index i maps to policy index perm[i]
    gripper = [False, False, False]
    policy = _robot_state_to_policy(
        robot_state, perm, joint_units_are_degrees=True, policy_is_gripper=gripper
    )
    back = _policy_action_to_robot(
        policy, perm, joint_units_are_degrees=True, policy_is_gripper=gripper
    )
    np.testing.assert_allclose(back, robot_state)


def test_gripper_scale_maps_normalized_hal_to_lerobot_motor_units() -> None:
    """SO-101 policy sees [0,100], while the HAL remains normalized [0,1]."""
    robot_state = np.array([0.5])
    policy = _robot_state_to_policy(
        robot_state,
        None,
        joint_units_are_degrees=True,
        policy_is_gripper=[True],
        policy_gripper_scale=100.0,
    )
    assert policy[0] == 50.0

    robot = _policy_action_to_robot(
        policy,
        None,
        joint_units_are_degrees=True,
        policy_is_gripper=[True],
        policy_gripper_scale=100.0,
    )
    assert robot[0] == 0.5


def test_missing_policy_feature_names_falls_back_to_robot_gripper_role() -> None:
    """SmolVLA config has only action shape; the robot manifest still identifies gripper."""
    description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    adapter = SimpleNamespace(
        _policy=SimpleNamespace(
            config=SimpleNamespace(
                output_features={"action": SimpleNamespace(shape=(6,))},
            )
        )
    )

    permutation, grippers = _build_joint_permutation(
        adapter=adapter,
        description=description,
    )

    assert permutation is None
    assert grippers == [False, False, False, False, False, True]


def test_adapter_without_a_policy_passes_through_instead_of_raising() -> None:
    """An adapter exposing no ``_policy`` must fall through, not blow up.

    ACT/DiffusionPolicy adapters have no ``_policy``; reading it raises
    AttributeError and leaves the local unbound, so the follow-up
    ``output_features`` read raised ``UnboundLocalError`` — uncaught by
    ``except (AttributeError, KeyError, TypeError)`` — crashing where the
    contract says "pass through".
    """
    description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")

    permutation, grippers = _build_joint_permutation(
        adapter=SimpleNamespace(),  # no `_policy` at all
        description=description,
    )

    assert permutation is None
    assert grippers == []


# ─── Codec built from the manifest + robot description ──────────────────────


def test_so101_manifests_declare_gripper_scale_in_the_action_contract() -> None:
    """Both SO-101 checkpoints carry the LeRobot [0, 100] gripper as a typed field.

    ``molmoact2-so101-nf4``'s norm_stats gripper q01..q99 is 0.94..44.1 (max 118) —
    the same [0, 100] motor range as the SmolVLA eraser checkpoint, which used to be
    the only one declaring it (as an untyped ``policy_extras`` key).
    """
    for path in (_SO101_SMOLVLA, _SO101_MOLMOACT2):
        manifest = RSkillManifest.from_yaml(path)
        assert manifest.action_contract is not None
        assert manifest.action_contract.gripper_scale == 100.0, path
        assert "gripper_scale" not in manifest.policy_extras, path


def test_codec_from_real_so101_manifest_converts_degrees_and_gripper() -> None:
    manifest = RSkillManifest.from_yaml(_SO101_SMOLVLA)
    description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    codec = PolicyIOCodec.from_manifest(manifest, description)
    assert codec.joint_units_are_degrees
    assert codec.gripper_scale == 100.0
    state = np.array([math.pi / 2, 0.0, 0.0, 0.0, -math.pi / 4, 0.25])
    policy_state = codec.to_policy_state(state)
    np.testing.assert_allclose(policy_state, [90.0, 0.0, 0.0, 0.0, -45.0, 25.0])
    np.testing.assert_allclose(codec.to_robot_action(policy_state), state)


def test_manifest_joint_names_permute_by_name() -> None:
    """``action_contract.joint_names`` is the policy order, matched BY NAME.

    Replaces the runner's old OpenArm-only ``openarm_`` prefix strip: a checkpoint
    whose feature names differ from ``robot.yaml`` declares the policy order in
    robot joint names instead.
    """
    description = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    robot_names = [j.name for j in description.joints]
    policy_names = robot_names[8:] + robot_names[:8]  # right-first checkpoint
    manifest = RSkillManifest.from_yaml(_SO101_SMOLVLA)
    raw = manifest.model_dump(mode="json")
    raw["action_contract"] = {
        "dim": 16,
        "representation": "joint_positions",
        "joint_units": "radians",
        "joint_names": policy_names,
    }
    raw["state_contract"] = None
    manifest = RSkillManifest.model_validate(raw)
    codec = PolicyIOCodec.from_manifest(manifest, description)
    assert codec.policy_joint_names == policy_names
    policy_action = np.arange(16, dtype=np.float32)  # value == policy index
    robot_action = codec.to_robot_action(policy_action)
    # robot left_joint1 (index 0) is policy index 8; right_joint1 (8) is policy 0.
    assert robot_action[0] == 8.0
    assert robot_action[8] == 0.0
    # grippers follow the robot role through the permutation
    assert codec.policy_is_gripper[7] and codec.policy_is_gripper[15]


def test_manifest_joint_names_not_on_the_robot_is_a_config_error() -> None:
    description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    manifest = RSkillManifest.from_yaml(_SO101_SMOLVLA)
    raw = manifest.model_dump(mode="json")
    raw["action_contract"]["joint_names"] = [
        "openarm_left_joint1",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    with pytest.raises(ROSConfigError, match="joint_names"):
        PolicyIOCodec.from_manifest(RSkillManifest.model_validate(raw), description)


def test_legacy_policy_extras_gripper_scale_still_honoured() -> None:
    """One-release deprecation: the old untyped key still converts (with a warning)."""
    description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    raw = RSkillManifest.from_yaml(_SO101_SMOLVLA).model_dump(mode="json")
    raw["action_contract"]["gripper_scale"] = 1.0
    raw["policy_extras"]["gripper_scale"] = 100.0
    codec = PolicyIOCodec.from_manifest(RSkillManifest.model_validate(raw), description)
    assert codec.gripper_scale == 100.0


def test_conflicting_gripper_scales_are_a_config_error() -> None:
    description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    raw = RSkillManifest.from_yaml(_SO101_SMOLVLA).model_dump(mode="json")
    raw["policy_extras"]["gripper_scale"] = 50.0  # contract says 100
    with pytest.raises(ROSConfigError, match="gripper_scale"):
        PolicyIOCodec.from_manifest(RSkillManifest.model_validate(raw), description)


def test_action_contract_rejects_bad_gripper_scale_and_joint_names() -> None:
    raw = RSkillManifest.from_yaml(_SO101_SMOLVLA).model_dump(mode="json")
    for bad in (
        {"gripper_scale": 0.0},
        {"joint_names": ["shoulder_pan"]},  # len != dim
        {"joint_names": ["gripper"] * 6},  # duplicates
    ):
        broken = {**raw, "action_contract": {**raw["action_contract"], **bad}}
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(broken)


def test_clamp_pulls_inside_the_robot_limits() -> None:
    description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    codec = PolicyIOCodec.from_manifest(None, description)
    out = codec.clamp(np.array([5.0, -5.0, 0.0, 0.0, 0.0, 2.0]))
    assert out[0] < 1.9199 and out[1] > -1.7453
    assert out[2] == 0.0
    assert 0.0 < out[5] < 1.0


def test_explicit_non_gripper_role_beats_a_gripper_name() -> None:
    """A wrist tagged ``role: arm`` but named ``gripper_roll`` is a joint, not a gripper."""
    from openral_core.schemas import JointSpec
    from openral_rskill._policy_io import _is_gripper_joint

    base = RobotDescription.from_yaml("robots/so101_follower/robot.yaml").joints[0]
    wrist = base.model_copy(update={"name": "gripper_roll", "role": "arm"})
    untagged = base.model_copy(update={"name": "gripper_roll", "role": "unknown"})
    assert isinstance(wrist, JointSpec)
    assert _is_gripper_joint(wrist) is False
    assert _is_gripper_joint(untagged) is True
