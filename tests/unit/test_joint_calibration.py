"""``action_contract.joint_calibration``: schema, codec, and who applies it on ``sim run``.

Real fixtures (CLAUDE.md §1.11): the LeIsaac PickOrange rSkill manifest and the
``so101_follower`` robot description. The reference is LeIsaac's own map
(``leisaac/utils/robot_utils.py`` ``convert_leisaac_action_to_lerobot``, pinned
24d3bcd), restated below with its USD joint limits and motor ranges.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openral_core import ActionContract, JointCalibration, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_rskill._policy_io import PolicyIOCodec
from openral_rskill.loader import load_rskill_manifest
from openral_sim.registry import SCENES
from openral_sim.sim_runner import _check_policy_units_owned, _runner_policy_codec
from pydantic import ValidationError

_REPO = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = load_rskill_manifest(str(_REPO / "rskills" / "smolvla-so101-leisaac-pick-orange"))
_ROBOT = RobotDescription.from_yaml(str(_REPO / "robots" / "so101_follower" / "robot.yaml"))

# LeIsaac SO101_FOLLOWER_USD_JOINT_LIMLITS / SO101_FOLLOWER_MOTOR_LIMITS (deg / motor units).
_USD_LIMITS = [(-110, 110), (-100, 100), (-100, 90), (-95, 95), (-160, 160), (-10, 100)]
_MOTOR_LIMITS = [(-100, 100)] * 5 + [(0, 100)]
# The Isaac scene's SO-101 jaw: normalized fraction f -> f * URDF upper (1.74533 rad).
_JAW_OPEN_RAD = 1.74533


def _leisaac_to_lerobot(joint_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    deg: npt.NDArray[np.float64] = np.degrees(joint_rad)
    out = np.empty_like(deg)
    for i, ((jl0, jl1), (ml0, ml1)) in enumerate(zip(_USD_LIMITS, _MOTOR_LIMITS, strict=True)):
        out[i] = (deg[i] - jl0) / (jl1 - jl0) * (ml1 - ml0) + ml0
    return out


def _robot_to_leisaac_rad(robot: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """openral robot units (arm rad, jaw fraction) -> LeIsaac joint radians."""
    out = robot.astype(np.float64).copy()
    out[5] = robot[5] * _JAW_OPEN_RAD
    return out


def test_calibration_matches_leisaac_state_map() -> None:
    codec = PolicyIOCodec.from_manifest(_MANIFEST, _ROBOT)
    assert not codec.is_identity
    for robot in (
        np.zeros(6),
        np.array([0.3, -1.2, 1.1, 0.9, -0.4, 0.6]),
        np.array([-1.9, 1.7, -1.7, -1.6, 2.7, 0.99]),
    ):
        got = codec.to_policy_state(robot.astype(np.float32))
        want = _leisaac_to_lerobot(_robot_to_leisaac_rad(robot))
        np.testing.assert_allclose(got, want, atol=2e-3)


def test_zero_pose_is_leisaac_init_in_motor_units() -> None:
    state = PolicyIOCodec.from_manifest(_MANIFEST, _ROBOT).to_policy_state(np.zeros(6))
    np.testing.assert_allclose(state, [0, 0, 100 / 19, 0, 0, 100 / 11], atol=1e-4)


def test_action_round_trips_through_the_calibration() -> None:
    codec = PolicyIOCodec.from_manifest(_MANIFEST, _ROBOT)
    robot = np.array([0.5, -1.0, 1.2, 0.3, -2.0, 0.4], dtype=np.float64)
    back = codec.to_robot_action(codec.to_policy_state(robot))
    np.testing.assert_allclose(back, robot, atol=1e-6)


def test_unknown_joint_is_a_config_error() -> None:
    ac = _MANIFEST.action_contract
    assert ac is not None and ac.joint_calibration is not None
    bad = ac.model_copy(
        update={"joint_calibration": {**ac.joint_calibration, "elbow": JointCalibration(scale=1.0)}}
    )
    with pytest.raises(ROSConfigError, match=r"names joints \['elbow'\]"):
        PolicyIOCodec.from_manifest(_MANIFEST.model_copy(update={"action_contract": bad}), _ROBOT)


def test_calibration_without_description_is_a_config_error() -> None:
    with pytest.raises(ROSConfigError, match="needs the robot description"):
        PolicyIOCodec.from_manifest(_MANIFEST, None)


@pytest.mark.parametrize("scale", [0.0, math.inf, math.nan])
def test_schema_rejects_degenerate_scale(scale: float) -> None:
    with pytest.raises(ValidationError, match="finite non-zero scale"):
        JointCalibration(scale=scale)


def test_schema_rejects_gripper_scale_alongside_calibration() -> None:
    with pytest.raises(ValidationError, match="both map the gripper"):
        ActionContract(
            dim=6,
            joint_units="degrees",
            gripper_scale=100.0,
            joint_calibration={"gripper": JointCalibration(scale=100.0)},
        )


def test_schema_rejects_empty_calibration() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        ActionContract(dim=6, joint_calibration={})


@given(
    scale=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6).filter(
        lambda v: v != 0.0
    ),
    offset=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
)
def test_joint_calibration_json_round_trip(scale: float, offset: float) -> None:
    cal = JointCalibration(scale=scale, offset=offset)
    assert JointCalibration.model_validate_json(cal.model_dump_json()) == cal
    assert "scale" in JointCalibration.model_json_schema()["properties"]


def test_isaac_scene_takes_the_calibrated_skill_and_the_runner_applies_it() -> None:
    assert SCENES.meta("isaac_sim").get("robot_unit_io") is True
    _check_policy_units_owned(_MANIFEST, _ROBOT, "isaac_sim")
    codec = _runner_policy_codec(_MANIFEST, _ROBOT, "isaac_sim")
    assert codec is not None and codec == PolicyIOCodec.from_manifest(_MANIFEST, _ROBOT)


def test_self_converting_scene_refuses_a_calibrated_skill() -> None:
    with pytest.raises(ROSConfigError, match="would ignore it"):
        _check_policy_units_owned(_MANIFEST, _ROBOT, "so101_box", {"joint_units": "degrees"})
    assert _runner_policy_codec(_MANIFEST, _ROBOT, "so101_box") is None


def test_leisaac_scene_videos_play_at_its_60hz_step_rate() -> None:
    from openral_core import SimEnvironment, SimScene, VLASpec, load_scene_strict
    from openral_sim.cli import _scene_control_rate_hz

    scene = load_scene_strict(str(_REPO / "scenes/sim/isaac_leisaac_pick_orange.yaml"), SimScene)
    env_cfg = SimEnvironment(
        robot_id="so101_follower",
        scene=scene.scene,
        task=scene.task,
        vla=VLASpec(id="zero", weights_uri="placeholder"),
    )
    assert _scene_control_rate_hz(env_cfg) == 60.0
    plain = env_cfg.model_copy(
        update={"scene": scene.scene.model_copy(update={"backend_options": {}})}
    )
    assert _scene_control_rate_hz(plain) == 30.0  # robots/so101_follower action_spec
