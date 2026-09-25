# SPDX-License-Identifier: Apache-2.0
"""Per-joint starting-pose approach, per-robot joint-state staleness, per-skill rate.

Audit C (generalization) F1/F2/F3/F6. The OpenArm-derived values used to be one
scalar each, applied to every robot:

* F2 — the starting-pose ramp used one max-abs delta in "rad" across all joints.
  A prismatic gripper (Sawyer, ALOHA: 0-0.041 m stroke) always "arrived" under a
  0.05 tolerance, a gripper-only move was skipped, and the Franka gripper ramped
  at 5x its ``velocity_limit``. Now each joint has its own speed
  (``min(declared, velocity_limit)``) and tolerance, in its own units.
* F3 — the runner's joint-state window was a launch constant. It is now
  ``safety.joint_state_staleness_limit_s``, read by ``build_hal`` and the launch.
* F1 — a skill can declare the rate it was trained at; the runner refuses a
  mismatch instead of silently executing at the robot's rate.

Real manifests from ``robots/`` only; no ROS needed.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import JointType, RobotDescription
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
runner = importlib.import_module("openral_rskill_ros.rskill_runner_node")

REAL_ROBOTS = [
    "aloha_bimanual",
    "franka_panda",
    "galaxea_a1",
    "openarm",
    "sawyer",
    "so100_follower",
    "so101_follower",
    "ur5e",
    "ur10e",
]
# The value every real manifest used before this change, in whatever unit.
_OLD_SPEED, _OLD_TOLERANCE = 0.5, 0.05


def _robot(name: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(REPO_ROOT / "robots" / name / "robot.yaml"))


def _index(desc: RobotDescription, joint: str) -> int:
    return [j.name for j in desc.joints].index(joint)


# ── F2: per-joint starting-pose approach ──────────────────────────────────────


@pytest.mark.parametrize("robot", REAL_ROBOTS)
def test_every_joint_is_at_least_as_conservative_as_the_old_scalar(robot: str) -> None:
    """No joint ramps faster, or counts as arrived from further away, than before."""
    desc = _robot(robot)
    bounds = runner.starting_pose_joint_bounds(0.0, 0.0, desc)
    assert len(bounds) == len(desc.joints)
    for joint, (speed, tolerance) in zip(desc.joints, bounds, strict=True):
        assert 0.0 < speed <= _OLD_SPEED, joint.name
        assert 0.0 < tolerance <= _OLD_TOLERANCE, joint.name
        if joint.velocity_limit is not None:
            assert speed <= joint.velocity_limit, joint.name
        if joint.joint_type is JointType.PRISMATIC and joint.position_limits is not None:
            stroke = joint.position_limits[1] - joint.position_limits[0]
            assert tolerance < stroke / 2, f"{joint.name}: tolerance swallows the stroke"


@pytest.mark.parametrize(
    ("robot", "gripper"),
    [("sawyer", "right_gripper"), ("aloha_bimanual", "left_gripper")],
)
def test_a_prismatic_gripper_away_from_its_pose_is_not_arrived(robot: str, gripper: str) -> None:
    """0.03 m off on a 0.041 m stroke used to be within the 0.05 'rad' tolerance."""
    desc = _robot(robot)
    bounds = runner.starting_pose_joint_bounds(0.0, 0.0, desc)
    target = [0.0] * len(desc.joints)
    current = list(target)
    current[_index(desc, gripper)] = 0.03
    assert not runner.starting_pose_reached(current, target, bounds)
    # ...and a gripper-only difference is ramped, not skipped.
    assert runner.starting_pose_ramp_steps(current, target, bounds, 30.0) > 0
    assert runner.starting_pose_reached(target, target, bounds)
    assert runner.starting_pose_ramp_steps(target, target, bounds, 30.0) == 0


def test_the_franka_gripper_ramps_within_its_velocity_limit() -> None:
    """A gripper-dominated move used to run at 0.5/s against a 0.1/s limit."""
    desc = _robot("franka_panda")
    bounds = runner.starting_pose_joint_bounds(0.0, 0.0, desc)
    i = _index(desc, "panda_gripper")
    limit = desc.joints[i].velocity_limit
    assert limit is not None
    current = [0.0] * len(desc.joints)
    target = list(current)
    target[i] = 1.0
    rate = 30.0
    steps = runner.starting_pose_ramp_steps(current, target, bounds, rate)
    assert 1.0 / (steps / rate) <= limit + 1e-9
    assert steps >= math.ceil(1.0 / _OLD_SPEED * rate)


def test_every_joint_respects_its_own_cap_on_a_mixed_move() -> None:
    """A slow arm joint and a fast gripper in one move: the slowest-to-finish sets the ramp."""
    desc = _robot("sawyer")
    bounds = runner.starting_pose_joint_bounds(0.0, 0.0, desc)
    current = [0.0] * len(desc.joints)
    target = [0.2] * (len(desc.joints) - 1) + [0.041]
    rate = 30.0
    steps = runner.starting_pose_ramp_steps(current, target, bounds, rate)
    duration = steps / rate
    for (speed, _), start, end in zip(bounds, current, target, strict=True):
        assert abs(end - start) / duration <= speed + 1e-9


def test_revolute_params_override_only_revolute_joints() -> None:
    """The runner's rad params (SO-100 twin tests) never leak onto a prismatic joint."""
    desc = _robot("aloha_bimanual")
    bounds = runner.starting_pose_joint_bounds(0.2, 0.01, desc)
    for joint, (speed, tolerance) in zip(desc.joints, bounds, strict=True):
        if joint.joint_type is JointType.PRISMATIC:
            assert (speed, tolerance) == (
                min(desc.safety.starting_pose_max_joint_speed_m_s or 0.0, joint.velocity_limit),
                desc.safety.starting_pose_tolerance_m,
            )
        else:
            assert (speed, tolerance) == (0.2, 0.01)


def test_an_undeclared_approach_is_refused_by_joint_type() -> None:
    desc = _robot("sawyer")
    no_linear = desc.model_copy(
        update={
            "safety": desc.safety.model_copy(
                update={
                    "starting_pose_max_joint_speed_m_s": None,
                    "starting_pose_tolerance_m": None,
                }
            )
        }
    )
    with pytest.raises(ROSConfigError, match="starting_pose_max_joint_speed_m_s"):
        runner.starting_pose_joint_bounds(0.0, 0.0, no_linear)
    with pytest.raises(ROSConfigError):
        runner.starting_pose_joint_bounds(0.0, 0.0, None)


def test_a_real_manifest_with_a_prismatic_joint_must_declare_the_linear_approach() -> None:
    raw = _robot("sawyer").model_dump(mode="json", exclude_unset=True)
    del raw["safety"]["starting_pose_tolerance_m"]
    with pytest.raises(ValidationError, match=r"safety\.starting_pose_tolerance_m"):
        RobotDescription.model_validate(raw)


def test_a_revolute_only_real_manifest_needs_no_linear_approach() -> None:
    desc = _robot("ur5e")
    assert desc.safety.starting_pose_tolerance_m is None
    assert all(j.joint_type is not JointType.PRISMATIC for j in desc.joints)


# ── F3: joint-state staleness is a typed manifest field ───────────────────────


@pytest.mark.parametrize("robot", REAL_ROBOTS)
def test_every_real_robot_declares_its_staleness_in_one_place(robot: str) -> None:
    desc = _robot(robot)
    assert desc.safety.joint_state_staleness_limit_s is not None
    assert "staleness_limit_s" not in desc.hal.parameters.defaults


def test_the_measured_openarm_window_and_the_other_robots_keep_their_values() -> None:
    """Values carried over exactly: OpenArm measured, the rest today's effective value."""
    expected = {
        "openarm": 0.1,
        "ur5e": 0.5,
        "ur10e": 0.5,
        "franka_panda": 0.2,
        "sawyer": 0.2,
        "aloha_bimanual": 0.2,
        "so100_follower": 0.5,
        "so101_follower": 0.5,
        "galaxea_a1": 0.5,
    }
    got = {r: _robot(r).safety.joint_state_staleness_limit_s for r in expected}
    assert got == expected


def test_a_real_manifest_without_staleness_is_refused() -> None:
    raw = _robot("franka_panda").model_dump(mode="json", exclude_unset=True)
    del raw["safety"]["joint_state_staleness_limit_s"]
    with pytest.raises(ValidationError, match=r"safety\.joint_state_staleness_limit_s"):
        RobotDescription.model_validate(raw)


def test_staleness_in_two_places_is_refused() -> None:
    raw = _robot("ur5e").model_dump(mode="json", exclude_unset=True)
    raw["hal"].setdefault("parameters", {}).setdefault("defaults", {})["staleness_limit_s"] = 0.5
    with pytest.raises(ValidationError, match="both"):
        RobotDescription.model_validate(raw)


@pytest.mark.parametrize("robot", ["ur5e", "franka_panda", "sawyer", "openarm"])
def test_build_hal_threads_the_manifest_staleness_into_the_real_hal(robot: str) -> None:
    from openral_hal import build_hal

    desc = _robot(robot)
    hal = build_hal(desc, mode="real", transport={"require_can_links": False})
    assert hal._staleness_limit_s == desc.safety.joint_state_staleness_limit_s  # type: ignore[attr-defined]  # reason: private field under test


def test_a_real_hal_built_directly_reads_its_description() -> None:
    """No constructor default: the ALOHA HAL used to hard-code 0.2 in code only."""
    from openral_hal.aloha import AlohaHAL
    from openral_hal.sawyer_real import SawyerRealHAL

    aloha = _robot("aloha_bimanual")
    assert AlohaHAL(description=aloha)._staleness_limit_s == 0.2
    sawyer = _robot("sawyer")
    assert SawyerRealHAL(description=sawyer)._staleness_limit_s == 0.2


def test_a_real_hal_with_no_staleness_anywhere_is_refused() -> None:
    from openral_hal.franka_panda_real import FrankaPandaRealHAL

    desc = _robot("franka_panda")
    bare = desc.model_copy(
        update={"safety": desc.safety.model_copy(update={"joint_state_staleness_limit_s": None})}
    )
    with pytest.raises(ROSConfigError, match="joint_state_staleness_limit_s"):
        FrankaPandaRealHAL(description=bare)


# ── F1: a skill's trained rate must match the robot's tick ────────────────────


def test_a_skill_trained_at_another_rate_is_refused() -> None:
    with pytest.raises(ROSConfigError, match="15"):
        runner.check_skill_control_rate(15.0, 30.0, skill_id="droid_policy", robot="franka_panda")


def test_a_matching_or_undeclared_skill_rate_passes() -> None:
    runner.check_skill_control_rate(30.0, 30.0, skill_id="s", robot="ur5e")
    runner.check_skill_control_rate(None, 50.0, skill_id="s", robot="aloha_bimanual")


def test_the_action_contract_carries_an_optional_trained_rate() -> None:
    from openral_core.schemas import ActionContract

    assert ActionContract(dim=7).control_freq_hz is None
    assert ActionContract(dim=7, control_freq_hz=15.0).control_freq_hz == 15.0
    with pytest.raises(ValidationError):
        ActionContract(dim=7, control_freq_hz=0.0)


@pytest.mark.parametrize(
    ("param", "robot", "source"),
    [(0.0, "ur5e", "manifest"), (15.0, "ur5e", "rate_hz_param"), (0.0, "widowx", "fallback")],
)
def test_the_tick_rate_source_is_named(param: float, robot: str, source: str) -> None:
    """A robot with no declared rate ticks at the 30 Hz fallback, and says so."""
    rate, got = runner.resolve_control_rate_source(param, _robot(robot))
    assert got == source
    assert rate == {"manifest": 30.0, "rate_hz_param": 15.0, "fallback": 30.0}[source]


# ── F3: the launch hands the runner the manifest's window in real mode ────────


@pytest.fixture(scope="module")
def launch_module() -> object:
    """The real launch file, loaded by path — it is not an importable package."""
    import importlib.util
    import sys

    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    path = REPO_ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_staleness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_staleness"] = module
    spec.loader.exec_module(module)
    return module


def _republished(desc: RobotDescription) -> bool:
    """Whether a real deploy routes the runner to the HAL's ``~/joint_states`` republish."""
    from openral_hal.resolver import hal_joint_states_topic

    node = f"openral_hal_{desc.name}"
    return hal_joint_states_topic(desc, mode="real", hal_node_name=node) == f"/{node}/joint_states"


@pytest.mark.parametrize(("robot", "window"), [("openarm", 0.1), ("aloha_bimanual", 0.2)])
def test_the_launch_uses_the_manifest_window_on_the_raw_stream(
    launch_module: object, robot: str, window: float
) -> None:
    fn = launch_module._runner_joint_state_staleness_s  # type: ignore[attr-defined]
    assert fn(_robot(robot), "real", republished=False) == window
    assert fn(_robot(robot), "sim", republished=False) == 0.5


@pytest.mark.parametrize(
    ("robot", "window"),
    [
        # 0.1 s measured on the raw 750 Hz stream + two 33.3 ms republish periods.
        ("openarm", 0.1 + 2 / 30.0),
        ("franka_panda", 0.2 + 2 / 30.0),
        # 0.5 + 2/30 would exceed the former shipped runner window; capped there.
        ("ur5e", 0.5),
        # Serial / interbotix HALs publish /joint_states themselves: not republished.
        ("so100_follower", 0.5),
        ("aloha_bimanual", 0.2),
    ],
)
def test_the_runner_window_adds_two_republish_periods_on_the_hal_republish(
    launch_module: object, robot: str, window: float
) -> None:
    fn = launch_module._runner_joint_state_staleness_s  # type: ignore[attr-defined]
    desc = _robot(robot)
    assert math.isclose(fn(desc, "real", republished=_republished(desc)), window)


def test_the_hal_keeps_the_manifest_window_on_the_republish_path() -> None:
    """Only the runner widens: the HAL still checks its raw source at the manifest value."""
    from openral_hal import build_hal

    desc = _robot("openarm")
    assert _republished(desc)
    hal = build_hal(desc, mode="real", transport={"require_can_links": False})
    assert hal._staleness_limit_s == 0.1  # type: ignore[attr-defined]  # reason: private field under test


def test_no_real_runner_window_is_looser_than_the_former_shipped_value(
    launch_module: object,
) -> None:
    fn = launch_module._runner_joint_state_staleness_s  # type: ignore[attr-defined]
    real = [
        RobotDescription.from_yaml(p)
        for p in sorted((REPO_ROOT / "robots").glob("*/robot.yaml"))
        if RobotDescription.from_yaml(p).hal.real is not None
    ]
    assert len(real) >= 5
    for desc in real:
        assert fn(desc, "real", republished=_republished(desc)) <= 0.5, desc.name


def test_a_republished_runner_without_a_control_rate_is_refused(launch_module: object) -> None:
    fn = launch_module._runner_joint_state_staleness_s  # type: ignore[attr-defined]
    desc = _robot("ur5e")
    assert desc.action_spec is not None
    bare = desc.model_copy(
        update={"action_spec": desc.action_spec.model_copy(update={"control_freq_hz": None})}
    )
    with pytest.raises(ROSConfigError, match="control_freq_hz"):
        fn(bare, "real", republished=True)
