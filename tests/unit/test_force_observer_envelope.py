"""Issue #373 — the external-torque observer's manifest → kernel path.

Real manifests only (CLAUDE.md §1.11): the OpenArm's lowered ``force_observer``
block and the SO-100 (which has none). Covers the schema's refusals, the
per-joint ceiling algebra (a deploy scene or rSkill may add or lower a joint's
ceiling, never raise or drop one), and the kernel parameters the loader emits.
"""

from __future__ import annotations

import pathlib

import pytest
from openral_core import (
    ForceEvidence,
    ForceObserverSpec,
    LinkInertial,
    RobotDescription,
    RSkillManifest,
    SafetyEnvelope,
)
from openral_core.exceptions import ROSConfigError
from openral_safety.envelope_loader import (
    collision_params_from_description,
    compute_intersection,
    kernel_params_from_envelope,
    merge_deploy_envelope,
)
from pydantic import TypeAdapter, ValidationError

from tests.unit.test_envelope_loader import _skill_with_envelope

_ROBOTS = pathlib.Path(__file__).resolve().parents[2] / "robots"


def _openarm() -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))


def _so100() -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROBOTS / "so100_follower" / "robot.yaml"))


def _with_ceilings(robot: RobotDescription, ceilings: dict[str, float]) -> RobotDescription:
    data = robot.model_dump()
    data["safety"]["max_external_joint_torque_nm"] = ceilings
    return RobotDescription.model_validate(data)


def test_openarm_ships_a_model_but_no_ceiling() -> None:
    """The real cell runs the observer for disclosure only until the load-cell go/no-go."""
    robot = _openarm()
    assert robot.force_observer is not None
    assert robot.safety.max_external_joint_torque_nm == {}
    intersection = compute_intersection(robot, skill=None)
    assert intersection.joint_external_torque_max == ()
    assert "joint_external_torque_max" not in kernel_params_from_envelope(intersection)


def test_observer_params_are_parallel_to_the_collision_links() -> None:
    robot = _openarm()
    params = collision_params_from_description(robot)
    names = params["collision_link_names"]
    assert isinstance(names, list)
    n = len(names)
    assert params["force_observer_enabled"] is True
    for key, width in (
        ("force_observer_link_mass", 1),
        ("force_observer_link_com", 3),
        ("force_observer_link_inertia", 6),
        ("force_observer_joint_armature", 1),
        ("force_observer_joint_viscous", 1),
        ("force_observer_joint_coulomb", 1),
        ("force_observer_joint_offset", 1),
    ):
        value = params[key]
        assert isinstance(value, list)
        assert len(value) == width * n, key
    mass = params["force_observer_link_mass"]
    armature = params["force_observer_joint_armature"]
    assert isinstance(mass, list)
    assert isinstance(armature, list)
    assert robot.force_observer is not None
    by_link = {x.link: x.mass_kg for x in robot.force_observer.links}
    by_joint = {x.joint: x.armature_kgm2 for x in robot.force_observer.joints}
    for i, name in enumerate(names):
        assert mass[i] == by_link.get(name, 0.0), name
    # Joint terms describe the joint INTO the link: right_joint4 drives link4.
    assert armature[names.index("openarm_right_link4")] == by_joint["right_joint4"]
    # A mimic edge (the follower finger) and a rigid mount carry no joint terms.
    assert armature[names.index("openarm_right_finger_follower")] == 0.0
    assert armature[names.index("openarm_right_link0")] == 0.0
    # Robots without the block emit nothing.
    assert not any(
        k.startswith("force_observer") for k in collision_params_from_description(_so100())
    )


def test_a_deploy_ceiling_reaches_the_kernel_in_joint_order() -> None:
    robot = _openarm()
    deploy = SafetyEnvelope(max_external_joint_torque_nm={"right_joint2": 6.0})
    intersection = compute_intersection(robot, skill=None, deploy=deploy)
    names = [j.name for j in robot.joints]
    expected = tuple(6.0 if n == "right_joint2" else 0.0 for n in names)
    assert intersection.joint_external_torque_max == expected
    assert kernel_params_from_envelope(intersection)["joint_external_torque_max"] == list(expected)


def test_a_deploy_scene_adds_or_lowers_ceilings_but_never_drops_or_raises_one() -> None:
    robot = _with_ceilings(_openarm(), {"right_joint1": 8.0, "right_joint2": 8.0})
    merged = merge_deploy_envelope(
        robot.safety,
        SafetyEnvelope(max_external_joint_torque_nm={"right_joint2": 5.0, "left_joint1": 4.0}),
    )
    assert merged.max_external_joint_torque_nm == {
        "right_joint1": 8.0,  # kept although the deploy scene does not mention it
        "right_joint2": 5.0,
        "left_joint1": 4.0,
    }
    with pytest.raises(ROSConfigError, match="right_joint1"):
        merge_deploy_envelope(
            robot.safety, SafetyEnvelope(max_external_joint_torque_nm={"right_joint1": 9.0})
        )


def test_an_rskill_may_only_tighten_the_ceilings() -> None:
    robot = _with_ceilings(_openarm(), {"right_joint1": 8.0})
    skill = _skill_with_envelope({"max_external_joint_torque_nm": {"right_joint3": 3.0}})
    intersection = compute_intersection(robot, skill)
    names = [j.name for j in robot.joints]
    assert intersection.joint_external_torque_max[names.index("right_joint1")] == 8.0
    assert intersection.joint_external_torque_max[names.index("right_joint3")] == 3.0
    looser: RSkillManifest = _skill_with_envelope(
        {"max_external_joint_torque_nm": {"right_joint1": 12.0}}
    )
    with pytest.raises(ROSConfigError, match="loosening rejected"):
        compute_intersection(robot, looser)


def test_a_ceiling_nothing_can_enforce_is_refused() -> None:
    # No observer model on the SO-100: a deploy-scene ceiling would be inert.
    so100 = _so100()
    joint = so100.joints[0].name
    with pytest.raises(ROSConfigError, match="force_observer"):
        compute_intersection(
            so100, None, deploy=SafetyEnvelope(max_external_joint_torque_nm={joint: 2.0})
        )
    with pytest.raises(ROSConfigError, match="not actuated joints"):
        compute_intersection(
            _openarm(), None, deploy=SafetyEnvelope(max_external_joint_torque_nm={"nope": 2.0})
        )
    # And in the manifest itself.
    with pytest.raises(ValidationError, match="needs force_observer"):
        _with_ceilings(so100, {joint: 2.0})
    with pytest.raises(ValidationError, match="unknown joints"):
        _with_ceilings(_openarm(), {"nope": 2.0})


def test_the_model_must_name_real_links_and_joints() -> None:
    robot = _openarm()
    assert robot.force_observer is not None
    data = robot.model_dump()
    data["force_observer"]["links"].append(LinkInertial(link="nowhere", mass_kg=1.0).model_dump())
    with pytest.raises(ValidationError, match="outside the tree"):
        RobotDescription.model_validate(data)
    with pytest.raises(ValidationError, match="more than once"):
        ForceObserverSpec(
            source="test",
            links=[LinkInertial(link="a", mass_kg=1.0), LinkInertial(link="a", mass_kg=2.0)],
        )
    with pytest.raises(ValidationError):
        LinkInertial(link="a", mass_kg=-1.0)


def test_the_kernels_force_evidence_parses_as_a_joint_torque() -> None:
    """The exact shape ``trip_force_limit`` publishes (lifecycle_kernel.cpp)."""
    from openral_core import FailureEvidence

    raw = '{"kind":"force","joint_or_ee":"right_joint2","measured_n":6.5,"limit_n":6,"unit":"Nm"}'
    evidence: object = TypeAdapter(FailureEvidence).validate_json(raw)
    assert isinstance(evidence, ForceEvidence)
    assert evidence.unit == "Nm"
    assert ForceEvidence(joint_or_ee="ee", measured_n=1.0, limit_n=2.0).unit == "N"
