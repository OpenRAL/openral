"""Tests for the cuRobo robot-config emitter (ADR-0065 D4).

The emitter reuses the capsule/sphere geometry the safety kernel already lowers
(ADR-0030) so plan-time (cuMotion) and kernel-time collision geometry share one
source of truth. cuRobo represents link collision volumes as **spheres**, so the
core transformation samples spheres along each lowered capsule's segment.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from openral_core import (
    CapsuleShape,
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    LinkCollisionGeometry,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SphereShape,
)
from openral_safety.cumotion_config import (
    actuated_joint_names,
    capsule_to_spheres,
    link_collision_spheres,
    render_cumotion_config,
    spheres_for_capsule,
)
from openral_safety.urdf_lowering import LoweredCollisionModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def _two_link_robot() -> RobotDescription:
    return RobotDescription(
        name="emit_test",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        base_frame="base_link",
        joints=[
            JointSpec(
                name="j1",
                joint_type=JointType.REVOLUTE,
                parent_link="base_link",
                child_link="link_a",
            ),
            JointSpec(
                name="j2",
                joint_type=JointType.REVOLUTE,
                parent_link="link_a",
                child_link="link_b",
            ),
            JointSpec(
                name="weld",
                joint_type=JointType.FIXED,
                parent_link="link_b",
                child_link="tool",
            ),
        ],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION],
            embodiment_tags=["emit_test"],
        ),
        safety=SafetyEnvelope(),
    )


class TestCapsuleToSpheres:
    def test_three_spheres_along_x_axis_include_endpoints(self) -> None:
        spheres = capsule_to_spheres((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.1, count=3)
        assert [s.center for s in spheres] == [
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ]
        assert all(s.radius == 0.1 for s in spheres)

    def test_count_two_is_just_the_endpoints(self) -> None:
        spheres = capsule_to_spheres((0.0, 0.0, 0.0), (2.0, 4.0, 0.0), 0.2, count=2)
        assert [s.center for s in spheres] == [(0.0, 0.0, 0.0), (2.0, 4.0, 0.0)]

    def test_count_one_is_the_midpoint(self) -> None:
        spheres = capsule_to_spheres((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.3, count=1)
        assert len(spheres) == 1
        assert spheres[0].center == (0.5, 0.0, 0.0)
        assert spheres[0].radius == 0.3

    def test_count_preserved_and_radius_applied(self) -> None:
        spheres = capsule_to_spheres((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.05, count=5)
        assert len(spheres) == 5
        assert all(s.radius == 0.05 for s in spheres)


class TestSpheresForCapsule:
    def test_sphere_shape_yields_a_single_sphere(self) -> None:
        # A zero-length capsule (sphere) needs exactly one sphere.
        assert spheres_for_capsule(SphereShape(radius_m=0.05)) == 1

    def test_long_capsule_spacing_does_not_exceed_radius(self) -> None:
        # length 0.30, radius 0.05 -> need centres <= 0.05 apart -> >= 7 spheres.
        n = spheres_for_capsule(CapsuleShape(radius_m=0.05, length_m=0.30))
        assert n >= 7
        # spacing = length / (n - 1) must cover the segment within one radius.
        assert 0.30 / (n - 1) <= 0.05 + 1e-9

    def test_short_capsule_gets_at_least_two_spheres(self) -> None:
        n = spheres_for_capsule(CapsuleShape(radius_m=0.10, length_m=0.01))
        assert n >= 2


class TestLinkCollisionSpheres:
    def test_capsule_link_lowered_to_spheres_in_link_frame(self) -> None:
        # Capsule along local +Z, length 0.2, centred at origin -> endpoints at
        # z = -0.1 and z = +0.1 in the link frame.
        geom = LinkCollisionGeometry(
            link_name="link_1",
            shape=CapsuleShape(radius_m=0.04, length_m=0.2),
        )
        spheres = link_collision_spheres(geom)
        assert len(spheres) >= 2
        zs = [round(s.center[2], 6) for s in spheres]
        assert min(zs) == -0.1
        assert max(zs) == 0.1
        assert all(s.radius == 0.04 for s in spheres)

    def test_sphere_link_lowered_to_one_sphere_at_origin(self) -> None:
        geom = LinkCollisionGeometry(
            link_name="wrist",
            shape=SphereShape(radius_m=0.06),
            origin_xyz_rpy=(0.1, 0.2, 0.3, 0.0, 0.0, 0.0),
        )
        spheres = link_collision_spheres(geom)
        assert len(spheres) == 1
        assert spheres[0].center == (0.1, 0.2, 0.3)
        assert spheres[0].radius == 0.06


class TestActuatedJointNames:
    def test_fixed_joints_excluded_movable_kept_in_order(self) -> None:
        assert actuated_joint_names(_two_link_robot()) == ["j1", "j2"]


class TestRenderCuMotionConfig:
    def _model(self) -> LoweredCollisionModel:
        return LoweredCollisionModel(
            collision_geometry=[
                LinkCollisionGeometry(
                    link_name="link_a",
                    shape=CapsuleShape(radius_m=0.04, length_m=0.2),
                ),
                LinkCollisionGeometry(
                    link_name="link_b",
                    shape=SphereShape(radius_m=0.05),
                ),
            ],
            allowed_collision_pairs=[("link_a", "link_b")],
            acm_source="srdf",
        )

    def test_emits_loadable_yaml_with_curobo_structure(self) -> None:
        text = render_cumotion_config(_two_link_robot(), self._model())
        assert "do not hand-edit" in text
        doc = yaml.safe_load(text)
        kin = doc["robot_cfg"]["kinematics"]
        assert kin["base_link"] == "base_link"
        # Both links carry collision spheres, each a {center, radius} entry.
        spheres = kin["collision_spheres"]
        assert set(spheres) == {"link_a", "link_b"}
        first = spheres["link_a"][0]
        assert "center" in first and "radius" in first and len(first["center"]) == 3
        assert spheres["link_b"] == [{"center": [0.0, 0.0, 0.0], "radius": 0.05}]
        # cspace plans the actuated joints only.
        assert kin["cspace"]["joint_names"] == ["j1", "j2"]

    def test_self_collision_ignore_is_symmetric_from_acm(self) -> None:
        doc = yaml.safe_load(render_cumotion_config(_two_link_robot(), self._model()))
        ignore = doc["robot_cfg"]["kinematics"]["self_collision_ignore"]
        assert ignore["link_a"] == ["link_b"]
        assert ignore["link_b"] == ["link_a"]

    def test_real_franka_panda_fixture_round_trips_all_links(self) -> None:
        desc = RobotDescription.from_yaml(str(REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"))
        model = LoweredCollisionModel(
            collision_geometry=list(desc.collision_geometry),
            allowed_collision_pairs=list(desc.allowed_collision_pairs),
            acm_source="srdf",
        )
        doc = yaml.safe_load(render_cumotion_config(desc, model))
        kin = doc["robot_cfg"]["kinematics"]
        # Every lowered link appears in the cuRobo collision-sphere map.
        assert set(kin["collision_spheres"]) == {g.link_name for g in desc.collision_geometry}
        # The 7 Panda arm joints are planned, in manifest order.
        for j in (f"panda_joint{i}" for i in range(1, 8)):
            assert j in kin["cspace"]["joint_names"]
