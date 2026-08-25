"""Unit tests for ``collision_params_from_description``.

Verifies the RobotDescription → kernel collision-param flattening: topological
link ordering (parent precedes child), per-link joint/dof/origin/axis arrays,
capsule lowering (capsule + sphere), and the allowed-collision matrix. Uses a
small synthetic chain whose geometry is fully controlled, plus the real openarm
fixture for shape sanity.

CLAUDE.md §1.11 — real schemas + the real ``robots/openarm`` fixture, no mocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
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
from openral_core.exceptions import ROSConfigError
from openral_safety.envelope_loader import (
    collision_params_from_description,
    ee_link_index_from_collision_params,
)


def _two_link_arm() -> RobotDescription:
    """base → link1 (revolute Z, +0.3 z) → link2 (revolute Y, +0.3 z); capsules on both."""
    return RobotDescription(
        name="synthetic_arm",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name="j1",
                joint_type=JointType.REVOLUTE,
                parent_link="base",
                child_link="link1",
                axis_xyz=(0.0, 0.0, 1.0),
                origin_xyz=(0.0, 0.0, 0.3),
            ),
            JointSpec(
                name="j2",
                joint_type=JointType.REVOLUTE,
                parent_link="link1",
                child_link="link2",
                axis_xyz=(0.0, 1.0, 0.0),
                origin_xyz=(0.0, 0.0, 0.3),
            ),
        ],
        collision_geometry=[
            LinkCollisionGeometry(
                link_name="link1", shape=CapsuleShape(radius_m=0.05, length_m=0.2)
            ),
            LinkCollisionGeometry(link_name="link2", shape=SphereShape(radius_m=0.04)),
        ],
        allowed_collision_pairs=[("link1", "link2")],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION], embodiment_tags=["synthetic"]
        ),
        safety=SafetyEnvelope(),
    )


@dataclass(frozen=True)
class _CylinderShape:
    """A fourth primitive kind — what a future ``CollisionShape`` member looks like.

    ``openral_core.CollisionShape`` is closed over sphere/capsule/box, so no
    fixture in ``robots/`` can produce this and no amount of validation will
    build one. It exists to reach the lowering's unknown-shape branch, the same
    way ``packages/openral_slam_bringup/test/test_depth_height_filter.py``
    reaches the height band's.

    It deliberately carries ``radius_m``: that is exactly the shape that used to
    slip through, because the old bare ``else`` read ``shape.radius_m`` off
    anything that was not a ``BoxShape``.
    """

    shape: str = "cylinder"
    radius_m: float = 0.1
    length_m: float = 0.4


def test_unknown_primitive_is_refused_not_lowered_as_a_capsule() -> None:
    """An unlowerable primitive raises instead of becoming an assumed capsule.

    Before this fix the routing was ``if BoxShape: ... else: <capsule>``, so a
    fourth variant was lowered as a zero-length capsule of its ``radius_m`` —
    silently, and in the unsafe direction: a capsule of radius ``r`` is
    contained in every non-spherical primitive carrying ``r``, so the kernel
    received a strictly smaller volume than the manifest declared. Refusing is
    at-least-as-conservative: no params are emitted, so no motion is authorised
    against a wrong envelope.
    """
    robot = _two_link_arm()
    # `model_construct` bypasses validation — the only way to hold a primitive
    # the closed union cannot express.
    bad = LinkCollisionGeometry.model_construct(
        link_name="link2",
        shape=_CylinderShape(),  # type: ignore[arg-type]  # reason: future-variant stand-in
        origin_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    robot = robot.model_copy(update={"collision_geometry": [robot.collision_geometry[0], bad]})

    with pytest.raises(ROSConfigError) as excinfo:
        collision_params_from_description(robot)

    message = str(excinfo.value)
    assert "cylinder" in message, message
    assert "link2" in message, message


def test_no_collision_geometry_disables_check() -> None:
    """A manifest without collision geometry yields the disabled sentinel."""
    robot = _two_link_arm().model_copy(update={"collision_geometry": []})
    params = collision_params_from_description(robot)
    assert params == {"self_collision_enabled": False}


def test_flatten_topological_order_and_arrays() -> None:
    """The chain lowers to parent-before-child arrays with correct dof/capsules."""
    params = collision_params_from_description(_two_link_arm())
    assert params["self_collision_enabled"] is True
    assert params["collision_n_links"] == 3  # base + link1 + link2
    names = params["collision_link_names"]
    assert names == ["base", "link1", "link2"]

    parent = params["collision_parent"]
    # Topological invariant: every parent index precedes its child.
    for child_idx, parent_idx in enumerate(parent):
        assert parent_idx < child_idx
    assert parent == [-1, 0, 1]

    # dof_index maps each moving link to its joint's position in robot.joints.
    assert params["collision_dof_index"] == [-1, 0, 1]
    assert params["collision_joint_kind"] == [0, 1, 1]  # fixed root, revolute, revolute

    # Capsules are a per-capsule list tagged with their link index: base (link 0)
    # has none, link1 a capsule, link2 a sphere (half_len 0).
    assert params["collision_capsule_link"] == [1, 2]
    assert params["collision_capsule_radius"] == [0.05, 0.04]
    assert params["collision_capsule_half_length"] == [0.1, 0.0]

    # allowed pair (link1, link2) → indices (1, 2).
    assert params["collision_allowed_pairs"] == [1, 2]

    # Joint origin of link1 is (0,0,0.3); arrays are 6*n_links / 3*n_links long.
    assert len(params["collision_origin_xyzrpy"]) == 18
    assert params["collision_origin_xyzrpy"][6:9] == [0.0, 0.0, 0.3]
    assert len(params["collision_axis"]) == 9


def test_openarm_fixture_lowers_to_well_formed_params() -> None:
    """The real openarm manifest produces shape-consistent collision params."""
    robot = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    params = collision_params_from_description(robot)
    assert params["self_collision_enabled"] is True
    n = params["collision_n_links"]
    assert n > 0
    assert len(params["collision_parent"]) == n
    assert len(params["collision_origin_xyzrpy"]) == 6 * n
    assert len(params["collision_axis"]) == 3 * n
    # Per-capsule arrays are parallel and tagged with valid link indices.
    n_caps = len(params["collision_capsule_link"])
    assert len(params["collision_capsule_radius"]) == n_caps
    assert len(params["collision_capsule_origin_xyzrpy"]) == 6 * n_caps
    assert all(0 <= li < n for li in params["collision_capsule_link"])
    # Parent-before-child holds for the lowered order.
    for child_idx, parent_idx in enumerate(params["collision_parent"]):
        assert parent_idx < child_idx


def test_ee_link_index_picks_deepest_link() -> None:
    """The EE control link is the kinematically deepest link."""
    params = collision_params_from_description(_two_link_arm())
    ee = ee_link_index_from_collision_params(params)
    parent = params["collision_parent"]

    # Deepest link: no other link has a longer chain to the root.
    def _depth(i: int) -> int:
        d = 0
        p = parent[i]
        while p is not None and p >= 0:
            d += 1
            p = parent[p]
        return d

    assert ee >= 0
    assert _depth(ee) == max(_depth(i) for i in range(len(parent)))


def test_ee_link_index_disabled_without_collision_model() -> None:
    """No collision model → -1 (predictive Cartesian stays off; reactive floor only)."""
    robot = _two_link_arm().model_copy(update={"collision_geometry": []})
    params = collision_params_from_description(robot)
    assert ee_link_index_from_collision_params(params) == -1


def test_ee_link_index_openarm_is_on_the_arm_chain() -> None:
    """On the real openarm manifest the EE proxy is a valid, non-root link."""
    robot = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    params = collision_params_from_description(robot)
    ee = ee_link_index_from_collision_params(params)
    assert 0 <= ee < params["collision_n_links"]
    # The deepest link is never the root (which has parent -1).
    assert params["collision_parent"][ee] >= 0
