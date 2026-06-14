"""Contract tests for the ADR-0030 geometric-safety schemas.

Covers the typed surface added for self/world-collision checking: the
``CollisionShape`` discriminated union, ``LinkCollisionGeometry`` /
``WorldCollisionPrimitive`` / ``OccupancyGridRef``, the ``CollisionEvidence``
``FailureEvidence`` variant, and the real ``robots/openarm/robot.yaml``
fixture carrying capsule/sphere link geometry + an allowed-collision matrix.

CLAUDE.md §1.11 — real schemas, real fixture under ``robots/openarm/``, no
mocks.
"""

from __future__ import annotations

from openral_core import (
    CapsuleShape,
    CollisionEvidence,
    FailureEvidence,
    LinkCollisionGeometry,
    OccupancyGridRef,
    Pose6D,
    RobotDescription,
    SphereShape,
    WorldCollisionPrimitive,
    WorldState,
)
from openral_core.schemas import JointState
from pydantic import TypeAdapter, ValidationError

_OPENARM_YAML = "robots/openarm/robot.yaml"


# ── CollisionShape discriminated union ────────────────────────────────────────


def test_collision_shape_discriminates_capsule_vs_sphere() -> None:
    """The ``shape`` discriminator routes embedded dicts to the right model."""
    capsule = LinkCollisionGeometry.model_validate(
        {"link_name": "link_1", "shape": {"shape": "capsule", "radius_m": 0.04, "length_m": 0.3}}
    )
    sphere = LinkCollisionGeometry.model_validate(
        {"link_name": "finger", "shape": {"shape": "sphere", "radius_m": 0.05}}
    )
    assert isinstance(capsule.shape, CapsuleShape)
    assert isinstance(sphere.shape, SphereShape)
    assert capsule.origin_xyz_rpy == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_capsule_rejects_nonpositive_radius() -> None:
    """``radius_m`` is constrained ``> 0`` (CLAUDE.md §1.3 — types are the contract)."""
    try:
        CapsuleShape(radius_m=0.0, length_m=0.1)
    except ValidationError:
        pass
    else:  # pragma: no cover - the constraint must fire
        raise AssertionError("CapsuleShape accepted radius_m=0.0")


# ── CollisionEvidence through the FailureEvidence union ───────────────────────


def test_collision_evidence_dispatches_through_failure_union() -> None:
    """A ``kind="collision"`` payload decodes to ``CollisionEvidence`` (ADR-0030)."""
    ev = CollisionEvidence(
        collision_kind="self",
        link_a="openarm_left_link3",
        link_b_or_object="openarm_right_link3",
        horizon_step=2,
        min_distance_m=-0.01,
    )
    decoded = TypeAdapter(FailureEvidence).validate_json(ev.model_dump_json())
    assert isinstance(decoded, CollisionEvidence)
    assert decoded.collision_kind == "self"
    assert decoded.min_distance_m == -0.01


# ── WorldState world surface defaults ─────────────────────────────────────────


def test_world_state_world_surface_defaults_empty() -> None:
    """A WorldState with no obstacles has an empty/absent world surface."""
    ws = WorldState(stamp_ns=0, joint_state=JointState(name=["j1"], position=[0.0], stamp_ns=0))
    assert ws.collision_primitives == []
    assert ws.occupancy_grid is None


def test_world_collision_primitive_and_occupancy_grid_validate() -> None:
    """A placed obstacle and an occupancy-grid reference validate against schema."""
    origin = Pose6D(xyz=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map")
    obstacle = WorldCollisionPrimitive(
        shape=SphereShape(radius_m=0.1),
        pose=Pose6D(xyz=(0.5, 0.0, 0.2), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map"),
        object_id="mug-7",
    )
    grid = OccupancyGridRef(
        frame_id="map",
        resolution_m=0.05,
        width=200,
        height=200,
        origin=origin,
        data_topic="/map",
    )
    assert obstacle.object_id == "mug-7"
    assert grid.width == 200


# ── Real openarm fixture ──────────────────────────────────────────────────────


def test_openarm_fixture_loads_collision_geometry() -> None:
    """``robots/openarm/robot.yaml`` parses its capsule/sphere link geometry."""
    desc = RobotDescription.from_yaml(_OPENARM_YAML)

    by_link = {g.link_name: g.shape for g in desc.collision_geometry}
    # Every collision-geometry link names a real link in the kinematic chain
    # (the lowering tool / authoring contract: no orphan geometry).
    chain_links = {j.parent_link for j in desc.joints} | {j.child_link for j in desc.joints}
    assert set(by_link).issubset(chain_links)

    assert isinstance(by_link["openarm_left_link3"], CapsuleShape)
    assert isinstance(by_link["openarm_left_finger_pair"], SphereShape)
    assert by_link["openarm_left_link3"].radius_m > 0.0


def test_openarm_allowed_collision_matrix_excludes_adjacent_not_cross_arm() -> None:
    """Adjacent links are allowed to touch; the two arms are not (ADR-0030)."""
    desc = RobotDescription.from_yaml(_OPENARM_YAML)
    pairs = {frozenset(p) for p in desc.allowed_collision_pairs}

    # Adjacent within an arm → excluded from self-collision.
    assert frozenset({"openarm_left_link1", "openarm_left_link2"}) in pairs
    # Cross-arm → deliberately NOT excluded, so a left-vs-right collision is caught.
    assert frozenset({"openarm_left_link3", "openarm_right_link3"}) not in pairs


# The authoritative Franka SRDF `disable_collisions` among arm links 1-7, from
# moveit_resources_panda_moveit_config/config/panda.srdf. Stable, canonical spec
# (Adjacent = directly connected; Never = proven not-in-collision across MoveIt's
# random-pose sweep). Embedded rather than parsed from /opt/ros so the test is
# self-contained — it's a known robot specification, not a fixture under test.
_PANDA_SRDF_ARM_DISABLES = frozenset(
    frozenset(p)
    for p in (
        ("panda_link1", "panda_link2"),  # Adjacent
        ("panda_link2", "panda_link3"),  # Adjacent
        ("panda_link3", "panda_link4"),  # Adjacent
        ("panda_link4", "panda_link5"),  # Adjacent
        ("panda_link5", "panda_link6"),  # Adjacent
        ("panda_link6", "panda_link7"),  # Adjacent
        ("panda_link1", "panda_link3"),  # Never
        ("panda_link1", "panda_link4"),  # Never
        ("panda_link2", "panda_link4"),  # Never
        ("panda_link2", "panda_link6"),  # Never
        ("panda_link3", "panda_link5"),  # Never
        ("panda_link3", "panda_link6"),  # Never
        ("panda_link3", "panda_link7"),  # Never
        ("panda_link4", "panda_link6"),  # Never
        ("panda_link4", "panda_link7"),  # Never
    )
)
# Pairs our capsule model allows that the SRDF mesh model does NOT disable —
# documented capsule-junction artifacts (link6 is a short 0.088 m capsule, so
# link5↔link7 always overlap under capsule conservatism). Any OTHER divergence
# from the SRDF is a bug.
_PANDA_CAPSULE_JUNCTION_EXTRAS = frozenset({frozenset({"panda_link5", "panda_link7"})})


def test_panda_mobile_acm_matches_franka_srdf() -> None:
    """panda_mobile's self-collision ACM mirrors the Franka SRDF (ADR-0030).

    Regression guard: the ACM was once re-derived independently and dropped the
    SRDF ``Never`` pairs (notably link1↔link4), which false-E-stopped a live
    robocasa pi05 episode. The allowed set among arm links must equal the SRDF
    Adjacent+Never disables, plus only the documented capsule-junction extras.
    """
    desc = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    arm_pairs = {
        frozenset(p)
        for p in desc.allowed_collision_pairs
        if all(link.startswith("panda_link") for link in p)
    }
    expected = _PANDA_SRDF_ARM_DISABLES | _PANDA_CAPSULE_JUNCTION_EXTRAS
    missing = expected - arm_pairs
    extra = arm_pairs - expected
    assert not missing, f"ACM is missing SRDF-disabled pairs (would false-E-stop): {missing}"
    assert not extra, f"ACM allows pairs the SRDF checks (undocumented over-permissive): {extra}"
    # The specific pair that regressed must be present.
    assert frozenset({"panda_link1", "panda_link4"}) in arm_pairs


def test_robot_description_without_collision_geometry_still_loads() -> None:
    """The new fields default empty, so a minimal manifest is unchanged (§1.6)."""
    desc = RobotDescription.from_yaml(_OPENARM_YAML)
    minimal = RobotDescription(
        name=desc.name,
        embodiment_kind=desc.embodiment_kind,
        joints=desc.joints,
        capabilities=desc.capabilities,
        safety=desc.safety,
    )
    assert minimal.collision_geometry == []
    assert minimal.allowed_collision_pairs == []
    assert minimal.srdf_path is None
