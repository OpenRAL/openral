"""Contract tests for the geometric-safety schemas.

Covers the typed surface added for self/world-collision checking: the
``CollisionShape`` discriminated union, ``LinkCollisionGeometry`` /
``WorldCollisionPrimitive`` / ``OccupancyGridRef``, the ``CollisionEvidence``
``FailureEvidence`` variant, and the real ``robots/openarm/robot.yaml``
fixture carrying capsule/sphere link geometry + an allowed-collision matrix.

CLAUDE.md §1.11 — real schemas, real fixture under ``robots/openarm/``, no
mocks.
"""

from __future__ import annotations

import json
from pathlib import Path

from openral_core import (
    BoxShape,
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

#: A real ``FailureTrigger.evidence_json`` captured from the C++ safety kernel's
#: REACTIVE (measured-state) collision check. Provenance + reproduction command
#: in the sibling ``.SOURCE.txt``.
_KERNEL_REACTIVE_EVIDENCE = (
    Path(__file__).parent / "fixtures" / "kernel_reactive_collision_evidence.json"
)

#: A real ``FailureTrigger.evidence_json`` captured from the same kernel's
#: PREDICTIVE (Jacobian look-ahead) check. Provenance + reproduction command in
#: the sibling ``.SOURCE.txt``.
_KERNEL_PREDICTIVE_EVIDENCE = (
    Path(__file__).parent / "fixtures" / "kernel_predictive_collision_evidence.json"
)


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


def test_collision_shape_refuses_an_untagged_mapping() -> None:
    """A shape mapping without the ``shape`` tag is refused, not guessed.

    Before ``CollisionShape`` carried ``Field(discriminator="shape")`` this
    mapping was *accepted*: pydantic's smart union matched it structurally and
    silently produced a ``SphereShape``. A manifest that omitted the tag
    therefore got a real primitive that nobody had written down. The tag has
    always been the documented contract, so the only correct outcome is a
    named refusal.
    """
    try:
        LinkCollisionGeometry.model_validate({"link_name": "link_1", "shape": {"radius_m": 0.05}})
    except ValidationError as exc:
        errors = exc.errors()
        assert len(errors) == 1, f"expected one precise error, got {len(errors)}: {errors}"
        assert errors[0]["type"] == "union_tag_not_found"
        assert errors[0]["loc"] == ("shape",)
    else:
        raise AssertionError("an untagged collision shape must not validate")


def test_collision_shape_unknown_tag_reports_one_error_naming_the_valid_tags() -> None:
    """An unknown tag yields a single error that enumerates the real tags.

    The bare union produced *six* errors here — one per (variant, field)
    mismatch — none of which said "cylinder is not a shape". The discriminator
    turns that into one error against one field.
    """
    try:
        LinkCollisionGeometry.model_validate(
            {"link_name": "link_1", "shape": {"shape": "cylinder", "radius_m": 0.05}}
        )
    except ValidationError as exc:
        errors = exc.errors()
        assert len(errors) == 1, f"expected one precise error, got {len(errors)}: {errors}"
        assert errors[0]["type"] == "union_tag_invalid"
        message = errors[0]["msg"]
        assert "cylinder" in message
        for tag in ("capsule", "sphere", "box"):
            assert tag in message, f"error should enumerate {tag!r}: {message}"
    else:
        raise AssertionError("an unknown collision shape tag must not validate")


def test_every_real_manifest_round_trips_through_the_tagged_union() -> None:
    """Every committed manifest's collision geometry survives dump → re-validate.

    The discriminator narrows what validates, so the guard that it narrowed
    onto the *published* contract and not past it is that all 11 real
    ``robots/*/robot.yaml`` manifests still load, and that each primitive
    re-validates to the very same variant after a serialization round-trip
    (CLAUDE.md §1.11 — real fixtures, no placeholders).
    """
    manifests = sorted(Path("robots").glob("*/robot.yaml"))
    assert manifests, "no robot manifests found"

    seen: set[str] = set()
    checked = 0
    for path in manifests:
        desc = RobotDescription.from_yaml(str(path))
        for geometry in desc.collision_geometry:
            reloaded = LinkCollisionGeometry.model_validate_json(geometry.model_dump_json())
            assert reloaded == geometry, f"{path}:{geometry.link_name} did not round-trip"
            assert type(reloaded.shape) is type(geometry.shape), (
                f"{path}:{geometry.link_name} re-validated to a different variant"
            )
            seen.add(geometry.shape.shape)
            checked += 1

    assert checked > 0, "no collision geometry exercised"
    # All three variants are represented in the committed corpus, so this is a
    # round-trip over every member of the union, not just the common one.
    assert seen == {"capsule", "sphere", "box"}, seen


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
    """A ``kind="collision"`` payload decodes to ``CollisionEvidence``."""
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
    assert not decoded.is_reactive  # horizon_step=2 is a predicted step


def test_kernel_reactive_collision_evidence_validates() -> None:
    """A REAL kernel-emitted reactive payload (``horizon_step: -1``) validates.

    Regression: the field used to be ``Field(ge=0)``, which rejected every
    reactive (measured-state) hit the kernel publishes — and every Cartesian
    control mode reaches the reactive check first, so an attached-payload stop
    always landed there. The reasoner then silently fell back to raw-JSON
    truncation instead of a structured summary.
    """
    payload = _KERNEL_REACTIVE_EVIDENCE.read_text(encoding="utf-8").strip()
    decoded = TypeAdapter(FailureEvidence).validate_json(payload)
    assert isinstance(decoded, CollisionEvidence)
    assert decoded.collision_kind == "world"
    assert decoded.link_a == "ee"
    assert decoded.link_b_or_object == "voxel_189"
    assert decoded.horizon_step == CollisionEvidence.REACTIVE_HORIZON_STEP == -1
    assert decoded.is_reactive
    assert decoded.min_distance_m == -0.05
    # Recorded before the kernel carried its configuration: the field is a
    # backward-compatible addition, so an old payload still validates and says
    # so by being empty rather than by failing to decode.
    assert decoded.joint_positions_rad == []


def test_kernel_predictive_evidence_carries_the_configuration_it_adjudicated() -> None:
    """A REAL predictive payload carries the *predicted* configuration, exactly.

    A predicted step's configuration is the kernel's own damped-least-squares
    integration of the chunk, at its lambda and its seed dt — it exists in no
    other artifact. Without it a predictive stop can only be adjudicated
    against the measured joints, which is a different pose: the drawer-opening
    run that motivated this reported ``panda_link2`` vs ``panda_link5`` at
    -5.34 mm while offline mesh adjudication at the recorded joints put the
    same pair +53 mm clear.

    The precision assertion is the second half of the contract. The kernel
    serializes at ``max_digits10``; the default 6 significant digits would
    round these angles to ~1e-6 rad, millimetres of end-effector error at a
    metre of reach — the scale the evidence exists to adjudicate at.
    """
    payload = _KERNEL_PREDICTIVE_EVIDENCE.read_text(encoding="utf-8").strip()
    decoded = TypeAdapter(FailureEvidence).validate_json(payload)
    assert isinstance(decoded, CollisionEvidence)
    assert decoded.horizon_step == 5
    assert not decoded.is_reactive
    assert decoded.joint_positions_rad == [0.29982877677088093, 1.2524345104800854]
    # Not the measured seed the chunk started from — that config passed.
    assert decoded.joint_positions_rad[1] != 1.57079632679
    assert json.loads(payload)["joint_positions_rad"] == decoded.joint_positions_rad


def test_collision_evidence_rejects_below_reactive_sentinel() -> None:
    """``-1`` is the only negative ``horizon_step``; ``-2`` is still garbage."""
    try:
        CollisionEvidence(
            collision_kind="world",
            link_a="ee",
            link_b_or_object="voxel_189",
            horizon_step=-2,
            min_distance_m=-0.05,
        )
    except ValidationError:
        pass
    else:  # pragma: no cover - the constraint must fire
        raise AssertionError("CollisionEvidence accepted horizon_step=-2")


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
    """Adjacent links are allowed to touch; the two arms are not."""
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


def test_panda_mobile_collision_fk_starts_at_the_arm_mount() -> None:
    """Both PandaMobile manifests measure joint 1 from the arm-mount ``base_link``.

    ADR-0095. ``base_link`` is RoboCasa's ``mobilebase0_support`` — the top of
    the 0.70 m pedestal, the pose ``odom -> base_link`` carries and the frame
    ``/openral/world_voxels`` is expressed in. Joint 1 is therefore the plain
    Franka URDF transform. PR #103 briefly folded the pedestal in here as well,
    to cancel a producer-side frame bug *inside* the kernel; the producer is
    fixed at source now, so double-counting it would put the collision capsules
    0.70 m above the obstacles they are checked against (hazard HZ-0095-1).
    """
    desc = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    vslam_desc = RobotDescription.from_yaml("robots/panda_mobile_vslam/robot.yaml")
    joint1 = next(joint for joint in desc.joints if joint.name == "panda_joint1")
    vslam_joint1 = next(joint for joint in vslam_desc.joints if joint.name == "panda_joint1")

    assert joint1.origin_xyz == vslam_joint1.origin_xyz == (0.0, 0.0, 0.333)


def test_panda_mobile_arms_use_matching_mesh_enclosing_obbs() -> None:
    """Both mobile Panda manifests use the same measured arm mesh bounds."""
    descriptions = [
        RobotDescription.from_yaml(path)
        for path in (
            "robots/panda_mobile/robot.yaml",
            "robots/panda_mobile_vslam/robot.yaml",
        )
    ]
    arm_geometry = [
        {
            geometry.link_name: geometry
            for geometry in desc.collision_geometry
            if geometry.link_name.startswith("panda_link")
        }
        for desc in descriptions
    ]

    assert arm_geometry[1] == arm_geometry[0]
    assert set(arm_geometry[0]) == {f"panda_link{i}" for i in range(1, 8)}
    assert all(isinstance(geometry.shape, BoxShape) for geometry in arm_geometry[0].values())


def test_panda_mobile_acm_matches_franka_srdf() -> None:
    """panda_mobile's self-collision ACM mirrors the Franka SRDF.

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
    assert minimal.assets.srdf is None
