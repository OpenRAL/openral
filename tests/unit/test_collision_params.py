"""Unit tests for ``collision_params_from_description``.

Verifies the RobotDescription → kernel collision-param flattening: topological
link ordering (parent precedes child), per-link joint/dof/origin/axis arrays,
capsule lowering (capsule + sphere), and the allowed-collision matrix. Uses a
small synthetic chain whose geometry is fully controlled, plus the real openarm
fixture for shape sanity.

CLAUDE.md §1.11 — real schemas + the real ``robots/openarm`` fixture, no mocks.
"""

from __future__ import annotations

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

from tests.unit.conftest import _CylinderShape


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


def test_unknown_primitive_is_refused_not_lowered_as_a_capsule() -> None:
    """An unlowerable primitive raises instead of becoming an assumed capsule.

    Regression: the old routing (``if BoxShape: ... else: <capsule>``) lowered any
    other variant to a zero-length capsule of its ``radius_m`` — unsafe, since that
    capsule is contained in every non-spherical primitive carrying the same radius,
    so the kernel got a smaller volume than declared. Refusing is at-least-as-
    conservative: no params emitted, so no motion is authorised against a wrong
    envelope.
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
    # Per-primitive arrays are parallel and tagged with valid link indices (the
    # OpenArm lowers every link to a refined box today, so it may emit no
    # capsule arrays at all — absent and empty mean the same to the kernel).
    n_caps = len(params.get("collision_capsule_link", []))
    assert len(params.get("collision_capsule_radius", [])) == n_caps
    assert len(params.get("collision_capsule_origin_xyzrpy", [])) == 6 * n_caps
    n_boxes = len(params.get("collision_box_link", []))
    assert len(params.get("collision_box_half_extents", [])) == 3 * n_boxes
    assert len(params.get("collision_box_origin_xyzrpy", [])) == 6 * n_boxes
    assert n_caps + n_boxes > 0
    assert all(
        0 <= li < n
        for li in [*params.get("collision_capsule_link", []), *params.get("collision_box_link", [])]
    )
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


def test_tight_geometry_lowers_csr_parallel_to_the_box_arrays() -> None:
    """The staged narrow phase's parameters, on the real panda_mobile manifest.

    ``collision_box_hull`` is parallel to ``collision_box_link``: one entry per
    box, naming its tight representation or ``-1``. The ``collision_hull_*``
    arrays are CSR-indexed by it. Getting that shape wrong is not a cosmetic
    bug — the kernel would attach one link's geometry to another link's box, and
    the containment proof is per-link.
    """
    robot = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    params = collision_params_from_description(robot)

    box_hull = params["collision_box_hull"]
    assert len(box_hull) == len(params["collision_box_link"])
    n_hulls = sum(1 for h in box_hull if h >= 0)
    assert n_hulls == 7, (
        "link1 + link2 (world side), link5 + link7 (the self pair, #191), and "
        "link3 + link4 + link6 (2026-09-07, the #204 battery's link-class excess)"
    )
    assert sorted(h for h in box_hull if h >= 0) == list(range(n_hulls)), (
        "hull indices must be dense and unique; a repeated index would silently share "
        "one link's geometry with another"
    )

    assert len(params["collision_hull_dop_lo"]) == 13 * n_hulls
    assert len(params["collision_hull_dop_hi"]) == 13 * n_hulls
    assert len(params["collision_hull_vertex_first"]) == n_hulls
    assert len(params["collision_hull_vertex_count"]) == n_hulls

    # CSR slices must tile the vertex buffer exactly: no gaps, no overlap, no
    # slice running off the end.
    verts = params.get("collision_hull_vertices", [])
    assert len(verts) % 3 == 0
    n_vertices = len(verts) // 3
    cursor = 0
    for first, count in zip(
        params["collision_hull_vertex_first"], params["collision_hull_vertex_count"], strict=True
    ):
        assert first == cursor
        cursor += count
    assert cursor == n_vertices

    # link1 is stage 1 only (1588-vertex hull, over the kernel's cost budget).
    # link2/link3/link4/link5 ship their exact 152-vertex hulls; link6 and link7
    # their 102-vertex ones. link5 + link7 are what the self-collision
    # refinement needs (#191); link3/link4/link6 were added 2026-09-07 for the
    # link-class excess the #204 battery measured (the programme note §5,
    # docs/reference/collision-validation-evidence.md).
    assert sorted(params["collision_hull_vertex_count"]) == [0, 102, 102, 152, 152, 152, 152]


def test_no_tight_geometry_lowers_exactly_as_before() -> None:
    """A manifest that declares none must emit none.

    The kernel switches the staged narrow phase on by seeing
    ``box_hull.size() == boxes.size()``, so an accidentally-emitted empty array
    would be the difference between "no robot changed" and "every boxed robot
    changed". Checked on the real so101_follower fixture, which declares no
    tight geometry (the OpenArm, the previous fixture, now refines every link).
    """
    robot = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    assert all(g.tight_geometry is None for g in robot.collision_geometry)
    params = collision_params_from_description(robot)
    assert "collision_box_hull" not in params
    assert "collision_hull_dop_lo" not in params
    assert "collision_hull_vertices" not in params


def test_several_primitives_per_link_all_reach_the_kernel() -> None:
    """A link may carry several primitives; every one lands in the kernel arrays.

    The C++ kernel has always accepted several capsules/boxes per link (same-link
    pairs are skipped in ``check_self_collision``); only this lowering refused
    them. The unified fitter emits capsule chains and mixed sets, so the refusal
    is lifted: each primitive is routed by its kind and tagged with its link index.
    """
    from openral_core import BoxShape

    robot = _two_link_arm()
    robot = robot.model_copy(
        update={
            "collision_geometry": [
                LinkCollisionGeometry(
                    link_name="link1",
                    shape=CapsuleShape(radius_m=0.05, length_m=0.1),
                    origin_xyz_rpy=(0.0, 0.0, -0.05, 0.0, 0.0, 0.0),
                ),
                LinkCollisionGeometry(
                    link_name="link1",
                    shape=CapsuleShape(radius_m=0.05, length_m=0.1),
                    origin_xyz_rpy=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0),
                ),
                LinkCollisionGeometry(
                    link_name="link1", shape=BoxShape(half_extents_m=(0.02, 0.02, 0.02))
                ),
                LinkCollisionGeometry(link_name="link2", shape=SphereShape(radius_m=0.04)),
            ]
        }
    )
    params = collision_params_from_description(robot)
    assert params["collision_capsule_link"] == [1, 1, 2]
    assert params["collision_capsule_half_length"] == [0.05, 0.05, 0.0]
    assert params["collision_capsule_origin_xyzrpy"][2] == -0.05
    assert params["collision_capsule_origin_xyzrpy"][8] == 0.05
    assert params["collision_box_link"] == [1]
    assert params["collision_n_links"] == 3  # links, not primitives


def test_a_capsule_split_into_a_chain_is_exactly_as_conservative() -> None:
    """Two capsules covering one capsule's segment trip exactly where the one did.

    The at-least-as-conservative proof for the chain output (CLAUDE.md §3): the
    union of the two halves is the original capsule, so the kernel's pair gap —
    the minimum over the link's primitives, as ``check_self_collision`` folds
    it — equals the single-capsule gap at every pose.
    """
    import numpy as np
    from openral_safety.kernel_predicates import capsule_distance

    whole = LinkCollisionGeometry(
        link_name="link1", shape=CapsuleShape(radius_m=0.05, length_m=0.2)
    )
    halves = [
        LinkCollisionGeometry(
            link_name="link1",
            shape=CapsuleShape(radius_m=0.05, length_m=0.1),
            origin_xyz_rpy=(0.0, 0.0, z, 0.0, 0.0, 0.0),
        )
        for z in (-0.05, 0.05)
    ]
    robot = _two_link_arm()
    single = collision_params_from_description(
        robot.model_copy(update={"collision_geometry": [whole, robot.collision_geometry[1]]})
    )
    chain = collision_params_from_description(
        robot.model_copy(update={"collision_geometry": [*halves, robot.collision_geometry[1]]})
    )
    assert chain["collision_capsule_link"] == [1, 1, 2]
    rng = np.random.default_rng(0)

    def gaps(params: dict[str, object], probe: np.ndarray) -> float:
        link = list(params["collision_capsule_link"])  # type: ignore[call-overload]  # reason: flat param list
        radius = list(params["collision_capsule_radius"])  # type: ignore[call-overload]  # reason: flat param list
        half = list(params["collision_capsule_half_length"])  # type: ignore[call-overload]  # reason: flat param list
        origin = np.asarray(params["collision_capsule_origin_xyzrpy"], dtype=float).reshape(-1, 6)
        tfs = [np.eye(4) for _ in link]
        for i, o in enumerate(origin):
            tfs[i][:3, 3] = o[:3]
        return min(
            float(capsule_distance(tfs[i][None], radius[i], half[i], probe[None], 0.01, 0.0)[0])
            for i, li in enumerate(link)
            if li == 1
        )

    for _ in range(200):
        probe = np.eye(4)
        probe[:3, 3] = rng.uniform(-0.3, 0.3, 3)
        assert gaps(chain, probe) == pytest.approx(gaps(single, probe), abs=1e-12)
