"""The collision-block splicer/renderer replaces only the two collision blocks.

`openral collision lower --write` must preserve every hand comment outside the
`collision_geometry` / `allowed_collision_pairs` blocks, so it does a
targeted textual splice rather than a full manifest round-trip. No mocks (§1.11).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from openral_cli.collision import inject_joint_fk, render_blocks, splice_collision_blocks
from openral_core import BoxShape, CapsuleShape, LinkCollisionGeometry, SphereShape
from openral_safety.kernel_predicates import bounding_capsule_segment
from openral_safety.mjcf_lowering import _rpy_to_mat
from openral_safety.urdf_lowering import LoweredCollisionModel

_MANIFEST = """\
name: demo
joints: []
collision_geometry:
  - link_name: "old"
    shape: { shape: "sphere", radius_m: 0.1 }
    origin_xyz_rpy: [0, 0, 0, 0, 0, 0]
allowed_collision_pairs:
  - [old_a, old_b]
# trailing comment that must survive
sdk_kind: "open"
"""


def test_splice_replaces_only_the_two_blocks() -> None:
    new_geo = (
        'collision_geometry:\n  - link_name: "new"\n'
        '    shape: { shape: "sphere", radius_m: 0.2 }\n'
        "    origin_xyz_rpy: [0, 0, 0, 0, 0, 0]\n"
    )
    new_acm = "allowed_collision_pairs:\n  - [new_a, new_b]\n"
    out = splice_collision_blocks(_MANIFEST, geometry_block=new_geo, acm_block=new_acm)
    assert '"new"' in out and '"old"' not in out
    assert "[new_a, new_b]" in out and "[old_a, old_b]" not in out
    assert "# trailing comment that must survive" in out
    assert "name: demo" in out and 'sdk_kind: "open"' in out


def test_splice_only_acm_leaves_geometry_untouched() -> None:
    new_acm = "allowed_collision_pairs:\n  - [x, y]\n"
    out = splice_collision_blocks(_MANIFEST, geometry_block=None, acm_block=new_acm)
    assert '"old"' in out  # geometry block untouched
    assert "[x, y]" in out and "[old_a, old_b]" not in out


def test_splice_is_idempotent_when_block_unchanged() -> None:
    geo, acm = (
        'collision_geometry:\n  - link_name: "old"\n'
        '    shape: { shape: "sphere", radius_m: 0.1 }\n'
        "    origin_xyz_rpy: [0, 0, 0, 0, 0, 0]\n",
        "allowed_collision_pairs:\n  - [old_a, old_b]\n",
    )
    # Re-splicing the same content (modulo the generated header) must round-trip
    # cleanly without corrupting the surrounding keys.
    out = splice_collision_blocks(_MANIFEST, geometry_block=geo, acm_block=acm)
    assert out.count("name: demo") == 1
    assert out.count('sdk_kind: "open"') == 1
    assert out.rstrip().endswith('sdk_kind: "open"')


def test_splice_appends_blocks_when_absent() -> None:
    """Onboarding a manifest with no collision blocks appends them at the end."""
    manifest = 'name: fresh\njoints: []\nsdk_kind: "open"\n'
    geo = (
        'collision_geometry:\n  - link_name: "l1"\n'
        '    shape: { shape: "sphere", radius_m: 0.05 }\n'
        "    origin_xyz_rpy: [0, 0, 0, 0, 0, 0]\n"
    )
    acm = "allowed_collision_pairs:\n  - [l1, l2]\n"
    out = splice_collision_blocks(manifest, geometry_block=geo, acm_block=acm)
    assert "name: fresh" in out and 'sdk_kind: "open"' in out
    assert out.index("collision_geometry:") < out.index("allowed_collision_pairs:")
    import yaml

    assert yaml.safe_load(out)["allowed_collision_pairs"] == [["l1", "l2"]]


_JOINTS_MANIFEST = """\
name: arm
joints:
  - name: "j1"
    joint_type: "revolute"
    parent_link: "base"
    child_link: "link1"
    velocity_limit: 2.0
  - name: "j2"
    joint_type: "revolute"
    parent_link: "link1"
    child_link: "link2"
sdk_kind: "open"
"""


def test_inject_joint_fk_adds_fields_and_preserves_rest() -> None:
    fk = {"j1": ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))}
    out = inject_joint_fk(_JOINTS_MANIFEST, fk)
    assert "origin_xyz: [0, 0, 0.333]" in out
    assert "axis_xyz: [0, 0, 1]" in out
    # Untouched joint j2 has no FK injected; surrounding fields/keys preserved.
    assert out.count("origin_xyz:") == 1
    assert "velocity_limit: 2.0" in out and 'sdk_kind: "open"' in out
    import yaml

    data = yaml.safe_load(out)
    j1 = next(j for j in data["joints"] if j["name"] == "j1")
    assert j1["origin_xyz"] == [0, 0, 0.333] and j1["joint_type"] == "revolute"


def test_inject_joint_fk_is_idempotent() -> None:
    fk = {"j1": ((0.1, 0.2, 0.3), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))}
    once = inject_joint_fk(_JOINTS_MANIFEST, fk)
    twice = inject_joint_fk(once, fk)
    assert once == twice, "re-injecting the same FK must not stack duplicate lines"


def test_render_blocks_emits_provenance_header_and_shapes() -> None:
    model = LoweredCollisionModel(
        collision_geometry=[
            LinkCollisionGeometry(
                link_name="panda_link1",
                shape=SphereShape(radius_m=0.075),
                origin_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            LinkCollisionGeometry(
                link_name="panda_link2",
                shape=CapsuleShape(radius_m=0.06, length_m=0.316),
                origin_xyz_rpy=(0.0, -0.158, 0.0, 0.0, 1.5708, -1.5708),
            ),
        ],
        allowed_collision_pairs=[("panda_link1", "panda_link2")],
        acm_source="srdf",
    )
    geo_block, acm_block = render_blocks(model)
    assert geo_block.startswith("# GENERATED")
    assert "collision_geometry:" in geo_block
    assert 'link_name: "panda_link1"' in geo_block
    assert 'shape: "sphere"' in geo_block and "radius_m: 0.0750" in geo_block
    assert 'shape: "capsule"' in geo_block and "length_m: 0.3160" in geo_block
    assert "source: srdf" in acm_block
    assert "- [panda_link1, panda_link2]" in acm_block


def test_geometry_replacement_preserves_interblock_comment_on_real_manifest() -> None:
    """Replacing collision_geometry must keep the column-0 ACM comment that follows it."""
    from pathlib import Path

    src = Path("robots/panda_mobile/robot.yaml").read_text(encoding="utf-8")
    assert "# Self-collision ACM" in src  # the inter-block comment we must protect
    stub = (
        'collision_geometry:\n  - link_name: "x"\n'
        '    shape: { shape: "sphere", radius_m: 0.1 }\n'
        "    origin_xyz_rpy: [0, 0, 0, 0, 0, 0]\n"
    )
    out = splice_collision_blocks(src, geometry_block=stub, acm_block=None)
    assert "# Self-collision ACM" in out, "the inter-block comment was swallowed"
    assert "allowed_collision_pairs:" in out
    assert "sdk_kind:" in out


def test_render_then_splice_validates_as_a_robot_description() -> None:
    """A rendered-then-spliced panda_mobile manifest still loads + validates."""
    from pathlib import Path

    import yaml
    from openral_core import RobotDescription

    src = Path("robots/panda_mobile/robot.yaml").read_text(encoding="utf-8")
    model = LoweredCollisionModel(
        collision_geometry=[],
        allowed_collision_pairs=[("panda_link1", "panda_link4")],
        acm_source="srdf",
    )
    _, acm_block = render_blocks(model)
    out = splice_collision_blocks(src, geometry_block=None, acm_block=acm_block)
    # The kernel consumer (collision_params_from_description) never sees broken yaml.
    desc = RobotDescription.model_validate(yaml.safe_load(out))
    assert ("panda_link1", "panda_link4") in {tuple(p) for p in desc.allowed_collision_pairs}


def test_render_blocks_serializes_box_not_capsule() -> None:
    """A ``BoxShape`` renders as a box (``half_extents_m``), not a capsule.

    Before the box branch, ``render_blocks``' ``else`` assumed capsule and read
    ``.length_m``, which ``BoxShape`` lacks — crashing on the SO-101's OBB model
    (issue #84). The rendered line must round-trip the committed manifest's
    ``shape: { shape: "box", half_extents_m: [...] }`` format at 4 dp.
    """
    model = LoweredCollisionModel(
        collision_geometry=[
            LinkCollisionGeometry(
                link_name="base",
                shape=BoxShape(half_extents_m=(0.0555, 0.0478, 0.0360)),
                origin_xyz_rpy=(0.0208, -0.0011, 0.0661, 0.0, 0.0, 0.0),
            )
        ],
        allowed_collision_pairs=[],
        acm_source="sampling",
    )
    geo_block, _ = render_blocks(model)
    assert 'shape: { shape: "box", half_extents_m: [0.0555, 0.0478, 0.0360] }' in geo_block
    assert "length_m" not in geo_block  # a box must never be emitted as a capsule
    assert "origin_xyz_rpy: [0.0208, -0.0011, 0.0661, 0.0000, 0.0000, 0.0000]" in geo_block


def test_a_box_never_lowers_to_a_capsule_smaller_than_itself() -> None:
    """The retired inscribed-sphere lowering, and why its safety argument was wrong.

    This test used to assert the opposite: that a box feeds the ACM sweep as its
    *inscribed* sphere, on the reasoning that under-approximating "can only reduce
    the collision count — it can never disable a self-collision check the true box
    would keep".

    Both halves of that were wrong (issue #155). The ACM sweep no longer reduces a
    box to a single primitive at all — it asks `kernel_predicates.shape_distance`
    for the true oriented box, the same question the kernel asks. And the
    under-approximation was not safe even on its own terms: it silently dropped
    `panda_link5`/`panda_link7`, and the same helper fed the cuMotion *planner*,
    where believing a link is thinner than it is means planning into obstacles.

    What survives is the invariant in the other direction: the one remaining
    capsule lowering (for consumers that can only express capsules) must
    **contain** the box.
    """
    box = BoxShape(half_extents_m=(0.08, 0.04, 0.02))
    origin = (0.1, 0.2, 0.3, 0.5, 0.6, 0.7)  # arbitrary rotation — must not matter
    p0, p1, radius = bounding_capsule_segment(box, origin)

    # Radius spans the cross-section diagonal of the two shorter axes, so it is
    # strictly larger than the smallest half-extent the old lowering used.
    assert radius == pytest.approx(math.hypot(0.04, 0.02))
    assert radius > min(box.half_extents_m)
    # The segment is the longest axis, so the capsule is not a sphere.
    assert p0 != p1
    assert math.dist(p0, p1) == pytest.approx(2 * 0.08)
    # Every corner of the box lies within `radius` of the segment: it is contained.
    axes = [
        [(0.08 if sx else -0.08), (0.04 if sy else -0.04), (0.02 if sz else -0.02)]
        for sx in (0, 1)
        for sy in (0, 1)
        for sz in (0, 1)
    ]
    rot = np.asarray(_rpy_to_mat(*origin[3:])).reshape(3, 3)
    for corner in axes:
        world = rot @ np.asarray(corner) + np.asarray(origin[:3])
        seg = np.asarray(p1) - np.asarray(p0)
        t = float(np.clip((world - np.asarray(p0)) @ seg / (seg @ seg), 0.0, 1.0))
        closest = np.asarray(p0) + seg * t
        assert float(np.linalg.norm(world - closest)) <= radius + 1e-9
