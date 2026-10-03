"""Reasoner-named grasp/place targets, grounded by perception (real pick-and-place design §2.2).

The reasoner names a target; ``openral_reasoner.grounding`` grounds it from what perception
already holds — the world-state lift's 3D boxes (``VoxelFrustumLifter``, run here for real
on a voxel scene seen by the OpenArm ``head_zed`` intrinsics) and the spatial memory
(``SpatialMemory``, ingesting those same lifted boxes) — into the declarations the
``ExecuteRskill`` goal carries. The producer measures; nothing here may supply a region.

Real fixtures throughout: ``robots/openarm/robot.yaml``,
``tests/unit/fixtures/robot_units/openarm_shelf_cell.yaml``,
``tests/unit/fixtures/home_scene_graph.json`` (CLAUDE.md §1.11).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import (
    DetectedObject,
    ExecuteRskillTool,
    GraspDeclaration,
    GraspTargetRef,
    ObjectDetection2D,
    PlaceTargetRef,
    Pose6D,
    RobotDescription,
    RobotUnit,
    SceneGraph,
)
from openral_core.exceptions import ROSReasonerInvalidPlan
from openral_reasoner.grounding import (
    gripper_hands,
    ground_grasp_target,
    ground_place_target,
)
from openral_reasoner.palette import ToolPalette
from openral_reasoner.tool_use import (
    _decode_tool_payload,
    _tool_palette_to_anthropic_tools,
    resolve_reasoner_system_prompt,
)
from openral_world_state import SpatialMemory, VoxelFrustumLifter

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPENARM = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml"))
_UNIT = RobotUnit.from_yaml(
    str(_REPO_ROOT / "tests" / "unit" / "fixtures" / "robot_units" / "openarm_shelf_cell.yaml")
)
_BASE = _OPENARM.base_frame
_VOXEL = 0.02
_PAD = 0.035  # one 20 mm cell + the 15 mm planar extrinsic bound (the node defaults)

# head_zed optical frame looking along base +x: base (x, y, z) -> optical (-y, -z, x).
_T_CAM_FROM_BASE: NDArray[np.float64] = np.array(
    [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def _cube(center: tuple[float, float, float], n: int = 3) -> NDArray[np.float64]:
    """Occupied voxel centres of an ``n``-cell cube (20 mm cells) around ``center``."""
    offs = (np.arange(n) - (n - 1) / 2.0) * _VOXEL
    return np.array(
        [(center[0] + a, center[1] + b, center[2] + c) for a in offs for b in offs for c in offs]
    )


def _detection(label: str, voxels: NDArray[np.float64]) -> ObjectDetection2D:
    """The pixel box a detector would draw around ``voxels`` in head_zed."""
    k = next(s.intrinsics for s in _OPENARM.sensors if s.name == "head_zed")
    assert k is not None
    cam = (_T_CAM_FROM_BASE @ np.c_[voxels, np.ones(len(voxels))].T).T[:, :3]
    u = k.fx * cam[:, 0] / cam[:, 2] + k.cx
    v = k.fy * cam[:, 1] / cam[:, 2] + k.cy
    pad = 10
    return ObjectDetection2D(
        label=label,
        confidence=0.9,
        bbox_xyxy=(int(u.min()) - pad, int(v.min()) - pad, int(u.max()) + pad, int(v.max()) + pad),
    )


def _lift(objects: dict[str, tuple[str, tuple[float, float, float]]]) -> list[DetectedObject]:
    """Run the world-state lift on a voxel scene, in the base frame (a fixed-base cell)."""
    k = next(s.intrinsics for s in _OPENARM.sensors if s.name == "head_zed")
    assert k is not None
    voxels = {key: _cube(center) for key, (_label, center) in objects.items()}
    lifted = VoxelFrustumLifter().lift(
        detections=[_detection(label, voxels[key]) for key, (label, _c) in objects.items()],
        occupied_centers_base=np.concatenate(list(voxels.values())),
        intrinsics=k,
        frame_size=(k.width, k.height),
        t_cam_from_base=_T_CAM_FROM_BASE,
        t_map_from_base=np.eye(4),
        map_frame=_BASE,
    )
    assert len(lifted) == len(objects)
    return lifted


def _ground(ref: GraspTargetRef, live: list[DetectedObject], graph: SceneGraph | None = None):
    return ground_grasp_target(
        ref,
        live_objects=live,
        scene_graph=graph,
        base_frame=_BASE,
        default_contact_links=gripper_hands(_OPENARM),
        patience_s=60.0,
        pad_m=_PAD,
    )


def _contains(decl: GraspDeclaration, bbox: tuple[float, ...]) -> bool:
    box = decl.search_box
    assert box is not None
    return all(
        box.pose.xyz[i] - box.half_extents[i] <= bbox[i] + 1e-9
        and bbox[i + 3] <= box.pose.xyz[i] + box.half_extents[i]
        for i in range(3)
    )


def test_a_single_lifted_detection_grounds_a_seed_only_declaration() -> None:
    live = _lift({"a": ("box", (0.40, -0.15, -0.10))})
    decl = _ground(GraspTargetRef(label="Box", contact_links=["openarm_left_finger_pair"]), live)
    assert decl.region is None  # only the producer measures
    assert decl.target_id == "obj:box"
    assert decl.contact_links == ("openarm_left_finger_pair",)
    assert decl.timeout_s == 70.0
    box = decl.search_box
    assert box is not None
    assert box.frame_id == _BASE
    assert box.pose.quat_xyzw == (0.0, 0.0, 0.0, 1.0)  # gravity-aligned in the base frame
    assert live[0].bbox_3d is not None
    assert _contains(decl, live[0].bbox_3d)
    # Padded by exactly the configured margin on every side, downward included: the box is
    # a search hint, and the lifted bottom (the lowest occupied centre in the frustum) may
    # sit above OR below the true support, so the support layer must be inside the box for
    # the producer to measure it. The lowest object cell (z = -0.12 here) is one cell above
    # the support layer it stands on; one voxel of downward pad reaches it.
    lo, hi = live[0].bbox_3d[:3], live[0].bbox_3d[3:]
    for i in range(3):
        assert box.half_extents[i] == pytest.approx((hi[i] - lo[i]) / 2.0 + _PAD)
    bottom = box.pose.xyz[2] - box.half_extents[2]
    assert bottom == pytest.approx(lo[2] - _PAD)
    assert bottom <= lo[2] - _VOXEL  # the support layer's cell centre is inside
    assert bottom >= lo[2] - 2 * _VOXEL  # bounded: one voxel + extrinsic error, no more


def test_two_instances_of_the_label_refuse_without_an_object_id() -> None:
    live = _lift({"a": ("box", (0.40, -0.15, -0.10)), "b": ("box", (0.40, 0.15, -0.10))})
    with pytest.raises(ROSReasonerInvalidPlan, match="ambiguous: 2"):
        _ground(GraspTargetRef(label="box"), live)


def test_a_label_perception_does_not_see_refuses() -> None:
    live = _lift({"a": ("box", (0.40, -0.15, -0.10))})
    with pytest.raises(ROSReasonerInvalidPlan, match="no live 3D detection"):
        _ground(GraspTargetRef(label="mug"), live)


def test_a_recalled_node_id_disambiguates() -> None:
    live = _lift({"a": ("box", (0.40, -0.15, -0.10)), "b": ("box", (0.40, 0.15, -0.10))})
    memory = SpatialMemory()
    memory.ingest_detected_objects(live, now_ns=1)
    graph = memory.to_scene_graph()
    right = next(n for n in graph.nodes if n.pose.xyz[1] > 0.0)
    decl = _ground(
        GraspTargetRef(
            label="box", object_id=right.node_id, contact_links=["openarm_right_finger_pair"]
        ),
        live,
        graph,
    )
    assert decl.target_id == f"obj:{right.node_id}"
    assert decl.region is None
    assert right.bbox_3d is not None
    assert _contains(decl, right.bbox_3d)
    left_bbox = next(o.bbox_3d for o in live if o.pose.xyz[1] < 0.0)
    assert left_bbox is not None
    assert not _contains(decl, left_bbox)


def test_an_unknown_node_id_refuses() -> None:
    with pytest.raises(ROSReasonerInvalidPlan, match="not in spatial memory"):
        _ground(GraspTargetRef(label="box", object_id="nope"), [], SceneGraph())


def test_a_map_frame_memory_node_is_refused_not_reinterpreted() -> None:
    graph = SceneGraph.model_validate(
        json.loads((_REPO_ROOT / "tests/unit/fixtures/home_scene_graph.json").read_text())
    )
    with pytest.raises(ROSReasonerInvalidPlan, match="not the robot base frame"):
        _ground(GraspTargetRef(label="bottle of wine", object_id="wine_bottle"), [], graph)


def test_the_reasoner_supplied_contact_links_win() -> None:
    live = _lift({"a": ("box", (0.40, -0.15, -0.10))})
    decl = _ground(GraspTargetRef(label="box", contact_links=["openarm_left_finger_pair"]), live)
    assert decl.contact_links == ("openarm_left_finger_pair",)


def test_an_unnamed_gripper_on_a_bimanual_robot_refuses() -> None:
    """Defaulting to every gripper would name both OpenArm hands for one grasp."""
    live = _lift({"a": ("box", (0.40, -0.15, -0.10))})
    with pytest.raises(ROSReasonerInvalidPlan, match=r"2 hands.*openarm_left_finger_pair"):
        _ground(GraspTargetRef(label="box"), live)


def test_a_single_gripper_robot_keeps_the_default_contact_link() -> None:
    so101 = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "so101_follower" / "robot.yaml"))
    ((link,),) = gripper_hands(so101)
    box = DetectedObject(
        label="eraser",
        confidence=0.9,
        pose=Pose6D(
            xyz=(0.2, 0.0, 0.02), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=so101.base_frame
        ),
        bbox_3d=(0.18, -0.02, 0.0, 0.22, 0.02, 0.04),
    )
    decl = ground_grasp_target(
        GraspTargetRef(label="eraser"),
        live_objects=[box],
        scene_graph=None,
        base_frame=so101.base_frame,
        default_contact_links=gripper_hands(so101),
        patience_s=60.0,
        pad_m=_PAD,
    )
    assert decl.contact_links == (link,)


_R1PRO = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "r1pro" / "robot.yaml"))
_R1_LEFT = ("left_gripper_finger_link1", "left_gripper_finger_link2")
_R1_RIGHT = ("right_gripper_finger_link1", "right_gripper_finger_link2")


def test_hands_group_finger_joints_by_the_arm_they_hang_off() -> None:
    """R1 Pro's four finger joints are two hands; OpenArm's two pair joints are two hands."""
    assert gripper_hands(_R1PRO) == (_R1_LEFT, _R1_RIGHT)
    assert gripper_hands(_OPENARM) == (
        ("openarm_left_finger_pair",),
        ("openarm_right_finger_pair",),
    )


def _ground_r1(ref: GraspTargetRef) -> GraspDeclaration:
    box = DetectedObject(
        label="cup",
        confidence=0.9,
        pose=Pose6D(
            xyz=(0.5, 0.0, 0.8), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=_R1PRO.base_frame
        ),
        bbox_3d=(0.46, -0.04, 0.76, 0.54, 0.04, 0.84),
    )
    return ground_grasp_target(
        ref,
        live_objects=[box],
        scene_graph=None,
        base_frame=_R1PRO.base_frame,
        default_contact_links=gripper_hands(_R1PRO),
        patience_s=60.0,
        pad_m=_PAD,
    )


def test_an_unnamed_hand_on_r1pro_refuses_listing_both_hands() -> None:
    with pytest.raises(ROSReasonerInvalidPlan, match=r"2 hands.*left_gripper_finger_link2\]"):
        _ground_r1(GraspTargetRef(label="cup"))


def test_naming_fingers_of_one_hand_grounds() -> None:
    decl = _ground_r1(GraspTargetRef(label="cup", contact_links=list(_R1_LEFT)))
    assert decl.contact_links == _R1_LEFT
    # Order-free: the declaration carries the hand in manifest order.
    swapped = _ground_r1(GraspTargetRef(label="cup", contact_links=list(reversed(_R1_RIGHT))))
    assert swapped.contact_links == _R1_RIGHT


def test_naming_part_of_a_hand_refuses_listing_the_whole_hand() -> None:
    """One finger of a two-finger hand would exempt that finger only: the grasp would stop."""
    with pytest.raises(
        ROSReasonerInvalidPlan,
        match=r"part of a hand.*\[right_gripper_finger_link1, right_gripper_finger_link2\]",
    ):
        _ground_r1(GraspTargetRef(label="cup", contact_links=[_R1_RIGHT[1]]))


def test_naming_fingers_of_two_hands_refuses() -> None:
    with pytest.raises(ROSReasonerInvalidPlan, match="not gripper child links of one hand"):
        _ground_r1(GraspTargetRef(label="cup", contact_links=[_R1_LEFT[0], _R1_RIGHT[0]]))


def test_naming_a_link_that_is_no_gripper_child_refuses() -> None:
    with pytest.raises(ROSReasonerInvalidPlan, match="not gripper child links of one hand"):
        _ground_r1(GraspTargetRef(label="cup", contact_links=["left_gripper_link"]))
    with pytest.raises(ROSReasonerInvalidPlan, match="not gripper child links of one hand"):
        _ground_r1(GraspTargetRef(label="cup", contact_links=[_R1_LEFT[0], "torso_link1"]))


def test_a_fixture_place_target_grounds_to_its_place_declaration() -> None:
    decl = ground_place_target(
        PlaceTargetRef(fixture_id="cell:shelf_top"), unit=_UNIT, patience_s=60.0
    )
    assert decl.target_id == "cell:shelf_top"
    assert decl.region is None
    assert decl.timeout_s == 70.0


@pytest.mark.parametrize(
    ("ref", "unit", "match"),
    [
        (PlaceTargetRef(fixture_id="cell:nowhere"), _UNIT, "not a fixture of this robot unit"),
        (PlaceTargetRef(fixture_id="cell:shelf_top"), None, "fixtures: none"),
        (PlaceTargetRef(place_node_id="kitchen_table"), _UNIT, "not supported yet"),
    ],
)
def test_a_place_target_that_is_not_a_unit_fixture_refuses(
    ref: PlaceTargetRef, unit: RobotUnit | None, match: str
) -> None:
    with pytest.raises(ROSReasonerInvalidPlan, match=match):
        ground_place_target(ref, unit=unit, patience_s=60.0)


def test_place_target_ref_needs_exactly_one_target() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PlaceTargetRef()
    with pytest.raises(ValueError, match="exactly one"):
        PlaceTargetRef(fixture_id="cell:shelf_top", place_node_id="kitchen_table")


# ── LLM surface ─────────────────────────────────────────────────────────────


def test_the_targets_reach_the_llm_tool_schema_and_decode_back() -> None:
    palette = ToolPalette(execute_rskill_ids=frozenset({"openral/pick"}))
    tools = _tool_palette_to_anthropic_tools(palette)
    schema = next(t for t in tools if t["name"] == "execute_rskill")["input_schema"]
    assert isinstance(schema, dict)
    assert {"grasp_target", "place_target"} <= set(schema["properties"])
    assert {"GraspTargetRef", "PlaceTargetRef"} <= set(schema["$defs"])
    call = _decode_tool_payload(
        tool_name="execute_rskill",
        arguments={
            "rskill_id": "openral/pick",
            "grasp_target": {"label": "box", "object_id": "box_1"},
            "place_target": {"fixture_id": "cell:shelf_top"},
        },
        palette=palette,
    )
    assert isinstance(call, ExecuteRskillTool)
    assert call.grasp_target == GraspTargetRef(label="box", object_id="box_1")
    assert call.place_target == PlaceTargetRef(fixture_id="cell:shelf_top")


def test_the_unit_fixtures_are_listed_in_the_system_prompt() -> None:
    prompt = resolve_reasoner_system_prompt(_OPENARM.capabilities, env={}, fixtures=_UNIT.fixtures)
    assert "place_target.fixture_id choices" in prompt
    assert "- cell:shelf_top: " in prompt
    assert "grasp_target" in prompt  # the base brief explains how to name targets
    assert "fixture_id choices" not in resolve_reasoner_system_prompt(_OPENARM.capabilities, env={})
