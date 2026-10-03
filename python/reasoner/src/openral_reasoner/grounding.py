"""Ground an ``ExecuteRskillTool``'s named targets into goal-scoped declarations.

The reasoner **names** a grasp target (``GraspTargetRef``) and a place target
(``PlaceTargetRef``); this module **grounds** them at dispatch, from perception
the reasoner already holds, into the ``GraspDeclaration`` / ``PlaceDeclaration``
the ``ExecuteRskill`` goal carries; the HAL's producers then **measure** (real
pick-and-place design §2.2). Nothing here measures a region: a grasp declaration
leaves with ``region=None`` and a ``search_box`` seed only, a place declaration
with a fixture id only. The kernel's trust boundary is unchanged.

Pure and ROS-free (the reasoner node feeds it the latest lifted
``WorldState.detected_objects`` and its spatial-memory scene graph), so a
refusal is a typed ``ROSReasonerInvalidPlan`` the node logs and reports to the
LLM instead of sending the goal.
"""

from __future__ import annotations

from collections.abc import Sequence

from openral_core import (
    DetectedObject,
    GraspDeclaration,
    GraspTargetRef,
    PlaceDeclaration,
    PlaceRegion,
    PlaceTargetRef,
    Pose6D,
    RobotDescription,
    RobotUnit,
    SceneGraph,
)
from openral_core.exceptions import ROSReasonerInvalidPlan

__all__ = [
    "DECLARATION_TIMEOUT_MARGIN_S",
    "gripper_contact_links",
    "ground_grasp_target",
    "ground_place_target",
]

#: Seconds a declaration outlives its goal's own patience ceiling before the
#: backstop expires it; the runner retracts it at goal end anyway.
DECLARATION_TIMEOUT_MARGIN_S = 10.0


def gripper_contact_links(description: RobotDescription) -> tuple[str, ...]:
    """Every ``role: gripper`` joint's ``child_link``: the default grasp contact links.

    Example:
        >>> d = RobotDescription.from_yaml("robots/openarm/robot.yaml")
        >>> gripper_contact_links(d)
        ('openarm_left_finger_pair', 'openarm_right_finger_pair')
    """
    return tuple(j.child_link for j in description.joints if j.role == "gripper")


def _box(
    bbox: tuple[float, float, float, float, float, float] | None,
    frame_id: str,
    *,
    what: str,
    base_frame: str,
    pad_m: float,
) -> PlaceRegion:
    """Gravity-aligned seed box: ``bbox`` padded by ``pad_m`` sideways and upward only.

    Never padded downward: the producer reads the seed's bottom face as the support
    plane and keeps the measured region above it (HZ-01xx-6), so a bottom below the
    object would let the region reach into the surface it stands on. The bottom stays
    at the lifted box's lowest cell centre, which errs upward (safe) either way.
    """
    if bbox is None:
        raise ROSReasonerInvalidPlan(f"{what} has no 3D box; it cannot seed a grasp search.")
    if frame_id != base_frame:
        # ponytail: no tf2 in the reasoner; a fixed-base cell lifts in its base frame.
        raise ROSReasonerInvalidPlan(
            f"{what} is in frame {frame_id!r}, not the robot base frame {base_frame!r}; "
            "the grasp search box must be in the voxel grid's base frame."
        )
    lo, hi = bbox[:3], bbox[3:]
    top = hi[2] + pad_m
    return PlaceRegion(
        frame_id=base_frame,
        pose=Pose6D(
            xyz=((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + top) / 2.0),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id=base_frame,
        ),
        half_extents=(
            (hi[0] - lo[0]) / 2.0 + pad_m,
            (hi[1] - lo[1]) / 2.0 + pad_m,
            (top - lo[2]) / 2.0,
        ),
        evidence_ref=f"reasoner_seed:{what}",
    )


def ground_grasp_target(
    ref: GraspTargetRef,
    *,
    live_objects: Sequence[DetectedObject],
    scene_graph: SceneGraph | None,
    base_frame: str,
    default_contact_links: Sequence[str],
    patience_s: float,
    pad_m: float,
) -> GraspDeclaration:
    """Resolve a named grasp target to a seed-only ``GraspDeclaration``.

    ``ref.object_id`` set → that spatial-memory node's 3D box; else the single
    live lifted detection whose label equals ``ref.label`` (case-insensitive).
    The box, padded by ``pad_m`` (one voxel + extrinsic error) sideways and upward
    — never downward, its bottom face is the producer's support plane — and kept
    gravity-aligned, becomes ``search_box``; ``region`` stays ``None``.

    Args:
        ref: The reasoner's named target.
        live_objects: The latest lifted ``WorldState.detected_objects``.
        scene_graph: The spatial-memory snapshot, or ``None`` without memory.
        base_frame: The robot base frame (the voxel grid's frame).
        default_contact_links: Used when ``ref.contact_links`` is empty.
        patience_s: The goal's patience ceiling; the backstop is this plus
            ``DECLARATION_TIMEOUT_MARGIN_S``, capped at the declaration's ceiling.
        pad_m: Padding added to each half-extent, > 0.

    Raises:
        ROSReasonerInvalidPlan: Nothing grounds, several instances match with no
            ``object_id``, the box is missing / in another frame, or no contact
            link is known.

    Example:
        >>> from openral_core import DetectedObject, Pose6D
        >>> box = DetectedObject(
        ...     label="box",
        ...     confidence=0.9,
        ...     pose=Pose6D(
        ...         xyz=(0.4, 0.0, 0.1), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="openarm_base"
        ...     ),
        ...     bbox_3d=(0.35, -0.05, 0.05, 0.45, 0.05, 0.15),
        ... )
        >>> d = ground_grasp_target(
        ...     GraspTargetRef(label="Box"),
        ...     live_objects=[box],
        ...     scene_graph=None,
        ...     base_frame="openarm_base",
        ...     default_contact_links=("openarm_left_finger_pair",),
        ...     patience_s=60.0,
        ...     pad_m=0.04,
        ... )
        >>> d.region is None, d.target_id, d.timeout_s
        (True, 'obj:box', 70.0)
    """
    if pad_m <= 0.0:
        raise ValueError(f"pad_m must be > 0; got {pad_m!r}")
    links = tuple(ref.contact_links) or tuple(default_contact_links)
    if not links:
        raise ROSReasonerInvalidPlan(
            "grasp_target names no contact_links and the robot manifest has no gripper joint."
        )
    if ref.object_id is not None:
        nodes = [
            n for n in (scene_graph.nodes if scene_graph else []) if n.node_id == ref.object_id
        ]
        if not nodes:
            raise ROSReasonerInvalidPlan(
                f"grasp_target object_id {ref.object_id!r} is not in spatial memory; "
                "recall_object first and pass a returned node_id."
            )
        node = nodes[0]
        target = ref.object_id
        what = f"memory node {ref.object_id!r}"
        seed = _box(node.bbox_3d, node.pose.frame_id, what=what, base_frame=base_frame, pad_m=pad_m)
    else:
        label = ref.label.strip().casefold()
        matches = [o for o in live_objects if o.label.strip().casefold() == label]
        if not matches:
            seen = sorted({o.label for o in live_objects}) or ["nothing"]
            raise ROSReasonerInvalidPlan(
                f"grasp_target {ref.label!r}: no live 3D detection carries that label "
                f"(perception sees: {', '.join(seen)}). Look for it or recall_object it first."
            )
        if len(matches) > 1:
            raise ROSReasonerInvalidPlan(
                f"grasp_target {ref.label!r} is ambiguous: {len(matches)} live detections carry "
                "that label. recall_object it and pass the instance's node_id as object_id."
            )
        target = label
        seed = _box(
            matches[0].bbox_3d,
            matches[0].pose.frame_id,
            what=f"detection {ref.label!r}",
            base_frame=base_frame,
            pad_m=pad_m,
        )
    return GraspDeclaration(
        target_id=f"obj:{target}",
        object_id="",
        contact_links=links,
        timeout_s=min(patience_s + DECLARATION_TIMEOUT_MARGIN_S, GraspDeclaration.MAX_TIMEOUT_S),
        stamp_ns=0,
        search_box=seed,
    )


def ground_place_target(
    ref: PlaceTargetRef, *, unit: RobotUnit | None, patience_s: float
) -> PlaceDeclaration:
    """Resolve a named place target to a region-less ``PlaceDeclaration`` on a unit fixture.

    Raises:
        ROSReasonerInvalidPlan: A recalled place (``place_node_id``) — no free-space
            place producer exists yet — no robot unit is loaded, or the unit surveys
            no such fixture.

    Example:
        >>> unit = RobotUnit.from_yaml("tests/unit/fixtures/robot_units/openarm_shelf_cell.yaml")
        >>> ground_place_target(
        ...     PlaceTargetRef(fixture_id="cell:shelf_top"), unit=unit, patience_s=60.0
        ... ).target_id
        'cell:shelf_top'
    """
    if ref.fixture_id is None:
        raise ROSReasonerInvalidPlan(
            f"place_target place_node_id {ref.place_node_id!r}: placing at a recalled place is "
            "not supported yet; name one of the listed unit fixtures as fixture_id."
        )
    have = [f.id for f in unit.fixtures] if unit is not None else []
    if ref.fixture_id not in have:
        raise ROSReasonerInvalidPlan(
            f"place_target fixture_id {ref.fixture_id!r} is not a fixture of this robot unit "
            f"(fixtures: {', '.join(have) or 'none'})."
        )
    return PlaceDeclaration(
        target_id=ref.fixture_id,
        timeout_s=min(patience_s + DECLARATION_TIMEOUT_MARGIN_S, PlaceDeclaration.MAX_TIMEOUT_S),
        stamp_ns=0,
    )
