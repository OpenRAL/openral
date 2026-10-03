"""Ground an ``ExecuteRskillTool``'s named targets into goal-scoped declarations.

The reasoner **names** a grasp target (``GraspTargetRef``) and a place target
(``PlaceTargetRef``); this module **grounds** them at dispatch, from perception
the reasoner already holds, into the ``GraspDeclaration`` / ``PlaceDeclaration``
the ``ExecuteRskill`` goal carries; the HAL's producers then **measure** (real
pick-and-place design §2.2/§2.3). Nothing here measures a region: both declarations
leave with ``region=None`` and a ``search_box`` seed only — the padded box of the
named object (grasp) or surface (place). The kernel's trust boundary is unchanged.

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
    SceneGraph,
    gripper_hands,
)
from openral_core.exceptions import ROSReasonerInvalidPlan

__all__ = [
    "DECLARATION_TIMEOUT_MARGIN_S",
    "gripper_hands",
    "ground_grasp_target",
    "ground_place_target",
]

#: Seconds a declaration outlives its goal's own patience ceiling before the
#: backstop expires it; the runner retracts it at goal end anyway.
DECLARATION_TIMEOUT_MARGIN_S = 10.0


def _box(
    bbox: tuple[float, float, float, float, float, float] | None,
    frame_id: str,
    *,
    what: str,
    base_frame: str,
    pad_m: float,
) -> PlaceRegion:
    """Gravity-aligned search box: ``bbox`` padded by ``pad_m`` on every side.

    A search hint only, with no support semantics: the producer measures the support
    layer under the target from the voxel map itself. The lifted box's bottom is the
    lowest occupied cell centre inside the detection's frustum, which can sit above the
    true support (the object's bottom cell) or below it (table cells caught by a loose
    pixel box), so nothing here may read it as a plane. The downward pad (one voxel +
    the extrinsic error, like the others) keeps the support layer inside the box, so
    the producer can find it there; it is bounded so the box does not reach far below.
    """
    if bbox is None:
        raise ROSReasonerInvalidPlan(f"{what} has no 3D box; it cannot seed a search.")
    if frame_id != base_frame:
        # ponytail: no tf2 in the reasoner; a fixed-base cell lifts in its base frame.
        raise ROSReasonerInvalidPlan(
            f"{what} is in frame {frame_id!r}, not the robot base frame {base_frame!r}; "
            "a search box must be in the voxel grid's base frame."
        )
    x0, y0, z0, x1, y1, z1 = bbox
    return PlaceRegion(
        frame_id=base_frame,
        pose=Pose6D(
            xyz=((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id=base_frame,
        ),
        half_extents=(
            (x1 - x0) / 2.0 + pad_m,
            (y1 - y0) / 2.0 + pad_m,
            (z1 - z0) / 2.0 + pad_m,
        ),
        evidence_ref=f"reasoner_seed:{what}",
    )


def _locate(
    field: str,
    label: str,
    node_id: str | None,
    *,
    live_objects: Sequence[DetectedObject],
    scene_graph: SceneGraph | None,
    base_frame: str,
    pad_m: float,
) -> tuple[str, PlaceRegion]:
    """``(target id stem, padded search box)`` for a named object or surface.

    ``node_id`` set → that spatial-memory node's 3D box; else the single live lifted
    detection whose label equals ``label`` (case-insensitive). ``field`` names the
    tool-call field in refusals.

    Raises:
        ROSReasonerInvalidPlan: Nothing grounds, several instances match with no node
            id, or the box is missing / in another frame.
    """
    if pad_m <= 0.0:
        raise ValueError(f"pad_m must be > 0; got {pad_m!r}")
    if node_id is not None:
        nodes = [n for n in (scene_graph.nodes if scene_graph else []) if n.node_id == node_id]
        if not nodes:
            raise ROSReasonerInvalidPlan(
                f"{field} node {node_id!r} is not in spatial memory; "
                "recall_object first and pass a returned node_id."
            )
        node = nodes[0]
        what = f"memory node {node_id!r}"
        return node_id, _box(
            node.bbox_3d, node.pose.frame_id, what=what, base_frame=base_frame, pad_m=pad_m
        )
    key = label.strip().casefold()
    matches = [o for o in live_objects if o.label.strip().casefold() == key]
    if not matches:
        seen = sorted({o.label for o in live_objects}) or ["nothing"]
        raise ROSReasonerInvalidPlan(
            f"{field} {label!r}: no live 3D detection carries that label "
            f"(perception sees: {', '.join(seen)}). Look for it or recall_object it first."
        )
    if len(matches) > 1:
        raise ROSReasonerInvalidPlan(
            f"{field} {label!r} is ambiguous: {len(matches)} live detections carry "
            "that label. recall_object it and pass the instance's node_id as object_id."
        )
    return key, _box(
        matches[0].bbox_3d,
        matches[0].pose.frame_id,
        what=f"detection {label!r}",
        base_frame=base_frame,
        pad_m=pad_m,
    )


def ground_grasp_target(
    ref: GraspTargetRef,
    *,
    live_objects: Sequence[DetectedObject],
    scene_graph: SceneGraph | None,
    base_frame: str,
    default_contact_links: Sequence[str | Sequence[str]],
    patience_s: float,
    pad_m: float,
) -> GraspDeclaration:
    """Resolve a named grasp target to a seed-only ``GraspDeclaration``.

    ``ref.object_id`` set → that spatial-memory node's 3D box; else the single
    live lifted detection whose label equals ``ref.label`` (case-insensitive).
    The box, padded by ``pad_m`` (one voxel + extrinsic error) on every side and kept
    gravity-aligned, becomes ``search_box`` — a search hint for the producer, which
    measures the support layer itself; ``region`` stays ``None``.

    Args:
        ref: The reasoner's named target.
        live_objects: The latest lifted ``WorldState.detected_objects``.
        scene_graph: The spatial-memory snapshot, or ``None`` without memory.
        base_frame: The robot base frame (the voxel grid's frame).
        default_contact_links: The robot's hands (``gripper_hands``), each its gripper
            child links (a bare ``str`` is a one-link hand). Named ``ref.contact_links``
            must be ALL the links of ONE hand (any order) — part of a hand would exempt
            one finger only and the kernel would stop the grasp; empty ones default to the
            hand of a single-hand robot — with several, defaulting would exempt every hand.
        patience_s: The goal's patience ceiling; the backstop is this plus
            ``DECLARATION_TIMEOUT_MARGIN_S``, capped at the declaration's ceiling.
        pad_m: Padding added to each half-extent, > 0.

    Raises:
        ROSReasonerInvalidPlan: Nothing grounds, several instances match with no
            ``object_id``, the box is missing / in another frame, no contact link is
            known, none is named on a robot with more than one hand, or the named
            links are not gripper child links of one hand, or are only part of one.

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
    target, seed = _locate(
        "grasp_target",
        ref.label,
        ref.object_id,
        live_objects=live_objects,
        scene_graph=scene_graph,
        base_frame=base_frame,
        pad_m=pad_m,
    )
    hands = [(h,) if isinstance(h, str) else tuple(h) for h in default_contact_links]
    listing = "; ".join(f"[{', '.join(h)}]" for h in hands) or "none"
    links = tuple(ref.contact_links)
    if links:
        owners = [h for h in hands if set(links) & set(h)]
        if len(owners) != 1 or not set(links) <= set(owners[0]):
            raise ROSReasonerInvalidPlan(
                f"grasp_target contact_links {list(links)} are not gripper child links of "
                f"one hand; name links of exactly one hand (hands: {listing})."
            )
        # A partial hand would exempt one finger only and the kernel would stop the grasp.
        if set(links) != set(owners[0]):
            raise ROSReasonerInvalidPlan(
                f"grasp_target contact_links {list(links)} name part of a hand; name all of "
                f"its links: [{', '.join(owners[0])}]."
            )
        links = owners[0]
    elif len(hands) > 1:
        raise ROSReasonerInvalidPlan(
            f"grasp_target names no contact_links and this robot has {len(hands)} hands; "
            f"name the one hand that grasps as contact_links (hands: {listing})."
        )
    elif hands:
        links = hands[0]
    else:
        raise ROSReasonerInvalidPlan(
            "grasp_target names no contact_links and the robot manifest has no gripper joint."
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
    ref: PlaceTargetRef,
    *,
    live_objects: Sequence[DetectedObject],
    scene_graph: SceneGraph | None,
    base_frame: str,
    patience_s: float,
    pad_m: float,
) -> PlaceDeclaration:
    """Resolve a named place surface to a region-less ``PlaceDeclaration`` with a search box.

    The mirror of ``ground_grasp_target``: ``ref.object_id`` / ``ref.place_node_id`` set
    → that spatial-memory node's 3D box; else the single live lifted detection whose
    label equals ``ref.label``. The box, padded by ``pad_m`` (one voxel + extrinsic
    error) and gravity-aligned, becomes ``search_box`` — the volume the place producer
    measures the support surface in. It is never a region and never a plane: no
    surveyed or predeclared cell geometry is involved.

    Args:
        ref: The reasoner's named surface.
        live_objects: The latest lifted ``WorldState.detected_objects``.
        scene_graph: The spatial-memory snapshot, or ``None`` without memory.
        base_frame: The robot base frame (the voxel grid's frame).
        patience_s: The goal's patience ceiling; the backstop is this plus
            ``DECLARATION_TIMEOUT_MARGIN_S``, capped at the declaration's ceiling.
        pad_m: Padding added to each half-extent, > 0.

    Raises:
        ROSReasonerInvalidPlan: Nothing grounds, several instances match with no node
            id, or the box is missing / in another frame.

    Example:
        >>> from openral_core import DetectedObject, Pose6D
        >>> shelf = DetectedObject(
        ...     label="shelf",
        ...     confidence=0.8,
        ...     pose=Pose6D(
        ...         xyz=(0.5, 0.3, 0.3), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="openarm_base"
        ...     ),
        ...     bbox_3d=(0.3, 0.1, 0.0, 0.7, 0.5, 0.6),
        ... )
        >>> d = ground_place_target(
        ...     PlaceTargetRef(label="Shelf"),
        ...     live_objects=[shelf],
        ...     scene_graph=None,
        ...     base_frame="openarm_base",
        ...     patience_s=60.0,
        ...     pad_m=0.04,
        ... )
        >>> d.target_id, d.region is None, [round(v, 3) for v in d.search_box.half_extents]
        ('surface:shelf', True, [0.24, 0.24, 0.34])
    """
    node_id = ref.object_id if ref.object_id is not None else ref.place_node_id
    target, seed = _locate(
        "place_target",
        ref.label,
        node_id,
        live_objects=live_objects,
        scene_graph=scene_graph,
        base_frame=base_frame,
        pad_m=pad_m,
    )
    return PlaceDeclaration(
        target_id=f"surface:{target}",
        timeout_s=min(patience_s + DECLARATION_TIMEOUT_MARGIN_S, PlaceDeclaration.MAX_TIMEOUT_S),
        stamp_ns=0,
        search_box=seed,
    )
