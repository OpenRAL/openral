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

With object tracking on (issue #418) a declaration also carries the classes it names
as ``ObjectQuery`` s (``GraspDeclaration.target_queries``,
``PlaceDeclaration.destination_query``) for the Layer-2 tracker, beside the unchanged
``search_box`` path; a label several live detections carry is then no longer refused —
the tracker grounds every instance (``count="all"``) and the policy's choice is resolved
at binding — and neither is one no live detection carries: the tracker's standing query
keeps looking and escalates to EXHAUSTED itself (Addendum G).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from openral_core import (
    DetectedObject,
    GraspDeclaration,
    GraspTargetRef,
    ObjectQuery,
    ObjectRole,
    PlaceDeclaration,
    PlaceRegion,
    PlaceTargetRef,
    Pose6D,
    SceneGraph,
    gripper_hands,
    hand_joint,
)
from openral_core.exceptions import ROSConfigError, ROSReasonerInvalidPlan

__all__ = [
    "DECLARATION_TIMEOUT_MARGIN_S",
    "gripper_hands",
    "ground_grasp_target",
    "ground_place_target",
    "ground_workspace",
    "hand_joint",
    "placement_queries",
]

#: Seconds a declaration outlives its goal's own patience ceiling before the
#: backstop expires it; the runner retracts it at goal end anyway.
DECLARATION_TIMEOUT_MARGIN_S = 10.0

#: A scene object's instance suffix (``orange_1``): instances of one class share a label.
_INSTANCE_SUFFIX = re.compile(r"_\d+$")


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
    several_ok: bool = False,
) -> tuple[str, PlaceRegion | None, int]:
    """``(target id stem, padded search box, live matches)`` for a named object or surface.

    ``node_id`` set → that spatial-memory node's 3D box; else the single live lifted
    detection whose label equals ``label`` (case-insensitive). ``field`` names the
    tool-call field in refusals. With ``several_ok`` (object tracking) several matches —
    or none — ground with no search box: the tracker grounds every instance by its query,
    and keeps looking for a label no camera sees yet (#418 Addendum G).

    Raises:
        ROSReasonerInvalidPlan: Nothing grounds (unless ``several_ok``), several instances
            match with no node id (unless ``several_ok``), an unknown node id, or the box
            is missing / in another frame.
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
        box = _box(node.bbox_3d, node.pose.frame_id, what=what, base_frame=base_frame, pad_m=pad_m)
        return node_id, box, 1
    key = label.strip().casefold()
    matches = [o for o in live_objects if o.label.strip().casefold() == key]
    if not matches and several_ok:
        # Addendum G: the tracker's standing query keeps looking (passive every frame,
        # then find-object) and escalates to EXHAUSTED itself; never a first-frame refusal.
        return key, None, 0
    if not matches:
        seen = sorted({o.label for o in live_objects}) or ["nothing"]
        raise ROSReasonerInvalidPlan(
            f"{field} {label!r}: no live 3D detection carries that label "
            f"(perception sees: {', '.join(seen)}). Look for it or recall_object it first."
        )
    if len(matches) > 1 and several_ok:
        return key, None, len(matches)
    if len(matches) > 1:
        raise ROSReasonerInvalidPlan(
            f"{field} {label!r} is ambiguous: {len(matches)} live detections carry "
            "that label. recall_object it and pass the instance's node_id as object_id."
        )
    box = _box(
        matches[0].bbox_3d,
        matches[0].pose.frame_id,
        what=f"detection {label!r}",
        base_frame=base_frame,
        pad_m=pad_m,
    )
    return key, box, 1


def ground_grasp_target(
    ref: GraspTargetRef,
    *,
    live_objects: Sequence[DetectedObject],
    scene_graph: SceneGraph | None,
    base_frame: str,
    default_contact_links: Sequence[str | Sequence[str]],
    patience_s: float,
    pad_m: float,
    object_tracking: bool = False,
    hand: str | None = None,
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
        object_tracking: The deploy runs the Layer-2 object tracker (issue #418): the
            declaration also carries ``target_queries=[ObjectQuery(label, count, TARGET,
            hand)]``, ``count="all"`` when several live detections carry the label —
            which then grounds (no ``search_box``) instead of being refused. A label no
            live detection carries grounds too (``count="one"``, no ``search_box``): the
            tracker's standing query keeps looking and escalates itself (Addendum G).
        hand: The grasping hand's gripper joint (``hand_joint``), a hint on the query.

    Raises:
        ROSReasonerInvalidPlan: Nothing grounds, several instances match with no
            ``object_id`` (without ``object_tracking``), the box is missing / in another
            frame, no contact link is
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
    target, seed, matches = _locate(
        "grasp_target",
        ref.label,
        ref.object_id,
        live_objects=live_objects,
        scene_graph=scene_graph,
        base_frame=base_frame,
        pad_m=pad_m,
        several_ok=object_tracking,
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
        target_queries=(
            (
                ObjectQuery(
                    label=ref.label.strip(),
                    count="all" if matches > 1 else "one",
                    role=ObjectRole.TARGET,
                    hand=hand,
                ),
            )
            if object_tracking
            else ()
        ),
    )


def ground_place_target(
    ref: PlaceTargetRef,
    *,
    live_objects: Sequence[DetectedObject],
    scene_graph: SceneGraph | None,
    base_frame: str,
    patience_s: float,
    pad_m: float,
    object_tracking: bool = False,
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
        object_tracking: The deploy runs the Layer-2 object tracker (issue #418): the
            declaration also carries ``destination_query=ObjectQuery(label, "one",
            DESTINATION)``, and a label several live detections carry — or none —
            grounds with no ``search_box`` instead of being refused (Addendum G).

    The hint is optional (the place leg measures the surface under the carried payload
    without one), but a supplied hint that does not ground refuses the goal: it never
    degrades to placing on whatever surface is measured.

    Raises:
        ROSReasonerInvalidPlan: The hint does not ground — no live detection carries the
            label (including no detector at all) or several do and no node id is given
            (both only without ``object_tracking``), the
            node is not in spatial memory (including no memory at all), or the box is
            missing / in another frame. The message says the hint could not be grounded
            and that omitting ``place_target`` lets the policy place on the measured surface.

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
    try:
        target, seed, _ = _locate(
            "place_target",
            ref.label,
            node_id,
            live_objects=live_objects,
            scene_graph=scene_graph,
            base_frame=base_frame,
            pad_m=pad_m,
            several_ok=object_tracking,
        )
    except ROSReasonerInvalidPlan as exc:
        # Option A: the hint is optional, but a supplied one that does not ground refuses
        # the goal — never a silent fall-back to "place on any measured surface".
        raise ROSReasonerInvalidPlan(
            f"the place hint could not be grounded, so the goal is refused: {exc!s} "
            "Omit place_target to let the policy place on the surface measured under "
            "the carried payload."
        ) from exc
    return PlaceDeclaration(
        target_id=f"surface:{target}",
        timeout_s=min(patience_s + DECLARATION_TIMEOUT_MARGIN_S, PlaceDeclaration.MAX_TIMEOUT_S),
        stamp_ns=0,
        search_box=seed,
        destination_query=(
            ObjectQuery(label=ref.label.strip(), count="one", role=ObjectRole.DESTINATION)
            if object_tracking
            else None
        ),
    )


def placement_queries(placements: Sequence[tuple[str, str | None]]) -> tuple[ObjectQuery, ...]:
    """The standing queries a scene task names (issue #418): its object classes and destination.

    ``placements`` are a prompt's ``(object, destination object)`` pairs — e.g. a graded
    scene's task list, each target region's ``relative_to`` (``None`` for a fixed region
    such as a shelf slot, which stays a ``PlaceRegion``). An object's instance suffix is
    dropped (``orange_1`` → ``orange``) and underscores read as spaces, so instances share
    one open-vocabulary label; a class placed more than once is ``count="all"``. One
    ``TARGET`` query per class, in task order, then at most one ``DESTINATION`` query.
    The derivation suits a class-like name (``cracker_box`` → ``cracker box``); a prim
    name that is no class (``stock_object``, ``carton_b``) is passed through as is, so a
    scene names its detector label explicitly (``IsaacPlacement.label``).

    Raises:
        ROSConfigError: No placement, or several destination objects (a
            ``PlaceDeclaration`` carries one).

    Example:
        >>> [
        ...     (q.label, q.count, q.role.value)
        ...     for q in placement_queries(
        ...         [("orange_1", "plate"), ("orange_2", "plate"), ("orange_3", "plate")]
        ...     )
        ... ]
        [('orange', 'all', 'target'), ('plate', 'one', 'destination')]
    """
    if not placements:
        raise ROSConfigError("placement_queries needs at least one (object, destination) pair.")
    counts: dict[str, int] = {}
    for name, _ in placements:
        label = _INSTANCE_SUFFIX.sub("", name).replace("_", " ")
        counts[label] = counts.get(label, 0) + 1
    destinations = sorted({d for _, d in placements if d is not None})
    if len(destinations) > 1:
        raise ROSConfigError(
            f"placement_queries: the task places into {destinations}; a PlaceDeclaration "
            "carries one destination_query."
        )
    queries = [
        ObjectQuery(label=label, count="all" if n > 1 else "one", role=ObjectRole.TARGET)
        for label, n in counts.items()
    ]
    queries += [
        ObjectQuery(
            label=_INSTANCE_SUFFIX.sub("", d).replace("_", " "),
            count="one",
            role=ObjectRole.DESTINATION,
        )
        for d in destinations
    ]
    return tuple(queries)


Bbox = tuple[float, float, float, float, float, float]


def ground_workspace(
    source: Bbox,
    destination: Bbox | None,
    *,
    anchor_xyz: tuple[float, float, float],
    base_frame: str,
    pad_m: float,
    headroom_m: float,
    evidence_ref: str,
) -> PlaceRegion:
    """The manipulation workspace a pick-and-place works in, as a ``workspace_box``.

    The gravity-aligned hull of the ``source`` fixture the target lies in (a tote's
    walls and floor, a counter top) and the ``destination`` it goes to (a shelf slot,
    a cabinet), padded by ``pad_m`` on every side plus ``headroom_m`` on top (the hand
    rides above what it carries). Each box is ``(x0, y0, z0, x1, y1, z1)`` in
    ``base_frame`` — from sim ground truth, or from the scene graph's surface boxes on
    a real cell. It is a declaration like ``search_box``: it arms nothing until the
    producer confirms it against the voxel map (``workspace_region``).

    Never wider than ``GraspDeclaration``'s workspace caps: an axis longer than twice
    ``MAX_WORKSPACE_HALF_EXTENT_M`` is cut to that window centred on ``anchor_xyz``
    (the target object), keeping the pick side and as much of the destination as fits.

    Raises:
        ROSReasonerInvalidPlan: the clipped box still exceeds the volume cap.

    Example:
        >>> ws = ground_workspace(
        ...     (0.0, -0.2, -0.6, 0.3, 0.2, -0.45),
        ...     (0.3, 0.0, -0.45, 0.7, 0.15, -0.3),
        ...     anchor_xyz=(0.15, 0.0, -0.55),
        ...     base_frame="openarm_base",
        ...     pad_m=0.02,
        ...     headroom_m=0.05,
        ...     evidence_ref="sim:tote+slot",
        ... )
        >>> [round(v, 3) for v in ws.pose.xyz], [round(v, 3) for v in ws.half_extents]
        ([0.35, 0.0, -0.425], [0.37, 0.22, 0.195])
    """
    cap = GraspDeclaration.MAX_WORKSPACE_HALF_EXTENT_M
    boxes = [source] if destination is None else [source, destination]
    lo = [min(b[i] for b in boxes) - pad_m for i in range(3)]
    hi = [max(b[i + 3] for b in boxes) + pad_m for i in range(3)]
    hi[2] += headroom_m
    for i in range(3):
        if hi[i] - lo[i] > 2.0 * cap:
            lo[i], hi[i] = max(lo[i], anchor_xyz[i] - cap), min(hi[i], anchor_xyz[i] + cap)
    half = ((hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0, (hi[2] - lo[2]) / 2.0)
    if 8.0 * half[0] * half[1] * half[2] > GraspDeclaration.MAX_WORKSPACE_VOLUME_M3:
        raise ROSReasonerInvalidPlan(
            f"workspace {[round(2 * h, 3) for h in half]} m exceeds the "
            f"{GraspDeclaration.MAX_WORKSPACE_VOLUME_M3} m^3 cap; declare a tighter one."
        )
    return PlaceRegion(
        frame_id=base_frame,
        pose=Pose6D(
            xyz=((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id=base_frame,
        ),
        half_extents=half,
        evidence_ref=evidence_ref,
    )
