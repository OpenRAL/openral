#!/usr/bin/env python3
"""Bucket-2 converter node: OpenRAL custom msgs → standard ROS viz types.

- ``/openral/world_voxels`` (``OccupancyVoxels``) → ``/openral/world_voxels_cloud``
  (``PointCloud2``): one point per occupied voxel, at the voxel centre.
- ``/openral/attachment_state`` (``AttachmentState``) → ``/openral/viz/attachments``
  (``visualization_msgs/MarkerArray``): every attached object's collision
  primitives on its attach link (and the measured inner box a payload's stage-2
  refinement carries), plus the grasp / place declaration regions and the grasp
  declaration's search box and manipulation workspace.
- Parameter ``static_geometry_file`` (``DeployRuntime.static_geometry_file``) →
  ``/openral/viz/static_geometry`` (latched ``MarkerArray``): the declared fixtures
  the kernel always treats as occupied, drawn as their boxes and triangle meshes.

Conversion math lives in pure functions (``occupied_voxel_centers``,
``attachment_markers``) so unit tests exercise it without a ROS context
(CLAUDE.md §1.11). ``rclpy``/``openral_msgs`` imports are deferred inside node
methods (PLC0415, ruff-exempt for ``packages/**``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import structlog

__all__ = [
    "ATTACHMENT_MARKERS_TOPIC",
    "STATIC_GEOMETRY_MARKERS_TOPIC",
    "Bucket2MarkersNode",
    "attachment_markers",
    "main",
    "occupied_voxel_centers",
    "parse_attach_link_tf_frames",
    "static_geometry_markers",
]

#: Where the attachment / declaration-region markers are published.
ATTACHMENT_MARKERS_TOPIC = "/openral/viz/attachments"
#: Where the declared static geometry is published (latched, once).
STATIC_GEOMETRY_MARKERS_TOPIC = "/openral/viz/static_geometry"

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure data types and conversion functions (no ROS imports)
# ---------------------------------------------------------------------------


def occupied_voxel_centers(
    origin: tuple[float, float, float],
    resolution: float,
    size: tuple[int, int, int],
    occupancy: Sequence[int],
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> list[tuple[float, float, float]]:
    """Return the centre positions of all occupied voxels.

    Row-major indexing, x fastest: ``idx = x + size_x * (y + size_y * z)``.
    ``openral_msgs/OccupancyVoxels`` is an ORIENTED lattice — its cell axes
    are the source map's, not ``header.frame_id``'s — so dropping the
    rotation draws every voxel in the wrong place whenever the robot isn't
    map-aligned.

    Args:
        origin: (ox, oy, oz) minimum corner of voxel (0, 0, 0) in metres.
        resolution: Edge length of one voxel in metres.
        size: (size_x, size_y, size_z) grid dimensions.
        occupancy: Flat occupancy array (length size_x*size_y*size_z).
            Non-zero → occupied.
        orientation_xyzw: Grid orientation ``(x, y, z, w)``. Defaults to
            identity for callers building an axis-aligned grid themselves; a
            caller decoding a wire message passes the message's own. An all-zero
            quaternion (the unset default) is rejected, never read as identity.

    Returns:
        List of (cx, cy, cz) centre coordinates for occupied voxels.

    Raises:
        ValueError: On a length mismatch, or a non-unit ``orientation_xyzw``.
    """
    sx, sy, sz = size
    expected = sx * sy * sz
    if len(occupancy) != expected:
        raise ValueError(f"occupancy length {len(occupancy)} != size_x*size_y*size_z={expected}")
    qx, qy, qz, qw = orientation_xyzw
    norm2 = qx * qx + qy * qy + qz * qz + qw * qw
    if not math.isfinite(norm2) or abs(norm2 - 1.0) > 1e-6:
        raise ValueError(
            f"orientation {orientation_xyzw!r} is not a unit quaternion "
            "(an unset OccupancyVoxels.orientation is all zeros, not identity)"
        )
    rot = np.array(
        (
            (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
            (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
            (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
        )
    )
    # Vectorised: the grid is ~10^6 cells (r = 1.05 m at 20 mm) at several Hz; a Python
    # loop over it pinned a Thor core (2026-10-02) and starved octomap_server beside it.
    idx = np.flatnonzero(np.asarray(occupancy))
    x = idx % sx
    y = (idx // sx) % sy
    z = idx // (sx * sy)
    local = (np.stack((x, y, z), axis=1) + 0.5) * resolution
    centers = local @ rot.T + np.array(origin)
    return [tuple(c) for c in centers.tolist()]


def parse_attach_link_tf_frames(entries: Sequence[str]) -> dict[str, str]:
    """Parse ``"link=frame"`` entries into a manifest-link → TF-frame map.

    The same strings, and the same rules, as the octomap bridge's
    ``attach_link_tf_frames`` parameter (``payload_clearing.hpp``): empty
    entries are skipped (the ROS default ``[""]`` means none); a malformed
    entry or one link mapped to two frames is an error.

    Args:
        entries: ``"link=frame"`` strings, e.g. the scene's
            ``vision_attachment.tf_frames``.

    Returns:
        ``{link: frame}``; a link absent from it is its own TF frame.

    Raises:
        ValueError: On an entry without ``=``, with an empty side, or a link
            mapped to two different frames.

    Example:
        >>> parse_attach_link_tf_frames(["openarm_right_link7=openarm_right_ee_base_link", ""])
        {'openarm_right_link7': 'openarm_right_ee_base_link'}
    """
    out: dict[str, str] = {}
    for entry in entries:
        if not entry:
            continue
        link, sep, frame = entry.partition("=")
        if not sep or not link or not frame:
            raise ValueError(f"attach_link_tf_frames entry {entry!r} is not 'link=frame'")
        if out.get(link, frame) != frame:
            raise ValueError(f"attach link {link!r} mapped to both {out[link]!r} and {frame!r}")
        out[link] = frame
    return out


#: Payload colour per ``AttachedCollisionObject.evidence_kind`` (RGBA): what
#: measured the primitive decides how far to trust it, so it is the first thing
#: a viewer should read off the shape.
EVIDENCE_RGBA: dict[str, tuple[float, float, float, float]] = {
    "grasp_target_region": (1.0, 0.55, 0.0, 0.45),  # orange — pre-grasp region payload
    "vision_segmentation": (0.0, 0.8, 1.0, 0.45),  # cyan — wrist-view segmentation
    "gripper_closure": (1.0, 0.9, 0.0, 0.45),  # yellow — conservative jaw-span box
}
#: Any other evidence kind (sim ground truth, operator, perception track).
OTHER_EVIDENCE_RGBA = (0.8, 0.3, 1.0, 0.45)
#: Declaration regions: active grasp green, active place blue, retracted grey.
GRASP_REGION_RGBA = (0.1, 0.9, 0.2, 0.2)
PLACE_REGION_RGBA = (0.2, 0.4, 1.0, 0.2)
INACTIVE_REGION_RGBA = (0.6, 0.6, 0.6, 0.12)
#: The grasp declaration's search box (where the target is looked for): faint white.
SEARCH_BOX_RGBA = (0.9, 0.9, 0.9, 0.08)
#: The manipulation workspace (#349): amber once map-confirmed, grey while only declared.
WORKSPACE_RGBA = (1.0, 0.75, 0.1, 0.1)
#: A payload's measured inner box (stage-2 refinement): the shape payload-vs-robot uses.
MEASURED_RGBA = (1.0, 1.0, 1.0, 0.6)
#: Declared static geometry: always-occupied fixtures.
STATIC_RGBA = (0.85, 0.3, 0.3, 0.25)

#: ``evidence_ref`` suffix of a released payload frozen on the base frame
#: (``openral_hal.vision_attachment_bridge.freeze_released_attachment``).
_FROZEN_RELEASE_SUFFIX = "|frozen_release"

# AttachedCollisionPrimitive.SHAPE_* (kept literal so the module imports without ROS).
_SHAPE_SPHERE, _SHAPE_CAPSULE, _SHAPE_BOX = 1, 2, 3
_SHAPE_DIMS = {_SHAPE_SPHERE: 1, _SHAPE_CAPSULE: 2, _SHAPE_BOX: 3}
_LABEL_RGBA = (1.0, 1.0, 1.0, 0.9)


def attachment_markers(state: Any, attach_link_tf_frames: Mapping[str, str] | None = None) -> Any:
    """Convert an ``openral_msgs/AttachmentState`` into a ``visualization_msgs/MarkerArray``.

    The array opens with one ``DELETEALL`` and then re-adds everything, so an
    object that left the attachment set (a detach, a closed release window)
    disappears with the next message instead of lingering.

    - Each attached object's primitives — ``CUBE`` (box), ``SPHERE``, and a
      capsule as a ``CYLINDER`` plus two end-cap ``SPHERE`` markers — in the
      attach link's TF frame at ``pose_in_link ∘ pose_in_object``, frame-locked
      so they ride the link. Semi-transparent, coloured by ``evidence_kind``
      (:data:`EVIDENCE_RGBA`), with a text label: short object id, evidence
      kind, ``support`` when ``support_contact_valid``, ``released`` for a
      frozen release record.
    - The grasp and place declarations' measured regions (only when the
      declaration and its ``region`` are both valid) as a translucent box plus
      wireframe edges in the region's own frame, green / blue while active and
      grey once retracted, labelled with the target id.
    - A box primitive's measured inner box (its stage-2 refinement, the first three
      ``tight_dop_lo``/``tight_dop_hi`` axes) as a white ``CUBE`` in ns
      ``attached_measured``: the shape the kernel checks payload-vs-robot with.
    - The grasp declaration's ``search_box`` (faint white) and manipulation
      workspace: ``workspace_region`` in amber once map-confirmed, else the declared
      ``workspace_box`` in grey.

    Args:
        state: An ``openral_msgs/AttachmentState``.
        attach_link_tf_frames: Manifest attach link → TF frame
            (:func:`parse_attach_link_tf_frames`). A link not in it is its own
            frame. Also applied to a region's ``frame_id``.

    Returns:
        The ``visualization_msgs/MarkerArray``, ``DELETEALL`` first.

    Raises:
        ValueError: On an unknown primitive shape or a wrong dimension count.
        openral_core.ROSConfigError: On a degenerate (all-zero) quaternion.
    """
    from visualization_msgs.msg import Marker, MarkerArray

    frames = attach_link_tf_frames or {}
    stamp = state.header.stamp
    adds: list[Any] = []
    for obj in state.objects:
        adds += _object_markers(obj, frames.get(obj.attach_link, obj.attach_link))
    declarations = (
        ("grasp", state.grasp_declaration_valid, state.grasp_declaration, GRASP_REGION_RGBA),
        ("place", state.place_declaration_valid, state.place_declaration, PLACE_REGION_RGBA),
    )
    for name, valid, decl, active_rgba in declarations:
        if valid and decl.region_valid:
            frame = frames.get(decl.region.frame_id, decl.region.frame_id)
            rgba = active_rgba if decl.active else INACTIVE_REGION_RGBA
            text = f"{name}: {decl.target_id}" + ("" if decl.active else " (inactive)")
            adds += _region_markers(f"{name}_region", decl.region, frame, rgba, text)
    if state.grasp_declaration_valid:
        g = state.grasp_declaration
        boxes = [("search_box", g.search_box_valid, g.search_box, SEARCH_BOX_RGBA, "search box")]
        if g.workspace_region_valid:
            boxes.append(
                ("workspace", True, g.workspace_region, WORKSPACE_RGBA, "workspace (confirmed)")
            )
        else:
            boxes.append(
                (
                    "workspace",
                    g.workspace_box_valid,
                    g.workspace_box,
                    INACTIVE_REGION_RGBA,
                    "workspace (declared, unconfirmed)",
                )
            )
        for ns, valid, region, rgba, text in boxes:
            if valid:
                adds += _region_markers(
                    ns, region, frames.get(region.frame_id, region.frame_id), rgba, text
                )
    for i, m in enumerate(adds, start=1):
        m.id = i
        m.header.stamp = stamp
    return MarkerArray(markers=[Marker(action=Marker.DELETEALL), *adds])


def _mat(pose: Any) -> Any:
    """``geometry_msgs/Pose`` -> ``(4, 4)`` homogeneous transform."""
    from openral_core.geometry import homogeneous_from_quat_xyz

    p, q = pose.position, pose.orientation
    return homogeneous_from_quat_xyz((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))


def _pose_of(t: Any) -> Any:
    """``(4, 4)`` homogeneous transform -> ``geometry_msgs/Pose``."""
    from geometry_msgs.msg import Point, Pose, Quaternion
    from openral_core.geometry import rotation_to_quat_wxyz

    w, x, y, z = rotation_to_quat_wxyz(t[:3, :3])
    return Pose(
        position=Point(x=float(t[0, 3]), y=float(t[1, 3]), z=float(t[2, 3])),
        orientation=Quaternion(x=x, y=y, z=z, w=w),
    )


def _marker(
    ns: str,
    frame: str,
    kind: int,
    pose: Any,
    scale: Sequence[float],
    rgba: Sequence[float],
    text: str = "",
) -> Any:
    """One ``ADD`` marker, frame-locked (id and stamp are set by the caller)."""
    from visualization_msgs.msg import Marker

    m = Marker(ns=ns, type=kind, action=Marker.ADD, text=text, pose=pose)
    m.header.frame_id = frame
    # Ride the frame: the attachment state is republished on change only, so a
    # marker posed once must follow its link as the arm moves.
    m.frame_locked = True
    m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
    m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in rgba)
    return m


def _object_markers(obj: Any, frame: str) -> list[Any]:
    """One attached object's primitives plus its label, in ``frame``."""
    from visualization_msgs.msg import Marker

    rgba = EVIDENCE_RGBA.get(obj.evidence_kind, OTHER_EVIDENCE_RGBA)
    t_obj = _mat(obj.pose_in_link)
    out: list[Any] = []
    for prim in obj.primitives:
        t = t_obj @ _mat(prim.pose_in_object)
        dims = [float(d) for d in prim.shape_dimensions]
        if _SHAPE_DIMS.get(prim.shape_type) != len(dims):
            raise ValueError(
                f"{obj.object_id!r}: primitive shape {prim.shape_type} with "
                f"{len(dims)} dimension(s) is not a sphere/capsule/box"
            )
        if prim.shape_type == _SHAPE_BOX:
            box = [2 * d for d in dims]
            out.append(_marker("attached", frame, Marker.CUBE, _pose_of(t), box, rgba))
            lo, hi = list(prim.tight_dop_lo)[:3], list(prim.tight_dop_hi)[:3]
            if len(lo) == 3 and len(hi) == 3:
                # The refinement's first three 26-DOP axes are the box's own x/y/z: the
                # measured inner box, the shape payload-vs-robot is checked with.
                inner = t.copy()
                inner[:3, 3] = t[:3, :3] @ ((np.asarray(lo) + np.asarray(hi)) / 2.0) + t[:3, 3]
                size = [b - a for a, b in zip(lo, hi, strict=True)]
                out.append(
                    _marker(
                        "attached_measured",
                        frame,
                        Marker.CUBE,
                        _pose_of(inner),
                        size,
                        MEASURED_RGBA,
                    )
                )
            continue
        d = 2 * dims[0]
        seg = dims[1] if prim.shape_type == _SHAPE_CAPSULE else 0.0
        if seg > 0.0:  # capsule: segment along local +Z (CapsuleShape convention)
            cyl = (d, d, seg)
            out.append(_marker("attached", frame, Marker.CYLINDER, _pose_of(t), cyl, rgba))
        for z in sorted({-seg / 2, seg / 2}):
            cap = t.copy()
            cap[:3, 3] = t[:3, 3] + t[:3, 2] * z
            out.append(_marker("attached", frame, Marker.SPHERE, _pose_of(cap), (d, d, d), rgba))
    tags = [obj.object_id.rsplit(":", 1)[-1][:24], obj.evidence_kind]
    if obj.support_contact_valid:
        tags.append("support")
    if obj.evidence_ref.endswith(_FROZEN_RELEASE_SUFFIX):
        tags.append("released")
    label_pose = _pose_of(t_obj)
    out.append(
        _marker(
            "attached_label",
            frame,
            Marker.TEXT_VIEW_FACING,
            label_pose,
            (0, 0, 0.025),
            _LABEL_RGBA,
            " · ".join(tags),
        )
    )
    return out


def _region_markers(
    ns: str, region: Any, frame: str, rgba: Sequence[float], text: str
) -> list[Any]:
    """A ``PlaceRegion``: translucent box, wireframe edges and a label above it."""
    from geometry_msgs.msg import Point, Pose
    from visualization_msgs.msg import Marker

    he = (region.half_extents.x, region.half_extents.y, region.half_extents.z)
    box = _marker(ns, frame, Marker.CUBE, region.pose, [2 * h for h in he], rgba)
    edges = _marker(ns, frame, Marker.LINE_LIST, region.pose, (0.003, 0, 0), (*rgba[:3], 1.0))
    signs = (-1, 1)
    corners = [(sx * he[0], sy * he[1], sz * he[2]) for sx in signs for sy in signs for sz in signs]
    for i, a in enumerate(corners):
        for b in corners[i + 1 :]:
            if sum(p != q for p, q in zip(a, b, strict=True)) == 1:  # one axis apart: an edge
                edges.points += [Point(x=a[0], y=a[1], z=a[2]), Point(x=b[0], y=b[1], z=b[2])]
    p = region.pose.position
    top = Pose(position=Point(x=p.x, y=p.y, z=p.z + he[2] + 0.03))
    label = _marker(
        f"{ns}_label", frame, Marker.TEXT_VIEW_FACING, top, (0, 0, 0.03), _LABEL_RGBA, text
    )
    return [box, edges, label]


def static_geometry_markers(path: str, frame: str) -> Any:
    """The declared static geometry (``StaticGeometry`` YAML) as a ``MarkerArray``.

    Every box is a translucent ``CUBE`` at its pose; every mesh a ``TRIANGLE_LIST`` from
    its ``.npz`` (``vertices``, ``triangles``). Both are in ``frame`` — the manifest
    ``base_frame`` the file is declared in — and labelled with their ``source``.

    Raises:
        openral_core.ROSConfigError: the file is missing or does not validate, or a
            mesh file is missing or malformed.
    """
    from pathlib import Path

    import yaml
    from geometry_msgs.msg import Point
    from openral_core import ROSConfigError, StaticBox, StaticGeometry
    from openral_core.geometry import homogeneous_from_quat_xyz
    from visualization_msgs.msg import Marker, MarkerArray

    p = Path(path)
    try:
        geometry = StaticGeometry.model_validate(yaml.safe_load(p.read_text()))
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ROSConfigError(f"static geometry {p}: {exc}") from exc
    out: list[Any] = [Marker(action=Marker.DELETEALL)]
    for entry in geometry.static_geometry:
        if isinstance(entry, StaticBox):
            t = homogeneous_from_quat_xyz(entry.xyz, entry.quat_xyzw)
            size = [2 * h for h in entry.half_extents]
            out.append(
                _marker("static_geometry", frame, Marker.CUBE, _pose_of(t), size, STATIC_RGBA)
            )
            continue
        mesh = p.parent / entry.file
        try:
            with np.load(mesh) as data:
                verts = np.asarray(data["vertices"], dtype=np.float64)
                tris = np.asarray(data["triangles"], dtype=np.int64)
            if tris.size and (tris.min() < 0 or tris.max() >= len(verts)):
                raise ValueError(f"triangle index outside 0..{len(verts) - 1}")
        except (OSError, KeyError, ValueError) as exc:
            raise ROSConfigError(f"static mesh {mesh}: {exc}") from exc
        m = _marker(
            "static_geometry",
            frame,
            Marker.TRIANGLE_LIST,
            _pose_of(np.eye(4)),
            (1, 1, 1),
            STATIC_RGBA,
        )
        m.points = [
            Point(x=float(x), y=float(y), z=float(z)) for x, y, z in verts[tris.reshape(-1)]
        ]
        out.append(m)
    for i, m in enumerate(out[1:], start=1):
        m.id = i
        m.frame_locked = False
    return MarkerArray(markers=out)


# ---------------------------------------------------------------------------
# ROS node (all rclpy / message imports deferred to methods — PLC0415)
# ---------------------------------------------------------------------------


class Bucket2MarkersNode:
    """Read-only converter node for Bucket-2 custom message types.

    Subscribes to ``/openral/world_voxels`` and ``/openral/attachment_state``
    and re-publishes them as standard ROS visualization types (``PointCloud2``,
    ``MarkerArray`` on :data:`ATTACHMENT_MARKERS_TOPIC`). Parameter
    ``attach_link_tf_frames`` (``"link=frame"`` list) renames manifest attach
    links to their TF frames. Never commands the robot.
    """

    def __init__(self) -> None:
        """Initialise subscriptions and publishers, then log readiness."""
        import rclpy
        import rclpy.node
        import rclpy.qos

        self._node: rclpy.node.Node = rclpy.node.Node("bucket2_markers")

        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        from openral_msgs.msg import OccupancyVoxels
        from sensor_msgs.msg import PointCloud2

        self._pub_cloud = self._node.create_publisher(
            PointCloud2, "/openral/world_voxels_cloud", qos
        )

        self._sub_voxels = self._node.create_subscription(
            OccupancyVoxels,
            "/openral/world_voxels",
            self._on_world_voxels,
            qos,
        )

        # The attachment set is latched description-class data published on
        # change only, so both ends are TRANSIENT_LOCAL: a late subscriber
        # (this node, or a Foxglove client joining mid-grasp) still gets it.
        from openral_msgs.msg import AttachmentState
        from visualization_msgs.msg import MarkerArray

        latched = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Manifest attach link -> TF frame renames, the strings the HAL gets as
        # `vision_attachment_tf_frames` (the octomap bridge's parameter of the
        # same name). A malformed map is ignored whole, as the bridge does.
        entries = self._node.declare_parameter("attach_link_tf_frames", [""]).value
        try:
            self._tf_frames = parse_attach_link_tf_frames(list(entries or []))
        except ValueError:
            log.exception("bucket2: attach_link_tf_frames malformed — ignoring the whole map")
            self._tf_frames = {}
        self._pub_attachments = self._node.create_publisher(
            MarkerArray, ATTACHMENT_MARKERS_TOPIC, latched
        )
        self._sub_attachments = self._node.create_subscription(
            AttachmentState, "/openral/attachment_state", self._on_attachment_state, latched
        )
        # Declared static geometry (`static_geometry_file`, in `base_frame`): published
        # once, latched. A file the deploy launch already validated; a bad one is logged.
        static_file = str(self._node.declare_parameter("static_geometry_file", "").value or "")
        base_frame = str(self._node.declare_parameter("base_frame", "").value or "")
        self._pub_static = None
        if static_file and base_frame:
            from openral_core import ROSConfigError

            self._pub_static = self._node.create_publisher(
                MarkerArray, STATIC_GEOMETRY_MARKERS_TOPIC, latched
            )
            try:
                self._pub_static.publish(static_geometry_markers(static_file, base_frame))
            except ROSConfigError:
                log.exception("bucket2: static_geometry_file unreadable — not drawn")

        log.info("bucket2_markers node ready", attach_link_tf_frames=self._tf_frames)

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _on_world_voxels(self, msg: object) -> None:
        """Convert OccupancyVoxels → PointCloud2 and publish."""
        from sensor_msgs.msg import PointCloud2, PointField

        origin = (msg.origin.x, msg.origin.y, msg.origin.z)  # type: ignore[union-attr]
        size = (int(msg.size_x), int(msg.size_y), int(msg.size_z))  # type: ignore[union-attr]
        occupancy = np.asarray(msg.occupancy)  # type: ignore[attr-defined]
        resolution = float(msg.resolution)  # type: ignore[union-attr]
        # The grid's lattice is the OctoMap's; without its rotation every voxel
        # is drawn somewhere the obstacle is not.
        orientation = (
            float(msg.orientation.x),  # type: ignore[union-attr]
            float(msg.orientation.y),  # type: ignore[union-attr]
            float(msg.orientation.z),  # type: ignore[union-attr]
            float(msg.orientation.w),  # type: ignore[union-attr]
        )

        try:
            centers = occupied_voxel_centers(origin, resolution, size, occupancy, orientation)
        except ValueError:
            log.exception("bucket2: malformed OccupancyVoxels message — skipping")
            return

        cloud = PointCloud2()
        cloud.header = msg.header  # type: ignore[union-attr]
        cloud.height = 1
        cloud.width = len(centers)
        cloud.is_dense = True
        cloud.is_bigendian = False

        # xyz as three float32 fields (4 bytes each, 12 bytes per point)
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width

        cloud.data = np.asarray(centers, dtype="<f4").reshape(-1, 3).tobytes()

        self._pub_cloud.publish(cloud)
        log.debug("bucket2: published world_voxels_cloud", points=len(centers))

    def _on_attachment_state(self, msg: object) -> None:
        """Convert AttachmentState → MarkerArray (DELETEALL + re-add) and publish."""
        from openral_core import ROSConfigError

        try:
            markers = attachment_markers(msg, self._tf_frames)
        except (ValueError, ROSConfigError):
            log.exception("bucket2: malformed AttachmentState message — skipping")
            return
        self._pub_attachments.publish(markers)

    def spin(self) -> None:
        """Block and spin the node until shutdown."""
        import rclpy

        rclpy.spin(self._node)

    def destroy(self) -> None:
        """Tear down the node."""
        self._node.destroy_node()


def main() -> None:
    """Entry point for the ``bucket2_markers`` console script."""
    import rclpy

    rclpy.init()
    node = Bucket2MarkersNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()
