"""HAL-side client for the vision attachment-evidence path.

ROS wiring that turns ``_grasp_trigger`` events into
``openral_msgs/srv/SegmentInView`` calls, feeds the replies to
``VisionAttachmentEvidenceProducer``, and publishes the resulting
``AttachmentState`` the safety kernel already consumes.

The real-hardware sibling of the attachment leg in
``sim_sensor_bridge``, which reads MuJoCo ground truth. Enable it
explicitly (``vision_attachment_enabled``); the two must not both drive
``/openral/attachment_state``.

**Torch-free, by construction.** Everything model-shaped lives behind
the service, in ``openral_perception_ros.segmenter_node``. This module
imports numpy, rclpy and ``openral_core`` — never ``torch``, never
``transformers``, and never the runner's segmenter backend.

The deferred-ack barrier: the HAL already holds the ``action_applied``
acknowledgement of a grouped tick until attached-payload perception
settles (``attachment_action_ack_ready`` /
``_on_attachment_perception_ready``). In the simulator that wait is for
a transparent depth frame — about 100 ms. Segmentation rides inside
that same wait: a warmed SAM 2.1 call was measured at ~53 ms on the
reference GPU, so on that host the barrier does not grow at all. This
bridge exposes the same ``attachment_action_ack_ready`` shape, so the
node holds the tick for it exactly as it does for the simulator's depth
frames.

Three properties of that wait are non-negotiable:

* **It is bounded.** Every request carries a deadline
  (``VisionAttachmentConfig.deadline_s``). When it expires the
  attachment is resolved from what is in hand — nothing — rather than
  waiting longer. A robot that stalls because a perception node is
  wedged is a worse failure than a conservatively-shaped payload.
* **It never skips.** A timeout, a service failure, a missing depth
  frame and a missing transform all end in the producer's conservative
  jaw-span box, stamped ``AttachmentEvidenceKind.GRIPPER_CLOSURE`` at low
  confidence. Something is in the jaws either way; the collision
  checker must see *some* geometry.
* **It is visible.** Every fallback logs its typed reason and every
  attachment logs the gate report (CLAUDE.md §1.4). Nothing here
  degrades silently.

The kernel's attached-payload check fails closed on a snapshot it has never
heard (``attachment_stamp_ns == 0``) or one older than its deadline, so the
set is republished on a 0.2 s heartbeat — the same period as the simulator
bridge's. That heartbeat is a claim about the jaws, so it is only made while
the claim has evidence behind it: every gripper's jaw *position* channel must
have reported a finite value within ``VisionAttachmentConfig.evidence_timeout_s``,
in an unbroken run of new samples (repeated stamps do not count, a gap over
``PositionStallConfig.max_gap_s`` breaks the run) spanning at least
``PositionStallConfig.consecutive_s`` seconds of sample time and at least
``consecutive_samples`` samples, with no
grasp being resolved and every trigger agreeing with its leg's attachment. A
dead position channel therefore ages into a kernel drop, never into a stale
"nothing attached".

The grasp trigger is ``_grasp_trigger.PositionStallTrigger``: the jaw settling
short of a close command. It needs the command, so the HAL node feeds every
applied command through ``observe_command`` (the safety-approved target, the
chunk's last row — what the transport sends the trajectory controller). For an
ADR-0102 slot group that is the HAL's own composed full-dof action
(``last_applied_action``), handed over only once the HAL committed the tick: the
bridge never re-stages slots, so it cannot diverge from what the HAL applied.

Every grasp event supersedes whatever request its leg still has in flight: the
leg's generation advances, the old deadline timer is cancelled and the old future
dropped, so a late reply for an older event is discarded rather than overwriting
what the newer event decided (a DETACH while a segmentation is in flight resolves
to no attachment). A REGRASP always segments: the latched grasp-target region is a
pre-grasp measurement and is the payload only on the first ATTACH of its
declaration.
Confirmation is geometric and an AND: a stall attaches with vision-measured
geometry only when vision agrees — the grasp-target leg's latched pre-grasp
region with the jaw at it (preferred: the hand occludes the head camera at that
moment, design §2.2 "Handover"), else a ``SegmentInView`` mask that clears the
producer's gates. A stall vision cannot confirm still attaches the conservative
``GRIPPER_CLOSURE`` jaw box (fail-closed for collision). Detach is an OR in
effect: the trigger alone detaches (vision is never asked to keep a payload
the jaws let go of).

With ``VisionAttachmentConfig.grasp_target_enabled`` (default off) the bridge
also owns the pre-grasp target producer leg (``_grasp_target_leg``): it
measures the live ``GraspDeclaration``'s region and every publication — event
or heartbeat — carries that declaration on the envelope, so the kernel reads
the region and the attachment set from one snapshot. With ``grasp_target_approach_m`` set
(default off) a declaration that names no search box — no target — is measured
around whichever one hand's TCP approaches occupied cells, so the policy, not
the reasoner, picks the object (``_grasp_target_leg`` "Approach-armed target").

With ``VisionAttachmentConfig.place_target_enabled`` (default off) it also owns
the real place producer leg (``_place_target_leg``): while a payload is held it
measures the surface directly under it in the voxel map and, when one is measured,
every publication carries a ``PlaceDeclaration`` with that support patch as the
region (dispatch's optional declaration when there is one, else the leg's own),
and the payload gets a map-support proximity witness (not sensed contact) once.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core import CameraTopicKind, ControlMode, camera_topic
from openral_core.depth_extrinsic import MAX_PLANAR_ERR_M
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import homogeneous_from_quat_xyz

from openral_hal._grasp_target_leg import GraspTargetLeg, _hand_near
from openral_hal._grasp_trigger import (
    GraspEvent,
    PositionStallConfig,
    PositionStallTrigger,
    gripper_joints,
)
from openral_hal._place_target_leg import PlaceTargetLeg
from openral_hal._vision_attachment_evidence import (
    VisionAttachmentEvidenceProducer,
    VisionGateConfig,
)

if TYPE_CHECKING:  # pragma: no cover — typing only
    from openral_core import (
        Action,
        AttachedCollisionObject,
        GraspDeclaration,
        IntrinsicsPinhole,
        JointState,
        PlaceRegion,
        RobotDescription,
    )

__all__ = [
    "DEFAULT_SEGMENT_SERVICE",
    "ReleaseWindow",
    "SegmentOutcome",
    "VisionAttachmentBridge",
    "VisionAttachmentConfig",
    "box_gap_lower_bound_m",
    "build_segment_request",
    "decode_mono8_mask",
    "freeze_released_attachment",
    "mask_depth_skew_reason",
    "mask_stamps_ns",
    "primitive_poses",
    "region_attachment",
    "resolve_segment_outcome",
]

#: Service the perception-side segmenter node offers by default.
DEFAULT_SEGMENT_SERVICE = "/openral/perception/segment_in_view"


@dataclass
class _GripperLeg:
    """One gripper's trigger, producer, and in-flight segmentation state.

    A bimanual robot carries one leg per ``role: gripper`` joint; the two hands
    grasp independently, and each leg owns its own slot in the published
    attachment set.

    Attributes:
        joint_name: The gripper joint this leg watches.
        trigger: Position-stall trigger on that joint.
        producer: Evidence producer attached to that joint's parent link.
        object_id: Attachment identity, unique across legs.
        tcp_frame: tf2 frame to look the TCP up from, or empty to use
            ``tcp_in_link`` directly.
        tcp_in_link: TCP in the attach link — the gripper joint's
            ``origin_xyz``, i.e. the jaw child link's origin in the parent link.
            Needs no TF, so it holds even where the attach link is missing from
            the published tree (the vendored OpenArm URDF has no ``link7``).
        attachment: What this leg currently holds, or ``None``.
        inflight: The outstanding ``SegmentInView`` future, if any.
        deadline_timer: The one-shot deadline timer for that future, if any.
        generation: Bumped by every grasp event; a reply or deadline carrying an
            older one is stale and dropped.
        region_spent: ``(target_id, region stamp_ns)`` of the measured region this leg
            already took as its payload; an ATTACH offered that same region again
            segments instead, while a later pick's region (its own
            ``approach:<link>:<n>`` identity, measured afresh) may be taken.
        pending: Whether this leg is holding the ack barrier.
        jaw_link: The gripper joint's child link — what a ``GraspDeclaration``
            names in ``contact_links``.
        release: The payload this leg released and still publishes, frozen in
            the base frame, until its jaws are clear (``ReleaseWindow``).
        announced_uncommanded: Whether "no commanded target yet" was logged for this leg.
    """

    joint_name: str
    trigger: PositionStallTrigger
    producer: VisionAttachmentEvidenceProducer
    object_id: str
    tcp_frame: str
    tcp_in_link: tuple[float, float, float]
    attachment: Any = None
    inflight: Any = None
    deadline_timer: Any = None
    generation: int = 0
    region_spent: tuple[str, int] | None = None
    pending: bool = False
    jaw_link: str = ""
    release: ReleaseWindow | None = None
    announced_uncommanded: bool = False


#: Mask/depth aspect-ratio agreement below which a resample is a resolution change.
_ASPECT_TOLERANCE = 1e-3

#: Heartbeat period, seconds — the simulator bridge's attachment heartbeat period.
_HEARTBEAT_PERIOD_S = 0.2

#: Confidence on a region payload: the vision producer's own accepted-mask value, since the
#: region is a gated, map-verified head-view measurement of the same object.
_REGION_CONFIDENCE = 0.7


class _JawEvidence:
    """Is every gripper's jaw position channel alive right now? — the heartbeat's gate.

    Pure bookkeeping on a caller-supplied monotonic clock. A sample counts only
    when *every* leg read a finite jaw position from it; one missing value, or a gap
    longer than the timeout, restarts the run, because a heartbeat claims
    something about every hand at once and a channel that just came back has
    not yet shown it is steady. The run is measured exactly as the trigger measures
    its debounce: at least ``consecutive_s`` of sample time AND at least
    ``consecutive_samples`` samples, restarted by a sample-time gap over ``max_gap_s``
    (``PositionStallConfig.is_gap``). A sample whose stamp repeats the last one
    (``PositionStallConfig.is_repeat``) is no evidence at all — a cached joint
    state re-read neither extends the run nor refreshes liveness.

    Args:
        config: The trigger windows: ``consecutive_s`` of unbroken complete samples
            before evidence is live, and the repeated-stamp rule.
        timeout_s: How old (monotonic) the newest complete sample may be.
        spans_gaps: Sim twin only (``VisionAttachmentConfig.evidence_run_spans_gaps``):
            a sample-time gap restarts the run only past ``timeout_s``, not past
            ``max_gap_s`` — the twin's joint states pause for every re-inference.

    Example:
        >>> config = PositionStallConfig(consecutive_s=0.05, consecutive_samples=2)
        >>> evidence = _JawEvidence(config, timeout_s=0.5)
        >>> evidence.observe(complete=True, stamp_ns=0, now_s=0.0)
        >>> evidence.observe(complete=True, stamp_ns=10, now_s=0.05)  # a repeat
        >>> evidence.live(now_s=0.05)
        False
        >>> evidence.observe(complete=True, stamp_ns=50_000_000, now_s=0.1)
        >>> evidence.live(now_s=0.5), evidence.live(now_s=0.7)
        (True, False)
    """

    def __init__(
        self, config: PositionStallConfig, *, timeout_s: float, spans_gaps: bool = False
    ) -> None:
        """Start with no evidence."""
        self._config = config
        self._timeout_s = timeout_s
        self._spans_gaps = spans_gaps
        self._since_ns: int | None = None
        self._samples = 0
        self._last_ns: int | None = None
        self._last_s: float | None = None

    def observe(self, *, complete: bool, stamp_ns: int, now_s: float) -> None:
        """Fold one joint-state sample in."""
        if self._config.is_repeat(stamp_ns, self._last_ns):
            return
        gap = (
            self._last_ns is not None and stamp_ns - self._last_ns > self._timeout_s * 1e9
            if self._spans_gaps
            else self._config.is_gap(stamp_ns, self._last_ns)
        )
        self._last_ns = stamp_ns
        stale = self._last_s is not None and now_s - self._last_s > self._timeout_s
        if not complete or stale or gap:
            self._since_ns = None
        if not complete:
            return
        if self._since_ns is None:
            self._since_ns, self._samples = stamp_ns, 0
        self._samples += 1
        self._last_s = now_s

    def live(self, *, now_s: float) -> bool:
        """Whether enough fresh, complete samples back a claim about the jaws."""
        return (
            self._since_ns is not None
            and self._last_ns is not None
            and self._last_ns - self._since_ns >= self._config.consecutive_s * 1e9
            and self._samples >= self._config.consecutive_samples
            and self._last_s is not None
            and now_s - self._last_s <= self._timeout_s
        )


#: An oriented box in the base frame: ``(centre (3,), rotation (3, 3), half extents (3,))``.
_Box = tuple["NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"]
#: A primitive in its owning frame: ``(frame <- primitive (4, 4), bounding half extents (3,))``.
_FramedBox = tuple["NDArray[np.float64]", "NDArray[np.float64]"]


def _bounding_half_extents(shape: Any) -> NDArray[np.float64]:
    """Half extents of a box containing a primitive, in the primitive's frame.

    A box is itself, a sphere its cube, a capsule (segment along local +Z) its
    ``(r, r, L/2 + r)`` box. Containment makes every distance measured on the
    result a lower bound on the distance to the primitive — the conservative side
    for a test that may only *end* a window once things are proven apart.
    """
    from openral_core import BoxShape, CapsuleShape, SphereShape

    if isinstance(shape, BoxShape):
        return np.asarray(shape.half_extents_m, dtype=np.float64)
    if isinstance(shape, SphereShape):
        return np.full(3, float(shape.radius_m), dtype=np.float64)
    if isinstance(shape, CapsuleShape):
        r = float(shape.radius_m)
        return np.array([r, r, float(shape.length_m) / 2.0 + r], dtype=np.float64)
    raise ROSConfigError(f"release window: cannot bound collision primitive {shape!r}.")


def _xyz_rpy_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> NDArray[np.float64]:
    """``(4, 4)`` from a translation and fixed-axis roll-pitch-yaw (``R = Rz Ry Rx``, URDF)."""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    t[:3, 3] = xyz
    return t


def _joint_motion(joint: Any, q: float) -> NDArray[np.float64]:
    """Child-in-parent motion of one joint at position ``q`` (applied after its origin)."""
    from openral_core.schemas import JointType

    axis = np.asarray(joint.axis_xyz, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    t = np.eye(4, dtype=np.float64)
    if joint.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
        k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        t[:3, :3] = np.eye(3) + np.sin(q) * k + (1.0 - np.cos(q)) * (k @ k)
    elif joint.joint_type is JointType.PRISMATIC:
        t[:3, 3] = q * axis
    else:
        raise ROSConfigError(
            f"release window: jaw joint {joint.name!r} is {joint.joint_type.value}; only "
            "revolute, continuous and prismatic jaws are posed."
        )
    return t


#: Below this a cross-product axis is degenerate (parallel edges).
_PARALLEL_EPS = 1e-9


def box_gap_lower_bound_m(a: _Box, b: _Box) -> float:
    """Separating-axis lower bound on the distance between two oriented boxes.

    The gap between the boxes' projections onto any unit axis never exceeds their
    distance (projection is 1-Lipschitz), so the largest gap over the 15 SAT
    axes is a sound lower bound — exact along a face normal, conservative across
    an edge-edge pair. ``<= 0`` when the boxes overlap on every axis tried.

    Args:
        a: ``(centre, rotation, half_extents)`` of the first box.
        b: The second box, same frame.

    Returns:
        A lower bound on the surface distance, metres.

    Example:
        >>> import numpy as np
        >>> unit = (np.zeros(3), np.eye(3), np.full(3, 0.5))
        >>> moved = (np.array([1.05, 0.0, 0.0]), np.eye(3), np.full(3, 0.5))
        >>> round(box_gap_lower_bound_m(unit, moved), 6)
        0.05
    """
    (ca, ra, ha), (cb, rb, hb) = a, b
    axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
    axes += [
        np.asarray(np.cross(ra[:, i], rb[:, j]), dtype=np.float64)
        for i in range(3)
        for j in range(3)
    ]
    offset = cb - ca
    gap = -np.inf
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm < _PARALLEL_EPS:  # parallel edges: their face normals already cover the pair
            continue
        n = axis / norm
        reach = float(np.sum(ha * np.abs(ra.T @ n)) + np.sum(hb * np.abs(rb.T @ n)))
        gap = max(gap, abs(float(offset @ n)) - reach)
    return float(gap)


def freeze_released_attachment(
    held: AttachedCollisionObject,
    *,
    base_link: str,
    t_base_from_link: NDArray[np.float64],
) -> AttachedCollisionObject:
    """Re-express a released payload as a record fixed in the base frame.

    The kernel places an attached object at ``link_world[attach_link] ·
    pose_in_link`` for every configuration a candidate predicts, so a record
    left on the hand would ride the retreating hand through the whole chunk while
    the real object stays put. Attached to the collision model's root link (the
    manifest's ``base_frame``: identity FK, no primitives of its own) it stays
    where it was released in every predicted configuration, and the attach-link
    skip exempts nothing.

    ``touch_links`` become the held record's attach link plus its touch links —
    exactly the links the held record already exempted (the hand and the jaws
    are in contact with the payload at release); every other link, and the
    world, is still checked against it.

    Args:
        held: The attachment the gripper held until DETACH.
        base_link: The collision model's root link (``RobotDescription.base_frame``).
        t_base_from_link: ``(4, 4)`` pose of ``held.attach_link`` in ``base_link``
            at the DETACH stamp.

    Returns:
        The frozen record: same object, geometry, evidence and support/force
        attestations; base-frame pose; ``evidence_ref`` suffixed
        ``|frozen_release``.
    """
    from openral_core import AttachedCollisionObject as _Attached
    from openral_core.geometry import homogeneous_from_quat_xyz, rotation_to_quat_wxyz

    pose = t_base_from_link @ homogeneous_from_quat_xyz(
        held.pose_in_link.xyz, held.pose_in_link.quat_xyzw
    )
    w, x, y, z = rotation_to_quat_wxyz(pose[:3, :3])
    touch = [held.attach_link, *(link for link in held.touch_links if link != held.attach_link)]
    return _Attached.model_validate(
        {
            **held.model_dump(),
            "attach_link": base_link,
            "touch_links": touch,
            "pose_in_link": {
                "xyz": tuple(float(v) for v in pose[:3, 3]),
                "quat_xyzw": (x, y, z, w),
                "frame_id": base_link,
            },
            "evidence_ref": f"{held.evidence_ref or held.object_id}|frozen_release",
        }
    )


def region_attachment(
    declaration: GraspDeclaration,
    *,
    attach_link: str,
    touch_links: Sequence[str],
    t_link_from_region: NDArray[np.float64],
    stamp_ns: int,
) -> AttachedCollisionObject:
    """The grasp-target leg's latched pre-grasp region as the held payload (design §2.2).

    At ATTACH the hand occludes the head camera, so the region measured *before* the
    grasp (head-view mask + voxel map, gated and tracked by ``_grasp_target_leg``) is a
    better payload than a re-segmentation. One box primitive with the region's half
    extents, posed at the region in the attach link; ``object_id`` is the declaration's
    ``object_id`` (what the kernel's handover matches the attachment against) or, when
    that is empty, its ``target_id``.

    Args:
        declaration: The live declaration, carrying the accepted ``region``.
        attach_link: The gripper's attach link (the producer's).
        touch_links: Links allowed to touch the payload (the producer's).
        t_link_from_region: ``(4, 4)`` pose of ``region.frame_id`` in ``attach_link``.
        stamp_ns: The ATTACH instant.

    Returns:
        The attachment, evidence ``GRASP_TARGET_REGION``.

    Raises:
        ROSConfigError: If the declaration carries no region.
    """
    from openral_core import (
        AttachedCollisionObject as _Attached,
    )
    from openral_core import (
        AttachedCollisionPrimitive,
        AttachmentEvidenceKind,
        BoxShape,
        Pose6D,
    )
    from openral_core.geometry import homogeneous_from_quat_xyz, rotation_to_quat_wxyz

    region = declaration.region
    if region is None:
        raise ROSConfigError(f"grasp target {declaration.target_id!r} carries no region.")
    object_id = declaration.object_id or declaration.target_id
    pose = t_link_from_region @ homogeneous_from_quat_xyz(region.pose.xyz, region.pose.quat_xyzw)
    w, x, y, z = rotation_to_quat_wxyz(pose[:3, :3])
    return _Attached(
        object_id=object_id,
        attach_link=attach_link,
        touch_links=list(touch_links),
        primitives=[
            AttachedCollisionPrimitive(
                shape=BoxShape(half_extents_m=tuple(float(h) for h in region.half_extents)),
                pose_in_object=Pose6D(
                    xyz=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=object_id
                ),
            )
        ],
        pose_in_link=Pose6D(
            xyz=(float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])),
            quat_xyzw=(x, y, z, w),
            frame_id=attach_link,
        ),
        confidence=_REGION_CONFIDENCE,
        evidence_kind=AttachmentEvidenceKind.GRASP_TARGET_REGION,
        evidence_ref=f"grasp_target_region:{declaration.target_id}:{region.evidence_ref}@{stamp_ns}",
        stamp_ns=stamp_ns,
    )


def primitive_poses(
    obj: AttachedCollisionObject, t_frame_from_link: NDArray[np.float64]
) -> list[NDArray[np.float64]]:
    """``frame <- primitive`` (4, 4) for each primitive of ``obj``, given ``frame <- link``."""
    t_object = t_frame_from_link @ homogeneous_from_quat_xyz(
        obj.pose_in_link.xyz, obj.pose_in_link.quat_xyzw
    )
    return [
        t_object @ homogeneous_from_quat_xyz(p.pose_in_object.xyz, p.pose_in_object.quat_xyzw)
        for p in obj.primitives
    ]


@dataclass(frozen=True)
class ReleaseWindow:
    """One gripper's released payload, frozen in the base frame until the jaws are clear.

    Opened on that gripper's DETACH (design note §2.3 "Release"): the position-stall
    DETACH fires as the jaws *open* past their hold, before the hand retreats, so
    dropping the payload then lets the occupancy map re-mark it inside the fingers' world margin and
    the first retreat chunk stops on it. Instead the payload stays a published,
    fully checked attached record (``freeze_released_attachment``) — which also
    keeps the octomap bridge clearing its cells — until the hand and jaws it
    exempts are proven ``release_clear_m`` away, ``release_timeout_s`` passes, or
    the gripper grasps again.

    Attributes:
        record: The frozen record, attached to the base link.
        opened_s: Monotonic seconds at DETACH.
        hand_link: The held record's attach link (the hand), posed through tf2.
        hand_boxes: The hand's collision primitives, bounded, in the hand frame.
        jaws: ``(joint, origin, boxes)`` per jaw link the record exempts: the
            jaw joint (hinged on the hand), its origin in the hand frame, and the
            jaw's bounded primitives in the jaw frame.
    """

    record: AttachedCollisionObject
    opened_s: float
    hand_link: str
    hand_boxes: tuple[_FramedBox, ...]
    jaws: tuple[tuple[Any, NDArray[np.float64], tuple[_FramedBox, ...]], ...]

    @classmethod
    def open(
        cls,
        description: RobotDescription,
        held: AttachedCollisionObject,
        *,
        base_link: str,
        t_base_from_link: NDArray[np.float64],
        now_s: float,
    ) -> ReleaseWindow:
        """Freeze ``held`` and collect the geometry the separation test needs.

        Args:
            description: The robot manifest (collision geometry, jaw joints).
            held: The attachment held until DETACH.
            base_link: The collision model's root link.
            t_base_from_link: The hand's pose in ``base_link`` at DETACH.
            now_s: Monotonic seconds at DETACH.

        Returns:
            The open window.
        """

        def boxes(link: str) -> tuple[_FramedBox, ...]:
            return tuple(
                (
                    _xyz_rpy_matrix(geom.origin_xyz_rpy[:3], geom.origin_xyz_rpy[3:]),
                    _bounding_half_extents(geom.shape),
                )
                for geom in description.collision_geometry
                if geom.link_name == link
            )

        jaws = tuple(
            (joint, _xyz_rpy_matrix(joint.origin_xyz, joint.origin_rpy), boxes(joint.child_link))
            for joint in description.joints
            if joint.parent_link == held.attach_link and joint.child_link in held.touch_links
        )
        return cls(
            record=freeze_released_attachment(
                held, base_link=base_link, t_base_from_link=t_base_from_link
            ),
            opened_s=now_s,
            hand_link=held.attach_link,
            hand_boxes=boxes(held.attach_link),
            jaws=jaws,
        )

    def clearance_m(
        self,
        t_base_from_hand: NDArray[np.float64],
        positions: Mapping[str, float],
    ) -> float | None:
        """Lower bound on the gap between the frozen payload and every link it exempts.

        Args:
            t_base_from_hand: The hand's pose in the base frame now.
            positions: Latest joint positions by joint name (the jaws' angles).

        Returns:
            The smallest box-gap lower bound over (exempted link primitive,
            payload primitive) pairs, or ``None`` when a jaw position is unknown
            — unknown is never "clear".
        """
        placed: list[tuple[NDArray[np.float64], tuple[_FramedBox, ...]]] = [
            (t_base_from_hand, self.hand_boxes)
        ]
        for joint, origin, jaw_boxes in self.jaws:
            if joint.name not in positions:
                return None
            motion = _joint_motion(joint, positions[joint.name])
            placed.append((t_base_from_hand @ origin @ motion, jaw_boxes))
        # The frozen record is attached to the base link: its link frame is the base.
        payload: list[_Box] = [
            (t[:3, 3], t[:3, :3], _bounding_half_extents(primitive.shape))
            for t, primitive in zip(
                primitive_poses(self.record, np.eye(4)), self.record.primitives, strict=True
            )
        ]
        gap = np.inf
        for t_link, link_boxes in placed:
            for t_prim, half in link_boxes:
                t = t_link @ t_prim
                link_box = (t[:3, 3], t[:3, :3], half)
                for box in payload:
                    gap = min(gap, box_gap_lower_bound_m(link_box, box))
        return float(gap)

    def close_reason(
        self, *, now_s: float, clearance_m: float | None, clear_m: float, timeout_s: float
    ) -> str:
        """Why the window ends now — ``"timeout"`` or ``"separation"`` — or ``""`` to keep it.

        Args:
            now_s: Monotonic seconds now.
            clearance_m: ``clearance_m(...)`` now; ``None`` when unknown.
            clear_m: ``VisionAttachmentConfig.release_clear_m``.
            timeout_s: ``VisionAttachmentConfig.release_timeout_s``.

        Returns:
            The close reason, or ``""``.
        """
        if now_s - self.opened_s >= timeout_s:
            return "timeout"
        if clearance_m is not None and clearance_m > clear_m:
            return "separation"
        return ""


def _tree_links(description: RobotDescription) -> tuple[set[str], set[str]]:
    """``(parents, children)`` of every joint and fixed attachment in the manifest."""
    parents = {j.parent_link for j in description.joints}
    parents |= {f.parent_link for f in description.fixed_attachments}
    children = {j.child_link for j in description.joints}
    children |= {f.child_link for f in description.fixed_attachments}
    return parents, children


def _release_base_link(description: RobotDescription) -> str:
    """The manifest's ``base_frame``, proven to be the root of the kinematic tree.

    The kernel's collision model is rooted there at the identity frame, so a
    record attached to it stays put in every predicted configuration.
    """
    base = description.base_frame
    parents, children = _tree_links(description)
    if base in children or base not in parents:
        raise ROSConfigError(
            f"vision attachment: base_frame {base!r} of {description.name!r} is not the root "
            "of its kinematic tree, so a released payload cannot be frozen on it."
        )
    return base


#: What one grasp needs from TF and the manifest before it can prompt:
#: ``(t_link_from_cam, tcp_in_link, negative_points_in_link)``.
_PromptContext = tuple[
    "NDArray[np.float64]",
    tuple[float, float, float],
    list[tuple[float, float, float]],
]


@dataclass(frozen=True)
class VisionAttachmentConfig:
    """Wiring for ``VisionAttachmentBridge``.

    Attributes:
        camera: Logical camera id — a ``SensorSpec`` name in the robot manifest.
            Only the id and its existence come from the manifest: the optical
            frame is the depth image's ``header.frame_id`` and the intrinsics are
            the driver's live ``CameraInfo``, because this geometry feeds
            collision and a manifest's nominal values may not describe the
            stream (the OpenArm ZED: body-frame ``frame_id``, fx 960 vs a measured
            1498.18). Empty selects the manifest's first camera with intrinsics.
        depth_topic: ``sensor_msgs/Image`` topic carrying that camera's metric
            depth (``32FC1`` metres or ``16UC1`` millimetres), in a REP-103
            optical frame named by its header. The segmenter's RGB stream must
            be **registered** to it (same optical frame and view — true for the
            ZED ``rgb/color/rect`` / ``depth/depth_registered`` pair): the TCP is
            sent already in this frame, and masks are resampled onto it.
        camera_info_topic: ``sensor_msgs/CameraInfo`` for that depth stream.
            Empty / ``None`` → ``camera_topic(camera, CameraTopicKind.DEPTH_CAMERA_INFO)``.
            No message yet, an uncalibrated one, or one whose size differs from
            the depth raster sends the grasp to the conservative box — never to
            the manifest's nominal K.
        service_name: ``SegmentInView`` service to call.
        deadline_s: Upper bound on one segmentation, from request to reply.
            *Calibration point*, ``0.25`` s — a warmed call was measured at
            ~53 ms on the reference GPU plus a service round trip, so this is
            roughly a 4x margin over the measured path while staying the same
            order as the ~100 ms barrier it rides inside. A **CPU-only** host is
            far slower than that and must raise this deliberately; leaving it at
            the default there means every grasp falls back, which is safe,
            visible in the log, and useless.
        tcp_frame: tf2 frame of the tool center point. Empty resolves, on a
            single-gripper robot, to the gripper joint's ``child_link`` — the
            moving jaw — looked up through tf2, and on a multi-gripper robot to
            each gripper joint's ``origin_xyz`` (the same jaw link origin,
            read from the manifest without TF). A robot with a calibrated
            mid-jaw frame should name it here rather than accept the
            approximation. Single-gripper only: set on a robot with several
            gripper joints it is a ``ROSConfigError``.
        jaw_tip_frames: Optional tf2 frames used as **negative** prompt points,
            pushing the mask off the gripper fingers. Empty = none; the single
            positive point at the TCP is what makes the prompt meaningful.
            Single-gripper only, like ``tcp_frame``.
        object_id: Identity stamped on the attachment, suffixed
            ``:<gripper joint>`` so each hand's slot is unique. Vision cannot
            name what it segmented, so this is a stable slot ("the thing in the
            jaws"), not a recognition result.
        tf_frames: Manifest link name -> tf2 frame name, for robots whose
            published TF tree spells a link differently from the manifest. Used
            only for tf2 lookups (attach link <- camera / TCP / jaw tips); the
            published attachment keeps
            the manifest link, because the kernel's collision model is keyed on
            manifest links. Every entry must be a **proven identity** — the
            same rigid body with the same origin — never a nearby frame. The
            OpenArm case: the manifest's ``openarm_<side>_link7`` is the body
            joint7 drives, which the MJCF and the vendor URDF the real cell
            publishes both call ``openarm_<side>_ee_base_link`` (origin at
            joint7). Empty = look every link up by its manifest name.
        evidence_timeout_s: How old the newest joint-state sample carrying a
            finite jaw position for every gripper may be before the attachment
            heartbeat stops. *Calibration point*, ``0.5`` s — several HAL read
            ticks at any rate the cell runs, and well inside the kernel's
            attached-collision deadline, so a dead position channel surfaces as a
            kernel drop rather than as a stale "nothing attached".
        evidence_run_spans_gaps: Sim twin only — set by the HAL node exactly when
            ``twin_jaw_evidence_timeout_s`` applies (``hal_mode`` ``"sim"`` with an
            idle-stepping HAL). A joint-state sample gap shorter than
            ``evidence_timeout_s`` then does not restart the evidence run, so a
            re-inference pause does not withhold the heartbeat for a fresh debounce
            as the arm resumes. Default ``False``: real hardware streams continuously
            and any gap over ``max_gap_s`` restarts the run.
        grasp_target_enabled: Run the pre-grasp target producer leg
            (``_grasp_target_leg``): measure the live ``GraspDeclaration``'s
            region from its ``search_box``, the voxel map and ``SegmentInView``,
            and put it on every attachment publication. Default off.
        grasp_target_rate_hz: Re-measurement rate, 2-5 Hz (design §2.2).
        grasp_target_freeze_s: How long past its ``stamp_ns`` the last accepted
            region survives while the view is lost (the gripper occluding the
            target). ``None`` (default) = ``2 * grid_max_age_s``: at most twice
            the deploy's kernel voxel deadline. Refused above ``2 *
            grid_max_age_s`` — the kernel's ``grasp_region_max_age_s`` — past which
            the map the region was vouched against is superseded. *Calibration point.*
        grasp_target_min_cells: Fewest occupied cells the seed cluster above the
            support plane may have. *Calibration point.*
        grasp_target_min_cover: Fraction of the region's footprint cell count
            that must be occupied in the map. *Calibration point.*
        grasp_target_support_search_below_m: How far below the search box the
            target leg looks for the surface the target stands on, at most 0.5 m.
            *Calibration point.*
        grasp_target_support_probe_margin_m: Outer reach, from the target's
            footprint, of the ring that must be occupied for a layer to count as
            its support. *Calibration point.*
        grasp_target_occluder_margin_m: Distance (beyond one voxel) from the
            held region within which a declared contact link's hand point (its
            leg's TCP, ``jaw_point``) makes a shrunken re-fit an occlusion by the
            robot's own hand, at most 0.10 m.
            *Calibration point.*
        grasp_target_approach_m: Approach-armed target (``_grasp_target_leg``):
            while a live declaration has no ``search_box``, a hand whose TCP comes
            within this distance of occupied cells arms a one-hand declaration
            measured around its jaws, so no one has to name the target (the policy
            picks it). ``None`` = off (default). At most
            ``GraspDeclaration.MAX_HALF_EXTENT_M``. *Calibration point.*
        release_clear_m: How far every link a released payload's frozen record
            exempts (the hand and its jaws) must be from it before the record is
            dropped (``ReleaseWindow``). The deploy sets it to the kernel's
            world-voxel margin plus one voxel resolution: once the record goes,
            the octomap re-marks the payload's cells, and a cell is a cube the
            kernel measures against at that margin. The ``0.04`` m default only
            serves code that builds the bridge directly (tests);
            ``ManifestHALLifecycleNode`` refuses to start the leg unless the
            deploy passes the value. A lower
            bound is measured (bounding boxes, separating axes), so the window can
            only stay open longer than the true gap needs. *Calibration point.*
        release_timeout_s: Hard bound on a release window: the frozen record is
            dropped this long after DETACH even if the jaws were never proven
            clear — the kernel then sees the re-marked payload and stops a
            retreat still inside its margin (fail-closed). The bridge sees no goal
            end, so this is the window's bound when the hand does not retreat.
            *Calibration point*, ``3.0`` s.
        grid_max_age_s: Oldest ``/openral/world_voxels`` grid the grasp-target
            and place-target legs may use, seconds. The deploy sets it to the
            safety kernel's ``world_voxel_deadline_s``: a grid the kernel itself
            would refuse as stale cannot vouch for a region. The ``1.0`` s default
            only serves code that builds the bridge directly (tests);
            ``ManifestHALLifecycleNode`` refuses to start the leg unless the
            deploy passes the value.
        mask_depth_max_skew_s: Largest accepted ``|mask stamp - depth stamp|``:
            a ``SegmentInView`` mask (stamped with the RGB frame the segmenter
            captured) is only back-projected through a depth frame from the
            same instant. The ZED driver publishes RGB and depth of one grab
            with one stamp, so a same-grab pair has zero skew; ``0.1`` s admits
            the two latest caches being one frame apart at >= 10 Hz and refuses
            pairs several frames apart, across which the hand and the payload
            have moved. A refusal is a ``GRIPPER_CLOSURE`` fallback in the
            attachment path and a lost view in the grasp-target leg.
            *Calibration point.*
        place_target_enabled: Run the real place producer leg
            (``_place_target_leg``): measure the support patch directly under the
            carried payload from the voxel map, publish it as a place region (no
            dispatch declaration needed), and attest the map-support proximity
            witness. Default off; it rests on drafted, unapproved ADR-0097 /
            ADR-0092 D6 amendments.
        place_target_rate_hz: Measurement / re-verification rate.
        place_target_freeze_s: How long past its last verification a latched
            patch survives a lost view (the payload and hand occluding the board).
            ``None`` (default) = ``2 * grid_max_age_s``, which is also its ceiling:
            the kernel drops a place region older than that (``place_region_max_age_s``).
            *Calibration point.*
        place_target_search_depth_m: How far below the carried payload's bottom the
            leg looks for the surface it would land on. *Calibration point.*
        place_target_extrinsic_error_m: The depth extrinsic's accuracy bound: it
            grows the patch on every side and the free height, and is the witness
            tolerance with one voxel. Default
            ``openral_core.depth_extrinsic.MAX_PLANAR_ERR_M``. *Calibration point.*
    """

    camera: str = ""
    depth_topic: str = ""
    camera_info_topic: str | None = None
    service_name: str = DEFAULT_SEGMENT_SERVICE
    deadline_s: float = 0.25
    tcp_frame: str = ""
    jaw_tip_frames: tuple[str, ...] = ()
    object_id: str = "grasped_payload"
    tf_frames: Mapping[str, str] = field(default_factory=dict)
    evidence_timeout_s: float = 0.5
    evidence_run_spans_gaps: bool = False
    grasp_target_enabled: bool = False
    grasp_target_rate_hz: float = 3.0
    grasp_target_freeze_s: float | None = None
    grasp_target_min_cells: int = 8
    grasp_target_min_cover: float = 0.5
    grasp_target_support_search_below_m: float = 0.15
    grasp_target_support_probe_margin_m: float = 0.05
    grasp_target_occluder_margin_m: float = 0.05
    grasp_target_approach_m: float | None = None
    release_clear_m: float = 0.04
    release_timeout_s: float = 3.0
    grid_max_age_s: float = 1.0
    mask_depth_max_skew_s: float = 0.1
    place_target_enabled: bool = False
    place_target_rate_hz: float = 2.0
    place_target_freeze_s: float | None = None
    place_target_search_depth_m: float = 0.20
    place_target_extrinsic_error_m: float = MAX_PLANAR_ERR_M


@dataclass(frozen=True)
class SegmentOutcome:
    """What one ``SegmentInView`` round trip yielded, and why.

    Attributes:
        use_masks: Whether the reply carried candidates worth gating. ``False``
            sends the producer down its conservative fallback path.
        reason: Typed, human-readable explanation, empty only on the clean path.
            Logged and carried into the attachment trace.
    """

    use_masks: bool
    reason: str


def resolve_segment_outcome(
    *,
    timed_out: bool,
    ok: bool,
    failure_reason: str,
    mask_count: int,
) -> SegmentOutcome:
    """Decide how one service round trip resolves — pure, so it is testable.

    Every branch that is not "the segmenter returned candidates" ends in the
    conservative fallback, and every one of them names itself. There is no
    branch that skips the attachment.

    Args:
        timed_out: The deadline expired before a reply arrived.
        ok: The reply's ``ok`` flag.
        failure_reason: The reply's typed ``failure_reason``.
        mask_count: How many candidate masks the reply carried.

    Returns:
        The ``SegmentOutcome``.

    Example:
        >>> late = resolve_segment_outcome(
        ...     timed_out=True, ok=False, failure_reason="", mask_count=0
        ... )
        >>> late.use_masks, late.reason.split(":")[0]
        (False, 'ROSDeadlineMissed')
        >>> resolve_segment_outcome(timed_out=False, ok=True, failure_reason="", mask_count=3)
        SegmentOutcome(use_masks=True, reason='')
    """
    if timed_out:
        return SegmentOutcome(
            use_masks=False,
            reason="ROSDeadlineMissed: SegmentInView did not answer within the attach deadline",
        )
    if not ok:
        return SegmentOutcome(
            use_masks=False,
            reason=failure_reason or "ROSPerceptionStale: SegmentInView returned ok=False",
        )
    if mask_count == 0:
        return SegmentOutcome(
            use_masks=False,
            reason="ROSPerceptionStale: SegmentInView returned ok=True with no candidate masks",
        )
    return SegmentOutcome(use_masks=True, reason="")


def decode_mono8_mask(data: bytes, *, height: int, width: int) -> NDArray[np.bool_]:
    """Decode ``mono8`` mask bytes into an ``(H, W)`` boolean array.

    The reader half of ``segmenter_node.mono8_bytes_from_mask``. Any non-zero
    pixel is set — the producer writes 255, but a mask that survived a lossy hop
    must not silently become empty because it arrived as 254.

    Args:
        data: ``height * width`` tightly-packed bytes.
        height: Rows.
        width: Columns.

    Returns:
        The boolean mask.

    Raises:
        ROSConfigError: If the payload length does not match ``height * width``.

    Example:
        >>> decode_mono8_mask(bytes([255, 0]), height=1, width=2).tolist()
        [[True, False]]
    """
    flat = np.frombuffer(data, dtype=np.uint8)
    if flat.size != height * width:
        raise ROSConfigError(
            f"decode_mono8_mask: {flat.size} bytes for a {height}x{width} mono8 mask."
        )
    return np.asarray(flat.reshape(height, width) != 0, dtype=bool)


def mask_depth_skew_reason(
    mask_stamps_ns: Sequence[int], depth_stamp_ns: int, *, max_skew_s: float
) -> str:
    """Refuse masks whose capture stamp is too far from the depth frame they meet.

    The segmenter masks the RGB frame it captured and stamps each mask with that
    frame's stamp; the HAL back-projects the mask through the depth frame *it*
    holds. Nothing else pairs the two, so a mask from one instant laid on depth
    from another describes pixels of a scene that has since moved (the hand,
    the payload) — evidence of nothing.

    Args:
        mask_stamps_ns: Each mask's ``header.stamp``, ns.
        depth_stamp_ns: The depth frame's ``header.stamp``, ns.
        max_skew_s: Largest accepted ``|mask - depth|``, seconds.

    Returns:
        A typed reason (``ROSPerceptionStale: ...``) when any mask is too far
        off, else ``""``.

    Example:
        >>> mask_depth_skew_reason([1_000_000_000], 1_050_000_000, max_skew_s=0.1)
        ''
        >>> mask_depth_skew_reason([1_000_000_000], 1_200_000_000, max_skew_s=0.1)[:19]
        'ROSPerceptionStale:'
    """
    worst = max((abs(m - depth_stamp_ns) for m in mask_stamps_ns), default=0)
    if worst <= max_skew_s * 1e9:
        return ""
    return (
        f"ROSPerceptionStale: SegmentInView mask captured {worst / 1e9:.3f} s from the "
        f"depth frame it would be back-projected through (> {max_skew_s:.3f} s)"
    )


def mask_stamps_ns(masks: Sequence[Any]) -> list[int]:
    """``header.stamp`` of each ``sensor_msgs/Image`` mask, in ns."""
    return [int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec) for m in masks]


def build_segment_request(
    *,
    stamp_ns: int,
    camera: str,
    t_link_from_cam: NDArray[np.float64],
    tcp_in_link: tuple[float, float, float],
    negatives_in_link: Sequence[tuple[float, float, float]] = (),
) -> Any:
    """Build a ``SegmentInView`` request whose prompts are in the camera's optical frame.

    The bridge already holds ``link <- camera`` from tf2, so it sends the TCP and
    jaw-tip prompts in the depth stream's own optical frame, with an empty
    ``frame_id`` — which the segmenter reads as "already in my optical frame" —
    instead of asking it to re-resolve a link its TF tree may not carry.
    Precondition: the segmenter's RGB is registered to the depth stream.

    Args:
        stamp_ns: The grasp instant.
        camera: Logical camera id.
        t_link_from_cam: ``(4, 4)`` transform mapping optical-frame points into
            the attach link.
        tcp_in_link: TCP in the attach link.
        negatives_in_link: Jaw-tip negative prompts in the attach link.

    Returns:
        The ``openral_msgs.srv.SegmentInView.Request``.
    """
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import Point
    from openral_msgs.srv import SegmentInView

    t_cam_from_link = np.linalg.inv(t_link_from_cam)

    def _to_cam(point: tuple[float, float, float]) -> Any:
        x, y, z = (t_cam_from_link @ np.array([*point, 1.0], dtype=np.float64))[:3]
        return Point(x=float(x), y=float(y), z=float(z))

    request = SegmentInView.Request()
    request.stamp = Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
    request.camera = camera
    request.frame_id = ""
    request.tcp_point = _to_cam(tcp_in_link)
    request.negative_points = [_to_cam(point) for point in negatives_in_link]
    return request


class VisionAttachmentBridge:
    """Drive SegmentInView from grasp events and hold the ack barrier for it.

    One leg per ``role: gripper`` joint: each has its own position-stall trigger,
    evidence producer, and in-flight request, and the barrier stays shut while
    any leg is pending. Every publish carries the union of every leg's current
    attachment, so one hand's grasp never erases the other's payload.

    Args:
        node: The HAL lifecycle node. Owns the executor these callbacks run on.
        description: The robot manifest — gripper joints, camera intrinsics,
            frames.
        on_perception_ready: Called when a held barrier is released, so the node
            can publish the deferred ``action_applied`` tick. Same callback the
            simulator bridge is given.
        config: Wiring and the segmentation deadline.
        gate_config: Geometric gate thresholds handed to the producer.
        trigger_config: Debounce for every leg's position-stall trigger and the heartbeat
            evidence; the thresholds are each gripper joint's own ``closure_calibration``
            in the manifest. ``None`` derives it from the manifest's control rate
            (``PositionStallConfig.for_rate``; the 30 Hz-tuned defaults when it declares
            none) — the HAL node passes one derived from the rate it actually feeds.

    Raises:
        ROSConfigError: If the manifest cannot support the producer or the
            trigger (no gripper joints, a gripper joint with no
            ``closure_calibration``, no camera intrinsics),
            or ``tcp_frame`` / ``jaw_tip_frames`` is set on a robot with more
            than one gripper joint.

    Example:
        >>> from openral_core import RobotDescription
        >>> d = RobotDescription.from_yaml("robots/openarm/robot.yaml")
        >>> bridge = VisionAttachmentBridge(
        ...     None, d, config=VisionAttachmentConfig(camera="head_zed")
        ... )
        >>> bridge.gripper_joint_names
        ('left_gripper', 'right_gripper')
    """

    def __init__(
        self,
        node: Any,
        description: RobotDescription,
        *,
        on_perception_ready: Any = None,
        config: VisionAttachmentConfig | None = None,
        gate_config: VisionGateConfig | None = None,
        trigger_config: PositionStallConfig | None = None,
    ) -> None:
        """Resolve manifest-derived wiring; create no ROS entities yet."""
        self._node = node
        self._description = description
        self._config = config or VisionAttachmentConfig()
        self._on_perception_ready = on_perception_ready
        rate_hz = description.control_rate_hz
        if trigger_config is None:
            trigger_config = (
                PositionStallConfig() if rate_hz is None else PositionStallConfig.for_rate(rate_hz)
            )
        self._legs = self._build_legs(gate_config, trigger_config)
        # The manifest K is never projected through: it only fills the
        # producer's signature on the no-mask paths, which never back-project.
        self._camera, self._no_mask_intrinsics = self._resolve_camera()
        if not (self._config.mask_depth_max_skew_s > 0.0 and self._config.grid_max_age_s > 0.0):
            raise ROSConfigError(
                "vision attachment: mask_depth_max_skew_s and grid_max_age_s must be positive, "
                f"got {self._config.mask_depth_max_skew_s} and {self._config.grid_max_age_s}."
            )
        if self._config.release_clear_m <= 0.0 or self._config.release_timeout_s <= 0.0:
            raise ROSConfigError(
                "vision attachment: release_clear_m and release_timeout_s must be positive, got "
                f"{self._config.release_clear_m} and {self._config.release_timeout_s}."
            )
        self._base_link = _release_base_link(description)
        links = set().union(*_tree_links(description))
        unknown = sorted(set(self._config.tf_frames) - links)
        if unknown:
            raise ROSConfigError(
                f"vision attachment: tf_frames names {unknown}, which are not links of "
                f"{description.name!r}."
            )

        # (raster, stamp ns, header.frame_id) of the newest depth frame.
        self._depth: tuple[NDArray[np.float64], int, str] | None = None
        self._depth_sub: Any = None
        self._camera_info: Any = None
        self._camera_info_sub: Any = None
        self._logged_depth_frame = False
        self._client: Any = None
        self._attachment_pub: Any = None
        self._tf_buffer: Any = None
        self._tf_listener: Any = None
        self._heartbeat_timer: Any = None
        self._heartbeat_open: bool | None = None
        self._evidence = _JawEvidence(
            trigger_config,
            timeout_s=self._config.evidence_timeout_s,
            spans_gaps=self._config.evidence_run_spans_gaps,
        )
        self._joint_order = [joint.name for joint in description.joints]
        self._warned_row_shape = False
        self._warned_slot = False
        self._revision = 0
        # Latest joint positions by name: the jaws' angles for the release test.
        self._positions: dict[str, float] = {}
        self._grasp_target: GraspTargetLeg | None = (
            GraspTargetLeg(
                node,
                self,
                self._config,
                support_search_below_m=self._config.grasp_target_support_search_below_m,
                support_probe_margin_m=self._config.grasp_target_support_probe_margin_m,
                occluder_margin_m=self._config.grasp_target_occluder_margin_m,
                approach_m=self._config.grasp_target_approach_m,
            )
            if self._config.grasp_target_enabled
            else None
        )
        self._place_target: PlaceTargetLeg | None = (
            PlaceTargetLeg(node, self, self._config) if self._config.place_target_enabled else None
        )

    # ── wiring ───────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Create the depth subscription, TF listener, service client, publisher."""
        import tf2_ros
        from openral_msgs.msg import AttachmentState
        from openral_msgs.srv import SegmentInView
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        from sensor_msgs.msg import CameraInfo, Image

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

        depth_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._depth_sub = self._node.create_subscription(
            Image, self._depth_topic(), self._on_depth, depth_qos
        )
        self._camera_info_sub = self._node.create_subscription(
            CameraInfo, self._camera_info_topic(), self._on_camera_info, depth_qos
        )
        # Same class the simulator bridge publishes on: the kernel's authoritative
        # attachment snapshot is latched description-class data.
        state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._attachment_pub = self._node.create_publisher(
            AttachmentState, "/openral/attachment_state", state_qos
        )
        self._client = self._node.create_client(SegmentInView, self._config.service_name)
        # The bridge is rebuilt on every HAL activate, and the aggregator rejects
        # a revision that moves backwards, so a counter restarting at 0 would
        # wedge the second activation. Seed from the node clock instead: host
        # wall time on a real cell, monotonic within a launch under sim time;
        # +1 per event after that. Nothing persists the last revision across
        # processes — the upgrade path, if wall time ever proves unfit, is to
        # read the latched topic's last revision before the first publish.
        self._revision = int(self._node.get_clock().now().nanoseconds)
        self._heartbeat_timer = self._node.create_timer(_HEARTBEAT_PERIOD_S, self._heartbeat)
        if self._grasp_target is not None:
            self._grasp_target.setup()
        if self._place_target is not None:
            self._place_target.setup()
        for leg in self._legs:
            self._node.get_logger().info(
                f"vision attachment bridge: gripper={leg.joint_name!r} camera={self._camera!r} "
                f"depth={self._depth_topic()!r} camera_info={self._camera_info_topic()!r} "
                f"service={self._config.service_name!r} "
                f"attach_link={leg.producer.attach_link!r} "
                f"tcp={leg.tcp_frame or leg.tcp_in_link!r} "
                f"deadline={self._config.deadline_s:.3f}s "
                f"stall_thresholds(closed,rest,gap,settle)={leg.trigger.thresholds}"
            )

    def teardown(self) -> None:
        """Destroy every ROS entity; idempotent, and never leaves the barrier shut."""
        for leg in self._legs:
            self._supersede(leg)
        if self._grasp_target is not None:
            self._grasp_target.teardown()
        if self._place_target is not None:
            self._place_target.teardown()
        if self._heartbeat_timer is not None:
            self._heartbeat_timer.cancel()
            self._node.destroy_timer(self._heartbeat_timer)
            self._heartbeat_timer = None
        if self._depth_sub is not None:
            self._node.destroy_subscription(self._depth_sub)
            self._depth_sub = None
        if self._camera_info_sub is not None:
            self._node.destroy_subscription(self._camera_info_sub)
            self._camera_info_sub = None
        if self._attachment_pub is not None:
            self._node.destroy_publisher(self._attachment_pub)
            self._attachment_pub = None
        self._client = None
        self._tf_listener = None
        self._tf_buffer = None
        # A bridge torn down mid-flight must not leave the node deferring an
        # acknowledgement forever. Clearing the flag is not enough now that the
        # node re-checks every barrier holder before releasing: a swallowed
        # notify is only safe because each holder re-issues one when it settles,
        # so the holder that goes away has to issue its last one here.
        for leg in self._legs:
            if leg.pending:
                self._release_barrier(leg)

    # ── barrier ──────────────────────────────────────────────────────────────

    def attachment_action_ack_ready(self) -> bool:
        """Whether attached-payload perception has settled for this tick.

        The same shape ``SimSensorBridge``
        exposes, so the node's deferred-ack path treats both identically.
        """
        return not any(leg.pending for leg in self._legs)

    def tf_frame(self, link: str) -> str:
        """The tf2 frame a manifest link is looked up as (``tf_frames``, else itself).

        Args:
            link: A manifest link name.

        Returns:
            The tf2 frame name.
        """
        return self._config.tf_frames.get(link, link)

    @property
    def gripper_joint_names(self) -> tuple[str, ...]:
        """The gripper joints this bridge watches, one leg each, in manifest order."""
        return tuple(leg.joint_name for leg in self._legs)

    # ── trigger ──────────────────────────────────────────────────────────────

    def observe_joint_state(self, state: JointState) -> None:
        """Fold one HAL read into every leg's grasp trigger and act on transitions.

        Also the heartbeat's liveness evidence: a sample counts only when every
        leg read a finite jaw position from it.

        Args:
            state: The tick's joint state, straight from the HAL.
        """
        self._positions.update(zip(state.name, (float(q) for q in state.position), strict=False))
        complete = True
        for leg in self._legs:
            missing_before = leg.trigger.missing_position_ticks
            event = leg.trigger.update(state)
            if (
                leg.trigger.last_command is None
                and not leg.announced_uncommanded
                and self._node is not None
            ):
                # Visible, once: a leg whose jaw is never commanded can never ATTACH.
                leg.announced_uncommanded = True
                self._node.get_logger().info(
                    f"grasp trigger {leg.joint_name}: no commanded target yet — it cannot "
                    "ATTACH until an applied action commands this joint"
                )
            complete = complete and leg.trigger.missing_position_ticks == missing_before
            if event is None:
                continue
            if self._supersede(leg):
                self._node.get_logger().warning(
                    f"grasp trigger {leg.joint_name}: {event.value.upper()} superseded the "
                    "segmentation still in flight; its reply will be dropped"
                )
            if event is GraspEvent.DETACH:
                held = leg.attachment
                self._open_release(leg)
                leg.attachment = None
                self._publish_attachment()
                if leg.pending:
                    self._release_barrier(leg)
                if self._grasp_target is not None and held is not None:
                    # The pick-complete path: the frozen record (None without tf2).
                    record = leg.release.record if leg.release is not None else None
                    self._grasp_target.on_detach(leg.jaw_link, record)
                continue
            if leg.release is not None:
                self._close_release(leg, "attach")
            self._node.get_logger().info(
                f"grasp trigger {leg.joint_name}: {event.value.upper()} — holding the action "
                "ack for segmentation"
            )
            self._begin_segmentation(leg, event, stamp_ns=int(state.stamp_ns))
        self._evidence.observe(
            complete=complete, stamp_ns=int(state.stamp_ns), now_s=time.monotonic()
        )
        if self._place_target is not None:
            self._place_target.on_joint_state()

    def observe_command(self, action: Action) -> None:
        """Fold the command the HAL applied into every leg's trigger.

        The commanded target is the chunk's **last** row — what the ros2_control
        transport hands the trajectory controller as the goal. Pass what the HAL
        *applied*: for an ADR-0102 slot group, the HAL's composed full-dof action
        (``last_applied_action``) once it committed the tick — never the individual
        slots, which this bridge does not stage (a slot, ``tick_group_size > 1``, is
        ignored and logged once). A sim group that does not compose (a base-twist slot)
        hands over only its gripper targets as a compact row
        (``sim_attached.gripper_targets_action``), read positionally like any compact row.

        A ``JOINT_POSITION`` action: a full-dof row is padded and read at each owned
        joint's manifest index (every joint when it names none); a row as long as its
        ``joint_names`` is compact and read positionally; any other length is logged
        once and ignored (``_row_targets``). Any other action leaves every leg's last
        command as it is.

        Args:
            action: The command the HAL just applied (safety-approved).
        """
        if int(action.tick_group_size) > 1:
            if not self._warned_slot and self._node is not None:
                self._warned_slot = True
                self._node.get_logger().warning(
                    "grasp trigger: observe_command was handed a slot of a group, not the "
                    "HAL's applied command; not folded in (logged once)"
                )
            return
        if action.control_mode is not ControlMode.JOINT_POSITION or not action.joint_targets:
            return
        targets = self._row_targets(action.joint_targets[-1], action.joint_names)
        if targets is None:
            return
        for leg in self._legs:
            target = targets.get(leg.joint_name)
            if target is not None:
                leg.trigger.command(target)

    def clear_command(self) -> None:
        """Forget every leg's commanded target: the HAL applied a command nobody can read.

        Called by the HAL node when a committed slot group's command is unknown
        (``PositionStallTrigger.clear_command``): no leg may measure "short of the
        command" against an older one until the next readable command arrives.
        """
        for leg in self._legs:
            leg.trigger.clear_command()

    def _row_targets(
        self, row: Sequence[float], names: Sequence[str] | None
    ) -> dict[str, float] | None:
        """An ungrouped JOINT_POSITION row by joint name; ``None`` when unplaceable.

        A full-dof row is padded (ADR-0102): each owned joint is read at its manifest
        index, as the HAL's slot composition reads it — also when ``joint_names`` lists every
        joint, the padded reading being the contract. A row as long as ``joint_names``
        is compact: ``joint_names[i]`` owns ``row[i]``. Any other length says nothing
        placeable and is logged (once).
        """
        if len(row) == len(self._joint_order):
            owned = set(names or self._joint_order)
            return {
                name: float(row[i]) for i, name in enumerate(self._joint_order) if name in owned
            }
        if names and len(row) == len(names):
            return {name: float(value) for name, value in zip(names, row, strict=True)}
        if not self._warned_row_shape and self._node is not None:
            self._warned_row_shape = True
            self._node.get_logger().warning(
                f"grasp trigger: ROSConfigError: a JOINT_POSITION row of {len(row)} values is "
                f"neither full dof ({len(self._joint_order)}) nor as long as its joint_names "
                f"({len(names or [])}); not folded in as a jaw command (logged once)"
            )
        return None

    @property
    def missing_position_ticks(self) -> int:
        """Ticks whose gripper carried no finite position, summed over every leg.

        A driver-health signal.
        """
        return sum(leg.trigger.missing_position_ticks for leg in self._legs)

    # ── segmentation round trip ──────────────────────────────────────────────

    def _supersede(self, leg: _GripperLeg) -> bool:
        """Invalidate the leg's in-flight request, if any; whether there was one.

        Advances the generation, cancels and destroys the deadline timer, and drops
        the future *before* cancelling it — ``rclpy.Future.cancel`` runs the done
        callback synchronously, and the generation check is what discards it. The
        barrier (``pending``) is left to the caller: the next request keeps it, a
        DETACH releases it.
        """
        leg.generation += 1
        self._cancel_deadline(leg)
        inflight, leg.inflight = leg.inflight, None
        if inflight is None:
            return False
        inflight.cancel()
        return True

    def _begin_segmentation(self, leg: _GripperLeg, event: GraspEvent, *, stamp_ns: int) -> None:
        """Attach the latched target region, or close the barrier and ask SegmentInView.

        Only an ATTACH may take the region: on a REGRASP the jaws have re-seated the
        payload away from where the pre-grasp region measured it, so it is segmented.
        """
        taken = self._region_payload(leg, stamp_ns=stamp_ns) if event is GraspEvent.ATTACH else None
        if taken is not None:
            held, region = taken
            leg.attachment = held
            self._node.get_logger().info(
                f"vision attachment {leg.joint_name}: {held.object_id!r} from the grasp-target "
                f"region ({held.evidence_ref}); no segmentation"
            )
            if self._grasp_target is not None:
                self._grasp_target.on_attach(leg.jaw_link, held, region=region)
            self._publish_attachment()
            if leg.pending:  # a superseded request was still holding the barrier
                self._release_barrier(leg)
            return
        leg.pending = True
        context = self._gather_context(leg)
        if isinstance(context, str):  # a typed precondition failure
            self._finish(leg, stamp_ns=stamp_ns, masks=[], scores=[], reason=context)
            return
        t_link_from_cam, tcp_in_link, negatives = context
        if self._client is None or not self._client.service_is_ready():
            self._finish(
                leg,
                stamp_ns=stamp_ns,
                masks=[],
                scores=[],
                reason=(
                    f"ROSDispatchUnavailable: no server on {self._config.service_name!r}; "
                    "is the segmenter node active?"
                ),
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            )
            return

        request = build_segment_request(
            stamp_ns=stamp_ns,
            camera=self._camera,
            t_link_from_cam=t_link_from_cam,
            tcp_in_link=tcp_in_link,
            negatives_in_link=negatives,
        )
        generation = leg.generation
        future = self._client.call_async(request)
        leg.inflight = future
        future.add_done_callback(
            lambda fut: self._on_reply(
                leg,
                fut,
                generation=generation,
                stamp_ns=stamp_ns,
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            )
        )
        leg.deadline_timer = self._node.create_timer(
            self._config.deadline_s,
            lambda: self._on_deadline(
                leg,
                generation=generation,
                stamp_ns=stamp_ns,
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            ),
        )

    def _on_reply(
        self,
        leg: _GripperLeg,
        future: Any,
        *,
        generation: int,
        stamp_ns: int,
        t_link_from_cam: NDArray[np.float64],
        tcp_in_link: tuple[float, float, float],
    ) -> None:
        """Resolve the attachment from a SegmentInView reply (or its exception)."""
        if generation != leg.generation or future is not leg.inflight:
            return  # resolved by the deadline, or superseded by a newer grasp event
        self._cancel_deadline(leg)
        leg.inflight = None
        try:
            response = future.result()
        except Exception as exc:  # a service exception must degrade, not propagate
            self._finish(
                leg,
                stamp_ns=stamp_ns,
                masks=[],
                scores=[],
                reason=f"ROSDispatchUnavailable: SegmentInView raised {type(exc).__name__}: {exc}",
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            )
            return
        if response is None:
            # A cancelled future resolves to None without raising. Belt and
            # braces behind the in-flight guard above: an AttributeError escaping
            # here would be swallowed by rclpy's callback wrapper and leave the
            # barrier held with no explanation anywhere.
            self._finish(
                leg,
                stamp_ns=stamp_ns,
                masks=[],
                scores=[],
                reason="ROSDispatchUnavailable: SegmentInView future resolved with no response",
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            )
            return
        outcome = resolve_segment_outcome(
            timed_out=False,
            ok=bool(response.ok),
            failure_reason=str(response.failure_reason),
            mask_count=len(response.masks),
        )
        reason = outcome.reason
        use_masks = outcome.use_masks
        if use_masks and self._depth is not None:
            # The depth _finish back-projects through is the cached one, now.
            skew = mask_depth_skew_reason(
                mask_stamps_ns(response.masks),
                self._depth[1],
                max_skew_s=self._config.mask_depth_max_skew_s,
            )
            if skew:
                use_masks, reason = False, skew
        masks = (
            [
                decode_mono8_mask(bytes(image.data), height=image.height, width=image.width)
                for image in response.masks
            ]
            if use_masks
            else []
        )
        self._finish(
            leg,
            stamp_ns=stamp_ns,
            masks=masks,
            scores=[float(s) for s in response.mask_scores_advisory] if use_masks else [],
            reason=reason,
            t_link_from_cam=t_link_from_cam,
            tcp_in_link=tcp_in_link,
        )

    def _on_deadline(
        self,
        leg: _GripperLeg,
        *,
        generation: int,
        stamp_ns: int,
        t_link_from_cam: NDArray[np.float64],
        tcp_in_link: tuple[float, float, float],
    ) -> None:
        """Resolve conservatively when the segmenter overran its budget."""
        if generation != leg.generation:
            return  # superseded: ``_supersede`` already destroyed that timer
        self._cancel_deadline(leg)
        inflight = leg.inflight
        if inflight is None:
            return
        # Drop the handle BEFORE cancelling. ``rclpy.Future.cancel`` completes
        # the future, which synchronously runs the done callback we registered —
        # and that callback's "is this still the in-flight one?" guard is the
        # only thing keeping it from resolving a reply that never arrived.
        leg.inflight = None
        inflight.cancel()
        outcome = resolve_segment_outcome(timed_out=True, ok=False, failure_reason="", mask_count=0)
        self._finish(
            leg,
            stamp_ns=stamp_ns,
            masks=[],
            scores=[],
            reason=outcome.reason,
            t_link_from_cam=t_link_from_cam,
            tcp_in_link=tcp_in_link,
        )

    def _finish(
        self,
        leg: _GripperLeg,
        *,
        stamp_ns: int,
        masks: Sequence[NDArray[np.bool_]],
        scores: Sequence[float],
        reason: str,
        t_link_from_cam: NDArray[np.float64] | None = None,
        tcp_in_link: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Gate, publish, log, and release the barrier — on every path.

        ``masks`` empty is not an error branch: the producer answers it with its
        conservative jaw-span box, so the collision checker always receives
        geometry.
        """
        depth = self._depth[0] if self._depth is not None else np.zeros((0, 0), dtype=np.float64)
        transform = t_link_from_cam if t_link_from_cam is not None else np.eye(4, dtype=np.float64)
        intrinsics = self._no_mask_intrinsics
        if masks:
            aligned = self._align_to_depth(masks, depth)
            if isinstance(aligned, str):
                masks, scores, reason = [], [], aligned
            else:
                masks, intrinsics = aligned

        def _fit(candidates: Sequence[NDArray[np.bool_]], advisory: Sequence[float]) -> Any:
            return leg.producer.on_grasp(
                masks=candidates,
                depth_m=depth,
                intrinsics=intrinsics,
                t_link_from_cam=transform,
                tcp_in_link=tcp_in_link,
                object_id=leg.object_id,
                stamp_ns=stamp_ns,
                mask_scores_advisory=advisory,
            )

        try:
            attachment, report = _fit(masks, scores)
        except ROSConfigError as exc:
            # Shape disagreements (mask vs depth vs intrinsics) are configuration
            # bugs, not grasp outcomes. They must neither stall the tick nor
            # leave the payload invisible, so the conservative box still ships.
            reason = f"ROSConfigError: {exc}"
            self._node.get_logger().error(f"vision attachment producer rejected inputs: {exc}")
            attachment, report = _fit([], [])
        if reason:
            self._node.get_logger().warning(
                f"vision attachment {leg.joint_name} fell back to GRIPPER_CLOSURE: {reason}"
            )
        self._node.get_logger().info(
            f"vision attachment {leg.joint_name}: accepted={report.accepted} "
            f"rejections={list(report.rejections)} "
            f"candidate={report.candidate_index}/{report.candidate_count} "
            f"points={report.point_count} extents_m={report.extents_m} "
            f"depth_valid={report.depth_valid_fraction:.2f} "
            f"mask_score_advisory={report.mask_score_advisory:.3f} (advisory only)"
        )
        leg.attachment = attachment
        if self._grasp_target is not None and attachment is not None:
            # Segmented: handed over only if the payload is on the armed region (it may
            # be a neighbour — ``_region_payload`` refused the region for this grasp).
            self._grasp_target.on_attach(leg.jaw_link, attachment, region=None)
        self._publish_attachment()
        self._release_barrier(leg)

    def _release_barrier(self, leg: _GripperLeg) -> None:
        """Reopen this leg's hold and let the node re-check the barrier."""
        leg.pending = False
        if callable(self._on_perception_ready):
            self._on_perception_ready()

    def _cancel_deadline(self, leg: _GripperLeg) -> None:
        """Cancel and drop a leg's one-shot deadline timer, if any."""
        if leg.deadline_timer is None:
            return
        leg.deadline_timer.cancel()
        self._node.destroy_timer(leg.deadline_timer)
        leg.deadline_timer = None

    # ── inputs ───────────────────────────────────────────────────────────────

    def _region_payload(
        self, leg: _GripperLeg, *, stamp_ns: int
    ) -> tuple[AttachedCollisionObject, PlaceRegion] | None:
        """The latched grasp-target region as this leg's payload, when it is confirmed.

        Confirmation is geometric: the declaration names this leg's jaw link and the
        leg's TCP (``jaw_point``) lies within ``grasp_target_occluder_margin_m`` of the
        region. The region is a pre-grasp measurement, so a leg takes each measured
        region once (``region_spent``, keyed by ``target_id`` — one per pick — and the
        region's own ``stamp_ns``): a later ATTACH offered the same region — the object
        set down and picked up again before any re-measurement — is segmented, while a
        region measured for the goal's next pick (a fresh identity) is a new payload.
        Returns the payload
        and the region it was built from (what ``GraspTargetLeg.on_attach`` hands
        over); ``None`` — logged when a declaration was live — sends the grasp to
        ``SegmentInView`` instead.
        """
        if self._grasp_target is None:
            return None
        # The envelope, not the raw held region: it applies the freeze TTL and the
        # declaration's own expiry, so a stale region is never handed over.
        declaration = self._grasp_target.tracker.envelope(
            now_ns=int(self._node.get_clock().now().nanoseconds)
        )
        region = declaration.region if declaration is not None else None
        if declaration is None or region is None or leg.jaw_link not in declaration.contact_links:
            return None
        # The region's own measurement stamp: one arming may re-measure its region
        # before the grasp, and each measurement is one payload.
        key = (declaration.target_id, int(region.stamp_ns))
        if leg.region_spent == key:
            self._node.get_logger().info(
                f"vision attachment {leg.joint_name}: grasp-target region for "
                f"{declaration.target_id!r} (measured at {region.stamp_ns}) already handed "
                "over; segmenting instead"
            )
            return None
        hand = self.jaw_point(leg.jaw_link, region.frame_id)
        link = self.tf_frame(leg.producer.attach_link)
        t_link_from_region = self._lookup(link, region.frame_id)
        why = ""
        if hand is None or t_link_from_region is None:
            why = f"no tf2 {link} <- {region.frame_id}"
        elif not _hand_near([hand], region, reach_m=self._config.grasp_target_occluder_margin_m):
            why = (
                f"jaw at {tuple(round(v, 3) for v in hand)} is not within "
                f"{self._config.grasp_target_occluder_margin_m} m of the region"
            )
        if why or t_link_from_region is None:
            self._node.get_logger().warning(
                f"vision attachment {leg.joint_name}: grasp-target region for "
                f"{declaration.target_id!r} not used — {why}; segmenting instead"
            )
            return None
        leg.region_spent = key
        held = region_attachment(
            declaration,
            attach_link=leg.producer.attach_link,
            touch_links=leg.producer.touch_links,
            t_link_from_region=t_link_from_region,
            stamp_ns=stamp_ns,
        )
        return held, region

    def _gather_context(self, leg: _GripperLeg) -> _PromptContext | str:
        """Collect depth, transforms and prompt geometry, or name what is missing.

        Returns the ``(t_link_from_cam, tcp_in_link, negative_points)`` tuple, or
        a typed reason string when a precondition is unmet. The caller turns
        either into an attachment; neither turns into a stall.
        """
        if self._depth is None:
            return f"ROSPerceptionStale: no depth frame yet on {self._depth_topic()!r}"
        link = self.tf_frame(leg.producer.attach_link)
        # The optical frame is the one the depth raster says it is in — never
        # guessed from the manifest, whose frame_id may name a body frame.
        camera_frame = self._depth[2]
        if not camera_frame:
            return "ROSPerceptionStale: depth frame has no header.frame_id"
        if not self._logged_depth_frame:
            self._logged_depth_frame = True
            manifest_frame = self._camera_frame()
            if camera_frame != manifest_frame:
                self._node.get_logger().info(
                    f"vision attachment: back-projecting in the depth header's frame "
                    f"{camera_frame!r}, not the manifest's {manifest_frame!r} for "
                    f"camera {self._camera!r}"
                )
        t_link_from_cam = self._lookup(link, camera_frame)
        if t_link_from_cam is None:
            return f"ROSPerceptionStale: no tf2 {link} <- {camera_frame}"
        tcp = self._tcp_in(leg, link)
        if tcp is None:
            return f"ROSPerceptionStale: no tf2 {link} <- {leg.tcp_frame}"
        negatives: list[tuple[float, float, float]] = []
        for frame in self._config.jaw_tip_frames:
            tip = self._lookup(link, frame)
            if tip is not None:
                negatives.append((float(tip[0, 3]), float(tip[1, 3]), float(tip[2, 3])))
        return t_link_from_cam, tcp, negatives

    def _tcp_in(self, leg: _GripperLeg, link: str) -> tuple[float, float, float] | None:
        """The leg's TCP in tf2 frame ``link`` (its attach link), or ``None`` without tf."""
        if not leg.tcp_frame:
            return leg.tcp_in_link
        t_link_from_tcp = self._lookup(link, leg.tcp_frame)
        if t_link_from_tcp is None:
            return None
        return (
            float(t_link_from_tcp[0, 3]),
            float(t_link_from_tcp[1, 3]),
            float(t_link_from_tcp[2, 3]),
        )

    def jaw_point(self, jaw_link: str, frame: str) -> tuple[float, float, float] | None:
        """Where the hand whose jaw link is ``jaw_link`` is, in tf2 frame ``frame``.

        The same TCP the attachment prompt uses: the leg's attach link through tf2
        (``tf_frames``-mapped) composed with its TCP — never a tf lookup of the jaw
        link itself, which may be a manifest-only link (the OpenArm's
        ``openarm_<side>_finger_pair`` is not a tf frame).

        Args:
            jaw_link: A gripper joint's child link, as ``GraspDeclaration.contact_links``
                names it.
            frame: The tf2 frame to express the point in.

        Returns:
            The point, or ``None`` for a link no leg owns or a missing transform.
        """
        leg = next((each for each in self._legs if each.jaw_link == jaw_link), None)
        if leg is None:
            return None
        link = self.tf_frame(leg.producer.attach_link)
        t_frame_from_link = self._lookup(frame, link)
        tcp = self._tcp_in(leg, link)
        if t_frame_from_link is None or tcp is None:
            return None
        x, y, z = (float(v) for v in t_frame_from_link[:3, :3] @ tcp + t_frame_from_link[:3, 3])
        return x, y, z

    def _on_depth(self, msg: Any) -> None:
        """Cache the newest metric-depth raster for the attach camera."""
        from openral_hal.depth_cloud import depth_grid_from_image

        try:
            grid = depth_grid_from_image(msg)
        except ROSConfigError as exc:
            self._node.get_logger().warning(f"vision attachment depth dropped: {exc}")
            return
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self._depth = (grid, stamp_ns, str(msg.header.frame_id).strip())

    def _on_camera_info(self, msg: Any) -> None:
        """Cache the newest ``CameraInfo`` for the depth stream."""
        self._camera_info = msg

    def _align_to_depth(
        self,
        masks: Sequence[NDArray[np.bool_]],
        depth: NDArray[np.float64],
    ) -> tuple[list[NDArray[np.bool_]], IntrinsicsPinhole] | str:
        """Live intrinsics for the depth raster and the masks on its grid, or a typed reason.

        The intrinsics come only from the driver's ``CameraInfo`` — this geometry
        feeds collision, so a missing or mismatched calibration falls back to the
        conservative box rather than to the manifest's nominal K. A mask at a
        different resolution of the same view (aspect within 1e-3) is resampled
        nearest-neighbour; a different crop is refused.
        """
        from openral_hal.depth_cloud import intrinsics_from_camera_info, resample_mask_nearest

        topic = self._camera_info_topic()
        if self._camera_info is None:
            return f"ROSPerceptionStale: no CameraInfo yet on {topic!r}"
        try:
            intrinsics = intrinsics_from_camera_info(self._camera_info)
        except ROSConfigError as exc:
            return f"ROSConfigError: {exc}"
        height, width = depth.shape
        if (intrinsics.height, intrinsics.width) != (height, width):
            return (
                f"ROSConfigError: CameraInfo on {topic!r} is {intrinsics.width}x"
                f"{intrinsics.height} but the depth raster is {width}x{height}"
            )
        aligned: list[NDArray[np.bool_]] = []
        for mask in masks:
            if mask.shape == depth.shape:
                aligned.append(mask)
                continue
            mask_h, mask_w = mask.shape
            if abs(mask_h * width - height * mask_w) > _ASPECT_TOLERANCE * mask_h * width:
                return (
                    f"ROSConfigError: mask {mask_w}x{mask_h} and depth {width}x{height} "
                    "differ in aspect — different crops, not a registered pair"
                )
            try:
                aligned.append(resample_mask_nearest(mask, (height, width)))
            except ROSConfigError as exc:
                return f"ROSConfigError: {exc}"
        return aligned, intrinsics

    def _lookup(self, target: str, source: str) -> NDArray[np.float64] | None:
        """Latest ``target <- source`` transform as a 4x4, or ``None``."""
        if target == source:
            return np.eye(4, dtype=np.float64)
        import rclpy
        from openral_core.geometry import homogeneous_from_quat_xyz

        try:
            tf = self._tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        except Exception as exc:  # tf2 raises several distinct lookup errors
            self._node.get_logger().debug(f"tf {target} <- {source} unavailable: {exc}")
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return homogeneous_from_quat_xyz((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

    def _publish_attachment(self) -> None:
        """Publish every leg's current attachment as one snapshot at a fresh revision."""
        if self._attachment_pub is None:
            return
        self._revision += 1
        self._publish_snapshot()

    # ── release window ───────────────────────────────────────────────────────

    def _open_release(self, leg: _GripperLeg) -> None:
        """Freeze the payload a DETACH released, or say why it is dropped at once."""
        held = leg.attachment
        if held is None:
            self._node.get_logger().info(
                f"grasp trigger {leg.joint_name}: DETACH — nothing was attached"
            )
            return
        t_base_from_link = self._lookup(
            self.tf_frame(self._base_link), self.tf_frame(held.attach_link)
        )
        if t_base_from_link is None:
            # Without the hand's pose there is nothing honest to freeze; dropping
            # is the pre-window behaviour — the re-marked payload stops the retreat.
            self._node.get_logger().warning(
                f"release window not opened for {leg.joint_name}: no tf2 "
                f"{self.tf_frame(self._base_link)} <- {self.tf_frame(held.attach_link)}; "
                "dropping its attachment"
            )
            return
        leg.release = ReleaseWindow.open(
            self._description,
            held,
            base_link=self._base_link,
            t_base_from_link=t_base_from_link,
            now_s=time.monotonic(),
        )
        self._node.get_logger().info(
            f"release window opened for {leg.joint_name}: {held.object_id!r} frozen in "
            f"{self._base_link!r} at {leg.release.record.pose_in_link.xyz}; exempt links "
            f"{leg.release.record.touch_links} until clear by {self._config.release_clear_m} m "
            f"or {self._config.release_timeout_s} s"
        )

    def _close_release(self, leg: _GripperLeg, reason: str) -> None:
        """Drop a leg's frozen record (the caller publishes), logging why once."""
        self._node.get_logger().info(f"release window closed for {leg.joint_name}: reason={reason}")
        leg.release = None
        if self._grasp_target is not None and reason != "attach":
            # A re-ATTACH is a new grasp of the same hand, not the hand emptying.
            self._grasp_target.on_release_closed(leg.jaw_link)

    def _poll_releases(self) -> None:
        """Close every window whose jaws are clear or whose time is up; publish if any did."""
        now_s = time.monotonic()
        closed = False
        for leg in self._legs:
            window = leg.release
            if window is None:
                continue
            t_hand = self._lookup(self.tf_frame(self._base_link), self.tf_frame(window.hand_link))
            clearance = None if t_hand is None else window.clearance_m(t_hand, self._positions)
            reason = window.close_reason(
                now_s=now_s,
                clearance_m=clearance,
                clear_m=self._config.release_clear_m,
                timeout_s=self._config.release_timeout_s,
            )
            if reason:
                self._close_release(leg, reason)
                closed = True
        if closed:
            self._publish_attachment()

    def _heartbeat(self) -> None:
        """Republish the current set at the current revision, while evidence backs it.

        Fail-closed: no live jaw-position evidence, a grasp still being resolved, or a
        trigger that believes the jaws are loaded with no attachment yet each
        stop the heartbeat, so the kernel's snapshot ages into a drop instead of
        a stale claim. Each open/close transition is logged once.
        """
        if self._attachment_pub is None:
            return
        self._poll_releases()
        if not self._evidence.live(now_s=time.monotonic()):
            reason = "no live jaw-position evidence from every gripper"
        elif any(leg.pending for leg in self._legs):
            reason = "a grasp is being resolved"
        elif any(leg.trigger.attached != (leg.attachment is not None) for leg in self._legs):
            reason = "a trigger and its attachment disagree"
        else:
            reason = ""
        is_open = not reason
        if is_open != self._heartbeat_open:
            self._heartbeat_open = is_open
            if is_open:
                self._node.get_logger().info(
                    "vision attachment heartbeat: publishing — jaw-position evidence is live"
                )
            else:
                self._node.get_logger().warning(
                    f"vision attachment heartbeat: withheld — {reason}; the kernel's "
                    "attachment snapshot will age out"
                )
        if is_open:
            self._publish_snapshot()

    def _publish_snapshot(self) -> None:
        """Publish the union of the legs' attachments at the current revision, stamped now."""
        objects: list[AttachedCollisionObject] = []
        for leg in self._legs:
            if leg.attachment is not None:
                objects.append(leg.attachment)
            elif leg.release is not None:
                objects.append(leg.release.record)
        from openral_msgs.msg import (
            AttachedCollisionObject,
            AttachedCollisionPrimitive,
            AttachmentState,
        )

        msg = AttachmentState()
        now = self._node.get_clock().now()
        msg.header.stamp = now.to_msg()
        msg.revision = self._revision
        if self._grasp_target is not None:
            self._grasp_target.fill(msg, now_ns=int(now.nanoseconds))
        if self._place_target is not None:
            self._place_target.fill(msg, now_ns=int(now.nanoseconds))
            objects = self._place_target.decorate(objects)
        for obj in objects:
            item = AttachedCollisionObject()
            obj.fill_idl(item, primitive_factory=AttachedCollisionPrimitive)
            msg.objects.append(item)
        self._attachment_pub.publish(msg)

    # ── manifest resolution ──────────────────────────────────────────────────

    def _sensors(self) -> list[Any]:
        """Every SensorSpec on the manifest, standalone and bundled."""
        specs = list(self._description.sensors)
        for bundle in self._description.sensor_bundles:
            specs.extend(bundle.sensors)
        return specs

    def _resolve_camera(self) -> tuple[str, IntrinsicsPinhole]:
        """Resolve the attach camera's id (and its existence) from the manifest."""
        specs = [spec for spec in self._sensors() if spec.intrinsics is not None]
        if self._config.camera:
            specs = [spec for spec in specs if spec.name == self._config.camera]
            if not specs:
                raise ROSConfigError(
                    f"vision attachment: camera {self._config.camera!r} has no SensorSpec with "
                    f"intrinsics on {self._description.name!r}."
                )
        if not specs:
            raise ROSConfigError(
                f"vision attachment needs a camera SensorSpec with intrinsics on "
                f"{self._description.name!r}; none declares any."
            )
        return str(specs[0].name), specs[0].intrinsics

    def _camera_frame(self) -> str:
        """The manifest's ``frame_id`` for the attach camera — logged, never projected in."""
        for spec in self._sensors():
            if spec.name == self._camera:
                return str(spec.frame_id)
        raise ROSConfigError(f"vision attachment: camera {self._camera!r} vanished from manifest.")

    def _build_legs(
        self,
        gate_config: VisionGateConfig | None,
        trigger_config: PositionStallConfig | None,
    ) -> list[_GripperLeg]:
        """One leg per gripper joint, with its TCP resolved.

        The TCP is the gripper's moving-jaw child link — an approximation, and
        named as one: no schema field carries a calibrated tool center point
        today, and the jaw link is the closest frame every manifest declares.
        With one gripper it is looked up through tf2 (``tcp_frame``, defaulting
        to the child link) exactly as before. With several, it is the joint's
        ``origin_xyz`` — the child link origin in the attach link, read from
        the manifest — because the attach link may be absent from the published
        TF tree, and per-hand frame overrides are not supported.
        """
        joints = gripper_joints(self._description)
        if len(joints) > 1 and (self._config.tcp_frame or self._config.jaw_tip_frames):
            raise ROSConfigError(
                f"vision attachment: tcp_frame / jaw_tip_frames are single-gripper settings, "
                f"but {self._description.name!r} has gripper joints "
                f"{[joint.name for joint in joints]}."
            )
        return [
            _GripperLeg(
                joint_name=joint.name,
                trigger=PositionStallTrigger(
                    self._description, joint_name=joint.name, config=trigger_config
                ),
                producer=VisionAttachmentEvidenceProducer(
                    self._description, gripper_joint=joint.name, config=gate_config
                ),
                object_id=f"{self._config.object_id}:{joint.name}",
                tcp_frame=(self._config.tcp_frame or joint.child_link) if len(joints) == 1 else "",
                tcp_in_link=joint.origin_xyz,
                jaw_link=joint.child_link,
            )
            for joint in joints
        ]

    def _depth_topic(self) -> str:
        """Configured depth topic, or the conventional per-camera default."""
        return self._config.depth_topic or camera_topic(self._camera, CameraTopicKind.DEPTH_IMAGE)

    def _camera_info_topic(self) -> str:
        """Configured depth ``CameraInfo`` topic, or the conventional per-camera default."""
        return self._config.camera_info_topic or camera_topic(
            self._camera, CameraTopicKind.DEPTH_CAMERA_INFO
        )
