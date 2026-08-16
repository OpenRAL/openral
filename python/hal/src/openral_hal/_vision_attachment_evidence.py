"""Vision attachment evidence for real hardware — masked wrist depth to payload geometry.

The real-hardware counterpart of
:mod:`~openral_hal._sim_attachment_evidence`. That module reads MuJoCo's
ground truth: exact contacts, exact geoms, exact mass, and an exact kinematic
classification of the contacted body. None of that exists on a real robot. This
module answers the same question from a wrist RGB-D frame and a mask, and emits
the **identical** :class:`~openral_core.AttachedCollisionObject` contract — the
safety kernel cannot tell which producer filled it in, and does not need to.

Flow
----

1. A ``kind: "segmenter"`` rSkill (SAM 2.1) is prompted with a single positive
   point at the TF-projected tool center point on the wrist camera and returns
   candidate masks.
2. Each candidate mask is intersected with the same frame's depth channel and
   back-projected to a point cloud, which is transformed into the attach link's
   frame.
3. **Geometric gates** decide which candidates are trustworthy, and geometry
   also picks between them — the segmenter returns SAM 2's nested subpart /
   part / whole hypotheses precisely because, having no depth, it cannot choose.
   Among candidates clearing every gate the most inclusive wins.
4. The selected cloud is reduced to a PCA-oriented bounding box and split into
   the same ``<= 16``-box primitive contract the simulator producer emits.

Why the gates are geometric, and only geometric
-----------------------------------------------

The segmenter's own confidence cannot detect a mis-aimed prompt. Measured on a
real in-tree frame: a point prompt that landed on background returned a mask
covering **59.8% of the image** — essentially an entire tablecloth — at the
model's **highest** score, ``0.977``. A gate on that score would have passed it.

So every gate here is geometric: containment near the jaws, a per-axis payload
extent cap, a scalar volume backstop, and a depth-validity fraction. The mask
score is carried into the trace as
:attr:`VisionAttachmentReport.mask_score_advisory` and is never compared against
anything.

**On gate failure the attachment is not skipped.** :meth:`on_grasp` always
returns an :class:`~openral_core.AttachedCollisionObject`; a rejected mask
degrades to a conservative jaw-span box stamped
:attr:`~openral_core.AttachmentEvidenceKind.GRIPPER_FORCE` at low confidence.
A mis-segmented grasp still gets *some* geometry in front of the collision
checker, which is strictly safer than an invisible payload.

Honest limitations — what this does NOT fix
-------------------------------------------

* **Articulated objects.** Vision gives geometry, not kinematic class. Where
  the simulator producer classifies FREE / HINGE / SLIDE / ARTICULATED / FIXED
  from ground truth, this module cannot: a drawer handle and a free box look
  identical to it. Scope is therefore **free objects only**, by deliberate
  exclusion, and nothing here detects a violation of that scope.
* **Support contact.** This establishes what is *attached*, never what is
  *resting on* something. That stays with the separate support-contact witness.
* **Transparent / thin objects.** Expected to fail the depth-validity gate and
  degrade to the fallback box. Not specially handled.
* **Self-occlusion.** The far side of the object is never observed. Only
  partially mitigated by :attr:`VisionGateConfig.view_ray_inflation_m`, which is
  an unbenchmarked free parameter.
* **Mass / centre of mass / inertia.** Not estimable from vision;
  ``mass_kg`` and friends stay ``None``, unlike the simulator producer which
  reads them from MuJoCo.
* **Deformable objects and multi-object grasps.** A single positive point means
  a single rigid object. Not addressed.
* **The attach trigger itself.** This module is *told* a grasp happened; it does
  not decide it. Whether SO-101's feetech effort readback is trustworthy enough
  to be that trigger is unverified.

Calibration status of the thresholds
------------------------------------

Every threshold on :class:`VisionGateConfig` is a **calibration point, not a
measured constant** — see that class's docstring. The design work behind this
module benchmarked latency, VRAM and mask quality; it did **not** benchmark a
single gate threshold. They are set conservatively and must be tuned per robot
before this producer is trusted on hardware.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    Pose6D,
    RobotDescription,
)
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import rotation_to_quat_wxyz

if TYPE_CHECKING:  # pragma: no cover — typing only
    from openral_core import IntrinsicsPinhole

__all__ = [
    "VisionAttachmentEvidenceProducer",
    "VisionAttachmentReport",
    "VisionGateConfig",
    "backproject_masked_depth",
    "clustered_obb_primitives",
    "jaw_span_primitive",
]

# Minimum points a cloud needs before a PCA orientation means anything. Below
# this the covariance is degenerate and the "oriented" box is noise.
_MIN_CLOUD_POINTS = 24

# Padding added to every fitted box half-extent, matching the simulator
# producer's own `+ 1e-4` so neither producer can emit a zero-thickness slab.
_BOX_EPSILON_M = 1e-4

# Confidence stamped on an accepted vision attachment. Below the simulator
# producer's 1.0 (which reads exact contacts) because this one infers geometry
# from one partially-occluded viewpoint. CALIBRATION POINT — chosen, not
# measured; there is no validated mapping from gate margins to a probability.
_VISION_CONFIDENCE = 0.7

# Confidence stamped on the conservative fallback box. Deliberately low: the
# geometry is a guess, and a consumer ranking evidence should prefer almost
# anything else. CALIBRATION POINT.
_FALLBACK_CONFIDENCE = 0.2


@dataclass(frozen=True)
class VisionGateConfig:
    """Fail-closed geometric gate thresholds for the vision producer.

    .. warning::

       **None of these values is benchmarked.** The design work behind this
       module measured segmenter latency, VRAM and mask quality on real frames;
       it explicitly left every gate threshold as an open constant. Each field
       below is a deliberately conservative starting point and a **calibration
       point** that must be tuned against the target robot and camera before
       this producer is trusted on hardware. They are documented as chosen, not
       as measured (CLAUDE.md §1.2).

    The extent cap is expressed **per axis**, not as a single volume, because
    the three axes are not equally constrained: the jaw axis has a hard physical
    bound (whatever is held fits between the jaws) while the other two are
    comparatively unbounded. A scalar volume conflates one hard constraint with
    two soft ones. :attr:`max_payload_volume_m3` is kept as a cheap backstop for
    the pathological case where all three axes sit just under their caps.

    Attributes:
        containment_radius_m: The fitted payload centroid must lie within this
            distance of the tool center point, in the attach link's frame.
            *Calibration point.* ``0.12`` m is roughly a small gripper's reach
            past its own jaws.
        max_payload_extents_m: Per-axis full-extent cap in metres, ordered
            **(jaw axis, then the two free axes)** after PCA axes are sorted by
            extent ascending. *Calibration point.* ``(0.10, 0.25, 0.25)``.
            **Not derivable from the manifest today**:
            :class:`~openral_core.EndEffectorSpec` carries
            ``max_grip_force_n`` / ``max_payload_kg`` / ``workspace_radius_m``
            but **no jaw aperture or gripper span**, and SO-101's gripper
            ``position_limits`` are explicitly normalized units. Deriving the
            jaw-axis cap from the manifest needs a schema field that does not
            exist yet.
        max_payload_volume_m3: Scalar bounding-box volume backstop.
            *Calibration point.* ``0.004`` m³ is a 10 x 20 x 20 cm box — larger
            than anything a small parallel gripper carries, so it only ever
            fires on a gross mis-segmentation the per-axis caps let through.
        min_depth_valid_fraction: Fraction of masked pixels that must carry a
            finite in-range depth reading. *Calibration point.* ``0.30``.
        min_depth_m: Depth readings at or below this are invalid (sensor
            minimum range / zero-fill).
        max_depth_m: Depth readings at or above this are invalid. Anything this
            far from a wrist camera is not in the jaws.
        view_ray_inflation_m: Extra half-extent added along the camera view ray
            to compensate for the unobserved far side of the object.
            *Calibration point, and only a partial mitigation.* ``0.02`` m.
        jaw_span_m: Half-extent of the conservative fallback box, i.e. how far
            the jaws could be holding something. *Calibration point*, for the
            same missing-schema-field reason as
            :attr:`max_payload_extents_m`. ``0.05`` m.
        max_primitives: Bounded primitive count, matching
            :attr:`~openral_core.AttachedCollisionObject.primitives`' own
            ``max_length`` and the simulator producer's contract.
    """

    containment_radius_m: float = 0.12
    max_payload_extents_m: tuple[float, float, float] = (0.10, 0.25, 0.25)
    max_payload_volume_m3: float = 0.004
    min_depth_valid_fraction: float = 0.30
    min_depth_m: float = 0.02
    max_depth_m: float = 1.0
    view_ray_inflation_m: float = 0.02
    jaw_span_m: float = 0.05
    max_primitives: int = 16


@dataclass(frozen=True)
class VisionAttachmentReport:
    """Everything the gates measured, for the trace.

    Emitted alongside every attachment so an operator can see *why* a payload
    was accepted or rejected without re-running the model (CLAUDE.md §1.4:
    fallback must show up in logs).

    The report always describes **one** candidate: the selected one when a mask
    was accepted, or the closest-to-acceptable one when every candidate was
    rejected. :attr:`candidate_index` / :attr:`candidate_count` say which of how
    many, so the trace never hides that other hypotheses were considered.

    Attributes:
        accepted: Whether the selected mask cleared every geometric gate.
        rejections: Names of the gates that failed, empty when ``accepted``.
        depth_valid_fraction: Fraction of masked pixels with usable depth.
        point_count: Points in the back-projected cloud.
        extents_m: Fitted full extents, ascending, in metres.
        volume_m3: Fitted bounding-box volume.
        centroid_distance_m: Distance from the TCP to the fitted centroid.
        candidate_index: Which candidate this report describes, indexing the
            ``masks`` sequence passed to :meth:`on_grasp`. ``-1`` when no
            candidate was supplied at all.
        candidate_count: How many candidates were evaluated.
        mask_score_advisory: The selected candidate's own model score. Recorded
            only. It is **never** compared against a threshold and **never**
            used to choose between candidates — a mask covering 59.8% of a real
            frame scored 0.977.
    """

    accepted: bool
    rejections: tuple[str, ...]
    depth_valid_fraction: float
    point_count: int
    extents_m: tuple[float, float, float]
    volume_m3: float
    centroid_distance_m: float
    candidate_index: int = -1
    candidate_count: int = 0
    mask_score_advisory: float = 0.0


def backproject_masked_depth(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float64],
    intrinsics: IntrinsicsPinhole,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[NDArray[np.float64], float]:
    """Back-project the masked pixels of a depth frame into the camera frame.

    Args:
        mask: ``(H, W)`` boolean segmentation mask.
        depth_m: ``(H, W)`` metric depth, same shape as ``mask``. Zero / NaN /
            out-of-range readings are treated as missing.
        intrinsics: Pinhole intrinsics matching the frame's resolution. Rescale
            with :func:`openral_core.scale_intrinsics_to` first if they were
            calibrated at a different size.
        min_depth_m: Readings at or below this are invalid.
        max_depth_m: Readings at or above this are invalid.

    Returns:
        ``(points, valid_fraction)`` — an ``(N, 3)`` float64 array of points in
        the camera **optical** frame (REP-103: +x right, +y down, +z forward),
        and the fraction of masked pixels that carried usable depth. An empty
        mask yields ``((0, 3), 0.0)`` rather than raising: "nothing was masked"
        is a gate result, not an error.

    Raises:
        ROSConfigError: If ``mask`` and ``depth_m`` shapes disagree, or the
            intrinsics resolution does not match the frame.

    Example:
        >>> import numpy as np
        >>> from openral_core import IntrinsicsPinhole
        >>> k = IntrinsicsPinhole(width=4, height=2, fx=2.0, fy=2.0, cx=2.0, cy=1.0)
        >>> m = np.zeros((2, 4), dtype=bool)
        >>> m[1, 2] = True
        >>> d = np.full((2, 4), 0.5)
        >>> pts, frac = backproject_masked_depth(m, d, k, min_depth_m=0.1, max_depth_m=1.0)
        >>> pts.round(3).tolist()
        [[0.0, 0.0, 0.5]]
        >>> frac
        1.0
    """
    if mask.shape != depth_m.shape:
        raise ROSConfigError(
            f"backproject_masked_depth: mask shape {mask.shape} != depth shape {depth_m.shape}."
        )
    if mask.shape != (intrinsics.height, intrinsics.width):
        raise ROSConfigError(
            f"backproject_masked_depth: frame {mask.shape} does not match intrinsics "
            f"{(intrinsics.height, intrinsics.width)}; rescale with scale_intrinsics_to first."
        )
    masked_total = int(mask.sum())
    if masked_total == 0:
        return np.zeros((0, 3), dtype=np.float64), 0.0

    rows, cols = np.nonzero(mask)
    z = np.asarray(depth_m, dtype=np.float64)[rows, cols]
    usable = np.isfinite(z) & (z > min_depth_m) & (z < max_depth_m)
    valid_fraction = float(usable.sum()) / float(masked_total)
    if not usable.any():
        return np.zeros((0, 3), dtype=np.float64), valid_fraction

    z = z[usable]
    u = cols[usable].astype(np.float64) + 0.5
    v = rows[usable].astype(np.float64) + 0.5
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return np.stack((x, y, z), axis=1), valid_fraction


def _pca_basis(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Right-handed PCA basis whose columns are the cloud's principal axes.

    Columns are ordered by explained variance **ascending**, so column 0 is the
    thinnest axis — the one a parallel gripper's jaws bound.
    """
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    basis = np.asarray(vt[::-1].T, dtype=np.float64)  # ascending variance
    if float(np.linalg.det(basis)) < 0.0:
        basis[:, 0] = -basis[:, 0]
    return basis


def clustered_obb_primitives(
    points_in_link: NDArray[np.float64],
    *,
    object_id: str,
    max_primitives: int = 16,
    inflation_m: float = 0.0,
    inflation_axis_in_link: tuple[float, float, float] | None = None,
) -> tuple[list[AttachedCollisionPrimitive], NDArray[np.float64], NDArray[np.float64]]:
    """Reduce a payload point cloud to bounded oriented boxes in its object frame.

    The vision analogue of the simulator producer's ``_clustered_box_primitives``
    and it uses the same reduction: sort along the widest axis, ``np.array_split``
    into at most ``max_primitives`` clusters, and take each cluster's AABB. The
    difference is only the frame — the simulator splits geoms in the body frame,
    this splits points in the cloud's PCA frame, so the boxes are oriented to the
    object rather than to the link.

    Splitting is strictly tighter than a single box and never drops an observed
    point, so the union still covers the whole cloud.

    Args:
        points_in_link: ``(N, 3)`` cloud in the attach link's frame.
        object_id: Frame id stamped on each primitive's ``pose_in_object``.
        max_primitives: Upper bound on emitted boxes (the schema's own cap).
        inflation_m: Extra half-extent added along ``inflation_axis_in_link``,
            compensating for the object's unobserved far side.
        inflation_axis_in_link: Unit view-ray direction in the link frame. When
            ``None`` the inflation is applied isotropically instead.

    Returns:
        ``(primitives, centre_in_link, rotation_link_from_object)`` — the boxes
        posed in the object frame, plus the object frame's own origin and
        3x3 rotation in the link frame, which the caller turns into
        ``pose_in_link``.

    Raises:
        ROSConfigError: If the cloud is empty or ``max_primitives < 1``.
    """
    if max_primitives < 1:
        raise ROSConfigError(
            f"clustered_obb_primitives needs max_primitives >= 1, got {max_primitives}."
        )
    if points_in_link.size == 0:
        raise ROSConfigError("clustered_obb_primitives needs a non-empty point cloud.")

    if len(points_in_link) >= _MIN_CLOUD_POINTS:
        rotation = _pca_basis(points_in_link)
    else:
        # Too few points for a meaningful covariance — stay axis-aligned rather
        # than orienting the box off noise.
        rotation = np.eye(3, dtype=np.float64)

    local = (points_in_link - points_in_link.mean(axis=0)) @ rotation
    lower, upper = local.min(axis=0), local.max(axis=0)
    origin_local = 0.5 * (lower + upper)
    centre_in_link = points_in_link.mean(axis=0) + rotation @ origin_local
    local -= origin_local

    inflation: NDArray[np.float64] = np.full(3, float(inflation_m), dtype=np.float64)
    if inflation_axis_in_link is not None and inflation_m > 0.0:
        axis = np.asarray(inflation_axis_in_link, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm > 0.0:
            # Project the view ray into the object frame and inflate each axis
            # by its share of it, so the growth follows the unobserved direction.
            inflation = float(inflation_m) * np.abs(rotation.T @ (axis / norm))

    split_axis = int(np.argmax(np.ptp(local, axis=0)))
    order = np.argsort(local[:, split_axis], kind="stable")
    primitives: list[AttachedCollisionPrimitive] = []
    for cluster in np.array_split(order, min(max_primitives, len(order))):
        if cluster.size == 0:
            continue
        chunk = local[cluster]
        lo, hi = chunk.min(axis=0), chunk.max(axis=0)
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) + inflation + _BOX_EPSILON_M
        primitives.append(
            AttachedCollisionPrimitive(
                shape=BoxShape(half_extents_m=(float(half[0]), float(half[1]), float(half[2]))),
                pose_in_object=Pose6D(
                    xyz=(float(centre[0]), float(centre[1]), float(centre[2])),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id=object_id,
                ),
            )
        )
    return primitives, centre_in_link, rotation


def jaw_span_primitive(*, object_id: str, jaw_span_m: float) -> AttachedCollisionPrimitive:
    """Build the conservative fallback box used when a mask fails the gates.

    A cube of half-extent ``jaw_span_m`` centred on the tool center point: "we
    know *something* is held, we do not know its shape". Deliberately crude —
    the point is that a rejected mask still leaves geometry in front of the
    collision checker rather than an invisible payload.

    Args:
        object_id: Frame id stamped on the primitive's ``pose_in_object``.
        jaw_span_m: Half-extent in metres.

    Raises:
        ROSConfigError: If ``jaw_span_m`` is not positive.

    Example:
        >>> jaw_span_primitive(object_id="payload", jaw_span_m=0.05).shape.half_extents_m
        (0.05, 0.05, 0.05)
    """
    if jaw_span_m <= 0.0:
        raise ROSConfigError(f"jaw_span_primitive needs jaw_span_m > 0, got {jaw_span_m}.")
    return AttachedCollisionPrimitive(
        shape=BoxShape(half_extents_m=(jaw_span_m, jaw_span_m, jaw_span_m)),
        pose_in_object=Pose6D(
            xyz=(0.0, 0.0, 0.0),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id=object_id,
        ),
    )


def _quat_xyzw_from_rotation(rotation: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Adapt the public ``wxyz`` matrix→quaternion helper to the schema's ``xyzw``."""
    w, x, y, z = rotation_to_quat_wxyz(rotation)
    return (x, y, z, w)


class VisionAttachmentEvidenceProducer:
    """Turn a segmenter mask plus wrist depth into a gated attachment.

    Mirrors :class:`~openral_hal._sim_attachment_evidence.SimAttachmentEvidenceTracker`'s
    role — resolve the attach link and touch links from the manifest once, then
    emit complete attachment sets on grasp / release — but sources its geometry
    from perception rather than from MuJoCo ground truth.

    Unlike the simulator tracker this producer is **event-driven, not ticked**:
    it has no per-frame `update`. The caller decides that a grasp happened and
    calls :meth:`on_grasp` once; per-frame carry masking stays geometric
    containment against the primitive fitted here.

    Args:
        description: The robot manifest, read for the gripper's parent link and
            the finger links that are allowed to touch the payload.
        config: Gate thresholds. Every one is a calibration point — see
            :class:`VisionGateConfig`.

    Raises:
        ROSConfigError: If the manifest declares no gripper-role joints, or its
            gripper joints do not share exactly one parent link.
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        config: VisionGateConfig | None = None,
    ) -> None:
        """Resolve the attach link and touch links from the manifest."""
        self._config = config or VisionGateConfig()
        gripper_joints = [joint for joint in description.joints if joint.role == "gripper"]
        if not gripper_joints:
            raise ROSConfigError("Vision attachment evidence requires gripper-role joints.")
        attach_links = {joint.parent_link for joint in gripper_joints}
        if len(attach_links) != 1:
            raise ROSConfigError(
                f"Vision attachment evidence needs one gripper parent link, got {attach_links}."
            )
        self._attach_link = next(iter(attach_links))
        self._touch_links = [joint.child_link for joint in gripper_joints]
        self._attached: bool = False

    @property
    def attach_link(self) -> str:
        """Robot link that owns the attached payload's pose."""
        return self._attach_link

    @property
    def config(self) -> VisionGateConfig:
        """The gate thresholds in force."""
        return self._config

    def on_release(self) -> list[AttachedCollisionObject]:
        """Report the detach event: an empty attachment set.

        Mirrors the simulator producer's release path — an empty list is a
        *complete* attachment set meaning "nothing is held", not "no update".
        """
        self._attached = False
        return []

    def _evaluate_candidate(
        self,
        *,
        mask: NDArray[np.bool_],
        depth_m: NDArray[np.float64],
        intrinsics: IntrinsicsPinhole,
        t_link_from_cam: NDArray[np.float64],
        tcp: NDArray[np.float64],
        index: int,
        count: int,
        score_advisory: float,
    ) -> tuple[VisionAttachmentReport, NDArray[np.float64]]:
        """Gate one candidate mask, returning its report and its link-frame cloud."""
        cfg = self._config
        points_cam, depth_valid_fraction = backproject_masked_depth(
            mask,
            depth_m,
            intrinsics,
            min_depth_m=cfg.min_depth_m,
            max_depth_m=cfg.max_depth_m,
        )
        rejections: list[str] = []
        # "the segmenter masked nothing" and "the depth sensor saw nothing under
        # the mask" are different diagnoses and the trace names whichever it was.
        if not bool(mask.any()):
            rejections.append("empty_mask")
        elif depth_valid_fraction < cfg.min_depth_valid_fraction:
            rejections.append("depth_validity")

        if points_cam.shape[0] == 0:
            empty = VisionAttachmentReport(
                accepted=False,
                rejections=tuple(rejections or ["no_valid_depth"]),
                depth_valid_fraction=depth_valid_fraction,
                point_count=0,
                extents_m=(0.0, 0.0, 0.0),
                volume_m3=0.0,
                centroid_distance_m=float("inf"),
                candidate_index=index,
                candidate_count=count,
                mask_score_advisory=score_advisory,
            )
            return empty, np.zeros((0, 3), dtype=np.float64)

        rotation_link_cam = t_link_from_cam[:3, :3]
        points_link = points_cam @ rotation_link_cam.T + t_link_from_cam[:3, 3]

        basis = (
            _pca_basis(points_link)
            if len(points_link) >= _MIN_CLOUD_POINTS
            else np.eye(3, dtype=np.float64)
        )
        local = (points_link - points_link.mean(axis=0)) @ basis
        extents = np.ptp(local, axis=0)
        volume = float(np.prod(extents))
        centroid = points_link.mean(axis=0)
        centroid_distance = float(np.linalg.norm(centroid - tcp))

        if any(
            float(extent) > cap
            for extent, cap in zip(extents, cfg.max_payload_extents_m, strict=True)
        ):
            rejections.append("payload_extent")
        if volume > cfg.max_payload_volume_m3:
            rejections.append("payload_volume")
        if centroid_distance > cfg.containment_radius_m:
            rejections.append("jaw_containment")

        report = VisionAttachmentReport(
            accepted=not rejections,
            rejections=tuple(rejections),
            depth_valid_fraction=depth_valid_fraction,
            point_count=int(points_link.shape[0]),
            extents_m=(float(extents[0]), float(extents[1]), float(extents[2])),
            volume_m3=volume,
            centroid_distance_m=centroid_distance,
            candidate_index=index,
            candidate_count=count,
            mask_score_advisory=score_advisory,
        )
        return report, points_link

    def on_grasp(
        self,
        *,
        masks: Sequence[NDArray[np.bool_]],
        depth_m: NDArray[np.float64],
        intrinsics: IntrinsicsPinhole,
        t_link_from_cam: NDArray[np.float64],
        tcp_in_link: tuple[float, float, float],
        object_id: str,
        stamp_ns: int,
        mask_scores_advisory: Sequence[float] = (),
    ) -> tuple[AttachedCollisionObject, VisionAttachmentReport]:
        """Fit and gate one payload from a wrist RGB-D frame's candidate masks.

        **Selection happens here, and it happens on geometry.** The segmenter
        returns SAM 2's nested subpart / part / whole hypotheses because it has
        no depth and therefore cannot choose between them; this producer does.
        Every candidate is back-projected against the same depth frame and run
        through the same gates, then:

        * among candidates that clear **every** gate, the one with the largest
          fitted volume wins — over-approximating a payload is the conservative
          error for collision checking, so the most inclusive geometrically
          plausible hypothesis is the safe pick;
        * if none clear, the report describes the closest-to-acceptable
          candidate (fewest failed gates, ties broken toward the smaller volume)
          and the attachment degrades to the fallback box.

        The per-candidate model scores are carried into the report and are
        **never** part of that choice — a mis-aimed prompt produced this model's
        top score of 0.977 on a mask covering 59.8% of the frame.

        **Always returns an attachment.** A rejected grasp yields the
        conservative jaw-span box stamped
        :attr:`~openral_core.AttachmentEvidenceKind.GRIPPER_FORCE` at low
        confidence — never ``None``, never a silent skip. The report names every
        gate that failed, so the fallback is visible in the trace
        (CLAUDE.md §1.4).

        Args:
            masks: Candidate ``(H, W)`` boolean masks from the segmenter, in the
                order the ``SegmentInView`` reply carried them (area ascending).
                A ``multimask: false`` manifest simply supplies one.
            depth_m: ``(H, W)`` metric depth from the **same** frame.
            intrinsics: Pinhole intrinsics at the frame's resolution.
            t_link_from_cam: ``(4, 4)`` homogeneous transform mapping camera
                optical-frame points into the attach link's frame. Supplied by
                the caller from TF2 — this module never composes frames itself
                (CLAUDE.md §2).
            tcp_in_link: Tool center point in the attach link's frame.
            object_id: Stable identity for the payload; also the primitives'
                frame id.
            stamp_ns: The attach instant, owned by the caller.
            mask_scores_advisory: Per-candidate model scores, parallel to
                ``masks``. Recorded in the report and **never** gated on or used
                to select. May be empty, in which case scores default to ``0.0``.

        Returns:
            ``(attachment, report)``.

        Raises:
            ROSConfigError: If ``t_link_from_cam`` is not ``(4, 4)``, the
                mask / depth / intrinsics shapes disagree, or
                ``mask_scores_advisory`` is non-empty and a different length
                from ``masks``.
        """
        if t_link_from_cam.shape != (4, 4):
            raise ROSConfigError(
                f"on_grasp: t_link_from_cam must be (4, 4), got {t_link_from_cam.shape}."
            )
        if mask_scores_advisory and len(mask_scores_advisory) != len(masks):
            raise ROSConfigError(
                f"on_grasp: mask_scores_advisory has {len(mask_scores_advisory)} entries "
                f"for {len(masks)} masks; the arrays are parallel."
            )
        cfg = self._config
        tcp = np.asarray(tcp_in_link, dtype=np.float64)
        fallback = self._fallback_attachment(
            object_id=object_id, tcp_in_link=tcp_in_link, stamp_ns=stamp_ns
        )
        if not masks:
            # The segmenter answered with nothing at all (`ok=False`, empty
            # `masks`). Still fail closed with geometry rather than a skip.
            self._attached = True
            return fallback, VisionAttachmentReport(
                accepted=False,
                rejections=("no_candidates",),
                depth_valid_fraction=0.0,
                point_count=0,
                extents_m=(0.0, 0.0, 0.0),
                volume_m3=0.0,
                centroid_distance_m=float("inf"),
                candidate_count=0,
            )

        scores = list(mask_scores_advisory) or [0.0] * len(masks)
        evaluated = [
            self._evaluate_candidate(
                mask=mask,
                depth_m=depth_m,
                intrinsics=intrinsics,
                t_link_from_cam=t_link_from_cam,
                tcp=tcp,
                index=index,
                count=len(masks),
                score_advisory=float(score),
            )
            for index, (mask, score) in enumerate(zip(masks, scores, strict=True))
        ]

        accepted = [(report, cloud) for report, cloud in evaluated if report.accepted]
        if not accepted:
            # Report the candidate that came closest: fewest failed gates first,
            # then the smaller volume (nearer to a plausible payload).
            report, _ = min(
                evaluated, key=lambda item: (len(item[0].rejections), item[0].volume_m3)
            )
            self._attached = True
            return fallback, report

        # Most inclusive plausible hypothesis wins — see the docstring.
        report, points_link = max(accepted, key=lambda item: item[0].volume_m3)

        # The view ray in link coordinates: the camera's optical +z, rotated.
        view_ray = t_link_from_cam[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        primitives, centre_in_link, rotation = clustered_obb_primitives(
            points_link,
            object_id=object_id,
            max_primitives=cfg.max_primitives,
            inflation_m=cfg.view_ray_inflation_m,
            inflation_axis_in_link=(float(view_ray[0]), float(view_ray[1]), float(view_ray[2])),
        )
        self._attached = True
        attachment = AttachedCollisionObject(
            object_id=object_id,
            attach_link=self._attach_link,
            touch_links=self._touch_links,
            primitives=primitives,
            pose_in_link=Pose6D(
                xyz=(
                    float(centre_in_link[0]),
                    float(centre_in_link[1]),
                    float(centre_in_link[2]),
                ),
                quat_xyzw=_quat_xyzw_from_rotation(rotation),
                frame_id=self._attach_link,
            ),
            # mass / CoG / inertia stay None: vision gives geometry only. The
            # simulator producer fills them from MuJoCo; there is no honest
            # real-hardware equivalent.
            confidence=_VISION_CONFIDENCE,
            evidence_kind=AttachmentEvidenceKind.VISION_SEGMENTATION,
            # Which candidate of how many, so the trace can be replayed against
            # the same SegmentInView reply.
            evidence_ref=(
                f"segmenter_mask:{object_id}@{stamp_ns}"
                f"#{report.candidate_index}/{report.candidate_count}"
            ),
            stamp_ns=stamp_ns,
        )
        return attachment, report

    def _fallback_attachment(
        self,
        *,
        object_id: str,
        tcp_in_link: tuple[float, float, float],
        stamp_ns: int,
    ) -> AttachedCollisionObject:
        """Conservative jaw-span box for a mask that failed the geometric gates.

        Never a skip: something is in the jaws, we simply do not know its shape,
        so the collision checker gets a crude box rather than an invisible
        payload.

        Pure: :meth:`on_grasp` builds this before it knows the verdict (the
        rejection path needs it for every candidate outcome), so the attached
        flag is flipped by the caller at its return points, not here.
        """
        return AttachedCollisionObject(
            object_id=object_id,
            attach_link=self._attach_link,
            touch_links=self._touch_links,
            primitives=[
                jaw_span_primitive(object_id=object_id, jaw_span_m=self._config.jaw_span_m)
            ],
            pose_in_link=Pose6D(
                xyz=tcp_in_link,
                quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id=self._attach_link,
            ),
            confidence=_FALLBACK_CONFIDENCE,
            evidence_kind=AttachmentEvidenceKind.GRIPPER_FORCE,
            evidence_ref=f"vision_gate_rejected:{object_id}@{stamp_ns}",
            stamp_ns=stamp_ns,
        )
