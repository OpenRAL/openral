"""Real-hardware place producer leg — the support measured under the carried payload.

The real sibling of the simulator's place producer (``_sim_attachment_evidence``
``set_place_declaration`` / ``place_declaration`` / ``_place_witness``), owned by
``VisionAttachmentBridge`` so ``/openral/attachment_state`` keeps one authority.
Real pick-and-place design ``docs/reference/real-pick-place-design.md`` §2.3.

**Where to place is the policy's job.** Nothing about the cell is surveyed, and no
one has to name a place target: while a goal is running and a payload is held, this leg
looks straight down under it in the live voxel map and **measures** the surface it would
come to rest on, and attaches that region to the goal's place declaration. The
semantics are **drafted, not approved** (ADR-0097 amendment: a region the producer
measured under the carried payload needs no dispatch-named target; ADR-0092 D6
amendment: a proximity witness to a map-measured plane; hazard "a measured surface is
not a support"). Default off (``VisionAttachmentConfig.place_target_enabled``), and
every failure fails closed — no region, the kernel's normal margins.

1. **Column.** At ``place_target_rate_hz``, only under a live dispatch declaration,
   against the newest ``/openral/world_voxels`` (received, or its ``source_stamp`` data,
   older than ``grid_max_age_s`` = unusable; an unset ``source_stamp`` too): the held,
   loaded payload's measured primitives give its footprint (axis-aligned in the
   lattice axes) and height. The search column is that footprint grown by one voxel +
   ``place_target_extrinsic_error_m`` on every side, from the payload's bottom down
   ``place_target_search_depth_m``. Cells within one voxel of a published payload are
   ignored (the octomap bridge clears exactly those).
2. **Surface** (``measure_under_payload``). Scanning the column top-down, the first
   layer holding any occupied cell is what the payload would land on. It is a support
   only when **every** column cell of that layer is occupied and the layers above it
   are free up to the payload's height + one voxel + the extrinsic bound; anything
   else (an edge, clutter, a hole) is a typed refusal. The region is that patch as a
   one-voxel **slab** — the support cells only — in the grid frame, no ``geometry``.
3. **Latch and freeze.** The patch is latched with the payload's identity. While the
   payload's centre stays over it, later ticks re-verify the latched patch instead of
   re-measuring: something new in its free volume retracts it at once; support cells
   missing (the payload and hand occlude the board from the head camera, payload
   clearing removes the cells under it) or a missing / stale grid is a **lost view**,
   and the latched region is held for at most ``place_target_freeze_s`` (default and
   ceiling twice ``grid_max_age_s`` — the kernel's ``place_region_max_age_s``) past
   the grid data it was last verified on (``freeze_ttl``). The region's ``stamp_ns``
   is that grid's ``source_stamp`` (the world data's capture time, the kernel's own voxel
   data-age clock — never ``header.stamp``, which a bridge republishing a stalled octree
   keeps fresh), re-stamped on every successful re-verification, so the published
   region is never older than the kernel accepts. The payload moving off the patch
   re-measures; a different payload drops the latch.
4. **Declaration.** The kernel only applies a region carried in a ``PlaceDeclaration``,
   and the producer never declares on its own: the region rides dispatch's live
   ``/openral/place_declaration`` — the runner's goal-scope one (``place_approach_enabled``,
   ``target_id = surface``, the mirror of the grasp side's approach declaration) or a
   reasoner/scene-named surface — scoped to the measured payload when it names no object.
   No goal, no allowance. Its ``search_box``, when set, is a hint: a patch whose centre
   lies outside it is refused (``outside_hint``). A retraction (goal end, cancel, E-stop)
   or expiry drops the patch and the witness at once and nothing re-arms until a new
   declaration; a new declaration drops them too and re-measures under it.
5. **Witness substitute**, once per declaration (the simulator's hysteresis): a latched
   region, the payload still attached on a leg whose trigger reads loaded, its lowest
   primitive point within ``max(1 voxel, extrinsic_error_m)`` of the latched plane and
   its centre over the patch → a ``SupportContactWitness`` on that object,
   ``support_id`` = the declaration's ``target_id``, ``evidence_kind =
   MAP_SUPPORT_PROXIMITY``. **It is proximity to a map-measured plane, not sensed
   contact, and the measured surface is not proven to bear load**; every log line and
   ``evidence_ref`` says so. A payload released on the patch keeps it on its frozen
   release record until the window closes. It dies with the region — contradicting
   evidence, the freeze TTL (the kernel's region age bound), a dispatch retraction or
   a payload change — so the place allowance never outlives a measurement the kernel
   would accept: a set-down and release must finish within the freeze.

Every threshold is a **calibration point** (CLAUDE.md §1.2), none measured on the cell.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    CapsuleShape,
    PlaceDeclaration,
    PlaceRegion,
    Pose6D,
    SphereShape,
    SupportContactWitness,
)
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import homogeneous_from_quat_xyz

from openral_hal._grasp_target import VoxelLattice
from openral_hal._grasp_target_leg import lattice_from_msg

if TYPE_CHECKING:  # pragma: no cover — typing only
    from openral_hal.vision_attachment_bridge import (
        VisionAttachmentBridge,
        VisionAttachmentConfig,
        _GripperLeg,
    )

__all__ = [
    "PlacePatch",
    "PlaceRefusal",
    "PlaceTargetLeg",
    "PlaceTargetTracker",
    "measure_under_payload",
    "payload_witness",
    "verify_patch",
]

#: The kernel's ADR-0092 D6 witness caps (``support_witness_max_patch_radius_m`` /
#: ``support_witness_max_penetration_m``); a witness past either fails the kernel's
#: whole attachment message closed, so the producer never emits one.
_MAX_PATCH_RADIUS_M = 0.5
_MAX_PENETRATION_M = 0.01

#: A proximity witness is never a full-confidence contact claim. Not a calibrated
#: probability; the kernel does not read it.
_WITNESS_CONFIDENCE = 0.5

#: Fastest witness evaluation from the joint-state hook, seconds (20 Hz): a few tf2
#: lookups, kept off a 750 Hz joint-state stream.
_WITNESS_PERIOD_S = 0.05

#: Tilt tolerance: the lattice must be yaw-only (z up).
_GRAVITY_TOL = 1e-6

#: A primitive posed in the grid frame.
Posed = Sequence[tuple[AttachedCollisionPrimitive, NDArray[np.float64]]]


class PlaceRefusal(StrEnum):
    """Why the leg holds no region (logged once per transition, never swallowed)."""

    NO_PAYLOAD = "no_payload"
    PAYLOAD_CHANGED = "payload_changed"
    NO_GRID = "no_grid"
    GRID_STALE = "grid_stale"
    TILTED = "tilted"
    NO_SURFACE = "no_surface"
    PARTIAL_SUPPORT = "partial_support"
    NO_HEADROOM = "no_headroom"
    OUTSIDE_HINT = "outside_hint"
    FRAME_MISMATCH = "frame_mismatch"
    FREE_VOLUME_OCCUPIED = "free_volume_occupied"
    SUPPORT_OCCLUDED = "support_occluded"
    FREEZE_TTL = "freeze_ttl"
    NO_DECLARATION = "no_declaration"
    REDECLARED = "redeclared"


#: Refusals that only mean "the view is lost": a latched region survives them under
#: its freeze TTL. Every other refusal retracts at once.
_LOST_VIEW = frozenset(
    {PlaceRefusal.NO_GRID, PlaceRefusal.GRID_STALE, PlaceRefusal.SUPPORT_OCCLUDED}
)


@dataclass(frozen=True)
class PlacePatch:
    """A measured place patch on one lattice layer.

    Attributes:
        layer: Lattice z-layer of the support cells.
        i0: First lattice ``i`` of the patch.
        j0: First lattice ``j`` of the patch.
        ni: Patch width in cells along lattice ``i``.
        nj: Patch width in cells along lattice ``j``.
        free_layers: Layers above ``layer`` that must stay free.
        plane_z: The support's top face in the grid frame (z up).
        region: The patch's one-voxel support slab as a ``PlaceRegion``.
    """

    layer: int
    i0: int
    j0: int
    ni: int
    nj: int
    free_layers: int
    plane_z: float
    region: PlaceRegion


# ── payload geometry ─────────────────────────────────────────────────────────


def _shape_half_extents(primitive: AttachedCollisionPrimitive) -> NDArray[np.float64]:
    """The primitive's local AABB half-extents (a capsule's segment runs along +Z)."""
    shape = primitive.shape
    if isinstance(shape, BoxShape):
        return np.asarray(shape.half_extents_m, dtype=np.float64)
    if isinstance(shape, SphereShape):
        return np.full(3, shape.radius_m)
    assert isinstance(shape, CapsuleShape)
    r = shape.radius_m
    return np.array([r, r, r + 0.5 * shape.length_m])


def _support_along(
    primitive: AttachedCollisionPrimitive, rot: NDArray[np.float64], n: NDArray[np.float64]
) -> float:
    """Exact support distance of the primitive from its centre along unit ``n``."""
    shape = primitive.shape
    if isinstance(shape, BoxShape):
        return float(np.abs(rot.T @ n) @ np.asarray(shape.half_extents_m))
    if isinstance(shape, SphereShape):
        return float(shape.radius_m)
    assert isinstance(shape, CapsuleShape)
    return float(shape.radius_m + 0.5 * shape.length_m * abs(float(rot[:, 2] @ n)))


def _inside_any(points: NDArray[np.float64], posed: Posed, *, pad: float) -> NDArray[np.bool_]:
    """Points inside any primitive's local AABB grown by ``pad`` (a conservative hull)."""
    hit = np.zeros(len(points), dtype=bool)
    for prim, t in posed:
        local = (points - t[:3, 3]) @ t[:3, :3]
        hit |= np.all(np.abs(local) <= _shape_half_extents(prim) + pad, axis=1)
    return hit


# ── measurement ──────────────────────────────────────────────────────────────


def _z_up(grid: VoxelLattice) -> bool:
    return abs(float(grid.rotation()[2, 2]) - 1.0) <= _GRAVITY_TOL


def _lattice_ijk(grid: VoxelLattice, centres: NDArray[np.float64]) -> NDArray[np.int64]:
    local = (centres - np.asarray(grid.origin)) @ grid.rotation()
    return np.asarray(np.floor(local / grid.resolution), dtype=np.int64)


def _payload_extent_local(
    grid: VoxelLattice, posed: Posed
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(lo, hi)`` of the payload's AABB in lattice coordinates (an over-approximation)."""
    rot = grid.rotation()
    origin = np.asarray(grid.origin)
    los, his = [], []
    for prim, t in posed:
        half = np.abs(rot.T @ t[:3, :3]) @ _shape_half_extents(prim)
        c = rot.T @ (t[:3, 3] - origin)
        los.append(c - half)
        his.append(c + half)
    return np.min(los, axis=0), np.max(his, axis=0)


def _occupied_cells(grid: VoxelLattice, exclude: Posed) -> NDArray[np.int64]:
    """Lattice indices of every occupied cell farther than one voxel from ``exclude``."""
    centres = grid.occupied_centers()
    if exclude:
        centres = centres[~_inside_any(centres, exclude, pad=grid.resolution)]
    return _lattice_ijk(grid, centres)


def _patch_region(
    grid: VoxelLattice, layer: int, i0: int, j0: int, ni: int, nj: int, evidence_ref: str
) -> PlaceRegion:
    """The patch's one-voxel slab (its support cells) in the grid frame."""
    res = grid.resolution
    centre_local = np.array([(i0 + ni / 2.0) * res, (j0 + nj / 2.0) * res, (layer + 0.5) * res])
    centre = grid.rotation() @ centre_local + np.asarray(grid.origin)
    return PlaceRegion(
        frame_id=grid.frame_id,
        pose=Pose6D(
            xyz=(float(centre[0]), float(centre[1]), float(centre[2])),
            quat_xyzw=grid.orientation_xyzw,
            frame_id=grid.frame_id,
        ),
        half_extents=(ni * res / 2.0, nj * res / 2.0, res / 2.0),
        evidence_ref=evidence_ref,
    )


def measure_under_payload(
    grid: VoxelLattice,
    payload: Posed,
    *,
    extrinsic_error_m: float,
    search_depth_m: float,
    exclude: Posed = (),
    evidence_ref: str = "",
) -> PlacePatch | tuple[PlaceRefusal, str]:
    """Measure the surface directly under the carried ``payload`` in one voxel grid.

    Args:
        grid: The published lattice; must be z-up (yaw-only).
        payload: The carried payload's primitives posed in ``grid.frame_id``; they give
            the column's footprint and the free height.
        extrinsic_error_m: The depth extrinsic's accuracy bound, added with one voxel to
            the footprint on every side and to the free height. *Calibration point.*
        search_depth_m: How far below the payload's bottom the column reaches.
            *Calibration point.*
        exclude: Every published payload posed in the grid frame (``payload`` included
            or not); cells within one voxel of any are ignored.
        evidence_ref: Prefix for the region's ``evidence_ref``.

    Returns:
        The ``PlacePatch``, or ``(refusal, detail)``.

    Example:
        >>> import numpy as np
        >>> from openral_core import AttachedCollisionPrimitive, BoxShape, Pose6D
        >>> occ = np.zeros(10 * 10 * 10, dtype=np.uint8)
        >>> occ[300:400] = 1  # a full 10 x 10 table top on layer k=3
        >>> g = VoxelLattice("b", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.05, (10, 10, 10), occ)
        >>> cube = AttachedCollisionPrimitive(
        ...     shape=BoxShape(half_extents_m=(0.04, 0.04, 0.04)),
        ...     pose_in_object=Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id="p"),
        ... )
        >>> t = np.eye(4)
        >>> t[:3, 3] = (0.25, 0.25, 0.3)
        >>> patch = measure_under_payload(
        ...     g, [(cube, t)], extrinsic_error_m=0.01, search_depth_m=0.2
        ... )
        >>> round(patch.plane_z, 3), (patch.ni, patch.nj), patch.free_layers
        (0.2, (4, 4), 3)
    """
    if not _z_up(grid):
        return PlaceRefusal.TILTED, "the lattice must be yaw-only (z up)"
    if not payload:
        return PlaceRefusal.NO_PAYLOAD, "no carried payload to measure under"
    res = grid.resolution
    lo, hi = _payload_extent_local(grid, payload)
    grow = res + extrinsic_error_m
    i0, j0 = math.floor((lo[0] - grow) / res + 1e-9), math.floor((lo[1] - grow) / res + 1e-9)
    i1, j1 = math.ceil((hi[0] + grow) / res - 1e-9), math.ceil((hi[1] + grow) / res - 1e-9)
    ni, nj = i1 - i0, j1 - j0
    free_layers = math.ceil((hi[2] - lo[2] + grow) / res - 1e-9)
    k_top = math.floor(lo[2] / res + 1e-9) - 1  # the highest layer wholly below the payload
    k_bottom = math.floor((lo[2] - search_depth_m) / res)
    ijk = _occupied_cells(grid, exclude or payload)
    column = (ijk[:, 0] >= i0) & (ijk[:, 0] < i1) & (ijk[:, 1] >= j0) & (ijk[:, 1] < j1)
    below = column & (ijk[:, 2] <= k_top) & (ijk[:, 2] >= k_bottom)
    if not below.any():
        return (
            PlaceRefusal.NO_SURFACE,
            f"nothing occupied within {search_depth_m:.2f} m under the payload",
        )
    layer = int(ijk[below, 2].max())
    support = {(int(i), int(j)) for i, j, _ in ijk[below & (ijk[:, 2] == layer)].tolist()}
    if len(support) < ni * nj:
        return (
            PlaceRefusal.PARTIAL_SUPPORT,
            f"{len(support)}/{ni * nj} cells of the {ni}x{nj} footprint occupied on the first "
            f"surface under the payload (an edge, clutter or a hole)",
        )
    above = column & (ijk[:, 2] > layer) & (ijk[:, 2] <= layer + free_layers)
    if above.any():
        return (
            PlaceRefusal.NO_HEADROOM,
            f"{int(above.sum())} occupied cells within {free_layers} cells above the surface",
        )
    plane_z = float(grid.origin[2] + (layer + 1) * res)
    ref = f"{evidence_ref}map_measured_support:plane_z={plane_z:.3f};patch={ni}x{nj}@layer{layer}"
    region = _patch_region(grid, layer, i0, j0, ni, nj, ref)
    return PlacePatch(layer, i0, j0, ni, nj, free_layers, plane_z, region)


def verify_patch(
    grid: VoxelLattice, patch: PlacePatch, payload: Posed = ()
) -> tuple[PlaceRefusal, str] | None:
    """Re-check a latched patch on a newer grid; ``None`` = it still holds.

    Anything occupied in the free volume (outside one voxel of the payload) is a
    contradiction; support cells missing are a lost view (the payload and hand
    occlude the board, payload clearing removes the cells under it).
    """
    if grid.frame_id != patch.region.frame_id:
        return PlaceRefusal.FRAME_MISMATCH, f"grid now in {grid.frame_id!r}"
    if not _z_up(grid):
        return PlaceRefusal.TILTED, "the lattice is no longer yaw-only"
    ijk = _occupied_cells(grid, payload)
    in_cols = (
        (ijk[:, 0] >= patch.i0)
        & (ijk[:, 0] < patch.i0 + patch.ni)
        & (ijk[:, 1] >= patch.j0)
        & (ijk[:, 1] < patch.j0 + patch.nj)
    )
    above = in_cols & (ijk[:, 2] > patch.layer) & (ijk[:, 2] <= patch.layer + patch.free_layers)
    if above.any():
        return (
            PlaceRefusal.FREE_VOLUME_OCCUPIED,
            f"{int(above.sum())} occupied cells above the latched patch",
        )
    support = in_cols & (ijk[:, 2] == patch.layer)
    have = len({(int(i), int(j)) for i, j, _ in ijk[support].tolist()})
    if have < patch.ni * patch.nj:
        return (
            PlaceRefusal.SUPPORT_OCCLUDED,
            f"{have}/{patch.ni * patch.nj} support cells of the latched patch in view",
        )
    return None


def _over_patch(posed: Posed, patch: PlacePatch) -> bool:
    """Whether the payload's centre (mean primitive origin) lies over the patch."""
    centre = np.mean([t[:3, 3] for _, t in posed], axis=0)
    region = patch.region
    t_region = homogeneous_from_quat_xyz(region.pose.xyz, region.pose.quat_xyzw)
    uv = (centre - t_region[:3, 3]) @ t_region[:3, :2]
    return bool(np.all(np.abs(uv) <= np.asarray(region.half_extents[:2])))


def payload_witness(
    obj: AttachedCollisionObject,
    t_base_link: NDArray[np.float64],
    patch: PlacePatch,
    *,
    support_id: str,
    resolution: float,
    extrinsic_error_m: float,
    stamp_ns: int,
) -> tuple[SupportContactWitness | None, str]:
    """The proximity witness for ``obj`` resting on the latched patch, or why not.

    ``t_base_link`` poses ``obj.attach_link`` in the grid frame (z up). Not sensed
    contact: the payload's lowest primitive point is within ``max(resolution,
    extrinsic_error_m)`` of the map-measured plane and its centre is over the patch.

    Returns:
        ``(witness, detail)``; ``witness`` is ``None`` when the payload is not there.
    """
    from openral_hal.vision_attachment_bridge import primitive_poses  # circular at import

    n = np.array([0.0, 0.0, 1.0])
    posed = list(zip(obj.primitives, primitive_poses(obj, t_base_link), strict=True))
    lowest = min(
        float(t[2, 3] - patch.plane_z) - _support_along(prim, t[:3, :3], n) for prim, t in posed
    )
    tol = max(resolution, extrinsic_error_m)
    detail = f"lowest point {lowest * 1e3:+.1f} mm from the measured plane (tol {tol * 1e3:.1f} mm)"
    if abs(lowest) > tol:
        return None, detail
    if not _over_patch(posed, patch):
        return None, f"payload centre is off the measured patch ({detail})"
    centre = np.mean([t[:3, 3] for _, t in posed], axis=0)
    lateral = [
        float(np.linalg.norm((t[:3, 3] - centre)[:2]))
        + float(np.linalg.norm(_shape_half_extents(prim)))
        for prim, t in posed
    ]
    p = obj.pose_in_link
    t_base_obj = t_base_link @ homogeneous_from_quat_xyz(p.xyz, p.quat_xyzw)
    r_obj, t_obj = t_base_obj[:3, :3], t_base_obj[:3, 3]
    contact = np.array([centre[0], centre[1], patch.plane_z])
    normal_obj = r_obj.T @ n
    witness = SupportContactWitness(
        support_id=support_id,
        contact_point_in_object=tuple(float(v) for v in r_obj.T @ (contact - t_obj)),
        contact_normal_in_object=tuple(float(v) for v in normal_obj / np.linalg.norm(normal_obj)),
        patch_radius_m=min(max(lateral), _MAX_PATCH_RADIUS_M),
        max_penetration_m=min(extrinsic_error_m, _MAX_PENETRATION_M),
        confidence=_WITNESS_CONFIDENCE,
        evidence_kind=AttachmentEvidenceKind.MAP_SUPPORT_PROXIMITY,
        evidence_ref=(
            f"map_support_proximity:{support_id}: proximity to a map-measured plane, not "
            f"sensed contact, not a proven support; {detail}"
        ),
        stamp_ns=stamp_ns,
    )
    return witness, detail


# ── state machine ────────────────────────────────────────────────────────────


class PlaceTargetTracker:
    """Optional dispatch declaration, latched patch, freeze TTL and witness — pure.

    Clock-free: every method takes the caller's ``now_ns`` (the node's ROS clock, the
    declaration's domain). ``log`` receives one line per transition.

    Args:
        freeze_s: How long past its last verification a latched region survives a
            lost view.
        extrinsic_error_m: The depth extrinsic's accuracy bound (witness tolerance).
        log: Sink for transition lines.

    Example:
        >>> from openral_core import PlaceDeclaration
        >>> lines: list[str] = []
        >>> t = PlaceTargetTracker(freeze_s=2.0, extrinsic_error_m=0.015, log=lines.append)
        >>> t.envelope(now_ns=1) is None  # nothing measured, nothing declared
        True
        >>> t.on_declaration(PlaceDeclaration(target_id="surface:x", timeout_s=9.0, stamp_ns=0))
        >>> t.envelope(now_ns=1).region is None, lines[-1].split()[:2]
        (True, ['place_target', 'declared'])
    """

    def __init__(
        self, *, freeze_s: float, extrinsic_error_m: float, log: Callable[[str], None]
    ) -> None:
        """Start with nothing declared or measured."""
        self._freeze_ns = int(freeze_s * 1e9)
        self._extrinsic = extrinsic_error_m
        self._log = log
        self._declaration: PlaceDeclaration | None = None
        self._patch: PlacePatch | None = None
        self._resolution = 0.0
        # (object_id, stamp_ns) of the payload the patch was measured under.
        self._payload_key: tuple[str, int] | None = None
        self._status = ""
        # (declaration stamp, object_id, object stamp) attested under, and the witness.
        self._attested: tuple[int, str, int] | None = None
        self._witness: SupportContactWitness | None = None
        # The witness the last ``attest`` reported, so a drop made elsewhere (a
        # retraction) still reports a change once.
        self._reported: SupportContactWitness | None = None

    @property
    def declaration(self) -> PlaceDeclaration | None:
        """Dispatch's optional declaration in force (region-less), or ``None``."""
        return self._declaration

    @property
    def patch(self) -> PlacePatch | None:
        """The latched patch, or ``None``."""
        return self._patch

    @property
    def payload_key(self) -> tuple[str, int] | None:
        """``(object_id, stamp_ns)`` of the payload the latched patch was measured under."""
        return self._payload_key

    def _transition(self, status: str, line: str) -> None:
        if status != self._status:
            self._status = status
            self._log(line)

    def _retract(self, reason: PlaceRefusal, detail: str) -> None:
        had = self._patch is not None
        self._patch = None
        self._payload_key = None
        # The witness dies with the region, whatever retracted it — the freeze TTL
        # included. The TTL equals the kernel's place_region_max_age_s, so no part of the
        # place allowance (approach slab or proximity witness) outlives the measurement
        # the kernel would accept.
        self._witness = None
        verb = "place_target_retracted" if had else "place_target_refused"
        self._transition(f"refused:{reason}", f"{verb} reason={reason} — {detail}; no place region")

    def on_declaration(self, declaration: PlaceDeclaration) -> None:
        """Fold in dispatch's goal-scope declaration.

        A retraction (goal end, cancel, E-stop) ends the place allowance: the latched
        patch and the witness go with it, and nothing re-arms until a new declaration.
        A new declaration drops them too, so a patch latched under the previous one
        (and its hint) is re-measured under this one.
        """
        current = self._declaration
        if not declaration.active:
            if current is not None:
                self._declaration = None
                self._retract(
                    PlaceRefusal.NO_DECLARATION, f"dispatch retracted {current.target_id!r}"
                )
            return
        if current is not None and (current.target_id, current.stamp_ns) == (
            declaration.target_id,
            declaration.stamp_ns,
        ):
            return  # the latched copy again
        if self._patch is not None:
            self._retract(
                PlaceRefusal.REDECLARED,
                f"new declaration {declaration.target_id!r}: re-measure under it",
            )
        # Dispatch's region, if any, is never relayed: only this producer's own.
        self._declaration = declaration.model_copy(update={"region": None})
        self._witness = None
        self._transition(
            f"declared:{declaration.stamp_ns}",
            f"place_target declared target={declaration.target_id} "
            f"rskill={declaration.rskill_id!r} trace={declaration.trace_id!r} — a hint; the "
            "surface is measured under the payload",
        )

    def latch(self, patch: PlacePatch, *, payload_key: tuple[str, int], resolution: float) -> None:
        """Hold a freshly measured patch (its region stamped with its grid's data)."""
        self._patch = patch
        self._payload_key = payload_key
        self._resolution = resolution
        self._transition(
            f"latched:{patch.region.evidence_ref}",
            f"place_target_latched object={payload_key[0]} plane_z={patch.plane_z:.3f} "
            f"centre={tuple(round(v, 3) for v in patch.region.pose.xyz)} "
            f"half_extents={tuple(round(v, 3) for v in patch.region.half_extents)} "
            f"evidence={patch.region.evidence_ref!r} — a measured surface, not a proven support",
        )

    def reverified(self, grid_stamp_ns: int) -> None:
        """The latched patch held on a newer grid: restart its freeze clock."""
        patch = self._patch
        if patch is None:
            return
        region = patch.region.model_copy(update={"stamp_ns": grid_stamp_ns})
        self._patch = replace(patch, region=region)
        self._transition("verified", "place_target_verified")

    def refuse(self, reason: PlaceRefusal, detail: str, *, now_ns: int) -> None:
        """One failed tick: a lost view freezes a latched patch, anything else retracts."""
        if reason not in _LOST_VIEW or self._patch is None:
            self._retract(reason, detail)
            return
        self._transition(
            f"frozen:{reason}",
            f"place_target view lost — {reason}: {detail}; holding the latched patch for at "
            f"most {self._freeze_ns / 1e9:.1f} s from its last verification",
        )
        self._expire_frozen(now_ns)

    def _expire_frozen(self, now_ns: int) -> None:
        patch = self._patch
        if patch is not None and now_ns - patch.region.stamp_ns > self._freeze_ns:
            self._retract(
                PlaceRefusal.FREEZE_TTL,
                f"latched patch last verified {(now_ns - patch.region.stamp_ns) / 1e9:.2f} s ago "
                f"(> {self._freeze_ns / 1e9:.1f} s)",
            )

    def live(self, *, now_ns: int) -> PlaceDeclaration | None:
        """Dispatch's live declaration; an expired one ends the allowance like a retraction."""
        declaration = self._declaration
        if declaration is not None and not declaration.is_live(now_ns=now_ns):
            self._declaration = None
            self._retract(
                PlaceRefusal.NO_DECLARATION,
                f"dispatch declaration {declaration.target_id!r} expired",
            )
            return None
        return declaration

    def envelope(self, *, now_ns: int) -> PlaceDeclaration | None:
        """The declaration for the envelope now, or ``None``.

        Only dispatch's live goal-scope declaration — the producer never declares on
        its own, so no place allowance exists outside a running goal. While a patch is
        latched it carries the region, scoped to the payload it was measured under when
        dispatch names none. An over-age latched region is retracted here too, so a
        stalled measurement loop cannot keep one alive.
        """
        declaration = self.live(now_ns=now_ns)
        self._expire_frozen(now_ns)
        patch, key = self._patch, self._payload_key
        if declaration is None or patch is None or key is None:
            return declaration
        return declaration.model_copy(
            update={"region": patch.region, "object_id": declaration.object_id or key[0]}
        )

    def attest(
        self,
        candidates: Sequence[tuple[AttachedCollisionObject, NDArray[np.float64] | None, bool]],
        *,
        now_ns: int,
    ) -> bool:
        """Arm or drop the witness; ``True`` when the published set must change.

        Args:
            candidates: ``(attachment, T_grid_from_attach_link or None, payload live)``
                per gripper leg holding something — ``payload live`` is the trigger's
                loaded bit for a held payload — plus ``(record, None, True)`` per
                frozen release record (design note §2.3 "Release"): there ``True``
                means the payload the witness names is still published, so a witness
                armed while held carries through the window, and the ``None`` pose
                arms no new one. The window closing removes the candidate and drops it.
            now_ns: The node clock.
        """
        declaration = self.envelope(now_ns=now_ns)
        if self._witness is not None and not any(
            self._attested is not None
            and (obj.object_id, obj.stamp_ns) == self._attested[1:]
            and loaded
            for obj, _, loaded in candidates
        ):
            self._witness = None  # the payload it named is gone, changed or let go
        patch = self._patch
        if self._witness is None and declaration is not None and patch is not None:
            for obj, t_base_link, loaded in candidates:
                key = (declaration.stamp_ns, obj.object_id, obj.stamp_ns)
                if not loaded or t_base_link is None or self._attested == key:
                    continue
                if declaration.object_id and declaration.object_id != obj.object_id:
                    continue
                witness, detail = payload_witness(
                    obj,
                    t_base_link,
                    patch,
                    support_id=declaration.target_id,
                    resolution=self._resolution,
                    extrinsic_error_m=self._extrinsic,
                    stamp_ns=now_ns,
                )
                if witness is None:
                    continue
                self._witness, self._attested = witness, key
                self._log(
                    f"place_target witness armed object={obj.object_id} "
                    f"support={declaration.target_id} patch_m={witness.patch_radius_m:.3f} "
                    f"penetration_m={witness.max_penetration_m:.3f} — map_support_proximity: "
                    f"proximity to a map-measured plane, not sensed contact ({detail})"
                )
                break
        changed = self._witness is not self._reported
        if changed and self._witness is None:
            self._log("place_target witness dropped")
        self._reported = self._witness
        return changed

    def decorate(self, objects: Sequence[AttachedCollisionObject]) -> list[AttachedCollisionObject]:
        """The attachment set with the held witness on the object it names."""
        witness, key = self._witness, self._attested
        if witness is None or key is None:
            return list(objects)
        return [
            obj.model_copy(update={"support_contact": witness})
            if (obj.object_id, obj.stamp_ns) == key[1:]
            else obj
            for obj in objects
        ]


def _published(leg: _GripperLeg) -> AttachedCollisionObject | None:
    """What one leg publishes: its held payload, else its frozen release record."""
    held: AttachedCollisionObject | None = leg.attachment
    window = leg.release  # read once: the executor's poll may close it meanwhile
    if held is None and window is not None:
        held = window.record
    return held


def _witness_candidates(
    legs: Iterable[_GripperLeg],
    pose: Callable[[AttachedCollisionObject], NDArray[np.float64] | None],
) -> list[tuple[AttachedCollisionObject, NDArray[np.float64] | None, bool]]:
    """``PlaceTargetTracker.attest`` candidates from the bridge's gripper legs.

    A held payload is ``(attachment, pose(attachment), trigger loaded)``. A leg
    whose window holds a frozen release record (design note §2.3 "Release") is
    ``(record, None, True)``: the record keeps the witness it had — same object
    and stamp, now resting on the patch, exactly when the kernel needs the
    support-contact exemption — and the ``None`` pose arms no new one.
    """
    candidates: list[tuple[AttachedCollisionObject, NDArray[np.float64] | None, bool]] = []
    for leg in legs:
        held, window = leg.attachment, leg.release  # each read once (proprio vs executor)
        if held is not None:
            candidates.append((held, pose(held), bool(leg.trigger.attached)))
        elif window is not None:
            candidates.append((window.record, None, True))
    return candidates


def place_tick(
    tracker: PlaceTargetTracker,
    grid: VoxelLattice,
    grid_stamp_ns: int,
    *,
    carried: tuple[AttachedCollisionObject, Posed] | None,
    published: dict[tuple[str, int], Posed],
    extrinsic_error_m: float,
    search_depth_m: float,
    now_ns: int,
) -> None:
    """One measurement tick on a fresh grid — pure, so the whole loop is testable.

    Args:
        tracker: The leg's state.
        grid: The fresh lattice.
        grid_stamp_ns: Its ``source_stamp`` — the world data's capture time, which is the
            region's ``stamp_ns`` and its freeze clock.
        carried: The held, loaded payload the surface is measured under (posed in the
            grid frame), or ``None``.
        published: Every published payload (held or frozen-released) posed in the grid
            frame, keyed ``(object_id, stamp_ns)``; all are excluded from the map checks.
        extrinsic_error_m: See ``measure_under_payload``.
        search_depth_m: See ``measure_under_payload``.
        now_ns: The node clock.
    """
    if tracker.live(now_ns=now_ns) is None:
        tracker.refuse(
            PlaceRefusal.NO_DECLARATION,
            "no live dispatch place declaration (no goal in progress)",
            now_ns=now_ns,
        )
        return
    exclude = [p for posed in published.values() for p in posed]
    patch, held = tracker.patch, tracker.payload_key
    if patch is not None and held is not None:
        if held not in published:
            tracker.refuse(
                PlaceRefusal.PAYLOAD_CHANGED,
                "the payload the patch was measured under is no longer published",
                now_ns=now_ns,
            )
        elif carried is None or (
            (carried[0].object_id, carried[0].stamp_ns) == held and _over_patch(carried[1], patch)
        ):
            # Still over its patch, or released onto it: re-verify, never re-choose.
            verdict = verify_patch(grid, patch, exclude)
            if verdict is None:
                tracker.reverified(grid_stamp_ns)
            else:
                tracker.refuse(*verdict, now_ns=now_ns)
            return
    if carried is None:
        if tracker.patch is None:
            tracker.refuse(PlaceRefusal.NO_PAYLOAD, "nothing held", now_ns=now_ns)
        return
    obj, posed = carried
    hint = tracker.declaration.search_box if tracker.declaration is not None else None
    if hint is not None and hint.frame_id != grid.frame_id:
        # A hint in another frame cannot be tested against this grid; never guess,
        # and never measure first: the refusal is the frame, not the patch.
        tracker.refuse(
            PlaceRefusal.FRAME_MISMATCH,
            f"the named surface is in frame {hint.frame_id!r}, the voxel grid in {grid.frame_id!r}",
            now_ns=now_ns,
        )
        return
    result = measure_under_payload(
        grid,
        posed,
        extrinsic_error_m=extrinsic_error_m,
        search_depth_m=search_depth_m,
        exclude=exclude,
        evidence_ref=f"{obj.object_id}@{grid_stamp_ns};",
    )
    if not isinstance(result, tuple) and hint is not None:
        t = homogeneous_from_quat_xyz(hint.pose.xyz, hint.pose.quat_xyzw)
        local = (np.asarray(result.region.pose.xyz) - t[:3, 3]) @ t[:3, :3]
        if np.any(np.abs(local) > np.asarray(hint.half_extents)):
            result = (PlaceRefusal.OUTSIDE_HINT, "the measured patch is outside the named surface")
    if isinstance(result, tuple):
        tracker.refuse(*result, now_ns=now_ns)
        return
    tracker.latch(
        replace(result, region=result.region.model_copy(update={"stamp_ns": grid_stamp_ns})),
        payload_key=(obj.object_id, obj.stamp_ns),
        resolution=grid.resolution,
    )


# ── ROS wiring ───────────────────────────────────────────────────────────────


class PlaceTargetLeg:
    """ROS wiring for ``PlaceTargetTracker``, owned by ``VisionAttachmentBridge``.

    Reuses the bridge's tf2 buffer, legs and publisher; owns its declaration and
    voxel subscriptions and its measurement timer.

    Args:
        node: The HAL lifecycle node.
        bridge: The owning bridge.
        config: The bridge's config (the ``place_target_*`` fields, ``grid_max_age_s``).

    Raises:
        ROSConfigError: A non-positive rate, freeze, depth or extrinsic bound, or a freeze
            past ``2 * grid_max_age_s`` (the kernel's place-region age bound).
    """

    def __init__(
        self, node: Any, bridge: VisionAttachmentBridge, config: VisionAttachmentConfig
    ) -> None:
        """Validate the config; create no ROS entities yet."""
        freeze_s = (
            config.place_target_freeze_s
            if config.place_target_freeze_s is not None
            else 2.0 * config.grid_max_age_s
        )
        if not (
            config.place_target_rate_hz > 0.0
            and freeze_s > 0.0
            and config.place_target_extrinsic_error_m > 0.0
            and config.place_target_search_depth_m > 0.0
        ):
            raise ROSConfigError(
                "place_target_rate_hz, place_target_freeze_s, place_target_search_depth_m "
                "and place_target_extrinsic_error_m must be > 0."
            )
        if freeze_s > 2.0 * config.grid_max_age_s:
            # The kernel drops a region older than place_region_max_age_s (the deploy
            # passes 2 x its voxel deadline = 2 x grid_max_age_s): a longer freeze would
            # publish a region the kernel silently ignores mid-set-down.
            raise ROSConfigError(
                f"place_target_freeze_s {freeze_s!r} exceeds 2 x grid_max_age_s "
                f"({2.0 * config.grid_max_age_s!r}), the kernel's place-region age bound."
            )
        self._node = node
        self._bridge = bridge
        self._config = config
        self.tracker = PlaceTargetTracker(
            freeze_s=freeze_s,
            extrinsic_error_m=config.place_target_extrinsic_error_m,
            log=lambda line: node.get_logger().info(line),
        )
        # (lattice, grid stamp ns, monotonic receive time) of the newest grid.
        self._grid: tuple[VoxelLattice, int, float] | None = None
        self._subs: list[Any] = []
        self._timer: Any = None
        self._last_witness_s = -math.inf

    def setup(self) -> None:
        """Subscribe the optional declaration and the voxel grid; start the timer."""
        from openral_msgs.msg import OccupancyVoxels
        from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )

        declaration_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        voxel_qos = QoSProfile(  # the safety kernel's own profile for this topic
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._subs = [
            self._node.create_subscription(
                PlaceDeclarationMsg,
                "/openral/place_declaration",
                self._on_declaration,
                declaration_qos,
            ),
            self._node.create_subscription(
                OccupancyVoxels, "/openral/world_voxels", self._on_voxels, voxel_qos
            ),
        ]
        self._timer = self._node.create_timer(1.0 / self._config.place_target_rate_hz, self._tick)
        self._node.get_logger().info(
            f"place target leg: rate={self._config.place_target_rate_hz:.1f}Hz "
            f"search_depth={self._config.place_target_search_depth_m:.2f}m "
            f"extrinsic_error={self._config.place_target_extrinsic_error_m:.3f}m — measures "
            "the surface under the carried payload; drafted ADR-0097/ADR-0092 D6 amendments; "
            "the witness is proximity to a map-measured plane, not sensed contact"
        )

    def teardown(self) -> None:
        """Destroy every ROS entity; idempotent."""
        if self._timer is not None:
            self._timer.cancel()
            self._node.destroy_timer(self._timer)
            self._timer = None
        for sub in self._subs:
            self._node.destroy_subscription(sub)
        self._subs = []

    # ── bridge hooks ─────────────────────────────────────────────────────────

    def fill(self, msg: Any, *, now_ns: int) -> None:
        """Put the declaration in force (region when held) on one ``AttachmentState``."""
        declaration = self.tracker.envelope(now_ns=now_ns)
        msg.place_declaration_valid = declaration is not None
        if declaration is not None:
            declaration.fill_idl(msg.place_declaration)

    def decorate(self, objects: Sequence[AttachedCollisionObject]) -> list[AttachedCollisionObject]:
        """The attachment set with the place witness on the payload it names."""
        return self.tracker.decorate(objects)

    def on_joint_state(self) -> None:
        """Re-evaluate the witness (throttled); re-publish when it arms or drops."""
        now_s = time.monotonic()
        if now_s - self._last_witness_s < _WITNESS_PERIOD_S:
            return
        self._last_witness_s = now_s
        frame = self._grid[0].frame_id if self._grid is not None else ""
        candidates = _witness_candidates(
            self._bridge._legs, lambda obj: self._pose(obj, frame) if frame else None
        )
        if self.tracker.attest(candidates, now_ns=self._now_ns()):
            self._bridge._publish_attachment()

    # ── inputs ───────────────────────────────────────────────────────────────

    def _on_declaration(self, msg: Any) -> None:
        try:
            declaration = PlaceDeclaration.from_idl(msg)
        except (ValueError, ROSConfigError) as exc:
            # A malformed declaration licenses nothing; whatever was held dies.
            self._node.get_logger().error(f"place target: declaration rejected — {exc}")
            held = self.tracker.declaration
            if held is not None:
                self.tracker.on_declaration(held.model_copy(update={"active": False}))
            return
        self.tracker.on_declaration(declaration)

    def _on_voxels(self, msg: Any) -> None:
        # The world data's capture time, not ``header.stamp`` (production time, which a
        # bridge republishing an unchanged octree keeps fresh) — the kernel's own
        # voxel data-age clock. Unset (0) reads as stale in ``_tick``.
        src = msg.source_stamp
        stamp_ns = int(src.sec) * 1_000_000_000 + int(src.nanosec)
        try:
            self._grid = (lattice_from_msg(msg), stamp_ns, time.monotonic())
        except ROSConfigError as exc:
            self._grid = None
            self._node.get_logger().warning(f"place target: voxel grid dropped — {exc}")

    def _now_ns(self) -> int:
        return int(self._node.get_clock().now().nanoseconds)

    def _pose(self, obj: AttachedCollisionObject, frame: str) -> NDArray[np.float64] | None:
        """``frame <- obj.attach_link`` through the bridge's tf2, or ``None``."""
        return self._bridge._lookup(frame, self._bridge.tf_frame(obj.attach_link))

    # ── measurement ──────────────────────────────────────────────────────────

    def _tick(self) -> None:
        from openral_hal.vision_attachment_bridge import primitive_poses  # circular at import

        now_ns = self._now_ns()
        self.tracker.live(now_ns=now_ns)
        if self._grid is None:
            self.tracker.refuse(
                PlaceRefusal.NO_GRID, "no /openral/world_voxels grid", now_ns=now_ns
            )
            return
        grid, grid_stamp_ns, received = self._grid
        age_s = time.monotonic() - received
        data_age_s = (now_ns - grid_stamp_ns) / 1e9
        max_age_s = self._config.grid_max_age_s
        if grid_stamp_ns <= 0 or age_s > max_age_s or not 0.0 <= data_age_s <= max_age_s:
            self.tracker.refuse(
                PlaceRefusal.GRID_STALE,
                f"newest voxel grid received {age_s:.2f} s ago, its data (source_stamp) "
                f"{data_age_s:.2f} s old (unset or future = stale)",
                now_ns=now_ns,
            )
            return
        declared = self.tracker.declaration
        published: dict[tuple[str, int], Posed] = {}
        carried: tuple[AttachedCollisionObject, Posed] | None = None
        for leg in self._bridge._legs:
            obj = _published(leg)
            if obj is None:
                continue
            t = self._pose(obj, grid.frame_id)
            # An unposed payload is excluded from nothing (fail-closed) and measures nothing.
            posed = (
                list(zip(obj.primitives, primitive_poses(obj, t), strict=True))
                if t is not None
                else []
            )
            published[(obj.object_id, obj.stamp_ns)] = posed
            if (
                carried is None
                and posed
                and obj is leg.attachment
                and leg.trigger.attached
                and (
                    declared is None
                    or not declared.object_id
                    or declared.object_id == obj.object_id
                )
            ):
                carried = (obj, posed)
        place_tick(
            self.tracker,
            grid,
            grid_stamp_ns,
            carried=carried,
            published=published,
            extrinsic_error_m=self._config.place_target_extrinsic_error_m,
            search_depth_m=self._config.place_target_search_depth_m,
            now_ns=now_ns,
        )
