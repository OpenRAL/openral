"""Real-hardware place producer leg — a unit fixture verified live against the voxel map.

The real sibling of the simulator's place producer (``_sim_attachment_evidence``
``set_place_declaration`` / ``place_declaration`` / ``_place_witness``), owned by
``VisionAttachmentBridge`` so ``/openral/attachment_state`` keeps one authority.
Real pick-and-place design ``docs/reference/real-pick-place-design.md`` §2.3; the
two semantics it needs are **drafted, not approved** (ADR-0097 amendment: a
unit-surveyed, map-verified fixture counts as a producer-measured region on a fixed
base; ADR-0092 D6 amendment: a proximity witness). Default off
(``VisionAttachmentConfig.place_fixture_enabled``), and every failure fails closed.

1. **Declaration.** Dispatch's region-less copy arrives on
   ``/openral/place_declaration``. Its ``target_id`` names a ``UnitFixture`` of this
   cell's unit overlay; an unknown one is logged once and the declaration rides the
   envelope region-less — the pre-amendment margins.
2. **Live map verification**, at ``place_fixture_rate_hz`` against the newest
   ``/openral/world_voxels`` (older than 1 s = unverified). The fixture's top face is
   the box face along ``top_plane_normal``; it must be **occupied** — at least
   ``place_fixture_min_face_cover`` of its cell footprint holds an occupied cell
   whose centre is within one voxel of the surveyed face height — and the **free
   volume** above it (the face's prism from one voxel above the face to
   ``place_fixture_free_height_m``) must hold no occupied cell. A fixture box's top
   *is* its face, so that volume lies above the box, not inside it. Cells within one
   voxel of a carried payload's primitives (posed by tf2) are not counted: the
   octomap bridge clears exactly those (cell circumradius < one voxel) and a payload
   resting on the shelf must not fail its own check; a payload tf2 cannot pose is
   excluded from nothing, so its cells fail the check (fail-closed). Verified → the
   envelope's declaration carries ``region`` = the fixture box, in the grid frame,
   ``evidence_ref = unit_fixture:<unit>/<id>@<surveyed_on>;map_verified@<grid stamp>``,
   no ``geometry`` (the 30 mm blanket allowance path; the ADR-0098 offset is a WG
   item). Any failure → region-less, with ``place_fixture_unverified reason=...``
   logged once per transition.
3. **Witness substitute**, once per declaration (the simulator's
   ``_place_attested_stamp_ns`` hysteresis): live declaration, region verified, a
   payload attached on a leg whose trigger still reads loaded, its lowest primitive
   point within ``max(1 voxel, survey_uncertainty_m)`` of the face plane and its
   centre over the face → a ``SupportContactWitness`` on that object, plane in the
   object frame, ``patch_radius_m`` = the payload's footprint radius (≤ 0.5 m),
   ``max_penetration_m = min(survey_uncertainty_m, 0.01)``, ``support_id`` = the
   fixture id, ``evidence_kind = DECLARED_FIXTURE``. **It is proximity to a verified
   plane, not sensed contact**, and every log line and ``evidence_ref`` says so. It
   stays on the object (same stamp, so the kernel's latch key does not change) until
   the declaration is retracted, times out, is replaced, the region stops verifying,
   or the payload changes; each of those re-publishes without it. A payload released
   on the face keeps it on its frozen release record until the window closes.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
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
    SphereShape,
    SupportContactWitness,
    UnitFixture,
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
    "FixtureFace",
    "PlaceFixtureLeg",
    "PlaceFixtureTracker",
    "fixture_face",
    "payload_witness",
    "verify_fixture",
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

#: Tolerance on "top_plane_normal is one of the box's own axes".
_AXIS_TOL = 1e-6


@dataclass(frozen=True)
class FixtureFace:
    """A fixture's top face in its frame.

    Attributes:
        centre: Face centre, ``(3,)``.
        normal: Outward unit normal, ``(3,)``.
        axes: The two in-plane unit axes, ``(2, 3)``.
        half: In-plane half-extents along ``axes``.
    """

    centre: NDArray[np.float64]
    normal: NDArray[np.float64]
    axes: NDArray[np.float64]
    half: tuple[float, float]


def fixture_face(fixture: UnitFixture) -> FixtureFace:
    """The box face ``top_plane_normal`` points out of, in the fixture's frame.

    Raises:
        ROSConfigError: ``top_plane_normal`` is not one of the box's own axes — the
            "top face" of a box is then not a face.

    Example:
        >>> from openral_core import Pose6D, UnitFixture
        >>> shelf = UnitFixture(
        ...     id="cell:shelf_top",
        ...     frame_id="b",
        ...     pose=Pose6D(xyz=(0.45, 0.0, 0.30), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ...     half_extents=(0.20, 0.40, 0.01),
        ...     survey_uncertainty_m=0.01,
        ...     surveyed_on="2026-10-02",
        ...     surveyed_by="op",
        ...     method="tape",
        ... )
        >>> face = fixture_face(shelf)
        >>> face.centre.round(3).tolist(), face.half
        ([0.45, 0.0, 0.31], (0.2, 0.4))
    """
    n_box = np.asarray(fixture.top_plane_normal, dtype=np.float64)
    k = int(np.argmax(np.abs(n_box)))
    if abs(abs(n_box[k]) - 1.0) > _AXIS_TOL:
        raise ROSConfigError(
            f"fixture {fixture.id!r}: top_plane_normal {fixture.top_plane_normal} is not one "
            "of the box's own axes, so the box has no top face to verify"
        )
    rot = homogeneous_from_quat_xyz((0.0, 0.0, 0.0), fixture.pose.quat_xyzw)[:3, :3]
    normal = rot[:, k] * math.copysign(1.0, n_box[k])
    others = [i for i in range(3) if i != k]
    return FixtureFace(
        centre=np.asarray(fixture.pose.xyz, dtype=np.float64) + normal * fixture.half_extents[k],
        normal=normal,
        axes=rot[:, others].T.copy(),
        half=(fixture.half_extents[others[0]], fixture.half_extents[others[1]]),
    )


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


def _posed_primitives(
    obj: AttachedCollisionObject, t_base_link: NDArray[np.float64]
) -> list[tuple[AttachedCollisionPrimitive, NDArray[np.float64]]]:
    """Each primitive with its 4x4 pose in the frame ``t_base_link`` maps into."""
    p = obj.pose_in_link
    t_base_obj = t_base_link @ homogeneous_from_quat_xyz(p.xyz, p.quat_xyzw)
    return [
        (
            prim,
            t_base_obj
            @ homogeneous_from_quat_xyz(prim.pose_in_object.xyz, prim.pose_in_object.quat_xyzw),
        )
        for prim in obj.primitives
    ]


def _inside_any(
    points: NDArray[np.float64],
    posed: Sequence[tuple[AttachedCollisionPrimitive, NDArray[np.float64]]],
    *,
    pad: float,
) -> NDArray[np.bool_]:
    """Points inside any primitive's local AABB grown by ``pad`` (a conservative hull)."""
    hit = np.zeros(len(points), dtype=bool)
    for prim, t in posed:
        local = (points - t[:3, 3]) @ t[:3, :3]
        hit |= np.all(np.abs(local) <= _shape_half_extents(prim) + pad, axis=1)
    return hit


# ── verification ─────────────────────────────────────────────────────────────


def verify_fixture(
    grid: VoxelLattice,
    fixture: UnitFixture,
    *,
    min_face_cover: float,
    free_height_m: float,
    payload: Sequence[tuple[AttachedCollisionPrimitive, NDArray[np.float64]]] = (),
) -> tuple[str, str] | None:
    """Check a fixture against one voxel grid; ``None`` = verified, else ``(reason, detail)``.

    Args:
        grid: The published lattice.
        fixture: The surveyed fixture.
        min_face_cover: Fraction of the face's cell footprint that must be occupied
            within one voxel of the face height.
        free_height_m: Top of the free volume above the face.
        payload: Carried primitives posed in the grid frame; occupied cells within one
            voxel of them are not counted against the free volume.

    Returns:
        ``None`` when the face is there and the volume above it is empty, else a typed
        reason (``frame_mismatch`` | ``fixture_normal`` | ``face_missing`` |
        ``free_volume_occupied``) and a detail line.
    """
    if grid.frame_id != fixture.frame_id:
        return "frame_mismatch", f"fixture in {fixture.frame_id!r}, grid in {grid.frame_id!r}"
    try:
        face = fixture_face(fixture)
    except ROSConfigError as exc:
        return "fixture_normal", str(exc)
    res = grid.resolution
    centres = grid.occupied_centers()
    rel = centres - face.centre
    height = rel @ face.normal
    uv = rel @ face.axes.T
    half = np.asarray(face.half)
    over_face = np.all(np.abs(uv) <= half, axis=1)
    on_face = over_face & (np.abs(height) <= res)
    bins_uv = np.maximum(1, np.ceil(2.0 * half / res - 1e-9)).astype(np.int64)
    bins = np.clip(np.floor((uv[on_face] + half) / res).astype(np.int64), 0, bins_uv - 1)
    covered = len({(int(i), int(j)) for i, j in bins})
    total = int(bins_uv[0] * bins_uv[1])
    if covered < min_face_cover * total:
        return (
            "face_missing",
            f"{covered}/{total} face cells occupied within one voxel of the surveyed height "
            f"(< {min_face_cover:.2f})",
        )
    above = over_face & (height > res) & (height <= free_height_m)
    if payload:
        above &= ~_inside_any(centres, payload, pad=res)
    if above.any():
        first = centres[above][np.argmin(height[above])]
        return (
            "free_volume_occupied",
            f"{int(above.sum())} occupied cells above the face, lowest at "
            f"{tuple(round(float(v), 3) for v in first)}",
        )
    return None


def fixture_region(fixture: UnitFixture, *, unit: str, grid_stamp_ns: int) -> PlaceRegion:
    """The verified fixture as a ``PlaceRegion``: its own box, no geometry."""
    return PlaceRegion(
        frame_id=fixture.frame_id,
        pose=fixture.pose,
        half_extents=fixture.half_extents,
        evidence_ref=(
            f"unit_fixture:{unit}/{fixture.id}@{fixture.surveyed_on.isoformat()};"
            f"map_verified@{grid_stamp_ns}"
        ),
        stamp_ns=grid_stamp_ns,
    )


def payload_witness(
    obj: AttachedCollisionObject,
    t_base_link: NDArray[np.float64],
    fixture: UnitFixture,
    *,
    resolution: float,
    stamp_ns: int,
) -> tuple[SupportContactWitness | None, str]:
    """The proximity witness for ``obj`` resting on ``fixture``'s face, or why not.

    ``t_base_link`` poses ``obj.attach_link`` in the fixture's frame. Not sensed
    contact: the payload's lowest primitive point is within ``max(resolution,
    survey_uncertainty_m)`` of the surveyed plane and its centre is over the face.

    Returns:
        ``(witness, detail)``; ``witness`` is ``None`` when the payload is not there.
    """
    face = fixture_face(fixture)
    n = face.normal
    posed = _posed_primitives(obj, t_base_link)
    lowest = min(
        float((t[:3, 3] - face.centre) @ n) - _support_along(prim, t[:3, :3], n)
        for prim, t in posed
    )
    centre = np.mean([t[:3, 3] for _, t in posed], axis=0)
    uv = face.axes @ (centre - face.centre)
    tol = max(resolution, fixture.survey_uncertainty_m)
    detail = f"lowest point {lowest * 1e3:+.1f} mm from the plane (tol {tol * 1e3:.1f} mm)"
    if abs(lowest) > tol:
        return None, detail
    if np.any(np.abs(uv) > np.asarray(face.half)):
        return None, f"payload centre is off the face ({detail})"
    lateral = [
        float(np.linalg.norm((t[:3, 3] - centre) - n * float((t[:3, 3] - centre) @ n)))
        + float(np.linalg.norm(_shape_half_extents(prim)))
        for prim, t in posed
    ]
    p = obj.pose_in_link
    t_base_obj = t_base_link @ homogeneous_from_quat_xyz(p.xyz, p.quat_xyzw)
    r_obj, t_obj = t_base_obj[:3, :3], t_base_obj[:3, 3]
    contact = centre - n * float((centre - face.centre) @ n)
    normal_obj = r_obj.T @ n
    witness = SupportContactWitness(
        support_id=fixture.id,
        contact_point_in_object=tuple(float(v) for v in r_obj.T @ (contact - t_obj)),
        contact_normal_in_object=tuple(float(v) for v in normal_obj / np.linalg.norm(normal_obj)),
        patch_radius_m=min(max(lateral), _MAX_PATCH_RADIUS_M),
        max_penetration_m=min(fixture.survey_uncertainty_m, _MAX_PENETRATION_M),
        confidence=_WITNESS_CONFIDENCE,
        evidence_kind=AttachmentEvidenceKind.DECLARED_FIXTURE,
        evidence_ref=(
            f"declared_fixture:{fixture.id}: proximity to a verified plane, not sensed "
            f"contact; {detail}"
        ),
        stamp_ns=stamp_ns,
    )
    return witness, detail


# ── state machine ────────────────────────────────────────────────────────────


class PlaceFixtureTracker:
    """Declaration, verification verdict and witness — pure, clock-free.

    Every method takes the caller's ``now_ns`` (the node's ROS clock, the
    declaration's domain). ``log`` receives one line per transition.

    Args:
        fixtures: This cell's surveyed fixtures.
        unit: The unit name, for ``evidence_ref``.
        log: Sink for transition lines.

    Example:
        >>> from openral_core import PlaceDeclaration
        >>> lines: list[str] = []
        >>> t = PlaceFixtureTracker(fixtures=(), unit="thor", log=lines.append)
        >>> t.on_declaration(PlaceDeclaration(target_id="cell:x", timeout_s=9.0, stamp_ns=0))
        >>> t.envelope(now_ns=1).region is None, lines[-1].split()[:2]
        (True, ['place_fixture_unverified', 'reason=unknown_fixture'])
    """

    def __init__(
        self, *, fixtures: Sequence[UnitFixture], unit: str, log: Callable[[str], None]
    ) -> None:
        """Start with no declaration."""
        self._fixtures = {f.id: f for f in fixtures}
        self._unit = unit
        self._log = log
        self._declaration: PlaceDeclaration | None = None
        self._fixture: UnitFixture | None = None
        self._region: PlaceRegion | None = None
        self._resolution = 0.0
        self._status = ""
        # (declaration stamp, object_id, object stamp) attested under, and the witness.
        self._attested: tuple[int, str, int] | None = None
        self._witness: SupportContactWitness | None = None
        # The witness the last ``attest`` reported, so a drop made elsewhere (a
        # retraction, an unverified map) still reports a change once.
        self._reported: SupportContactWitness | None = None

    @property
    def declaration(self) -> PlaceDeclaration | None:
        """Dispatch's declaration in force (region-less), or ``None``."""
        return self._declaration

    @property
    def fixture(self) -> UnitFixture | None:
        """The fixture the declaration names, when this unit surveys it."""
        return self._fixture

    @property
    def region(self) -> PlaceRegion | None:
        """The map-verified region, or ``None``."""
        return self._region

    def _transition(self, status: str, line: str) -> None:
        if status != self._status:
            self._status = status
            self._log(line)

    def _target(self) -> str:
        return self._declaration.target_id if self._declaration is not None else "-"

    def unverified(self, reason: str, detail: str) -> None:
        """Drop the region (and any witness) for a typed reason; logged on transition."""
        self._region = None
        self._drop_witness()
        self._transition(
            f"unverified:{reason}",
            f"place_fixture_unverified reason={reason} target={self._target()} — {detail}; "
            "declaration goes out region-less",
        )

    def on_declaration(self, declaration: PlaceDeclaration) -> None:
        """Fold in dispatch's copy; retraction or a new declaration clears everything."""
        current = self._declaration
        if not declaration.active:
            if current is not None:
                self._clear(f"dispatch retracted {current.target_id!r}")
            return
        if current is not None and (current.target_id, current.stamp_ns) == (
            declaration.target_id,
            declaration.stamp_ns,
        ):
            return  # the latched copy again
        self._clear("")
        # Dispatch's region, if any, is never relayed: only this producer's own.
        self._declaration = declaration.model_copy(update={"region": None})
        self._fixture = self._fixtures.get(declaration.target_id)
        self._status = ""
        if self._fixture is None:
            have = ", ".join(self._fixtures) or "none"
            self.unverified("unknown_fixture", f"this unit surveys no such fixture (has: {have})")
        else:
            self._transition(
                f"declared:{declaration.stamp_ns}",
                f"place_fixture declared target={declaration.target_id} "
                f"rskill={declaration.rskill_id!r} trace={declaration.trace_id!r} — "
                "awaiting map verification",
            )

    def _clear(self, why: str) -> None:
        self._declaration = None
        self._fixture = None
        self._region = None
        self._drop_witness()
        if why:
            self._transition(f"cleared:{why}", f"place_fixture cleared — {why}")

    def _drop_witness(self) -> None:
        self._witness = None

    def grid_fresh(self, age_s: float | None, *, max_age_s: float) -> bool:
        """Whether the newest grid (``age_s`` old, ``None`` = none yet) can vouch for the face.

        ``max_age_s`` is ``VisionAttachmentConfig.grid_max_age_s``: the kernel's own
        voxel deadline, so a grid the kernel would refuse as stale vouches for nothing.

        A missing or stale grid unverifies the region (``no_grid`` / ``grid_stale``):
        the face is only as verified as the newest map. ``False`` too when there is
        no fixture to verify.
        """
        if self._declaration is None or self._fixture is None:
            return False
        if age_s is None:
            self.unverified("no_grid", "no /openral/world_voxels grid yet")
            return False
        if age_s > max_age_s:
            self.unverified("grid_stale", f"newest voxel grid is {age_s:.2f} s old")
            return False
        return True

    def verified(self, region: PlaceRegion, *, resolution: float) -> None:
        """Hold a region that passed the map check."""
        self._region = region
        self._resolution = resolution
        self._transition(
            "verified",
            f"place_fixture_verified target={self._target()} evidence={region.evidence_ref!r}",
        )

    def live(self, *, now_ns: int) -> PlaceDeclaration | None:
        """The live declaration, clearing an expired one."""
        declaration = self._declaration
        if declaration is not None and not declaration.is_live(now_ns=now_ns):
            self._clear(f"{declaration.target_id!r} expired (timeout_s={declaration.timeout_s})")
            return None
        return declaration

    def envelope(self, *, now_ns: int) -> PlaceDeclaration | None:
        """The declaration for the envelope now: dispatch's fields plus the verified region."""
        declaration = self.live(now_ns=now_ns)
        if declaration is None:
            return None
        return declaration.model_copy(update={"region": self._region})

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
        declaration = self.live(now_ns=now_ns)
        if self._witness is not None and not any(
            self._attested is not None
            and (obj.object_id, obj.stamp_ns) == self._attested[1:]
            and loaded
            for obj, _, loaded in candidates
        ):
            self._drop_witness()  # the payload it named is gone, changed or let go
        fixture = self._fixture
        if (
            self._witness is None
            and declaration is not None
            and fixture is not None
            and self._region is not None
        ):
            for obj, t_base_link, loaded in candidates:
                key = (declaration.stamp_ns, obj.object_id, obj.stamp_ns)
                if not loaded or t_base_link is None or self._attested == key:
                    continue
                if declaration.object_id and declaration.object_id != obj.object_id:
                    continue
                witness, detail = payload_witness(
                    obj, t_base_link, fixture, resolution=self._resolution, stamp_ns=now_ns
                )
                if witness is None:
                    continue
                self._witness, self._attested = witness, key
                self._log(
                    f"place_fixture witness armed object={obj.object_id} support={fixture.id} "
                    f"patch_m={witness.patch_radius_m:.3f} "
                    f"penetration_m={witness.max_penetration_m:.3f} — declared_fixture: "
                    f"proximity to a verified plane, not sensed contact ({detail})"
                )
                break
        changed = self._witness is not self._reported
        if changed and self._witness is None:
            self._log(f"place_fixture witness dropped target={self._target()}")
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


def _witness_candidates(
    legs: Iterable[_GripperLeg],
    pose: Callable[[AttachedCollisionObject], NDArray[np.float64] | None],
) -> list[tuple[AttachedCollisionObject, NDArray[np.float64] | None, bool]]:
    """``PlaceFixtureTracker.attest`` candidates from the bridge's gripper legs.

    A held payload is ``(attachment, pose(attachment), trigger loaded)``. A leg
    whose window holds a frozen release record (design note §2.3 "Release") is
    ``(record, None, True)``: the record keeps the witness it had — same object
    and stamp, now resting on the face, exactly when the kernel needs the
    support-contact exemption — and the ``None`` pose arms no new one. The
    window closing removes the candidate, which drops the witness.
    """
    candidates: list[tuple[AttachedCollisionObject, NDArray[np.float64] | None, bool]] = []
    for leg in legs:
        if leg.attachment is not None:
            candidates.append((leg.attachment, pose(leg.attachment), bool(leg.trigger.attached)))
        elif leg.release is not None:
            candidates.append((leg.release.record, None, True))
    return candidates


# ── ROS wiring ───────────────────────────────────────────────────────────────


class PlaceFixtureLeg:
    """ROS wiring for ``PlaceFixtureTracker``, owned by ``VisionAttachmentBridge``.

    Reuses the bridge's tf2 buffer, legs and publisher; owns its declaration and
    voxel subscriptions and its verification timer.

    Args:
        node: The HAL lifecycle node.
        bridge: The owning bridge.
        config: The bridge's config (the ``place_fixture_*`` fields, ``unit_fixtures``,
            ``robot_unit``).

    Raises:
        ROSConfigError: Enabled with no fixtures, or a non-positive rate / free height,
            or a face cover outside ``(0, 1]``.
    """

    def __init__(
        self, node: Any, bridge: VisionAttachmentBridge, config: VisionAttachmentConfig
    ) -> None:
        """Validate the config; create no ROS entities yet."""
        if not config.unit_fixtures:
            raise ROSConfigError(
                "place_fixture_enabled needs the robot unit's surveyed fixtures; "
                f"unit {config.robot_unit!r} carries none."
            )
        if not (
            config.place_fixture_rate_hz > 0.0
            and config.place_fixture_free_height_m > 0.0
            and 0.0 < config.place_fixture_min_face_cover <= 1.0
        ):
            raise ROSConfigError(
                "place_fixture_rate_hz and place_fixture_free_height_m must be > 0 and "
                "place_fixture_min_face_cover in (0, 1]."
            )
        self._node = node
        self._bridge = bridge
        self._config = config
        self.tracker = PlaceFixtureTracker(
            fixtures=config.unit_fixtures,
            unit=config.robot_unit,
            log=lambda line: node.get_logger().info(line),
        )
        # (lattice, grid stamp ns, monotonic receive time) of the newest grid.
        self._grid: tuple[VoxelLattice, int, float] | None = None
        self._subs: list[Any] = []
        self._timer: Any = None
        self._last_witness_s = -math.inf

    def setup(self) -> None:
        """Subscribe the declaration and the voxel grid; start the verification timer."""
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
        self._timer = self._node.create_timer(1.0 / self._config.place_fixture_rate_hz, self._tick)
        self._node.get_logger().info(
            f"place fixture leg: unit={self._config.robot_unit!r} "
            f"fixtures={[f.id for f in self._config.unit_fixtures]} "
            f"rate={self._config.place_fixture_rate_hz:.1f}Hz "
            f"min_face_cover={self._config.place_fixture_min_face_cover:.2f} "
            f"free_height={self._config.place_fixture_free_height_m:.3f}m — drafted ADR-0097/"
            "ADR-0092 D6 amendments; the witness is proximity, not sensed contact"
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
        """Put the declaration in force (region when verified) on one ``AttachmentState``."""
        self._check_grid_age()
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
        self._check_grid_age()
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
            self._node.get_logger().error(f"place fixture: declaration rejected — {exc}")
            held = self.tracker.declaration
            if held is not None:
                self.tracker.on_declaration(held.model_copy(update={"active": False}))
            return
        self.tracker.on_declaration(declaration)

    def _on_voxels(self, msg: Any) -> None:
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        try:
            self._grid = (lattice_from_msg(msg), stamp_ns, time.monotonic())
        except ROSConfigError as exc:
            self._grid = None
            self._node.get_logger().warning(f"place fixture: voxel grid dropped — {exc}")

    def _now_ns(self) -> int:
        return int(self._node.get_clock().now().nanoseconds)

    def _check_grid_age(self) -> bool:
        """Whether a fresh grid is in hand (see ``PlaceFixtureTracker.grid_fresh``)."""
        return self.tracker.grid_fresh(
            None if self._grid is None else time.monotonic() - self._grid[2],
            max_age_s=self._config.grid_max_age_s,
        )

    def _pose(self, obj: AttachedCollisionObject, frame: str) -> NDArray[np.float64] | None:
        """``frame <- obj.attach_link`` through the bridge's tf2, or ``None``."""
        return self._bridge._lookup(frame, self._bridge.tf_frame(obj.attach_link))

    # ── verification ─────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now_ns = self._now_ns()
        fixture = self.tracker.fixture
        if self.tracker.live(now_ns=now_ns) is None or fixture is None:
            return
        if not self._check_grid_age():
            return
        assert self._grid is not None
        grid, grid_stamp_ns, _ = self._grid
        payload: list[tuple[AttachedCollisionPrimitive, NDArray[np.float64]]] = []
        for leg in self._bridge._legs:
            if leg.attachment is None:
                continue
            t_grid_link = self._pose(leg.attachment, grid.frame_id)
            if t_grid_link is not None:  # unposed → excluded from nothing (fail-closed)
                payload.extend(_posed_primitives(leg.attachment, t_grid_link))
        verdict = verify_fixture(
            grid,
            fixture,
            min_face_cover=self._config.place_fixture_min_face_cover,
            free_height_m=self._config.place_fixture_free_height_m,
            payload=payload,
        )
        if verdict is not None:
            self.tracker.unverified(*verdict)
            return
        # verify_fixture refused any frame but the grid's, so the region is in it.
        self.tracker.verified(
            fixture_region(fixture, unit=self._config.robot_unit, grid_stamp_ns=grid_stamp_ns),
            resolution=grid.resolution,
        )
