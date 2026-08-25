# python/hal/src/openral_hal/sim_sensor_bridge.py
"""Shared sim-sensor + viewer bridge for scene-attached HAL lifecycle nodes.

Republishes whatever a ``SimAttachedHAL`` exposes — RGB camera frames
(``read_images``) on ``/openral/cameras/<n>/image`` and an optional live
``mujoco.viewer`` — for any manifest-driven node, so franka/ur5e/... reach
deploy-sim scene+camera+viewer parity without per-package wiring. Phase 2
adds ``/scan`` + depth ``PointCloud2``. rclpy is imported lazily so the
module stays import-safe in pure-Python CI.

It also owns the E-stop ground-truth record: one
``sim.estop_ground_truth_snapshot`` line per kernel stop — MuJoCo contacts,
near-miss distances, joint state, base TF, and the candidate chunk the
kernel was checking — so a stop can be adjudicated real-vs-false after the
fact. Diagnostics only: nothing on that path gates, delays, or alters
actuation.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from openral_hal.convex_distance import ConvexDistance, convex_geom_distance
from openral_hal.mobile_base_bridge import describes_mobile_base

# Throttle dashboard thumbnail emission to ~1 Hz per camera (1e9 ns).
# The live ROS topic stays at the higher camera_rate_hz; only the OTel
# ``sensors.read_latest`` span is rate-limited to avoid ballooning OTLP
# payload with redundant thumbnails (the dashboard polls at ~1 Hz anyway).
_THUMB_INTERVAL_NS = 1_000_000_000
_IMAGE_DIM = 3  # HWC ndarray
_RGB_CHANNELS = 3
# Below this a quaternion carries no usable direction; keep the identity
# rotation rather than dividing by ~0.
_DEGENERATE_QUAT_NORM = 1e-12

# -- E-stop ground-truth snapshot bounds --
# Near-miss probe: the kernel stops on a *margin* (a few mm to a few cm), so
# at the stop instant MuJoCo's contact list is usually EMPTY — the honest
# ground truth of "how close was it really" is the signed geom distance.
# Probed only for geom pairs whose bounding spheres are within this gap,
# ranked closest-first, and truncated to the closest few.
#
# The caps were 256/8 in the first field round and produced a nearly WRONG
# verdict: on a mobile manipulator all 8 slots saturated on
# mobilebase↔floor pairs at 0-2 mm (the robot merely standing on the
# ground) and hid an arm that was 17-30 mm inside a freezer door. The
# structural fix is scoping the probe to the links the kernel actually
# checks (:func:`kernel_checked_body_ids`); these wider caps are the belt to
# that braces.
#
# The exact distance is `openral_hal.convex_distance.convex_geom_distance`,
# NOT ``mujoco.mj_geomDistance`` — that call is unreliable for exactly the
# pairs this probe adjudicates (a RoboCasa fixture geom against a panda
# collision mesh), in two distinct silent modes, and the module docstring
# carries the measurements. The certified instrument costs ~2-4 ms per pair
# against ~9 us, so the round-robin candidate set is first thinned by a
# CERTIFIED window rejection (`distmax_m=`, a separating-axis bound that
# proves a pair is outside the window): on the four RoboCasa matrix scenes
# that leaves 1-24 pairs actually solved out of 74-259 candidates, and a
# whole three-probe snapshot at ~0.1-0.7 s. That is three orders of magnitude
# above the old cost and it is affordable for the same reason the probe
# exists at all: it runs once, at a terminal event, off the actuation path.
_NEAREST_PROBE_DISTMAX_M = 0.10
_NEAREST_PROBE_MAX_CALLS = 4096
_NEAREST_PROBE_MAX_PAIRS = 32
# Voxel-backing probe: rays per cube axis, per side. The cube is sampled by
# three orthogonal `rays_per_axis**2` ray fans (one per base-frame axis), so
# the cost is `3 * n**2` `mj_ray` calls — 27 at the default, once, at a
# terminal event. A depth-derived occupancy cell is always backed by a
# *surface*, so surface sampling is the right test (see
# :func:`voxel_backing_record`).
_VOXEL_BACKING_RAYS_PER_AXIS = 3
# Ray start stand-off outside the cube face, as a fraction of the cell size.
# Large enough that a surface lying exactly on the face is still struck,
# small enough that geometry outside the cell is never attributed to it.
_VOXEL_BACKING_EPS_FRACTION = 0.01
# Cap on backing geoms reported per cell; a cell is 25 mm across and cannot
# plausibly be backed by more distinct geoms than this.
_VOXEL_BACKING_MAX_GEOMS = 8
# Cartesian dimensions — a grid size, an origin and a half-extent triple.
_XYZ = 3
# Emitted verbatim on every stop line. ``ncon`` is NOT a penetration oracle:
# MuJoCo contype/conaffinity exclusions can suppress contacts entirely
# (observed in the field — an arm 30 mm inside a freezer door with
# ``ncon == 0``), so the near-miss probe is the adjudicator.
_CONTACTS_CAVEAT = (
    "robot_world_contacts==0 does NOT mean no interpenetration: MuJoCo "
    "contype/conaffinity exclusions can suppress contacts entirely (field-"
    "observed: arm 30mm inside a freezer door with ncon==0). Adjudicate with "
    "the nearest_*_pairs probes, not with the contact list. Those probes "
    "measure only between SOLID geoms, on EVERY side — robot, payload and "
    "world (a geom with neither contype nor conaffinity is a marker or a "
    "visual shell, not an obstacle, and is excluded; the counts are in "
    "nearest_*_coverage.noncollidable_{world,side,other}_geoms_excluded); a "
    "pair whose bitmasks merely fail to meet each other is still measured, "
    "because suppression is a property of the pair, not of the geom."
)
# Candidate chunks retained for predicted-horizon reconstruction. The kernel
# checks the chunk it has just received; a small ring covers the delivery
# race between ``/openral/candidate_action`` and ``/openral/estop`` without
# unbounded growth.
_CANDIDATE_CHUNK_HISTORY = 3
# A collision evidence older than this is not attributed to the stop being
# snapshotted (0.5 s ≫ the kernel's publish-then-estop gap, ≪ a rollout).
_ESTOP_EVIDENCE_WINDOW_NS = 500_000_000

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from openral_core import RobotDescription

__all__ = [
    "SimSensorBridge",
    "candidate_chunk_digest",
    "collision_model_mesh_slop",
    "constant_scan_no_hit_ranges",
    "estop_ground_truth_snapshot",
    "initial_configuration_stop_record",
    "kernel_checked_body_ids",
    "should_idle_step",
    "voxel_backing_record",
]


def constant_scan_no_hit_ranges(*, n_beams: int, max_range_m: float) -> list[float]:
    """Return an ``n_beams``-long list with every beam clamped to ``max_range_m``.

    The synthetic ``/scan`` published when no live MuJoCo handle is bound (the
    in-process digital twin has no scene to ray-cast). slam_toolbox and Nav2
    both treat per-beam ``max_range`` as "no hit" rather than ``inf`` / ``NaN``,
    so this is the honest "nothing in front of me" reading. Pure (no rclpy) so
    it is unit-testable in isolation.

    Example:
        >>> constant_scan_no_hit_ranges(n_beams=3, max_range_m=12.0)
        [12.0, 12.0, 12.0]
    """
    return [float(max_range_m)] * int(n_beams)


def should_idle_step(
    now_ns: int,
    last_action_ns: int,
    idle_hold_ns: int,
    *,
    step_while_active: bool = False,
) -> bool:
    """Return True iff the sim-only idle stepper should advance the env now.

    Scene-attached environments yield to active skills: they step with a
    zero/HOLD action only when no real action arrived within the idle-hold
    window. Bare ``MujocoArmHAL`` opts into ``step_while_active`` because its
    ``send_action`` advances only one physics tick; the wall-time stepper must
    keep integrating the current control target or simulation time, camera
    timers, and ``/clock`` collapse during a rollout.

    Pure (no rclpy / no I/O) so it is unit-testable in isolation. The
    single-threaded rclpy executor guarantees the idle timer and
    ``_on_safe_action`` never run concurrently, so this timestamp comparison
    alone is a sufficient hand-off — no lock is needed.

    Args:
        now_ns: Current monotonic clock in nanoseconds (``time.monotonic_ns()``).
        last_action_ns: Monotonic ns of the last real action through the HAL's
            ``send_action`` (``SimAttachedHAL.last_action_ns``); ``0`` if none.
        idle_hold_ns: Quiet window in ns. A real action within this window of
            ``now_ns`` suppresses the idle tick.
        step_while_active: Keep stepping despite recent actions. Used only by
            bare MuJoCo arms whose action path does not advance wall time.

    Returns:
        ``True`` to idle-step now, ``False`` to yield to a recent action.

    Example:
        >>> should_idle_step(now_ns=1_000_000_000, last_action_ns=0, idle_hold_ns=200_000_000)
        True
        >>> should_idle_step(
        ...     now_ns=1_000_000_000, last_action_ns=950_000_000, idle_hold_ns=200_000_000
        ... )
        False
    """
    return step_while_active or now_ns - last_action_ns >= idle_hold_ns


def _obs_key_for_sensor(sensor: Any) -> str:
    """Key into ``read_images()`` for a manifest RGB sensor.

    Scenes key rendered frames by the VLA camera slot (``camera1``,
    ``camera2``, ...): LIBERO emits only those; robocasa emits them as
    aliases alongside the real camera name. So resolve the obs key from the
    sensor's ``vla_feature_key`` suffix (``observation.images.camera1`` ->
    ``camera1``), falling back to the sensor name (robocasa real-name keys, or
    sensors with no ``vla_feature_key``). The published topic stays
    ``/openral/cameras/<sensor.name>/image`` regardless.
    """
    vfk = getattr(sensor, "vla_feature_key", None)
    if vfk:
        return str(vfk).rsplit(".", 1)[-1]
    return str(sensor.name)


def _frame_for_camera(images: dict[str, Any], obs_key: str, name: str) -> Any:
    """Resolve a camera's frame from a ``read_images()`` dict, or ``None``.

    The two sim HALs key their frame dicts by different conventions:
    :class:`~openral_hal.sim_attached.SimAttachedHAL` (scene-attached LIBERO /
    robocasa) keys by the VLA slot (``obs_key`` — ``camera1`` / ``camera2``),
    while :class:`~openral_hal._mujoco_arm.MujocoArmHAL` (bare or composed
    digital twin) keys by the sensor ``name``. Try the slot first, then fall
    back to the name so both conventions resolve.

    Without the fallback, so101's slot ``camera1`` never matched its
    MujocoArmHAL frame keyed ``front`` and no frame ever published (issue #88);
    openarm was unaffected only because its sensor names equal their VLA slots.
    """
    arr = images.get(obs_key)
    if arr is None and name != obs_key:
        arr = images.get(name)
    return arr


def _optical_frame_rgb_cameras(sensors: Any) -> list[Any]:
    """RGB camera specs that own a dedicated ``*_optical_frame``.

    These are the cameras :meth:`SimSensorBridge._publish_camera_optical_tfs`
    broadcasts a live ``base_frame -> <camera>_optical_frame`` TF for, so the
    world-state object-lift can project the world voxel map into them. A camera
    whose ``frame_id`` is a robot link (e.g. an eye-in-hand at ``panda_hand``)
    already has TF from ``robot_state_publisher`` and is excluded. Pure (no
    rclpy / MuJoCo) so it is unit-testable in isolation.
    """
    return [
        s
        for s in sensors
        if getattr(s, "modality", None) == "rgb"
        and str(getattr(s, "frame_id", "")).endswith("_optical_frame")
    ]


def _body_record(model: Any, data: Any, body_id: int) -> dict[str, object]:
    """MJCF body id → ``{id, name, world_xyz}`` (world pose at this instant)."""
    import mujoco  # reason: optional sim dep

    return {
        "id": int(body_id),
        "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)),
        "world_xyz": [round(float(value), 6) for value in data.xpos[int(body_id)]],
    }


def _body_names(model: Any, body_ids: frozenset[int]) -> list[str]:
    """Sorted MJCF names for ``body_ids`` (unnamed bodies dropped)."""
    import mujoco  # reason: optional sim dep

    names = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) for body_id in body_ids
    )
    return sorted(str(name) for name in names if name)


def _contact_records(
    model: Any,
    data: Any,
    *,
    side: frozenset[int],
    other_excluded: frozenset[int],
) -> list[dict[str, object]]:
    """Live MuJoCo contacts crossing the ``side`` boundary, as JSON records.

    A contact is kept when exactly one of its two geoms sits on a body in
    ``side`` and the other body is not in ``other_excluded`` — i.e. the
    boundary the caller wants adjudicated (payload↔everything, or
    robot↔world once the payload bodies are excluded). Each record carries
    both geom names, both body names, both body world positions, the signed
    contact distance (negative = interpenetration, MuJoCo's own sign
    convention) and the contact point, so an offline tool can compare the
    kernel's evidence against the simulator's ground truth without the scene.

    The ``_a`` side is always the ``side`` member (MuJoCo orders a contact's
    geoms by id, which would otherwise put the payload / robot on either
    side depending on the scene's geom numbering).
    """
    import mujoco  # reason: optional sim dep

    records: list[dict[str, object]] = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom_a, geom_b = int(contact.geom1), int(contact.geom2)
        body_a = int(model.geom_bodyid[geom_a])
        body_b = int(model.geom_bodyid[geom_b])
        if (body_a in side) == (body_b in side):
            continue
        if body_b in side:  # normalise: ``side`` member first
            geom_a, geom_b = geom_b, geom_a
            body_a, body_b = body_b, body_a
        if body_b in other_excluded:
            continue
        records.append(
            {
                "distance_m": round(float(contact.dist), 6),
                "geom_a": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a),
                "geom_b": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b),
                "body_a": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_a),
                "body_b": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_b),
                "body_a_world_xyz": [round(float(v), 6) for v in data.xpos[body_a]],
                "body_b_world_xyz": [round(float(v), 6) for v in data.xpos[body_b]],
                "position_xyz": [round(float(v), 6) for v in contact.pos],
            }
        )
    return records


def _pair_distance_lower_bound(model: Any, data: Any, side_geoms: Any, other_geoms: Any) -> Any:
    """A **finite** lower bound on the true distance of every candidate pair.

    The bounding-sphere bound ``|c_a - c_b| - r_a - r_b`` needs both radii, and
    MuJoCo reports ``geom_rbound == 0`` for the geoms that have no bounding
    sphere at all: planes and heightfields. Treating those as radius ``inf``
    scores every pair involving one at ``-inf``, which sorts them ahead of
    every finite pair and lets a handful of scene planes consume the whole
    exact-distance budget (a RoboCasa kitchen ships four — the room floor and
    its backing, each with a ``_vis`` twin).

    A plane needs no sphere: its exact distance to another geom's bounding
    sphere is ``|n · (c - p)| - r``, where ``n`` is the plane's local +z in
    world. That is still a valid lower bound and it is finite, so a floor
    ranks on merit against a cabinet. Heightfields keep ``-inf`` (no cheap
    bound exists, and scenes carry at most a couple).

    Returns:
        ``(len(side_geoms), len(other_geoms))`` float64 lower bounds.
    """
    import mujoco  # reason: optional sim dep
    import numpy as np

    xpos = np.asarray(data.geom_xpos, dtype=np.float64)
    xmat = np.asarray(data.geom_xmat, dtype=np.float64).reshape(-1, 3, 3)
    gtype = np.asarray(model.geom_type, dtype=np.int64)
    raw_rbound = np.asarray(model.geom_rbound, dtype=np.float64)
    rbound = np.where(raw_rbound <= 0.0, np.inf, raw_rbound)
    centre_gap = np.linalg.norm(
        xpos[side_geoms][:, None, :] - xpos[other_geoms][None, :, :], axis=-1
    )
    gap = centre_gap - rbound[side_geoms][:, None] - rbound[other_geoms][None, :]

    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)

    def plane_bound(plane_ids: Any, point_ids: Any) -> Any:
        """``(len(plane_ids), len(point_ids))`` distances from planes to spheres."""
        # Plane surface normal = the geom frame's +z (third column of xmat).
        normals = xmat[plane_ids][:, :, 2]
        offsets = np.einsum("pk,pk->p", normals, xpos[plane_ids])
        signed = np.einsum("pk,ok->po", normals, xpos[point_ids]) - offsets[:, None]
        return np.abs(signed) - rbound[point_ids][None, :]

    side_planes = np.flatnonzero(gtype[side_geoms] == plane)
    other_planes = np.flatnonzero(gtype[other_geoms] == plane)
    if side_planes.size:
        gap[side_planes, :] = plane_bound(side_geoms[side_planes], other_geoms)
    if other_planes.size:
        gap[:, other_planes] = plane_bound(other_geoms[other_planes], side_geoms).T
    return gap


def _round_robin_candidates(gap: Any, distmax_m: float, max_calls: int) -> tuple[Any, int]:
    """Pick the exact-distance probe set, one side geom at a time.

    ``gap`` is the ``(n_side, n_other)`` bounding-sphere lower bound on the
    true geom distance. Ranking it *globally* and taking the first
    ``max_calls`` entries starves whole links: an unbounded geom
    (``geom_rbound == 0`` — every plane and heightfield in the scene) has no
    sphere, so every pair involving one scores ``-inf`` and sorts ahead of
    every finite pair. A RoboCasa kitchen has four such geoms (the room floor
    and its backing, each with a ``_vis`` twin) and a robosuite mobile Panda
    has ~70 geoms, so ~280 robot↔floor pairs monopolise a 256-call budget and
    the arm's genuine near-misses are never probed at all. The snapshot then
    reads "nothing was near link N" when the probe simply never looked —
    which is how a real stop gets adjudicated *false*.

    So spend the budget round-robin: every side geom contributes its own
    closest candidate before any side geom contributes its second, with the
    side geoms visited closest-first. Every robot geom is therefore probed at
    least once whenever ``max_calls >= n_side``, and the cost stays bounded by
    ``max_calls`` exactly as before.

    Args:
        gap: ``(n_side, n_other)`` float lower bounds (``-inf`` allowed).
        distmax_m: probe window; pairs above it are not candidates.
        max_calls: hard cap on exact distance solves.

    Returns:
        ``(pairs, n_candidates)`` — ``pairs`` is an ``(k, 2)`` int array of
        ``(row, col)`` indices into ``gap`` with ``k <= max_calls``, ordered
        by round-robin rank; ``n_candidates`` is how many pairs were within
        ``distmax_m`` in total, so the caller can report truncation.
    """
    import numpy as np

    valid = gap <= distmax_m
    per_row_counts = valid.sum(axis=1)
    n_candidates = int(per_row_counts.sum())
    if n_candidates == 0:
        return np.zeros((0, 2), dtype=np.int64), 0
    # Each row's candidates, closest-first; invalid entries sort to the end.
    row_order = np.argsort(np.where(valid, gap, np.inf), axis=1, kind="stable")
    # Visit the side geoms closest-first so a tie on rank still reports the
    # nearer link before the farther one.
    best_per_row = np.where(per_row_counts > 0, gap.min(axis=1), np.inf)
    rows = np.argsort(best_per_row, kind="stable")
    rows = rows[per_row_counts[rows] > 0]
    picked: list[tuple[int, int]] = []
    budget = min(int(max_calls), n_candidates)
    rank = 0
    while len(picked) < budget:
        for row in rows:
            if rank >= int(per_row_counts[row]):
                continue
            picked.append((int(row), int(row_order[row, rank])))
            if len(picked) >= budget:
                break
        rank += 1
    return np.asarray(picked, dtype=np.int64).reshape(-1, 2), n_candidates


def _nearest_pair_records(
    model: Any,
    data: Any,
    *,
    side: frozenset[int],
    other_excluded: frozenset[int] = frozenset(),
    other_included: frozenset[int] | None = None,
    distmax_m: float,
    max_pairs: int,
    max_calls: int = _NEAREST_PROBE_MAX_CALLS,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Closest signed geom distances across the ``side`` boundary.

    The safety kernel stops on a *margin*, so a genuine stop usually leaves
    NO MuJoCo contact at the measured configuration — ``ncon`` alone cannot
    say whether a ``-15 mm`` predicted hit was real (and contype/conaffinity
    exclusions can suppress the contact even at 30 mm of interpenetration).
    This measures the signed distance (negative = interpenetration) for the
    ``side``↔other geom pairs whose bounding spheres are within ``distmax_m``,
    ranked closest-first and truncated to ``max_pairs``.

    **The measurement is `convex_geom_distance`, not `mujoco.mj_geomDistance`.**
    That call returns confidently wrong numbers on precisely the pair class
    this probe exists to adjudicate — measured on
    ``robocasa_fridge_drawer`` layout 9, ``robot0_link7_collision`` vs
    ``fridge_right_group_freezer_door_main``: ``+0.000000`` from the default
    native-CCD path with a 126.264 mm witness segment lying outside *both*
    geoms, and ``-57 mm`` / ``-352 mm`` from libccd through a 48 mm panel,
    against a certified truth of ``+0.148512 mm``. It is a degenerate
    configuration rather than a distance regime — displacing the link by a
    picometre returns the right answer — so no probe window avoids it, and a
    scene's reset pose is where such configurations live. Every number this
    probe emits now carries ``distance_certified``; the coverage block counts
    both, so a downstream adjudicator can refuse rather than believe
    (``tools/validation_matrix.py::probe_is_distance_certified``).

    The other side is either an explicit body set (``other_included`` — used
    for payload↔robot-link self-pairs, which are not "everything else") or,
    by default, every body outside ``side`` and ``other_excluded``.

    **Every** side is restricted to solid geoms — one with neither ``contype``
    nor ``conaffinity`` cannot collide with anything and the kernel never checks
    it, so a distance measured against it is not a penetration. That rule used
    to apply to the enumerated world side only, which is how a purely visual
    mesh came to carry a stop: the 2026-08-23 fridge round reported
    ``robot0_g42_vis ~ fridge_main_group_g43`` at 0.000 m and was adjudicated
    ``real-contact``, while the nearest *solid* pair on the same link
    (``robot0_link7_collision``) was 2.5 mm clear. Both counts are reported —
    ``noncollidable_side_geoms_excluded`` and
    ``noncollidable_other_geoms_excluded`` — so the omission is visible, and
    their presence is what tells a downstream adjudicator that a 0 m pair here
    can be trusted at all.

    Bounded by construction: a vectorised distance-lower-bound prefilter
    (:func:`_pair_distance_lower_bound`) reduces the O(n·m) pair set, then at
    most ``max_calls`` exact distance calls run, shared fairly across the side
    geoms by :func:`_round_robin_candidates` so no link can be starved out of
    the report. Each of those is then offered a **certified** window
    rejection before it is solved — a separating-axis bound that *proves* the
    pair is outside ``distmax_m`` — which is what keeps the exact instrument
    affordable without weakening anything: a rejected pair was provably out of
    range, not heuristically dropped. Pure MuJoCo reads, no ROS.

    The prefilter is what a scene's floors used to defeat. MuJoCo reports
    ``geom_rbound == 0`` for the geoms that have no bounding sphere — planes
    and heightfields — and reading that as radius ``inf`` scored every pair
    involving one at ``-inf``, ahead of every finite pair. A plane is now
    bounded **exactly** (``|n · (c - p)| - r``), so a floor competes on real
    distance instead of pre-empting the queue: it is excluded from the
    candidate set outright when it is further than ``distmax_m``, and ranks
    on merit when it is not. Round-robin then bounds the residual: a
    heightfield still has no cheap bound and keeps ``-inf``, but it can cost
    each side geom only its first call, never the whole budget.

    Returns:
        ``(records, coverage)`` — ``records`` is the closest ``max_pairs``
        pairs; ``coverage`` reports how much of the candidate set the budget
        actually reached, so a *silent* omission can never be read as "nothing
        was near".
    """
    import mujoco  # reason: optional sim dep
    import numpy as np

    coverage: dict[str, object] = {
        "distmax_m": float(distmax_m),
        "candidate_pairs": 0,
        "probed_pairs": 0,
        "max_calls": int(max_calls),
        "truncated": False,
        "side_geoms": 0,
        "side_geoms_probed": 0,
        "noncollidable_world_geoms_excluded": 0,
        "noncollidable_side_geoms_excluded": 0,
        "noncollidable_other_geoms_excluded": 0,
        # Both are reported so an UNCERTIFIED distance can never be read as a
        # certified one by omission. `certified_pairs` counts what a verdict
        # may rest on; `uncertified_pairs` counts what it may not.
        "certified_pairs": 0,
        "uncertified_pairs": 0,
        "distance_instrument": "openral_hal.convex_distance.convex_geom_distance",
    }
    body_of_geom = np.asarray(model.geom_bodyid, dtype=np.int64)
    if body_of_geom.size == 0:
        return [], coverage
    # A geom with NEITHER contype NOR conaffinity is not solid: MuJoCo can never
    # generate a contact for it, and the safety kernel never checks one.
    # Measuring against it manufactures penetrations that mean nothing
    # physically — round 5/6 reported the payload "134 mm inside
    # cab_1_left_group_reg_main", a RoboCasa region marker, and the 2026-08-23
    # rounds called a stop `real-contact` on `robot0_g42_vis`, a visual shell.
    # The support producer already draws the same line for the same reason
    # (``_sim_attachment_evidence._support_candidate_geoms``: "purely visual
    # geometry ... is not solid, and a decorative shell coincident with a real
    # surface would only add noise"). It applies to BOTH sides of every probe:
    # a robot link body and an attached payload body each carry visual geoms
    # alongside their collision ones, so scoping a side by BODY does not scope
    # it to solid geometry. Pairs whose bitmasks merely fail to *meet* are still
    # measured — suppression there is a property of the pair, not of the geom,
    # and that is precisely the case the probe exists to adjudicate.
    collidable = (np.asarray(model.geom_contype, dtype=np.int64) != 0) | (
        np.asarray(model.geom_conaffinity, dtype=np.int64) != 0
    )
    in_side = np.isin(body_of_geom, np.fromiter(side, dtype=np.int64, count=len(side)))
    side_geoms = np.flatnonzero(in_side & collidable)
    coverage["noncollidable_side_geoms_excluded"] = int(np.count_nonzero(in_side & ~collidable))
    if other_included is not None:
        in_other = (
            np.isin(
                body_of_geom,
                np.fromiter(other_included, dtype=np.int64, count=len(other_included)),
            )
            & ~in_side
        )
    else:
        excluded = np.isin(
            body_of_geom,
            np.fromiter(other_excluded, dtype=np.int64, count=len(other_excluded)),
        )
        in_other = ~in_side & ~excluded
        coverage["noncollidable_world_geoms_excluded"] = int(
            np.count_nonzero(in_other & ~collidable)
        )
    other_geoms = np.flatnonzero(in_other & collidable)
    coverage["noncollidable_other_geoms_excluded"] = int(np.count_nonzero(in_other & ~collidable))
    coverage["side_geoms"] = int(side_geoms.size)
    if side_geoms.size == 0 or other_geoms.size == 0:
        return [], coverage
    gap = _pair_distance_lower_bound(model, data, side_geoms, other_geoms)
    candidates, n_candidates = _round_robin_candidates(gap, distmax_m, max_calls)
    coverage["candidate_pairs"] = n_candidates
    coverage["probed_pairs"] = int(candidates.shape[0])
    coverage["truncated"] = bool(n_candidates > candidates.shape[0])
    coverage["side_geoms_probed"] = int(np.unique(candidates[:, 0]).size)
    if candidates.size == 0:
        return [], coverage
    probed: list[tuple[float, int, int, ConvexDistance]] = []
    uncertified = 0
    for row, col in candidates:
        geom_side = int(side_geoms[int(row)])
        geom_other = int(other_geoms[int(col)])
        measured = convex_geom_distance(
            model, data, geom_side, geom_other, distmax_m=float(distmax_m)
        )
        if measured.method == "beyond-window":
            continue  # PROVABLY nothing within the probe window for this pair
        if not measured.certified:
            uncertified += 1
        probed.append((measured.distance_m, geom_side, geom_other, measured))
    probed.sort(key=lambda item: item[0])
    coverage["certified_pairs"] = int(len(probed) - uncertified)
    coverage["uncertified_pairs"] = int(uncertified)
    records = [
        {
            **measured.as_record(),
            "geom_a": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_side),
            "geom_b": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_other),
            "body_a": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_side])
            ),
            "body_b": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_other])
            ),
        }
        for _distance, geom_side, geom_other, measured in probed[:max_pairs]
    ]
    return records, coverage


def kernel_checked_body_ids(model: Any, description: Any) -> frozenset[int]:
    """MJCF bodies for exactly the links the safety kernel collision-checks.

    The manifest's ``collision_geometry`` **is** the kernel's collision model:
    a link with no entry is deliberately invisible to the check. On
    ``panda_mobile`` that exempts ``base_link`` (the base parks ~1 cm from
    cabinets; base-vs-world is Nav2's costmap job) and ``panda_finger_pair``
    (the gripper is the intended-contact part).

    Scoping the near-miss probe to this set is what makes a stop record
    readable on a mobile manipulator: probing the *whole* robot ranks the
    wheels' 0-2 mm floor contact above everything and buries the arm — which
    in the field very nearly produced a wrong verdict on a stop where the arm
    was 17-30 mm inside a freezer door.

    Resolution is exact, not name-mangled: a link is the MJCF body carrying
    the joint whose ``child_link`` names it, looked up through that joint's
    ``sim_joint_name`` (robosuite's ``robot0_joint7`` for ``panda_link7``),
    falling back to a body of the link's own name for jointless links.
    Returns an empty set when the manifest declares no collision geometry —
    the caller then has no kernel scope to honour and must say so.
    """
    return frozenset(kernel_checked_link_bodies(model, description).values())


def kernel_checked_link_bodies(model: Any, description: Any) -> dict[str, int]:
    """Map each kernel-checked link name to its MJCF body id.

    The name-keyed form of :func:`kernel_checked_body_ids` — needed wherever a
    diagnostic has to line a manifest ``collision_geometry`` entry up with the
    MuJoCo geometry it is supposed to enclose (see
    :func:`collision_model_mesh_slop`). Links the model does not carry are
    simply absent, so the caller can report the shortfall rather than guess.
    """
    import mujoco  # reason: optional sim dep

    links = {g.link_name for g in getattr(description, "collision_geometry", [])}
    if not links:
        return {}
    out: dict[str, int] = {}
    for joint in getattr(description, "joints", []):
        if joint.child_link not in links or joint.child_link in out:
            continue
        jid = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.sim_joint_name or joint.name)
        )
        if jid >= 0:
            out[joint.child_link] = int(model.jnt_bodyid[jid])
    for link in links:  # jointless (welded) links, if the MJCF names them directly
        if link in out:
            continue
        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link))
        if bid >= 0:
            out[link] = bid
    return out


def _body_collision_points(model: Any, body_id: int) -> Any:
    """Sampled points on a body's SOLID geometry, in body-local coordinates.

    Mesh geoms contribute their vertices; primitives contribute their bounding
    box corners, which *overstate* the primitive's extent and therefore
    understate the slop computed from them — the safe direction for a budget
    (see :func:`collision_model_mesh_slop`). Visual-only geoms (neither
    ``contype`` nor ``conaffinity``) are skipped: the kernel's OBB is
    documented as enclosing the *collision* mesh.
    """
    import mujoco  # reason: optional sim dep
    import numpy as np

    chunks: list[Any] = []
    for geom in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom]) != body_id:
            continue
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
            continue
        flat = np.zeros(9)
        mujoco.mju_quat2Mat(flat, model.geom_quat[geom])
        rot = flat.reshape(3, 3)
        pos = np.asarray(model.geom_pos[geom], dtype=np.float64)
        if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[geom])
            start = int(model.mesh_vertadr[mesh])
            count = int(model.mesh_vertnum[mesh])
            verts = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
            chunks.append((verts.reshape(-1, 3) @ rot.T) + pos)
        else:
            size = np.asarray(model.geom_size[geom], dtype=np.float64)
            reach = float(np.max(size)) if size.size else 0.0
            corners = np.array(
                [
                    [sx, sy, sz]
                    for sx in (-reach, reach)
                    for sy in (-reach, reach)
                    for sz in (-reach, reach)
                ],
                dtype=np.float64,
            )
            chunks.append((corners @ rot.T) + pos)
    return np.vstack(chunks) if chunks else np.zeros((0, 3))


def collision_model_mesh_slop(model: Any, description: Any) -> dict[str, object]:
    """How far the kernel's collision model reaches beyond the real meshes.

    **The number every world-voxel adjudication needs, and the one the
    2026-08-22 round did not have.** The safety kernel checks manifest
    ``collision_geometry`` OBBs against occupancy voxels; the near-miss probe
    (:func:`estop_ground_truth_snapshot`) measures MuJoCo *mesh* against MuJoCo
    *mesh*. Those are different quantities, and subtracting one from the other
    without this term makes a legitimate conservative stop look like a false
    positive.

    A box around a rounded link is tight on its **faces** and loose at its
    **corners** — necessarily, not by sloppy authoring. Measured against
    ``panda_mj_description`` under mujoco 3.8.0, ``panda_mobile``'s OBBs are
    sub-millimetre on every face and 23-88 mm out at the corners. That
    completes, rather than contradicts, the mesh verification that withdrew
    the "OBBs are too big" hypothesis: both are true, of different questions.

    The admissible gap between a kernel distance and a probe distance is
    therefore ``corner_slop(link) + voxel_half_diagonal``; only a gap wider
    than that is unexplained by the collision model's own conservatism.

    Args:
        model: live ``mujoco.MjModel``.
        description: the ``RobotDescription`` whose ``collision_geometry`` is
            the kernel's collision model.

    Returns:
        ``{"links": {<link>: {...}}, "max_corner_slop_m": float,
        "unresolved_links": [...], "method": str}``. Each link entry carries
        ``obb_half_extents_m``, ``face_slop_m`` (per axis) and
        ``corner_slop_m`` — the largest distance from an OBB corner to the
        nearest sampled collision point. ``{}`` when the manifest declares no
        collision geometry, so the caller reports "no budget" rather than
        assuming zero.

    Example:
        >>> # slop = collision_model_mesh_slop(model, description)
        >>> # slop["max_corner_slop_m"]  # e.g. 0.0559 on panda_mobile
    """
    import numpy as np

    entries = list(getattr(description, "collision_geometry", []) or [])
    if not entries:
        return {}
    bodies = kernel_checked_link_bodies(model, description)
    links: dict[str, object] = {}
    unresolved: list[str] = []
    worst = 0.0
    for entry in entries:
        name = str(entry.link_name)
        body_id = bodies.get(name)
        half = np.asarray(entry.shape.half_extents_m, dtype=np.float64)
        if body_id is None or half.size != _XYZ:
            unresolved.append(name)
            continue
        points = _body_collision_points(model, int(body_id))
        if points.shape[0] == 0:
            unresolved.append(name)
            continue
        origin = np.asarray(entry.origin_xyz_rpy, dtype=np.float64)
        rot = _rpy_to_matrix(float(origin[3]), float(origin[4]), float(origin[5]))
        local = (points - origin[:3]) @ rot  # collision points in the OBB frame
        extent = np.abs(local).max(axis=0)
        corners = np.array(
            [
                [sx * half[0], sy * half[1], sz * half[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        corner_slop = float(
            max(float(np.min(np.linalg.norm(local - corner, axis=1))) for corner in corners)
        )
        worst = max(worst, corner_slop)
        links[name] = {
            "obb_half_extents_m": [round(float(v), 6) for v in half],
            "face_slop_m": [round(float(v), 6) for v in (half - extent)],
            "corner_slop_m": round(corner_slop, 6),
            "collision_points_sampled": int(points.shape[0]),
        }
    return {
        "links": links,
        "max_corner_slop_m": round(worst, 6),
        "unresolved_links": sorted(unresolved),
        "method": (
            "OBB corner -> nearest sampled collision point (mesh vertices; "
            "primitives sampled at their bounding-box corners, which understates "
            "the slop). An upper bound on corner-to-surface distance, so a gap "
            "WIDER than corner_slop + voxel half-diagonal is unexplained by "
            "collision-model conservatism."
        ),
    }


def _payload_collision_points(model: Any, data: Any, root_body_id: int) -> Any:
    """Sampled points on a payload subtree's SOLID geometry, in the root frame.

    The payload counterpart of :func:`_body_collision_points`, and deliberately
    *not* the same sampling. That one samples every primitive at the corners of
    a cube of its largest half-size, which overstates the geometry and so
    understates the robot-side slop — the safe direction there, because the
    robot's OBBs come from the manifest and are only ever loose. Here the
    measured quantity is the producer's own lowering, so an overstated point
    set would charge an exactly-lowered box geom slop it does not have. Every
    point below therefore lies ON the real surface: mesh vertices, a box's
    corners, a sphere's axis extrema, a capsule's cap poles and equator.

    It reads the *live* ``data`` rather than the model defaults because a
    payload subtree can articulate, and it expresses every point in the ROOT
    body frame — the frame ``extract_body_primitives`` lowers
    ``pose_in_object`` into, so the two are directly comparable.
    """
    import numpy as np

    from openral_hal._sim_attachment_evidence import _body_subtree

    subtree = _body_subtree(model, root_body_id)
    root_rot = np.asarray(data.xmat[root_body_id], dtype=np.float64).reshape(3, 3)
    root_pos = np.asarray(data.xpos[root_body_id], dtype=np.float64)
    chunks: list[Any] = []
    for geom in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom]) not in subtree:
            continue
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
            continue
        rot = root_rot.T @ np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
        pos = root_rot.T @ (np.asarray(data.geom_xpos[geom], dtype=np.float64) - root_pos)
        local = _geom_surface_points(model, geom)
        if local.shape[0]:
            chunks.append((local @ rot.T) + pos)
    return np.vstack(chunks) if chunks else np.zeros((0, 3))


def _geom_surface_points(model: Any, geom: int) -> Any:
    """Points lying on one geom's own surface, in the geom's local frame."""
    import mujoco  # reason: optional sim dep
    import numpy as np

    kind = int(model.geom_type[geom])
    size = np.asarray(model.geom_size[geom], dtype=np.float64)
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        return np.asarray(model.mesh_vert[start : start + count], dtype=np.float64).reshape(-1, 3)
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return np.array(
            [
                [sx * size[0], sy * size[1], sz * size[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        r = float(size[0])
        return np.array(
            [[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]],
            dtype=np.float64,
        )
    if kind in {int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)}:
        r = float(size[0])
        half = float(size[1])
        # A cylinder's rim is on its surface; a capsule's pole sits r beyond
        # the segment end. Use the shape's own extent so neither is overstated.
        pole = half + r if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else half
        return np.array(
            [
                [r, 0, -half],
                [-r, 0, -half],
                [0, r, -half],
                [0, -r, -half],
                [r, 0, half],
                [-r, 0, half],
                [0, r, half],
                [0, -r, half],
                [0, 0, pole],
                [0, 0, -pole],
            ],
            dtype=np.float64,
        )
    return np.zeros((0, 3))


def attached_payload_mesh_slop(
    model: Any,
    data: Any,
    *,
    attached_body_ids: frozenset[int],
    max_primitives: int = 16,
) -> dict[str, object]:
    """How far a carried payload's kernel primitives reach beyond its meshes.

    **The payload-side half of the adjudication budget, and the term the
    2026-08-22 attached-payload round did not have.** For a world-voxel stop
    only the robot link is an OBB — the other side is a voxel cube, and
    :func:`collision_model_mesh_slop` plus the cell half-diagonal covers it.
    An *attached-payload self-collision* stop has an OBB on **both** sides:
    the kernel checks the payload's published primitives against the link
    OBBs (``check_attached_self_collision``), while the ground-truth probe
    still measures mesh against mesh. Charging only the link's corner slop
    therefore under-counts the admissible gap by the payload's own.

    That under-count is not hypothetical. In the 2026-08-22 ``baguette``
    round the kernel stopped on ``attached:sim:obj_main`` vs ``panda_link2``
    at -4.63 mm while the probe put the nearest payload mesh 75.86 mm from
    the same link. The kernel's own arithmetic at the measured configuration
    puts that pair at +21.71 mm — a 54.15 mm representation gap, of which
    ``panda_link2``'s 48.22 mm corner slop is only the robot's share.

    The payload's primitives come from :func:`extract_body_primitives`, which
    lowers a *mesh* geom to its local AABB (and clusters geoms once there are
    more than ``max_primitives``). Both inflate; a sphere/box geom lowers
    exactly and contributes nothing. Rather than re-deriving that lowering,
    this calls the producer and measures what it actually publishes, so the
    budget cannot drift from the geometry the kernel was handed.

    Args:
        model: live ``mujoco.MjModel``.
        data: live ``mujoco.MjData`` — a payload subtree can articulate, so
            the primitives and the sampled points are both taken live.
        attached_body_ids: root body ids of the currently carried payloads.
        max_primitives: the producer's per-object primitive cap, mirrored
            here so the measurement matches the published decomposition.

    Returns:
        ``{"objects": {<body>: {...}}, "max_corner_slop_m": float,
        "unresolved_objects": [...], "method": str}``, or ``{}`` when nothing
        is carried — so the caller reports "no budget" rather than assuming
        zero. Each object entry carries ``n_primitives``,
        ``n_box_primitives`` (the subset that can be loose at all),
        ``corner_slop_m`` and ``collision_points_sampled``.

    Example:
        >>> # slop = attached_payload_mesh_slop(
        >>> #     model, data, attached_body_ids=frozenset({380})
        >>> # )
        >>> # slop["max_corner_slop_m"]  # payload's share of the budget
    """
    import numpy as np

    from openral_hal._sim_attachment_evidence import extract_body_primitives

    if not attached_body_ids:
        return {}
    objects: dict[str, object] = {}
    unresolved: list[str] = []
    worst = 0.0
    for body_id in sorted(attached_body_ids):
        named = _body_names(model, frozenset({int(body_id)}))
        name = named[0] if named else f"body_{int(body_id)}"
        points = _payload_collision_points(model, data, int(body_id))
        if points.shape[0] == 0:
            unresolved.append(name)
            continue
        prims = extract_body_primitives(
            model,
            data,
            root_body_id=int(body_id),
            object_id=name,
            max_primitives=max_primitives,
        )
        if not prims:
            unresolved.append(name)
            continue
        corner_slop = 0.0
        boxes = 0
        for prim in prims:
            # Only BOX primitives can be loose: a sphere/capsule geom lowers to
            # a sphere/capsule the kernel checks as such, with no corner to
            # overhang. Boxes are where the mesh AABB and the cluster merge
            # land, so they carry the whole payload-side term.
            half_extents = getattr(prim.shape, "half_extents_m", None)
            if half_extents is None:
                continue
            boxes += 1
            half = np.asarray(half_extents, dtype=np.float64)
            rot = _quat_xyzw_to_matrix(prim.pose_in_object.quat_xyzw)
            centre = np.asarray(prim.pose_in_object.xyz, dtype=np.float64)
            local = (points - centre) @ rot  # payload points in the primitive frame
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corner = np.array([sx * half[0], sy * half[1], sz * half[2]])
                        corner_slop = max(
                            corner_slop, float(np.min(np.linalg.norm(local - corner, axis=1)))
                        )
        worst = max(worst, corner_slop)
        objects[name] = {
            "n_primitives": len(prims),
            "n_box_primitives": boxes,
            "corner_slop_m": round(corner_slop, 6),
            "collision_points_sampled": int(points.shape[0]),
        }
    return {
        "objects": objects,
        "max_corner_slop_m": round(worst, 6),
        "unresolved_objects": sorted(unresolved),
        "method": (
            "published primitive corner -> nearest sampled payload collision "
            "point, over the primitives extract_body_primitives actually "
            "publishes (mesh geoms lower to their local AABB; clustering above "
            "the primitive cap inflates further; sphere/box geoms lower "
            "exactly and contribute 0). The payload's share of the "
            "attached-payload self-collision budget."
        ),
    }


def _quat_xyzw_to_matrix(quat_xyzw: Any) -> Any:
    """Unit quaternion ``(x, y, z, w)`` to a 3x3 rotation matrix."""
    import numpy as np

    qx, qy, qz, qw = (float(v) for v in quat_xyzw)
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm < _DEGENERATE_QUAT_NORM:
        return np.eye(3)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> Any:
    """Fixed-axis roll-pitch-yaw to a 3x3 rotation matrix (``R = Rz Ry Rx``)."""
    import numpy as np

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _voxel_cube_hits(
    model: Any,
    data: Any,
    *,
    centre_world: Any,
    rot: Any,
    resolution: float,
    rays_per_axis: int,
) -> tuple[list[int], int, int]:
    """Geoms whose surface passes through one cell cube, by ray sampling.

    Three orthogonal fans, one per cube axis, each ``rays_per_axis**2`` rays
    started one stand-off outside a face. A strike counts only while it lies
    inside the cube, so geometry merely *behind* the cell is never attributed
    to it. Returns ``(geom_ids, rays_cast, rays_hit)`` — the counts are the
    coverage attestation that keeps "nothing found" distinct from "did not
    look".
    """
    import mujoco  # reason: optional sim dep
    import numpy as np

    half = resolution / 2.0
    eps = resolution * _VOXEL_BACKING_EPS_FRACTION
    span = 2.0 * half + eps  # strikes past the far face are not this cell's
    n = max(int(rays_per_axis), 1)
    taps = [(k + 0.5) / n * 2.0 * half - half for k in range(n)]
    geomid = np.zeros(1, dtype=np.int32)
    hits: dict[int, None] = {}
    cast = 0
    hit_count = 0
    for axis in range(3):
        direction = np.ascontiguousarray(rot[:, axis])
        u = rot[:, (axis + 1) % 3]
        v = rot[:, (axis + 2) % 3]
        for du in taps:
            for dv in taps:
                start = np.ascontiguousarray(
                    centre_world - direction * (half + eps) + u * du + v * dv
                )
                geomid[0] = -1
                distance = float(mujoco.mj_ray(model, data, start, direction, None, 1, -1, geomid))
                cast += 1
                if geomid[0] >= 0 and 0.0 <= distance <= span:
                    hit_count += 1
                    hits.setdefault(int(geomid[0]), None)
    return sorted(hits), cast, hit_count


def _classify_voxel_hits(
    model: Any,
    geom_ids: list[int],
    *,
    robot_body_ids: frozenset[int],
    attached_body_ids: frozenset[int],
) -> list[dict[str, object]]:
    """Name each backing geom and say which side of the map/world line it sits on."""
    import mujoco  # reason: optional sim dep

    out: list[dict[str, object]] = []
    for geom in geom_ids[:_VOXEL_BACKING_MAX_GEOMS]:
        body = int(model.geom_bodyid[geom])
        collidable = int(model.geom_contype[geom]) != 0 or int(model.geom_conaffinity[geom]) != 0
        if body in attached_body_ids:
            kind = "attached_payload"
        elif body in robot_body_ids:
            kind = "self_occupancy_suspect"
        elif collidable:
            kind = "solid_world"
        else:
            kind = "noncollidable_world"
        out.append(
            {
                "class": kind,
                "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                "collidable": bool(collidable),
            }
        )
    return out


def voxel_backing_record(
    model: Any,
    data: Any,
    *,
    voxel_index: int,
    grid_origin: Sequence[float],
    grid_resolution: float,
    grid_size: Sequence[int],
    robot_body_ids: frozenset[int],
    attached_body_ids: frozenset[int] = frozenset(),
    base_frame_body: str | None = None,
    rays_per_axis: int = _VOXEL_BACKING_RAYS_PER_AXIS,
) -> dict[str, object]:
    """What MuJoCo geometry, if any, backs one occupancy voxel.

    **The question a world-voxel stop turns on, and the one nothing in the
    stack could answer.** The kernel stops on a cell in
    ``/openral/world_voxels``; the near-miss probes measure MuJoCo against
    MuJoCo and never look at the map at all. So a stop could be adjudicated
    "nothing was there" when the truth was "the map and the world disagree" —
    and the probe's two deliberate blind spots (it excludes every robot body
    from its world side, and every non-collidable geom) are exactly where such
    a disagreement hides.

    This locates the cell and asks MuJoCo directly, with no exclusions:

    * ``solid_world`` — a collidable world geom passes through the cell. The
      stop is explained by real geometry.
    * ``attached_payload`` — the carried object; it should have been cleared
      from world occupancy by the bridge.
    * ``self_occupancy_suspect`` — **the robot's own body**, base and mount
      included. ``suspect`` is the whole of the claim: this says a robot body
      is in the cell *now*, which is equally the signature of a correct stop on
      a link that has reached real geometry. It cannot distinguish "the robot
      wrote this cell" from "the robot has since moved into it" — on the
      2026-08-23 fridge reconstruction the backing body was ``robot0_link7``,
      the stopping link, 1.9 mm inside a freezer door. Read it as a prompt to
      check the near-miss pairs, never as a finding on its own. What a hit does
      rule out is that the stop was on nothing at all.
    * ``noncollidable_world`` — a marker/visual geom. ``mj_ray`` strikes these
      and so does the depth synth, so they *can* become occupancy; the
      near-miss probe deliberately never measures them.
    * ``unbacked`` — nothing at all. A phantom or stale cell.

    Method: three orthogonal ray fans, one per base-frame cube axis, each
    ``rays_per_axis**2`` rays started just outside one face and accepted only
    where the strike lies inside the cube. A depth-derived occupancy cell is
    created by a *surface* return, so surface sampling is the matching test —
    a cell buried strictly inside a solid could not have been written by the
    depth path and is not sought. ``rays_cast``/``rays_hit`` are reported so
    ``unbacked`` reads as "looked and found nothing", never "did not look".

    Args:
        model: live ``mujoco.MjModel``.
        data: live ``mujoco.MjData`` at the stop instant.
        voxel_index: the kernel's world-obstacle index — the integer in a
            ``safety.collision`` line's ``b=voxel_<n>``. Row-major with x
            fastest, matching ``OccupancyVoxels``.
        grid_origin: base-frame position of voxel ``(0,0,0)``'s minimum corner.
        grid_resolution: cell size in metres.
        grid_size: ``(size_x, size_y, size_z)``.
        robot_body_ids: the robot's own MJCF bodies (the depth self-filter
            set), which is what makes ``self_occupancy_suspect`` cover the base
            and mount the world-side probe excludes.
        attached_body_ids: currently carried payload bodies.
        base_frame_body: MJCF body the grid frame denotes. ``None`` treats the
            grid as already world-aligned (synthetic/world-frame grids).
        rays_per_axis: rays per side of each face fan.

    Returns:
        A JSON-safe dict with ``voxel_index``, ``voxel_ijk``, ``base_xyz``,
        ``world_xyz``, ``resolution_m``, ``half_diagonal_m``, ``verdict``,
        ``classes``, ``backing`` (geom/body/class records), ``rays_cast``,
        ``rays_hit`` and ``method``. ``verdict`` is ``"out_of_range"`` when the
        index does not address a cell of this grid.

    Example:
        >>> # voxel_backing_record(model, data, voxel_index=76001,
        >>> #     grid_origin=(-0.8, -0.8, -0.3), grid_resolution=0.025,
        >>> #     grid_size=(64, 64, 64), robot_body_ids=self_bodies)
    """
    import mujoco  # reason: optional sim dep
    import numpy as np

    size = [int(v) for v in grid_size]
    resolution = float(grid_resolution)
    record: dict[str, object] = {
        "voxel_index": int(voxel_index),
        "resolution_m": round(resolution, 6),
        "half_diagonal_m": round(resolution * float(np.sqrt(3.0)) / 2.0, 6),
        "rays_cast": 0,
        "rays_hit": 0,
        "backing": [],
        "classes": [],
        "method": (
            "three orthogonal ray fans across the cell cube; a strike is "
            "attributed only when it lies inside the cube. No geom class is "
            "excluded: robot bodies and non-collidable markers are reported, "
            "not filtered."
        ),
    }
    total = size[0] * size[1] * size[2] if len(size) == _XYZ else 0
    if total <= 0 or resolution <= 0.0 or not 0 <= int(voxel_index) < total:
        record["verdict"] = "out_of_range"
        return record
    index = int(voxel_index)
    ix = index % size[0]
    iy = (index // size[0]) % size[1]
    iz = index // (size[0] * size[1])
    origin = np.asarray(list(grid_origin), dtype=np.float64)
    centre_base = origin + (np.array([ix, iy, iz], dtype=np.float64) + 0.5) * resolution
    rot = np.eye(3, dtype=np.float64)
    offset: Any = np.zeros(3, dtype=np.float64)
    if base_frame_body is not None:
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_frame_body))
        if body_id >= 0:
            rot = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            offset = np.asarray(data.xpos[body_id], dtype=np.float64)
    centre_world = rot @ centre_base + offset
    record["voxel_ijk"] = [int(ix), int(iy), int(iz)]
    record["base_xyz"] = [round(float(v), 6) for v in centre_base]
    record["world_xyz"] = [round(float(v), 6) for v in centre_world]

    hits, cast, hit_count = _voxel_cube_hits(
        model,
        data,
        centre_world=centre_world,
        rot=rot,
        resolution=resolution,
        rays_per_axis=rays_per_axis,
    )
    record["rays_cast"] = cast
    record["rays_hit"] = hit_count

    backing = _classify_voxel_hits(
        model,
        hits,
        robot_body_ids=robot_body_ids,
        attached_body_ids=attached_body_ids,
    )
    classes = {str(entry["class"]) for entry in backing}
    record["backing"] = backing
    record["classes"] = sorted(classes)
    # Precedence is adjudication order, not severity: real geometry EXPLAINS the
    # stop and outranks everything; only when nothing solid backs the cell does
    # the map's disagreement with the world become the headline.
    for kind in (
        "solid_world",
        "attached_payload",
        "self_occupancy_suspect",
        "noncollidable_world",
    ):
        if kind in classes:
            record["verdict"] = kind
            break
    else:
        record["verdict"] = "unbacked"
    return record


def estop_ground_truth_snapshot(
    model: Any,
    data: Any,
    *,
    robot_body_ids: frozenset[int],
    attached_body_ids: frozenset[int] = frozenset(),
    probe_body_ids: frozenset[int] | None = None,
    base_frame_body: str | None = None,
    joint_state: Any = None,
    description: Any = None,
    evidence_voxel: Any = None,
    distmax_m: float = _NEAREST_PROBE_DISTMAX_M,
    max_pairs: int = _NEAREST_PROBE_MAX_PAIRS,
    max_calls: int = _NEAREST_PROBE_MAX_CALLS,
) -> dict[str, object]:
    """MuJoCo ground truth for one safety stop, attached payload or not.

    Every kernel E-stop gets one of these (CLAUDE.md §1.4): without it a stop
    cannot be adjudicated real-vs-false after the fact. The payload sections
    are populated only when something is carried; the robot↔world sections
    are always populated, which is what a PRE-GRASP arm↔world stop needs (the
    2026-08-13 post-fix matrix had 3 of 4 stops in that class and zero ground
    truth for them).

    **The contact lists are not a penetration oracle.** MuJoCo
    contype/conaffinity exclusions can suppress a contact entirely — the
    field round saw an arm 30 mm inside a freezer door with ``ncon == 0``.
    An empty ``robot_world_contacts`` therefore means "MuJoCo reported no
    contact", never "nothing is interpenetrating"; the ``nearest_*_pairs``
    probes are the adjudicator, and the record carries this as
    ``contacts_caveat``.

    **The probes measure only between solid geoms, on every side.** A geom with
    neither ``contype`` nor ``conaffinity`` cannot collide with anything and is
    never checked by the safety kernel, so a signed distance against one is not
    a penetration — rounds 5/6 reported the payload "134 mm inside
    ``cab_1_left_group_reg_main``", a RoboCasa region marker, and the 2026-08-23
    fridge round reported ``robot0_g42_vis`` (a visual shell) touching the
    freezer door at 0.000 m while the nearest solid pair on the same link was
    2.5 mm clear. The filter therefore covers the robot and payload sides too,
    not just the enumerated world side: scoping a probe side by *body* does not
    scope it to solid geometry, because a link body carries its visual meshes
    alongside its collision geom. It is the same rule the support-contact
    producer applies when enumerating support candidates
    (``_sim_attachment_evidence._support_candidate_geoms``). Each probe's
    coverage block reports ``noncollidable_world_geoms_excluded``,
    ``noncollidable_side_geoms_excluded`` and
    ``noncollidable_other_geoms_excluded``, so the omission is visible rather
    than silent — and the presence of those keys is what lets a downstream
    adjudicator tell a trustworthy 0 m pair from one recorded before the filter
    existed. Pairs whose bitmasks merely fail to *meet* are still measured:
    suppression there is a property of the pair, not of the geom, and that is
    precisely the case the probe exists to adjudicate.

    Args:
        model: live ``mujoco.MjModel``.
        data: live ``mujoco.MjData`` at the stop instant.
        robot_body_ids: the robot's own MJCF body ids (the depth self-filter
            set — derived from the manifest joint prefixes). Scopes the
            contact lists and the payload↔world exclusion.
        attached_body_ids: currently carried payload body ids (empty when
            nothing is attached).
        probe_body_ids: robot bodies the near-miss probes may rank — the
            kernel-checked links from :func:`kernel_checked_body_ids`.
            ``None`` falls back to ``robot_body_ids`` and is reported as
            ``probe_robot_scope: "all_robot_bodies"``, which on a mobile base
            lets wheel↔floor pairs (0-2 mm, and deliberately unchecked by the
            kernel) crowd out the arm.
        base_frame_body: MJCF body that ``base_frame`` denotes on ``/tf``.
            The kernel's collision FK is base-relative, so its world pose is
            what maps a reconstructed configuration back into MuJoCo world
            coordinates.
        joint_state: the HAL's :class:`~openral_core.JointState` at the stop
            (the same vector the kernel seeded ``q_meas`` from), or ``None``.
        description: the ``RobotDescription`` whose ``collision_geometry`` is
            the kernel's collision model. Supplying it turns on
            ``adjudication_budget`` (:func:`collision_model_mesh_slop`) and
            widens the probe window to that budget.
        evidence_voxel: the world-voxel cell the kernel stopped on, as
            ``{"index": int, "origin": (x, y, z), "resolution": float,
            "size": (nx, ny, nz)}`` (optionally ``"frame_body"``). Supplying it
            turns on ``evidence_voxel_backing``
            (:func:`voxel_backing_record`) — the only part of this record that
            looks at the MAP rather than at MuJoCo alone.
        distmax_m: near-miss probe window *floor*. The window actually used is
            ``max(distmax_m, admissible_gap_m)`` and is reported as
            ``adjudication_budget.probe_distmax_used_m``.
        max_pairs: cap on reported nearest pairs, per probe.
        max_calls: cap on exact distance solves per probe. The
            prefilter ranks candidates by proximity first and spends the
            budget round-robin across the probed geoms, so this truncates the
            far end of each probe, never the close one, and never at the cost
            of leaving a geom unprobed while the budget covers the geom count.

    Returns:
        A JSON-safe dict: ``stop_class`` (``"attached_payload"`` when a
        payload is carried, else ``"robot_world"``), ``sim_time_s``,
        ``contacts_caveat``, ``attached_bodies``, ``payload_contacts``,
        ``robot_world_contacts``, ``nearest_robot_world_pairs``,
        ``nearest_probe_coverage``, ``nearest_payload_world_pairs``,
        ``nearest_payload_world_coverage``, ``nearest_payload_robot_pairs``,
        ``nearest_payload_robot_coverage``, ``probe_robot_scope``,
        ``probe_excluded_robot_bodies``, ``robot_joint_state``,
        ``base_frame_tf``, ``adjudication_budget`` and
        ``evidence_voxel_backing``.

        ``adjudication_budget`` is what makes a kernel number and a probe
        number comparable at all: the kernel measures OBB-to-voxel, the probes
        measure mesh-to-mesh, and the admissible gap between them is
        ``corner_slop(link) + voxel_half_diagonal``. The 2026-08-22 round
        subtracted the two directly with only the quantization term and called
        a conservative, correct ``panda_link1`` stop a false positive.

        ``evidence_voxel_backing`` is the map-side half of the record — what
        MuJoCo actually has at the cell the kernel stopped on, with no geom
        class excluded. It is ``None`` when no evidence voxel was supplied.

        Each probe carries its own coverage block, and that is what makes an
        *absent* pair readable: an empty near-miss list only means "nothing
        was close" when ``truncated`` is false and
        ``side_geoms_probed == side_geoms``.
        ``noncollidable_world_geoms_excluded`` /
        ``noncollidable_side_geoms_excluded`` /
        ``noncollidable_other_geoms_excluded`` name how much non-solid geometry
        the probe deliberately did not measure, per side. The payload coverage
        blocks are ``{}`` when nothing is carried, matching their empty pair
        lists.
    """
    attached_bodies = sorted(attached_body_ids)
    attached = frozenset(attached_bodies)
    probe_robot = robot_body_ids if probe_body_ids is None else (probe_body_ids & robot_body_ids)
    # The probe measures MESH-to-mesh; the kernel measures OBB-to-voxel. Widen
    # the window to the gap the collision model's own conservatism can already
    # account for, so the backing geometry of a stop cannot fall outside the
    # probe purely because the OBB corner reached further than the mesh. Widen
    # only: a derived window narrower than the default would recreate
    # silence-as-evidence from the other side.
    slop = collision_model_mesh_slop(model, description) if description is not None else {}
    voxel_half_diagonal = 0.0
    if isinstance(evidence_voxel, dict):
        cell_m: Any = evidence_voxel.get("resolution", 0.0)
        voxel_half_diagonal = float(cell_m) * 3.0**0.5 / 2.0
    max_slop: Any = slop.get("max_corner_slop_m", 0.0)
    budget = float(max_slop or 0.0) + voxel_half_diagonal
    # An attached-payload SELF stop has an OBB on both sides — the payload's
    # published primitives against the link OBBs — so the voxel term does not
    # apply and the payload's own corner slop does. Charging only the link's
    # share under-counts the admissible gap and makes a conservative, correct
    # stop read as a misattributed one.
    payload_slop = attached_payload_mesh_slop(model, data, attached_body_ids=attached)
    max_payload_slop: Any = payload_slop.get("max_corner_slop_m", 0.0)
    self_budget = float(max_slop or 0.0) + float(max_payload_slop or 0.0)
    # Widen the probe window to whichever comparison this stop needs, never
    # narrow it: a window derived below the default would recreate
    # silence-as-evidence from the other side.
    distmax_used = max(float(distmax_m), budget, self_budget if attached_bodies else 0.0)
    distmax_m = distmax_used
    # World side excludes the WHOLE robot, not just the probed subset: a link
    # the kernel does not check (the gripper, the base) is still the robot,
    # never an obstacle it could be "near".
    robot_world_pairs, robot_world_coverage = _nearest_pair_records(
        model,
        data,
        side=probe_robot,
        other_excluded=attached | robot_body_ids,
        distmax_m=distmax_m,
        max_pairs=max_pairs,
        max_calls=max_calls,
    )
    # Payload↔world: the margin stop a carried object triggers, which realized
    # contacts alone cannot adjudicate.
    # Payload↔robot-link self-pairs: the kernel's own
    # ``check_attached_self_collision``. A sink_cup stop at -0.62 mm against
    # panda_link5 was unadjudicable without this.
    payload_world_pairs: list[dict[str, object]] = []
    payload_world_coverage: dict[str, object] = {}
    payload_robot_pairs: list[dict[str, object]] = []
    payload_robot_coverage: dict[str, object] = {}
    if attached_bodies:
        payload_world_pairs, payload_world_coverage = _nearest_pair_records(
            model,
            data,
            side=attached,
            other_excluded=robot_body_ids,
            distmax_m=distmax_m,
            max_pairs=max_pairs,
            max_calls=max_calls,
        )
        payload_robot_pairs, payload_robot_coverage = _nearest_pair_records(
            model,
            data,
            side=attached,
            other_included=probe_robot,
            distmax_m=distmax_m,
            max_pairs=max_pairs,
            max_calls=max_calls,
        )
    voxel_backing: dict[str, object] | None = None
    if isinstance(evidence_voxel, dict) and evidence_voxel.get("index") is not None:
        voxel_backing = voxel_backing_record(
            model,
            data,
            voxel_index=int(evidence_voxel["index"]),
            grid_origin=list(evidence_voxel.get("origin", (0.0, 0.0, 0.0))),
            grid_resolution=float(evidence_voxel.get("resolution", 0.0)),
            grid_size=list(evidence_voxel.get("size", (0, 0, 0))),
            robot_body_ids=robot_body_ids,
            attached_body_ids=attached,
            base_frame_body=evidence_voxel.get("frame_body", base_frame_body),
        )
    snapshot: dict[str, object] = {
        "stop_class": "attached_payload" if attached_bodies else "robot_world",
        "sim_time_s": round(float(data.time), 6),
        "contacts_caveat": _CONTACTS_CAVEAT,
        "probe_robot_scope": "all_robot_bodies"
        if probe_body_ids is None
        else "kernel_checked_links",
        "probe_excluded_robot_bodies": _body_names(model, robot_body_ids - probe_robot),
        "attached_bodies": [_body_record(model, data, body_id) for body_id in attached_bodies],
        "payload_contacts": _contact_records(model, data, side=attached, other_excluded=frozenset())
        if attached_bodies
        else [],
        "robot_world_contacts": _contact_records(
            model, data, side=robot_body_ids, other_excluded=attached
        ),
        "nearest_robot_world_pairs": robot_world_pairs,
        "nearest_probe_coverage": robot_world_coverage,
        "nearest_payload_world_pairs": payload_world_pairs,
        "nearest_payload_world_coverage": payload_world_coverage,
        "nearest_payload_robot_pairs": payload_robot_pairs,
        "nearest_payload_robot_coverage": payload_robot_coverage,
        # The budget every kernel-vs-probe comparison needs, carried with the
        # verdict so no future round has to reconstruct it.
        "adjudication_budget": {
            "rule": (
                "kernel distances are OBB-to-voxel; ground-truth probes are "
                "mesh-to-mesh. The admissible gap is "
                "corner_slop(link) + voxel_half_diagonal."
            ),
            "max_corner_slop_m": slop.get("max_corner_slop_m"),
            "voxel_half_diagonal_m": round(voxel_half_diagonal, 6),
            "admissible_gap_m": round(budget, 6),
            "probe_distmax_default_m": round(float(_NEAREST_PROBE_DISTMAX_M), 6),
            "probe_distmax_used_m": round(distmax_used, 6),
            "collision_model_slop": slop or None,
            # The self-collision half. Distinct from the block above because
            # the geometry is: no voxel is involved, and BOTH sides are OBBs.
            "self_collision": {
                "rule": (
                    "an attached-payload self stop (collision_kind 'self', "
                    "link_a 'attached:<id>') is checked payload-primitive-OBB "
                    "to link-OBB, with no voxel on either side. The admissible "
                    "gap against a mesh-to-mesh probe is "
                    "corner_slop(link) + payload_corner_slop -- per-link slop "
                    "is in collision_model_slop.links[<link_b>]."
                ),
                "max_link_corner_slop_m": slop.get("max_corner_slop_m"),
                "max_payload_corner_slop_m": payload_slop.get("max_corner_slop_m"),
                "admissible_gap_m": round(self_budget, 6),
                "payload_slop": payload_slop or None,
            }
            if attached_bodies
            else None,
        },
        "evidence_voxel_backing": voxel_backing,
        "robot_joint_state": None
        if joint_state is None
        else {
            "name": list(joint_state.name),
            "position": [round(float(value), 6) for value in joint_state.position],
            "velocity": [round(float(value), 6) for value in joint_state.velocity],
            "stamp_ns": int(joint_state.stamp_ns),
        },
        "base_frame_tf": None,
    }
    if base_frame_body is not None:
        import mujoco  # reason: optional sim dep

        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_frame_body))
        if body_id >= 0:
            snapshot["base_frame_tf"] = {
                "body": base_frame_body,
                "world_xyz": [round(float(v), 6) for v in data.xpos[body_id]],
                "world_quat_wxyz": [round(float(v), 6) for v in data.xquat[body_id]],
            }
    return snapshot


def candidate_chunk_digest(
    *,
    stamp_ns: int,
    control_mode: int,
    horizon: int,
    n_dof: int,
    flat: Sequence[float],
    cartesian_delta_scale: Sequence[float] = (),
    ee_name: str = "",
    frame_id: str = "",
    rskill_id: str = "",
    trace_id: str = "",
    tick_index: int = 0,
) -> dict[str, object]:
    """One ``openral_msgs/ActionChunk``'s fields as a JSON-safe FK input.

    A predicted-horizon stop (``CollisionEvidence.horizon_step >= 0``) was
    evaluated at a configuration that exists nowhere in the simulator: the
    kernel integrated this chunk forward from the measured state (rows are
    joint configurations for JOINT_POSITION, joint velocity increments for
    JOINT_VELOCITY, or EE twists driven through the damped-least-squares
    Jacobian for CARTESIAN_DELTA — see
    ``cpp/openral_safety_kernel/src/lifecycle_kernel.cpp``). TF cannot
    reproduce it, so the snapshot logs the chunk itself: with the measured
    joint state, ``horizon_step`` and the kernel's own params, an offline
    tool can re-run that FK exactly.

    ``flat`` is reshaped into ``horizon`` rows of ``n_dof``; a length that
    disagrees with ``horizon * n_dof`` is reported as-is under
    ``flat`` with ``shape_mismatch: true`` rather than silently truncated.

    Example:
        >>> digest = candidate_chunk_digest(
        ...     stamp_ns=1, control_mode=5, horizon=2, n_dof=3, flat=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        ... )
        >>> digest["control_mode"], digest["ticks"]
        ('cartesian_delta', [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])
    """
    from openral_core.schemas import UINT8_TO_CONTROL_MODE

    mode = UINT8_TO_CONTROL_MODE.get(int(control_mode))
    values = [round(float(value), 6) for value in flat]
    digest: dict[str, object] = {
        "stamp_ns": int(stamp_ns),
        "control_mode": mode.value if mode is not None else int(control_mode),
        "control_mode_uint8": int(control_mode),
        "horizon": int(horizon),
        "n_dof": int(n_dof),
        "cartesian_delta_scale": [round(float(v), 6) for v in cartesian_delta_scale],
        "ee_name": str(ee_name),
        "frame_id": str(frame_id),
        "rskill_id": str(rskill_id),
        "trace_id": str(trace_id),
        "tick_index": int(tick_index),
    }
    if int(horizon) > 0 and int(n_dof) > 0 and len(values) == int(horizon) * int(n_dof):
        width = int(n_dof)
        digest["ticks"] = [values[i : i + width] for i in range(0, len(values), width)]
    else:
        digest["flat"] = values
        digest["shape_mismatch"] = True
    return digest


def initial_configuration_stop_record(
    snapshot: Mapping[str, object],
    *,
    stop_seq: int,
    last_action_ns: int,
    candidate_chunks_seen: int,
) -> dict[str, object] | None:
    """Classify a kernel stop that landed before the robot was ever commanded.

    A stop with ``last_action_ns == 0`` is categorically different from the
    mid-task stop the ``sim.estop_ground_truth_snapshot`` line reads like: no
    action has reached ``SimAttachedHAL.send_action`` yet — the single choke
    point every real action passes, stamped *before* any early return — so the
    configuration the kernel refused is the one the **scene reset produced**,
    not one a policy drove into. The robot is not doing something unsafe; it
    was **spawned** somewhere unsafe, and no policy, chunk, or margin tweak can
    clear it. The remedy is a scene-config change (a different seed, or pinned
    ``layout_ids`` / ``style_ids`` in ``backend_options``), not a safety one.

    This is observability only (CLAUDE.md §1.4). The stop itself is correct and
    is neither suppressed, delayed, nor altered — an initial pose that
    interpenetrates the scene is exactly what the kernel exists to refuse. All
    this does is name it, so an operator reading a run's artifacts does not
    spend a round debugging a policy that never got to act.

    ``candidate_chunks_seen`` is reported but deliberately does **not** gate:
    a chunk the kernel *rejected* is a candidate that was never applied, and
    that stop is still at the initial configuration.

    Args:
        snapshot: The :func:`estop_ground_truth_snapshot` record for this stop.
            Read for ``sim_time_s`` and the closest ``nearest_robot_world_pairs``
            entry; any missing key is simply omitted from the result.
        stop_seq: The bridge's monotonic stop counter, joining this line to the
            ``sim.estop_ground_truth_snapshot`` line for the same stop.
        last_action_ns: ``SimAttachedHAL.last_action_ns`` — ``0`` until the
            first ``send_action``.
        candidate_chunks_seen: How many chunks the kernel had been offered on
            ``/openral/candidate_action`` when the stop fired.

    Returns:
        A JSON-safe record, or ``None`` when at least one action has already
        been applied (an ordinary mid-task stop, already fully covered by
        ``sim.estop_ground_truth_snapshot``).

    Example:
        >>> pairs = [{"body_a": "robot0_link7", "body_b": "fridge_door", "distance_m": 0.0025}]
        >>> record = initial_configuration_stop_record(
        ...     {"sim_time_s": 4.85, "nearest_robot_world_pairs": pairs},
        ...     stop_seq=1,
        ...     last_action_ns=0,
        ...     candidate_chunks_seen=0,
        ... )
        >>> record["violation"], record["nearest_robot_world_pair"]["body_b"]
        ('initial_configuration', 'fridge_door')
        >>> initial_configuration_stop_record(
        ...     {}, stop_seq=2, last_action_ns=17, candidate_chunks_seen=3
        ... ) is None
        True
    """
    if int(last_action_ns) != 0:
        return None
    record: dict[str, object] = {
        "violation": "initial_configuration",
        "stop_seq": int(stop_seq),
        "candidate_chunks_seen": int(candidate_chunks_seen),
        "detail": (
            "the safety kernel stopped before any action reached the HAL, so "
            "the refused configuration is the one the scene reset produced. "
            "This is a scene-initialization defect, not a policy or margin "
            "one: re-seed the scene or pin backend_options.layout_ids / "
            "style_ids to a combination whose initial pose is clear."
        ),
    }
    sim_time_s = snapshot.get("sim_time_s")
    if isinstance(sim_time_s, (int, float)):
        record["sim_time_s"] = float(sim_time_s)
    pairs = snapshot.get("nearest_robot_world_pairs")
    if isinstance(pairs, list) and pairs and isinstance(pairs[0], dict):
        record["nearest_robot_world_pair"] = dict(pairs[0])
    return record


class SimSensorBridge:
    """Wire + tear down sim-sensor publishers and the viewer on a lifecycle node.

    Args:
        node: the HAL ``LifecycleNode`` (provides ``create_publisher`` /
            ``create_timer`` / ``get_logger``).
        hal: the connected HAL; scene streams activate only when it exposes
            ``read_images`` / ``mujoco_handles``.
        description: host manifest — gates streams on declared sensors.
        viewer_enabled: open ``mujoco.viewer.launch_passive`` (graceful headless).
        camera_rate_hz / viewer_sync_rate_hz: timer rates.
        scan_rate_hz: LaserScan publish rate (Hz). Gated on manifest lidar_2d.
        scan_n_beams: Number of ray-cast beams per scan cycle.
        scan_max_range_m: Sensor max range; "no hit" beams clamp to this.
        scan_min_range_m: Sensor min range (near-field filter).
    """

    def __init__(  # noqa: PLR0915 — flat attribute-init list; extracting helpers would only obscure it
        self,
        node: Any,
        hal: Any,
        description: RobotDescription,
        *,
        viewer_enabled: bool = True,
        camera_rate_hz: float = 10.0,
        viewer_sync_rate_hz: float = 30.0,
        scan_rate_hz: float = 10.0,
        scan_n_beams: int = 360,
        scan_max_range_m: float = 12.0,
        scan_min_range_m: float = 0.05,
        depth_rate_hz: float = 10.0,
        depth_max_range_m: float = 5.0,
        depth_pixel_stride: int = 4,
        idle_hold_ms: float = 2000.0,
        on_step: Any = None,
        on_attachment_perception_ready: Any = None,
    ) -> None:
        """Bind the node + HAL + manifest; opens no publishers until :meth:`setup`.

        ``idle_hold_ms`` is the sim-only idle stepper's quiet window: it
        advances the env with a zero/HOLD action only when no real action has
        arrived within this window (so an active skill always wins). Default
        2000 ms — it MUST exceed the slowest action cadence of an active skill,
        which for a VLA is the re-inference gap (SmolVLA ≈ 0.45 s/chunk; π0.5 /
        MolmoAct ≈ 1 s). The original 200 ms assumed a 30-200 Hz S1 stream and
        so RACED a 2 Hz VLA: it injected ~8 HOLD steps between policy actions,
        corrupting the trajectory and burning the episode horizon (~100 env
        steps consumed in ~12 policy actions → the VLA never finished a grasp).
        2 s sits comfortably above any VLA re-inference gap yet below the
        reasoner's ~5 s idle tick, so idle-stepping resumes (cameras stay live)
        within ~2 s once a skill stops. A truly idle scene (no action ever,
        ``last_action_ns == 0``) idle-steps immediately.

        ``on_step``: an optional zero-arg callback invoked after each
        successful ``idle_step`` — the node uses it to refresh the proprio
        snapshot so odom / joint_state stay fresh while the scene idles. It runs
        in this bridge's (default / "sim") callback group, so reading the
        simulator inside it is safe.
        """
        self._node = node
        self._on_step = on_step
        self._on_attachment_perception_ready = on_attachment_perception_ready
        self._hal = hal
        self._description = description
        self._viewer_enabled = viewer_enabled
        self._camera_rate_hz = camera_rate_hz
        self._idle_hold_ns = int(max(idle_hold_ms, 0.0) * 1_000_000)
        self._viewer_sync_rate_hz = viewer_sync_rate_hz
        self._scan_rate_hz = scan_rate_hz
        self._scan_n_beams = scan_n_beams
        self._scan_max_range_m = scan_max_range_m
        self._scan_min_range_m = scan_min_range_m
        self._depth_rate_hz = depth_rate_hz
        self._depth_max_range_m = depth_max_range_m
        self._depth_pixel_stride = depth_pixel_stride
        # Advertised lazily, keyed by camera name — a manifest camera earns its
        # topic on its first real frame (see _advertise_camera).
        self._image_pubs: dict[str, Any] = {}
        self._camera_qos: Any = None
        # RGB CameraInfo per camera: cuVSLAM/nvblox need the pinhole intrinsics
        # + a TF-valid frame, which plain RGB streams don't otherwise carry.
        self._camera_info_pubs: dict[str, Any] = {}
        self._camera_info_specs: dict[str, Any] = {}
        self._image_obs_key: dict[str, str] = {}
        # Per-camera last thumbnail emit timestamp (ns). Throttles the OTel
        # ``sensors.read_latest`` span to ~1 Hz while the ROS topic publishes
        # at the full camera_rate_hz.
        self._last_thumb_ns: dict[str, int] = {}
        self._image_missing_warned: set[str] = set()
        self._image_timer: Any = None
        self._camera_tf_timer: Any = None
        # (2026-06-04 idle-stepper amendment) — sim-only free-running
        # stepper timer. Created in setup ONLY when the HAL exposes ``idle_step``
        # AND has live MuJoCo handles (both sim gates); never against a real HAL.
        self._idle_timer: Any = None
        self._viewer: Any = None
        self._viewer_timer: Any = None
        self._scan_pub: Any = None
        self._scan_timer: Any = None
        # Depth-camera → PointCloud2 publishers, one per depth SensorSpec.
        # Feeds octomap_server → safety kernel world-collision voxel check.
        # Gated on live MuJoCo handles; _depth_disabled prevents repeated warnings.
        self._depth_pubs: dict[str, Any] = {}
        # Per depth camera, a dense 32FC1 depth image + CameraInfo
        # alongside the PointCloud2, so nvblox's projective depth integrator
        # (which rejects the sparse hit-only cloud) can build a `/map`.
        self._depth_image_pubs: dict[str, Any] = {}
        self._depth_info_pubs: dict[str, Any] = {}
        self._depth_timer: Any = None
        self._attachment_sub: Any = None
        self._attachment_ack_sub: Any = None
        self._attachment_voxel_sub: Any = None
        self._attachment_pub: Any = None
        self._place_declaration_sub: Any = None
        # E-stop ground truth (diagnostics only — never gates anything).
        # ``/openral/estop`` triggers the snapshot; the candidate-chunk ring
        # and the last collision evidence let an offline tool reconstruct a
        # PREDICTED-horizon stop, whose configuration TF cannot reproduce.
        self._estop_sub: Any = None
        self._candidate_action_sub: Any = None
        self._safety_failure_sub: Any = None
        self._candidate_chunks: deque[dict[str, object]] = deque(maxlen=_CANDIDATE_CHUNK_HISTORY)
        self._last_collision_evidence: dict[str, object] | None = None
        # Geometry (not occupancy) of the last /openral/world_voxels grid, so a
        # `b=voxel_<n>` evidence line can be located in space at the stop.
        self._last_voxel_grid: dict[str, object] | None = None
        self._last_collision_evidence_ns: int = 0
        self._collision_evidence_warned: bool = False
        self._estop_seq: int = 0
        self._estop_awaiting_evidence: bool = False
        self._attachment_timer: Any = None
        self._attachment_revision: int = 0
        self._attachment_applied_revision: int = -1
        self._attachment_desired: list[Any] = []
        self._attachment_pending: list[Any] | None = None
        self._attachment_tracker: Any = None
        self._attachment_depth_frames_remaining: int = 0
        self._attachment_voxel_updates_remaining: int = 0
        self._attachment_expect_voxel_update: bool = False
        self._attachment_transparent_depth_stamp_ns: int | None = None
        self._depth_disabled: set[str] = set()
        self._depth_base_body: str | None = None
        self._depth_base_body_id: int = -1
        # The MJCF body ``base_frame`` DENOTES on /tf — the arm mount, which on
        # a robosuite mobile manipulator sits a 0.70 m pedestal above the
        # chassis root cached above. Every ``base_frame -> …`` extrinsic is
        # measured against this one, so the published depth cloud lands where TF
        # says it does (ADR-0095). Equals ``_depth_base_body`` on fixed bases.
        self._base_frame_body: str | None = None
        # Robot's own MJCF body ids — dropped from the depth cloud so the
        # base-mounted camera doesn't voxelise the arm into its own world map.
        self._depth_self_bodies: frozenset[int] = frozenset()
        self._tf_broadcaster: Any = None
        # Static world->base_frame TF (gives a fixed-base sim arm the
        # world root its TF tree otherwise lacks, so task-space state layouts
        # like ``libero_eef8d`` can read the WORLD-frame EE pose the policy was
        # trained on). Published once from the base body's MuJoCo world pose;
        # skipped for mobile bases (they publish odom->base).
        self._static_tf_broadcaster: Any = None
        self._world_base_published: bool = False
        # Cross-frame lift — RGB cameras whose optical-frame TF failed
        # to resolve (no MJCF camera); warned once, then skipped.
        self._camera_tf_disabled: set[str] = set()
        # OPENRAL_DASHBOARD_FLIP_180 — flip ONLY the dashboard thumbnail 180° so
        # bottom-up MuJoCo (LIBERO) renders show upright. The published Image (and
        # thus the policy observation) stays raw; the world_state node applies the
        # same display-only flip to its thumbnail, so both `sensors.read_latest`
        # emitters agree and the dashboard card never flickers between orientations.
        import os  # reason: env-gated display feature

        self._dashboard_flip_180 = os.environ.get(
            "OPENRAL_DASHBOARD_FLIP_180", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        # Offscreen "cinecam" recorder (website-video capture): when
        # OPENRAL_CINECAM_DIR is set, render the pulled-back free-camera view
        # (same pose as the onscreen viewer) to numbered JPGs each tick. Robust
        # vs the onscreen GLFW window, which the desktop WM can unmap.
        self._cinecam_renderer: Any = None
        self._cinecam_cam: Any = None
        self._cinecam_opt: Any = None
        self._cinecam_timer: Any = None
        self._cinecam_frame: int = 0
        self._cinecam_out_dir: str = ""
        self._cinecam_model: Any = None
        self._cinecam_w: int = 0
        self._cinecam_h: int = 0
        self._cinecam_base_body: Any = None
        self._cinecam_setup_az: float = 0.0
        self._cinecam_setup_el: float = 0.0
        self._cinecam_setup_dist: float = 1.0

    def setup(self) -> None:
        """Activate every stream the manifest + HAL support. Idempotent-safe per activate."""
        self._setup_cameras()
        self._setup_idle_stepper()
        self._setup_cinecam()
        self._setup_viewer()
        self._setup_scan()
        self._setup_attachment_state()
        self._setup_estop_ground_truth()
        self._setup_depth()

    def teardown(self) -> None:  # noqa: PLR0915  # reason: one symmetric resource cleanup
        """Cancel timers, destroy publishers, and close the viewer (idempotent)."""
        for t in (
            self._image_timer,
            self._camera_tf_timer,
            self._idle_timer,
            self._viewer_timer,
            self._cinecam_timer,
            self._scan_timer,
            self._depth_timer,
            self._attachment_timer,
        ):
            if t is not None:
                t.cancel()
        self._image_timer = self._idle_timer = self._camera_tf_timer = None
        self._viewer_timer = self._scan_timer = self._depth_timer = None
        self._attachment_timer = None
        self._cinecam_timer = None
        if self._attachment_sub is not None:
            self._node.destroy_subscription(self._attachment_sub)
            self._attachment_sub = None
        if self._attachment_ack_sub is not None:
            self._node.destroy_subscription(self._attachment_ack_sub)
            self._attachment_ack_sub = None
        if self._attachment_voxel_sub is not None:
            self._node.destroy_subscription(self._attachment_voxel_sub)
            self._attachment_voxel_sub = None
        if self._place_declaration_sub is not None:
            self._node.destroy_subscription(self._place_declaration_sub)
            self._place_declaration_sub = None
        self._teardown_estop_ground_truth()
        if self._attachment_pub is not None:
            self._node.destroy_publisher(self._attachment_pub)
            self._attachment_pub = None
        if self._cinecam_renderer is not None:
            with contextlib.suppress(Exception):  # reason: renderer GL ctx may be gone
                self._cinecam_renderer.close()
            self._cinecam_renderer = None
        for pub in (*self._image_pubs.values(), *self._camera_info_pubs.values()):
            self._node.destroy_publisher(pub)
        self._image_pubs.clear()
        self._camera_info_pubs.clear()
        self._camera_info_specs.clear()
        self._image_obs_key.clear()
        self._camera_qos = None
        self._last_thumb_ns.clear()
        self._image_missing_warned.clear()
        if self._scan_pub is not None:
            self._node.destroy_publisher(self._scan_pub)
            self._scan_pub = None
        for pub in (
            *self._depth_pubs.values(),
            *self._depth_image_pubs.values(),
            *self._depth_info_pubs.values(),
        ):
            self._node.destroy_publisher(pub)
        self._depth_pubs.clear()
        self._depth_image_pubs.clear()
        self._depth_info_pubs.clear()
        self._depth_disabled.clear()
        self._depth_base_body = None
        self._depth_base_body_id = -1
        self._base_frame_body = None
        self._depth_self_bodies = frozenset()
        self._camera_tf_disabled.clear()
        self._tf_broadcaster = None
        self._static_tf_broadcaster = None
        self._world_base_published = False
        if self._viewer is not None:
            with contextlib.suppress(Exception):  # reason: viewer already closed
                self._viewer.close()
            self._viewer = None

    # -- RGB cameras --
    def _setup_cameras(self) -> None:
        if not hasattr(self._hal, "read_images"):
            return
        rgb = [s for s in self._description.sensors if s.modality == "rgb"]
        if not rgb:
            return
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

        self._camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        # Record what the manifest declares, but do NOT advertise yet — each
        # camera earns its topic in :meth:`_advertise_camera` on its first real
        # frame. A manifest may legitimately declare a camera the *current*
        # scene cannot supply: robocasa's synthetic ``head`` nav cam is
        # mobile-base-only and opt-in via ``OPENRAL_ROBOCASA_HEAD_CAM``, which
        # ``openral deploy`` sets only when a capability-matched rSkill
        # consumes ``observation.images.head`` — so on a palette without one,
        # panda_mobile still declares the sensor and no backend renders it.
        # Advertising eagerly published a permanently silent
        # ``/openral/cameras/head/image`` onto the graph, which reads to any
        # subscriber (dashboard panel, recorder, nav policy) as a live stream
        # that is merely slow — so it waits forever instead of failing.
        for s in rgb:
            self._image_obs_key[s.name] = _obs_key_for_sensor(s)
            self._camera_info_specs[s.name] = s
        self._image_timer = self._node.create_timer(
            1.0 / max(self._camera_rate_hz, 1.0), self._publish_images
        )
        # Cross-frame lift — broadcast base_frame -> <camera>_optical_frame
        # for every RGB camera that owns a dedicated optical frame, from live
        # MuJoCo poses, so the world-state object-lift can project the world
        # voxel map into any detection camera (generic over robots/camera names).
        from tf2_ros import TransformBroadcaster

        if self._tf_broadcaster is None:
            self._tf_broadcaster = TransformBroadcaster(self._node)
        self._camera_tf_timer = self._node.create_timer(
            1.0 / max(self._camera_rate_hz, 1.0), self._publish_camera_optical_tfs
        )
        self._node.get_logger().info(
            f"SimSensorBridge: {len(rgb)} camera(s) declared, each advertised on "
            "its first frame: " + ", ".join(f"{s.name}<-{self._image_obs_key[s.name]}" for s in rgb)
        )

    def _advertise_camera(self, name: str) -> Any:
        """Create (once) and return the ``Image`` publisher for one camera.

        Called from :meth:`_publish_images` the first time the backend supplies
        a frame for ``name``, so the topic set on the graph is exactly the set
        of cameras this scene can actually render — never a superset drawn from
        the manifest (CLAUDE.md §1.4, explicit beats implicit).

        The matching ``camera_info`` publisher is created alongside so any
        manifest camera can serve a pinhole consumer (cuVSLAM rig build, nvblox
        mono-depth framing) — link-framed cameras get TF from
        ``robot_state_publisher``, optical-frame cameras from
        :meth:`_publish_camera_optical_tfs`. Both are published per-frame
        (VOLATILE) so a late-joining subscriber never misses a latched-once
        message.
        """
        pub = self._image_pubs.get(name)
        if pub is not None:
            return pub
        from sensor_msgs.msg import CameraInfo
        from sensor_msgs.msg import Image as RosImage

        pub = self._node.create_publisher(
            RosImage, f"/openral/cameras/{name}/image", self._camera_qos
        )
        self._image_pubs[name] = pub
        self._camera_info_pubs[name] = self._node.create_publisher(
            CameraInfo, f"/openral/cameras/{name}/camera_info", self._camera_qos
        )
        self._node.get_logger().info(
            f"SimSensorBridge: advertising /openral/cameras/{name}/image "
            f"(obs key '{self._image_obs_key.get(name, name)}')"
        )
        return pub

    def _publish_images(self) -> None:
        """Republish cached camera frames from the HAL as sensor_msgs/Image.

        Reads :meth:`SimAttachedHAL.read_images` (a dict of
        ``camera_name -> HWC uint8 NDArray``) and publishes each frame on
        ``/openral/cameras/<name>/image``.  The obs-key lookup (via
        ``_image_obs_key``) lets LIBERO-style scenes (keyed by VLA slot
        ``camera1`` / ``camera2``) coexist with robocasa real-name keys.

        Encoding handles mono8 / rgb8 / rgba8 arrays automatically. Frame
        data is copied bytewise — no compression hop.

        An OTel ``sensors.read_latest`` span (with JPEG thumbnail) is emitted
        at most once per second per camera so the dashboard Perception card
        updates without ballooning the OTLP payload.
        """
        reader = getattr(self._hal, "read_images", None)
        if reader is None or not self._image_obs_key:
            return
        from sensor_msgs.msg import Image as RosImage

        images = reader()  # dict[str, ndarray HWC uint8]
        if not isinstance(images, dict) or not images:
            return
        stamp = self._node.get_clock().now().to_msg()
        from openral_observability.producer import (
            encode_rgb_thumbnail,
            record_sensor_frame_attrs,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        now_ns = time.monotonic_ns()
        for name, obs_key in self._image_obs_key.items():
            arr = _frame_for_camera(images, obs_key, name)
            if arr is None:
                if name not in self._image_missing_warned:
                    self._image_missing_warned.add(name)
                    self._node.get_logger().warning(
                        f"SimSensorBridge: no frame for camera '{name}' "
                        f"(expected obs key '{obs_key}' or name '{name}'); "
                        f"available keys: {sorted(images.keys())}. "
                        f"/openral/cameras/{name}/image stays unadvertised "
                        "until a frame arrives. Check the scene's --robot "
                        "override matches sensor layout, and whether this "
                        "camera is opt-in (robocasa's synthetic 'head' nav cam "
                        "needs a mobile base plus OPENRAL_ROBOCASA_HEAD_CAM=1, "
                        "which `openral deploy` sets only when a "
                        "capability-matched rSkill consumes it)."
                    )
                continue
            if arr.ndim != _IMAGE_DIM or arr.shape[2] not in (1, _RGB_CHANNELS, 4):
                continue
            pub = self._advertise_camera(name)
            h, w, c = arr.shape
            msg = RosImage()
            msg.header.stamp = stamp
            msg.header.frame_id = name
            msg.height = int(h)
            msg.width = int(w)
            msg.encoding = "mono8" if c == 1 else "rgb8" if c == _RGB_CHANNELS else "rgba8"
            msg.is_bigendian = 0
            msg.step = int(w * c)
            msg.data = bytes(arr.astype("uint8").tobytes())
            pub.publish(msg)
            info_pub = self._camera_info_pubs.get(name)
            if info_pub is not None:
                info = self._rgb_camera_info(name, int(w), int(h), stamp)
                if info is not None:
                    info_pub.publish(info)
            # Emit a ``sensors.read_latest`` span at most once per second per
            # camera (dashboard polls at ~1 Hz; higher rate would balloon OTLP
            # payload with redundant thumbnails).
            last = self._last_thumb_ns.get(name, 0)
            if now_ns - last < _THUMB_INTERVAL_NS:
                continue
            self._last_thumb_ns[name] = now_ns
            # Display-only 180° flip for the dashboard thumbnail (the published
            # Image above stays raw for the policy/world_state path). Keeps this
            # emitter's thumbnail in the same orientation as world_state's.
            thumb_arr = (
                arr[::-1, ::-1] if (self._dashboard_flip_180 and c == _RGB_CHANNELS) else arr
            )
            thumb = encode_rgb_thumbnail(thumb_arr) if c == _RGB_CHANNELS else None
            with tracer.start_as_current_span("sensors.read_latest") as span:
                span.set_attribute("openral.sensors.source", name)
                record_sensor_frame_attrs(
                    span,
                    modality="rgb",
                    encoding=msg.encoding,
                    width=int(w),
                    height=int(h),
                    channels=int(c),
                    age_ms=0.0,
                    thumbnail_bytes=thumb,
                )

    def _rgb_camera_info(self, name: str, width: int, height: int, stamp: Any) -> Any:
        """Pinhole ``CameraInfo`` for an RGB camera, from its MJCF ``fovy``.

        cuVSLAM builds its rig from these intrinsics and nvblox frames the mono
        depth against the camera's ``*_optical_frame``. MuJoCo cameras are square-
        pixel with a vertical ``fovy``, so ``fy = (h/2)/tan(fovy/2)``, ``fx = fy``,
        principal point at the image centre. No-op for non-MuJoCo backends or an
        unresolvable MJCF camera (warned once by the TF path).
        """
        import math

        spec = self._camera_info_specs.get(name)
        if spec is None:
            return None
        handle = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handle is None:
            return None
        model, _ = handle

        import mujoco

        from openral_hal.depth_cloud import camera_info_from_intrinsics, mjcf_camera_name

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, mjcf_camera_name(spec))
        if cam_id < 0:
            return None
        fovy = float(model.cam_fovy[cam_id])
        fy = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
        return camera_info_from_intrinsics(
            width=width,
            height=height,
            fx=fy,
            fy=fy,
            cx=width / 2.0,
            cy=height / 2.0,
            frame_id=spec.frame_id,
            stamp=stamp,
        )

    def _publish_camera_optical_tfs(self) -> None:
        """Broadcast ``base_frame -> <camera>_optical_frame`` for every RGB camera.

        Cross-frame object-lift: the world-state lifter projects the
        world voxel map (built from the robot's body-mounted depth sensor) into
        each detection camera using that camera's extrinsics. This publishes
        those extrinsics live from MuJoCo poses — generic over any robot and any
        camera name (the MJCF camera is resolved from each ``SensorSpec``'s
        ``metadata.mjcf_camera``). Only cameras that own a dedicated
        ``*_optical_frame`` are broadcast: a camera whose ``frame_id`` is a robot
        link (e.g. an eye-in-hand at ``panda_hand``) already has TF from
        ``robot_state_publisher`` and must not be clobbered. No-op for non-MuJoCo
        backends or cameras whose MJCF name doesn't resolve (warned once).
        """
        # Keyed off the *declared* set, not the advertised one: the optical
        # extrinsic is a property of the manifest + MJCF, so it stays available
        # from configure time rather than waiting on the first rendered frame.
        if self._tf_broadcaster is None or not self._camera_info_specs:
            return
        handle = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handle is None:
            return
        model, data = handle
        if self._depth_base_body is None and self._depth_base_body_id < 0:
            self._resolve_depth_base_body(model)
        if self._base_frame_body is None:
            return

        # Publish the world root for a fixed-base sim arm (once).
        self._publish_world_base_tf(model, data)

        from geometry_msgs.msg import TransformStamped
        from openral_core.exceptions import ROSConfigError

        from openral_hal.depth_cloud import camera_optical_tf_to_base, mjcf_camera_name

        base_frame_id = getattr(self._description, "base_frame", "base_link")
        stamp = self._node.get_clock().now().to_msg()
        specs = {s.name: s for s in _optical_frame_rgb_cameras(self._description.sensors)}
        for name in self._camera_info_specs:
            spec = specs.get(name)
            if spec is None or name in self._camera_tf_disabled:
                continue
            try:
                xyz, quat = camera_optical_tf_to_base(
                    model=model,
                    data=data,
                    camera_name=mjcf_camera_name(spec),
                    base_body_name=self._base_frame_body,
                )
            except ROSConfigError as exc:
                self._camera_tf_disabled.add(name)
                self._node.get_logger().warning(
                    f"camera optical TF {name!r} disabled: {exc}; "
                    "check the SensorSpec's mjcf_camera metadata."
                )
                continue
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = base_frame_id
            tf.child_frame_id = spec.frame_id
            tf.transform.translation.x = xyz[0]
            tf.transform.translation.y = xyz[1]
            tf.transform.translation.z = xyz[2]
            tf.transform.rotation.x = quat[0]
            tf.transform.rotation.y = quat[1]
            tf.transform.rotation.z = quat[2]
            tf.transform.rotation.w = quat[3]
            self._tf_broadcaster.sendTransform(tf)

    def _publish_world_base_tf(self, model: object, data: object) -> None:
        """Publish a static ``world -> base_frame`` TF from the base body's sim pose.

        A robosuite-attached fixed-base arm (LIBERO franka, ur5e, ...) roots its
        TF tree at ``base_frame`` (panda_link0) with NO parent, yet the robot
        sits at a non-origin world pose (LIBERO mounts the franka at world
        ``[-0.66, 0, 0.912]``, varying by suite). The benchmark feeds the policy
        the WORLD-frame EE pose (robosuite ``robot0_eef_pos``); without this
        transform the ``libero_eef8d`` task-space state layout could only read
        the base-relative EE pose (off by the mount — the policy would see the
        EE ~0.9 m below where it trained). Publishing the base body's live
        MuJoCo world pose as ``world -> base_frame`` makes
        ``tf_lookup("world", "panda_hand_tcp")`` equal robosuite's eef.

        Static (the base is fixed) + latched, so the skill_runner's tf_lookup
        gets it even joining late. Skipped for MOBILE bases — they publish a
        live ``odom -> base`` and a second parent for ``base`` would corrupt the
        tree (detected via
        :func:`~openral_hal.mobile_base_bridge.describes_mobile_base`, the same
        predicate the lifecycle node attaches the ``odom`` publisher on).
        """
        if self._world_base_published:
            return
        if describes_mobile_base(self._description):
            self._world_base_published = True  # mobile: odom owns base->world; nothing to do
            return
        if self._base_frame_body is None:
            return

        import mujoco  # reason: defer optional sim dep
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster

        # The body ``base_frame`` denotes, not the chassis root — this TF *is*
        # the base frame's pose (ADR-0095). Identical on a fixed-base arm.
        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._base_frame_body))
        if bid < 0:
            return
        pos = data.xpos[bid]  # type: ignore[attr-defined]  # world position of the base body
        quat_wxyz = data.xquat[bid]  # type: ignore[attr-defined]  # MuJoCo quaternion is wxyz
        if self._static_tf_broadcaster is None:
            self._static_tf_broadcaster = StaticTransformBroadcaster(self._node)
        base_frame_id = getattr(self._description, "base_frame", "base_link")
        tf = TransformStamped()
        tf.header.stamp = self._node.get_clock().now().to_msg()
        tf.header.frame_id = "world"
        tf.child_frame_id = base_frame_id
        tf.transform.translation.x = float(pos[0])
        tf.transform.translation.y = float(pos[1])
        tf.transform.translation.z = float(pos[2])
        tf.transform.rotation.w = float(quat_wxyz[0])
        tf.transform.rotation.x = float(quat_wxyz[1])
        tf.transform.rotation.y = float(quat_wxyz[2])
        tf.transform.rotation.z = float(quat_wxyz[3])
        self._static_tf_broadcaster.sendTransform(tf)
        self._world_base_published = True
        self._node.get_logger().info(
            f"published static world->{base_frame_id} at "
            f"[{float(pos[0]):.3f}, {float(pos[1]):.3f}, {float(pos[2]):.3f}] "
            "(fixed-base sim world root)"
        )

    # -- Sim-only free-running idle stepper --
    def _setup_idle_stepper(self) -> None:
        """Create the sim-only idle-step timer, gated on a callable ``idle_step``.

        Gate (the PRIMARY safety gate): the HAL exposes a callable ``idle_step``,
        defined ONLY on :class:`~openral_hal.sim_attached.SimAttachedHAL`. A real
        HAL never defines it, so the timer is never created against real
        hardware. This is the real guarantee, not "zero is harmless" (a zero
        vector is a HOLD in sim but "drive to 0 rad" — violent — on a real
        absolute-position arm).

        No MuJoCo-handle gate (dropped in a later revision): idle-stepping
        is valid for any wrapped SimRollout, so a non-MuJoCo backend (Isaac Sim
        sidecar, ManiSkill3) keeps its cameras live when idle too. ``idle_step``
        itself returns ``False`` for a non-sim HAL, and the
        catch-once-and-disable guard below contains any per-tick fault.

        The timer runs at ``camera_rate_hz`` so step-then-publish stays matched
        (the existing camera timer republishes the freshened ``_last_obs``); no
        separate rate param is introduced. The single-threaded rclpy executor
        ensures the idle callback and ``_on_safe_action`` never run
        concurrently, so :func:`should_idle_step`'s timestamp check is a
        sufficient hand-off (no lock).
        """
        if not callable(getattr(self._hal, "idle_step", None)):
            return
        # Drive the idle stepper on WALL time, never the
        # node's clock. Under ``use_sim_time`` (simulation clock authority) a node-clock
        # timer fires off ``/clock`` — but the idle step is what ADVANCES
        # ``/clock`` (it steps the sim), so a sim-time timer here deadlocks: no
        # step → no /clock → no fire. A SYSTEM_TIME clock breaks the cycle so the
        # sim keeps stepping and bootstraps /clock. Harmless on wall-clock runs
        # (SYSTEM_TIME == the node clock there).
        from rclpy.clock import Clock, ClockType

        self._idle_timer = self._node.create_timer(
            1.0 / max(self._camera_rate_hz, 1.0),
            self._idle_step_tick,
            clock=Clock(clock_type=ClockType.SYSTEM_TIME),
        )
        self._node.get_logger().info(
            f"SimSensorBridge: sim-only idle stepper @ {self._camera_rate_hz:.1f} Hz "
            f"(idle_hold={self._idle_hold_ns / 1e6:.0f} ms) — keeps cameras live when idle."
        )

    def _idle_step_tick(self) -> None:
        """Advance the sim one HOLD tick when no recent real action has arrived.

        Yields to active skills via :func:`should_idle_step`; the camera-publish
        timer then republishes the freshened ``_last_obs`` (publish path
        unchanged).

        Containment: ``idle_step`` fires autonomously on this timer (with
        ``last_action_ns == 0`` an idle scene starts stepping immediately). If
        it raises — most likely an ``env.step`` action-dim mismatch on a native
        backend whose true width was not probed (the documented probe gap;
        ``so101_box`` wants 6-D but the fallback is 11-D) — we log ONE loud
        warning and cancel/disable this timer so it cannot crash-loop the graph
        every tick. We do NOT swallow silently: one warning, then stop.
        """
        # Already disabled (e.g. by a prior error) — a callback queued before
        # the cancel must be a no-op, never re-trigger the disabled path.
        if self._idle_timer is None:
            return
        idle_step = getattr(self._hal, "idle_step", None)
        if not callable(idle_step):
            return
        last_action_ns = int(getattr(self._hal, "last_action_ns", 0))
        step_while_active = bool(getattr(self._hal, "_step_while_active", False))
        if not should_idle_step(
            time.monotonic_ns(),
            last_action_ns,
            self._idle_hold_ns,
            step_while_active=step_while_active,
        ):
            return
        try:
            # Hand the tick's WALL period to the HAL so it advances that much
            # SIM time. The idle timer is deliberately on SYSTEM_TIME (see
            # _start_idle_stepper), but every other timer on this node — the
            # camera republisher, the camera-TF broadcast, the cinecam — runs on
            # the node clock, which under `use_sim_time` is the sim clock this
            # very callback advances. A single physics step per tick leaves sim
            # time running at ~2% of wall, so those timers fire ~50x slower than
            # their nominal rate and the world-state aggregator latches every
            # camera STALE against its wall-clock arrival stamps. Bare MuJoCo
            # arms also run this during active skills: send_action() advances
            # one physics tick, not one wall-time slice.
            idle_step(wall_dt_s=1.0 / max(self._camera_rate_hz, 1.0))
        except Exception as exc:  # reason: contain a per-tick crash-loop; warn once + disable
            self._node.get_logger().warning(
                f"SimSensorBridge: idle stepper disabled after error: {exc}. "
                "Possibly an env action-dim mismatch (an explicit env_action_dim "
                "override that disagrees with the backend's step width; native "
                "backends now expose their own action_dim so the probe resolves it). "
                "Cameras will only refresh while a skill is actively stepping the env."
            )
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            return
        # The env advanced; refresh the proprio snapshot so the
        # control group's odom/joint_state publishers stay fresh while idle.
        if self._on_step is not None:
            self._on_step()

    # -- 2-D LiDAR / LaserScan --
    def _setup_scan(self) -> None:
        """Create the ``/scan`` publisher + timer, gated only on manifest lidar_2d.

        Gate: the manifest declares a ``lidar_2d`` sensor
        (``RobotDescription.lidar_sensor is not None``); franka/ur5e/so100 nodes
        without one never advertise a scan topic.

        The publisher is created whenever a lidar is declared, regardless of
        whether the HAL has live MuJoCo handles. :meth:`_compute_scan_ranges`
        ray-casts against the scene when handles are bound (``SimAttachedHAL``)
        and emits a constant ``max_range`` no-hit fan otherwise (the in-process
        digital twin has no scene to ray-cast, so "no hit everywhere" is the
        honest reading slam_toolbox / Nav2 boot on). This makes the bridge the
        single owner of ``/scan`` — issue #191 Phase 3 removed the panda_mobile
        node's separate digital-twin no-hit publisher.
        """
        if self._description.lidar_sensor is None:
            return
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import LaserScan

        # `/scan` is BEST_EFFORT (sensor-class data per CLAUDE.md §2 ROS QoS
        # table). slam_toolbox and Nav2 both subscribe BEST_EFFORT by default.
        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        self._scan_pub = self._node.create_publisher(LaserScan, "/scan", scan_qos)
        self._scan_timer = self._node.create_timer(
            1.0 / max(self._scan_rate_hz, 1.0), self._publish_scan
        )
        lidar = self._description.lidar_sensor
        self._node.get_logger().info(
            f"SimSensorBridge: publishing /scan @ {self._scan_rate_hz:.1f} Hz "
            f"frame={lidar.frame_id} beams={self._scan_n_beams} "
            f"range=[{self._scan_min_range_m}, {self._scan_max_range_m}] m."
        )

    def _publish_scan(self) -> None:
        """Construct + publish a /scan message (live MJCF ray-cast or synthetic no-hit).

        Lifted from ``openral_hal_panda_mobile.lifecycle_node._publish_scan`` /
        ``_compute_scan_ranges``. Uses the same
        :func:`openral_sim.backends.robocasa.synthesize_laser_scan_2d` call
        and identical no-hit fallback so nav-stack behaviour is bit-identical
        to the panda_mobile node.
        """
        if self._scan_pub is None:
            return
        import math  # reason: stdlib defer

        from sensor_msgs.msg import LaserScan

        n_beams = int(self._scan_n_beams)
        max_range = float(self._scan_max_range_m)
        min_range = float(self._scan_min_range_m)
        scan_rate = float(self._scan_rate_hz)

        ranges = self._compute_scan_ranges(n_beams=n_beams, max_range_m=max_range)

        lidar = self._description.lidar_sensor
        frame_id = lidar.frame_id if lidar is not None else "base_link"

        msg = LaserScan()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.angle_min = float(-math.pi)
        msg.angle_max = float(math.pi)
        msg.angle_increment = float(2.0 * math.pi / max(n_beams, 1))
        msg.scan_time = float(1.0 / max(scan_rate, 1.0))
        msg.time_increment = 0.0
        msg.range_min = min_range
        msg.range_max = max_range
        msg.ranges = list(ranges)
        msg.intensities = []
        self._scan_pub.publish(msg)

    def _compute_scan_ranges(self, *, n_beams: int, max_range_m: float) -> list[float]:
        """Return scan ranges — live MJCF ray-cast if handles are bound, else no-hit fan.

        Mirrors ``openral_hal_panda_mobile.lifecycle_node._compute_scan_ranges``
        exactly: tries ``hal.mujoco_handles()``, falls back to a
        ``max_range_m``-clamped no-hit list so slam_toolbox / Nav2 treat every
        beam as "nothing in front of me" rather than NaN-poisoning their grids.
        """
        handle = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handle is None:
            # Non-MuJoCo backend: use the ranges the HAL surfaces (Isaac lidar
            # ray-cast), else an honest no-hit fan.
            read = getattr(self._hal, "read_scan", None)
            if callable(read):
                scan = read()
                if scan is not None and len(scan) == n_beams:
                    return [float(r) for r in scan]
            return constant_scan_no_hit_ranges(n_beams=n_beams, max_range_m=max_range_m)
        model, data = handle
        from openral_core import (  # reason: scoped to scan synthesis
            extract_base_sim_joint_names,
        )
        from openral_sim.backends.robocasa import (  # reason: optional dep
            synthesize_laser_scan_2d,
        )

        # Pull MJCF joint names from the HAL's description so the
        # sim-side helper doesn't depend on hardcoded robosuite /
        # robocasa naming conventions.
        base_names: tuple[str, str, str] | None = None
        description = getattr(self._hal, "description", None)
        if description is not None:
            base_names = extract_base_sim_joint_names(description)

        ranges = synthesize_laser_scan_2d(
            model=model,
            data=data,
            base_joint_names=base_names,
            n_beams=n_beams,
            max_range_m=max_range_m,
        )
        return [float(r) for r in ranges]

    # -- Depth PointCloud2 --
    def _setup_attachment_state(self) -> None:
        """Subscribe atomic attachment snapshots when the sim HAL supports them."""
        update = getattr(self._hal, "update_attached_objects", None)
        read = getattr(self._hal, "read_attached_objects", None)
        if not callable(update) or not callable(read):
            return
        from openral_msgs.msg import AttachmentState
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._attachment_sub = self._node.create_subscription(
            AttachmentState,
            "/openral/attachment_state",
            self._on_attachment_state,
            qos,
        )
        from std_msgs.msg import UInt64

        self._attachment_ack_sub = self._node.create_subscription(
            UInt64,
            "/openral/attachment_state_applied",
            self._on_attachment_state_applied,
            qos,
        )
        from openral_msgs.msg import OccupancyVoxels

        voxel_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self._attachment_voxel_sub = self._node.create_subscription(
            OccupancyVoxels,
            "/openral/world_voxels",
            self._on_attachment_world_voxels,
            voxel_qos,
        )
        self._attachment_pub = self._node.create_publisher(
            AttachmentState,
            "/openral/attachment_state",
            qos,
        )
        self._attachment_timer = self._node.create_timer(
            0.2,
            self._publish_attachment_state,
        )
        handles = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handles is not None:
            from openral_core.exceptions import ROSConfigError

            from openral_hal._sim_attachment_evidence import SimAttachmentEvidenceTracker

            model, _data = handles
            try:
                self._attachment_tracker = SimAttachmentEvidenceTracker(
                    model,
                    self._description,
                    stable_ticks=1,
                )
                add_observer = getattr(self._hal, "add_post_step_observer", None)
                if callable(add_observer):
                    add_observer(self._observe_attachment_evidence)
                    self._node.get_logger().info(
                        "automatic sim attachment evidence armed at the post-step boundary"
                    )
                    self._setup_place_declaration()
                else:
                    self._node.get_logger().warning(
                        "automatic sim attachment evidence has no post-step observer"
                    )
            except ROSConfigError as exc:
                self._node.get_logger().warning(
                    f"automatic sim attachment evidence disabled: {exc}"
                )

    def _setup_place_declaration(self) -> None:
        """Subscribe dispatch's place-phase declarations (ADR-0097).

        This is the dispatch → HAL half of the declaration's path; the World
        State half is the attestation it licenses, which rides the existing
        `/openral/attachment_state` snapshot. TRANSIENT_LOCAL so a producer
        armed after the dispatcher sees the goal's declaration rather than
        missing it, and depth 1 because only the newest declaration is ever in
        force.
        """
        from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )

        self._place_declaration_sub = self._node.create_subscription(
            PlaceDeclarationMsg,
            "/openral/place_declaration",
            self._on_place_declaration,
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            ),
        )

    def _on_place_declaration(self, msg: object) -> None:
        """Hand one declaration to the attestation producer, or refuse it.

        Refusal is the fail-closed direction and it is logged, never silent: a
        declaration naming a target that does not exist in this scene must not
        be treated as "no declaration happened", because an operator who
        mistyped a target is owed the error (HZ-0097-2's attributability).
        """
        if self._attachment_tracker is None:
            return
        from openral_core import PlaceDeclaration
        from openral_core.exceptions import ROSConfigError

        try:
            declaration = PlaceDeclaration.from_idl(msg)
        except (ValueError, TypeError) as exc:
            self._node.get_logger().error(f"place declaration rejected: {exc}")
            return
        try:
            self._attachment_tracker.set_place_declaration(
                declaration if declaration.active else None
            )
        except ROSConfigError as exc:
            self._node.get_logger().error(str(exc))
            return
        if declaration.active:
            self._node.get_logger().info(
                f"place declaration armed target={declaration.target_id} "
                f"object={declaration.object_id or '<carried>'} "
                f"rskill={declaration.rskill_id or '<unset>'} "
                f"trace={declaration.trace_id or '<unset>'} "
                f"timeout_s={declaration.timeout_s:.1f}"
            )
        else:
            self._node.get_logger().info(
                f"place declaration retracted target={declaration.target_id}"
            )

    def _on_attachment_state(self, msg: object) -> None:
        """Apply one complete attachment snapshot to the sim perception mask."""
        from openral_core import AttachedCollisionObject
        from openral_core.exceptions import ROSConfigError

        update = getattr(self._hal, "update_attached_objects", None)
        read = getattr(self._hal, "read_attached_objects", None)
        if not callable(update) or not callable(read):
            return
        try:
            objects = [
                AttachedCollisionObject.from_idl(item)
                for item in msg.objects  # type: ignore[attr-defined]
            ]
            revision = int(msg.revision)  # type: ignore[attr-defined]
            if revision < self._attachment_revision:
                raise ValueError(
                    f"attachment revision moved backwards: {revision} < {self._attachment_revision}"
                )
            if revision == self._attachment_revision and objects == self._attachment_desired:
                return
            self._stage_attachment_objects(objects, revision=revision, update=update, read=read)
        except (ROSConfigError, ValueError, TypeError) as exc:
            self._node.get_logger().error(f"attachment state rejected by sim HAL: {exc}")

    def _stage_attachment_objects(
        self,
        objects: list[Any],
        *,
        revision: int,
        update: Any,
        read: Any,
    ) -> None:
        """Stage one atomic revision with conservative detach/attach ordering."""
        current = {obj.object_id: obj for obj in read()}
        desired_ids = {obj.object_id for obj in objects}
        # Detach ordering is conservative: unmask removed objects first while
        # the kernel still carries their old payload geometry.
        update([obj for object_id, obj in current.items() if object_id in desired_ids])
        self._attachment_desired = objects
        self._attachment_pending = objects
        self._attachment_revision = revision

    def _on_attachment_state_applied(self, msg: object) -> None:
        """Mask newly attached bodies only after the kernel accepts the revision.

        The barrier this releases exists for exactly one reason: a body that
        has just entered the perception mask is still baked into the world map,
        so motion must wait until a transparent depth frame (and the voxel
        raster behind it) has cleared it. Its trigger is therefore *new masked
        geometry*, and the mask is resolved from each object's
        ``evidence_ref`` (``SimAttachedHAL.update_attached_objects`` expands
        that body and its subtree into ``read_attached_body_ids``) — so
        comparing ``(object_id, evidence_ref)`` against the currently masked
        set decides it exactly, not approximately.

        The previous membership test was ``object_id`` *addition* only, which
        deadlocked the ADR-0097 place witness: attesting support contact
        re-publishes the SAME payload under a bumped revision, so nothing was
        "added", nothing released the barrier, and a tick the lifecycle node
        had deferred on ``attachment_action_ack_ready()`` was never
        acknowledged — the goal aborted 8 s later on a place that had in fact
        succeeded. A partial detach (2 payloads → 1) had the same shape.

        A revision that masks nothing new — an attestation-only re-publish, a
        partial detach — changes no perception geometry, so there is nothing to
        settle and the barrier releases immediately. The release is still
        guarded by :meth:`attachment_action_ack_ready` so an earlier revision's
        outstanding depth/voxel frames are never skipped.
        """
        revision = int(msg.data)  # type: ignore[attr-defined]
        if revision != self._attachment_revision or self._attachment_pending is None:
            return
        update = getattr(self._hal, "update_attached_objects", None)
        if not callable(update):
            return
        from openral_core.exceptions import ROSConfigError

        try:
            masked = {
                (obj.object_id, obj.evidence_ref)
                for obj in getattr(self._hal, "read_attached_objects", lambda: [])()
            }
            masks_new_geometry = any(
                (obj.object_id, obj.evidence_ref) not in masked for obj in self._attachment_pending
            )
            update(self._attachment_pending)
            self._attachment_applied_revision = revision
            self._attachment_pending = None
            if masks_new_geometry:
                self._attachment_depth_frames_remaining = 1 if self._depth_pubs else 0
                self._attachment_expect_voxel_update = (
                    self._node.count_publishers("/openral/world_voxels") > 0
                )
                self._node.get_logger().info(
                    "attachment perception barrier waiting for "
                    f"{self._attachment_depth_frames_remaining} transparent depth frames"
                )
                if self._attachment_depth_frames_remaining == 0:
                    self._notify_attachment_perception_ready()
            elif self.attachment_action_ack_ready():
                self._node.get_logger().info(
                    f"attachment revision {revision} masks no new geometry; "
                    "releasing the attachment perception barrier"
                )
                self._notify_attachment_perception_ready()
        except ROSConfigError as exc:
            self._node.get_logger().error(
                f"kernel accepted attachment revision {revision}, but sim mask update failed: {exc}"
            )

    def _publish_attachment_state(self) -> None:
        """Heartbeat the authoritative sim attachment set for kernel freshness."""
        if self._attachment_pub is None:
            return
        self._publish_attachment_message()

    def _observe_attachment_evidence(self) -> None:
        """Stage and publish exact MuJoCo attach/release transitions post-step."""
        if self._attachment_tracker is None or self._attachment_pending is not None:
            return
        handles = getattr(self._hal, "mujoco_handles", lambda: None)()
        update = getattr(self._hal, "update_attached_objects", None)
        read = getattr(self._hal, "read_attached_objects", None)
        if handles is None or not callable(update) or not callable(read):
            return
        _model, data = handles
        transition = self._attachment_tracker.update(
            data,
            stamp_ns=int(self._node.get_clock().now().nanoseconds),
        )
        if transition is None:
            return
        self._stage_attachment_objects(
            transition,
            revision=self._attachment_revision + 1,
            update=update,
            read=read,
        )
        object_ids = [obj.object_id for obj in transition]
        self._node.get_logger().info(
            f"automatic sim attachment revision {self._attachment_revision}: {object_ids}"
        )
        self._publish_attachment_message()

    def _publish_attachment_message(self) -> None:
        """Publish the current authoritative attachment snapshot."""
        if self._attachment_pub is None:
            return
        from openral_msgs.msg import (
            AttachedCollisionObject,
            AttachedCollisionPrimitive,
            AttachmentState,
        )

        msg = AttachmentState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.revision = self._attachment_revision
        for obj in self._attachment_desired:
            item = AttachedCollisionObject()
            obj.fill_idl(item, primitive_factory=AttachedCollisionPrimitive)
            msg.objects.append(item)
        self._fill_place_declaration(msg)
        self._attachment_pub.publish(msg)

    def _fill_place_declaration(self, msg: Any) -> None:
        """Attach the live place declaration, region and all (ADR-0097).

        Refreshed on every publication rather than snapshotted with the object
        records, because the region is expressed in the robot **base** frame —
        the frame the occupancy grid the allowance applies to lives in — and that
        frame moves under a driving base while the object records do not. Its
        accuracy is therefore bounded by this publication period; a sim place
        phase parks the base, and the kernel additionally refuses a region whose
        frame does not match the grid's.

        It rides the envelope rather than each ``AttachedCollisionObject`` for a
        second reason: this node subscribes its own topic to stage revisions, and
        that staging compares the decoded object list against the desired one. A
        per-object field refreshed every publication would make that comparison
        churn a revision on every heartbeat.
        """
        msg.place_declaration_valid = False
        if self._attachment_tracker is None:
            return
        handles = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handles is None:
            return
        _model, data = handles
        declaration = self._attachment_tracker.place_declaration(
            data,
            stamp_ns=int(self._node.get_clock().now().nanoseconds),
        )
        if declaration is None:
            return
        msg.place_declaration_valid = True
        declaration.fill_idl(msg.place_declaration)

    def _setup_depth(self) -> None:
        """Create a PointCloud2 publisher + timer per depth SensorSpec.

        Gate conditions (both must hold to publish):
        1. The manifest declares ≥1 depth/point_cloud SensorSpec with intrinsics
           (``depth_cloud.is_depth_sensor(s)`` is True for at least one sensor).
        2. The HAL exposes live MuJoCo handles
           (``hal.mujoco_handles()`` returns a non-``None`` pair).

        When either gate fails the method returns silently — no publisher,
        no timer, no TF broadcaster. This lets arm-only robots use the bridge
        without advertising any depth topics.

        Lifted from ``openral_hal_panda_mobile.lifecycle_node._setup_depth_publishers``.
        QoS matches panda_mobile's BEST_EFFORT depth QoS.
        """
        # Publish if the HAL ray-casts depth (MuJoCo) OR surfaces ready clouds in
        # obs (non-MuJoCo, e.g. the Isaac scene). Otherwise no topics.
        has_mujoco = getattr(self._hal, "mujoco_handles", lambda: None)() is not None
        has_obs_depth = callable(getattr(self._hal, "read_depth_clouds", None))
        if not (has_mujoco or has_obs_depth):
            return
        from openral_hal import depth_cloud

        depth_specs = [s for s in self._description.sensors if depth_cloud.is_depth_sensor(s)]
        if not depth_specs:
            return

        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2
        from tf2_ros import TransformBroadcaster

        if self._tf_broadcaster is None:  # may already exist (RGB camera TFs)
            self._tf_broadcaster = TransformBroadcaster(self._node)

        depth_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        # CameraInfo is low-rate + near-static: RELIABLE + TRANSIENT_LOCAL so any
        # subscriber QoS (nvblox's included) matches and late joiners get it.
        info_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        for spec in depth_specs:
            base = f"/openral/cameras/{spec.name}"
            self._depth_pubs[spec.name] = self._node.create_publisher(
                PointCloud2, f"{base}/points", depth_qos
            )
            # Dense depth image + CameraInfo for nvblox's depth integrator.
            self._depth_image_pubs[spec.name] = self._node.create_publisher(
                Image, f"{base}/depth/image", depth_qos
            )
            self._depth_info_pubs[spec.name] = self._node.create_publisher(
                CameraInfo, f"{base}/depth/camera_info", info_qos
            )
        self._depth_timer = self._node.create_timer(
            1.0 / max(self._depth_rate_hz, 1.0), self._publish_depth_clouds
        )
        self._node.get_logger().info(
            f"SimSensorBridge: publishing {len(depth_specs)} depth camera(s) "
            "(PointCloud2 + 32FC1 depth image + CameraInfo): "
            + ", ".join(s.name for s in depth_specs)
            + f" @ {self._depth_rate_hz:.1f} Hz"
        )

    def _resolve_depth_base_body(self, model: object) -> None:
        """Resolve + cache the MJCF base bodies (self-exclusion **and** TF parent).

        Two distinct bodies, deliberately (ADR-0095):

        * ``_depth_base_body`` — the chassis root (``mobilebase0_base`` under a
          composed kitchen, bare ``"base"`` on synthetic MJCFs). This is the
          ``mj_multiRay`` body-exclude anchor, so rays don't strike the camera's
          own mount at range ~0.
        * ``_base_frame_body`` — the body ``base_frame`` denotes on ``/tf`` (the
          0.70 m arm mount ``mobilebase0_support`` on a robosuite mobile base).
          Every ``base_frame -> …`` extrinsic is measured against this one, so
          the published cloud lands where TF says it does. Identical to the
          chassis root on fixed-base arms.

        Also populates ``_depth_self_bodies`` — the robot's own MJCF body ids
        for the depth self-filter, derived from the manifest's sim_joint_name
        prefixes (arm + base + gripper).
        """
        import mujoco  # reason: defer optional sim dep

        from openral_hal.depth_cloud import (
            resolve_base_body_name,
            resolve_base_frame_body_name,
            robot_self_body_ids,
        )

        description = getattr(self._hal, "description", None)
        base_body = resolve_base_body_name(model, description=description)
        self._depth_base_body = base_body
        self._depth_base_body_id = (
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body))
            if base_body is not None
            else -1
        )
        self._base_frame_body = resolve_base_frame_body_name(model, description=description)
        # Robot self-body set (arm + base + gripper) for the depth self-filter.

        sim_names = (
            [j.sim_joint_name for j in description.joints] if description is not None else []
        )
        self._depth_self_bodies = robot_self_body_ids(model, sim_names)
        self._node.get_logger().info(
            "SimSensorBridge: depth self-filter "
            f"base_body={self._depth_base_body!r} "
            f"base_frame_body={self._base_frame_body!r} "
            f"robot_bodies={len(self._depth_self_bodies)}"
        )

    def _render_size(self) -> tuple[int, int] | None:
        """Resolution the scene actually rendered its RGB frames at, or ``None``.

        deploy-sim scenes render the same MuJoCo camera at
        ``scene.observation_width``/``height`` (e.g. 128 or 640), which can
        differ from the manifest's nominal intrinsics resolution. The depth
        synth must back-project at the render resolution so its cloud lines up
        with the RGB the detector ran on; this reads the live rendered RGB frame
        shape (the ground truth of what robosuite rendered) and returns
        ``(width, height)`` so :func:`depth_synth_kwargs` can rescale the
        intrinsics. Returns ``None`` when no frame is available yet (the synth
        then falls back to the manifest's nominal intrinsics).
        """
        reader = getattr(self._hal, "read_images", None)
        if reader is None:
            return None
        images = reader()
        if not isinstance(images, dict):
            return None
        for arr in images.values():
            shape = getattr(arr, "shape", None)
            if shape is not None and len(shape) == _IMAGE_DIM:
                h, w = int(shape[0]), int(shape[1])
                if w > 0 and h > 0:
                    return (w, h)
        return None

    def _publish_depth_clouds(self) -> None:  # noqa: PLR0915  # reason: one atomic per-camera publish transaction
        """Ray-cast + publish a PointCloud2 (+ depth image) per camera, and its TF.

        The deploy-sim source for octomap_server. Each depth ``SensorSpec`` is
        ray-cast **once** per frame with
        :func:`openral_sim.backends.depth_camera.synthesize_depth_image` into a
        dense metric-depth raster, which is published three ways: as the
        ``32FC1`` ``sensor_msgs/Image`` (+ ``CameraInfo``) nvblox's projective
        integrator consumes, and — back-projected by
        :func:`openral_hal.depth_cloud.points_from_depth_grid` — as the
        camera-optical-frame ``sensor_msgs/PointCloud2`` octomap_server lifts
        into the world map, alongside a live
        ``base_link -> <camera>_optical_frame`` TF. A camera whose MJCF name
        doesn't resolve is disabled after one warning (sim sensor, not a safety
        path).

        The cast is the whole cost of this timer (~60 ms for a 256x256 camera at
        the default ``stride=4`` on a ~1200-geom kitchen), it runs on the single
        ``rclpy.spin`` thread, and ``depth_publish_rate_hz`` defaults to 10 Hz —
        a 100 ms period. So the raster is the primitive and every output is
        derived from it; synthesising the cloud separately cast every ray twice
        for numbers that are equal by construction.

        Lifted from ``openral_hal_panda_mobile.lifecycle_node._publish_depth_clouds``.
        The self-body exclusion, the range filtering, and the TF broadcast are
        unchanged.
        """
        if not self._depth_pubs:
            return
        handle = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handle is None:
            # Non-MuJoCo backend: publish the base_link clouds the HAL surfaces.
            self._publish_depth_clouds_from_obs()
            return
        model, data = handle

        from geometry_msgs.msg import (
            TransformStamped,
        )
        from openral_core.exceptions import ROSConfigError
        from openral_sim.backends.depth_camera import synthesize_depth_frame

        from openral_hal.depth_cloud import (
            camera_info_from_intrinsics,
            camera_optical_tf_to_base,
            depth_image_from_grid,
            depth_synth_kwargs,
            is_depth_sensor,
            pointcloud2_from_points_xyz,
            points_from_depth_grid,
        )

        if self._depth_base_body is None and self._depth_base_body_id < 0:
            self._resolve_depth_base_body(model)
        exclude_id = self._depth_base_body_id if self._depth_base_body_id >= 0 else None
        excluded_bodies = self._depth_excluded_body_ids()

        max_range_default = float(self._depth_max_range_m)
        stride = max(int(self._depth_pixel_stride), 1)
        stamp = self._node.get_clock().now().to_msg()

        base_frame_id = getattr(self._description, "base_frame", "base_link")
        specs = {s.name: s for s in self._description.sensors if is_depth_sensor(s)}
        for name, pub in self._depth_pubs.items():
            if name in self._depth_disabled:
                continue
            spec = specs.get(name)
            if spec is None:
                continue
            try:
                kwargs = depth_synth_kwargs(
                    spec,
                    max_range_default=max_range_default,
                    render_size=self._render_size(),
                )
                # The ONE ray-cast of this frame: a dense 32FC1 raster (every
                # pixel, 0.0 = no return) at the strided resolution, so the
                # CameraInfo intrinsics scale by 1/stride to match it — plus
                # the self-filter's clearing mask, which the raster's 0.0
                # sentinel cannot express and which OctoMap needs to clear the
                # cells the robot's own body occludes.
                depth_grid, clearing = synthesize_depth_frame(
                    model=model,
                    data=data,
                    stride=stride,
                    exclude_body_id=exclude_id,
                    exclude_body_ids=excluded_bodies or None,
                    **kwargs,
                )
                h_eff, w_eff = (int(depth_grid.shape[0]), int(depth_grid.shape[1]))
                # The raster's own intrinsics: the manifest's, scaled by 1/stride.
                intr = {k: float(kwargs[k]) / stride for k in ("fx", "fy", "cx", "cy")}
                # octomap's cloud is that same raster back-projected through the
                # same intrinsics — not a second cast of the same rays.
                points = points_from_depth_grid(
                    depth_grid,
                    clearing=clearing,
                    max_range_m=float(kwargs["max_range_m"]),
                    **intr,
                )
                cloud = pointcloud2_from_points_xyz(points, frame_id=spec.frame_id, stamp=stamp)
                pub.publish(cloud)
                if self._attachment_depth_frames_remaining > 0:
                    # One transparent cloud has now gone out: the attachment
                    # perception barrier can count this frame.
                    self._record_attachment_depth_frame(
                        stamp_ns=int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                    )
                # Dense 32FC1 depth image + CameraInfo for nvblox, from that same
                # single raster (the barrier used to skip this publish only to
                # avoid a second cast, which no longer exists).
                self._depth_image_pubs[name].publish(
                    depth_image_from_grid(depth_grid, frame_id=spec.frame_id, stamp=stamp)
                )
                self._depth_info_pubs[name].publish(
                    camera_info_from_intrinsics(
                        width=w_eff,
                        height=h_eff,
                        **intr,
                        frame_id=spec.frame_id,
                        stamp=stamp,
                    )
                )
                if self._base_frame_body is not None and self._tf_broadcaster is not None:
                    xyz, quat = camera_optical_tf_to_base(
                        model=model,
                        data=data,
                        camera_name=kwargs["camera_name"],
                        base_body_name=self._base_frame_body,
                    )
                    tf = TransformStamped()
                    tf.header.stamp = stamp
                    tf.header.frame_id = base_frame_id
                    tf.child_frame_id = spec.frame_id
                    tf.transform.translation.x = xyz[0]
                    tf.transform.translation.y = xyz[1]
                    tf.transform.translation.z = xyz[2]
                    tf.transform.rotation.x = quat[0]
                    tf.transform.rotation.y = quat[1]
                    tf.transform.rotation.z = quat[2]
                    tf.transform.rotation.w = quat[3]
                    self._tf_broadcaster.sendTransform(tf)
            except ROSConfigError as exc:
                self._depth_disabled.add(name)
                self._node.get_logger().warning(
                    f"depth camera {name!r} disabled: {exc}; "
                    "check the SensorSpec's mjcf_camera metadata."
                )

    def attachment_action_ack_ready(self) -> bool:
        """Return whether attachment geometry and transparent depth are settled."""
        pending_addition = self._attachment_pending is not None and bool(self._attachment_pending)
        return (
            not pending_addition
            and self._attachment_depth_frames_remaining == 0
            and self._attachment_voxel_updates_remaining == 0
        )

    def _notify_attachment_perception_ready(self) -> None:
        """Release a deferred action acknowledgement after map-clearing frames."""
        callback = self._on_attachment_perception_ready
        if callable(callback):
            callback()

    def _record_attachment_depth_frame(self, *, stamp_ns: int) -> None:
        """Count one transparent depth frame toward attachment-map readiness."""
        if self._attachment_depth_frames_remaining <= 0:
            return
        self._attachment_depth_frames_remaining -= 1
        self._node.get_logger().info(
            "attachment perception barrier depth frame; "
            f"remaining={self._attachment_depth_frames_remaining}"
        )
        if self._attachment_depth_frames_remaining == 0:
            self._attachment_transparent_depth_stamp_ns = stamp_ns
            if self._attachment_expect_voxel_update:
                self._attachment_voxel_updates_remaining = 1
            else:
                self._notify_attachment_perception_ready()

    def _on_attachment_world_voxels(self, msg: object) -> None:
        """Wait for post-depth OctoMap rasterizations before releasing motion.

        Also retains the grid's *geometry* (never its occupancy bytes), which
        is what turns a ``b=voxel_<n>`` evidence line into a position the
        ground-truth record can interrogate — see :meth:`_evidence_voxel`.
        """
        self._last_voxel_grid = {
            "origin": (
                float(msg.origin.x),  # type: ignore[attr-defined]  # reason: ROS subscription type
                float(msg.origin.y),  # type: ignore[attr-defined]  # reason: ROS subscription type
                float(msg.origin.z),  # type: ignore[attr-defined]  # reason: ROS subscription type
            ),
            "resolution": float(msg.resolution),  # type: ignore[attr-defined]  # reason: ROS subscription type
            "size": (
                int(msg.size_x),  # type: ignore[attr-defined]  # reason: ROS subscription type
                int(msg.size_y),  # type: ignore[attr-defined]  # reason: ROS subscription type
                int(msg.size_z),  # type: ignore[attr-defined]  # reason: ROS subscription type
            ),
        }
        if self._attachment_voxel_updates_remaining <= 0:
            return
        header = msg.header  # type: ignore[attr-defined]  # reason: ROS subscription type
        source_stamp_ns = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
        transparent_stamp_ns = self._attachment_transparent_depth_stamp_ns
        if transparent_stamp_ns is None or source_stamp_ns <= transparent_stamp_ns:
            return
        self._attachment_voxel_updates_remaining -= 1
        if self._attachment_voxel_updates_remaining == 0:
            self._attachment_expect_voxel_update = False
            self._attachment_transparent_depth_stamp_ns = None
            self._notify_attachment_perception_ready()

    # -- E-stop ground truth (diagnostics only) --
    def _setup_estop_ground_truth(self) -> None:
        """Subscribe the topics one adjudicable stop record needs.

        Gate: live MuJoCo handles (there is no ground truth without a
        simulator). Deliberately NOT gated on attachment support — the
        pre-grasp arm↔world stop is the class this closes.

        * ``/openral/estop`` — fires the snapshot at the stop instant.
        * ``/openral/candidate_action`` — the chunk the kernel was checking
          (the only reconstruction input for a PREDICTED-horizon stop).
        * ``/openral/failure/safety`` — the kernel's own
          :class:`~openral_core.CollisionEvidence`, which carries
          ``horizon_step`` / ``link_a`` / ``min_distance_m``.
        * ``/openral/world_voxels`` — the grid geometry that turns that
          evidence's ``b=voxel_<n>`` index into a position, so the record can
          report what backs the cell (subscribed here only when the attachment
          bridge has not already claimed it).

        Diagnostics only: nothing here gates, delays, or alters actuation.
        """
        if getattr(self._hal, "mujoco_handles", lambda: None)() is None:
            return
        from openral_msgs.msg import ActionChunk, FailureTrigger
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from std_msgs.msg import Empty

        estop_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self._estop_sub = self._node.create_subscription(
            Empty,
            "/openral/estop",
            self._on_estop_ground_truth,
            estop_qos,
        )
        # Depth 50 matches the kernel's chunk/failure QoS (and the node's own
        # ``/openral/safe_action`` subscription) so a multi-slot tick cannot
        # silently drop the very chunk that was checked. ``/openral/safe_action``
        # cannot serve here: a REJECTED chunk is never republished on it, so
        # the candidate bus is the only place the stopped motion exists.
        bus_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=50,
        )
        self._candidate_action_sub = self._node.create_subscription(
            ActionChunk,
            "/openral/candidate_action",
            self._on_candidate_action,
            bus_qos,
        )
        self._safety_failure_sub = self._node.create_subscription(
            FailureTrigger,
            "/openral/failure/safety",
            self._on_safety_failure,
            bus_qos,
        )
        if self._attachment_voxel_sub is None:
            # A world-voxel stop names its cell as an INDEX; without the grid's
            # geometry that index cannot be turned into a position, and the
            # record is left unable to say what backs the cell. The attachment
            # bridge owns this subscription when the HAL supports attachment;
            # this covers the HALs that do not, so the pre-grasp arm↔world
            # stop this method exists for is never the one missing it.
            from openral_msgs.msg import OccupancyVoxels

            self._attachment_voxel_sub = self._node.create_subscription(
                OccupancyVoxels,
                "/openral/world_voxels",
                self._on_attachment_world_voxels,
                QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.VOLATILE,
                    depth=1,
                ),
            )
        self._node.get_logger().info(
            "SimSensorBridge: e-stop ground truth armed (MuJoCo contacts + near-miss "
            f"distances + last {_CANDIDATE_CHUNK_HISTORY} candidate chunks + "
            "world-voxel backing)."
        )

    def _teardown_estop_ground_truth(self) -> None:
        """Destroy the stop-record subscriptions and drop their caches."""
        for sub in (self._estop_sub, self._candidate_action_sub, self._safety_failure_sub):
            if sub is not None:
                self._node.destroy_subscription(sub)
        self._estop_sub = None
        self._candidate_action_sub = None
        self._safety_failure_sub = None
        self._last_voxel_grid = None
        self._candidate_chunks.clear()
        self._last_collision_evidence = None
        self._last_collision_evidence_ns = 0
        self._collision_evidence_warned = False
        self._estop_awaiting_evidence = False

    def _on_candidate_action(self, msg: object) -> None:
        """Cache one candidate chunk digest for predicted-horizon reconstruction."""
        header = msg.header  # type: ignore[attr-defined]  # reason: ROS subscription type
        self._candidate_chunks.append(
            candidate_chunk_digest(
                stamp_ns=int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec),
                control_mode=int(msg.control_mode),  # type: ignore[attr-defined]
                horizon=int(msg.horizon),  # type: ignore[attr-defined]
                n_dof=int(msg.n_dof),  # type: ignore[attr-defined]
                flat=list(msg.flat),  # type: ignore[attr-defined]
                cartesian_delta_scale=list(msg.cartesian_delta_scale),  # type: ignore[attr-defined]
                ee_name=str(msg.ee_name),  # type: ignore[attr-defined]
                frame_id=str(msg.frame_id),  # type: ignore[attr-defined]
                rskill_id=str(msg.rskill_id),  # type: ignore[attr-defined]
                trace_id=str(msg.trace_id),  # type: ignore[attr-defined]
                tick_index=int(msg.tick_index),  # type: ignore[attr-defined]
            )
        )

    def _on_safety_failure(self, msg: object) -> None:
        """Cache the kernel's collision evidence; emit it late if it lost the race.

        The kernel publishes the failure trigger and then the E-stop, but
        cross-topic delivery order is not guaranteed. The snapshot is never
        delayed for the evidence (the sim state must be captured at the stop
        instant); when it arrives afterwards it is emitted as its own line,
        joined to the snapshot by ``stop_seq``.
        """
        from openral_msgs.msg import FailureTrigger

        if int(msg.kind) != FailureTrigger.KIND_COLLISION:  # type: ignore[attr-defined]
            return
        from openral_core import CollisionEvidence
        from pydantic import ValidationError

        try:
            evidence = CollisionEvidence.model_validate_json(
                str(msg.evidence_json)  # type: ignore[attr-defined]
            )
        except ValidationError as exc:
            if not self._collision_evidence_warned:
                self._collision_evidence_warned = True
                self._node.get_logger().warning(
                    f"collision evidence not decodable as CollisionEvidence: {exc}"
                )
            return
        header = msg.header  # type: ignore[attr-defined]  # reason: ROS subscription type
        stamp_ns = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
        self._last_collision_evidence_ns = stamp_ns
        self._last_collision_evidence = {
            "stamp_ns": stamp_ns,
            "rskill_id": str(msg.rskill_id),  # type: ignore[attr-defined]
            "trace_id": str(msg.trace_id),  # type: ignore[attr-defined]
            **evidence.model_dump(mode="json"),
        }
        if self._estop_awaiting_evidence:
            self._estop_awaiting_evidence = False
            self._node.get_logger().error(
                "sim.estop_ground_truth_evidence "
                + json.dumps(
                    {
                        "stop_seq": self._estop_seq,
                        "collision_evidence": self._last_collision_evidence,
                    },
                    sort_keys=True,
                )
            )

    def _on_estop_ground_truth(self, _msg: object) -> None:
        """Log MuJoCo ground truth for EVERY kernel stop, attached or not.

        Supersedes the attached-payload-only ``sim.attached_payload_estop_snapshot``
        line (no in-tree consumer keyed on that name), which early-returned
        whenever nothing was carried — so the pre-grasp arm↔world stops that
        dominate real runs produced no ground truth at all and could not be
        adjudicated real-vs-false.

        Emits one ``sim.estop_ground_truth_snapshot`` JSON line:
        :func:`estop_ground_truth_snapshot` (contacts, near-miss distances,
        joint state, base TF) plus the cached candidate chunks and the
        kernel's collision evidence when it has already landed. Diagnostics
        only — no gating, no actuation effect.

        READING THE RECORD: an empty ``robot_world_contacts`` is NOT
        "nothing was touching". MuJoCo's contype/conaffinity exclusions can
        suppress a contact even at deep interpenetration (field-observed: an
        arm 30 mm inside a freezer door with ``ncon == 0``). Adjudicate a
        stop with the ``nearest_*_pairs`` probes; the contact list only ever
        confirms, never refutes. Those probes measure only against SOLID world
        geometry — a geom with neither ``contype`` nor ``conaffinity`` is a
        marker, and measuring against one manufactured the physically
        meaningless "payload 134 mm inside ``cab_1_left_group_reg_main``" of
        rounds 5/6. The record repeats both facts as ``contacts_caveat`` so a
        single grepped line stays self-explaining.
        """
        handles = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handles is None:
            return
        model, data = handles
        if not self._depth_self_bodies:
            # Cameras/depth may never have run (they own the lazy resolve).
            self._resolve_depth_base_body(model)
        attached = self._depth_excluded_body_ids() - self._depth_self_bodies
        # Rank the near-miss probes over the links the KERNEL checks. Probing
        # the whole robot buries the arm under the base's 0-2 mm floor
        # contact — geometry the manifest deliberately leaves out of
        # collision_geometry — which nearly produced a wrong field verdict.
        probe_bodies = kernel_checked_body_ids(model, self._description) or None
        snapshot = estop_ground_truth_snapshot(
            model,
            data,
            robot_body_ids=self._depth_self_bodies,
            attached_body_ids=attached,
            probe_body_ids=probe_bodies,
            base_frame_body=self._base_frame_body,
            joint_state=self._read_joint_state(),
            description=self._description,
            evidence_voxel=self._evidence_voxel(),
        )
        now_ns = int(self._node.get_clock().now().nanoseconds)
        evidence = self._last_collision_evidence
        age_ns = now_ns - self._last_collision_evidence_ns
        fresh = evidence is not None and 0 <= age_ns <= _ESTOP_EVIDENCE_WINDOW_NS
        self._estop_seq += 1
        self._estop_awaiting_evidence = not fresh
        self._node.get_logger().error(
            "sim.estop_ground_truth_snapshot "
            + json.dumps(
                {
                    "stop_seq": self._estop_seq,
                    "stamp_ns": now_ns,
                    "attachment_revision": self._attachment_revision,
                    **snapshot,
                    "candidate_action_chunks": list(self._candidate_chunks),
                    "collision_evidence": evidence if fresh else None,
                },
                sort_keys=True,
            )
        )
        self._log_initial_configuration_stop(snapshot)

    def _log_initial_configuration_stop(self, snapshot: dict[str, object]) -> None:
        """Name a stop that fired before the robot was ever commanded.

        The snapshot line above records *what* the kernel refused; without this
        one it reads exactly like a mid-task stop, and the 2026-08-22
        ``PickPlaceFridgeShelfToDrawer`` round cost a debugging cycle on a
        policy that had not yet applied a single chunk — the arm was already
        24.7 mm inside the fridge's freezer door at the reset pose. Emitting
        ``sim.estop_initial_configuration`` puts "the scene spawned the robot
        in collision" in the artifacts as its own grep-able line.

        Diagnostics only (CLAUDE.md §1.4): nothing here gates, delays, or
        alters the stop, which is correct and must stand.
        """
        record = initial_configuration_stop_record(
            snapshot,
            stop_seq=self._estop_seq,
            last_action_ns=int(getattr(self._hal, "last_action_ns", 0)),
            candidate_chunks_seen=len(self._candidate_chunks),
        )
        if record is None:
            return
        self._node.get_logger().error(
            "sim.estop_initial_configuration " + json.dumps(record, sort_keys=True)
        )

    def _evidence_voxel(self) -> dict[str, object] | None:
        """The world-voxel cell of the pending stop, located in the live grid.

        A ``safety.collision`` line names the cell only as ``voxel_<n>`` — an
        index into a grid the kernel does not republish. Pairing that index
        with the geometry of the last ``/openral/world_voxels`` message is what
        lets :func:`voxel_backing_record` ask MuJoCo what is actually there.
        Returns ``None`` when the stop was not a world-voxel stop, when no grid
        has been seen, or when the evidence is too old to attribute.
        """
        evidence = self._last_collision_evidence
        grid = self._last_voxel_grid
        if evidence is None or grid is None:
            return None
        name = evidence.get("link_b_or_object")
        if not isinstance(name, str) or not name.startswith("voxel_"):
            return None
        index = name.removeprefix("voxel_")
        if not index.isdigit():
            return None
        return {"index": int(index), **grid}

    def _read_joint_state(self) -> Any:
        """The HAL's joint state at the stop, or ``None`` if it cannot be read.

        This is the same vector the kernel seeded ``q_meas`` from (the HAL
        publishes it on ``/joint_states``), so it is the FK seed an offline
        reconstruction of a predicted-horizon stop needs.
        """
        from openral_core.exceptions import ROSError

        read_state = getattr(self._hal, "read_state", None)
        if not callable(read_state):
            return None
        try:
            return read_state()
        except ROSError as exc:  # reason: a stop record must survive a HAL read fault
            self._node.get_logger().warning(f"e-stop snapshot has no joint state: {exc}")
            return None

    def _depth_excluded_body_ids(self) -> frozenset[int]:
        """Robot and attached-payload bodies excluded from world perception."""
        read_attached_ids = getattr(self._hal, "read_attached_body_ids", None)
        attached_ids = (
            frozenset(int(body_id) for body_id in read_attached_ids())
            if callable(read_attached_ids)
            else frozenset()
        )
        return self._depth_self_bodies | attached_ids

    def _publish_depth_clouds_from_obs(self) -> None:
        """Publish the HAL's ready ``base_link`` clouds as ``PointCloud2``.

        Non-MuJoCo path: the backend (Isaac scene) already deprojected each depth
        camera to a ``(N, 3)`` cloud in ``base_link`` (Isaac owns the camera
        convention), surfaced via ``hal.read_depth_clouds()``. We just wrap each in
        a ``PointCloud2`` stamped ``base_link`` — no ray-cast, no per-camera optical
        TF (``base_link`` is already on /tf via /odom). octomap lifts it into the
        world map exactly as it does the MuJoCo clouds.
        """
        read = getattr(self._hal, "read_depth_clouds", None)
        if not callable(read):
            return
        clouds = read()
        if not clouds:
            return
        from openral_hal.depth_cloud import pointcloud2_from_points_xyz

        base_frame_id = getattr(self._description, "base_frame", "base_link")
        stamp = self._node.get_clock().now().to_msg()
        for name, pub in self._depth_pubs.items():
            pts = clouds.get(name)
            if pts is None or pts.size == 0:
                continue
            cloud = pointcloud2_from_points_xyz(pts, frame_id=base_frame_id, stamp=stamp)
            pub.publish(cloud)

    # -- Viewer --
    def _setup_viewer(self) -> None:
        handles = getattr(self._hal, "mujoco_handles", lambda: None)()
        if not self._viewer_enabled or handles is None:
            return
        model, data = handles
        try:
            import mujoco.viewer  # reason: optional dep — robosuite/mujoco ships it

            # Hide both side panels (left settings / right info) so the window
            # shows only the simulation render — the panels are still reachable
            # at runtime via Tab / Shift+Tab.
            self._viewer = mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False
            )
            # Mirror RoboCasa's offscreen camera render config: hide geom group 0
            # (the collision shell, which RoboCasa colours dark red by convention)
            # and show group 1 (the textured visual geoms). The passive viewer
            # renders ALL groups by default, so without this the kitchen shows up
            # as a red collision box even though the /camera streams (group 1
            # only) render correctly.
            with contextlib.suppress(Exception):
                self._viewer.opt.geomgroup[0] = 0  # collision — hide
                self._viewer.opt.geomgroup[1] = 1  # visual — show
            # Open the viewer on a 3rd-person scene camera (agentview/top/…),
            # falling back to the base-aligned free camera for camera-less twins.
            self._aim_viewer_camera(model, data)
        except Exception as exc:  # reason: GL/DISPLAY failures are non-fatal (headless)
            self._node.get_logger().warning(
                f"SimSensorBridge: viewer launch failed ({exc!s}); continuing headless. "
                "Common causes: no DISPLAY, MUJOCO_GL=egl (use 'glfw'), missing libglfw/libGL."
            )
            self._viewer = None
            return
        self._viewer_timer = self._node.create_timer(
            1.0 / max(self._viewer_sync_rate_hz, 1.0), self._sync_viewer
        )
        self._node.get_logger().info(
            f"SimSensorBridge: MuJoCo viewer open @ {self._viewer_sync_rate_hz:.1f} Hz."
        )

    def _aim_viewer_camera(self, model: Any, data: Any) -> None:
        """Set the viewer's opening **free-camera** pose (mouse stays live).

        Sets the initial viewpoint via
        :func:`openral_hal.depth_cloud.initial_viewer_camera` — eye at the
        authored overview camera (``agentview`` / ``top`` / …) with the orbit
        pivot on the robot base, else the base-aligned default. The camera stays
        ``mjCAMERA_FREE`` so the user can drag to orbit and scroll to zoom; we
        only set the initial view. Best effort: any failure leaves the default.
        """
        if self._viewer is None:
            return
        with contextlib.suppress(Exception):
            import mujoco  # reason: optional sim dep

            from openral_hal.depth_cloud import initial_viewer_camera

            lookat, distance, azimuth, elevation = initial_viewer_camera(
                model=model, data=data, description=getattr(self._hal, "description", None)
            )
            with self._viewer.lock():
                cam = self._viewer.cam
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = lookat
                cam.distance = distance
                cam.azimuth = azimuth
                cam.elevation = elevation

    def _sync_viewer(self) -> None:
        if self._viewer is None:
            return
        try:
            self._viewer.sync()
        except Exception as exc:  # reason: viewer closed by user
            self._node.get_logger().warning(f"viewer sync failed; closing: {exc!s}")
            if self._viewer_timer is not None:
                self._viewer_timer.cancel()
                self._viewer_timer = None
            self._viewer = None

    # -- offscreen cinecam recorder (website-video capture) --
    def _configure_cinecam_camera(self, mujoco: Any, model: Any, data: Any) -> Any:
        """Build the free-camera pose from the viewer default + env overrides.

        Resolves the opening pose via :func:`initial_viewer_camera`, then applies
        absolute overrides (``OPENRAL_CINECAM_AZ_DEG`` / ``_EL_DEG`` / ``_DIST_M``)
        and deltas (``_AZ_OFFSET_DEG`` / ``_EL_OFFSET_DEG`` / ``_DIST_DELTA_M``),
        resolves the base body for the follow-cam, and snapshots the final pose
        as the baseline for the live ``OPENRAL_CINECAM_TUNE`` deltas.
        """
        import os  # reason: env-gated capture feature

        from openral_hal.depth_cloud import initial_viewer_camera, resolve_base_body_name

        cam = mujoco.MjvCamera()
        lookat, distance, azimuth, elevation = initial_viewer_camera(
            model=model, data=data, description=getattr(self._hal, "description", None)
        )
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        for env_key, attr in (
            ("OPENRAL_CINECAM_AZ_DEG", "azimuth"),
            ("OPENRAL_CINECAM_EL_DEG", "elevation"),
            ("OPENRAL_CINECAM_DIST_M", "distance"),
        ):
            val = os.environ.get(env_key)
            if val:
                setattr(cam, attr, float(val))
        az_off = os.environ.get("OPENRAL_CINECAM_AZ_OFFSET_DEG")
        if az_off:
            cam.azimuth = float(cam.azimuth) + float(az_off)
        el_off = os.environ.get("OPENRAL_CINECAM_EL_OFFSET_DEG")  # +ve = less top-down
        if el_off:
            cam.elevation = float(cam.elevation) + float(el_off)
        dist_delta = os.environ.get("OPENRAL_CINECAM_DIST_DELTA_M")  # -ve = closer
        if dist_delta:
            cam.distance = max(0.3, float(cam.distance) + float(dist_delta))
        self._cinecam_base_body = resolve_base_body_name(
            model, description=getattr(self._hal, "description", None)
        )
        self._cinecam_setup_az = float(cam.azimuth)
        self._cinecam_setup_el = float(cam.elevation)
        self._cinecam_setup_dist = float(cam.distance)
        return cam

    def _setup_cinecam(self) -> None:
        """Render the pulled-back free-camera view to JPGs when OPENRAL_CINECAM_DIR is set.

        Offscreen (EGL) render — robust against the onscreen GLFW viewer being
        unmapped/throttled by the desktop WM. Frame pose matches the viewer
        (:func:`openral_hal.depth_cloud.initial_viewer_camera`); collision shells
        are hidden so RoboCasa textures show. ``OPENRAL_CINECAM_SIZE`` (``WxH``,
        default ``1280x960``) and ``OPENRAL_CINECAM_FPS`` (default ``12``) tune it.
        """
        import os  # reason: env-gated debug/capture feature

        out_dir = os.environ.get("OPENRAL_CINECAM_DIR")
        if not out_dir:
            return
        handles = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handles is None:
            return
        model, data = handles
        try:
            import mujoco  # reason: optional sim dep

            from openral_hal.depth_cloud import apply_robosuite_visual_geomgroups

            size = os.environ.get("OPENRAL_CINECAM_SIZE", "1280x960")
            width, height = (int(v) for v in size.lower().split("x"))
            fps = float(os.environ.get("OPENRAL_CINECAM_FPS", "12"))
            os.makedirs(out_dir, exist_ok=True)
            self._cinecam_out_dir = out_dir
            # The MJCF offscreen framebuffer defaults to 640x480; enlarge it so
            # the cinecam can render at the requested (higher) resolution.
            with contextlib.suppress(Exception):
                model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
                model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
            try:
                self._cinecam_renderer = mujoco.Renderer(model, height=height, width=width)
            except Exception:  # reason: offscreen framebuffer smaller than request
                width = int(model.vis.global_.offwidth)
                height = int(model.vis.global_.offheight)
                self._cinecam_renderer = mujoco.Renderer(model, height=height, width=width)
            self._cinecam_model = model
            self._cinecam_w = width
            self._cinecam_h = height
            self._cinecam_opt = mujoco.MjvOption()
            apply_robosuite_visual_geomgroups(self._cinecam_opt, model)
            self._cinecam_cam = self._configure_cinecam_camera(mujoco, model, data)
        except Exception as exc:  # reason: GL/render failure is non-fatal
            self._node.get_logger().warning(
                f"SimSensorBridge: cinecam setup failed ({exc!s}); no offscreen capture."
            )
            self._cinecam_renderer = None
            return
        self._cinecam_timer = self._node.create_timer(1.0 / max(fps, 1.0), self._render_cinecam)
        self._node.get_logger().info(
            f"SimSensorBridge: cinecam recording {width}x{height} @ {fps:.0f} Hz → {out_dir}"
        )

    def _render_cinecam(self) -> None:
        if self._cinecam_renderer is None or self._cinecam_cam is None:
            return
        handles = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handles is None:
            return
        model, data = handles
        try:
            import mujoco  # reason: optional sim dep
            from PIL import Image  # reason: optional dep — present in the sim env

            # robosuite rebuilds its sim (a fresh MjModel) on env reset, so the
            # renderer bound to the setup-time model would render a frozen scene.
            # Rebuild it (same size + scene opts) whenever the live model changes.
            if model is not self._cinecam_model:
                with contextlib.suppress(Exception):
                    self._cinecam_renderer.close()
                self._cinecam_renderer = mujoco.Renderer(
                    model, height=self._cinecam_h, width=self._cinecam_w
                )
                from openral_hal.depth_cloud import apply_robosuite_visual_geomgroups

                apply_robosuite_visual_geomgroups(self._cinecam_opt, model)
                self._cinecam_model = model
            # Ensure derived kinematics (geom_xpos) reflect the latest qpos.
            mujoco.mj_forward(model, data)
            # Follow-cam: re-pin lookat to the live base position so a navigating
            # robot stays centred (OPENRAL_CINECAM_FOLLOW=1). Lift the pivot to
            # ~torso height for a nicer frame.
            import os  # reason: env-gated

            if os.environ.get("OPENRAL_CINECAM_FOLLOW") and self._cinecam_base_body:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._cinecam_base_body)
                if bid >= 0:
                    self._cinecam_cam.lookat[0] = float(data.xpos[bid][0])
                    self._cinecam_cam.lookat[1] = float(data.xpos[bid][1])
                    self._cinecam_cam.lookat[2] = float(data.xpos[bid][2]) + 0.5
            # Live tuning: re-read "az_delta el_delta dist_delta" from the tune
            # file each tick so framing can be dialed in without a relaunch.
            tune_path = os.environ.get("OPENRAL_CINECAM_TUNE")
            if tune_path and os.path.exists(tune_path):
                with contextlib.suppress(Exception):
                    with open(tune_path) as _tf:
                        parts = _tf.read().split()
                    az_d, el_d, dist_d = (float(parts[0]), float(parts[1]), float(parts[2]))
                    self._cinecam_cam.azimuth = self._cinecam_setup_az + az_d
                    self._cinecam_cam.elevation = self._cinecam_setup_el + el_d
                    self._cinecam_cam.distance = max(0.3, self._cinecam_setup_dist + dist_d)
            self._cinecam_renderer.update_scene(
                data, camera=self._cinecam_cam, scene_option=self._cinecam_opt
            )
            rgb = self._cinecam_renderer.render()
            self._cinecam_frame += 1
            path = f"{self._cinecam_out_dir}/f_{self._cinecam_frame:05d}.jpg"
            Image.fromarray(rgb).save(path, quality=88)
        except Exception as exc:  # reason: a dropped frame must not crash the HAL
            with contextlib.suppress(Exception):
                self._node.get_logger().warning(f"cinecam frame failed: {exc!s}")
