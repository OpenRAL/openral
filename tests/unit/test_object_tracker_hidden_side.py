"""A provisional box grows to the object's size where the robot hides it (#418, OT-6/OT-14).

Real frame: ``fixtures/pick_orange_hidden_side.npz``, the PickOrange top_depth frame at sim
5.63 s of trial r6po_dbg4 (Spark, 2026-10-17; depth, robot mask, intrinsics, camera pose,
cropped), the voxel grid then (15 mm) and the tracked boxes. orange_3 (trk:3) is PROVISIONAL
under the open jaw: its box is 5.2 cm on its short axis, and the gripper stopped at 5.72 s on
five of orange_3's cells (GT) 6-13 mm past that face. Its CONFIRMED neighbours trk:2 / trk:4
(orange_2 / orange_1) were fitted clear.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openral_core import IntrinsicsPinhole, PlaceRegion, Pose6D
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_core.perception_geometry import VoxelLattice
from openral_world_state.object_tracker import grow_hidden_side

_FIX = np.load(Path(__file__).parent / "fixtures" / "pick_orange_hidden_side.npz")
_RES = 0.015
_SUPPORT_Z = 0.015  # trk:3's measured counter top (tracked.jsonl)
_FRAME = "base"


def _box(name: str) -> PlaceRegion:
    v = [float(x) for x in _FIX[name]]
    return PlaceRegion(
        frame_id=_FRAME,
        half_extents=(v[7], v[8], v[9]),
        pose=Pose6D(xyz=(v[0], v[1], v[2]), quat_xyzw=(v[3], v[4], v[5], v[6]), frame_id=_FRAME),
    )


def _grid() -> VoxelLattice:
    pts: NDArray[np.float64] = np.asarray(_FIX["voxels"], dtype=np.float64)
    lo = pts.min(axis=0) - 1.5 * _RES  # the cloud's points are cell centres
    ijk = np.rint((pts - lo) / _RES - 0.5).astype(int)
    dims = ijk.max(axis=0) + 2
    occ = np.zeros(int(np.prod(dims)), dtype=np.uint8)
    occ[ijk[:, 0] + dims[0] * (ijk[:, 1] + dims[1] * ijk[:, 2])] = 1
    return VoxelLattice(
        _FRAME, tuple(lo.tolist()), (0.0, 0.0, 0.0, 1.0), _RES,
        (int(dims[0]), int(dims[1]), int(dims[2])), occ,
    )  # fmt: skip


def _k() -> IntrinsicsPinhole:
    fx, fy, cx, cy = (float(v) for v in _FIX["k"])
    h, w = _FIX["depth_m"].shape
    return IntrinsicsPinhole(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)


def _sibling_size() -> float:
    """The neighbours' clear size: max over them of their smaller horizontal extent."""
    return max(2.0 * min(_box(n).half_extents[:2]) for n in ("box_trk2", "box_trk4"))


def _grow(
    size: float, others: list[PlaceRegion] | None = None
) -> tuple[PlaceRegion, list[tuple[int, int, float]]]:
    return grow_hidden_side(
        _box("box_trk3"), size,
        depth_m=np.asarray(_FIX["depth_m"], dtype=np.float64), robot_mask=_FIX["robot_mask"],
        intrinsics=_k(), t_base_from_cam=_FIX["t_base_from_cam"], grid=_grid(),
        support_z=_SUPPORT_Z,
        others=[_box("box_trk2"), _box("box_trk4")] if others is None else others,
    )  # fmt: skip


def _inside(p: NDArray[np.float64], b: PlaceRegion) -> NDArray[np.bool_]:
    t = homogeneous_from_quat_xyz(b.pose.xyz, b.pose.quat_xyzw)
    return np.asarray(
        np.all(np.abs((p - t[:3, 3]) @ t[:3, :3]) <= np.asarray(b.half_extents), axis=1)
    )


def test_the_grown_box_covers_the_cells_the_gripper_stopped_on() -> None:
    stop = np.asarray(_FIX["stop_cells"], dtype=np.float64)
    assert not _inside(stop, _box("box_trk3")).any(), "the control: the fitted box misses them"
    out, grown = _grow(_sibling_size())
    assert grown and all(axis == 1 for axis, _, _ in grown), grown  # the short axis only
    assert _inside(stop, out).all()


def test_it_grows_only_on_the_side_the_robot_hides_and_never_past_the_size() -> None:
    size = _sibling_size()
    box = _box("box_trk3")
    out, grown = _grow(size)
    ((axis, sign, g),) = grown
    assert 2 * out.half_extents[axis] <= size + 1e-9 < 2 * box.half_extents[axis] + g + _RES
    # The far face (the side the camera sees clear) has not moved.
    t0 = homogeneous_from_quat_xyz(box.pose.xyz, box.pose.quat_xyzw)
    t1 = homogeneous_from_quat_xyz(out.pose.xyz, out.pose.quat_xyzw)
    far0 = t0[:3, 3] - t0[:3, axis] * sign * box.half_extents[axis]
    far1 = t1[:3, 3] - t1[:3, axis] * sign * out.half_extents[axis]
    assert np.allclose(far0, far1)
    # A much larger size still stops where the robot stops hiding it (never into seen space).
    big, _ = _grow(0.30)
    assert 2 * big.half_extents[axis] < 0.30


def test_it_never_grows_into_a_neighbours_box() -> None:
    box = _box("box_trk3")
    t = homogeneous_from_quat_xyz(box.pose.xyz, box.pose.quat_xyzw)
    # A neighbour 1 cm past the face the robot hides.
    c = t[:3, 3] + t[:3, 1] * (box.half_extents[1] + 0.01 + 0.02)
    wall = PlaceRegion(
        frame_id=_FRAME, half_extents=(0.05, 0.02, 0.05),
        pose=Pose6D(xyz=tuple(float(x) for x in c), quat_xyzw=box.pose.quat_xyzw, frame_id=_FRAME),
    )  # fmt: skip
    out, grown = _grow(_sibling_size(), others=[wall])
    assert all(g <= 0.01 + 1e-9 for _, _, g in grown), grown
    samples = np.array([wall.pose.xyz])
    assert not _inside(samples, out).any()


def test_no_size_no_growth() -> None:
    """Sized already (or no size known: the tracker passes none and never calls this)."""
    box = _box("box_trk3")
    out, grown = _grow(2 * min(box.half_extents[:2]))
    assert grown == [] and out == box


def test_the_tracker_sizes_from_its_own_clear_fit_then_same_label_tracks_else_nothing() -> None:
    from openral_core import FitKind, ObjectQuery, ObjectRole, TrackStatus
    from openral_world_state.object_tracker import ObjectTracker

    tr = ObjectTracker(_FRAME)
    tr.open_query("g", "q0", ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET),
                  "deploy@0", stamp_ns=0)  # fmt: skip
    assert tr._measured_size(None, "orange") is None, "no sibling, no clear fit: no growth"
    for name in ("box_trk2", "box_trk4"):
        t = tr.store.mint(
            label="orange", role=ObjectRole.TARGET, declaration_key="deploy@0", box=_box(name),
            tight=(), fit_kind=FitKind.SILHOUETTE_CONE, support_z=_SUPPORT_Z, support_ref="map",
            floor_z=_SUPPORT_Z + _RES, visible_cells=1, occupied_cells=1, last_fit_stamp_ns=1,
            last_confirmed_stamp_ns=1,
        )  # fmt: skip
        tr.store.replace(t.model_copy(update={"status": TrackStatus.CONFIRMED}))
    assert tr._measured_size(None, "orange") == (_sibling_size(), "same_label_tracks")
    assert tr._measured_size(None, "plate") is None
    tr._clear_size["trk:9"] = 0.061
    assert tr._measured_size("trk:9", "orange") == (0.061, "own_clear_fit")
