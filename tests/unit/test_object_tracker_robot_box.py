"""A target the robot half hides grounds PROVISIONAL at goal start (#418 PickOrange orange_3).

Real frame: ``fixtures/pick_orange_t0_top_depth.npz``, the PickOrange top_depth camera at goal
start in trial r6po_dbg2 (Spark, 2026-10-17, after the see-through robot-mask fix): depth,
robot mask, intrinsics and camera pose, cropped to the region around the open-vocabulary
locator's box. orange_3 lies under the SO-101's open jaw; 249 of its pixels are visible
(120 free, 129 in the uncertain band). The locator's only box on it, ``locator_box_arm``,
is mostly the arm, so SAM's masks for it were the arm and the fit was refused.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import IntrinsicsPinhole, ObjectQuery, ObjectRole
from openral_core.perception_geometry import VoxelLattice
from openral_world_state.object_tracker import CameraFrame, ObjectTracker
from openral_world_state.object_tracker.scheduler import TrackerRequest

_FIX = np.load(Path(__file__).parent / "fixtures" / "pick_orange_t0_top_depth.npz")
_CAM = "top_depth"
_RES = 0.015  # the sim octomap cell


def _frame() -> CameraFrame:
    depth = _FIX["depth_m"]
    fx, fy, cx, cy = (float(v) for v in _FIX["k"])
    k = IntrinsicsPinhole(width=depth.shape[1], height=depth.shape[0], fx=fx, fy=fy, cx=cx, cy=cy)
    return CameraFrame(
        _CAM, int(_FIX["stamp_ns"]), depth, k, _FIX["t_base_from_cam"], _FIX["robot_mask"]
    )


def _grid() -> VoxelLattice:
    pts: NDArray[np.float64] = np.asarray(_FIX["voxels"], dtype=np.float64)
    lo = np.floor(pts.min(axis=0) / _RES) * _RES - _RES
    ijk = np.floor((pts - lo) / _RES).astype(int)
    dims = ijk.max(axis=0) + 2
    occ = np.zeros(int(np.prod(dims)), dtype=np.uint8)
    occ[ijk[:, 0] + dims[0] * (ijk[:, 1] + dims[1] * ijk[:, 2])] = 1
    nx, ny, nz = (int(d) for d in dims)
    return VoxelLattice("base", tuple(lo.tolist()), (0.0, 0.0, 0.0, 1.0), _RES, (nx, ny, nz), occ)


def _tracker() -> ObjectTracker:
    tr = ObjectTracker("base")
    tr.open_query("g", "q0", ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET),
                  "obj:orange@0", stamp_ns=0)  # fmt: skip
    return tr


def _segment(box: NDArray[np.float64]) -> TrackerRequest:
    return TrackerRequest(
        "segment_in_view", _CAM, int(_FIX["stamp_ns"]), 1, -1, "orange", ("g", "q0"),
        box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
    )  # fmt: skip


def test_a_mostly_robot_detector_box_is_reprompted_on_what_it_hides() -> None:
    tr, frame, grid = _tracker(), _frame(), _grid()
    req = tr._robot_box_reprompt(_segment(_FIX["locator_box_arm"]), frame, grid)
    assert req.box_xyxy is None and req.point_uv is not None
    assert req.within_xyxy == tuple(float(v) for v in _FIX["locator_box_arm"])
    u, v = (round(c) for c in req.point_uv)
    assert _FIX["orange3_visible"][v, u], "the positive point is on orange_3"
    assert _FIX["robot_mask"][v, u] == 0, "on a free pixel, never the band or the robot"
    assert req.negative_uv, "negative points on the robot"
    for nu, nv in req.negative_uv:
        assert _FIX["robot_mask"][round(nv), round(nu)] == 255
    # A box on an orange the robot does not cover is left as the locator gave it.
    plain = _segment(_FIX["locator_box_orange1"])
    assert tr._robot_box_reprompt(plain, frame, grid) == plain


def test_the_visible_part_fits_provisional_through_the_robot() -> None:
    """SAM's mask on that prompt is the visible orange; the fit goes through the robot."""
    tr, frame, grid = _tracker(), _frame(), _grid()
    req = tr._robot_box_reprompt(_segment(_FIX["locator_box_arm"]), frame, grid)
    mask = np.asarray(_FIX["orange3_visible"], dtype=bool)
    tr._apply_mask(req, mask, frame, grid, int(_FIX["stamp_ns"]), (), last=True)
    (track,) = tr.store.all()
    assert track.label == "orange" and track.provisional
    centre = np.asarray(track.box.pose.xyz)
    assert np.linalg.norm(centre[:2] - _FIX["orange3_center"][:2]) < 0.04


def test_a_reprompted_box_is_served_once() -> None:
    """The re-prompt carries the box only as its bound; serving it still clears it."""
    tr, frame, grid = _tracker(), _frame(), _grid()
    req = tr._robot_box_reprompt(_segment(_FIX["locator_box_arm"]), frame, grid)
    tr._pending = [req]
    tr._mark_served(replace(req, request_id=7))
    assert tr._pending == []


def test_a_resolved_count_all_query_says_it_keeps_searching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openral_core import ObjectQueryStatus
    from openral_world_state.object_tracker import tracker as tracker_mod

    events: list[str] = []

    class _Rec:
        def __getattr__(self, level: str) -> Callable[..., None]:
            return lambda event, **_kw: events.append(f"{level}:{event}")

    monkeypatch.setattr(tracker_mod, "log", _Rec())
    tr = _tracker()
    q = tr.queries.get("g", "q0")
    assert q is not None
    q.status = ObjectQueryStatus.RESOLVED
    locate = TrackerRequest("locate_in_view", _CAM, 0, 2, 0, "orange", ("g", "q0"))
    for _ in range(3):  # rate-limited: once per 5 source seconds
        tr._mark_served(locate)
    assert events.count("info:object_tracker.count_all_research") == 1
