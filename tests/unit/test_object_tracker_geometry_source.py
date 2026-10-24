"""Where a payload primitive's geometry came from is on the wire (#418, hazard OT-10).

The kernel's payload-own-cell exemption (``payload_own_cells_enabled``) reads
``AttachedCollisionPrimitive.geometry_source``: only ``measured`` and ``grown`` primitives can
explain the payload's own cells; ``fallback`` (the pinch volume, boxes of candidates that may
not be the held object) never does. Real manifests, the real binder.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from openral_core import AttachedCollisionPrimitive, BoxShape, ObjectRole, Pose6D, TrackStatus
from openral_core.geometry import rotation_to_quat_wxyz
from openral_world_state.object_tracker.binding import BindOutcome

from tests.unit._object_tracker_fixtures import CELL, ROBOTS, DictStore, arm_at_hold, track
from tests.unit.test_object_tracker_binding import (
    _body,
    _grid,
    _hand_step,
    _tracked_hold,
)


def _sources(binding: object) -> list[str]:
    return [p.geometry_source for p in binding.attachment.primitives]  # type: ignore[attr-defined]


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_tracked_bind_marks_every_primitive_measured(robot: str) -> None:
    binder, q, centre = arm_at_hold(robot)
    store = DictStore([track("trk:1", binder.frame_id, centre, (0.02, 0.02, 0.02))])
    b = binder.on_attach(store, q, stamp_ns=10)
    assert b.outcome is BindOutcome.TRACKED
    assert set(_sources(b)) == {"measured"}


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_an_untracked_grasp_is_only_fallback(robot: str) -> None:
    binder, q, _ = arm_at_hold(robot)
    b = binder.on_attach(DictStore([]), q, stamp_ns=10)
    assert b.outcome is BindOutcome.UNTRACKED and set(_sources(b)) == {"fallback"}


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_one_provisional_candidate_is_measured_beside_the_fallback_volume(robot: str) -> None:
    binder, q, centre = arm_at_hold(robot)
    prov = track("trk:1", binder.frame_id, centre, (0.02, 0.02, 0.02)).model_copy(
        update={"provisional": True}
    )
    b = binder.on_attach(DictStore([prov]), q, stamp_ns=10)
    assert b.outcome is BindOutcome.PROVISIONAL
    src = _sources(b)
    assert src[0] == "fallback" and "measured" in src[1:]  # the pinch volume first, never evidence


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_several_provisional_candidates_may_be_neighbours_so_none_is_measured(robot: str) -> None:
    binder, q, centre = arm_at_hold(robot)
    a = track("trk:1", binder.frame_id, centre, (0.02, 0.02, 0.02)).model_copy(
        update={"provisional": True}
    )
    b2 = track("trk:2", binder.frame_id, centre + 0.003, (0.02, 0.02, 0.02)).model_copy(
        update={"provisional": True}
    )
    b = binder.on_attach(DictStore([a, b2]), q, stamp_ns=10)
    assert b.outcome is BindOutcome.PROVISIONAL and set(_sources(b)) == {"fallback"}


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_an_ambiguous_grasp_names_no_evidence(robot: str) -> None:
    binder, q, centre = arm_at_hold(robot)
    one = track("trk:1", binder.frame_id, centre, (0.02, 0.02, 0.02))
    two = track("trk:2", binder.frame_id, centre + 0.004, (0.02, 0.02, 0.02))
    b = binder.on_attach(DictStore([one, two]), q, stamp_ns=10)
    assert b.outcome is BindOutcome.AMBIGUOUS and set(_sources(b)) == {"fallback"}


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_released_track_is_no_bind_candidate(robot: str) -> None:
    binder, q, centre = arm_at_hold(robot)
    rel = track("trk:1", binder.frame_id, centre, (0.02, 0.02, 0.02), status=TrackStatus.RELEASED)
    b = binder.on_attach(DictStore([rel]), q, stamp_ns=10)
    assert b.outcome is BindOutcome.UNTRACKED and set(_sources(b)) == {"fallback"}


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_destination_is_never_the_payload(robot: str) -> None:
    """Spark R193 / R205: the jaw closed inside the bowl's box and bound the bowl."""
    binder, q, centre = arm_at_hold(robot)
    bowl = track("trk:1", binder.frame_id, centre, (0.11, 0.14, 0.08), role=ObjectRole.DESTINATION)
    b = binder.on_attach(DictStore([bowl]), q, stamp_ns=1)
    assert b.outcome is BindOutcome.UNTRACKED and set(_sources(b)) == {"fallback"}
    # The same box as the target binds it.
    target = bowl.model_copy(update={"role": ObjectRole.TARGET})
    binder.on_release()
    assert binder.on_attach(DictStore([target]), q, stamp_ns=1).outcome is BindOutcome.TRACKED


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_in_hand_growth_cells_are_grown_and_the_fit_stays_measured(robot: str) -> None:
    binder, q, r = _tracked_hold(robot)
    frame = binder.frame_id
    t_a = binder.hand.t_frame_from_attach(frame, q)
    pc = t_a[:3, :3] @ binder.hand.pinch_in_link(q[binder.hand.joint.name]).centre + t_a[:3, 3]
    body_a = _body(pc, r, -0.02, 0.035)
    q_b = _hand_step(binder, robot, q, 1.0 * CELL)
    t_b = binder.hand.t_frame_from_attach(frame, q_b)
    move = t_b @ np.linalg.inv(t_a)
    body_b = body_a @ move[:3, :3].T + move[:3, 3]
    lo = np.floor((pc - 0.3) / CELL) * CELL
    store = DictStore([])
    first = binder.held.attachment.primitives  # type: ignore[union-attr]
    n0 = len(first)
    assert binder.grow_untracked(store, _grid(frame, body_a, lo, 60), q, 20) is None
    grown = binder.grow_untracked(store, _grid(frame, body_b, lo, 60), q_b, 30)
    assert grown is not None
    src = [p.geometry_source for p in grown.attachment.primitives]
    assert src[:n0] == ["measured"] * n0 and set(src[n0:]) == {"grown"}


def test_a_protrusion_arc_is_grown_and_the_first_fit_stays_measured() -> None:
    binder, q, centre = arm_at_hold("droid_franka_robotiq")
    frame = binder.frame_id
    t_fa = binder.hand.t_frame_from_attach(frame, q)
    r_pinch = t_fa[:3, :3] @ binder.hand.axes
    w, x, y, z = rotation_to_quat_wxyz(r_pinch)
    half = (0.012, 0.012, 0.07)
    store = DictStore([track("trk:1", frame, centre, half, quat=(x, y, z, w))])
    first = binder.on_attach(store, q, stamp_ns=10).attachment
    theta = math.radians(30.0)
    c = r_pinch[:, 0]
    k = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    rot = np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)
    g = np.stack(np.meshgrid(*[np.arange(-h, h + 1e-9, 0.002) for h in half]), -1).reshape(-1, 3)
    grid = _grid(frame, g @ (rot @ r_pinch).T + centre, np.floor((centre - 0.2) / CELL) * CELL)
    grown = binder.grow_by_protrusion(store, grid, q, stamp_ns=20)
    assert grown is not None
    src = [p.geometry_source for p in grown.attachment.primitives]
    n0 = len(first.primitives)
    assert src[:n0] == ["measured"] * n0 and set(src[n0:]) == {"grown"}


def test_the_tag_round_trips_the_wire_and_an_unknown_code_is_fallback() -> None:
    def msg(code: int) -> SimpleNamespace:
        pos = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        ori = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        return SimpleNamespace(
            SHAPE_SPHERE=1, SHAPE_CAPSULE=2, SHAPE_BOX=3, shape_type=3,
            shape_dimensions=[0.02, 0.02, 0.02],
            pose_in_object=SimpleNamespace(position=pos, orientation=ori),
            tight_dop_lo=[], tight_dop_hi=[], tight_hull_vertices=[], geometry_source=code,
        )  # fmt: skip

    for code, name in ((0, "fallback"), (1, "measured"), (2, "grown"), (7, "fallback")):
        assert AttachedCollisionPrimitive.from_idl(msg(code), object_id="o").geometry_source == name
    prim = AttachedCollisionPrimitive(
        shape=BoxShape(half_extents_m=(0.02, 0.02, 0.02)),
        pose_in_object=Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id="o"),
        geometry_source="grown",
    )
    out = msg(0)
    prim.fill_idl(out)
    assert out.geometry_source == 2
    # A publisher that never sets it is fallback (the model default and IDL zero).
    assert (
        AttachedCollisionPrimitive(
            shape=prim.shape, pose_in_object=prim.pose_in_object
        ).geometry_source
        == "fallback"
    )
