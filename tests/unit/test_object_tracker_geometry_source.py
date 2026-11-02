"""Where a payload primitive's geometry came from is on the wire (#418, hazard OT-10).

The kernel's payload-own-cell exemption (``payload_own_cells_enabled``) reads
``AttachedCollisionPrimitive.geometry_source``: only ``measured`` and ``grown`` primitives can
explain the payload's own cells; ``fallback`` (the pinch volume, boxes of candidates that may
not be the held object) never does. Real manifests, the real binder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openral_core import AttachedCollisionPrimitive, BoxShape, ObjectRole, Pose6D, TrackStatus
from openral_world_state.object_tracker.binding import BindOutcome

from tests.unit._object_tracker_fixtures import ROBOTS, DictStore, arm_at_hold, track


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
