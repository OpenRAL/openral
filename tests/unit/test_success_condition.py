# SPDX-License-Identifier: Apache-2.0
"""``SuccessCondition`` / ``SuccessJudge`` / ``TaskOutcome`` (issue #635) on real fixtures.

The conditions are the ones the committed scenes declare: the cube over the layout-A
marker (``scenes/sim/makermods_metal_pick_place.yaml``) and three oranges in the plate
(``scenes/deploy/isaac_leisaac_pick_orange.yaml``). The judge reads real
``TrackedObjectSet`` models. Covers each verdict, the settle clock on tracker time, that
nothing perception cannot decide reads as a success, the IDL round trip and the reasoner's
``## EXECUTION`` phrase. (The real tracker node feeding the judge:
``tests/integration/test_object_tracker_node_live.py``.)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
from openral_core import (
    DeployScene,
    FitKind,
    ObjectRole,
    PlaceRegion,
    Pose6D,
    SimScene,
    SuccessCondition,
    TaskOutcome,
    TaskVerdict,
    TrackedObject,
    TrackedObjectSet,
    TrackStatus,
    success_condition_for_prompt,
)
from openral_core.success import SuccessJudge
from openral_reasoner.context import task_outcome_text
from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parents[2]
_MARKER = (0.278, 0.076, 0.0155)
_START = (0.267, -0.069, 0.0155)  # the layout-A cube start
_PLATE = (-0.2153, -0.2716, 0.0386)
_COUNTER = (0.0738, -0.3034, 0.0508)  # where orange_1 starts
_S = 1_000_000_000


@pytest.fixture(scope="module")
def cube() -> SuccessCondition:
    scene = SimScene.from_yaml(str(_ROOT / "scenes" / "sim" / "makermods_metal_pick_place.yaml"))
    condition = success_condition_for_prompt(scene.task.instruction, scene.success_conditions)
    assert condition is not None
    return condition


@pytest.fixture(scope="module")
def oranges() -> SuccessCondition:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # the scene's grasp search_box (#418)
        scene = DeployScene.from_yaml(
            str(_ROOT / "scenes" / "deploy" / "isaac_leisaac_pick_orange.yaml")
        )
    condition = success_condition_for_prompt(
        "Grab orange and place into plate", scene.success_conditions
    )
    assert condition is not None
    return condition


def track(
    n: int,
    xyz: tuple[float, float, float],
    status: TrackStatus = TrackStatus.CONFIRMED,
    *,
    label: str = "cube",
    frame: str = "base_link",
    half: tuple[float, float, float] = (0.0125, 0.0125, 0.0125),
) -> TrackedObject:
    """Track ``trk:<n>`` of ``label`` centred at ``xyz`` in ``frame``, box half-extents ``half``."""
    bound = status is TrackStatus.BOUND
    return TrackedObject(
        object_id=f"trk:{n}",
        epoch=0,
        label=label,
        role=ObjectRole.TARGET,
        declaration_key=f"obj:{label}@0",
        status=status,
        box=PlaceRegion(
            frame_id=frame,
            pose=Pose6D(xyz=xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=frame),
            half_extents=half,
        ),
        fit_kind=FitKind.SILHOUETTE_CONE,
        support_z=0.003,
        support_ref="map",
        floor_z=0.03,
        visible_cells=4,
        occupied_cells=4,
        last_fit_stamp_ns=1,
        last_confirmed_stamp_ns=1,
        bound_hand="gripper" if bound else None,
        attach_link="gripper" if bound else None,
        pose_in_link=(
            Pose6D(xyz=(0.0, 0.0, 0.1), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="gripper")
            if bound
            else None
        ),
    )


def tracked(t_s: float, *tracks: TrackedObject, frame: str = "base_link") -> TrackedObjectSet:
    """The tracker's set at ``t_s`` seconds of its own clock."""
    stamp = int(t_s * _S)
    return TrackedObjectSet(frame_id=frame, stamp_ns=stamp, revision=stamp, objects=list(tracks))


def judged(condition: SuccessCondition, *sets: TrackedObjectSet | None) -> TaskOutcome:
    judge = SuccessJudge(condition)
    for one in sets:
        judge.observe(one)
    return judge.outcome()


def test_the_scene_fixtures_declare_their_tasks_by_prompt(
    cube: SuccessCondition, oranges: SuccessCondition
) -> None:
    assert (cube.object_label, cube.region.frame_id, cube.region.pose.xyz) == (
        "cube",
        "base_link",
        _MARKER,
    )
    assert (cube.min_count, cube.settle_s) == (1, 1.0)
    # Three oranges, in the very box the scene declares as its `plate` place zone.
    assert (oranges.object_label, oranges.min_count) == ("orange", 3)
    assert oranges.region.pose.xyz == _PLATE
    # Any other prompt (a Nav2 move, another task) is judged by nothing.
    assert success_condition_for_prompt("go to the dock", {"Grab orange": oranges}) is None
    assert success_condition_for_prompt(" grab  ORANGE ", {"Grab orange": oranges}) is oranges


@pytest.mark.parametrize("status", [TrackStatus.CONFIRMED, TrackStatus.OCCLUDED])
def test_a_seen_cube_resting_in_the_region_for_settle_s_succeeds(
    cube: SuccessCondition, status: TrackStatus
) -> None:
    at = track(1, _MARKER, status)
    early = judged(cube, tracked(10.0, at), tracked(10.5, at))
    assert early.verdict is TaskVerdict.UNKNOWN and "0.5 s of the 1.0 s" in early.reason
    out = judged(cube, tracked(10.0, at), tracked(10.5, at), tracked(11.0, at))
    assert out.verdict is TaskVerdict.SUCCEEDED, out
    assert (out.source, out.object_ids, out.resting, out.required, out.held_s) == (
        "success_condition",
        ("trk:1",),
        1,
        1,
        1.0,
    )


def test_one_tracked_set_never_settles_however_long_it_is_looked_at(
    cube: SuccessCondition,
) -> None:
    """The settle clock is the tracker's evidence, not the judge's patience."""
    one = tracked(10.0, track(1, _MARKER))
    judge = SuccessJudge(cube)
    for _ in range(100):
        judge.observe(one)
    assert judge.outcome().verdict is TaskVerdict.UNKNOWN
    assert judge.outcome().held_s == 0.0


def test_the_trackers_heartbeat_is_evidence_and_a_clock_reset_restarts(
    cube: SuccessCondition,
) -> None:
    """The tracker republishes an unchanged set (same revision) at a later stamp, at 2 Hz."""

    def beat(t_s: float) -> TrackedObjectSet:
        return tracked(t_s, track(1, _MARKER)).model_copy(update={"revision": 7})

    held = judged(cube, beat(10.0), beat(10.5), beat(11.0))
    assert held.verdict is TaskVerdict.SUCCEEDED and held.held_s == 1.0
    # The source clock went backwards (a sim reset): what was held before does not count.
    reset = judged(cube, beat(100.0), beat(101.0), beat(1.0), beat(1.5))
    assert reset.verdict is TaskVerdict.UNKNOWN and reset.held_s == 0.5


def test_a_break_or_an_evidence_gap_restarts_the_settle(cube: SuccessCondition) -> None:
    at, away = track(1, _MARKER), track(1, _START)
    broken = judged(cube, tracked(10.0, at), tracked(10.6, away), tracked(11.2, at))
    assert broken.verdict is TaskVerdict.UNKNOWN and broken.held_s == 0.0
    # 2.5 s without a set (limit 2.0 s): the hold before it does not count.
    gap = judged(cube, tracked(10.0, at), tracked(12.5, at), tracked(13.0, at))
    assert gap.verdict is TaskVerdict.UNKNOWN and gap.held_s == 0.5
    # The tracker went stale (the snapshot carries no set): unknown, and the hold restarts.
    stale = judged(cube, tracked(10.0, at), tracked(11.0, at), None)
    assert stale.verdict is TaskVerdict.UNKNOWN and "no tracked-object set" in stale.reason
    assert judged(cube, tracked(10.0, at), None, tracked(11.0, at)).held_s == 0.0


@pytest.mark.parametrize(
    ("xyz", "status", "why"),
    [
        (_START, TrackStatus.CONFIRMED, "trk:1 is outside it"),
        ((_MARKER[0], _MARKER[1], 0.10), TrackStatus.CONFIRMED, "trk:1 is outside it"),  # lifted
        (_MARKER, TrackStatus.BOUND, "trk:1 is held"),
    ],
)
def test_a_cube_seen_elsewhere_or_still_held_fails(
    cube: SuccessCondition, xyz: tuple[float, float, float], status: TrackStatus, why: str
) -> None:
    out = judged(cube, tracked(10.0, track(1, xyz, status)), tracked(12.0, track(1, xyz, status)))
    assert out.verdict is TaskVerdict.FAILED and why in out.reason, out
    assert (out.object_ids, out.resting, out.held_s) == (("trk:1",), 0, 0.0)


@pytest.mark.parametrize(
    ("sets", "why"),
    [
        ((), "no tracked-object set"),
        ((tracked(10.0, track(1, _MARKER, frame="map"), frame="map"),) * 1, "'map'"),
        ((tracked(10.0, track(1, _MARKER, label="orange")),), "too few are tracked"),
        ((tracked(10.0, track(1, _MARKER, TrackStatus.LOST)),), "too few are tracked"),
        # Frozen at the let-go pose, or not yet confirmed: it may still come to rest there.
        ((tracked(10.0, track(1, _MARKER, TrackStatus.RELEASED)),), "1 not yet re-confirmed"),
        ((tracked(10.0, track(1, _MARKER, TrackStatus.TENTATIVE)),), "1 not yet re-confirmed"),
        # Another cube is seen elsewhere, but this one may still settle: not a failure yet.
        (
            (tracked(10.0, track(1, _MARKER, TrackStatus.RELEASED), track(2, _START)),),
            "1 not yet re-confirmed",
        ),
    ],
)
def test_what_perception_cannot_decide_is_unknown_never_a_success(
    cube: SuccessCondition, sets: tuple[TrackedObjectSet, ...], why: str
) -> None:
    out = judged(cube, *sets)
    assert out.verdict is TaskVerdict.UNKNOWN and why in out.reason, out


def test_three_oranges_are_needed_in_the_plate(oranges: SuccessCondition) -> None:
    a, b, c = (-0.26, -0.30, 0.05), (-0.17, -0.30, 0.05), (-0.21, -0.22, 0.05)  # in the plate

    def scene(t_s: float, *where: tuple[float, float, float]) -> TrackedObjectSet:
        tracks = [track(n, xyz, label="orange", frame="base") for n, xyz in enumerate(where, 1)]
        return tracked(t_s, *tracks, frame="base")

    one = judged(oranges, scene(1.0, a, _COUNTER, _COUNTER), scene(3.0, a, _COUNTER))
    assert one.verdict is TaskVerdict.FAILED and one.resting == 1 and one.required == 3
    assert "1 of 3 'orange' rest in the region; trk:2 is outside it" in one.reason
    # Two in the plate and the third not tracked at all: nothing says where it is.
    two = judged(oranges, scene(1.0, a, b), scene(3.0, a, b))
    assert two.verdict is TaskVerdict.UNKNOWN and "too few are tracked" in two.reason
    done = judged(oranges, scene(1.0, a, b, c), scene(2.0, a, b, c))
    assert done.verdict is TaskVerdict.SUCCEEDED and done.object_ids == ("trk:1", "trk:2", "trk:3")
    assert not oranges.end_on_success  # this scene's verdict labels the result only


def test_two_tracks_of_one_object_count_once(oranges: SuccessCondition) -> None:
    """The tracks the real tracker held with ONE orange in the plate (q5090, 2026-11-19).

    Isaac PickOrange, simulator grade ``partial``: ``trk:6`` and a later ``trk:10`` minted
    3 cm apart, one inside the other's box, plus ``trk:8`` 9 cm away. Counting tracks
    reported three oranges at rest; two of them are one object.
    """

    def scene(t_s: float) -> TrackedObjectSet:
        def held(n: int, xyz: tuple[float, float, float], half: tuple[float, float, float]) -> Any:
            return track(n, xyz, label="orange", frame="base", half=half)

        return tracked(
            t_s,
            held(6, (-0.222, -0.244, 0.054), (0.020, 0.033, 0.024)),
            held(8, (-0.137, -0.273, 0.052), (0.020, 0.039, 0.022)),
            held(10, (-0.229, -0.276, 0.063), (0.027, 0.034, 0.033)),
            frame="base",
        )

    out = judged(oranges, scene(1.0), scene(2.5))
    assert out.verdict is not TaskVerdict.SUCCEEDED, out
    assert (out.resting, out.object_ids) == (2, ("trk:6", "trk:8"))


def test_a_condition_that_names_no_volume_is_refused(cube: SuccessCondition) -> None:
    from openral_core import AttachedCollisionPrimitive, BoxShape

    box = AttachedCollisionPrimitive(
        shape=BoxShape(half_extents_m=(0.01, 0.01, 0.01)), pose_in_object=cube.region.pose
    )
    with pytest.raises(ValidationError, match="geometry must be empty"):
        SuccessCondition(
            object_label="cube", region=cube.region.model_copy(update={"geometry": (box,)})
        )
    for update, why in (
        ({"quat_xyzw": (0.0, 0.0, 0.0, 0.0)}, "not a rotation"),
        ({"xyz": (float("nan"), 0.076, 0.0155)}, "must be finite"),
        ({"frame_id": "map"}, "not the region's 'base_link'"),
    ):
        pose = cube.region.pose.model_copy(update=update)
        with pytest.raises(ValidationError, match=why):
            SuccessCondition(
                object_label="cube", region=cube.region.model_copy(update={"pose": pose})
            )
    with pytest.raises(ValidationError, match="greater than 0"):
        SuccessCondition(object_label="cube", region=cube.region, settle_s=0.0)


def test_a_scene_refuses_a_condition_for_a_blank_prompt(cube: SuccessCondition) -> None:
    """A blank key would name every goal sent without a prompt (a wrapped move)."""
    path = _ROOT / "scenes" / "deploy" / "isaac_leisaac_pick_orange.yaml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        scene = DeployScene.from_yaml(str(path))
        with pytest.raises(ValidationError, match="a blank one is refused"):
            DeployScene.model_validate(
                {**scene.model_dump(exclude_unset=True), "success_conditions": {" ": cube}}
            )


def test_the_condition_round_trips_the_ros_message(oranges: SuccessCondition) -> None:
    msgs = pytest.importorskip("openral_msgs.msg")
    msg = msgs.SuccessCondition()
    oranges.fill_idl(msg)
    assert SuccessCondition.from_idl(msg) == oranges
    # Unset wire fields read as the schema defaults, not as zero.
    bare = msgs.SuccessCondition()
    bare.object_label = "orange"
    oranges.region.fill_idl(bare.region)
    decoded = SuccessCondition.from_idl(bare)
    assert (decoded.min_count, decoded.settle_s, decoded.max_evidence_gap_s) == (1, 1.0, 2.0)


def test_the_reasoner_phrase_names_verdict_source_and_reason(cube: SuccessCondition) -> None:
    away = track(1, _START)
    failed = judged(cube, tracked(10.0, away))
    assert task_outcome_text(failed.model_dump_json()) == (
        "task_outcome=failed (success_condition): 0 of 1 'cube' rest in the region; "
        "trk:1 is outside it"
    )
    assert task_outcome_text(judged(cube).model_dump_json()).startswith(
        "task_outcome=unknown (success_condition): no tracked-object set"
    )
    # Nothing judged the run: nothing to add to the line, and never "succeeded".
    undeclared = TaskOutcome(verdict=TaskVerdict.UNKNOWN, reason="no success condition")
    assert task_outcome_text(undeclared.model_dump_json()) == ""
    assert task_outcome_text("") == ""
    assert task_outcome_text("not json") == "task_outcome=unknown (unreadable task_outcome_json)"
