"""Graded placement success for the Isaac manifest scene (``IsaacSimOptions.placement_success``).

Host side: the options model validates against a real scene fixture
(``scenes/deploy/isaac_openarm_warehouse.yaml``) and resolves the scene's prompt
into the sidecar's grading spec. Sidecar side: the scene module's pure grading
helpers (imported without Isaac, like ``test_isaac_manifest_scene``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.isaac_sim import (
    DEFAULT_PLACEMENT_GRADES,
    IsaacPlacement,
    IsaacPlacementGrade,
    IsaacPlacementRegion,
    IsaacPlacementSuccess,
    IsaacSimOptions,
    _placement_success_spec,
)
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[2]
_WAREHOUSE = _REPO / "scenes/deploy/isaac_openarm_warehouse.yaml"
_PROMPT = "put the cracker box on the shelf"


def _warehouse_options(placement_success: dict[str, Any]) -> dict[str, Any]:
    opts = dict(yaml.safe_load(_WAREHOUSE.read_text())["scene"]["backend_options"])
    opts["placement_success"] = placement_success
    return opts


def _shelf_success(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "regions": {
            "shelf_1": {"min_xyz": [-0.7, 4.75, 0.2], "max_xyz": [-0.4, 4.95, 0.5]},
            "shelf_2": {"min_xyz": [-0.7, 4.95, 0.2], "max_xyz": [-0.4, 5.25, 0.5]},
        },
        "tasks": {
            _PROMPT: [{"object": "cracker_box", "target": "shelf_1", "adjacent": ["shelf_2"]}]
        },
    }
    spec.update(overrides)
    return spec


def test_warehouse_scene_validates_with_a_placement_success_block() -> None:
    opts = IsaacSimOptions.model_validate(_warehouse_options(_shelf_success()))
    assert opts.placement_success is not None
    assert opts.placement_success.tasks[_PROMPT][0].adjacent == ["shelf_2"]


def test_placement_of_an_object_the_scene_does_not_have_is_refused() -> None:
    bad = _shelf_success(tasks={_PROMPT: [{"object": "tomato_can", "target": "shelf_1"}]})
    with pytest.raises(ValidationError, match="tomato_can"):
        IsaacSimOptions.model_validate(_warehouse_options(bad))


def test_placement_naming_an_unknown_region_is_refused() -> None:
    bad = _shelf_success(tasks={_PROMPT: [{"object": "cracker_box", "target": "shelf_9"}]})
    with pytest.raises(ValidationError, match="shelf_9"):
        IsaacSimOptions.model_validate(_warehouse_options(bad))


def test_placing_the_same_object_twice_in_one_task_is_refused() -> None:
    twice = [
        {"object": "cracker_box", "target": "shelf_1"},
        {"object": "cracker_box", "target": "shelf_2"},
    ]
    with pytest.raises(ValidationError, match="twice"):
        IsaacSimOptions.model_validate(_warehouse_options(_shelf_success(tasks={_PROMPT: twice})))


def test_a_static_object_cannot_be_graded() -> None:
    opts = _warehouse_options(_shelf_success())
    opts["objects"] = [
        {**o, "dynamic": False} if o["name"] == "cracker_box" else o for o in opts["objects"]
    ]
    with pytest.raises(ValidationError, match="static"):
        IsaacSimOptions.model_validate(opts)


def test_rubric_names_must_be_unique_and_the_first_grade_scores_full_credit() -> None:
    level = {"name": "in", "score": 1.0, "region": "target"}
    with pytest.raises(ValidationError, match="unique names"):
        IsaacSimOptions.model_validate(_warehouse_options(_shelf_success(grades=[level, level])))
    with pytest.raises(ValidationError, match="unique names"):
        IsaacSimOptions.model_validate(
            _warehouse_options(_shelf_success(grades=[{**level, "name": "fail"}]))
        )
    with pytest.raises(ValidationError, match=r"must score 1\.0"):
        IsaacSimOptions.model_validate(
            _warehouse_options(_shelf_success(grades=[{**level, "score": 0.5}]))
        )


def test_inverted_region_is_refused() -> None:
    bad = _shelf_success()
    bad["regions"]["shelf_1"] = {"min_xyz": [0.0, 0.0, 1.0], "max_xyz": [1.0, 1.0, 0.5]}
    with pytest.raises(ValidationError, match="min < max"):
        IsaacSimOptions.model_validate(_warehouse_options(bad))


def test_spec_resolves_the_scene_prompt_into_boxes() -> None:
    opts = IsaacSimOptions.model_validate(_warehouse_options(_shelf_success()))
    spec = _placement_success_spec(opts.placement_success, _PROMPT)
    assert spec is not None
    (placement,) = spec["placements"]
    assert placement["object"] == "cracker_box"
    assert placement["target"] == {"min": [-0.7, 4.75, 0.2], "max": [-0.4, 4.95, 0.5]}
    assert placement["adjacent"] == [{"min": [-0.7, 4.95, 0.2], "max": [-0.4, 5.25, 0.5]}]
    assert (spec["settle_s"], spec["upright_tol_deg"], spec["rest_turn_deg_s"]) == (1.0, 20.0, 10.0)
    assert [g["name"] for g in spec["grades"]] == ["complete", "partial", "semi"]
    assert spec["grades"][0] == {
        "name": "complete",
        "score": 1.0,
        "region": "target",
        "requires_upright": True,
    }


def test_spec_refuses_a_prompt_it_does_not_grade() -> None:
    opts = IsaacSimOptions.model_validate(_warehouse_options(_shelf_success()))
    with pytest.raises(ROSConfigError, match="put the cracker box on the shelf"):
        _placement_success_spec(opts.placement_success, "put the sugar box on the shelf")


@pytest.fixture(scope="module")
def scene_mod() -> Any:
    tools = str(_REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import isaac_manifest_scene

    return isaac_manifest_scene


_SLOT = {
    "target": {"min": [0.0, 0.2, 0.5], "max": [0.5, 0.4, 0.62]},
    "adjacent": [{"min": [0.0, 0.0, 0.5], "max": [0.5, 0.2, 0.62]}],
}
_LEVEL = np.array([1.0, 0.0, 0.0, 0.0])
_UP = np.array([0.0, 0.0, 1.0])
_RUBRIC = [g.model_dump() for g in DEFAULT_PLACEMENT_GRADES]
_SCORES = {"complete": 1.0, "partial": 0.5, "semi": 0.25, "fail": 0.0}


@pytest.mark.parametrize(
    ("where", "fallen", "grade"),
    [
        ((0.2, 0.3, 0.55), False, "complete"),
        ((0.2, 0.3, 0.53), True, "partial"),
        ((0.2, 0.1, 0.55), False, "semi"),
        ((0.2, 0.1, 0.53), True, "semi"),  # next slot, fallen: still the neighbour
        ((0.2, -0.1, 0.55), False, "fail"),  # two slots over
        ((0.2, 0.3, 0.25), False, "fail"),  # the right slot's column, wrong shelf
    ],
)
def test_placement_grade(
    scene_mod: Any, where: tuple[float, float, float], fallen: bool, grade: str
) -> None:
    quat = scene_mod._rpy_quat(np.pi / 2, 0.0, 0.3) if fallen else scene_mod._yaw_quat(0.3)
    assert scene_mod.placement_grade(np.array(where), quat, _UP, _SLOT, 20.0, _RUBRIC) == grade


def test_the_scene_rubric_is_evaluated_in_order(scene_mod: Any) -> None:
    anywhere_in_target = [
        {"name": "in", "score": 1.0, "region": "target", "requires_upright": False}
    ]
    fallen = scene_mod._rpy_quat(np.pi / 2, 0.0, 0.3)
    assert (
        scene_mod.placement_grade(
            np.array([0.2, 0.3, 0.53]), fallen, _UP, _SLOT, 20.0, anywhere_in_target
        )
        == "in"
    )
    assert (
        scene_mod.placement_grade(
            np.array([0.2, 0.1, 0.55]), _LEVEL, _UP, _SLOT, 20.0, anywhere_in_target
        )
        == "fail"
    )
    assert scene_mod.grade_scores(anywhere_in_target) == {"in": 1.0, "fail": 0.0}


def test_upright_follows_the_declared_up_axis(scene_mod: Any) -> None:
    # An asset authored lying down, stood up by a quarter roll: standing is upright.
    up = scene_mod.object_up_axis(np.pi / 2, 0.0)
    standing = scene_mod._rpy_quat(np.pi / 2, 0.0, 1.0)
    where = np.array([0.2, 0.3, 0.55])
    assert scene_mod.placement_grade(where, standing, up, _SLOT, 20.0, _RUBRIC) == "complete"
    assert scene_mod.placement_grade(where, _LEVEL, up, _SLOT, 20.0, _RUBRIC) == "partial"


def test_tilt_tolerance(scene_mod: Any) -> None:
    where = np.array([0.2, 0.3, 0.55])
    tilted = scene_mod._rpy_quat(np.radians(15.0), 0.0, 0.0)
    assert scene_mod.placement_grade(where, tilted, _UP, _SLOT, 20.0, _RUBRIC) == "complete"
    assert scene_mod.placement_grade(where, tilted, _UP, _SLOT, 10.0, _RUBRIC) == "partial"


def test_episode_grade_is_the_worst_placement(scene_mod: Any) -> None:
    assert scene_mod.worst_grade(["complete", "complete"], _SCORES) == "complete"
    assert scene_mod.worst_grade(["complete", "partial", "semi"], _SCORES) == "semi"
    assert scene_mod.worst_grade(["complete", "fail"], _SCORES) == "fail"


def test_rest_timer_needs_a_still_object(scene_mod: Any) -> None:
    dt, speed = 1 / 30, 0.02
    rest = 0.0
    rest = scene_mod.rest_after(rest, None, dt, speed)  # first graded step: no history
    for _ in range(30):
        rest = scene_mod.rest_after(rest, 0.0002, dt, speed)
    assert rest == pytest.approx(1.0)
    assert scene_mod.rest_after(rest, 0.01, dt, speed) == 0.0  # 0.3 m/s: moving again
    turn = np.radians(10.0)
    assert scene_mod.rest_after(rest, 0.0, dt, speed, np.radians(0.1), turn) > rest  # 3 deg/s
    assert (
        scene_mod.rest_after(rest, 0.0, dt, speed, np.radians(1.0), turn) == 0.0
    )  # 30 deg/s: spinning
    assert scene_mod._quat_angle(
        scene_mod._yaw_quat(0.0), scene_mod._yaw_quat(0.2)
    ) == pytest.approx(0.2)


def test_reward_ends_the_episode_only_on_complete(scene_mod: Any) -> None:
    reward = scene_mod.placement_reward
    assert reward("complete", False, _SCORES, "complete") == (1.0, True)
    assert reward("partial", False, _SCORES, "complete") == (0.0, False)
    assert reward("partial", True, _SCORES, "complete") == (0.5, False)
    assert reward("semi", True, _SCORES, "complete") == (0.25, False)
    assert reward("fail", True, _SCORES, "complete") == (0.0, False)
    assert reward("in", False, {"in": 1.0, "fail": 0.0}, "in") == (1.0, True)  # a one-level rubric


# ── Schema fuzz: round-trip + JSON Schema for every placement model ──────────────────

_coord = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_span = st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False)
_name = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_0123456789", min_size=1, max_size=12)


@st.composite
def _regions(draw: st.DrawFn) -> IsaacPlacementRegion:
    lo = draw(st.tuples(_coord, _coord, _coord))
    span = draw(st.tuples(_span, _span, _span))
    hi = tuple(a + b for a, b in zip(lo, span, strict=True))
    return IsaacPlacementRegion(min_xyz=lo, max_xyz=hi)


@st.composite
def _placements(draw: st.DrawFn, region_names: list[str]) -> IsaacPlacement:
    return IsaacPlacement(
        object=draw(_name),
        target=draw(st.sampled_from(region_names)),
        adjacent=draw(st.lists(st.sampled_from(region_names), max_size=3)),
    )


@st.composite
def _rubrics(draw: st.DrawFn) -> list[IsaacPlacementGrade]:
    names = draw(st.lists(_name.filter(lambda n: n != "fail"), min_size=1, max_size=4, unique=True))
    levels = [
        IsaacPlacementGrade(
            name=n,
            score=1.0 if i == 0 else draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            region=draw(st.sampled_from(["target", "adjacent"])),
            requires_upright=draw(st.booleans()),
        )
        for i, n in enumerate(names)
    ]
    return levels


@st.composite
def _successes(draw: st.DrawFn) -> IsaacPlacementSuccess:
    names = draw(st.lists(_name, min_size=1, max_size=4, unique=True))
    prompts = draw(st.lists(st.text(min_size=1, max_size=24), min_size=1, max_size=3, unique=True))
    return IsaacPlacementSuccess(
        regions={n: draw(_regions()) for n in names},
        tasks={
            p: draw(
                st.lists(_placements(names), min_size=1, max_size=2, unique_by=lambda x: x.object)
            )
            for p in prompts
        },
        grades=draw(_rubrics()),
        upright_tol_deg=draw(st.floats(min_value=0.1, max_value=89.9, allow_nan=False)),
        settle_s=draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False)),
        rest_speed_m_s=draw(st.floats(min_value=1e-4, max_value=1.0, allow_nan=False)),
    )


@given(region=_regions())
def test_placement_region_round_trips(region: IsaacPlacementRegion) -> None:
    assert IsaacPlacementRegion.model_validate(region.model_dump()) == region
    assert IsaacPlacementRegion.model_validate_json(region.model_dump_json()) == region


@given(success=_successes())
def test_placement_success_round_trips(success: IsaacPlacementSuccess) -> None:
    assert IsaacPlacementSuccess.model_validate(success.model_dump()) == success
    assert IsaacPlacementSuccess.model_validate_json(success.model_dump_json()) == success
    for placements in success.tasks.values():
        for placement in placements:
            assert IsaacPlacement.model_validate_json(placement.model_dump_json()) == placement


@pytest.mark.parametrize(
    "model", [IsaacPlacementRegion, IsaacPlacement, IsaacPlacementGrade, IsaacPlacementSuccess]
)
def test_placement_models_export_a_json_schema(model: type[Any]) -> None:
    schema = model.model_json_schema()
    assert schema["title"] == model.__name__
    assert set(model.model_fields) <= set(schema["properties"])
