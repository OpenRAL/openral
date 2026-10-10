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
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.isaac_sim import IsaacSimOptions, _placement_success_spec
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
    assert (spec["settle_s"], spec["upright_tol_deg"]) == (1.0, 20.0)


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


@pytest.mark.parametrize(
    ("where", "fallen", "grade"),
    [
        ((0.2, 0.3, 0.55), False, "complete"),
        ((0.2, 0.3, 0.53), True, "partial"),
        ((0.2, 0.1, 0.55), False, "semi"),
        ((0.2, 0.1, 0.53), True, "fail"),  # next slot, fallen: no credit
        ((0.2, -0.1, 0.55), False, "fail"),  # two slots over
        ((0.2, 0.3, 0.25), False, "fail"),  # the right slot's column, wrong shelf
    ],
)
def test_placement_grade(
    scene_mod: Any, where: tuple[float, float, float], fallen: bool, grade: str
) -> None:
    quat = scene_mod._rpy_quat(np.pi / 2, 0.0, 0.3) if fallen else scene_mod._yaw_quat(0.3)
    assert scene_mod.placement_grade(np.array(where), quat, _UP, _SLOT, 20.0) == grade


def test_upright_follows_the_declared_up_axis(scene_mod: Any) -> None:
    # An asset authored lying down, stood up by a quarter roll: standing is upright.
    up = scene_mod.object_up_axis(np.pi / 2, 0.0)
    standing = scene_mod._rpy_quat(np.pi / 2, 0.0, 1.0)
    where = np.array([0.2, 0.3, 0.55])
    assert scene_mod.placement_grade(where, standing, up, _SLOT, 20.0) == "complete"
    assert scene_mod.placement_grade(where, _LEVEL, up, _SLOT, 20.0) == "partial"


def test_tilt_tolerance(scene_mod: Any) -> None:
    where = np.array([0.2, 0.3, 0.55])
    tilted = scene_mod._rpy_quat(np.radians(15.0), 0.0, 0.0)
    assert scene_mod.placement_grade(where, tilted, _UP, _SLOT, 20.0) == "complete"
    assert scene_mod.placement_grade(where, tilted, _UP, _SLOT, 10.0) == "partial"


def test_episode_grade_is_the_worst_placement(scene_mod: Any) -> None:
    assert scene_mod.worst_grade(["complete", "complete"]) == "complete"
    assert scene_mod.worst_grade(["complete", "partial", "semi"]) == "semi"
    assert scene_mod.worst_grade(["complete", "fail"]) == "fail"


def test_rest_timer_needs_a_still_object(scene_mod: Any) -> None:
    dt, speed = 1 / 30, 0.02
    rest = 0.0
    rest = scene_mod.rest_after(rest, None, dt, speed)  # first graded step: no history
    for _ in range(30):
        rest = scene_mod.rest_after(rest, 0.0002, dt, speed)
    assert rest == pytest.approx(1.0)
    assert scene_mod.rest_after(rest, 0.01, dt, speed) == 0.0  # 0.3 m/s: moving again


def test_reward_ends_the_episode_only_on_complete(scene_mod: Any) -> None:
    assert scene_mod.placement_reward("complete", False) == (1.0, True)
    assert scene_mod.placement_reward("partial", False) == (0.0, False)
    assert scene_mod.placement_reward("partial", True) == (0.5, False)
    assert scene_mod.placement_reward("semi", True) == (0.25, False)
    assert scene_mod.placement_reward("fail", True) == (0.0, False)
