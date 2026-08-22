"""The deploy scene's committed place-phase declaration (ADR-0097).

A *direct* dispatch has no reasoner in the loop to ground a place target per
goal, so the committed scene is the only thing that knows where the task places
its payload. :attr:`DeployScene.place_declaration` is that statement, and
``openral deploy sim`` / ``deploy run`` inject it into the rSkill runner, which
scopes it to each goal (armed on start, retracted on end / cancel / E-stop).

What a scene may say is deliberately narrower than the wire type: it names a
**target**, never a :class:`PlaceRegion`. The region is what buys the payload a
reduced world-collision margin, and it is sound only because a producer
*measured* the declared target in the frame the occupancy grid uses. A scene
file measures nothing, and ``panda_mobile`` is a mobile base, so a box typed
into YAML is stale the moment the base drives (hazard log HZ-0097-2/4).

Real scene files under ``scenes/deploy/`` throughout — CLAUDE.md §1.11.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from openral_core import DeployScene, PlaceDeclaration
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BAGUETTE = _REPO_ROOT / "scenes" / "deploy" / "robocasa_baguette.yaml"


def _baguette_dict() -> dict[str, Any]:
    """The real #102 acceptance scene, as parsed YAML."""
    loaded = yaml.safe_load(_BAGUETTE.read_text())
    assert isinstance(loaded, dict)
    return copy.deepcopy(loaded)


def _declaration_dict() -> dict[str, Any]:
    return {
        "target_id": "sim:cab_1_left_group_main",
        "object_id": "",
        "timeout_s": 120.0,
        "stamp_ns": 0,
    }


def _region_dict() -> dict[str, Any]:
    """A syntactically valid region — the point is that it is *unmeasured*."""
    return {
        "frame_id": "base_link",
        "pose": {
            "xyz": (0.52, 0.0, 0.40),
            "quat_xyzw": (0.0, 0.0, 0.0, 1.0),
            "frame_id": "base_link",
        },
        "half_extents": (0.20, 0.25, 0.45),
        "evidence_ref": "scene_yaml:hand_written",
    }


def test_a_scene_declaration_names_a_target_and_validates() -> None:
    """The shape a scene is allowed to commit: target, payload scope, backstop."""
    raw = _baguette_dict()
    raw["place_declaration"] = _declaration_dict()
    scene = DeployScene.model_validate(raw)

    assert scene.place_declaration is not None
    assert scene.place_declaration.target_id == "sim:cab_1_left_group_main"
    assert scene.place_declaration.region is None
    assert scene.place_declaration.timeout_s <= PlaceDeclaration.MAX_TIMEOUT_S


def test_a_scene_may_not_supply_the_declared_targets_region() -> None:
    """The fail-closed narrowing: an unmeasured region never reaches the kernel.

    The runner would otherwise publish whatever the scene wrote, and the sim
    producer only *overwrites* the field when it can measure the target — so on
    an unmeasurable target a hand-typed box would arm the approach allowance
    around a volume nobody observed, with the trace still attributing it to a
    producer measurement (HZ-0097-2/4).
    """
    raw = _baguette_dict()
    raw["place_declaration"] = {**_declaration_dict(), "region": _region_dict()}

    with pytest.raises(ValidationError, match="producer-supplied"):
        DeployScene.model_validate(raw)


def test_no_declaration_at_all_is_still_valid() -> None:
    """Absent declaration = pre-ADR-0097 behaviour, which most scenes keep."""
    scene = DeployScene.model_validate(_baguette_dict() | {"place_declaration": None})
    assert scene.place_declaration is None


# -- The committed scenes themselves ------------------------------------------
#
# `scenes/deploy/robocasa_drawer_utensil.yaml` is deliberately NOT here: its
# target is a `stack` level chosen from many per layout, so a guessed name would
# very likely resolve to a different real receptacle instead of failing closed.
# That reasoning lives in the scene file; `test_the_drawer_scene_declares_nothing`
# pins the decision so it cannot be undone by accident.
_DECLARING_SCENES = {
    "robocasa_baguette.yaml": "sim:cab_1_left_group_main",
    "robocasa_sink_cup.yaml": "sim:sink_main_group_1_main",
    "robocasa_fridge_drawer.yaml": "sim:fridge_right_group_main",
}


@pytest.mark.parametrize("scene_file", sorted(_DECLARING_SCENES))
def test_the_place_scenes_declare_their_target(scene_file: str) -> None:
    """Each RoboCasa place scene commits a declaration the runner can arm."""
    scene = DeployScene.from_yaml(str(_REPO_ROOT / "scenes" / "deploy" / scene_file))
    declaration = scene.place_declaration

    assert declaration is not None, f"{scene_file} declares no place target"
    assert declaration.target_id == _DECLARING_SCENES[scene_file]
    # The sim producer resolves the body by stripping this prefix, so a target
    # without it names no MuJoCo body and is refused.
    assert declaration.target_id.startswith("sim:")
    # Direct dispatch discovers the payload at attach time.
    assert declaration.object_id == ""
    # Producer-measured, never scene-supplied.
    assert declaration.region is None
    # Live at goal start (the runner re-stamps), and inside the backstop ceiling.
    assert declaration.active
    assert 0.0 < declaration.timeout_s <= PlaceDeclaration.MAX_TIMEOUT_S
    assert declaration.is_live(now_ns=int(declaration.timeout_s * 1e9))
    assert not declaration.is_live(now_ns=int(declaration.timeout_s * 1e9) + 1)


def test_the_drawer_scene_declares_nothing_until_its_target_is_read_off_a_run() -> None:
    """`PickPlaceCounterToDrawer`'s target is one of many stack levels.

    Its generated name (`<stack>_<group>_<level>`) recurs across layouts pointing
    at a different drawer each time, so a guess would not fail closed — it would
    arm the witness and its approach allowance at whichever real receptacle
    happened to answer to that name (HZ-0097-2).
    """
    scene = DeployScene.from_yaml(
        str(_REPO_ROOT / "scenes" / "deploy" / "robocasa_drawer_utensil.yaml")
    )
    assert scene.place_declaration is None
