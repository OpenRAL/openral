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
        "timeout_s": 300.0,
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
# Every name here is one a live seed-1 env answered to. `robocasa_baguette` was
# read off the rounds 5/6 artifacts and confirmed by the 2026-08-22 Spark run
# (region measured, allowance published); the other two were #142 guesses that
# the HAL refused fail-closed (`... names no MuJoCo body; refused`) and were
# replaced with the names that run's MuJoCo body table actually carries.
_DECLARING_SCENES = {
    "robocasa_baguette.yaml": "sim:cab_1_left_group_main",
    "robocasa_sink_cup.yaml": "sim:sink_island_group_main",
    # The fridge target is a function of that scene's `layout_ids` pin, which
    # moved 30 -> 47 once the pin was verified against the KERNEL criterion on a
    # live octomap (layout 30 clears the mesh but stops at -23.47 mm). At layout
    # 47 the kitchen composes to style 32 and the fixture is
    # `fridgesidebyside_main_group_1`, so the old body no longer exists. Anyone
    # changing that pin must re-resolve this in the same commit: the HAL fails
    # closed on a stale target, so the symptom is a silently unarmed place phase
    # rather than an error.
    "robocasa_fridge_drawer.yaml": "sim:fridgesidebyside_main_group_1_fridge_drawer0",
}

#: Arm→grasp latency measured on `robocasa_baguette` at seed 1 (Spark,
#: 2026-08-22), in the SIMULATOR clock domain the declaration is stamped in.
#: The declaration arms at goal start, so this much of the backstop is spent
#: before the place phase can begin at all.
_MEASURED_ARM_TO_GRASP_S = 92.0


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


@pytest.mark.parametrize("scene_file", sorted(_DECLARING_SCENES))
def test_the_backstop_outlives_the_transport_phase(scene_file: str) -> None:
    """The backstop must not expire before the phase it exists to scope.

    #142 sized `timeout_s` at 120 s from the 500-step / 20 Hz benchmark horizon,
    but a `deploy sim` goal is continuous stepping and is not bounded by that
    horizon. The live baguette run then spent 92 s of simulator time between
    arming and the grasp, leaving ~28 s for the entire place phase — a window
    that expires before the allowance it carries can ever be used, which is
    indistinguishable from shipping no declaration at all.
    """
    scene = DeployScene.from_yaml(str(_REPO_ROOT / "scenes" / "deploy" / scene_file))
    declaration = scene.place_declaration
    assert declaration is not None

    # Still live at the measured grasp, with the place phase itself yet to run.
    assert declaration.is_live(now_ns=int(_MEASURED_ARM_TO_GRASP_S * 1e9))
    # And with more window left afterwards than was consumed getting there.
    remaining_s = declaration.timeout_s - _MEASURED_ARM_TO_GRASP_S
    assert remaining_s > _MEASURED_ARM_TO_GRASP_S, (
        f"{scene_file}: only {remaining_s:.1f} s of place window after a "
        f"{_MEASURED_ARM_TO_GRASP_S:.0f} s transport"
    )


def test_the_drawer_scene_declares_nothing_until_its_target_is_read_off_a_run() -> None:
    """`PickPlaceCounterToDrawer`'s target is one of many stack levels.

    Its generated name (`<stack>_<group>_<level>`) recurs across layouts pointing
    at a different drawer each time, so a guess would not fail closed — it would
    arm the witness and its approach allowance at whichever real receptacle
    happened to answer to that name (HZ-0097-2).

    The 2026-08-22 live run sharpened this rather than resolving it: at seed 1
    the only bodies whose names match `drawer` are the FRIDGE's freezer drawers,
    while the counter drawer this task places into is `stack_2_left_group_3_*`.
    A reader who greps the body table for `drawer` therefore gets a name that
    resolves — to a receptacle across the kitchen — and a resolving name cannot
    fail closed. The scene stays undeclared until a run establishes which stack
    level `register_fixture_ref("drawer", ...)` actually returned.
    """
    scene = DeployScene.from_yaml(
        str(_REPO_ROOT / "scenes" / "deploy" / "robocasa_drawer_utensil.yaml")
    )
    assert scene.place_declaration is None
