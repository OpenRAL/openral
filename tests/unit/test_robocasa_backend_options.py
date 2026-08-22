"""Schema validator tests for :class:`RoboCasaBackendOptions`.

The RoboCasa scene adapter (issue #88 PR B, not yet on disk) consumes
``SceneSpec.backend_options`` via
``RoboCasaBackendOptions.model_validate(scene.backend_options)``. This
file pins the validator contract — prebuilt-vs-procedural XOR,
``extra="forbid"``, JSON round-trip — so PR B can rely on it without
re-checking.

CLAUDE.md §1.11 — no mocks, no smoke tests. The model is exercised
against real Pydantic validation paths and a real JSON round-trip.
"""

from __future__ import annotations

import pytest
from openral_core import RoboCasaBackendOptions
from pydantic import ValidationError


def test_prebuilt_minimal_valid() -> None:
    """The minimal valid prebuilt config sets just ``prebuilt_task``."""
    opts = RoboCasaBackendOptions(mode="prebuilt", prebuilt_task="PnPCounterToCab")
    assert opts.prebuilt_task == "PnPCounterToCab"
    assert opts.task_verb is None
    assert opts.robots == ["PandaMobile"]
    assert opts.controller == "OSC_POSE"
    assert opts.horizon == 500


def test_procedural_minimal_valid() -> None:
    """A procedural config needs at least one procedural key (e.g. task_verb)."""
    opts = RoboCasaBackendOptions(
        mode="procedural",
        kitchen_style=3,
        layout_id=7,
        spawn_objects=["coffee_cup", "apple"],
        task_verb="pnp",
    )
    assert opts.prebuilt_task is None
    assert opts.kitchen_style == 3
    assert opts.layout_id == 7
    assert opts.task_verb == "pnp"
    assert opts.spawn_objects == ["coffee_cup", "apple"]


def test_prebuilt_rejects_procedural_keys() -> None:
    """Setting procedural keys while ``mode='prebuilt'`` is a validator error."""
    with pytest.raises(ValidationError) as excinfo:
        RoboCasaBackendOptions(
            mode="prebuilt",
            prebuilt_task="PnPCounterToCab",
            kitchen_style=2,
        )
    assert "procedural keys" in str(excinfo.value)


def test_prebuilt_requires_prebuilt_task() -> None:
    """``mode='prebuilt'`` without ``prebuilt_task`` fails the XOR validator."""
    with pytest.raises(ValidationError) as excinfo:
        RoboCasaBackendOptions(mode="prebuilt")
    assert "prebuilt_task" in str(excinfo.value)


def test_procedural_rejects_prebuilt_task() -> None:
    """``mode='procedural'`` with ``prebuilt_task`` set fails the XOR validator."""
    with pytest.raises(ValidationError) as excinfo:
        RoboCasaBackendOptions(
            mode="procedural",
            prebuilt_task="PnPCounterToCab",
            task_verb="pnp",
        )
    assert "prebuilt_task" in str(excinfo.value)


def test_procedural_requires_at_least_one_procedural_key() -> None:
    """``mode='procedural'`` without any procedural keys is rejected."""
    with pytest.raises(ValidationError) as excinfo:
        RoboCasaBackendOptions(mode="procedural")
    assert "procedural" in str(excinfo.value)


def test_kitchen_style_range_check() -> None:
    """``kitchen_style`` and ``layout_id`` are bounded to RoboCasa's 0–9 packs."""
    with pytest.raises(ValidationError):
        RoboCasaBackendOptions(mode="procedural", kitchen_style=10, task_verb="pnp")
    with pytest.raises(ValidationError):
        RoboCasaBackendOptions(mode="procedural", layout_id=-1, task_verb="pnp")


def test_extra_forbid() -> None:
    """Unknown fields are rejected — no silent ``backend_options`` drift."""
    with pytest.raises(ValidationError) as excinfo:
        RoboCasaBackendOptions.model_validate(
            {
                "mode": "prebuilt",
                "prebuilt_task": "PnPCounterToCab",
                "not_a_real_field": "oops",
            }
        )
    assert "not_a_real_field" in str(excinfo.value)


def test_task_verb_literal() -> None:
    """``task_verb`` is a closed enum — typos fail at validate time."""
    with pytest.raises(ValidationError):
        RoboCasaBackendOptions(
            mode="procedural",
            task_verb="grasp",  # type: ignore[arg-type]
        )


def test_json_round_trip() -> None:
    """JSON serialisation round-trip preserves every populated field.

    Configs in ``SceneSpec.backend_options`` flow through JSON-Schema
    export and YAML configs; the round-trip pins both directions.
    """
    src = RoboCasaBackendOptions(
        mode="procedural",
        kitchen_style=1,
        layout_id=4,
        fixtures=["sink", "stovetop"],
        spawn_objects=["coffee_cup"],
        task_verb="open",
        robots=["PandaMobile", "GR1"],
        controller="JOINT_VELOCITY",
        horizon=350,
    )
    dumped = src.model_dump_json()
    restored = RoboCasaBackendOptions.model_validate_json(dumped)
    assert restored == src


def test_construct_from_dict_matches_adapter_path() -> None:
    """The adapter path is ``model_validate(scene.backend_options: dict)``.

    Confirm a ``dict[str, object]`` payload validates the same way as
    direct kwargs — that is the call the RoboCasa adapter (issue #88 PR B)
    will make at scene-factory time.
    """
    payload: dict[str, object] = {
        "mode": "prebuilt",
        "prebuilt_task": "OpenSingleDoor",
    }
    via_dict = RoboCasaBackendOptions.model_validate(payload)
    direct = RoboCasaBackendOptions(mode="prebuilt", prebuilt_task="OpenSingleDoor")
    assert via_dict == direct


def test_xr1_state_layout_is_supported() -> None:
    opts = RoboCasaBackendOptions(
        mode="prebuilt",
        prebuilt_task="PickPlaceCounterToCabinet",
        state_layout="xr1_8d",
    )
    assert opts.state_layout == "xr1_8d"


def test_fridge_scene_accepts_the_documented_layout_pin() -> None:
    """The remedy the fridge scene documents must actually validate in prebuilt mode.

    ``scenes/deploy/robocasa_fridge_drawer.yaml`` carries a KNOWN DEFECT block
    telling the next operator to pin ``layout_ids`` (and a ``style_ids`` outside
    the task's ``EXCLUDE_STYLES``) to escape an initial pose the safety kernel
    refuses. That advice is only actionable if the scene-pool restrictors
    survive the prebuilt/procedural XOR — they are deliberately independent of
    ``mode``, unlike the procedural-only singular ``layout_id``.

    Driven off the real scene fixture (CLAUDE.md §1.11), so this breaks if the
    shipped YAML and the documented remedy ever drift apart.
    """
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(Path("scenes/deploy/robocasa_fridge_drawer.yaml").read_text())
    payload = dict(raw["scene"]["backend_options"])
    assert payload["prebuilt_task"] == "PickPlaceFridgeShelfToDrawer"

    # As shipped: no pin, so the Kitchen ctor keeps its own sampling.
    as_shipped = RoboCasaBackendOptions.model_validate(payload)
    assert as_shipped.layout_ids is None
    assert as_shipped.style_ids is None

    # With the documented pin applied — layout 3 is a bottom-freezer kitchen and
    # style 1 is outside EXCLUDE_STYLES, so the Kitchen pool is non-empty.
    pinned = RoboCasaBackendOptions.model_validate({**payload, "layout_ids": 3, "style_ids": 1})
    assert pinned.layout_ids == 3
    assert pinned.style_ids == 1
    assert pinned.prebuilt_task == "PickPlaceFridgeShelfToDrawer"


def test_singular_layout_id_is_not_the_prebuilt_pin() -> None:
    """The trap the fridge remedy has to avoid: ``layout_id`` is not ``layout_ids``.

    ``layout_id`` (singular) is a procedural-authoring key and is rejected under
    ``mode="prebuilt"``. Anyone applying the fridge fix by analogy with the
    procedural docs would hit this, so pin the distinction.
    """
    with pytest.raises(ValidationError) as excinfo:
        RoboCasaBackendOptions(
            mode="prebuilt",
            prebuilt_task="PickPlaceFridgeShelfToDrawer",
            layout_id=3,
        )
    assert "procedural keys" in str(excinfo.value)
