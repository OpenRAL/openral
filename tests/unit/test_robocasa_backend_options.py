"""Schema validator tests for :class:`RoboCasaBackendOptions`.

The RoboCasa scene adapter (issue #88 PR B, not yet on disk) consumes
``SceneSpec.backend_options`` via
``RoboCasaBackendOptions.model_validate(scene.backend_options)``. This
file pins the validator contract — prebuilt-vs-procedural XOR,
``extra="forbid"``, JSON round-trip — so PR B can rely on it without
re-checking.

ADR-0011 (`docs/adr/0011-robocasa-isolated-backend-lazy-assets.md`) is
the parent design document.

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
