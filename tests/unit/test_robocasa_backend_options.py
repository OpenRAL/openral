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
from hypothesis import given
from hypothesis import strategies as st
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


@pytest.mark.parametrize(
    ("scene_file", "prebuilt_task", "expected_layout"),
    [
        ("robocasa_fridge_drawer.yaml", "PickPlaceFridgeShelfToDrawer", 30),
        ("robocasa_drawer_utensil.yaml", "PickPlaceCounterToDrawer", 3),
    ],
)
def test_matrix_scene_ships_its_verified_layout_pin(
    scene_file: str, prebuilt_task: str, expected_layout: int
) -> None:
    """Both pinned matrix scenes carry the layout their header block justifies.

    The fridge pin is a *fix*: unpinned, seed 1 draws a side-by-side kitchen and
    spawns ``robot0_link7_collision`` at 0.000 m from the closed freezer door,
    so the run E-stops on the initial configuration before applying a chunk.
    The utensil pin is a *reproducibility* pin — that scene starts 43.3 mm clear
    unpinned and has no defect — but without it the kitchen is a free draw off
    ``env.rng`` and two rounds at ``seed: 1`` are not comparable.

    Either way the value must survive the prebuilt/procedural XOR: the
    scene-pool restrictors are deliberately independent of ``mode``, unlike the
    procedural-only singular ``layout_id``. Driven off the real scene fixtures
    (CLAUDE.md §1.11) so this breaks the moment a pin is "cleaned up" out of a
    YAML, which is exactly the regression the header comments warn against.
    """
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(Path("scenes/deploy", scene_file).read_text())
    payload = dict(raw["scene"]["backend_options"])
    assert payload["prebuilt_task"] == prebuilt_task

    opts = RoboCasaBackendOptions.model_validate(payload)
    assert opts.layout_ids == [expected_layout]
    assert opts.prebuilt_task == prebuilt_task
    # A single non-negative id is what makes the pin enforceable at reset; a
    # list of several would silently become a draw again.
    assert isinstance(opts.layout_ids, list) and len(opts.layout_ids) == 1


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


# ── Scene-pool id domain ─────────────────────────────────────────────────────
#
# The domain is mirrored from RoboCasa 1.0.1
# `robocasa/models/scenes/scene_registry.py`: layouts and styles are numbered
# 1..60, layouts take the six negative group aliases in `LAYOUT_GROUPS_TO_IDS`
# (-1..-6) and styles the three in `STYLE_GROUPS_TO_IDS` (-1..-3).
#
# Before these validators an out-of-domain pin was accepted here and died much
# later as a bare `KeyError: 99` inside `SceneRegistry.get_layout_path`, from a
# stack that never names the YAML key responsible. A silently-wrong kitchen is
# the failure this whole change exists to prevent, so the loud rejection is
# part of the contract, not a nicety.


@pytest.mark.parametrize("layout", [0, 61, 99, -7, -100])
def test_layout_ids_outside_the_robocasa_domain_are_rejected(layout: int) -> None:
    """A layout id RoboCasa does not define fails at validate time, not in the arena."""
    with pytest.raises(ValidationError) as excinfo:
        RoboCasaBackendOptions(
            mode="prebuilt",
            prebuilt_task="PickPlaceFridgeShelfToDrawer",
            layout_ids=[layout],
        )
    message = str(excinfo.value)
    assert "layout_ids" in message
    assert str(layout) in message


@pytest.mark.parametrize("style", [0, 61, -4])
def test_style_ids_outside_the_robocasa_domain_are_rejected(style: int) -> None:
    """Styles share the 1..60 numbering but only three group aliases (-1..-3)."""
    with pytest.raises(ValidationError) as excinfo:
        RoboCasaBackendOptions(
            mode="prebuilt",
            prebuilt_task="PickPlaceFridgeShelfToDrawer",
            style_ids=style,
        )
    assert "style_ids" in str(excinfo.value)


@pytest.mark.parametrize("layout", [1, 3, 30, 60, -1, -2, -3, -4, -5, -6])
def test_layout_ids_inside_the_domain_survive(layout: int) -> None:
    """Every concrete layout and every group shorthand RoboCasa defines is accepted."""
    opts = RoboCasaBackendOptions(
        mode="prebuilt",
        prebuilt_task="PickPlaceFridgeShelfToDrawer",
        layout_ids=[layout],
    )
    assert opts.layout_ids == [layout]


def test_scalar_and_list_layout_pins_are_both_accepted() -> None:
    """RoboCasa's ``unpack_layout_ids`` takes either shape; so do we."""
    scalar = RoboCasaBackendOptions(
        mode="prebuilt", prebuilt_task="PickPlaceCounterToDrawer", layout_ids=3
    )
    listed = RoboCasaBackendOptions(
        mode="prebuilt", prebuilt_task="PickPlaceCounterToDrawer", layout_ids=[3]
    )
    assert scalar.layout_ids == 3
    assert listed.layout_ids == [3]


def test_unknown_layout_and_style_shorthand_is_rejected() -> None:
    """Only ``5x5`` / ``5x1`` resolve upstream; any other string is a silent no-op.

    ``Kitchen.__init__`` matches those two strings and falls through both
    branches otherwise, leaving ``self.layout_and_style_ids`` never assigned —
    so an unknown shorthand surfaces as an ``AttributeError`` on an unrelated
    line rather than as a bad scene key.
    """
    with pytest.raises(ValidationError) as excinfo:
        RoboCasaBackendOptions(
            mode="prebuilt",
            prebuilt_task="PickPlaceCounterToDrawer",
            layout_and_style_ids="7x7",
        )
    assert "5x1" in str(excinfo.value)


@pytest.mark.parametrize("shorthand", ["5x5", "5x1"])
def test_known_layout_and_style_shorthands_survive(shorthand: str) -> None:
    """The two upstream shorthands validate unchanged."""
    opts = RoboCasaBackendOptions(
        mode="prebuilt",
        prebuilt_task="PickPlaceCounterToDrawer",
        layout_and_style_ids=shorthand,
    )
    assert opts.layout_and_style_ids == shorthand


def test_layout_and_style_pairs_are_range_checked() -> None:
    """A pair list is checked per axis, with the layout and style domains kept apart."""
    ok = RoboCasaBackendOptions(
        mode="prebuilt",
        prebuilt_task="PickPlaceCounterToDrawer",
        layout_and_style_ids=[[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]],
    )
    assert ok.layout_and_style_ids == [[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]]

    with pytest.raises(ValidationError):
        RoboCasaBackendOptions(
            mode="prebuilt",
            prebuilt_task="PickPlaceCounterToDrawer",
            layout_and_style_ids=[[61, 1]],
        )
    with pytest.raises(ValidationError) as excinfo:
        RoboCasaBackendOptions(
            mode="prebuilt",
            prebuilt_task="PickPlaceCounterToDrawer",
            layout_and_style_ids=[[1, 2, 3]],
        )
    assert "[layout, style] pair" in str(excinfo.value)


def test_layout_and_style_ids_is_exclusive_with_the_two_axes() -> None:
    """Upstream asserts on the combination; reject it here where the keys are named."""
    with pytest.raises(ValidationError) as excinfo:
        RoboCasaBackendOptions(
            mode="prebuilt",
            prebuilt_task="PickPlaceCounterToDrawer",
            layout_and_style_ids=[[1, 1]],
            layout_ids=[3],
        )
    assert "mutually" in str(excinfo.value)


@given(
    layout=st.one_of(
        st.integers(min_value=1, max_value=60),
        st.sampled_from([-1, -2, -3, -4, -5, -6]),
    ),
    style=st.one_of(
        st.integers(min_value=1, max_value=60),
        st.sampled_from([-1, -2, -3]),
    ),
    horizon=st.integers(min_value=1, max_value=5000),
)
def test_scene_pool_pins_round_trip_through_json(layout: int, style: int, horizon: int) -> None:
    """Every in-domain pin survives the JSON round-trip the scene YAML path uses.

    CLAUDE.md §2 — ``hypothesis`` round-trip on the Pydantic surface this change
    touches. The generated domain is exactly the domain the validators accept,
    so a narrowing of either side breaks this.
    """
    src = RoboCasaBackendOptions(
        mode="prebuilt",
        prebuilt_task="PickPlaceFridgeShelfToDrawer",
        layout_ids=[layout],
        style_ids=style,
        horizon=horizon,
    )
    restored = RoboCasaBackendOptions.model_validate_json(src.model_dump_json())
    assert restored == src
    assert restored.layout_ids == [layout]
    assert restored.style_ids == style
