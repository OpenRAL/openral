"""Declared place zones (``PlaceZone``) and the prompt → zone mapping, on the restock fixture."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from openral_core import DeployScene, PlaceZone, place_zone_for_prompt
from pydantic import TypeAdapter, ValidationError

_ROOT = Path(__file__).parents[2]
_ZONES = _ROOT / "scenes/deploy/place_zones/openarm_restock_shelf.yaml"


def _zones() -> list[PlaceZone]:
    raw = yaml.safe_load(_ZONES.read_text())["place_zones"]
    return TypeAdapter(list[PlaceZone]).validate_python(raw)


@pytest.mark.parametrize(
    ("prompt", "name"),
    [
        ("restock box x3 into upper slot 2", "upper_slot_2"),
        ("Restock box x1 into LOWER slot 3.", "lower_slot_3"),
        ("restock box x1 into upper_1", "upper_slot_1"),  # an alias
        ("restock box x3 into upper slot 20", None),  # no such slot: whole words only
        ("restock box x3 onto the shelf", None),  # names no zone: fail closed
        ("move upper slot 1 into upper slot 2", None),  # two zones: not ours to pick
    ],
)
def test_the_prompt_names_one_declared_zone_or_none(prompt: str, name: str | None) -> None:
    zone = place_zone_for_prompt(prompt, _zones())
    assert (zone.name if zone is not None else None) == name


def test_the_restock_zones_lie_in_the_robot_base_frame_and_validate_in_a_deploy_scene() -> None:
    zones = _zones()
    assert {z.region.frame_id for z in zones} == {"openarm_base"}
    upper_2 = next(z for z in zones if z.name == "upper_slot_2")
    # The slot the restock prompt names: x 0.339..0.715, y 0.152..0.276 in openarm_base.
    (x, y, _), (hx, hy, _) = upper_2.region.pose.xyz, upper_2.region.half_extents
    assert (x - hx, x + hx, y - hy, y + hy) == pytest.approx((0.339, 0.715, 0.152, 0.276), abs=1e-3)
    scene = DeployScene.from_yaml(str(_ROOT / "scenes/deploy/openarm_bench.yaml"))
    data = scene.model_dump(exclude_unset=True)
    with_zones = DeployScene.model_validate(
        {**data, "place_zones": [z.model_dump() for z in zones]}
    )
    assert with_zones.place_zones == zones
    # A repeated alias among base-frame zones: the scene is one frame, so the robot-aware
    # deploy refuses it (`_check_place_zone_frames`), as before such repeats were allowed.
    from openral_cli.deploy_sim import _check_place_zone_frames
    from openral_core import RobotDescription
    from openral_core.exceptions import ROSConfigError

    clash = zones[0].model_copy(update={"name": "lower_slot_9", "aliases": ("upper_2",)})
    repeated = DeployScene.model_validate(
        {**data, "place_zones": [z.model_dump() for z in (*zones, clash)]}
    )
    openarm = RobotDescription.from_yaml(str(_ROOT / "robots/openarm/robot.yaml"))
    with pytest.raises(ROSConfigError, match="repeat the alias"):
        _check_place_zone_frames(repeated.place_zones, openarm, "openarm", map_published=True)
    # A repeated NAME, or an alias that is another zone's name, is refused by the scene itself.
    for bad in (
        zones[0].model_copy(update={"aliases": ()}),
        zones[0].model_copy(update={"name": "lower_slot_9", "aliases": ("upper_slot_2",)}),
    ):
        with pytest.raises(ValidationError, match="unique"):
            DeployScene.model_validate(
                {**data, "place_zones": [z.model_dump() for z in (*zones, bad)]}
            )


def test_a_zone_refuses_a_malformed_name_and_target_geometry() -> None:
    zone = _zones()[0]
    with pytest.raises(ValidationError):
        PlaceZone.model_validate({**zone.model_dump(), "name": "Upper Slot 2"})
    region = zone.region.model_dump()
    region["geometry"] = [
        {
            "shape": {"shape": "sphere", "radius_m": 0.01},
            "pose_in_object": zone.region.pose.model_dump(),
        }
    ]
    with pytest.raises(ValidationError, match="geometry must be empty"):
        PlaceZone.model_validate({**zone.model_dump(), "region": region})


def test_a_zone_refuses_a_pose_in_another_frame_than_its_region() -> None:
    zone = _zones()[0]
    region = zone.region.model_dump()
    region["pose"]["frame_id"] = "world"
    with pytest.raises(ValidationError, match="not the region's"):
        PlaceZone.model_validate({**zone.model_dump(), "region": region})


@settings(max_examples=40, deadline=None)
@given(
    frame=st.sampled_from(["map", "odom"]),
    x=st.floats(-50.0, 50.0),
    y=st.floats(-50.0, 50.0),
    yaw=st.floats(-3.14, 3.14),
)
def test_a_map_frame_zone_is_the_same_model_and_round_trips_in_a_deploy_scene(
    frame: str, x: float, y: float, yaw: float
) -> None:
    """Station mode adds no field: the fixture's slot, redeclared in a fixed frame at any
    yaw, validates and round-trips through JSON and a real deploy scene unchanged (the
    frame is checked against the robot by the deploy, not here)."""
    zone = _zones()[0]
    pose = zone.region.pose.model_copy(
        update={
            "xyz": (x, y, 0.62),
            "quat_xyzw": (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
            "frame_id": frame,
        }
    )
    region = zone.region.model_copy(update={"frame_id": frame, "pose": pose})
    shelf = PlaceZone.model_validate({**zone.model_dump(), "region": region.model_dump()})
    assert PlaceZone.model_validate_json(shelf.model_dump_json()) == shelf
    scene = DeployScene.from_yaml(str(_ROOT / "scenes/deploy/openamrobot_v2_empty.yaml"))
    data = scene.model_dump(exclude_unset=True)
    with_zones = DeployScene.model_validate({**data, "place_zones": [shelf.model_dump()]})
    assert DeployScene.model_validate_json(with_zones.model_dump_json()).place_zones == [shelf]


def _station_zones(alias: str = "lower_1") -> list[PlaceZone]:
    """The aisle's convention: the twin's slot at six shelves, ``s<k>_<twin name>``, in map."""
    twin = next(z for z in _zones() if z.name == "lower_slot_1")
    out = []
    for k in range(1, 7):
        pose = twin.region.pose.model_copy(update={"xyz": (2.0 * k, 0.0, 0.5), "frame_id": "map"})
        region = twin.region.model_copy(update={"frame_id": "map", "pose": pose})
        out.append(
            PlaceZone.model_validate(
                {"name": f"s{k}_lower_slot_1", "aliases": [alias], "region": region.model_dump()}
            )
        )
    return out


def test_one_slot_at_six_map_frame_stations_loads_and_an_alias_may_not_cross_frames() -> None:
    scene = DeployScene.from_yaml(str(_ROOT / "scenes/deploy/openamrobot_v2_empty.yaml"))
    data = scene.model_dump(exclude_unset=True)
    zones = _station_zones()
    loaded = DeployScene.model_validate({**data, "place_zones": [z.model_dump() for z in zones]})
    assert [z.name for z in loaded.place_zones] == [z.name for z in zones]
    from openral_cli.deploy_sim import _check_place_zone_frames
    from openral_core import RobotDescription

    amr = RobotDescription.from_yaml(str(_ROOT / "robots/openamrobot_v2/robot.yaml"))
    _check_place_zone_frames(
        loaded.place_zones, amr, "openamrobot_v2", map_published=True
    )  # map: accepted
    odom = zones[0].model_copy(
        update={
            "name": "s7_lower_slot_1",
            "region": zones[0].region.model_copy(
                update={
                    "frame_id": "odom",
                    "pose": zones[0].region.pose.model_copy(update={"frame_id": "odom"}),
                }
            ),
        }
    )
    with pytest.raises(ValidationError, match="one frame"):
        DeployScene.model_validate(
            {**data, "place_zones": [z.model_dump() for z in (*zones, odom)]}
        )


@pytest.mark.parametrize(
    ("prompt", "matched"),
    [
        ("put box x1 into lower 1", 6),  # the twin's alias, repeated on every shelf
        ("put box x1 into lower slot 1", 0),  # the `s<k>_` prefix: no whole-word name hit
        ("put box x1 into lower 10", 0),  # whole words only
    ],
)
def test_the_prefixed_station_names_match_the_prompt_only_through_an_alias(
    prompt: str, matched: int
) -> None:
    """``place_zone_for_prompt`` matches a whole-word NAME or ALIAS. ``s3_lower_slot_1`` is not
    a whole word of "... lower slot 1", so the trained prompt matches a station zone only if
    the zone also carries the alias ``lower_slot_1`` (``test_..._with_the_slot_alias``)."""
    zones = _station_zones()
    assert sum(place_zone_for_prompt(prompt, [z]) is not None for z in zones) == matched


def test_with_the_slot_alias_the_trained_prompt_matches_all_six_stations() -> None:
    zones = [
        z.model_copy(update={"aliases": ("lower_1", "lower_slot_1")}) for z in _station_zones()
    ]
    prompt = "put box x1 into lower slot 1"
    assert sum(place_zone_for_prompt(prompt, [z]) is not None for z in zones) == 6
    assert place_zone_for_prompt(prompt, zones) is None  # several: the tracker chooses by reach


# The private Pro checkout holding the aisle scene; unset (CI, other hosts) skips.
_AISLE = Path(os.environ.get("OPENRAL_PRO_DIR") or "/nonexistent") / (
    "scenes/deploy/amr_shelf_aisle.yaml"
)


@pytest.mark.skipif(
    not _AISLE.exists(), reason="set OPENRAL_PRO_DIR to a Pro checkout with the aisle scene"
)
def test_the_pro_aisle_scene_loads_with_its_repeated_station_aliases() -> None:
    scene = DeployScene.from_yaml(str(_AISLE))
    zones = scene.place_zones
    assert {z.region.frame_id for z in zones} == {"map"}
    slot = [z for z in zones if z.name.endswith("_lower_slot_1")]
    assert sorted(z.name for z in slot) == [f"s{k}_lower_slot_1" for k in range(1, 7)]
    by_alias = [z for z in zones if place_zone_for_prompt("put box x1 into lower 1", [z])]
    assert sorted(z.name for z in by_alias) == sorted(z.name for z in slot)


@settings(max_examples=25, deadline=None)
@given(
    stations=st.integers(2, 8),
    frame=st.sampled_from(["map", "odom"]),
    other=st.sampled_from(["odom", "openarm_base", "map"]),
)
def test_a_repeated_alias_round_trips_in_one_frame_and_is_refused_across_frames(
    stations: int, frame: str, other: str
) -> None:
    scene = DeployScene.from_yaml(str(_ROOT / "scenes/deploy/openamrobot_v2_empty.yaml"))
    data = scene.model_dump(exclude_unset=True)

    def at(k: int, f: str) -> dict[str, object]:
        z = _station_zones()[0]
        pose = z.region.pose.model_copy(update={"xyz": (2.0 * k, 0.0, 0.5), "frame_id": f})
        region = z.region.model_copy(update={"frame_id": f, "pose": pose})
        return {"name": f"s{k}_lower_slot_1", "aliases": ["lower_1"], "region": region.model_dump()}

    same = [at(k, frame) for k in range(1, stations + 1)]
    loaded = DeployScene.model_validate({**data, "place_zones": same})
    assert DeployScene.model_validate_json(loaded.model_dump_json()) == loaded
    mixed = [*same, at(stations + 1, other)]
    if other == frame:
        DeployScene.model_validate({**data, "place_zones": mixed})
    else:
        with pytest.raises(ValidationError, match="one frame"):
            DeployScene.model_validate({**data, "place_zones": mixed})
