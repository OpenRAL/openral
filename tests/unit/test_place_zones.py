"""Declared place zones (``PlaceZone``) and the prompt → zone mapping, on the restock fixture."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
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
    clash = zones[0].model_copy(update={"name": "lower_slot_9", "aliases": ("upper_2",)})
    with pytest.raises(ValidationError, match="unique"):
        DeployScene.model_validate(
            {**data, "place_zones": [z.model_dump() for z in (*zones, clash)]}
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
