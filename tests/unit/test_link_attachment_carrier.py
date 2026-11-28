"""A carried prim with ``collision`` becomes a rigid link of the robot (sim_scene_description)."""

from __future__ import annotations

import pytest
from openral_core import RobotDescription, sim_scene_description
from openral_core.exceptions import ROSConfigError
from openral_safety.envelope_loader import collision_params_from_description

AMR = "robots/openamrobot_v2/robot.yaml"
FLOOR = {
    "shape": {"shape": "box", "half_extents_m": [0.12, 0.18, 0.005]},
    "origin_xyz_rpy": [0.37, 0.0, -0.42, 0.0, 0.0, 0.0],
}


def _tote(robot: str | None = None, **extra: object) -> dict[str, object]:
    return {
        "prim": "Tote",
        "link": "lift_carriage_link",
        "robot": robot,
        "collision": [FLOOR],
        **extra,
    }


def test_carrier_is_a_rigid_link_of_the_carrying_link_with_its_rigid_pairs_allowed() -> None:
    desc = sim_scene_description(RobotDescription.from_yaml(AMR), {"link_attachments": [_tote()]})
    params = collision_params_from_description(desc)
    names = list(params["collision_link_names"])
    tote = names.index("Tote")
    assert names[params["collision_parent"][tote]] == "lift_carriage_link"
    assert sum(1 for b in params["collision_box_link"] if b == tote) == 1
    pairs = set(desc.allowed_collision_pairs)
    # Rigid with the carriage (no joint between them): allowed. The arms move: checked.
    assert ("Tote", "lift_carriage_link") in pairs
    assert ("Tote", "adapter_plate_link") in pairs
    assert ("Tote", "openarm_left_base_link") in pairs  # the arm's pedestal is bolted on
    assert ("Tote", "openarm_left_ee_link1") in pairs  # the hand reaches in for the contents
    assert ("Tote", "openarm_left_ee_base_link") in pairs
    assert ("Tote", "openarm_left_link5") not in pairs  # the forearm is checked
    assert ("Tote", "lift_column_base_link") not in pairs  # the lift moves the carriage


def test_only_this_scene_robots_carrier_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_SCENE_ROBOT", "a")
    base = RobotDescription.from_yaml(AMR)
    other = sim_scene_description(base, {"link_attachments": [_tote("b")]})
    assert other is base
    mine = sim_scene_description(base, {"link_attachments": [_tote("a")]})
    assert "Tote" in {f.child_link for f in mine.fixed_attachments}


def test_no_collision_changes_nothing_and_a_bad_carrier_is_refused() -> None:
    base = RobotDescription.from_yaml(AMR)
    plain = {"prim": "Tote", "link": "lift_carriage_link"}
    assert sim_scene_description(base, {"link_attachments": [plain]}) is base
    with pytest.raises(ROSConfigError, match="not a link"):
        sim_scene_description(base, {"link_attachments": [_tote(link="nope")]})
    with pytest.raises(ROSConfigError, match="already has"):
        sim_scene_description(base, {"link_attachments": [_tote(prim="base_link")]})
