"""The robot manifest and the robot enrolment (ADR-0129 Interfaces A and B).

Manifests are built from real ``robots/*/robot.yaml`` and ``rskills/*/rskill.yaml``
fixtures: a fixed arm (``so101_follower``) and a mobile manipulator with a lidar and a
depth camera (``panda_mobile``).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from openral_core import (
    DeployRuntime,
    EmbodimentKind,
    ManifestAction,
    ManifestPanel,
    RobotDescription,
    RobotEnrolment,
    RobotManifest,
    RSkillManifest,
    build_robot_manifest,
    parse_robot_link_code,
    rskill_matches_robot,
)
from openral_core.exceptions import ROSConfigError
from openral_reasoner.palette import build_tool_palette

_REPO = Path(__file__).resolve().parents[2]
_NAV2 = "OpenRAL/rskill-nav2-mobile_base-navigate_to_pose"


def _robot(robot_id: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots" / robot_id / "robot.yaml"))


def _installed() -> list[RSkillManifest]:
    return [RSkillManifest.from_yaml(str(p)) for p in sorted(_REPO.glob("rskills/*/rskill.yaml"))]


def test_fixed_arm_manifest() -> None:
    robot = _robot("so101_follower")
    manifest = build_robot_manifest(
        robot,
        DeployRuntime(enable_reasoner=True),
        _installed(),
        bridge_on=False,
        write_controls=True,
        remote_control=False,
        unit_serial="SN-123",
    )
    assert (manifest.robot_id, manifest.embodiment_kind) == ("so101_follower", "manipulator")
    assert (manifest.unit_serial, manifest.base_frame) == ("SN-123", robot.base_frame)
    panels = [p.id for p in manifest.panels]
    assert panels[:4] == ["identity", "cam:top", "cam:wrist", "joints"]
    assert {"reasoner", "rskill", "safety_status", "safety_ledger", "system"} <= set(panels)
    assert {"metrics", "events", "world_state"} <= set(panels)
    assert not {"map", "pointcloud", "reward", "viz3d"} & set(panels)  # off on this deploy
    cam = manifest.panels[1]
    assert (cam.type, cam.source) == ("camera", "top")
    actions = {a.id: a for a in manifest.actions}
    assert manifest.actions[0].id == "estop"
    assert (actions["estop"].min_role, actions["estop"].remote) == ("viewer", True)
    assert actions["estop_reset"].confirm == "Is the work cell clear?"
    assert not actions["estop_reset"].remote  # remote control off
    assert actions["prompt"].args_schema is not None
    assert actions["prompt"].args_schema["required"] == ["text"]
    skills = sorted(a for a in actions if a.startswith("skill:"))
    assert skills == [
        "skill:OpenRAL/rskill-molmoact2-multi-so101-nf4",
        "skill:OpenRAL/rskill-smolvla-so101-eraser_place-bf16",
        "skill:OpenRAL/rskill-smolvla-so101-leisaac_pick_orange-bf16",
    ]
    assert actions[skills[1]].label == "smolvla-so101-eraser_place-bf16"
    assert actions[skills[1]].args_schema == {"type": "object", "properties": {}}


def test_mobile_manipulator_manifest() -> None:
    robot = _robot("panda_mobile")
    runtime = DeployRuntime(
        enable_reasoner=True, enable_slam=True, enable_octomap=True, enable_reward_monitor=True
    )
    manifest = build_robot_manifest(
        robot, runtime, _installed(), bridge_on=True, write_controls=True, remote_control=True
    )
    panels = {p.id: p for p in manifest.panels}
    # Image sensors become cameras; the lidar does not.
    assert {p.source for p in manifest.panels if p.type == "camera"} == {
        "head",
        "shoulder_left",
        "shoulder_right",
        "wrist",
        "front_depth",
    }
    assert {"map", "pointcloud", "reward", "viz3d"} <= set(panels)
    nav = next(a for a in manifest.actions if a.id == f"skill:{_NAV2}")
    nav_manifest = RSkillManifest.from_yaml(
        str(_REPO / "rskills/rskill-nav2-navigate-to-pose/rskill.yaml")
    )
    assert nav.args_schema == nav_manifest.goal_params_schema
    assert all(a.remote for a in manifest.actions)  # remote control on


def test_skills_stay_out_without_write_controls_and_prompt_without_a_reasoner() -> None:
    manifest = build_robot_manifest(
        _robot("panda_mobile"),
        DeployRuntime(enable_reasoner=False),
        _installed(),
        bridge_on=False,
        write_controls=False,
        remote_control=False,
    )
    assert [a.id for a in manifest.actions] == ["estop", "estop_reset"]
    assert "reasoner" not in {p.id for p in manifest.panels}


@pytest.mark.parametrize("robot_id", ["so101_follower", "panda_mobile", "franka_panda"])
@pytest.mark.parametrize("commercial", [False, True])
def test_skill_actions_are_exactly_the_reasoner_palette(robot_id: str, commercial: bool) -> None:
    # One predicate, two consumers: what the robot UI offers is what the LLM may dispatch.
    robot = _robot(robot_id)
    installed = _installed()
    manifest = build_robot_manifest(
        robot,
        DeployRuntime(),
        installed,
        bridge_on=False,
        write_controls=True,
        remote_control=False,
        commercial_deployment=commercial,
    )
    palette = build_tool_palette(
        installed_skills=installed,
        robot_capabilities=robot.capabilities,
        commercial_deployment=commercial,
    )
    offered = {a.id.removeprefix("skill:") for a in manifest.actions if a.kind == "skill"}
    assert offered == set(palette.execute_rskill_ids)
    assert offered == {
        s.name
        for s in installed
        if rskill_matches_robot(s, robot.capabilities, commercial_deployment=commercial)
    }


def test_the_scaffold_template_is_never_offered() -> None:
    template = RSkillManifest.from_yaml(str(_REPO / "rskills/template/rskill.yaml"))
    for robot_id in ("so101_follower", "panda_mobile"):
        assert not rskill_matches_robot(template, _robot(robot_id).capabilities)


# ── hypothesis round-trips (CLAUDE.md §2) ─────────────────────────────────────

_id = st.text(st.characters(codec="utf-8", exclude_categories=["Cs"]), min_size=1, max_size=20)
_json_leaf = st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=8))
_schema = st.dictionaries(st.text(max_size=8), _json_leaf, max_size=4)
_panel_st = st.builds(
    ManifestPanel,
    id=_id,
    type=st.sampled_from(["identity", "camera", "joints", "viz3d", "events"]),
    title=st.text(max_size=20),
    source=st.none() | _id,
)
_action_st = st.builds(
    ManifestAction,
    id=_id,
    kind=st.sampled_from(["estop", "estop_reset", "prompt", "skill"]),
    label=st.text(max_size=20),
    min_role=st.sampled_from(["viewer", "operator"]),
    remote=st.booleans(),
    confirm=st.none() | st.text(max_size=20),
    args_schema=st.none() | _schema,
)
_manifest_st = st.builds(
    RobotManifest,
    robot_id=_id,
    name=st.text(max_size=20),
    embodiment_kind=st.sampled_from(list(EmbodimentKind)),
    unit_serial=st.none() | _id,
    base_frame=_id,
    panels=st.lists(_panel_st, max_size=5),
    actions=st.lists(_action_st, max_size=5),
)
_https = st.from_regex(
    r"https://[a-z][a-z0-9-]{0,20}\.openral\.com(/[a-z]{1,8}){0,2}", fullmatch=True
)
_enrolment_st = st.builds(
    RobotEnrolment,
    v=st.just(1),
    robot=_id,
    aud=_https,
    org=st.text(max_size=12),
    iss=_https,
    jwks=_https,
)


def _round_trip(cls: Any, instance: Any) -> None:
    serialized = instance.model_dump_json(by_alias=True)
    assert cls.model_validate_json(serialized) == instance
    jsonschema.validate(json.loads(serialized), cls.model_json_schema())


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(_manifest_st)
def test_fuzz_robot_manifest(instance: RobotManifest) -> None:
    _round_trip(RobotManifest, instance)


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(_enrolment_st)
def test_fuzz_robot_enrolment(instance: RobotEnrolment) -> None:
    _round_trip(RobotEnrolment, instance)


def test_real_manifest_round_trips_with_the_wire_alias() -> None:
    manifest = build_robot_manifest(
        _robot("panda_mobile"),
        DeployRuntime(enable_slam=True),
        _installed(),
        bridge_on=True,
        write_controls=True,
        remote_control=False,
    )
    wire = json.loads(manifest.model_dump_json(by_alias=True))
    assert wire["schema"] == 1 and "schema_version" not in wire
    _round_trip(RobotManifest, manifest)


# ── the link code (Interface B) ───────────────────────────────────────────────

_LINK = {
    "v": 1,
    "robot": "k3x9",
    "aud": "https://r-k3x9.openral.com",
    "org": "acme",
    "iss": "https://www.openral.com/api/auth",
    "jwks": "https://www.openral.com/api/auth/jwks",
    "serial": "SO101-0007",
    "tunnel": "eyJhIjoiYWNjb3VudCJ9",
}


def _code(payload: dict[str, Any], *, pad: bool = False) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return raw if pad else raw.rstrip("=")


@pytest.mark.parametrize("pad", [False, True])
def test_link_code_parses(pad: bool) -> None:
    enrolment, tunnel = parse_robot_link_code(_code(_LINK, pad=pad))
    assert tunnel == _LINK["tunnel"]
    assert enrolment.model_dump() == {k: v for k, v in _LINK.items() if k != "tunnel"}


@pytest.mark.parametrize(
    "bad",
    [
        "not base64 !!",
        base64.urlsafe_b64encode(b"[1, 2]").decode(),
        _code({k: v for k, v in _LINK.items() if k != "tunnel"}),
        _code({**_LINK, "v": 2}),
        _code({**_LINK, "aud": "http://r-k3x9.openral.com"}),
        _code({**_LINK, "jwks": "http://www.openral.com/api/auth/jwks"}),
        _code({**_LINK, "extra": "field"}),
        _code({**_LINK, "serial": "has,comma"}),
    ],
)
def test_bad_link_codes_are_refused(bad: str) -> None:
    with pytest.raises(ROSConfigError):
        parse_robot_link_code(bad)
