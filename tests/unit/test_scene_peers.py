"""``ScenePeer``, the typed ``OPENRAL_SCENE_PEERS`` entry (openral#765)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openral_core import ScenePeer
from openral_core.scene_paths import format_scene_peers, parse_scene_peers

LAUNCH = (
    Path(__file__).resolve().parents[2] / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"
)

_peers = st.builds(
    ScenePeer,
    name=st.from_regex(r"[a-z][a-z0-9]{0,23}", fullmatch=True),
    robot_id=st.text(min_size=1, max_size=20),
    domain=st.integers(0, 232),
)


def test_round_trip() -> None:
    peers = [ScenePeer(name="b", robot_id="openarm", domain=34)]
    assert parse_scene_peers(format_scene_peers(peers)) == peers


@given(st.lists(_peers, max_size=5))
def test_hypothesis_round_trip_and_schema(peers: list[ScenePeer]) -> None:
    assert parse_scene_peers(format_scene_peers(peers)) == peers
    for p in peers:
        assert ScenePeer.model_validate_json(p.model_dump_json()) == p
        jsonschema.validate(json.loads(p.model_dump_json()), ScenePeer.model_json_schema())


def test_empty() -> None:
    assert parse_scene_peers(None) == [] and parse_scene_peers("") == []


@pytest.mark.parametrize(
    "bad",
    [
        "{}",
        "not json",
        '[["b", 34, "/x"]]',
        '[{"name": "B", "robot_id": "x", "domain": 1}]',
        '[{"name": "b", "robot_id": "", "domain": 1}]',
        '[{"name": "b", "robot_id": "x", "domain": 233}]',
        '[{"name": "b", "robot_id": "x", "domain": 1, "extra": 0}]',
    ],
)
def test_bad_value_names_the_env(bad: str) -> None:
    with pytest.raises(ValueError, match="OPENRAL_SCENE_PEERS"):
        parse_scene_peers(bad)


def _launch_module() -> Any:
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_scene_peers", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_scene_peers"] = module
    spec.loader.exec_module(module)
    return module


def test_peer_params_keys() -> None:
    mod = _launch_module()
    p = mod._peer_params([ScenePeer(name="b", robot_id="openarm", domain=34)])
    assert p["peer_names"] == ["b"] and p["peer_domains"] == [34]
    assert p["peer_b.base_frame"] == "openarm_base"


def test_peer_params_unknown_robot_id() -> None:
    mod = _launch_module()
    with pytest.raises(Exception, match="nosuchrobot"):
        mod._peer_params([ScenePeer(name="b", robot_id="nosuchrobot", domain=34)])
