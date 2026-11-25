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
from openral_cli.deploy_sim import resolve_launch_invocation
from openral_core import DeployScene, ScenePeer, SimScene
from openral_core.exceptions import ROSConfigError
from openral_core.scene_paths import (
    SCENE_PEERS_ENV,
    format_scene_peers,
    parse_scene_peers,
    require_peer_domains,
)
from pydantic import ValidationError

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


P = [
    ScenePeer(name="b", robot_id="openarm", domain=34),
    ScenePeer(name="c", robot_id="openarm", domain=38),
]


def test_no_peers_no_check() -> None:
    require_peer_domains([], None)


@pytest.mark.parametrize("own", [None, "", "34"])
def test_bad_own_domain(own: str | None) -> None:
    with pytest.raises(ROSConfigError, match="ROS_DOMAIN_ID"):
        require_peer_domains(P, own)


def test_peers_share_a_domain() -> None:
    with pytest.raises(ROSConfigError, match="34"):
        require_peer_domains([P[0], P[0].model_copy(update={"name": "d"})], "30")


def test_ok() -> None:
    require_peer_domains(P, "30")


def _resolve(**kw: Any) -> Any:
    repo = Path(__file__).resolve().parents[2]
    return resolve_launch_invocation(
        config=repo / "scenes/deploy/so101_box.yaml",
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        **kw,
    )


@pytest.mark.parametrize("hal_mode", ["sim", "real"])
def test_resolve_refuses_peers_without_an_exported_domain(
    monkeypatch: pytest.MonkeyPatch, hal_mode: str
) -> None:
    monkeypatch.setenv(SCENE_PEERS_ENV, format_scene_peers(P))
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    with pytest.raises(ROSConfigError, match="ROS_DOMAIN_ID"):
        _resolve(hal_mode=hal_mode)


def test_resolve_reports_malformed_peers_as_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SCENE_PEERS_ENV, "{}")
    with pytest.raises(ROSConfigError, match=SCENE_PEERS_ENV):
        _resolve()


def test_launch_refuses_peers_without_a_world_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _launch_module()
    monkeypatch.setenv(SCENE_PEERS_ENV, format_scene_peers(P))
    with pytest.raises(ROSConfigError, match="world"):
        mod._scene_peers(None)
    monkeypatch.delenv(SCENE_PEERS_ENV)
    assert mod._scene_peers(None) == []


@pytest.mark.parametrize("own", ["034", " 34 "])
def test_own_domain_is_compared_as_ros_reads_it(own: str) -> None:
    with pytest.raises(ROSConfigError, match="34"):
        require_peer_domains(P, own)


def test_own_domain_must_be_an_integer() -> None:
    with pytest.raises(ROSConfigError, match="ROS_DOMAIN_ID"):
        require_peer_domains(P, "abc")


def test_duplicate_peer_name() -> None:
    twin = P[1].model_copy(update={"name": "b"})
    with pytest.raises(ROSConfigError, match="twice"):
        require_peer_domains([P[0], twin], "30")


@pytest.mark.parametrize("model", [DeployScene, SimScene])
def test_robots_key_points_to_deploy_fleet(model: type[DeployScene]) -> None:
    with pytest.raises(ValidationError, match="openral deploy fleet"):
        model.model_validate({"scene": {"id": "x", "backend": "isaacsim"}, "robots": []})
