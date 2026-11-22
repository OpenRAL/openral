"""Each robot's place in the shared world, and what stopped it, reach ``/api/mission``.

openral-pro#99: a fleet console draws every robot of a scene with several robots in one
``world`` frame (``world_pose``, and a static ``world -> map`` TF per robot graph), and an
operator judges a reset from the stop's detail (``latch_detail``).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from openral_core import DeployScene
from openral_core.geometry import homogeneous_from_quat_xyz, quat_xyzw_to_yaw
from openral_core.scene_paths import SCENE_ROBOT_ENV
from openral_observability.dashboard import TelemetryStore
from openral_observability.dashboard.app import mission_view

from tests.unit.test_deploy_scene_robots import _aisle

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH = REPO_ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"

# The kernel's collision evidence for a carried carton pressed into a shelf.
_COLLISION = json.dumps(
    {
        "kind": "collision",
        "collision_kind": "world",
        "link_a": "attached:trk:2",
        "link_b_or_object": "voxel_464002",
        "horizon_step": -1,
        "min_distance_m": -0.0145,
        "joint_positions_rad": [0.0] * 6,
    }
)
_DETAIL = {
    "kind": "collision",
    "link": "attached:trk:2",
    "other": "voxel_464002",
    "depth_m": 0.0145,
}


@pytest.fixture(scope="module")
def launch_module() -> object:
    """The real launch file, loaded by path: it is not an importable package."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_world_pose", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_world_pose"] = module
    spec.loader.exec_module(module)
    return module


def _aisle_file(tmp_path: Path) -> Path:
    raw = DeployScene.model_validate(_aisle()).model_dump(mode="json", exclude_defaults=True)
    path = tmp_path / "aisle.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_a_robot_graph_places_its_map_in_the_shared_world(
    launch_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``world -> map`` is the robot's start: it maps its own goals back onto the world's."""
    path = _aisle_file(tmp_path)
    monkeypatch.setenv(SCENE_ROBOT_ENV, "north")
    north = DeployScene.from_yaml(str(path))
    pose = launch_module._scene_robot_world_pose(north)  # type: ignore[attr-defined]  # reason: module loaded by path
    assert pose == DeployScene.model_validate(_aisle()).robots[1].base_pose
    # The goal north reads in its map, through world <- map, is the scene's world goal.
    world_from_map = homogeneous_from_quat_xyz(pose.xyz, pose.quat_xyzw)
    goal = north.nav_goals[0].pose
    assert world_from_map @ np.array([*goal.xyz, 1.0]) == pytest.approx(
        [0.4, 1.1, 0.0, 1.0], abs=1e-9
    )
    node = launch_module._world_map_tf_node(pose, "map")  # type: ignore[attr-defined]  # reason: module loaded by path
    args = [str(a) for a in node._Node__arguments]  # reason: launch_ros keeps no public accessor
    values = dict(zip(args[::2], args[1::2], strict=True))
    assert (values["--frame-id"], values["--child-frame-id"]) == ("world", "map")
    assert [float(values[k]) for k in ("--x", "--y", "--z")] == pytest.approx(pose.xyz)
    assert [float(values[k]) for k in ("--qx", "--qy", "--qz", "--qw")] == pytest.approx(
        pose.quat_xyzw
    )
    assert quat_xyzw_to_yaw(*pose.quat_xyzw) == pytest.approx(-math.pi / 2)


def test_a_one_robot_graph_has_no_world_pose(
    launch_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``$OPENRAL_SCENE_ROBOT``: the parent scene, or a one-robot scene, has none."""
    monkeypatch.delenv(SCENE_ROBOT_ENV, raising=False)
    world_pose = launch_module._scene_robot_world_pose  # type: ignore[attr-defined]  # reason: module loaded by path
    assert world_pose(DeployScene.from_yaml(str(_aisle_file(tmp_path)))) is None
    one = DeployScene.from_yaml(str(REPO_ROOT / "scenes/deploy/openamrobot_v2_empty.yaml"))
    assert world_pose(one) is None
    assert world_pose(None) is None


def test_world_pose_reaches_the_mission_view_only_when_set() -> None:
    store = TelemetryStore()
    assert "world_pose" not in mission_view(store.snapshot())
    store.set_world_pose(-0.4, 2.4, -math.pi / 2)
    assert mission_view(store.snapshot())["world_pose"] == {
        "x": -0.4,
        "y": 2.4,
        "yaw": -math.pi / 2,
        "frame": "world",
    }


def _latch(store: TelemetryStore, latched: bool, reason: int = 10) -> None:
    store.set_safety_status(
        latched=latched,
        drop_reason=reason,
        drop_reason_label="kind_collision" if latched else "none",
        detail="",
        rskill_id="",
        trace_id="",
        stamp_unix=time.time(),
    )


def _stop(store: TelemetryStore, evidence: str = _COLLISION) -> None:
    store.record_safety_failure(
        kind_label="kind_collision",
        severity_label="severity_abort",
        evidence_json=evidence,
        rskill_id="hf://qualia/restock",
        trace_id="",
        stamp_unix=time.time(),
    )


@pytest.mark.parametrize("stop_first", [True, False], ids=["stop-then-latch", "latch-then-stop"])
def test_a_latched_collision_names_what_it_hit(stop_first: bool) -> None:
    """The trigger and the latch ride two topics: either order gives the same detail."""
    store = TelemetryStore()
    assert mission_view(store.snapshot())["latch_detail"] is None
    if stop_first:
        _stop(store)
        _latch(store, True)
    else:
        _latch(store, True)
        _stop(store)
    view = mission_view(store.snapshot())
    assert view["latch_detail"] == pytest.approx(_DETAIL)
    _latch(store, True)  # the 1 Hz re-publish keeps it
    assert mission_view(store.snapshot())["latch_detail"] == pytest.approx(_DETAIL)
    _latch(store, False)
    assert mission_view(store.snapshot())["latch_detail"] is None


def test_a_latch_without_a_stop_near_it_has_no_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator E-STOP long after an old collision does not borrow its detail."""
    clock = [1000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    store = TelemetryStore()
    _stop(store)
    clock[0] += 30.0
    _latch(store, True, reason=7)
    assert mission_view(store.snapshot())["latch_detail"] is None
    clock[0] += 30.0
    _stop(store, _COLLISION.replace("voxel_464002", "voxel_1"))  # far from the latch's start
    assert mission_view(store.snapshot())["latch_detail"] is None


def test_other_stops_name_their_joint_and_carry_no_depth() -> None:
    store = TelemetryStore()
    _latch(store, True, reason=1)
    force = '{"kind": "force", "joint_or_ee": "arm_joint_2", "measured_n": 9.0, "limit_n": 5.0}'
    _stop(store, force)
    assert mission_view(store.snapshot())["latch_detail"] == {
        "kind": "force",
        "link": "arm_joint_2",
        "other": "",
        "depth_m": None,
    }


def test_the_dashboard_refuses_a_malformed_world_pose() -> None:
    from openral_cli.main import app
    from typer.testing import CliRunner

    for bad in ("1,2", "a,b,c", "nan,0,0"):
        result = CliRunner().invoke(app, ["dashboard", "--world-pose", bad])
        assert result.exit_code == 2, (bad, result.output)
