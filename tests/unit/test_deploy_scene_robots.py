"""A deploy scene that lists several robots (``DeployScene.robots``, openral-pro#99)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from openral_core import DeployScene, NavGoal, PlaceRegion, PlaceZone, Pose6D, SceneRobot
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import scene_for_robot, yaw_to_quat_xyzw
from pydantic import ValidationError


def _at(x: float, y: float, yaw: float = 0.0, frame: str = "world") -> Pose6D:
    return Pose6D(xyz=(x, y, 0.0), quat_xyzw=yaw_to_quat_xyzw(yaw), frame_id=frame)


def _aisle(**extra: Any) -> dict[str, Any]:
    """Two AMRs at opposite ends of a north-south aisle, a goal and a shelf slot in `world`."""
    return {
        "scene": {"id": "aisle", "backend": "isaacsim"},
        "robots": [
            {"name": "south", "robot_id": "openamrobot_v2", "base_pose": _at(0.4, -1.1)},
            {
                "name": "north",
                "robot_id": "openamrobot_v2",
                "base_pose": _at(-0.4, 2.4, -math.pi / 2),
            },
        ],
        "nav_goals": [NavGoal(name="s5", pose=_at(0.4, 1.1))],
        "place_zones": [
            PlaceZone(
                name="slot_5",
                region=PlaceRegion(
                    frame_id="world",
                    half_extents=(0.1, 0.1, 0.05),
                    pose=_at(1.0, 1.1, math.pi / 2),
                ),
            )
        ],
        **extra,
    }


def test_each_robot_gets_its_own_scene_with_the_shared_poses_in_its_map() -> None:
    scene = DeployScene.model_validate(_aisle())
    south, north = scene_for_robot(scene, "south"), scene_for_robot(scene, "north")
    assert (south.robot_id, south.robots, south.base_pose) == (
        "openamrobot_v2",
        [],
        scene.robots[0].base_pose,
    )
    # south faces +x from (0.4, -1.1): the goal is 2.2 m to its left.
    assert south.nav_goals[0].pose.frame_id == "map"
    assert south.nav_goals[0].pose.xyz == pytest.approx((0.0, 2.2, 0.0), abs=1e-9)
    # north faces -y from (-0.4, 2.4): the goal is 1.3 m ahead, 0.8 m to its left.
    assert north.nav_goals[0].pose.xyz == pytest.approx((1.3, 0.8, 0.0), abs=1e-9)
    assert north.nav_goals[0].pose.quat_xyzw == pytest.approx(yaw_to_quat_xyzw(math.pi / 2))
    zone = north.place_zones[0].region
    assert (zone.frame_id, zone.pose.frame_id) == ("map", "map")
    assert zone.pose.xyz == pytest.approx((1.3, 1.4, 0.0), abs=1e-9)  # 1.3 m ahead, 1.4 m left
    assert zone.pose.quat_xyzw == pytest.approx(yaw_to_quat_xyzw(math.pi), abs=1e-9)
    # Each robot's scene validates as a plain one-robot scene.
    DeployScene.model_validate(north.model_dump())
    assert scene_for_robot(scene, "north", map_frame="odom").nav_goals[0].pose.frame_id == "odom"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"robots": _aisle()["robots"][:1]}, "two or more"),
        ({"robots": [_aisle()["robots"][0]] * 2}, "unique"),
        ({"robot_id": "openamrobot_v2"}, "one-robot fields"),
        ({"nav_goals": [NavGoal(name="s5", pose=_at(0, 0, frame="map"))]}, "shared 'world'"),
    ],
)
def test_a_several_robot_scene_is_refused_when_ambiguous(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        DeployScene.model_validate(_aisle(**change))


def test_a_robot_starts_in_the_shared_frame_and_has_a_plain_name() -> None:
    with pytest.raises(ValidationError, match="world"):
        SceneRobot(name="a", robot_id="x", base_pose=_at(0, 0, frame="map"))
    with pytest.raises(ValidationError):
        SceneRobot(name="North-1", robot_id="x", base_pose=_at(0, 0))


def test_an_unknown_robot_name_is_refused() -> None:
    with pytest.raises(ROSConfigError, match="no robot 'east'"):
        scene_for_robot(DeployScene.model_validate(_aisle()), "east")


def test_a_one_robot_scene_is_unchanged() -> None:
    scene = DeployScene.model_validate(
        {
            "scene": {"id": "aisle", "backend": "isaacsim"},
            "robot_id": "openamrobot_v2",
            "nav_goals": [NavGoal(name="s6", pose=_at(0, 0, frame="map"))],
        }
    )
    assert scene.robots == []
    assert scene.nav_goals[0].pose.frame_id == "map"


def test_sim_run_scenes_stay_one_robot() -> None:
    from openral_core import SimScene

    task = {"id": "restock", "scene_id": "aisle", "instruction": "restock"}
    with pytest.raises(ValidationError, match="runs one robot"):
        SimScene.model_validate(_aisle(task=task))


def test_a_robot_graph_reads_its_own_scene_in_every_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``$OPENRAL_SCENE_ROBOT`` (set by ``deploy sim`` per graph): one robot, everywhere."""
    from openral_core import load_scene_strict
    from openral_core.scene_paths import SCENE_ROBOT_ENV
    from openral_hal.sim_bringup import _load_scene_for_hal

    raw = DeployScene.model_validate(_aisle()).model_dump(mode="json", exclude_defaults=True)
    path = tmp_path / "aisle.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert len(DeployScene.from_yaml(str(path)).robots) == 2  # the parent sees every robot
    monkeypatch.setenv(SCENE_ROBOT_ENV, "north")
    north = DeployScene.model_validate(_aisle()).robots[1]
    for scene in (DeployScene.from_yaml(str(path)), load_scene_strict(str(path), DeployScene)):
        assert scene.robots == []
        assert scene.base_pose == north.base_pose
        assert scene.nav_goals[0].pose.frame_id == "map"
        assert scene.nav_goals[0].pose.xyz == pytest.approx((1.3, 0.8, 0.0), abs=1e-9)
    hal_scene = _load_scene_for_hal(str(path))  # the HAL keeps the scene, robot and start pose
    assert (hal_scene.robot_id, hal_scene.base_pose) == (north.robot_id, north.base_pose)


def test_a_robot_with_nav_goals_starts_on_the_ground_plane() -> None:
    """A raised or tilted start would put each goal re-expressed in its map off the ground."""
    robots = _aisle()["robots"]
    raised = {
        **robots[0],
        "base_pose": Pose6D(xyz=(0, 0, 1), quat_xyzw=(0, 0, 0, 1), frame_id="world"),
    }
    with pytest.raises(ValidationError, match=r"ground plane.*\['south'\]"):
        DeployScene.model_validate(_aisle(robots=[raised, robots[1]]))
    # Without nav goals a raised start (an arm on a table) stays valid.
    DeployScene.model_validate(_aisle(robots=[raised, robots[1]], nav_goals=[]))


def test_a_scaled_quaternion_start_is_refused() -> None:
    """(1e-6, 0, 0, 1e-6) passes a raw-component planar test yet normalizes to a 90 deg roll."""
    with pytest.raises(ValidationError, match="not a unit quaternion"):
        SceneRobot(
            name="a",
            robot_id="openamrobot_v2",
            base_pose=Pose6D(xyz=(0, 0, 0), quat_xyzw=(1e-6, 0, 0, 1e-6), frame_id="world"),
        )


def test_a_non_finite_start_is_refused() -> None:
    """A NaN norm slips past an ``abs(norm - 1) > tol`` test: finiteness is checked first."""
    with pytest.raises(ValidationError, match="not finite"):
        SceneRobot(
            name="a",
            robot_id="openamrobot_v2",
            base_pose=Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, float("nan"), 1), frame_id="world"),
        )


def test_an_attachment_naming_no_robot_of_the_scene_is_refused() -> None:
    """A misspelt robot would match no robot's attachment filter: the tote would ride nothing."""
    tote = {"prim": "Tote", "link": "lift_carriage_link", "robot": "sotuh"}
    scene = {"id": "aisle", "backend": "isaacsim", "backend_options": {"link_attachments": [tote]}}
    with pytest.raises(ValidationError, match=r"do(es)? not list: \['sotuh'\]"):
        DeployScene.model_validate(_aisle(scene=scene))
    tote["robot"] = "south"
    DeployScene.model_validate(_aisle(scene=scene))


def test_a_robot_graph_reads_its_scene_in_its_own_map_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openral_core.scene_paths import SCENE_ROBOT_ENV, SCENE_ROBOT_MAP_FRAME_ENV

    raw = DeployScene.model_validate(_aisle()).model_dump(mode="json", exclude_defaults=True)
    path = tmp_path / "aisle.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv(SCENE_ROBOT_ENV, "north")
    monkeypatch.setenv(SCENE_ROBOT_MAP_FRAME_ENV, "north_map")
    scene = DeployScene.from_yaml(str(path))
    assert scene.nav_goals[0].pose.frame_id == "north_map"
    assert scene.place_zones[0].region.frame_id == "north_map"


# ── `openral deploy sim` of a several-robot scene (openral_cli.deploy_robots) ──────────


def test_robot_domains_skip_the_reserved_ones() -> None:
    from openral_cli._dds_scope import RESERVED_DOMAINS
    from openral_cli.deploy_robots import _MAX_ROBOTS, robot_slots

    robots = [
        SceneRobot(name=f"r{i}", robot_id="openamrobot_v2", base_pose=_at(i, 0)) for i in range(4)
    ]
    scene = DeployScene.model_validate({**_aisle(), "robots": robots})
    domains = [int(s["domain"]) for s in robot_slots(scene)]
    assert domains == [30, 34, 38, 46]  # 42 is tools/_demo_env.sh's
    assert not set(domains) & RESERVED_DOMAINS and max(domains) < 50
    too_many = [*robots, SceneRobot(name="r9", robot_id="openamrobot_v2", base_pose=_at(9, 0))]
    assert len(too_many) > _MAX_ROBOTS
    with pytest.raises(ROSConfigError, match="at most 4"):
        robot_slots(DeployScene.model_validate({**_aisle(), "robots": too_many}))


def test_each_linked_robot_gets_its_own_remote_port_behind_one_router(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every robot's tunnel reaches 4319 (openral.com sets the ingress): the router's table
    sends each linked robot's host to that robot's own listener; an unlinked robot has none."""
    from openral_cli.deploy_robots import robot_enrolment_path, robot_slots, tunnel_routes
    from openral_core import RobotEnrolment
    from openral_observability.dashboard.tunnel import TUNNEL_PORT

    monkeypatch.setenv("HOME", str(tmp_path))
    robots = [
        SceneRobot(name=n, robot_id="openamrobot_v2", base_pose=_at(i, 0))
        for i, n in enumerate("abc")
    ]
    scene = DeployScene.model_validate({**_aisle(), "robots": robots})
    slots = robot_slots(scene)
    ports = [int(s["remote_port"]) for s in slots]
    assert len(set(ports)) == 3 and TUNNEL_PORT not in ports
    assert not set(ports) & {int(s["dashboard_port"]) for s in slots}
    for name, robot in (("a", "a1a1a"), ("c", "c3c3c")):  # b is not linked
        path = robot_enrolment_path(scene.scene.id, name)
        path.parent.mkdir(parents=True)
        path.write_text(
            RobotEnrolment(
                v=1,
                robot=robot,
                aud=f"https://r-{robot}.openral.com",
                org="acme",
                iss="https://www.openral.com/api/auth",
                jwks="https://www.openral.com/api/auth/jwks",
            ).model_dump_json()
        )
    assert tunnel_routes(scene.scene.id, slots) == {
        "r-a1a1a.openral.com": ports[0],
        "r-c3c3c.openral.com": ports[2],
    }
    robot_enrolment_path(scene.scene.id, "c").write_text("{not an enrolment")
    assert list(tunnel_routes(scene.scene.id, slots)) == ["r-a1a1a.openral.com"]


def test_each_robot_gets_its_own_memory_bundle(tmp_path: Path) -> None:
    from openral_cli.deploy_robots import robot_argv, robot_memory_dirs

    dirs = robot_memory_dirs(str(tmp_path), ["south", "north"])
    assert dirs == {"south": tmp_path / "south", "north": tmp_path / "north"}
    assert all(d.is_dir() for d in dirs.values())
    slot: dict[str, int | str] = {
        "name": "north",
        "domain": 34,
        "dashboard_port": 4331,
        "foxglove_port": 8831,
    }
    argv = robot_argv(
        ["deploy", "sim", "--memory-dir", str(tmp_path), "-c", "s.yaml"],
        Path("s.yaml"),
        slot,
        dirs["north"],
    )
    assert argv[argv.index("--memory-dir") + 1] == str(tmp_path / "north")
    assert argv.count("--memory-dir") == 1
    with pytest.raises(ROSConfigError, match="not an existing directory"):
        robot_memory_dirs(str(tmp_path / "missing"), ["south"])


def test_the_launcher_stops_when_any_robot_graph_ends(tmp_path: Path) -> None:
    """Even one whose entry is written: the world would start without it."""
    import subprocess
    import sys

    from openral_cli.deploy_robots import wait_for_entries

    (tmp_path / "south.json").write_text("{}", encoding="utf-8")
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    running = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        ended.wait(timeout=10)
        with pytest.raises(ROSConfigError, match=r"\['south'\] ended"):
            wait_for_entries(
                tmp_path, ["south", "north"], {"south": ended, "north": running}, timeout_s=5.0
            )
    finally:
        running.kill()
        running.wait()


def test_a_dry_run_prints_each_robots_command_and_starts_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from openral_cli.deploy_robots import run_scene_robots

    scene = DeployScene.model_validate(_aisle())
    argv = ["deploy", "sim", "--config", "aisle.yaml", "--dry-run", "--memory-dir", str(tmp_path)]
    assert run_scene_robots(scene, Path("aisle.yaml"), argv, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "OPENRAL_SCENE_ROBOT=south" in out and "OPENRAL_SCENE_ROBOT=north" in out
    assert "OPENRAL_SCENE_ROBOT_MAP_FRAME=map" in out
    assert '"north", 34' in out and '"south", 30' in out  # each robot's peers
    assert str(tmp_path / "north") in out
    assert list(tmp_path.iterdir()) == []  # nothing written, nothing started


def test_each_robot_is_told_every_other_robot_and_never_itself() -> None:
    """ADR-0137: each robot's peer producer reads the others' domains, from their manifests."""
    from openral_cli.deploy_robots import robot_slots, scene_peers
    from openral_core.scene_paths import parse_scene_peers

    robots = [
        SceneRobot(name=n, robot_id="openamrobot_v2", base_pose=_at(i, 0))
        for i, n in enumerate("abc")
    ]
    scene = DeployScene.model_validate({**_aisle(), "robots": robots})
    peers = {n: parse_scene_peers(v) for n, v in scene_peers(scene, robot_slots(scene)).items()}
    assert [(n, d) for n, d, _ in peers["a"]] == [("b", 34), ("c", 38)]
    assert [(n, d) for n, d, _ in peers["c"]] == [("a", 30), ("b", 34)]
    assert all(Path(m).name == "robot.yaml" and Path(m).is_file() for _, _, m in peers["b"])


def test_a_shared_scene_attachment_names_its_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    from openral_core.scene_paths import SCENE_ROBOT_ENV
    from openral_sim.backends.isaac_sim import (
        SHARED_SIDECAR_PORT_ENV,
        IsaacSimOptions,
        _shared_sidecar_setup,
    )

    monkeypatch.setenv(SCENE_ROBOT_ENV, "north")
    monkeypatch.setenv(SHARED_SIDECAR_PORT_ENV, "5999")
    tote = {"prim": "Tote", "link": "lift_carriage_link"}
    opts = IsaacSimOptions(
        link_attachments=[{**tote, "robot": "south"}, {**tote, "prim": "Tote_01", "robot": "north"}]
    )
    _robot, _port, mine = _shared_sidecar_setup(opts)
    assert [a.prim for a in mine.link_attachments] == ["Tote_01"]
    with pytest.raises(ROSConfigError, match=r"\['Tote'\] name no robot"):
        _shared_sidecar_setup(IsaacSimOptions(link_attachments=[tote]))
