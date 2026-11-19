"""``SharedWorldScene`` (issue #666): the fixture, its validators and the launcher's resolution.

Real fixtures only: ``scenes/worlds/so101_handover_table.yaml`` and the scenes and
manifests it names.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_cli.deploy_world import robot_deploy_argv, world_robot_yamls
from openral_core import RobotDescription, SharedWorldScene
from openral_core.exceptions import ROSConfigError
from openral_hal import build_hal
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_YAML = REPO_ROOT / "scenes" / "worlds" / "so101_handover_table.yaml"


def _world(**robot_b: object) -> dict[str, object]:
    data = SharedWorldScene.from_yaml(str(WORLD_YAML)).model_dump()
    data["robots"][1].update(robot_b)
    return data


def test_fixture_loads_and_resolves_both_robots() -> None:
    world = SharedWorldScene.from_yaml(str(WORLD_YAML))
    assert [(r.name, r.ros_domain_id) for r in world.robots] == [("a", 36), ("b", 40)]
    assert (WORLD_YAML.parent / world.arena_mjcf).is_file()
    yamls = world_robot_yamls(world, WORLD_YAML)
    assert {Path(p).parent.name for p in yamls.values()} == {"so101_follower"}
    argv = robot_deploy_argv(world.robots[1], WORLD_YAML, "/tmp/w.sock", ["--no-foxglove"])
    assert argv[argv.index("--config") + 1] == str(
        REPO_ROOT / "scenes" / "deploy" / "so101_shared_table.yaml"
    )
    assert argv[-5:] == [
        "--hal",
        'shared_world="/tmp/w.sock"',
        "--hal",
        'shared_world_robot="b"',
        "--no-foxglove",
    ]
    # `deploy sim` JSON-parses `--hal` values: the quotes keep a name such as `true` a string.
    from openral_cli.deploy_sim import _parse_hal_overrides

    assert _parse_hal_overrides(argv[-4:-1:2]) == {
        "shared_world": "/tmp/w.sock",
        "shared_world_robot": "b",
    }


@pytest.mark.parametrize(
    ("robot_b", "message"),
    [
        ({"name": "a"}, "names must be unique"),
        ({"dashboard_port": 4336}, "ports must all differ"),
        ({"foxglove_port": 4340}, "ports must all differ"),
        ({"ros_domain_id": 37}, "at least 2 apart"),  # adjacent domains see each other
        ({"ros_domain_id": 36}, "at least 2 apart"),
        ({"name": "robot0"}, "robosuite body prefixes"),
        ({"base_yaw_deg": float("nan")}, "finite"),  # robot0_* bodies trip render heuristics
    ],
)
def test_robots_that_would_collide_are_refused(robot_b: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SharedWorldScene.model_validate(_world(**robot_b))


@pytest.mark.parametrize("timeout", [float("inf"), 8.0, 0.0])
def test_the_barrier_wait_is_bounded(timeout: float) -> None:
    """The waiting robot's HAL is blocked meanwhile; its graph's action deadlines are 8 s."""
    data = _world()
    data["barrier_timeout_s"] = timeout
    with pytest.raises(ValidationError, match="barrier_timeout_s"):
        SharedWorldScene.model_validate(data)


def test_a_single_robot_is_not_a_shared_world() -> None:
    data = _world()
    data["robots"] = data["robots"][:1]  # type: ignore[index]
    with pytest.raises(ValidationError):
        SharedWorldScene.model_validate(data)


def test_a_robot_scene_that_owns_an_arena_is_refused() -> None:
    """``so101_box`` composes its own arena; in a shared world the world owns it."""
    world = SharedWorldScene.model_validate(_world(scene="../deploy/so101_box.yaml"))
    with pytest.raises(ROSConfigError, match="composition"):
        world_robot_yamls(world, WORLD_YAML)


def test_build_hal_refuses_shared_world_off_the_bare_sim_twin() -> None:
    transport: dict[str, object] = {"shared_world": "/tmp/w.sock", "shared_world_robot": "a"}
    so101 = RobotDescription.from_yaml(str(REPO_ROOT / "robots" / "so101_follower" / "robot.yaml"))
    with pytest.raises(ROSConfigError, match="shared_world"):
        build_hal(so101, mode="real", transport=transport)
    with pytest.raises(ROSConfigError, match="shared_world_robot"):  # the name is required
        build_hal(so101, mode="sim", transport={"shared_world": "/tmp/w.sock"})
    h1 = RobotDescription.from_yaml(str(REPO_ROOT / "robots" / "h1" / "robot.yaml"))
    assert h1.hal.sim is not None  # a class-based sim HAL: not a manifest-derived twin
    with pytest.raises(ROSConfigError, match="shared_world"):
        build_hal(h1, mode="sim", transport=transport)
