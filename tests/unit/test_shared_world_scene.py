"""``SharedWorldScene`` (issue #666): the fixture, its validators and the launcher's resolution.

Real fixtures only: ``scenes/worlds/so101_handover_table.yaml`` and the scenes and
manifests it names.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from openral_cli.deploy_world import robot_deploy_argv, world_robot_yamls
from openral_core import RobotDescription, SharedWorldScene
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_hal import build_hal
from openral_hal.peer_voxels import PeerBox, occupancy_grid
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


def test_a_robot_that_composes_its_own_arena_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest's `scene_defaults.composition` would compose an arena the world ignores."""
    import yaml

    robot_dir = tmp_path / "robots" / "so101_own_arena"
    robot_dir.mkdir(parents=True)
    manifest = yaml.safe_load((REPO_ROOT / "robots" / "so101_follower" / "robot.yaml").read_text())
    manifest["scene_defaults"] = {
        "composition": {"composer": "openral_sim.backends.so101_box._assets:compose_so101_box_mjcf"}
    }
    (robot_dir / "robot.yaml").write_text(yaml.safe_dump(manifest))
    scene = tmp_path / "scene.yaml"
    scene.write_text(
        "scene:\n  id: so101_own_arena_table\n  backend: mujoco\nrobot_id: so101_own_arena\n"
    )
    world = SharedWorldScene.model_validate(_world(scene=str(scene)))
    monkeypatch.setenv("OPENRAL_ROBOTS_DIR", str(tmp_path / "robots"))
    with pytest.raises(ROSConfigError, match=r"scene_defaults\.composition"):
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


def test_a_shared_world_robot_launch_feeds_its_kernel_the_peers() -> None:
    """The HAL binding `deploy world` passes also turns on the launch's peer producer."""
    from openral_cli.deploy_sim import resolve_launch_invocation

    config = REPO_ROOT / "scenes" / "deploy" / "so101_shared_table.yaml"
    invocation = resolve_launch_invocation(
        config=config,
        robot_override=None,
        dashboard_port=4336,
        reset_to_pose_service=None,
        hal_param_overrides={"shared_world": "/tmp/w.sock", "shared_world_robot": "a"},
    )
    argv = invocation.argv_template
    assert "shared_world:=/tmp/w.sock" in argv
    assert "shared_world_robot:=a" in argv
    assert "enable_octomap:=false" in argv  # the peers are the only voxel producer
    plain = resolve_launch_invocation(
        config=config, robot_override=None, dashboard_port=4336, reset_to_pose_service=None
    )
    assert not any(a.startswith("shared_world") for a in plain.argv_template)


def test_a_real_deploy_cannot_join_a_shared_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """The binding is simulation-only: refused while resolving, not later in the launch."""
    from openral_cli.deploy_sim import resolve_launch_invocation

    monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "bench_laptop")  # the SO-101's committed unit

    with pytest.raises(ROSConfigError, match="simulation binding"):
        resolve_launch_invocation(
            config=REPO_ROOT / "scenes" / "deploy" / "so101_shared_table.yaml",
            robot_override=None,
            dashboard_port=4336,
            reset_to_pose_service=None,
            hal_mode="real",
            hal_param_overrides={"shared_world": "/tmp/w.sock", "shared_world_robot": "a"},
        )


def test_a_non_finite_peer_read_is_refused_not_rasterized() -> None:
    """A NaN in a world read raises (the producer withholds that grid), never a hole."""
    import mujoco as mj
    from openral_hal.peer_voxels import swept_peer_boxes

    model = mj.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="b_base"><joint name="b_j" type="hinge"/>'
        '<geom type="box" size=".1 .1 .1"/></body></worldbody>'
        '<actuator><position joint="b_j" kp="10"/></actuator></mujoco>'
    )
    state = {"qpos": [float("nan")], "qvel": [0.0], "ctrl": [0.0]}
    with pytest.raises(ROSRuntimeError, match="not finite"):
        swept_peer_boxes(model, mj.MjData(model), state, np.array([0]), [(0, 0)], 0)


def test_the_peer_grid_covers_every_cell_a_box_touches() -> None:
    """Conservative rasterization: a cell any part of a rotated box enters is occupied."""
    angle = math.radians(30.0)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    box = PeerBox(np.array([0.12, -0.07, 0.05]), rotation, np.array([0.04, 0.01, 0.02]))
    resolution = 0.015
    origin, (sx, sy, sz), cells = occupancy_grid([box], (0.0, 0.0, 0.1), 0.3, resolution)
    assert sx == sy == sz == math.floor(2 * 0.3 / resolution + 0.5) + 1  # the kernel's reserve
    occupied = cells.reshape(sz, sy, sx)
    rng = np.random.default_rng(0)
    points = (rng.uniform(-1.0, 1.0, (5000, 3)) * box.half_extents) @ rotation.T + box.centre
    index = np.floor((points - origin) / resolution).astype(int)
    assert occupied[index[:, 2], index[:, 1], index[:, 0]].all()
    # ... and it stays local: nothing beyond the box's own reach plus a cell diagonal.
    zz, yy, xx = np.nonzero(occupied)
    centres = origin + (np.stack([xx, yy, zz], axis=1) + 0.5) * resolution
    reach = np.linalg.norm(box.half_extents) + resolution * math.sqrt(3.0)
    assert (np.linalg.norm(centres - box.centre, axis=1) <= reach + 1e-9).all()
    # A box beyond the ball's grid adds nothing.
    far = PeerBox(np.array([5.0, 0.0, 0.0]), np.eye(3), np.array([0.1, 0.1, 0.1]))
    assert occupancy_grid([far], (0.0, 0.0, 0.1), 0.3, resolution)[2].sum() == 0
