"""Isaac scene: environment USD + spawn pose from the deploy scene.

A deploy scene puts any manifest robot into any Isaac stage with three fields —
``scene.assets_uri`` (environment USD), ``robot_id``, ``base_pose`` (spawn).
These pin the openral side of that contract without a GPU:

* ``_resolve_environment_usd`` / ``_spawn_pose`` validate the two fields;
* the shipped ``isaac_panda_mobile_warehouse.yaml`` goes through the real deploy
  loader (``build_sim_env_from_yaml``) and the real factory to a real sidecar
  subprocess (``tests/unit/fakes/isaac_sidecar_no_kit.py`` — the real
  ``tools/isaac_sidecar.py`` argv parsing + ZMQ serve loop, minus Kit), so the
  test sees exactly the argv Isaac would boot with.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from openral_core import Pose6D
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.isaac_sim import _resolve_environment_usd, _spawn_pose

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WAREHOUSE_YAML = _REPO_ROOT / "scenes" / "deploy" / "isaac_panda_mobile_warehouse.yaml"
_FAKE_SIDECAR = _REPO_ROOT / "tests" / "unit" / "fakes" / "isaac_sidecar_no_kit.py"


def _pose(quat_xyzw: tuple[float, float, float, float]) -> Pose6D:
    return Pose6D(xyz=(-4.8, 0.0, 0.0), quat_xyzw=quat_xyzw, frame_id="world")


# ── _spawn_pose ───────────────────────────────────────────────────────────────


def test_spawn_pose_defaults_to_the_origin() -> None:
    assert _spawn_pose(None) == (0.0, 0.0, 0.0, 0.0)


def test_spawn_pose_extracts_yaw() -> None:
    s = math.sqrt(0.5)
    assert _spawn_pose(_pose((0.0, 0.0, s, s))) == pytest.approx((-4.8, 0.0, 0.0, math.pi / 2))


def test_spawn_pose_accepts_an_unnormalised_yaw_quaternion() -> None:
    assert _spawn_pose(_pose((0.0, 0.0, 2.0, 2.0)))[3] == pytest.approx(math.pi / 2)


def test_spawn_pose_rejects_a_tilted_robot() -> None:
    s = math.sqrt(0.5)
    with pytest.raises(ROSConfigError, match="only a yaw"):
        _spawn_pose(_pose((s, 0.0, 0.0, s)))  # 90 deg roll


def test_spawn_pose_rejects_the_zero_quaternion() -> None:
    with pytest.raises(ROSConfigError, match="zero quaternion"):
        _spawn_pose(_pose((0.0, 0.0, 0.0, 0.0)))


# ── _resolve_environment_usd ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "uri",
    [
        "isaac:Isaac/Environments/Simple_Warehouse/warehouse.usd",
        "https://example.org/stage.usdc",
        "omniverse://localhost/Projects/stage.usd",
    ],
)
def test_remote_environment_passes_through(uri: str) -> None:
    assert _resolve_environment_usd(uri) == uri


def test_local_environment_resolves_to_an_absolute_path(tmp_path: Path) -> None:
    stage = tmp_path / "cell.usda"
    stage.write_text('#usda 1.0\n(\n    upAxis = "Z"\n)\n')
    assert _resolve_environment_usd(f"file://{stage}") == str(stage)
    assert _resolve_environment_usd(str(stage)) == str(stage)


def test_missing_local_environment_fails_before_boot(tmp_path: Path) -> None:
    with pytest.raises(ROSConfigError, match="no such file"):
        _resolve_environment_usd(str(tmp_path / "typo.usd"))


def test_non_usd_environment_is_rejected() -> None:
    with pytest.raises(ROSConfigError, match="not a USD file"):
        _resolve_environment_usd("scenes/arena.xml")


# ── shipped warehouse scene → real loader → real factory → sidecar argv ──────


def test_warehouse_deploy_scene_boots_the_sidecar_with_stage_and_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("zmq", reason="isaacsim group (pyzmq) not installed")
    pytest.importorskip("msgpack", reason="isaacsim group (msgpack) not installed")
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    argv_out = tmp_path / "argv.json"
    monkeypatch.setenv("OPENRAL_ISAAC_SIDECAR_PYTHON", sys.executable)
    monkeypatch.setenv("OPENRAL_ISAAC_SIDECAR_SCRIPT", str(_FAKE_SIDECAR))
    monkeypatch.setenv("OPENRAL_TEST_ISAAC_ARGV_OUT", str(argv_out))
    monkeypatch.setenv("OPENRAL_ISAAC_AUTO_SPAWN", "1")

    env, _seed = build_sim_env_from_yaml(str(_WAREHOUSE_YAML))
    try:
        argv = json.loads(argv_out.read_text())
        # The environment + spawn fields imply the manifest layout.
        assert argv[argv.index("--layout") + 1] == "manifest"
        assert argv[argv.index("--robot") + 1] == "panda_mobile"
        assert (
            argv[argv.index("--environment-usd") + 1]
            == "isaac:Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        )
        i = argv.index("--spawn-pose")
        assert [float(v) for v in argv[i + 1 : i + 5]] == pytest.approx(
            [-4.8, 0.0, 0.0, math.pi / 2], abs=1e-6
        )
        # panda_mobile's 11-D contract (7 arm + gripper + 3 base twist) came back
        # over the identity-checked ping.
        assert env.action_dim == 11
    finally:
        env.close()


def test_a_removed_layout_is_rejected_at_load() -> None:
    """The lift_cube / bowl_plate PoC layouts are gone (they never booted on
    Isaac Sim 6.1); a YAML still naming one fails validation, not mid-boot."""
    from openral_sim import SCENES

    with pytest.raises(ROSConfigError, match="layout"):
        SCENES.validate_options("isaac_sim", {"layout": "lift_cube"})


# ── URDF joint matching + grippers (real OpenArm manifest vs its real URDF) ───


def test_openarm_manifest_joints_resolve_onto_its_urdf() -> None:
    from openral_core import RobotDescription
    from openral_sim.backends.isaac_sim import (
        _gripper_spec,
        _match_urdf_joint,
        _parse_urdf_joints,
    )

    desc = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml"))
    urdf = _parse_urdf_joints(_REPO_ROOT / "robots" / "openarm" / "openarm.urdf")
    claimed: set[str] = set()
    resolved = {}
    for j in desc.joints:
        resolved[j.name] = _match_urdf_joint(j, urdf, claimed)
        claimed.add(resolved[j.name])
    # Manifest names drop the URDF's openarm_ prefix; the (parent, child) pair
    # resolves the arms, sim_joint_name the logical-link grippers.
    assert resolved["left_joint1"] == "openarm_left_joint1"
    assert resolved["right_joint7"] == "openarm_right_joint7"
    assert resolved["left_gripper"] == "openarm_left_finger_joint1"
    left = _gripper_spec(
        next(j for j in desc.joints if j.name == "left_gripper"),
        urdf["openarm_left_finger_joint1"],
        urdf,
    )
    # The URDF jaw spans [-1.571, 0]; the hardware (manifest) travel is 0.785.
    assert (left["closed"], left["open"]) == pytest.approx((0.0, -0.7854))
    assert (left["manifest_closed"], left["manifest_open"]) == pytest.approx((0.0, 0.7854))
    # The second finger mirrors the first through the URDF <mimic>.
    assert left["followers"] == [
        {"dof": "openarm_left_finger_joint2", "multiplier": -1.0, "offset": 0.0}
    ]
    right = _gripper_spec(
        next(j for j in desc.joints if j.name == "right_gripper"),
        urdf["openarm_right_finger_joint1"],
        urdf,
    )
    assert (right["manifest_closed"], right["manifest_open"]) == pytest.approx((0.0, -0.7854))


def test_ros_package_paths_prefer_a_sourced_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AMENT_PREFIX_PATH wins over the public-package fetch (no network here)."""
    from openral_sim.backends.isaac_sim import _ros_package_paths

    share = tmp_path / "share" / "openarm_description"
    share.mkdir(parents=True)
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(tmp_path))
    urdf = _REPO_ROOT / "robots" / "openarm" / "openarm.urdf"
    assert _ros_package_paths(urdf) == [{"name": "openarm_description", "path": str(share)}]


def test_unknown_ros_package_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from openral_sim.backends.isaac_sim import _ros_package_paths

    monkeypatch.setenv("AMENT_PREFIX_PATH", str(tmp_path))
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        _REPO_ROOT.joinpath("robots/openarm/openarm.urdf")
        .read_text()
        .replace("package://openarm_description/", "package://no_such_description/")
    )
    with pytest.raises(ROSConfigError, match="no_such_description"):
        _ros_package_paths(urdf)


# ── backend options + objects ─────────────────────────────────────────────────


def test_options_reject_unknown_keys() -> None:
    from openral_sim import SCENES

    with pytest.raises(ROSConfigError):
        SCENES.validate_options("isaac_sim", {"control_mode": "joint"})


def test_objects_resolve_their_usd_and_key_the_sidecar() -> None:
    from openral_sim.backends.isaac_sim import (
        IsaacSceneObject,
        _objects_json,
        _scene_default_port,
        _world_key,
    )

    box = IsaacSceneObject(
        usd="isaac:Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        name="cracker_box",
        xyz=(-0.45, 4.8, 0.4),
    )
    payload = json.loads(_objects_json([box]))
    assert payload == [
        {
            "usd": "isaac:Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            "name": "cracker_box",
            "xyz": [-0.45, 4.8, 0.4],
            "yaw": 0.0,
            "dynamic": True,
        }
    ]
    # Same stage + spawn, different props → a different sidecar.
    a = _scene_default_port("t", "r", "manifest", _world_key("isaac:e.usd", None))
    b = _scene_default_port(
        "t", "r", "manifest", _world_key("isaac:e.usd", None, json.dumps(payload))
    )
    assert a != b
    with pytest.raises(ROSConfigError, match=r"objects\[mesh\]\.usd"):
        _objects_json([IsaacSceneObject(usd="props/mesh.obj", name="mesh", xyz=(0, 0, 0))])
