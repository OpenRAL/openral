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
from typing import Any

import numpy as np
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
        passthrough=True,  # OpenArm declares write_mode: passthrough (radians)
    )
    # Passthrough writes the manifest's physical value unchanged on the URDF axis.
    assert (left["closed"], left["open"]) == pytest.approx((0.0, 0.7854))
    assert (left["manifest_closed"], left["manifest_open"]) == pytest.approx((0.0, 0.7854))
    # The second finger mirrors the first through the URDF <mimic>.
    assert left["followers"] == [
        {"dof": "openarm_left_finger_joint2", "multiplier": -1.0, "offset": 0.0}
    ]
    right = _gripper_spec(
        next(j for j in desc.joints if j.name == "right_gripper"),
        urdf["openarm_right_finger_joint1"],
        urdf,
        passthrough=True,
    )
    assert (right["manifest_closed"], right["manifest_open"]) == pytest.approx((0.0, -0.7854))
    assert (right["closed"], right["open"]) == pytest.approx((0.0, -0.7854))


@pytest.mark.parametrize("side", ["left", "right"])
def test_openarm_isaac_open_target_swings_each_finger_outward(side: str) -> None:
    """The Isaac open target opens the jaw: each finger swings away from the jaw centre.

    The fingers hang along the hand's -z; rotating a point below a finger's hinge by
    the scene's open target (the follower through its ``<mimic>``) must move it away
    from the centre plane (y = 0) on the hinge's own side. An open target on the wrong
    side of the URDF axis swings the fingers inward through each other, the way the
    left jaw crossed in the warehouse pick trials and pushed the brick out instead of
    squeezing it.
    """
    import xml.etree.ElementTree as ET

    from openral_core import RobotDescription
    from openral_sim.backends.isaac_sim import _gripper_spec, _parse_urdf_joints

    urdf_path = _REPO_ROOT / "robots" / "openarm" / "openarm.urdf"
    desc = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml"))
    urdf = _parse_urdf_joints(urdf_path)
    leader = f"openarm_{side}_finger_joint1"
    spec = _gripper_spec(
        next(j for j in desc.joints if j.name == f"{side}_gripper"),
        urdf[leader],
        urdf,
        passthrough=True,
    )
    angles = {leader: spec["open"]}
    for f in spec["followers"]:
        angles[f["dof"]] = f["multiplier"] * spec["open"] + f["offset"]
    assert len(angles) == 2
    # Kinematic <joint>s only: the <ros2_control> blocks reuse the names, without origins.
    root = ET.parse(urdf_path).getroot()
    joints = {el.attrib["name"]: el for el in root.iter("joint") if "type" in el.attrib}
    for name, q in angles.items():
        el = joints[name]
        origin = np.array([float(v) for v in el.find("origin").attrib["xyz"].split()])
        axis = np.array([float(v) for v in el.find("axis").attrib["xyz"].split()])
        assert abs(axis[0]) == pytest.approx(1.0), "OpenArm's jaw hinges about the hand's x"
        # Rotation by q about axis (+-x): a point 10 cm below the hinge moves in y by
        # -sin(q * axis_x) * (-0.10).
        tip_y = origin[1] + np.sin(q * axis[0]) * 0.10
        assert abs(tip_y) > abs(origin[1]) and np.sign(tip_y) == np.sign(origin[1]), (
            f"{name} at the open target {q:+.4f} swings inward (tip y {tip_y:+.4f}, "
            f"hinge y {origin[1]:+.4f})"
        )


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
            "roll": 0.0,
            "pitch": 0.0,
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


def test_a_normalised_gripper_spans_the_whole_urdf_travel() -> None:
    """SO-100 declares a NORMALISED [0, 1] jaw on a revolute URDF jaw of
    (-0.2, 2.0) rad: 1.0 must open it fully (2.0 rad), 0.0 close it at the zero
    pose — not read [0, 1] as radians (which opened it to 0.8, ~45%)."""
    from openral_core import RobotDescription
    from openral_sim.backends.isaac_sim import _build_robot_spec

    desc = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "so100_follower" / "robot.yaml"))
    (gripper,) = _build_robot_spec(desc, "so100_follower")["grippers"]
    assert (gripper["closed"], gripper["open"]) == pytest.approx((0.0, 2.0))
    assert (gripper["manifest_closed"], gripper["manifest_open"]) == (0.0, 1.0)


def test_spawn_args_never_look_like_option_flags() -> None:
    """repr(-1e-05) == '-1e-05', which argparse (Python <= 3.13) takes for a flag."""
    import argparse

    from openral_sim.backends.isaac_sim import _spawn_pose

    spawn = _spawn_pose(
        Pose6D(xyz=(-1e-5, 2.0, 0.0), quat_xyzw=(0.0, 0.0, -1e-6, 1.0), frame_id="world")
    )
    args = [f"{v:.9f}" for v in spawn]
    p = argparse.ArgumentParser()
    p.add_argument("--spawn-pose", type=float, nargs=4)
    assert p.parse_args(["--spawn-pose", *args]).spawn_pose == pytest.approx(spawn, abs=1e-8)


def test_object_names_must_be_unique_and_not_reserved() -> None:
    from openral_sim import SCENES

    box = {"usd": "isaac:Isaac/Props/x.usd", "xyz": [0, 0, 0]}
    for names, match in ((["box", "box"], "duplicate"), (["robot"], "reserved")):
        with pytest.raises(ROSConfigError, match=match):
            SCENES.validate_options("isaac_sim", {"objects": [{**box, "name": n} for n in names]})


def test_relative_assets_resolve_against_the_repo_root_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HAL node's working directory is arbitrary; a repo-relative USD path
    must resolve the same from anywhere."""
    rel = Path("scenes") / "_tmp_test_stage.usda"
    stage = _REPO_ROOT / rel
    stage.write_text("#usda 1.0\n")
    try:
        monkeypatch.chdir(tmp_path)
        assert _resolve_environment_usd(str(rel)) == str(stage.resolve())
    finally:
        stage.unlink()


def _openarm_env_cfg() -> Any:
    """The shipped OpenArm warehouse scene as the SimEnvironment the backend builds from."""
    from openral_core import SimEnvironment, VLASpec
    from openral_hal.sim_bringup import _load_scene_for_hal

    scene = _load_scene_for_hal(str(_REPO_ROOT / "scenes/deploy/isaac_openarm_warehouse.yaml"))
    return SimEnvironment(
        robot_id="openarm",
        scene=scene.scene,
        task=scene.task,
        vla=VLASpec(id="smolvla", weights_uri="hf://lerobot/smolvla_base"),
        base_pose=scene.base_pose,
        n_episodes=1,
    )


def test_camera_mounts_validate_and_key_the_sidecar() -> None:
    """A camera mount takes exactly one orientation, names a real manifest camera,
    and a scene that only re-mounts a camera gets its own sidecar."""
    from openral_sim.backends.isaac_sim import (
        IsaacCameraMount,
        IsaacSimOptions,
        _camera_mounts_json,
        _scene_default_port,
        _world_key,
        _write_robot_spec,
    )
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="exactly one"):
        IsaacCameraMount(xyz=(0, 0, 0))
    with pytest.raises(ValidationError, match="axes: world"):
        IsaacCameraMount(xyz=(0, 0, 0), look_at=(1, 0, 0), axes="usd")
    opts = IsaacSimOptions(
        camera_mounts={
            "head_zed": {"link": "openarm_base", "xyz": (0, 0, 0.2), "look_at": (1, 0, 0)}
        }
    )
    mounts = _camera_mounts_json(opts.camera_mounts)
    assert json.loads(mounts)["head_zed"]["link"] == "openarm_base"
    a = _scene_default_port("t", "r", "manifest", _world_key("isaac:e.usd", None))
    b = _scene_default_port("t", "r", "manifest", _world_key("isaac:e.usd", None, mounts))
    assert a != b

    env_cfg = _openarm_env_cfg()
    path, _desc = _write_robot_spec(env_cfg, opts.camera_mounts)
    try:
        spec = json.loads(Path(path).read_text())
    finally:
        Path(path).unlink()
    assert spec["camera_mounts"]["head_zed"]["xyz"] == [0.0, 0.0, 0.2]
    with pytest.raises(ROSConfigError, match="not 'openarm' camera sensors"):
        _write_robot_spec(env_cfg, {"lidar": IsaacCameraMount(xyz=(0, 0, 0), look_at=(1, 0, 0))})


def test_openarm_spec_carries_its_base_frame_offset() -> None:
    """OpenArm's base_frame (openarm_base) is no URDF link: it sits 0.698 m above the
    URDF root. The sidecar must publish depth in that frame, so the spec carries the
    offset (the clouds were stamped openarm_base but expressed 0.698 m too high)."""
    from openral_sim.backends.isaac_sim import _build_robot_spec
    from openral_sim.registry import ROBOTS

    spec = _build_robot_spec(ROBOTS.get("openarm")(), "openarm")
    assert np.asarray(spec["root_to_base"])[:3, 3] == pytest.approx([0.0, 0.0, 0.698])
    franka = _build_robot_spec(ROBOTS.get("franka_panda")(), "franka_panda")
    assert np.asarray(franka["root_to_base"]) == pytest.approx(np.eye(4))


def _openarm_spec(monkeypatch: pytest.MonkeyPatch, unit: str | None) -> dict[str, Any]:
    """The robot spec the shipped OpenArm Isaac scene hands its sidecar, for ``unit``."""
    from openral_core import ROBOT_UNIT_ENV
    from openral_sim.backends.isaac_sim import IsaacSimOptions, _write_robot_spec

    if unit is None:
        monkeypatch.delenv(ROBOT_UNIT_ENV, raising=False)
    else:
        monkeypatch.setenv(ROBOT_UNIT_ENV, unit)
    env_cfg = _openarm_env_cfg()
    opts = IsaacSimOptions(**env_cfg.scene.backend_options)
    path, _desc = _write_robot_spec(env_cfg, opts.camera_mounts)
    try:
        return dict(json.loads(Path(path).read_text()))
    finally:
        Path(path).unlink()


def test_openarm_wrist_cameras_ride_the_jaw_like_the_mujoco_twin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unmounted in the scene, each wrist camera rides the jaw where the MuJoCo twin has it.

    Unmounted, they got the scene's generic base-relative viewpoints: on the Isaac host
    wrist_right rendered all-black (every pixel 0) and wrist_left stared at a shelf
    upright, while the policy expects the view down its own jaw. The default mount is
    the compiled `camera_wrist_*` pose of the real MuJoCo twin, in the body that is the
    same-named URDF link, at the twin's field of view (its 66 deg fovy at the manifest's
    4:3 raster) unless a unit calibrated the intrinsics.
    """
    mujoco = pytest.importorskip("mujoco")
    from openral_hal._openarm_v2_assets import ensure_openarm_v2_mjcf

    try:
        mjcf = ensure_openarm_v2_mjcf()
    except ROSConfigError as exc:
        pytest.skip(f"OpenArm MJCF unavailable: {exc}")
    model = mujoco.MjModel.from_xml_path(mjcf)
    urdf = (_REPO_ROOT / "robots" / "openarm" / "openarm.urdf").read_text()
    nominal = _openarm_spec(monkeypatch, None)["camera_mounts"]
    thor = _openarm_spec(monkeypatch, "thor")["camera_mounts"]
    for side in ("left", "right"):
        mount = nominal[f"wrist_{side}"]
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"camera_wrist_{side}")
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.cam_bodyid[cam])
        assert mount["link"] == body == f"openarm_{side}_ee_base_link"
        assert f'<link name="{body}">' in urdf
        assert mount["axes"] == "usd"  # MuJoCo cameras look down -z, +y up
        assert mount["xyz"] == pytest.approx(model.cam_pos[cam], abs=1e-5)
        assert abs(float(np.dot(mount["quat_wxyz"], model.cam_quat[cam]))) == pytest.approx(
            1.0, abs=1e-5
        )
        fovy = math.radians(float(model.cam_fovy[cam]))
        assert mount["hfov_deg"] == pytest.approx(
            math.degrees(2 * math.atan(math.tan(fovy / 2) * 640 / 480)), abs=0.1
        )
        # The Thor unit calibrated these intrinsics: its field of view, not the twin's.
        assert thor[f"wrist_{side}"]["hfov_deg"] is None
        assert thor[f"wrist_{side}"]["xyz"] == mount["xyz"]


def test_openarm_head_and_top_render_from_the_units_zed_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`top` is head_zed's device: both ride head_zed's mount, the unit's when one is set."""
    from openral_core import load_robot_unit

    nominal = _openarm_spec(monkeypatch, None)
    thor = _openarm_spec(monkeypatch, "thor")
    measured = {
        s.name: s for s in load_robot_unit(_REPO_ROOT / "robots/openarm/robot.yaml", "thor").sensors
    }["head_zed"].static_transform_xyz_rpy
    assert measured is not None
    for spec, xyz in ((nominal, [0.0, 0.0, 0.2]), (thor, list(measured[:3]))):
        mounts = spec["camera_mounts"]
        assert mounts["top"] == mounts["head_zed"]
        assert mounts["head_zed"]["link"] == "openarm_base"
        assert mounts["head_zed"]["axes"] == "world"  # zed_camera_link: x forward, z up
        assert mounts["head_zed"]["xyz"] == pytest.approx(xyz)
    # The unit's calibrated wrist raster reaches the sidecar, which renders it.
    by_name = {s["name"]: s for s in thor["sensors"]}
    assert (
        by_name["wrist_left"]["intrinsics"]["width"],
        by_name["top"]["intrinsics"]["width"],
    ) == (
        960,
        672,
    )


def test_a_camera_the_manifest_cannot_place_is_announced(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A robot-framed camera left on a generic viewpoint is said out loud, once, at load."""
    from openral_core import ROBOT_UNIT_ENV
    from openral_sim.backends.isaac_sim import _write_robot_spec

    monkeypatch.delenv(ROBOT_UNIT_ENV, raising=False)
    env_cfg = _openarm_env_cfg()
    path, _desc = _write_robot_spec(env_cfg, None)
    Path(path).unlink()
    assert "have no mount" not in capsys.readouterr().out  # the manifest places all four
    # panda_mobile declares robot-framed cameras with no mount, no static transform and
    # no MJCF: each is named.
    path, _desc = _write_robot_spec(env_cfg.model_copy(update={"robot_id": "panda_mobile"}), None)
    Path(path).unlink()
    assert (
        "cameras ['front_depth', 'head', 'shoulder_left', 'shoulder_right', 'wrist'] have no mount"
        in capsys.readouterr().out
    )


def test_initial_joint_positions_reach_the_sidecar_within_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scene's start pose is checked against the manifest, then handed to the sidecar.

    The OpenArm restock policy starts with both jaws at its training opening, 0.6 rad
    (the right jaw on the mirrored axis); the arms keep the URDF's zero (hanging).
    """
    from openral_core import ROBOT_UNIT_ENV
    from openral_sim.backends.isaac_sim import IsaacSimOptions, _write_robot_spec

    monkeypatch.delenv(ROBOT_UNIT_ENV, raising=False)
    env_cfg = _openarm_env_cfg()
    opening = {"left_gripper": 0.6, "right_gripper": -0.6}
    opts = IsaacSimOptions(initial_joint_positions=opening)
    path, _desc = _write_robot_spec(env_cfg, None, opts.initial_joint_positions)
    try:
        spec = json.loads(Path(path).read_text())
    finally:
        Path(path).unlink()
    assert spec["initial_joint_positions"] == opening
    names = [j["name"] for j in spec["joints"] if j.get("role") in ("arm", "gripper")]
    assert {"left_gripper", "right_gripper"} <= set(names)
    for bad in ({"left_gripper": -0.6}, {"no_such_joint": 0.0}):
        with pytest.raises(ROSConfigError, match="initial_joint_positions"):
            _write_robot_spec(env_cfg, None, bad)
