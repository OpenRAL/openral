"""Unit tests for the robot-agnostic Isaac scene marshalling.

Two halves, both GPU-free:

* the **openral side** — ``_build_robot_spec`` serialises the *real*
  ``franka_panda`` ``RobotDescription`` (no placeholder manifest, CLAUDE.md
  §1.11) into the JSON the py3.11 sidecar reads: URDF path resolved to a file,
  actuated joints in manifest order, an 8-D JOINT_POSITION action contract;
* the **sidecar side** — ``map_dof_to_manifest`` maps a full Isaac articulation
  DOF vector back to the manifest joint order, including the two-finger →
  one-gripper collapse, against the exact joint names the Panda URDF emits;
  and ``resolve_beam_range`` walks one lidar beam past the robot's own body.

The sidecar scene module (``tools/isaac_manifest_scene.py``) imports cleanly
without a live Kit app — the heavy Isaac imports live inside ``build()`` — so we
import ``map_dof_to_manifest`` directly by putting ``tools/`` on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from openral_core import RobotDescription, apply_sensor_overlays, resolve_sensor_overlays
from openral_sim.backends.isaac_sim import _build_robot_spec, _sensor_dict


def test_thor_brown_calibration_survives_sidecar_serialization() -> None:
    path = _repo_root() / "robots/openarm/robot.yaml"
    robot = RobotDescription.from_yaml(path)
    sensors = apply_sensor_overlays(
        robot.sensors, resolve_sensor_overlays(path, "thor", required=True)
    )
    for sensor in sensors:
        if sensor.name not in ("wrist_left", "wrist_right"):
            continue
        assert sensor.intrinsics is not None
        assert _sensor_dict(sensor)["intrinsics"] == sensor.intrinsics.model_dump()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "robots").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("repo root not found")


@pytest.fixture(scope="module")
def franka() -> RobotDescription:
    return RobotDescription.from_yaml(str(_repo_root() / "robots" / "franka_panda" / "robot.yaml"))


@pytest.fixture(scope="module")
def panda_mobile() -> RobotDescription:
    return RobotDescription.from_yaml(str(_repo_root() / "robots" / "panda_mobile" / "robot.yaml"))


# ── openral side: _build_robot_spec ───────────────────────────────────────────


def test_build_robot_spec_franka_shape(franka: RobotDescription) -> None:
    spec = _build_robot_spec(franka, "franka_panda")

    # URDF resolved to a real on-disk file (python:robot_descriptions:... form).
    assert Path(spec["urdf_path"]).is_file()
    assert spec["urdf_path"].endswith(".urdf")

    # Fixed-base arm: no planar base_joints → root pinned.
    assert spec["fix_base"] is True
    assert spec["base_joints"] is None

    # 7 arm joints + 1 gripper, in manifest order; base joints excluded (none).
    names = [j["name"] for j in spec["joints"]]
    assert names == [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_gripper",
    ]
    roles = {j["name"]: j["role"] for j in spec["joints"]}
    assert roles["panda_joint1"] == "arm"
    assert roles["panda_gripper"] == "gripper"


def test_build_robot_spec_franka_action_contract(franka: RobotDescription) -> None:
    spec = _build_robot_spec(franka, "franka_panda")
    action = spec["action"]
    assert action["control_mode"] == "joint_position"
    assert action["dim"] == 8  # 7 arm + 1 gripper
    (gripper,) = spec["grippers"]
    # The manifest's MuJoCo sim_joint_name (finger_joint1) is not a URDF joint;
    # the gripper resolves to the one non-mimic finger below panda_hand.
    assert gripper["leader"] == "panda_finger_joint1"
    assert (gripper["closed"], gripper["open"]) == pytest.approx((0.0, 0.04))
    assert gripper["followers"] == [
        {"dof": "panda_finger_joint2", "multiplier": 1.0, "offset": 0.0}
    ]


def test_build_robot_spec_sensors_serialised(franka: RobotDescription) -> None:
    spec = _build_robot_spec(franka, "franka_panda")
    rgb = [s for s in spec["sensors"] if s["modality"] == "rgb"]
    assert rgb, "franka manifest declares RGB cameras"
    cam = rgb[0]
    assert cam["intrinsics"] is not None
    assert set(cam["intrinsics"]) == {
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
        "distortion_model",
        "distortion_coeffs",
    }


def test_build_robot_spec_panda_mobile_base(panda_mobile: RobotDescription) -> None:
    spec = _build_robot_spec(panda_mobile, "panda_mobile")

    # All non-fixed joints in manifest order: 3 base + 7 arm + 1 gripper = 11.
    names = [j["name"] for j in spec["joints"]]
    roles = {j["name"]: j["role"] for j in spec["joints"]}
    assert names[:3] == ["base_x", "base_y", "base_yaw"]
    assert all(roles[b] == "base" for b in ("base_x", "base_y", "base_yaw"))
    assert roles["panda_joint1"] == "arm"
    assert roles["panda_gripper"] == "gripper"
    assert len(names) == 11

    # Action contract: 7 arm + 1 gripper + 3 base-twist channels.
    assert spec["action"]["dim"] == 11
    assert spec["action"]["has_base"] is True
    assert spec["base_joints"] == ["base_x", "base_y", "base_yaw"]

    # The arm is always pinned (kinematic base teleports the pinned root).
    assert spec["fix_base"] is True

    # panda_mobile declares a NORMALISED [0, 1] gripper width: the URDF's 0.04 m
    # finger travel drives the joint (never 1.0 m), and /joint_states maps back
    # onto the manifest's [0, 1].
    (gripper,) = spec["grippers"]
    assert gripper["open"] == pytest.approx(0.04)
    assert (gripper["manifest_closed"], gripper["manifest_open"]) == (0.0, 1.0)
    # Arm joints keep their URDF names (the manifest's sim_joint_name is the
    # robosuite MJCF's robot0_joint1, which the URDF does not have).
    urdf_names = {j["name"]: j["urdf_name"] for j in spec["joints"]}
    assert urdf_names["panda_joint1"] == "panda_joint1"
    assert urdf_names["base_x"] is None


def test_build_robot_spec_panda_mobile_has_depth_and_lidar(panda_mobile: RobotDescription) -> None:
    spec = _build_robot_spec(panda_mobile, "panda_mobile")
    modalities = {s["modality"] for s in spec["sensors"]}
    assert "depth" in modalities  # front_depth
    assert "lidar_2d" in modalities  # base_scan


# ── sidecar side: map_dof_to_manifest ─────────────────────────────────────────


@pytest.fixture(scope="module")
def _manifest_scene_mod() -> object:
    tools = str(_repo_root() / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import isaac_manifest_scene

    return isaac_manifest_scene


_PANDA_DOFS = [f"panda_joint{i}" for i in range(1, 8)] + [
    "panda_finger_joint1",
    "panda_finger_joint2",
]


def test_map_dof_to_manifest_franka(_manifest_scene_mod: object, franka: RobotDescription) -> None:
    map_dof_to_manifest = _manifest_scene_mod.map_dof_to_manifest  # type: ignore[attr-defined]
    spec = _build_robot_spec(franka, "franka_panda")

    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.02, 0.02], dtype=np.float32)
    mapped = map_dof_to_manifest(
        values,
        dof_index={n: i for i, n in enumerate(_PANDA_DOFS)},
        manifest_joints=spec["joints"],
        grippers=spec["grippers"],
    )

    assert mapped.shape == (8,)
    # Arm joints map straight through by URDF name.
    np.testing.assert_allclose(mapped[:7], values[:7], rtol=0, atol=1e-6)
    # Gripper: the leader finger's 0.02 m of 0.04 m travel = half of the
    # manifest's normalised [0, 1] width.
    assert mapped[7] == pytest.approx(0.5)
    # Velocities scale the same way, without the offset.
    rates = map_dof_to_manifest(
        values,
        dof_index={n: i for i, n in enumerate(_PANDA_DOFS)},
        manifest_joints=spec["joints"],
        grippers=spec["grippers"],
        rates=True,
    )
    assert rates[7] == pytest.approx(0.5)


def test_map_dof_to_manifest_base_joints_from_pose(
    _manifest_scene_mod: object, panda_mobile: RobotDescription
) -> None:
    map_dof_to_manifest = _manifest_scene_mod.map_dof_to_manifest  # type: ignore[attr-defined]
    spec = _build_robot_spec(panda_mobile, "panda_mobile")

    # panda_mobile order: 3 base + 7 arm + 1 gripper. Base joints are NOT URDF
    # DOFs — they come from the kinematic base pose (x, y, yaw).
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.04, 0.04], dtype=np.float32)
    mapped = map_dof_to_manifest(
        values,
        dof_index={n: i for i, n in enumerate(_PANDA_DOFS)},
        manifest_joints=spec["joints"],
        grippers=spec["grippers"],
        base_values=[1.5, -0.5, 0.785],  # x, y, yaw
        base_joints=spec["base_joints"],
    )

    assert mapped.shape == (11,)
    np.testing.assert_allclose(mapped[:3], [1.5, -0.5, 0.785], rtol=0, atol=1e-6)
    np.testing.assert_allclose(mapped[3:10], values[:7], rtol=0, atol=1e-6)
    assert mapped[10] == pytest.approx(1.0)  # fully open = the manifest's 1.0


def test_map_dof_to_manifest_unresolved_joint_is_zero(_manifest_scene_mod: object) -> None:
    map_dof_to_manifest = _manifest_scene_mod.map_dof_to_manifest  # type: ignore[attr-defined]
    # A manifest joint absent from the articulation → 0.0, never an index error.
    mapped = map_dof_to_manifest(
        np.array([1.0], dtype=np.float32),
        dof_index={"panda_joint1": 0},
        manifest_joints=[{"name": "ghost_joint", "role": "arm", "urdf_name": "ghost"}],
        grippers=[],
    )
    assert mapped.shape == (1,)
    assert mapped[0] == 0.0


def test_strip_urdf_mimics_keeps_every_joint(_manifest_scene_mod: object) -> None:
    """OpenArm's real URDF: mimics gone, joints and package:// meshes untouched."""
    import xml.etree.ElementTree as ET

    strip = _manifest_scene_mod.strip_urdf_mimics  # type: ignore[attr-defined]
    urdf = _repo_root() / "robots" / "openarm" / "openarm.urdf"
    source = urdf.read_text()
    out = ET.fromstring(strip(source, str(urdf.parent)))
    assert not list(out.iter("mimic"))
    assert len(list(out.iter("joint"))) == len(list(ET.fromstring(source).iter("joint")))
    assert all(
        m.attrib["filename"].startswith("package://openarm_description/") for m in out.iter("mesh")
    )


# ── sidecar side: resolve_beam_range ──────────────────────────────────────────
#
# The PhysX scene query is a parameter of `resolve_beam_range`'s real signature
# (Isaac's `raycast_closest(start, dir, span)`), so these drive it with plain
# closures describing a world — no patching, no stub object standing in for a
# collaborator (CLAUDE.md §1.11).


def _world(*bodies: tuple[float, str]) -> object:
    """A `raycast_closest` over bodies at fixed +x distances from the origin.

    Each body is ``(x_from_origin_m, prim_path)``. The ray is always +x here, so
    a hit is the nearest body ahead of ``start`` within ``span``.
    """
    ordered = sorted(bodies)

    def raycast_closest(
        start: tuple[float, float, float],
        _direction: tuple[float, float, float],
        span: float,
    ) -> dict[str, object] | None:
        for x, prim in ordered:
            distance = x - start[0]
            if 0.0 <= distance <= span:
                return {"hit": True, "distance": distance, "rigidBody": prim}
        return None

    return raycast_closest


_BEAM = {
    "robot_prim": "/panda",
    "origin_xy": (0.0, 0.0),
    "angle_rad": 0.0,
    "z": 0.30,
    "range_max_m": 12.0,
}


def test_beam_recasts_past_the_robots_own_chassis(_manifest_scene_mod: object) -> None:
    """Regression for #194: a self-hit must not read "clear all the way out".

    With `range_min_m` at the sensor minimum the ray starts INSIDE the chassis,
    so the robot is the first thing every beam hits. The old code reported
    `range_max_m` for such a beam — 12 m of empty space where a wall stands.
    """
    resolve_beam_range = _manifest_scene_mod.resolve_beam_range  # type: ignore[attr-defined]

    # Chassis at 0.20 m (inside the 0.43 m circumscribed radius), wall at 3 m.
    got = resolve_beam_range(
        _world((0.20, "/panda/base_link"), (3.0, "/World/wall")),
        range_min_m=0.05,
        **_BEAM,
    )
    assert got == pytest.approx(3.0, abs=1e-3), (
        "a beam that starts inside the chassis must be re-cast past the robot and "
        "report the obstacle behind it, not range_max"
    )


def test_beam_reports_the_nearest_real_obstacle_not_the_robot_behind_it(
    _manifest_scene_mod: object,
) -> None:
    """Identity, not distance: a near WORLD hit is kept even inside the chassis radius."""
    resolve_beam_range = _manifest_scene_mod.resolve_beam_range  # type: ignore[attr-defined]

    got = resolve_beam_range(
        _world((0.25, "/World/toe_kick"), (3.0, "/World/wall")),
        range_min_m=0.05,
        **_BEAM,
    )
    assert got == pytest.approx(0.25, abs=1e-3), (
        "0.25 m is inside the chassis radius but it is not the robot — the old "
        "0.55 m radial cutoff is exactly what deleted returns like this one"
    )


def test_beam_below_range_min_is_not_reported(_manifest_scene_mod: object) -> None:
    """The sensor's own floor still applies: nothing is invented closer than it."""
    resolve_beam_range = _manifest_scene_mod.resolve_beam_range  # type: ignore[attr-defined]

    got = resolve_beam_range(_world((0.02, "/World/dust")), range_min_m=0.05, **_BEAM)
    assert got == pytest.approx(12.0), "a return inside range_min is not a measurement"


def test_beam_gives_up_rather_than_looping_on_the_robot(_manifest_scene_mod: object) -> None:
    """A beam that never leaves the robot reads range_max — bounded, not hung."""
    resolve_beam_range = _manifest_scene_mod.resolve_beam_range  # type: ignore[attr-defined]

    def all_robot(
        start: tuple[float, float, float],
        _direction: tuple[float, float, float],
        span: float,
    ) -> dict[str, object]:
        del span
        return {"hit": True, "distance": 0.01, "rigidBody": "/panda/link_" + str(start[0])}

    assert resolve_beam_range(all_robot, range_min_m=0.05, **_BEAM) == pytest.approx(12.0)


def test_beam_misses_read_range_max(_manifest_scene_mod: object) -> None:
    """An empty world is an honest max-range fan, never NaN or a fabricated hit."""
    resolve_beam_range = _manifest_scene_mod.resolve_beam_range  # type: ignore[attr-defined]

    assert resolve_beam_range(_world(), range_min_m=0.05, **_BEAM) == pytest.approx(12.0)


def test_beam_self_skip_follows_the_imported_robot_prim(_manifest_scene_mod: object) -> None:
    """Self-hits are keyed on the robot's own prim, not a hardcoded ``/panda``.

    An environment USD can hold a prim whose path merely starts with the robot's
    name (``/panda_shelf``): that is the world, and must be reported.
    """
    resolve_beam_range = _manifest_scene_mod.resolve_beam_range  # type: ignore[attr-defined]
    beam = {**_BEAM, "robot_prim": "/openarm"}

    got = resolve_beam_range(
        _world((0.10, "/openarm/base_link"), (2.0, "/World/wall")), range_min_m=0.05, **beam
    )
    assert got == pytest.approx(2.0, abs=1e-3)
    got = resolve_beam_range(
        _world((0.10, "/openarm_shelf/rack"), (2.0, "/World/wall")), range_min_m=0.05, **beam
    )
    assert got == pytest.approx(0.10, abs=1e-3)


# ── sidecar side: compose_planar (spawn ∘ odom) ───────────────────────────────


def test_compose_planar_identity_spawn_is_odom(_manifest_scene_mod: object) -> None:
    compose_planar = _manifest_scene_mod.compose_planar  # type: ignore[attr-defined]
    assert compose_planar((0.0, 0.0, 0.0, 0.0), [1.5, -0.5, 0.3]) == pytest.approx(
        (1.5, -0.5, 0.0, 0.3)
    )


def test_compose_planar_odom_moves_in_the_spawn_heading(_manifest_scene_mod: object) -> None:
    """Driving 2 m forward from a spawn facing +y (the warehouse aisle) moves along +y."""
    compose_planar = _manifest_scene_mod.compose_planar  # type: ignore[attr-defined]
    x, y, z, yaw = compose_planar((-4.8, 0.0, 0.0, np.pi / 2), [2.0, 0.0, 0.0])
    assert (x, y, z, yaw) == pytest.approx((-4.8, 2.0, 0.0, np.pi / 2))
    # A strafe left (+y in the base frame) from that heading moves toward -x.
    x, y, _, _ = compose_planar((-4.8, 0.0, 0.0, np.pi / 2), [0.0, 1.0, 0.0])
    assert (x, y) == pytest.approx((-5.8, 0.0))


def test_compose_planar_keeps_the_spawn_height(_manifest_scene_mod: object) -> None:
    compose_planar = _manifest_scene_mod.compose_planar  # type: ignore[attr-defined]
    assert compose_planar((0.0, 0.0, 1.2, 0.0), [0.4, 0.0, 0.0])[2] == pytest.approx(1.2)


def test_points_in_frame_drops_misses_and_expresses_in_the_frame(
    _manifest_scene_mod: object,
) -> None:
    """Depth pixels that hit nothing deproject to inf/NaN; they must not reach octomap.
    Kept points land in the frame given by its world pose (yaw + offset)."""
    mod = _manifest_scene_mod
    pose = np.eye(4)
    pose[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]  # yaw 90 deg
    pose[:3, 3] = (1.0, 2.0, 0.7)
    pts = np.array([[1.0, 3.0, 0.0], [np.inf, 0.0, 0.0], [np.nan, 1.0, 1.0]], dtype=np.float32)
    out = mod.points_in_frame(pts, pose)  # type: ignore[attr-defined]
    assert out == pytest.approx(np.array([[1.0, 0.0, -0.7]]))


def test_look_at_aims_the_camera_x_axis(_manifest_scene_mod: object) -> None:
    mod = _manifest_scene_mod
    rot = mod.look_at_matrix(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))  # type: ignore[attr-defined]
    assert rot[:, 0] == pytest.approx(np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0))
    assert rot[:, 1] == pytest.approx([0.0, 1.0, 0.0])
    assert np.linalg.det(rot) == pytest.approx(1.0)


def test_camera_fov_comes_from_the_mount_else_the_manifest(_manifest_scene_mod: object) -> None:
    mod = _manifest_scene_mod
    k = {"width": 640, "fx": 320.0}
    assert mod.camera_hfov_deg({"intrinsics": k}) == pytest.approx(90.0)  # type: ignore[attr-defined]
    assert mod.camera_hfov_deg({"mount": {"hfov_deg": 70.0}, "intrinsics": k}) == 70.0  # type: ignore[attr-defined]
    assert mod.camera_hfov_deg({}) is None  # type: ignore[attr-defined]


def test_object_rpy_is_the_urdf_order(_manifest_scene_mod: object) -> None:
    """A scene object's roll/pitch/yaw compose as ``Rz·Ry·Rx``: a YCB box authored
    lying down stands on end with a quarter-turn roll, then turns by its yaw."""
    rpy_quat = _manifest_scene_mod._rpy_quat  # type: ignore[attr-defined]  # reason: module loaded off sys.path
    pose = _manifest_scene_mod.pose_matrix((0.1, 0.2, -0.08), rpy_quat(np.pi / 2, 0.0, np.pi / 2))  # type: ignore[attr-defined]  # reason: module loaded off sys.path
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(pose[:3, :3], rz @ rx, atol=1e-12)
    np.testing.assert_allclose(pose[:3, 3], [0.1, 0.2, -0.08])
    np.testing.assert_allclose(rpy_quat(0.0, 0.0, 0.4), _manifest_scene_mod._yaw_quat(0.4))  # type: ignore[attr-defined]  # reason: module loaded off sys.path


# ── finger friction (Isaac i58/i59) ───────────────────────────────────────────


def _robot_stage() -> object:
    """A real in-memory USD stage: two finger joints driving links with collision geometry,
    an arm link with its own, and a finger-named visual that no joint drives."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for path in (
        "/openarm/hand/ee_link1",
        "/openarm/hand/ee_link2",
        "/openarm/link7",
    ):
        UsdGeom.Xform.Define(stage, path)
        UsdGeom.Cube.Define(stage, f"{path}/collision")
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(f"{path}/collision"))
    UsdGeom.Cube.Define(stage, "/openarm/hand/finger_visual")
    for name, child in (("finger_joint1", "ee_link1"), ("finger_joint2", "ee_link2")):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/openarm/joints/{name}")
        joint.GetBody1Rel().SetTargets([f"/openarm/hand/{child}"])
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/openarm/joints/joint7")
    joint.GetBody1Rel().SetTargets(["/openarm/link7"])
    return stage


def test_finger_friction_binds_the_links_the_finger_joints_drive(
    _manifest_scene_mod: object,
) -> None:
    pytest.importorskip("pxr")
    from pxr import UsdPhysics, UsdShade

    stage = _robot_stage()
    result = _manifest_scene_mod.apply_finger_friction(  # type: ignore[attr-defined]
        stage, "/openarm", ["finger_joint1", "finger_joint2"], 0.9
    )
    assert sorted(result.bound) == ["/openarm/hand/ee_link1", "/openarm/hand/ee_link2"]
    assert result.missing_joints == []
    assert result.displaced == []
    material = UsdShade.Material.Get(stage, "/World/Physics_Materials/finger_pad")
    physics = UsdPhysics.MaterialAPI(material.GetPrim())
    assert physics.GetStaticFrictionAttr().Get() == pytest.approx(0.9)
    assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(0.9)
    for collider in ("/openarm/hand/ee_link1/collision", "/openarm/hand/ee_link2/collision"):
        resolved = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(collider)).ComputeBoundMaterial(
            "physics"
        )[0]
        assert resolved.GetPath() == material.GetPath(), "a finger collider inherits the pad"
    for untouched in ("/openarm/link7/collision", "/openarm/hand/finger_visual"):
        resolved = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(untouched)).ComputeBoundMaterial(
            "physics"
        )[0]
        assert not resolved, f"{untouched} is not driven by a finger joint"


def test_finger_friction_without_a_finger_joint_binds_nothing(
    _manifest_scene_mod: object,
) -> None:
    pytest.importorskip("pxr")
    stage = _robot_stage()
    result = _manifest_scene_mod.apply_finger_friction(stage, "/openarm", ["no_such_joint"])  # type: ignore[attr-defined]
    assert result.bound == []
    assert result.missing_joints == ["no_such_joint"]
    assert result.combine_max is False
    assert not stage.GetPrimAtPath("/World/Physics_Materials/finger_pad")


def _pad_for(stage: object, prim: str) -> object:
    from pxr import UsdShade

    return UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim)).ComputeBoundMaterial(  # type: ignore[attr-defined]
        "physics"
    )[0]


def test_finger_friction_partial_match_reports_the_missing_joint(
    _manifest_scene_mod: object,
) -> None:
    pytest.importorskip("pxr")
    stage = _robot_stage()
    result = _manifest_scene_mod.apply_finger_friction(  # type: ignore[attr-defined]
        stage, "/openarm", ["finger_joint1", "finger_joint_dropped"]
    )
    assert result.missing_joints == ["finger_joint_dropped"]
    assert result.bound == ["/openarm/hand/ee_link1"]
    assert _pad_for(stage, "/openarm/hand/ee_link1/collision")
    assert not _pad_for(stage, "/openarm/hand/ee_link2/collision"), "the other finger is unbound"


def test_finger_friction_forces_the_pad_over_a_descendant_material(
    _manifest_scene_mod: object,
) -> None:
    pytest.importorskip("pxr")
    from pxr import UsdPhysics, UsdShade

    stage = _robot_stage()
    own = UsdShade.Material.Define(stage, "/World/Physics_Materials/slippery")
    UsdPhysics.MaterialAPI.Apply(own.GetPrim()).CreateStaticFrictionAttr(0.1)
    collider = "/openarm/hand/ee_link1/collision"
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(collider)).Bind(
        own, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    result = _manifest_scene_mod.apply_finger_friction(  # type: ignore[attr-defined]
        stage, "/openarm", ["finger_joint1", "finger_joint2"]
    )
    assert result.displaced == [collider]
    assert _pad_for(stage, collider).GetPath() == "/World/Physics_Materials/finger_pad", (  # type: ignore[attr-defined]
        "the pad, not the descendant's material, resolves on the collider"
    )


def test_finger_friction_reports_whether_max_combine_was_applied(
    _manifest_scene_mod: object,
) -> None:
    pytest.importorskip("pxr")
    try:
        from pxr import PhysxSchema  # noqa: F401
    except ImportError:
        has_physx = False
    else:
        has_physx = True
    result = _manifest_scene_mod.apply_finger_friction(  # type: ignore[attr-defined]
        _robot_stage(), "/openarm", ["finger_joint1"]
    )
    assert result.combine_max is has_physx


@pytest.mark.parametrize("friction", [0.0, -1.0, float("nan"), float("inf")])
def test_finger_friction_refuses_a_bad_coefficient(
    _manifest_scene_mod: object, friction: float
) -> None:
    with pytest.raises(ValueError, match="friction"):
        _manifest_scene_mod.apply_finger_friction(object(), "/robot", ["j"], friction)  # type: ignore[attr-defined]


# ── gripper commands are read in the end effector's command_convention ────────


@pytest.mark.parametrize(
    ("robot_id", "closed_cmd", "open_cmd"),
    [
        # robots driven by an isaac_sim scene under scenes/
        ("panda_mobile", 1.0, -1.0),  # normalized_close_symmetric: +1 closes
        ("franka_panda", 0.0, 1.0),  # normalized_open_unit
        ("openarm", 0.0, None),  # raw_joint_rad passthrough: per-jaw command_range
    ],
)
def test_isaac_gripper_command_follows_the_end_effector_convention(
    _manifest_scene_mod: object, robot_id: str, closed_cmd: float, open_cmd: float | None
) -> None:
    """A closing command drives every Isaac gripper to its URDF closed target, an opening
    one to its open target — panda_mobile's +1 used to open the jaw (merge 0cbebfcc)."""
    to_urdf = _manifest_scene_mod.manifest_to_urdf_gripper  # type: ignore[attr-defined]
    desc = RobotDescription.from_yaml(str(_repo_root() / "robots" / robot_id / "robot.yaml"))
    spec = _build_robot_spec(desc, robot_id)
    ranges = {e.name: e.command_range for e in desc.end_effectors}
    assert spec["grippers"]
    for g in spec["grippers"]:
        opening = open_cmd
        if opening is None:  # OpenArm: the open end of this jaw's command_range
            opening = max(ranges[g["name"]], key=abs)
        assert to_urdf(g, closed_cmd) == pytest.approx(g["closed"])
        assert to_urdf(g, opening) == pytest.approx(g["open"])
        assert g["open"] != pytest.approx(g["closed"])


def test_a_width_meters_gripper_is_refused_in_the_isaac_scene() -> None:
    """A physical convention with no mapping onto the joint fails the scene build."""
    from openral_core.exceptions import ROSConfigError

    desc = RobotDescription.from_yaml(str(_repo_root() / "robots" / "franka_panda" / "robot.yaml"))
    ee = desc.end_effectors[0].model_copy(
        update={"command_convention": "width_meters", "command_range": (0.0, 0.08)}
    )
    desc = desc.model_copy(update={"end_effectors": [ee, *desc.end_effectors[1:]]})
    with pytest.raises(ROSConfigError, match="width_meters"):
        _build_robot_spec(desc, "franka_panda")


# ── RGB lens validity + render softening ──────────────────────────────────────

# The OpenArm ZED-M left lens, VGA raw: the Oct-2 Brown fit (the policy's training camera)
# and its monotonic rational extension (fitted 2026-10-07 to the SDK's raw->rect map; equal
# to the Brown fit within 0.007 px inside its calibrated radius).
_ZED_TOP_K = {"width": 672, "height": 376, "fx": 388.508267, "fy": 388.527128,
              "cx": 336.707405, "cy": 190.106848}  # fmt: skip
_ZED_TOP_BROWN = [-0.358038981, 0.187982349, -0.000311155, 0.000069646, -0.056267709]
_ZED_TOP_RATIONAL = [-1.441958182143234, 0.7289916610988222, -0.000311155, 6.9646e-05,
                     0.19532166359108094, -1.0824447975648817, 0.1369386641081563,
                     0.5689119070450521]  # fmt: skip


def test_a_brown_fit_that_folds_inside_the_frame_is_refused(_manifest_scene_mod: object) -> None:
    """Isaac inverts the lens per output pixel; past the fold there is no ray, so the
    renderer draws swirls at the image edge. The Oct-2 fit folds at r_d 0.858 < 0.995."""
    mod = _manifest_scene_mod
    assert mod.radial_fold_radius(_ZED_TOP_BROWN) == pytest.approx(0.858, abs=1e-3)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match=r"folds at normalised radius 0\.858"):
        mod.check_radial_reaches_corners("top", _ZED_TOP_K, _ZED_TOP_BROWN)  # type: ignore[attr-defined]


def test_the_rational_extension_reaches_the_corners_and_keeps_the_brown_interior(
    _manifest_scene_mod: object,
) -> None:
    mod = _manifest_scene_mod
    mod.check_radial_reaches_corners("top", _ZED_TOP_K, _ZED_TOP_RATIONAL)  # type: ignore[attr-defined]

    def radial(c: list[float], r: np.ndarray) -> np.ndarray:
        k1, k2, _, _, k3, k4, k5, k6 = (c + [0.0] * 8)[:8]
        r2 = r * r
        return r * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)

    r = np.linspace(0.0, 0.8, 200)  # well inside the Brown fit's valid range
    px = np.abs(radial(_ZED_TOP_RATIONAL, r) - radial(_ZED_TOP_BROWN, r)) * _ZED_TOP_K["fx"]
    assert px.max() < 0.05


def test_the_thor_wrist_calibrations_pass_the_corner_check(_manifest_scene_mod: object) -> None:
    """Real fixtures: the committed wrist Arducam calibrations are monotonic to their corners."""
    path = _repo_root() / "robots/openarm/robot.yaml"
    robot = RobotDescription.from_yaml(path)
    sensors = apply_sensor_overlays(
        robot.sensors, resolve_sensor_overlays(path, "thor", required=True)
    )
    for s in sensors:
        if s.name in ("wrist_left", "wrist_right"):
            assert s.intrinsics is not None
            k = s.intrinsics.model_dump()
            _manifest_scene_mod.check_radial_reaches_corners(s.name, k, k["distortion_coeffs"])  # type: ignore[attr-defined]


def test_rational_polynomial_is_a_schema_model() -> None:
    from openral_core import IntrinsicsPinhole

    k = IntrinsicsPinhole(
        **_ZED_TOP_K, distortion_model="rational_polynomial", distortion_coeffs=_ZED_TOP_RATIONAL
    )
    assert IntrinsicsPinhole.model_validate_json(k.model_dump_json()) == k


def test_blur_softens_without_shifting_or_darkening(_manifest_scene_mod: object) -> None:
    blur = _manifest_scene_mod.blur_rgb  # type: ignore[attr-defined]
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)
    assert blur(img, 0.0) is img
    out = blur(img, 0.9)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert abs(float(out.mean()) - float(img.mean())) < 0.5
    assert float(np.abs(np.diff(out.astype(float), axis=1)).mean()) < float(
        np.abs(np.diff(img.astype(float), axis=1)).mean()
    )
    dot = np.zeros((11, 11, 3), dtype=np.uint8)
    dot[5, 5] = 255
    peak = np.unravel_index(np.argmax(blur(dot, 1.0)[..., 0]), (11, 11))
    assert peak == (5, 5)


def test_image_blur_option_is_bounded() -> None:
    from openral_sim.backends.isaac_sim import IsaacSimOptions
    from pydantic import ValidationError

    assert IsaacSimOptions().image_blur_sigma_px == 0.0
    assert IsaacSimOptions(image_blur_sigma_px=0.9).image_blur_sigma_px == 0.9
    with pytest.raises(ValidationError):
        IsaacSimOptions(image_blur_sigma_px=-0.1)
    with pytest.raises(ValidationError):
        IsaacSimOptions(image_blur_sigma_px=6.0)


def test_relight_matches_exposure_and_auto_exposure_hits_target(
    _manifest_scene_mod: object,
) -> None:
    """One stop of scene-linear gain through the inverse op-6 tonemap is monotone, keeps
    black at black, and the AE gain brings a dark frame's mean luma to the target."""
    relight = _manifest_scene_mod.relight_rgb  # type: ignore[attr-defined]
    ae_gain = _manifest_scene_mod.auto_exposure_gain  # type: ignore[attr-defined]
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    up = relight(ramp, (2.0, 2.0, 2.0)).astype(int)
    assert up[0, 0, 0] == 0 and (np.diff(up[0, :, 0]) >= 0).all() and (up >= ramp).all()
    assert (relight(ramp, (1.0, 1.0, 1.0)) == ramp).all()
    rng = np.random.default_rng(0)
    dark = rng.integers(10, 90, size=(64, 96, 3), dtype=np.uint8)
    g = ae_gain(dark, 107.0, (1.0, 1.0, 1.2))
    lit = relight(dark, (g, g, 1.2 * g)).astype(float) @ np.array([0.299, 0.587, 0.114])
    assert g > 1.0 and abs(float(lit.mean()) - 107.0) < 3.0


def test_camera_image_post_needs_exposure_ev() -> None:
    from openral_sim.backends.isaac_sim import IsaacSimOptions
    from pydantic import ValidationError

    post = {"wrist_left": {"blur_sigma_px": 1.5, "auto_exposure_target": 107}}
    opts = IsaacSimOptions(exposure_ev=-1.0, image_blur_sigma_px=0.9, camera_image_post=post)
    assert opts.camera_image_post["wrist_left"].blur_sigma_px == 1.5
    IsaacSimOptions(camera_image_post={"wrist_left": {"blur_sigma_px": 1.5}})  # blur alone: fine
    with pytest.raises(ValidationError):
        IsaacSimOptions(camera_image_post=post)
    with pytest.raises(ValidationError):
        IsaacSimOptions(exposure_ev=-1.0, camera_image_post={"w": {"white_balance_rgb": (0, 1, 1)}})


def test_robot_material_recolour_targets_only_matching_robot_materials(
    _manifest_scene_mod: object,
) -> None:
    """OpenArm's upstream `.dae` meshes ship a "matte_black" at diffuse 0.247 (renders grey);
    the recolour sets only matching materials bound under the robot prim."""
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()

    def material(path: str, rgb: tuple[float, float, float]) -> object:
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    black = material("/World/robot/Looks/palette_01_matte_black_001_material", (0.247,) * 3)
    silver = material("/World/robot/Looks/palette_00_metal_silver_001_material", (0.65,) * 3)
    shelf = material("/World/Looks/shelf_matte_black_material", (0.5,) * 3)  # not the robot's
    for prim_path, mat in (
        ("/World/robot/link1/visual_a", black),
        ("/World/robot/link1/visual_b", silver),
        ("/World/shelf/mesh", shelf),
    ):
        UsdShade.MaterialBindingAPI.Apply(UsdGeom.Mesh.Define(stage, prim_path).GetPrim()).Bind(mat)

    changed = _manifest_scene_mod.recolor_robot_materials(  # type: ignore[attr-defined]
        stage, "/World/robot", {"*matte_black*": [0.03, 0.03, 0.03]}
    )
    assert list(changed) == ["/World/robot/Looks/palette_01_matte_black_001_material"]

    def diffuse(path: str) -> tuple[float, ...]:
        shader = UsdShade.Shader(stage.GetPrimAtPath(f"{path}/Shader"))
        return tuple(shader.GetInput("diffuseColor").Get())

    assert diffuse("/World/robot/Looks/palette_01_matte_black_001_material") == pytest.approx(
        (0.03, 0.03, 0.03)
    )
    assert diffuse("/World/robot/Looks/palette_00_metal_silver_001_material") == pytest.approx(
        (0.65, 0.65, 0.65)
    )
    assert diffuse("/World/Looks/shelf_matte_black_material") == pytest.approx((0.5, 0.5, 0.5))


def test_robot_material_recolour_writes_the_connected_material_input(
    _manifest_scene_mod: object,
) -> None:
    """Isaac's URDF importer exposes the colour as a Material interface input that the
    shader's diffuseColor is connected to; the recolour must write that source."""
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    mat = UsdShade.Material.Define(stage, "/World/robot/Materials/palette_01_matte_black")
    iface = mat.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    iface.Set(Gf.Vec3f(0.05, 0.05, 0.05))
    sh = UsdShade.Shader.Define(stage, "/World/robot/Materials/palette_01_matte_black/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(iface)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    mesh = UsdGeom.Mesh.Define(stage, "/World/robot/link1/visual").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(mat)

    _manifest_scene_mod.recolor_robot_materials(  # type: ignore[attr-defined]
        stage, "/World/robot", {"*matte_black*": [0.01, 0.01, 0.01]}
    )
    assert tuple(iface.Get()) == pytest.approx((0.01, 0.01, 0.01))


def test_robot_material_colours_are_validated() -> None:
    from openral_sim.backends.isaac_sim import IsaacSimOptions
    from pydantic import ValidationError

    opts = IsaacSimOptions(robot_material_colors={"*matte_black*": (0.03, 0.03, 0.03)})
    assert opts.robot_material_colors["*matte_black*"] == (0.03, 0.03, 0.03)
    with pytest.raises(ValidationError, match="RGB in"):
        IsaacSimOptions(robot_material_colors={"*": (1.2, 0.0, 0.0)})


def test_robot_material_recolour_reaches_instanced_visuals(_manifest_scene_mod: object) -> None:
    """Isaac's URDF importer instances the visuals, so their materials are read-only
    instance proxies: the recolour de-instances only the instances holding a match."""
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    proto = "/Prototypes/link_visual"
    mesh = UsdGeom.Mesh.Define(stage, f"{proto}/mesh").GetPrim()
    mat = UsdShade.Material.Define(stage, f"{proto}/Looks/palette_01_matte_black")
    mat.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.05, 0.05))
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(mat)
    collider = stage.DefinePrim("/World/robot/link1/collisions")
    collider.GetReferences().AddInternalReference(proto)
    collider.SetInstanceable(True)
    visual = stage.DefinePrim("/World/robot/link1/visuals")
    visual.GetReferences().AddInternalReference(proto)
    visual.SetInstanceable(True)
    stage.GetPrimAtPath("/Prototypes").SetActive(False)
    assert stage.GetPrimAtPath("/World/robot/link1/visuals/mesh").IsInstanceProxy()

    changed = _manifest_scene_mod.recolor_robot_materials(  # type: ignore[attr-defined]
        stage, "/World/robot", {"*matte_black*": [0.01, 0.01, 0.01]}
    )
    assert changed
    for root in ("visuals", "collisions"):
        prim = stage.GetPrimAtPath(f"/World/robot/link1/{root}/mesh")
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        assert tuple(bound.GetInput("diffuseColor").Get()) == pytest.approx((0.01, 0.01, 0.01))


def _noisy(x: float, y: float, yaw: float, **noise: object) -> dict[str, object]:
    base: dict[str, object] = {
        "xy_sigma_m": [0.01, 0.01],
        "yaw_sigma_deg": 10.0,
        "clip_sigma": 2.0,
        "keep_inside_xy": None,
        "max_tries": 200,
    }
    return {"xyz": [x, y, 0.2], "yaw": yaw, "pose_noise": {**base, **noise}}


def test_object_pose_noise_is_seeded_truncated_and_collision_free(
    _manifest_scene_mod: object,
) -> None:
    """Two tote cartons (OpenArm restock E14 footprints) 8 mm apart, inside a tote floor."""
    mod = _manifest_scene_mod
    sample = mod.sample_object_poses  # type: ignore[attr-defined]
    overlap = mod._rects_overlap  # type: ignore[attr-defined]
    corners = mod._rect_corners  # type: ignore[attr-defined]
    tote = [[-0.14, -0.18], [0.094, 0.168]]
    objs = [
        _noisy(0.04, 0.07, -1.57, keep_inside_xy=tote),
        _noisy(0.04, -0.01, -1.57, keep_inside_xy=tote),
        {"xyz": [-0.06, -0.09, 0.2], "yaw": 0.0},  # no noise: always declared
    ]
    foot = [(0.0, 0.0, 0.0368, 0.0175)] * 3
    first = sample(objs, foot, np.random.default_rng(7))
    assert first == sample(objs, foot, np.random.default_rng(7))
    assert first != sample(objs, foot, np.random.default_rng(8))
    for seed in range(200):
        poses = sample(objs, foot, np.random.default_rng(seed))
        assert poses[2] == (-0.06, -0.09, 0.0, "declared")
        quads = [corners(x, y, yaw, f) for (x, y, yaw, _), f in zip(poses, foot, strict=True)]
        for (x, y, yaw, status), obj in zip(poses[:2], objs[:2], strict=True):
            assert status == "sampled"
            assert abs(x - obj["xyz"][0]) <= 0.02 + 1e-12  # type: ignore[index]
            assert abs(y - obj["xyz"][1]) <= 0.02 + 1e-12  # type: ignore[index]
            assert abs(yaw - obj["yaw"]) <= np.radians(20) + 1e-12  # type: ignore[operator]
        assert not overlap(quads[0], quads[1], 0.002)
        for q in quads[:2]:
            assert np.all(q >= np.add(tote[0], 0.002)) and np.all(q <= np.subtract(tote[1], 0.002))


def test_object_pose_noise_falls_back_to_the_declared_pose(_manifest_scene_mod: object) -> None:
    sample = _manifest_scene_mod.sample_object_poses  # type: ignore[attr-defined]
    # A carton wider than its keep-inside box can never be placed.
    objs = [_noisy(0.0, 0.0, 0.0, keep_inside_xy=[[-0.01, -0.01], [0.01, 0.01]], max_tries=5)]
    assert sample(objs, [(0.0, 0.0, 0.03, 0.03)], np.random.default_rng(0)) == [
        (0.0, 0.0, 0.0, "declared_fallback")
    ]


def test_object_pose_noise_schema() -> None:
    from openral_sim.backends.isaac_sim import IsaacSimOptions

    opts = IsaacSimOptions(
        objects=[
            {
                "usd": "isaac:Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
                "name": "cracker_box",
                "xyz": (0.0, 0.0, 0.2),
                "pose_noise": {"xy_sigma_m": (0.01, 0.005), "yaw_sigma_deg": 6.0},
            }
        ]
    )
    noise = opts.objects[0].pose_noise
    assert noise is not None and noise.clip_sigma == 2.0 and noise.keep_inside_xy is None
    with pytest.raises(ValueError, match="xy_sigma_m"):
        IsaacSimOptions(
            objects=[
                {
                    "usd": "a.usd",
                    "name": "a",
                    "xyz": (0, 0, 0),
                    "pose_noise": {"xy_sigma_m": (-1, 0)},
                }
            ]
        )
    with pytest.raises(ValueError, match="keep_inside_xy"):
        IsaacSimOptions(
            objects=[
                {
                    "usd": "a.usd",
                    "name": "a",
                    "xyz": (0, 0, 0),
                    "pose_noise": {"keep_inside_xy": ((0.1, 0.0), (0.0, 0.1))},
                }
            ]
        )


def test_render_desc_scales_rgb_rasters_and_can_drop_depth() -> None:
    """OpenArm restock speed knobs: wrists rendered at 224 px high keep their field of view."""
    from openral_sim.backends.isaac_sim import _render_desc

    path = _repo_root() / "robots/openarm/robot.yaml"
    desc = RobotDescription.from_yaml(path)
    desc = desc.model_copy(
        update={
            "sensors": apply_sensor_overlays(
                desc.sensors, resolve_sensor_overlays(path, "thor", required=True)
            )
        }
    )
    before = {s.name: s for s in desc.sensors}
    out, scale = _render_desc(desc, {"wrist_left": 224, "wrist_right": 224}, depth_cameras=False)
    after = {s.name: s for s in out.sensors}
    assert not any(s.modality == "depth" for s in out.sensors)
    assert any(s.modality == "depth" for s in desc.sensors)  # the manifest has one to drop
    for name in ("wrist_left", "wrist_right"):
        k0, k1 = before[name].intrinsics, after[name].intrinsics
        assert k0 is not None and k1 is not None
        assert k1.height == 224 and scale[name] == pytest.approx(224 / k0.height)
        assert k1.width == round(k0.width * scale[name])
        # Same field of view: focal length over raster is unchanged.
        assert k1.fx / k1.width == pytest.approx(k0.fx / k0.width, rel=2e-3)
        assert k1.distortion_coeffs == k0.distortion_coeffs
    untouched = [n for n, s in before.items() if s.modality == "rgb" and n not in scale]
    assert all(after[n].intrinsics == before[n].intrinsics for n in untouched)


def test_render_options_are_validated() -> None:
    from openral_sim.backends.isaac_sim import IsaacSimOptions

    opts = IsaacSimOptions(
        physics_substeps=2, camera_render_height={"top": 224}, depth_cameras=False
    )
    assert opts.physics_substeps == 2
    with pytest.raises(ValueError, match="camera_render_height"):
        IsaacSimOptions(camera_render_height={"top": 8})
    with pytest.raises(ValueError):
        IsaacSimOptions(physics_substeps=0)


def test_joint_drive_gains_become_importer_pattern_dicts(_manifest_scene_mod: object) -> None:
    from openral_sim.backends.isaac_sim import IsaacSimOptions

    drive_gain_overrides = _manifest_scene_mod.drive_gain_overrides  # type: ignore[attr-defined]

    gains = {"joint[123]$": (70.0, 2.75), "joint[567]$": (10.0, 0.7)}
    opts = IsaacSimOptions(joint_drive_gains=gains)
    stiffness, damping = drive_gain_overrides(
        {k: list(v) for k, v in opts.joint_drive_gains.items()}
    )
    # Catch-all default first, so joints no pattern names keep the stiff default.
    assert list(stiffness) == [".*", "joint[123]$", "joint[567]$"]
    assert stiffness[".*"] == 1000.0 and damping[".*"] == 100.0
    assert stiffness["joint[567]$"] == 10.0 and damping["joint[567]$"] == 0.7
    assert drive_gain_overrides(None) == ({".*": 1000.0}, {".*": 100.0})
    with pytest.raises(ValueError, match="joint_drive_gains"):
        IsaacSimOptions(joint_drive_gains={"joint1": (-1.0, 0.0)})
    with pytest.raises(ValueError):
        IsaacSimOptions(joint_drive_gains={"joint[": (1.0, 0.0)})


def test_color_lut_option_validates_and_maps_per_channel(_manifest_scene_mod: object) -> None:
    from openral_sim.backends.isaac_sim import IsaacCameraImagePost

    apply_color_lut = _manifest_scene_mod.apply_color_lut  # type: ignore[attr-defined]
    lut = ([v // 2 for v in range(256)], list(range(256)), [255 - v for v in range(256)])
    post = IsaacCameraImagePost(color_lut_rgb=lut)
    img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3) * 3
    out = apply_color_lut(img, post.model_dump(mode="json")["color_lut_rgb"])
    assert (out[..., 0] == img[..., 0] // 2).all()
    assert (out[..., 1] == img[..., 1]).all()
    assert (out[..., 2] == 255 - img[..., 2]).all()
    with pytest.raises(ValueError, match="color_lut_rgb"):
        IsaacCameraImagePost(color_lut_rgb=(list(range(255)), list(range(256)), list(range(256))))
    with pytest.raises(ValueError, match="color_lut_rgb"):
        IsaacCameraImagePost(color_lut_rgb=([256] * 256, list(range(256)), list(range(256))))


def test_gripper_joint_curve_maps_commands_and_states_through_the_curve(
    _manifest_scene_mod: object,
) -> None:
    """OpenArm right jaw: reading -0.28 on a 6 cm carton <-> URDF finger -0.69; open stays 1:1.
    A command lands on the finger through the curve and the finger reads back through it."""
    from openral_core.exceptions import ROSConfigError
    from openral_sim.backends.isaac_sim import IsaacSimOptions, _curve_gripper_specs
    from openral_sim.registry import ROBOTS

    mod: Any = _manifest_scene_mod
    curve = [(0.0, 0.0), (-0.28, -0.69), (-0.785, -0.785)]
    spec = _build_robot_spec(ROBOTS.get("openarm")(), "openarm")
    _curve_gripper_specs(spec, {"right_gripper": curve})
    g = next(g for g in spec["grippers"] if g["name"] == "right_gripper")
    assert mod.manifest_to_urdf_gripper(g, -0.28) == pytest.approx(-0.69)
    assert mod.manifest_to_urdf_gripper(g, -0.785) == pytest.approx(-0.785)
    assert mod.manifest_to_urdf_gripper(g, -0.14) == pytest.approx(-0.345)
    m_pts, u_pts = mod._curve_by_urdf(g["curve"])
    assert float(np.interp(-0.69, u_pts, m_pts)) == pytest.approx(-0.28)
    with pytest.raises(ROSConfigError, match="gripper_joint_curve"):
        _curve_gripper_specs(spec, {"right_joint1": curve})
    with pytest.raises(ValueError, match="monotonic"):
        IsaacSimOptions(
            gripper_joint_curve={"right_gripper": [(0.0, 0.0), (-0.3, 0.1), (-0.7, -0.7)]}
        )
