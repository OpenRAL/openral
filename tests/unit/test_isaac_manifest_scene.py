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

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_sim.backends.isaac_sim import _build_robot_spec


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
    assert set(cam["intrinsics"]) == {"width", "height", "fx", "fy", "cx", "cy"}


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
    bound = _manifest_scene_mod.apply_finger_friction(  # type: ignore[attr-defined]
        stage, "/openarm", ["finger_joint1", "finger_joint2"], 0.9
    )
    assert sorted(bound) == ["/openarm/hand/ee_link1", "/openarm/hand/ee_link2"]
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
    bound = _manifest_scene_mod.apply_finger_friction(stage, "/openarm", ["no_such_joint"])  # type: ignore[attr-defined]
    assert bound == []
    assert not stage.GetPrimAtPath("/World/Physics_Materials/finger_pad")


@pytest.mark.parametrize("friction", [0.0, -1.0, float("nan"), float("inf")])
def test_finger_friction_refuses_a_bad_coefficient(
    _manifest_scene_mod: object, friction: float
) -> None:
    with pytest.raises(ValueError, match="friction"):
        _manifest_scene_mod.apply_finger_friction(object(), "/robot", ["j"], friction)  # type: ignore[attr-defined]
