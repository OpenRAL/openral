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
    assert action["gripper_open_m"] > action["gripper_closed_m"]


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

    # panda_mobile declares a NORMALISED [0, 1] gripper width — must NOT leak to
    # the Isaac finger DOF (would tear the joint); falls back to the Panda 0.04 m.
    assert spec["action"]["gripper_open_m"] == pytest.approx(0.04)


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


def test_map_dof_to_manifest_franka(_manifest_scene_mod: object) -> None:
    map_dof_to_manifest = _manifest_scene_mod.map_dof_to_manifest  # type: ignore[attr-defined]

    # The Panda URDF's actuated DOFs (what Isaac's articulation exposes): 7 arm
    # joints + 2 prismatic fingers. The manifest collapses the fingers into one
    # `panda_gripper` width DoF.
    dof_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    dof_index = {n: i for i, n in enumerate(dof_names)}
    finger_dof_idx = [7, 8]
    manifest_joints = [{"name": f"panda_joint{i}", "role": "arm"} for i in range(1, 8)]
    manifest_joints.append({"name": "panda_gripper", "role": "gripper"})

    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.02, 0.04], dtype=np.float32)
    mapped = map_dof_to_manifest(
        values,
        dof_index=dof_index,
        manifest_joints=manifest_joints,
        finger_dof_idx=finger_dof_idx,
    )

    assert mapped.shape == (8,)
    # Arm joints map straight through by name.
    np.testing.assert_allclose(mapped[:7], values[:7], rtol=0, atol=1e-6)
    # Gripper = mean of the two finger DOFs (0.02 + 0.04) / 2.
    assert mapped[7] == pytest.approx(0.03)


def test_map_dof_to_manifest_base_joints_from_pose(_manifest_scene_mod: object) -> None:
    map_dof_to_manifest = _manifest_scene_mod.map_dof_to_manifest  # type: ignore[attr-defined]

    # panda_mobile order: 3 base + 7 arm + 1 gripper. Base joints are NOT URDF
    # DOFs — they come from the kinematic base pose (x, y, yaw).
    manifest_joints = [
        {"name": "base_x", "role": "base"},
        {"name": "base_y", "role": "base"},
        {"name": "base_yaw", "role": "base"},
        *({"name": f"panda_joint{i}", "role": "arm"} for i in range(1, 8)),
        {"name": "panda_gripper", "role": "gripper"},
    ]
    dof_names = [f"panda_joint{i}" for i in range(1, 8)] + [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    dof_index = {n: i for i, n in enumerate(dof_names)}
    arm_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.02, 0.02], dtype=np.float32)

    mapped = map_dof_to_manifest(
        arm_vals,
        dof_index=dof_index,
        manifest_joints=manifest_joints,
        finger_dof_idx=[7, 8],
        base_values=[1.5, -0.5, 0.785],  # x, y, yaw
        base_joints=["base_x", "base_y", "base_yaw"],
    )

    assert mapped.shape == (11,)
    # Base joints carry the kinematic pose, in manifest order.
    np.testing.assert_allclose(mapped[:3], [1.5, -0.5, 0.785], rtol=0, atol=1e-6)
    # Arm joints follow, then the collapsed gripper.
    np.testing.assert_allclose(mapped[3:10], arm_vals[:7], rtol=0, atol=1e-6)
    assert mapped[10] == pytest.approx(0.02)


def test_map_dof_to_manifest_unresolved_joint_is_zero(_manifest_scene_mod: object) -> None:
    map_dof_to_manifest = _manifest_scene_mod.map_dof_to_manifest  # type: ignore[attr-defined]
    # A manifest joint absent from the articulation (and not a gripper) → 0.0,
    # never an index error.
    mapped = map_dof_to_manifest(
        np.array([1.0], dtype=np.float32),
        dof_index={"panda_joint1": 0},
        manifest_joints=[{"name": "ghost_joint", "role": "arm"}],
        finger_dof_idx=[],
    )
    assert mapped.shape == (1,)
    assert mapped[0] == 0.0


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


_BEAM = {"origin_xy": (0.0, 0.0), "angle_rad": 0.0, "z": 0.30, "range_max_m": 12.0}


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
