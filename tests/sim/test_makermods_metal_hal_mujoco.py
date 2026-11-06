# SPDX-License-Identifier: Apache-2.0
"""MakerMods Metal arm + two-arm station: closed-loop tracking on the derived MJCF.

Drives the real ``MujocoArmHAL.from_description`` built purely from
``robots/makermods_metal/robot.yaml`` and ``robots/makermods_metal_station/robot.yaml``
against the MJCF that ``makermods:`` derives from the pinned upstream
``metal_description`` URDF (gravity on, the manifest's 30 Hz action rate). The
robot is moved only through ``send_action``; ``read_state`` is the only
read-back. For each robot it checks that

* every arm joint settles within 0.05 rad of a non-trivial target pose after
  ~2 s of sim time (the sim-only position-servo gains in
  ``openral_hal._makermods_metal_assets`` hold the arm against gravity), and
* the normalised gripper (0 = closed, 1 = open) reads back within 0.05 of the
  command for both open and closed;

and that the station's two arms (``left_`` / ``right_`` at
``y = ±STATION_ARM_SPACING_M / 2``) are contact-free at the zero pose.

No mocks (CLAUDE.md §1.11): when ``mujoco`` is missing or the pinned clone
cannot be fetched (``AssetFetchError``), the module skips with the reason.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from openral_core import Action, ControlMode, RobotDescription  # noqa: E402
from openral_core.assets import AssetFetchError, resolve_asset  # noqa: E402
from openral_hal._mujoco_arm import MujocoArmHAL  # noqa: E402

pytestmark = pytest.mark.sim

_ROOT = Path(__file__).resolve().parents[2]
_TRACK_TOL_RAD = 0.05
_GRIPPER_TOL = 0.05
_SETTLE_SIM_S = 2.0

#: Per-arm target (joint1..joint6), inside every manifest position limit and
#: away from them (joint2 / joint3 rest at a limit at qpos 0). The right arm
#: mirrors joint1 so the two station arms swing away from each other.
_ARM_TARGET: tuple[float, ...] = (0.4, -0.9, 1.1, 0.3, 0.4, 0.5)


def _description(robot_id: str) -> RobotDescription:
    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / robot_id / "robot.yaml"))
    assert desc.assets.mjcf is not None
    try:
        resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetFetchError as exc:
        pytest.skip(f"pinned MakerMods Metal clone not fetchable on this host: {exc}")
    return desc


@pytest.fixture(params=["makermods_metal", "makermods_metal_station"])
def connected(request: pytest.FixtureRequest) -> Iterator[tuple[RobotDescription, MujocoArmHAL]]:
    """A connected, gravity-on HAL built purely from the parametrised manifest."""
    desc = _description(str(request.param))
    hal = MujocoArmHAL.from_description(desc, gravity_enabled=True)
    hal.connect()
    try:
        yield desc, hal
    finally:
        hal.disconnect()


def _target(desc: RobotDescription, gripper: float) -> list[float]:
    """Full action vector: the arm target per arm, ``gripper`` on every gripper slot."""
    out: list[float] = []
    for joint in desc.joints:
        if joint.role == "gripper":
            out.append(gripper)
            continue
        local = str(joint.sim_joint_name).removeprefix("left_").removeprefix("right_")
        index = int(local.removeprefix("joint"))
        value = _ARM_TARGET[index - 1]
        if index == 1 and joint.name.startswith("right_"):
            value = -value
        out.append(value)
    return out


def _drive(hal: MujocoArmHAL, target: list[float]) -> list[float]:
    """Hold ``target`` for ``_SETTLE_SIM_S`` of sim time; return the final positions."""
    tick_s = hal.sim_dt_per_tick_s
    assert tick_s is not None and tick_s > 0.0
    for _ in range(round(_SETTLE_SIM_S / tick_s)):
        state = hal.read_state()
        hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                joint_targets=[list(target)],
                horizon=1,
                stamp_ns=state.stamp_ns,
            )
        )
    return list(hal.read_state().position)


@pytest.mark.parametrize("gripper", [1.0, 0.0], ids=["open", "closed"])
def test_tracks_joint_targets_and_gripper(
    connected: tuple[RobotDescription, MujocoArmHAL], gripper: float
) -> None:
    """Arm joints settle within 0.05 rad; the normalised gripper within 0.05."""
    desc, hal = connected
    target = _target(desc, gripper)
    for joint, want in zip(desc.joints, target, strict=True):
        assert joint.position_limits is not None, joint.name
        low, high = joint.position_limits
        assert low <= want <= high, f"target for {joint.name} outside its limits"
    got = _drive(hal, target)
    for joint, want, have in zip(desc.joints, target, got, strict=True):
        tol = _GRIPPER_TOL if joint.role == "gripper" else _TRACK_TOL_RAD
        assert abs(have - want) < tol, (
            f"{desc.name}.{joint.name}: commanded {want:.4f}, settled at {have:.4f}"
        )


def test_station_arms_contact_free_at_zero_pose() -> None:
    """At qpos = 0 the station MJCF has no active contact (arms and fingers clear)."""
    desc = _description("makermods_metal_station")
    assert desc.assets.mjcf is not None
    path = resolve_asset(desc.assets.mjcf, "mjcf")
    assert path is not None
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    pairs = [
        (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]),
        )
        for c in data.contact[: data.ncon]
    ]
    assert data.ncon == 0, f"contacts at the zero pose: {pairs}"
    # Both arms are present and mounted where the manifest says.
    for prefix, sign in (("left_", 1.0), ("right_", -1.0)):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base_link")
        assert bid >= 0
        mount = next(a for a in desc.fixed_attachments if a.name == f"{prefix}mount")
        assert float(data.xpos[bid][1]) == pytest.approx(sign * abs(mount.origin_xyz[1]))


def test_station_arms_stay_apart_while_tracking() -> None:
    """Driving both arms to the mirrored target pose produces no inter-arm contact."""
    desc = _description("makermods_metal_station")
    hal = MujocoArmHAL.from_description(desc, gravity_enabled=True)
    hal.connect()
    try:
        _drive(hal, _target(desc, 1.0))
        handles = hal.mujoco_handles()
        assert handles is not None
        model, data = handles
        for c in data.contact[: data.ncon]:
            b1 = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]))
            b2 = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]))
            assert b1.split("_", 1)[0] == b2.split("_", 1)[0], f"inter-arm contact {b1} / {b2}"
    finally:
        hal.disconnect()
