# SPDX-License-Identifier: Apache-2.0
"""Galaxea A1 MuJoCo twin: the real HAL's contract, tracking, gripper, cameras (#740).

Drives the real ``MujocoArmHAL.from_description`` built purely from
``robots/galaxea_a1/robot.yaml`` on the MJCF derived from the vendor URDF
(``galaxea:a1``), gravity on. The twin must accept and publish exactly what
``GalaxeaA1HAL`` does: six arm joints in the joint state, the gripper only as a
``GRIPPER_POSITION`` action with ``gripper[0]`` in ``[0, 1]`` (refused outside it),
reported in ``health`` as ``gripper_position_normalized``.

No mocks (CLAUDE.md §1.11): when ``mujoco`` is missing or the vendor files cannot be
downloaded (``AssetFetchError``), the module skips with the reason.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

import numpy as np
from openral_core import Action, ControlMode, RobotDescription
from openral_core.assets import AssetFetchError, resolve_asset
from openral_core.exceptions import ROSConfigError
from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal.galaxea_a1 import GALAXEA_A1_DESCRIPTION

pytestmark = pytest.mark.sim

_ROOT = Path(__file__).resolve().parents[2]
_TRACK_TOL_RAD = 0.05
_GRIPPER_TOL = 0.05
_SETTLE_SIM_S = 1.5
#: arm_joint1..6, inside every manifest limit and away from them (joint 3 is <= 0).
_ARM_TARGET = [0.4, 0.8, -1.0, 0.3, 0.5, -0.4]


@pytest.fixture
def connected() -> Iterator[MujocoArmHAL]:
    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / "galaxea_a1" / "robot.yaml"))
    assert desc.assets.mjcf is not None
    try:
        resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetFetchError as exc:
        pytest.skip(f"Galaxea A1 vendor files not fetchable on this host: {exc}")
    hal = MujocoArmHAL.from_description(desc, gravity_enabled=True)
    hal.connect()
    try:
        yield hal
    finally:
        hal.disconnect()


def _ticks(hal: MujocoArmHAL) -> int:
    tick_s = hal.sim_dt_per_tick_s
    assert tick_s is not None and tick_s > 0.0
    return round(_SETTLE_SIM_S / tick_s)


def _gripper(hal: MujocoArmHAL, value: float) -> None:
    hal.send_action(Action(control_mode=ControlMode.GRIPPER_POSITION, gripper=[value]))


def test_joint_state_is_the_real_hals_six_arm_joints(connected: MujocoArmHAL) -> None:
    """Same names, order and width as ``GalaxeaA1HAL.read_state``: no gripper joint."""
    state = connected.read_state()
    assert list(state.name) == [j.name for j in GALAXEA_A1_DESCRIPTION.joints]
    assert list(state.name) == [f"arm_joint{i}" for i in range(1, 7)]
    assert connected.sim_dt_per_tick_s == pytest.approx(1.0 / 30.0)


def test_every_joint_names_its_mjcf_joint(connected: MujocoArmHAL) -> None:
    """``sim_joint_name`` names a real MJCF joint at the manifest joint's qpos slot."""
    import mujoco

    model = connected._model
    assert model is not None
    for index, joint in enumerate(connected.description.joints):
        assert joint.sim_joint_name is not None, joint.name
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.sim_joint_name)
        assert jid >= 0 and int(model.jnt_qposadr[jid]) == index, joint.name


def test_tracks_joint_targets(connected: MujocoArmHAL) -> None:
    """Arm joints settle within 0.05 rad of a 6-wide JOINT_POSITION target."""
    for _ in range(_ticks(connected)):
        connected.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[_ARM_TARGET])
        )
    got = list(connected.read_state().position)
    for want, have in zip(_ARM_TARGET, got, strict=True):
        assert abs(have - want) < _TRACK_TOL_RAD, f"commanded {want:.3f}, settled {have:.3f}"


@pytest.mark.parametrize("value", [1.0, 0.0, 0.5], ids=["open", "closed", "half"])
def test_gripper_position_drives_the_fingers(connected: MujocoArmHAL, value: float) -> None:
    """GRIPPER_POSITION moves both fingers; health reports it under the real HAL's key."""
    for _ in range(_ticks(connected)):
        _gripper(connected, value)
    reported = float(connected.health().fields["gripper_position_normalized"])
    assert abs(reported - value) < _GRIPPER_TOL
    data = connected._data  # the twin's live MjData: both fingers, coupled by the equality
    assert data is not None
    assert abs(float(data.qpos[6]) - float(data.qpos[7])) < 0.002


def test_gripper_leaves_the_arm_where_it_is(connected: MujocoArmHAL) -> None:
    """A gripper-only command holds the arm (the real HAL's separate mailboxes)."""
    before = list(connected.read_state().position)
    for _ in range(_ticks(connected)):
        _gripper(connected, 1.0)
    after = list(connected.read_state().position)
    assert np.allclose(before, after, atol=_TRACK_TOL_RAD)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_refuses_out_of_range_gripper_like_the_real_hal(
    connected: MujocoArmHAL, bad: float
) -> None:
    with pytest.raises(ROSConfigError, match=r"\[0, 1\]"):
        _gripper(connected, bad)


def test_refuses_a_seven_wide_joint_row(connected: MujocoArmHAL) -> None:
    """The gripper is not a joint: a row with a 7th entry is malformed, as on the real HAL."""
    with pytest.raises(ROSConfigError):
        connected.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[[*_ARM_TARGET, 1.0]])
        )


def test_front_and_wrist_cameras_render(connected: MujocoArmHAL) -> None:
    """Both manifest cameras render at their intrinsics from their sim placements."""
    frames = connected.read_images()
    for sensor in connected.description.sensors:
        assert sensor.intrinsics is not None
        frame = frames[sensor.name]
        assert frame.shape == (sensor.intrinsics.height, sensor.intrinsics.width, 3)
        assert frame.dtype == np.uint8
        assert float(frame.std()) > 5.0, f"{sensor.name} frame is blank"


@pytest.mark.parametrize("convention", ["normalized_close_symmetric", "width_meters"])
def test_refuses_a_gripper_convention_it_cannot_write(convention: str) -> None:
    """The twin writes [0, 1] opening only: any other declared convention is refused."""
    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / "galaxea_a1" / "robot.yaml"))
    (ee,) = desc.end_effectors
    update: dict[str, object] = {"command_convention": convention}
    if convention == "width_meters":
        update["command_range"] = (0.0, 0.06)
    flipped = desc.model_copy(update={"end_effectors": [ee.model_copy(update=update)]})
    with pytest.raises(ROSConfigError, match="normalized_open_unit"):
        MujocoArmHAL.from_description(flipped, gravity_enabled=True)


def test_derived_mjcf_loads_from_anywhere_under_a_relative_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative cache path still yields an MJCF that finds its meshes once moved."""
    import mujoco
    from openral_hal._galaxea_a1_assets import _derive_mjcf, ensure_a1_urdf
    from openral_hal._pinned_files import is_download_failure

    try:
        urdf = ensure_a1_urdf()
    except ROSConfigError as exc:
        # Only an unreachable download skips; a digest mismatch is a failure.
        if not is_download_failure(exc):
            raise
        pytest.skip(f"Galaxea A1 vendor files not fetchable on this host: {exc}")
    monkeypatch.chdir(urdf.parents[2])
    xml = _derive_mjcf(urdf.relative_to(urdf.parents[2]))
    elsewhere = tmp_path / "a1.xml"
    elsewhere.write_text(xml, encoding="utf-8")
    assert mujoco.MjModel.from_xml_path(str(elsewhere)).nmesh > 0
