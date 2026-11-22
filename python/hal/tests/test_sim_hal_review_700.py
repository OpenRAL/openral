"""openral#700 review fixes on the sim HAL shell, against a real native-MuJoCo arm.

- Item 1: a stop drops the setpoints in flight (``stop_stale_motion``): the arm's
  ctrl targets its present pose, the held base twist is gone.
- Item 9: ``reset_to_pose`` is refused while the e-stop is latched.
- Item 7: the viewer guard needs a display GLFW can open, not just a variable.

Native MuJoCo, no mocks (CLAUDE.md §1.11); skips when the MJCF cannot be fetched.
"""

from __future__ import annotations

import pytest
from openral_core import Action, ControlMode
from openral_core.exceptions import ROSConfigError, ROSEStopRequested


def _franka() -> object:
    pytest.importorskip("mujoco")
    from _twin_assets import skip_if_unavailable
    from openral_hal.franka_panda import FrankaPandaHAL

    hal = FrankaPandaHAL(gravity_enabled=False)
    try:
        hal.connect()
    except ROSConfigError as exc:  # only an unavailable asset skips; a broken twin fails
        skip_if_unavailable(exc, "franka_panda twin")
    return hal


def test_a_stop_drops_the_arm_target_in_flight() -> None:
    hal = _franka()
    try:
        state = hal.read_state()  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
        target = list(state.position)
        target[0] += 0.5  # a move in flight: one action does not reach it
        hal.send_action(  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
            Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[target])
        )
        _, data = hal.mujoco_handles()  # type: ignore[attr-defined]  # reason: connected twin
        twin = hal._twin  # type: ignore[attr-defined]  # reason: the twin's own indices
        act = twin.actuator_index[twin.joint_names[0]]
        addr = twin.joint_qpos_addr[twin.joint_names[0]]
        assert float(data.ctrl[act]) == pytest.approx(target[0])
        hal._held_twist = (0.4, 0.0, 0.0)  # type: ignore[attr-defined]  # a held /cmd_vel
        hal.stop_stale_motion()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
        assert float(data.ctrl[act]) == pytest.approx(float(data.qpos[addr]))
        assert hal._held_twist is None  # type: ignore[attr-defined]
        assert hal.base_twist == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore[attr-defined]
    finally:
        hal.disconnect()  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface


def test_reset_to_pose_is_refused_while_latched() -> None:
    hal = _franka()
    pose = list(hal.read_state().position)  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
    with pytest.raises(ROSEStopRequested):
        hal.estop()  # type: ignore[attr-defined]  # reason: the raising stop
    hal.connect()  # type: ignore[attr-defined]  # a fresh twin, still latched
    try:
        with pytest.raises(ROSEStopRequested, match="latched"):
            hal.reset_to_pose(pose)  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
        hal.reset_estop()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
        hal.reset_to_pose(pose)  # type: ignore[attr-defined]  # cleared: allowed again
    finally:
        hal.disconnect()  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface


def test_the_viewer_needs_a_display_glfw_can_open(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mujoco")
    from openral_hal.sim_sensor_bridge import glfw_display_usable

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    glfw_display_usable.cache_clear()
    assert glfw_display_usable() is False
    monkeypatch.setenv("DISPLAY", ":97531")  # set, but no X server answers it
    glfw_display_usable.cache_clear()
    assert glfw_display_usable() is False
    glfw_display_usable.cache_clear()
