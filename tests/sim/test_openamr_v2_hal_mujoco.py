"""Sim tests for ``openral_hal.OpenAMRv2MujocoHAL`` against real MuJoCo physics.

The twin is built from the vendored ``robots/openamr_v2/openamr_v2.urdf``
(meshes from the pinned openarm_description / openamr-platform-sw clones) and
driven through ``build_hal``, as ``deploy sim`` does. No mocks: the base,
lift, arms and grippers move under the MJCF's own position actuators.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

try:
    import mujoco  # noqa: F401
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

from openral_core import Action, ControlMode, RobotDescription, ROSConfigError
from openral_hal import build_hal
from openral_hal.openamr_v2 import OpenAMRv2MujocoHAL

_MANIFEST = Path(__file__).resolve().parents[2] / "robots" / "openamr_v2" / "robot.yaml"

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
]


@pytest.fixture(scope="module")
def description() -> RobotDescription:
    return RobotDescription.from_yaml(str(_MANIFEST))


@pytest.fixture
def hal(description: RobotDescription) -> Iterator[OpenAMRv2MujocoHAL]:
    try:
        hal = build_hal(description, mode="sim")
        assert isinstance(hal, OpenAMRv2MujocoHAL)
        hal.connect()
    except ROSConfigError as exc:  # pinned mesh clone unreachable (offline host)
        pytest.skip(f"openamr_v2 twin unavailable: {exc}")
    yield hal
    hal.disconnect()


def _twist(vx: float, wz: float, vy: float = 0.0) -> Action:
    return Action(
        control_mode=ControlMode.BODY_TWIST, horizon=1, body_twist=[(vx, vy, 0.0, 0.0, 0.0, wz)]
    )


def test_body_twist_drives_the_base_on_an_arc(hal: OpenAMRv2MujocoHAL) -> None:
    """40 x 0.05 s of (0.2 m/s, 0.5 rad/s) lands on the exact arc."""
    for _ in range(40):
        hal.send_action(_twist(0.2, 0.5))
    x, y, yaw = hal.base_pose
    t = 2.0
    assert yaw == pytest.approx(0.5 * t, abs=1e-6)
    assert x == pytest.approx(0.2 / 0.5 * math.sin(0.5 * t), abs=1e-4)
    assert y == pytest.approx(0.2 / 0.5 * (1 - math.cos(0.5 * t)), abs=1e-4)
    assert hal.base_twist == (0.2, 0.0, 0.0, 0.0, 0.0, 0.5)


def test_lateral_twist_is_refused_on_a_differential_base(hal: OpenAMRv2MujocoHAL) -> None:
    with pytest.raises(ROSConfigError, match="differential"):
        hal.send_action(_twist(0.0, 0.0, vy=0.1))
    assert hal.base_pose == (0.0, 0.0, 0.0)


def test_joint_targets_converge_and_the_lift_holds(hal: OpenAMRv2MujocoHAL) -> None:
    """Lift to 0.30 m, bend both arms, open both grippers; the base stays put."""
    names = [j.name for j in hal.description.joints]
    target = list(hal.read_state().position)
    for name, value in {
        "lift_joint": 0.30,
        "left_joint2": -0.5,
        "left_joint4": 1.0,
        "right_joint2": 0.5,
        "right_joint4": 1.0,
        "left_gripper": 0.6,
        "right_gripper": -0.6,
    }.items():
        target[names.index(name)] = value
    action = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[target])
    for _ in range(300):  # 10 s at 30 Hz
        hal.send_action(action)
    reached = hal.read_state().position
    for name, want, got in zip(names, target, reached, strict=True):
        assert got == pytest.approx(want, abs=5e-3), name


def test_lift_holds_under_gravity(description: RobotDescription) -> None:
    """With gravity on, the stiff lift drive holds 0.35 m within 3 mm under both arms."""
    hal = build_hal(description, mode="sim", transport={"gravity_enabled": True})
    try:
        hal.connect()
    except ROSConfigError as exc:
        pytest.skip(f"openamr_v2 twin unavailable: {exc}")
    try:
        names = [j.name for j in description.joints]
        target = list(hal.read_state().position)
        target[names.index("lift_joint")] = 0.35
        action = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[target])
        for _ in range(300):
            hal.send_action(action)
        assert hal.read_state().position[names.index("lift_joint")] == pytest.approx(0.35, abs=3e-3)
    finally:
        hal.disconnect()


def test_every_manifest_camera_renders(hal: OpenAMRv2MujocoHAL) -> None:
    images = hal.read_images()
    for sensor in hal.description.sensors:
        if sensor.modality == "rgb":
            assert sensor.intrinsics is not None
            image = images[sensor.name]
            assert isinstance(image, np.ndarray)
            assert image.shape == (
                sensor.intrinsics.height,
                sensor.intrinsics.width,
                3,
            )
