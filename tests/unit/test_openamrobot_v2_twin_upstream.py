"""The OpenAMRobot 2.0 twin carries upstream's masses and renders its visual meshes.

Real vendored URDF and the real MJCF builder (``openral_hal._openamrobot_v2_assets``).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import pytest

mujoco = pytest.importorskip("mujoco", reason="needs mujoco (sim group) for the twin")

from openral_hal._openamrobot_v2_assets import (  # noqa: E402
    BASE_Z_JOINT,
    OPENAMROBOT_V2_URDF,
    WHEEL_JOINTS,
    diff_drive_accel_limits,
    ensure_openamrobot_v2_mjcf,
)


def test_urdf_carries_upstreams_whole_robot_mass() -> None:
    """openamr-platform-sw #126: base_link 49.948 kg, whole robot 87.11 kg (massless root)."""
    links = ET.parse(OPENAMROBOT_V2_URDF).getroot().findall("link")
    mass = {
        link.get("name"): float(m.attrib["value"])
        for link in links
        if (m := link.find("inertial/mass")) is not None
    }
    assert mass["base_link"] == pytest.approx(49.948)
    assert sum(mass.values()) == pytest.approx(87.11, abs=0.05)


@pytest.fixture(scope="module")
def model() -> object:
    return mujoco.MjModel.from_xml_path(ensure_openamrobot_v2_mjcf())


def test_collision_shells_are_not_rendered(model: object) -> None:
    """Default-visible groups (0-2) hold only the robot's visual geoms; its collision is group 3.

    World geoms (the floor the wheels roll on) are drawn and collide.
    """
    for g in range(model.ngeom):  # type: ignore[attr-defined]  # reason: MjModel
        if int(model.geom_bodyid[g]) == 0:  # type: ignore[attr-defined]  # reason: MjModel
            continue
        collides = model.geom_contype[g] or model.geom_conaffinity[g]  # type: ignore[attr-defined]  # reason: MjModel
        if collides:
            assert int(model.geom_group[g]) == 3  # type: ignore[attr-defined]  # reason: MjModel


def test_the_base_rolls_on_upstreams_wheels(model: object) -> None:
    """No base joint is actuated; the two drive wheels are, on an integrated-velocity servo.

    The floor collides with the robot, the casters are frictionless, and everything on
    the lift carriage is gravity-compensated (the wheels carry the base and column).
    """
    m: Any = model
    targets = {m.joint(int(m.actuator_trnid[a, 0])).name for a in range(m.nu)}
    assert not targets & {"base_x", "base_y", "base_yaw", BASE_Z_JOINT}
    for wheel in WHEEL_JOINTS:
        act = m.actuator(f"{wheel}_vel")
        assert int(m.actuator_dyntype[act.id]) == mujoco.mjtDyn.mjDYN_INTEGRATOR
    floor = m.geom("floor")
    chassis = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name == "left_wheel_link"]
    assert any(
        (m.geom_contype[g] & floor.conaffinity) or (floor.contype & m.geom_conaffinity[g])
        for g in chassis
    )
    caster = [
        g
        for g in range(m.ngeom)
        if m.body(int(m.geom_bodyid[g])).name == "fl_wheel_link" and m.geom_contype[g]
    ]
    assert caster and all(float(m.geom_friction[g][0]) == 0.0 for g in caster)
    assert float(m.body_gravcomp[m.body("lift_carriage_link").id]) == 1.0
    assert float(m.body_gravcomp[m.body("base_link").id]) == 0.0


def test_the_wheel_ramp_is_upstreams_diff_drive() -> None:
    """The twin ramps by the DiffDrive limits in upstream's own (unmodified) URDF."""
    assert diff_drive_accel_limits() == (0.5, 1.0)
