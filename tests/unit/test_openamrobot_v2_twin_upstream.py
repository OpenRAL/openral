"""The OpenAMRobot 2.0 twin carries upstream's masses and renders its visual meshes.

Real vendored URDF and the real MJCF builder (``openral_hal._openamrobot_v2_assets``).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

mujoco = pytest.importorskip("mujoco", reason="needs mujoco (sim group) for the twin")

from openral_hal._openamrobot_v2_assets import (  # noqa: E402
    _BASE_OMEGA_RAD_S,
    OPENAMROBOT_V2_URDF,
    ensure_openamrobot_v2_mjcf,
)


def test_urdf_carries_upstreams_whole_robot_mass() -> None:
    """openamr-platform-sw #126: base_link 49.948 kg, whole robot 87.11 kg (+0.1 kg root)."""
    links = ET.parse(OPENAMROBOT_V2_URDF).getroot().findall("link")
    mass = {
        link.get("name"): float(m.attrib["value"])
        for link in links
        if (m := link.find("inertial/mass")) is not None
    }
    assert mass["base_link"] == pytest.approx(49.948)
    assert sum(mass.values()) == pytest.approx(87.11 + 0.1, abs=0.15)


@pytest.fixture(scope="module")
def model() -> object:
    return mujoco.MjModel.from_xml_path(ensure_openamrobot_v2_mjcf())


def test_collision_shells_are_not_rendered(model: object) -> None:
    """Default-visible groups (0-2) hold only visual geoms; collision lives in group 3."""
    for g in range(model.ngeom):  # type: ignore[attr-defined]  # reason: MjModel
        collides = model.geom_contype[g] or model.geom_conaffinity[g]  # type: ignore[attr-defined]  # reason: MjModel
        if collides:
            assert int(model.geom_group[g]) == 3  # type: ignore[attr-defined]  # reason: MjModel


def test_base_is_critically_damped_at_the_compiled_mass(model: object) -> None:
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_footprint")
    mass = float(model.body_subtreemass[base])  # type: ignore[attr-defined]  # reason: MjModel
    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_x_ctrl")
    kp = float(model.actuator_gainprm[act, 0])  # type: ignore[attr-defined]  # reason: MjModel
    kv = -float(model.actuator_biasprm[act, 2])  # type: ignore[attr-defined]  # reason: MjModel
    assert kp == pytest.approx(mass * _BASE_OMEGA_RAD_S**2)
    assert kv / (2.0 * math.sqrt(kp * mass)) == pytest.approx(1.0)
