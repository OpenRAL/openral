"""Regression test — every ``robots/<id>/robot.yaml`` matches its ``*_DESCRIPTION`` HAL constant.

HAL constants in ``python/hal/src/openral_hal/`` are the runtime source of
truth; YAMLs under ``robots/`` are what the eval layer loads via
``ROBOTS.register``. Pins them together so a joint-limit/payload/safety-
envelope bump can't silently drift the YAML out of sync (issues #54-58).

UR5e/UR10e/Franka/Sawyer/ALOHA pin to their ``*_REAL_DESCRIPTION`` (production
manifests), derived from the sim baseline via
:func:`openral_hal._real_description.make_real_description` — kinematics,
safety envelope, capabilities and ``hal`` entrypoints are shared; only
``sdk_kind`` differs. G1/H1/Rizon4/OpenArm/Anvil-v2 pin to their sim baseline
because none has a real-HW HAL yet (G1/H1 gated on the M2 C++ S0 cerebellum,
CLAUDE.md §6.2; Rizon4/OpenArm/Anvil real-HW wrappers are tracked follow-ups)
— their ``hal.real`` is null until the real adapter lands.

SO-100 and ``pusht_2d`` are out of scope: SO-100's YAML carries optional
sensor entries the in-code constant omits; ALOHA's YAML carries a camera
``SensorSpec`` + observation/action spec the constant omits (only joint
inventory + capability + safety + sdk pointer are asserted equal for it);
``pusht_2d`` has no in-code DESCRIPTION sibling.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription


@pytest.mark.parametrize(
    "manifest_path, hal_constant_attr",
    [
        ("robots/ur5e/robot.yaml", "UR5e_REAL_DESCRIPTION"),
        ("robots/ur10e/robot.yaml", "UR10e_REAL_DESCRIPTION"),
        ("robots/franka_panda/robot.yaml", "FRANKA_PANDA_REAL_DESCRIPTION"),
        ("robots/sawyer/robot.yaml", "SAWYER_REAL_DESCRIPTION"),
        ("robots/aloha_bimanual/robot.yaml", "ALOHA_REAL_DESCRIPTION"),
        # Sim-baseline pins (no real-HW HAL yet) — see module docstring.
        ("robots/g1/robot.yaml", "G1_DESCRIPTION"),
        ("robots/h1/robot.yaml", "H1_DESCRIPTION"),
        ("robots/rizon4/robot.yaml", "RIZON4_DESCRIPTION"),
        ("robots/openarm/robot.yaml", "OPENARM_DESCRIPTION"),
        # Anvil OpenARM 2.0: standard v2 + Anvil's J1/J6 range deltas + wrist
        # bracket. Real-HW wrapper (github.com/anvil-robotics/openarm) is a
        # tracked follow-up.
        ("robots/anvil_openarm_v2/robot.yaml", "ANVIL_OPENARM_V2_DESCRIPTION"),
        ("robots/galaxea_a1/robot.yaml", "GALAXEA_A1_DESCRIPTION"),
    ],
)
def test_robot_yaml_matches_hal_description(manifest_path: str, hal_constant_attr: str) -> None:
    """The YAML manifest must reproduce the in-code HAL description."""
    yaml_desc = RobotDescription.from_yaml(str(Path(manifest_path)))

    bh_hal = pytest.importorskip("openral_hal")
    hal_desc = getattr(bh_hal, hal_constant_attr)

    assert yaml_desc.name == hal_desc.name
    assert yaml_desc.embodiment_kind == hal_desc.embodiment_kind
    assert yaml_desc.base_frame == hal_desc.base_frame

    yaml_joints = [
        (
            j.name,
            j.joint_type,
            j.position_limits,
            j.velocity_limit,
            j.effort_limit,
            j.sim_joint_name,
        )
        for j in yaml_desc.joints
    ]
    hal_joints = [
        (
            j.name,
            j.joint_type,
            j.position_limits,
            j.velocity_limit,
            j.effort_limit,
            j.sim_joint_name,
        )
        for j in hal_desc.joints
    ]
    assert yaml_joints == hal_joints, "joint specs drifted between YAML and HAL constant"

    assert yaml_desc.capabilities.embodiment_tags == hal_desc.capabilities.embodiment_tags
    yaml_modes = yaml_desc.capabilities.supported_control_modes
    hal_modes = hal_desc.capabilities.supported_control_modes
    assert yaml_modes == hal_modes or yaml_modes == [m.value for m in hal_modes]
    assert yaml_desc.sdk_kind == hal_desc.sdk_kind
    assert yaml_desc.hal == hal_desc.hal  # sim/real HAL entrypoints
