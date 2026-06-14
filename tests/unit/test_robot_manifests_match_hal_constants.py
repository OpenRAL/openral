"""Regression test — every ``robots/<id>/robot.yaml`` matches its in-code
``*_DESCRIPTION`` HAL constant, where one exists.

The HAL constants in ``python/hal/src/openral_hal/`` are the runtime
source of truth; the YAMLs under ``robots/`` are what the eval layer loads
via ``ROBOTS.register``. This test pins them together so a future bump to
joint limits / payload / safety envelope can't silently drift the YAML out
of sync with the HAL.

Coverage
--------
- ``robots/ur5e/robot.yaml``           ↔ ``UR5e_REAL_DESCRIPTION``      (HAL `ur_real.py`)
- ``robots/ur10e/robot.yaml``          ↔ ``UR10e_REAL_DESCRIPTION``     (HAL `ur_real.py`)
- ``robots/franka_panda/robot.yaml``   ↔ ``FRANKA_PANDA_REAL_DESCRIPTION``
- ``robots/sawyer/robot.yaml``         ↔ ``SAWYER_REAL_DESCRIPTION``
- ``robots/aloha_bimanual/robot.yaml`` ↔ ``ALOHA_REAL_DESCRIPTION``
- ``robots/g1/robot.yaml``             ↔ ``G1_DESCRIPTION``             (HAL `g1.py`)
- ``robots/h1/robot.yaml``             ↔ ``H1_DESCRIPTION``             (HAL `h1.py`)
- ``robots/rizon4/robot.yaml``         ↔ ``RIZON4_DESCRIPTION``         (HAL `flexiv_rizon4.py`)
- ``robots/openarm/robot.yaml``        ↔ ``OPENARM_DESCRIPTION``        (HAL `openarm.py`)

Most covered YAMLs pin to the **real-hardware** ``*_REAL_DESCRIPTION``
constant because those YAMLs are the production-deployment manifests.
The Franka / Sawyer / ALOHA real descriptions are derived from their sim
baselines (``FRANKA_PANDA_DESCRIPTION`` / ``SAWYER_DESCRIPTION`` /
``ALOHA_DESCRIPTION``) via
:func:`openral_hal._real_description.make_real_description`. The UR
real descriptions (``UR5e_REAL_DESCRIPTION`` / ``UR10e_REAL_DESCRIPTION``
in ``ur_real.py``) follow the same pattern, derived from
``UR5e_DESCRIPTION`` / ``UR10e_DESCRIPTION`` in ``ur.py``. In every case
kinematics + safety envelope + capabilities + ``hal`` entrypoints are shared
between the sim baseline and the real-HW description; only ``sdk_kind``
differs (ADR-0031). The sim baselines stay in-tree as the manifest the MuJoCo /
gym sim adapter loads when available. Issues #54–#58.

The G1, H1, Rizon 4, and OpenArm all pin to their sim baselines
(``G1_DESCRIPTION`` / ``H1_DESCRIPTION`` / ``RIZON4_DESCRIPTION`` /
``OPENARM_DESCRIPTION``) because none of them have a real-HW HAL yet:
the G1 / H1 wait on the M2 C++ S0 cerebellum (CLAUDE.md §6.2); the
Rizon 4 real-HW wrapper around ``flexiv_rdk`` and the OpenArm real-HW
wrapper around lerobot's upstream driver are tracked as follow-ups.
All four YAMLs' ``hal.sim`` points at the sim HAL and ``hal.real`` is null
until the real adapter lands (ADR-0031).

The SO-100 manifest is skipped here because its YAML carries optional
sensor entries that the in-code constant deliberately omits (the HAL
constant is only the kinematic core; sensor wiring is opt-in via
``so100_with_sensors``).  The ALOHA YAML similarly carries a top-down
camera ``SensorSpec`` and an ``observation_spec`` / ``action_spec`` block
the in-code constant omits (sensor / sim-IO wiring is owned by the eval
adapter); only the joint inventory + capability + safety + sdk pointer
is asserted equal.  ``pusht_2d`` is a sim-pseudo manifest with no in-code
DESCRIPTION sibling and is therefore not in scope for this guard.
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
        # G1 + H1 + Rizon 4 + OpenArm all pin to their sim baselines
        # because none of them has a real-HW HAL yet.  G1 / H1 are
        # gated on the M2 C++ S0 cerebellum (CLAUDE.md §6.2); Rizon 4
        # awaits a wrapper around ``flexiv_rdk``; OpenArm awaits a
        # wrapper around lerobot's upstream OpenArm driver.  All four
        # YAMLs' hal.sim points at their respective sim HAL and hal.real
        # is null until the real adapter lands (ADR-0031).
        ("robots/g1/robot.yaml", "G1_DESCRIPTION"),
        ("robots/h1/robot.yaml", "H1_DESCRIPTION"),
        ("robots/rizon4/robot.yaml", "RIZON4_DESCRIPTION"),
        ("robots/openarm/robot.yaml", "OPENARM_DESCRIPTION"),
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
    assert yaml_desc.hal == hal_desc.hal  # ADR-0031 sim/real HAL entrypoints
