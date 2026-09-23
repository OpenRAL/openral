"""Regression test — ``robots/<id>/robot.yaml`` is the runtime source of truth for every HAL.

``build_hal`` threads the loaded manifest into every HAL constructor (sim and
real), so ``sim:``, ``sensors:`` and ``collision_geometry:`` edits reach the
running HAL. ``test_build_hal_binds_the_loaded_manifest`` pins that for every
robot with a ``hal:`` entrypoint.

The ``*_DESCRIPTION`` constants in ``python/hal/src/openral_hal/`` survive
only as in-code **mirrors** of the manifests (tests and direct
``FrankaPandaHAL()``-style construction use them). The parametrised test
below checks each mirror against its manifest, so a joint-limit/payload/
safety-envelope bump in the YAML fails loudly until the mirror follows
(issues #54-58). The manifest wins; fix the constant, never the YAML.

UR5e/UR10e/Franka/Sawyer/ALOHA mirror their ``*_REAL_DESCRIPTION`` (derived
via ``openral_hal._real_description.make_real_description`` — only
``sdk_kind`` differs from the sim baseline). G1/H1/Rizon4/OpenArm/Anvil-v2
mirror their sim baseline. SO-100 and ``pusht_2d`` are out of scope.
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
def test_hal_constant_mirrors_robot_yaml(manifest_path: str, hal_constant_attr: str) -> None:
    """The in-code mirror constant must reproduce the YAML manifest (the source of truth)."""
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


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _hal_cases() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for manifest in sorted((_REPO_ROOT / "robots").glob("*/robot.yaml")):
        desc = RobotDescription.from_yaml(str(manifest))
        if desc.hal.sim is not None or desc.sim is not None:
            cases.append((manifest.parent.name, "sim"))
        if desc.hal.real is not None:
            cases.append((manifest.parent.name, "real"))
    return cases


@pytest.mark.parametrize(("robot_id", "mode"), _hal_cases())
def test_build_hal_binds_the_loaded_manifest(robot_id: str, mode: str) -> None:
    """``build_hal`` hands the HAL the loaded manifest itself, not an in-code constant."""
    from openral_core.exceptions import ROSConfigError
    from openral_hal import build_hal

    desc = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / robot_id / "robot.yaml"))
    try:
        hal = build_hal(desc, mode=mode)  # type: ignore[arg-type] # reason: parametrised literal
    except ROSConfigError as exc:
        if "not installed" in str(exc):
            pytest.skip(f"{robot_id} sim asset needs an optional dependency: {exc}")
        raise
    assert hal.description is desc
