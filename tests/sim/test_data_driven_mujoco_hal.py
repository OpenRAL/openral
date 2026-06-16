"""End-to-end manifest dispatch for ``MujocoArmHAL.from_description``.

Parametrises across every robot whose ``robots/<id>/robot.yaml`` carries a
``sim:`` block (ADR-0023).  For each one, loads the YAML through the real
:class:`openral_core.RobotDescription` validator, then builds the HAL
purely from the manifest via :meth:`MujocoArmHAL.from_description` (no
per-robot subclass), connects against the real MJCF resolved through
``robot_descriptions``, and asserts that ``read_state`` / ``send_action``
shapes line up with ``description.joints``.

Per CLAUDE.md §1.11 — no mocks.  When ``mujoco`` or ``robot_descriptions``
is unavailable, the suite is collected and skipped with a typed reason,
not faked.
"""

from __future__ import annotations

import importlib.util

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robot_descriptions")

from openral_core import (
    Action,
    ControlMode,
    RobotDescription,
)
from openral_hal._mujoco_arm import MujocoArmHAL

pytestmark = pytest.mark.sim

# ``aloha_bimanual`` resolves its MJCF through the ``gym_aloha:`` scheme, an
# optional sim-only dependency separate from ``robot_descriptions``. Skip just
# that parametrization when it is absent (CLAUDE.md §1.11) rather than erroring.
_GYM_ALOHA_MISSING = importlib.util.find_spec("gym_aloha") is None


# Robots whose robot.yaml carries a sim: block (ADR-0023 scope).
_MANIFEST_DRIVEN_ROBOTS = [
    "so100_follower",
    "franka_panda",
    "ur5e",
    "ur10e",
    "rizon4",
    "g1",
    "h1",
    pytest.param(
        "aloha_bimanual",  # bimanual: 2 grippers + mirror_actuator + keyframe
        marks=pytest.mark.skipif(_GYM_ALOHA_MISSING, reason="gym_aloha not installed"),
    ),
    "openarm",  # bimanual: 2 grippers + seed_ctrl_from_qpos
]


@pytest.fixture(params=_MANIFEST_DRIVEN_ROBOTS)
def robot_id(request: pytest.FixtureRequest) -> str:
    """One robot id per test case (see _MANIFEST_DRIVEN_ROBOTS)."""
    return request.param


@pytest.fixture
def description(robot_id: str) -> RobotDescription:
    """The real ``robots/<id>/robot.yaml`` for the parametrised robot."""
    from pathlib import Path

    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "robots").is_dir() and (ancestor / "pyproject.toml").is_file():
            repo_root = ancestor
            break
    else:
        raise RuntimeError("could not locate repo root from test file")
    return RobotDescription.from_yaml(str(repo_root / "robots" / robot_id / "robot.yaml"))


def test_manifest_has_sim_block(description: RobotDescription) -> None:
    """Every robot in scope ships a ``sim:`` section + a resolvable ``assets.mjcf``."""
    assert description.sim is not None
    assert description.assets.mjcf is not None
    assert description.assets.mjcf.startswith(
        ("rd:", "gym_aloha:", "openarm:", "menagerie:", "file:")
    ), f"unexpected mjcf ref scheme: {description.assets.mjcf!r}"


def test_from_description_round_trips(description: RobotDescription) -> None:
    """``MujocoArmHAL.from_description`` builds, connects, reads, sends, disconnects.

    Asserts the action / state width matches ``description.joints`` — the
    one invariant that catches any future off-by-one in the default
    qpos/qvel/actuator mapping logic (with or without floating-base
    offsets).
    """
    hal = MujocoArmHAL.from_description(description, gravity_enabled=False)
    hal.connect()
    try:
        state = hal.read_state()
        assert len(state.position) == len(description.joints)
        assert state.name == [j.name for j in description.joints]

        # send_action with the current pose — the HAL has no rate limit
        # on the sim path, so a single step is enough to round-trip.
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[list(state.position)],
            horizon=1,
            stamp_ns=state.stamp_ns,
        )
        hal.send_action(action)

        # State after one settle step should still match shape.
        state2 = hal.read_state()
        assert len(state2.position) == len(description.joints)
    finally:
        hal.disconnect()


def test_no_user_facing_python_required(robot_id: str) -> None:
    """The user-facing path is "load YAML → from_description → HAL".

    This is the contract ADR-0023 promises: no per-robot Python import is
    required to drive the HAL.  We exercise that path explicitly here
    (the parametrised ``description`` fixture above goes through the same
    seam, but the assertion is implicit in the round-trip — make it
    explicit by avoiding any ``openral_hal.<robot>`` import).
    """
    from pathlib import Path

    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "robots").is_dir() and (ancestor / "pyproject.toml").is_file():
            repo_root = ancestor
            break
    else:
        raise RuntimeError("could not locate repo root from test file")

    yaml_path = repo_root / "robots" / robot_id / "robot.yaml"
    description = RobotDescription.from_yaml(str(yaml_path))
    hal = MujocoArmHAL.from_description(description, gravity_enabled=False)
    hal.connect()
    state = hal.read_state()
    hal.disconnect()
    assert len(state.position) == len(description.joints)


def test_python_description_matches_yaml(robot_id: str, description: RobotDescription) -> None:
    """The hand-coded ``<ROBOT>_DESCRIPTION`` Python constant carries the same sim wiring.

    Drift between the Python-side constant and ``robots/<id>/robot.yaml``
    is a known footgun (cf. ``tests/unit/test_robot_manifests_match_hal_constants.py``).
    For the ADR-0023 scope, both surfaces must agree on the ``sim:``
    block; otherwise the manifest-driven path and the legacy
    subclass-driven path would silently diverge.
    """
    py_map = {
        "so100_follower": ("openral_hal.so100_follower", "SO100_DESCRIPTION"),
        "franka_panda": ("openral_hal.franka_panda", "FRANKA_PANDA_DESCRIPTION"),
        "ur5e": ("openral_hal.ur", "UR5e_DESCRIPTION"),
        "ur10e": ("openral_hal.ur", "UR10e_DESCRIPTION"),
        "rizon4": ("openral_hal.flexiv_rizon4", "RIZON4_DESCRIPTION"),
        "g1": ("openral_hal.g1", "G1_DESCRIPTION"),
        "h1": ("openral_hal.h1", "H1_DESCRIPTION"),
        "aloha_bimanual": ("openral_hal.aloha", "ALOHA_DESCRIPTION"),
        "openarm": ("openral_hal.openarm", "OPENARM_DESCRIPTION"),
    }
    if robot_id not in py_map:
        pytest.skip(f"no Python-side description for {robot_id!r}")
    module_name, attr = py_map[robot_id]
    import importlib

    module = importlib.import_module(module_name)
    py_desc: RobotDescription = getattr(module, attr)
    assert py_desc.sim is not None
    assert description.sim is not None
    # Compare the manifest-relevant fields field-by-field.  Equality on
    # the full SimDescription is too brittle (default fields appear as
    # explicit None in YAML, etc.); the wiring that matters is the MJCF
    # ref, the joint-index overrides, and the gripper configs.
    assert py_desc.assets.mjcf == description.assets.mjcf
    assert py_desc.sim.floating_base == description.sim.floating_base
    assert py_desc.sim.joint_qpos_addr == description.sim.joint_qpos_addr
    assert py_desc.sim.actuator_index == description.sim.actuator_index
    assert py_desc.sim.keyframe_index == description.sim.keyframe_index
    assert py_desc.sim.seed_ctrl_from_qpos == description.sim.seed_ctrl_from_qpos
    assert len(py_desc.sim.grippers) == len(description.sim.grippers)
    for py_g, yaml_g in zip(py_desc.sim.grippers, description.sim.grippers, strict=True):
        assert py_g.joint == yaml_g.joint
        assert py_g.ctrl_range == yaml_g.ctrl_range
        assert py_g.qpos_addrs == yaml_g.qpos_addrs
        assert py_g.qpos_scale == yaml_g.qpos_scale
        assert py_g.read_mode == yaml_g.read_mode
        assert py_g.write_mode == yaml_g.write_mode
        assert py_g.actuator_index == yaml_g.actuator_index
        assert py_g.mirror_actuator_index == yaml_g.mirror_actuator_index
