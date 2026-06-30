"""Unit tests for the deploy-path-aware action-mode palette gate (ADR-0036).

Exercises the two module-level pure helpers in
:mod:`openral_reasoner_ros.reasoner_node`:

* :func:`_required_control_modes` — the :class:`ControlMode` s a skill's
  ``action_contract`` demands of the target robot.
* :func:`_action_executable` — whether the deploy path (``real`` vs
  ``sim``) can execute those modes.

All inputs are **real** fixtures (CLAUDE.md §1.11): real
``RobotDescription`` manifests from ``robots/`` and real
``RSkillManifest`` manifests from ``rskills/``. No mocks/stubs.

The ``reasoner_node`` module imports ``rclpy`` and ``openral_msgs`` at
import time, so both are required to import the pure helpers — guarded
below so a host without the ROS environment skips rather than errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_core import RobotDescription, RSkillManifest
from openral_core.schemas import ActionRepresentation, ControlMode
from openral_reasoner_ros.reasoner_node import (
    _action_executable,
    _required_control_modes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTS_DIR = REPO_ROOT / "robots"
RSKILLS_DIR = REPO_ROOT / "rskills"


def _franka() -> RobotDescription:
    """Real Franka-Panda description — supports only ``joint_position``."""
    path = ROBOTS_DIR / "franka_panda" / "robot.yaml"
    if not path.exists():
        pytest.skip(f"robot fixture missing: {path}")
    return RobotDescription.from_yaml(str(path))


def _aloha() -> RobotDescription:
    """Real bimanual ALOHA description — the act-aloha embodiment robot."""
    path = ROBOTS_DIR / "aloha_bimanual" / "robot.yaml"
    if not path.exists():
        pytest.skip(f"robot fixture missing: {path}")
    return RobotDescription.from_yaml(str(path))


def _pi05_libero() -> RSkillManifest:
    """Real pi05 LIBERO manifest (dim=7, no representation yet)."""
    path = RSKILLS_DIR / "pi05-libero-nf4" / "rskill.yaml"
    if not path.exists():
        pytest.skip(f"rskill fixture missing: {path}")
    return RSkillManifest.from_yaml(str(path))


def _act_aloha() -> RSkillManifest:
    """Real act-aloha manifest (dim=14, bare-dim joint)."""
    path = RSKILLS_DIR / "act-aloha" / "rskill.yaml"
    if not path.exists():
        pytest.skip(f"rskill fixture missing: {path}")
    return RSkillManifest.from_yaml(str(path))


def _panda_mobile() -> RobotDescription:
    """Real panda_mobile description — the robosuite-composite mobile manipulator."""
    path = ROBOTS_DIR / "panda_mobile" / "robot.yaml"
    if not path.exists():
        pytest.skip(f"robot fixture missing: {path}")
    return RobotDescription.from_yaml(str(path))


def _rldx_robocasa() -> RSkillManifest:
    """Real rldx robocasa manifest — slots incl. a ``composite_mode`` mux flag."""
    path = RSKILLS_DIR / "rldx1-ft-rc365-nf4" / "rskill.yaml"
    if not path.exists():
        pytest.skip(f"rskill fixture missing: {path}")
    return RSkillManifest.from_yaml(str(path))


def _cartesian_pi05() -> RSkillManifest:
    """pi05 LIBERO with an explicit cartesian representation set.

    Task 5 will declare this on disk; here we set it on the loaded real
    manifest so the gate's cartesian branch is exercised against a real
    fixture without a placeholder skill.
    """
    m = _pi05_libero()
    assert m.action_contract is not None
    return m.model_copy(
        update={
            "action_contract": m.action_contract.model_copy(
                update={"representation": ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER},
            ),
        },
    )


# ── _required_control_modes ────────────────────────────────────────────────


def test_required_modes_cartesian_representation() -> None:
    """A delta-EE+gripper representation requires cartesian-delta + gripper."""
    m = _cartesian_pi05()
    assert _required_control_modes(m) == {
        ControlMode.CARTESIAN_DELTA,
        ControlMode.GRIPPER_POSITION,
    }


def test_required_modes_bare_dim_joint() -> None:
    """A bare-dim action_contract (legacy, no representation/slots) → joint_position."""
    # act-aloha now declares `representation: joint_positions` on disk, so strip it
    # back to a legacy bare-dim contract to exercise the dim-only fallback path.
    base = _act_aloha()
    assert base.action_contract is not None
    bare_contract = base.action_contract.model_copy(update={"representation": None, "slots": None})
    m = base.model_copy(update={"action_contract": bare_contract})
    assert m.action_contract is not None
    assert m.action_contract.representation is None
    assert m.action_contract.slots is None
    assert _required_control_modes(m) == {ControlMode.JOINT_POSITION}


def test_required_modes_no_action_contract() -> None:
    """No action_contract → no action constraint (empty set)."""
    m = _act_aloha().model_copy(update={"action_contract": None})
    assert _required_control_modes(m) == set()


# ── _action_executable ──────────────────────────────────────────────────────


def test_cartesian_not_executable_on_real_joint_only_robot() -> None:
    """A cartesian skill is NOT executable on a real joint-only Franka."""
    m = _cartesian_pi05()
    franka = _franka()
    assert _action_executable(m, franka, "real") is False


def test_franka_manifest_declares_vision_for_libero_cameras() -> None:
    """Franka's LIBERO camera pair must surface as a vision-capable robot."""
    franka = _franka()
    assert franka.capabilities.has_vision is True
    rgb_names = [sensor.name for sensor in franka.sensors if sensor.modality == "rgb"]
    assert rgb_names == ["top", "wrist"]


def test_cartesian_executable_on_sim() -> None:
    """A cartesian skill IS executable in sim (robosuite OSC default set)."""
    m = _cartesian_pi05()
    franka = _franka()
    assert _action_executable(m, franka, "sim") is True


def test_joint_skill_executable_on_real_and_sim() -> None:
    """A bare-dim joint skill is executable on its robot under both modes."""
    m = _act_aloha()
    aloha = _aloha()
    assert _action_executable(m, aloha, "real") is True
    assert _action_executable(m, aloha, "sim") is True


def test_no_action_contract_always_executable() -> None:
    """A manifest with no action_contract is admitted (no constraint)."""
    m = _act_aloha().model_copy(update={"action_contract": None})
    franka = _franka()
    assert _action_executable(m, franka, "real") is True
    assert _action_executable(m, franka, "sim") is True


# ── composite_mode (ADR-0036 amendment 2026-06-04) ──────────────────────────


def test_required_modes_robocasa_composite_slots() -> None:
    """A RoboCasa slot contract requires the composite-controller mode set."""
    m = _rldx_robocasa()
    assert _required_control_modes(m) == {
        ControlMode.CARTESIAN_DELTA,
        ControlMode.GRIPPER_POSITION,
        ControlMode.JOINT_VELOCITY,
        ControlMode.COMPOSITE_MODE,
    }


def test_composite_skill_executable_on_sim() -> None:
    """The RoboCasa composite skill IS executable in sim.

    ``composite_mode`` is the sim robosuite-composite (HybridMobileBase)
    multiplexer the deploy path runs (ADR-0028d) — excluding it dropped
    pi05 / rldx robocasa VLAs at boot even though SimAttachedHAL executes
    them (the regression this amendment fixes).
    """
    m = _rldx_robocasa()
    panda_mobile = _panda_mobile()
    assert _action_executable(m, panda_mobile, "sim") is True


def test_composite_skill_not_executable_on_real_joint_robot() -> None:
    """``real`` still gates composite_mode: a joint-only robot can't execute it."""
    m = _rldx_robocasa()
    franka = _franka()
    assert _action_executable(m, franka, "real") is False
