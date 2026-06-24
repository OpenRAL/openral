"""ADR-0071 Phase 2 — the warn-only TaskSpace shadow gate.

Exercises ``openral_reasoner.palette.task_space_disagreement`` (the pure helper
the reasoner deploy palette and ``rskill_publisher`` both call) against real
``robots/`` + ``rskills/`` fixtures — no mocks (CLAUDE.md §1.11). The helper is
deliberately ROS-free so it runs without ``rclpy``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from openral_core import RobotDescription, RSkillManifest
from openral_reasoner.palette import task_space_disagreement
from structlog.testing import capture_logs

REPO_ROOT = Path(__file__).resolve().parents[2]

# tools/ is not an installed package — load the publisher module by path.
_spec = importlib.util.spec_from_file_location(
    "rskill_publisher", REPO_ROOT / "tools" / "rskill_publisher.py"
)
assert _spec is not None and _spec.loader is not None
rskill_publisher = importlib.util.module_from_spec(_spec)
sys.modules["rskill_publisher"] = rskill_publisher
_spec.loader.exec_module(rskill_publisher)


def _robot(name: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(REPO_ROOT / "robots" / name / "robot.yaml"))


def _rskill(name: str) -> RSkillManifest:
    return RSkillManifest.from_yaml(str(REPO_ROOT / "rskills" / name / "rskill.yaml"))


def test_no_warning_when_gate_agrees_with_legacy() -> None:
    """act-libero IS sim-executable on franka; passing legacy_ok=True agrees."""
    robot = _robot("franka_panda")
    skill = _rskill("act-libero")
    assert task_space_disagreement(skill, robot, "sim", legacy_ok=True) is None


def test_warning_when_gate_disagrees_with_legacy() -> None:
    """If the legacy verdict claims incompatible but the canonical gate says
    sim-executable, the helper surfaces the disagreement (warn-only)."""
    robot = _robot("franka_panda")
    skill = _rskill("act-libero")
    msg = task_space_disagreement(skill, robot, "sim", legacy_ok=False)
    assert msg is not None
    assert "act-libero" in msg
    assert "franka_panda" in msg
    assert "Warn-only" in msg


def test_real_mode_disagreement_surfaces_reasons() -> None:
    """In real mode act-libero is NOT executable on a joint-only Franka, so a
    legacy_ok=True (e.g. an embodiment-tag-only match) disagrees and the reasons
    name the missing cartesian/gripper modes."""
    robot = _robot("franka_panda")
    skill = _rskill("act-libero")
    msg = task_space_disagreement(skill, robot, "real", legacy_ok=True)
    assert msg is not None
    assert "cartesian_delta" in msg


def test_non_actuating_skill_never_warns() -> None:
    """Detectors / VLMs / rewards carry no action_contract → always None."""
    robot = _robot("franka_panda")
    for name in ("omdet-turbo-indoor", "qwen35-4b-nf4", "robometer-4b"):
        skill = _rskill(name)
        assert skill.action_contract is None
        assert task_space_disagreement(skill, robot, "sim", legacy_ok=True) is None
        assert task_space_disagreement(skill, robot, "real", legacy_ok=False) is None


def test_rc365_sim_agrees_after_phase1_fix() -> None:
    """The Phase-1 EE-name fix means rc365 IS sim-executable on panda_mobile, so
    a legacy_ok=True produces no disagreement warning."""
    robot = _robot("panda_mobile")
    skill = _rskill("rldx1-ft-rc365-nf4")
    assert task_space_disagreement(skill, robot, "sim", legacy_ok=True) is None


# ── Publisher gate (Phase 2b) ───────────────────────────────────────────────


def test_publisher_no_warning_for_compatible_skill() -> None:
    """act-libero is sim-executable on franka → no incompatibility warning."""
    skill_dir = REPO_ROOT / "rskills" / "act-libero"
    manifest = RSkillManifest.from_yaml(str(skill_dir / "rskill.yaml"))
    with capture_logs() as logs:
        rskill_publisher._validate_task_space(manifest, skill_dir)
    incompat = [e for e in logs if e.get("event") == "rskill_publisher.task_space_incompatible"]
    assert incompat == []


def test_publisher_warns_on_sim_incompatible_skill() -> None:
    """3d-diffuser-actor-rlbench emits cartesian_pose (not default-sim
    executable) on its franka target → publisher warns (warn-only)."""
    skill_dir = REPO_ROOT / "rskills" / "3d-diffuser-actor-rlbench"
    manifest = RSkillManifest.from_yaml(str(skill_dir / "rskill.yaml"))
    with capture_logs() as logs:
        rskill_publisher._validate_task_space(manifest, skill_dir)
    warns = [e for e in logs if e.get("event") == "rskill_publisher.task_space_incompatible"]
    assert any(e.get("robot") == "franka_panda" for e in warns)


def test_publisher_skips_non_actuating_skill() -> None:
    """A detector carries no action_contract → the gate returns before logging."""
    skill_dir = REPO_ROOT / "rskills" / "omdet-turbo-indoor"
    manifest = RSkillManifest.from_yaml(str(skill_dir / "rskill.yaml"))
    with capture_logs() as logs:
        rskill_publisher._validate_task_space(manifest, skill_dir)
    assert logs == []
