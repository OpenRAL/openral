"""ADR-0024 — pin that ``build_tool_palette`` does not filter on ``kind``.

The Reasoner's tool palette filter at
``python/reasoner/src/openral_reasoner/palette.py`` consults only
``role``, ``embodiment_tags``, ``capabilities_required``, and the license
posture. A wrapped-ROS rSkill (``kind: ros_action`` / ``ros_service``)
with ``role: s1`` and a matching embodiment MUST surface as one
``execute_rskill__<slug>`` LLM tool, exactly the same way a VLA does.
This test pins that behaviour so a future change can't accidentally
start gating on ``kind`` or ``model_family``.

No mocks (CLAUDE.md §1.11) — drives the real in-tree manifests through
the real palette builder.
"""

from __future__ import annotations

from pathlib import Path

from openral_core import RobotCapabilities, RSkillAction, RSkillManifest
from openral_reasoner.palette import build_tool_palette

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RSKILLS_DIR = _REPO_ROOT / "rskills"


def _load_intree() -> list[RSkillManifest]:
    return [RSkillManifest.from_yaml(str(p)) for p in sorted(_RSKILLS_DIR.glob("*/rskill.yaml"))]


def test_franka_panda_palette_surfaces_wrapped_moveit_skill() -> None:
    """The MoveIt wrapper (kind: ros_action) lists ``franka_panda`` in its
    embodiment_tags, so a Franka palette must include it."""
    caps = RobotCapabilities(embodiment_tags=["franka_panda"])
    palette = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=caps)
    ids = {s.rskill_id for s in palette.skills}
    assert "OpenRAL/rskill-moveit-joints" in ids, (
        f"expected wrapped-MoveIt rSkill in Franka palette, got {sorted(ids)!r}"
    )
    moveit = next(s for s in palette.skills if s.rskill_id == "OpenRAL/rskill-moveit-joints")
    assert moveit.actions == (RSkillAction.REACH,)
    assert "MoveIt" in moveit.description


def test_mobile_base_palette_surfaces_wrapped_nav2_skill() -> None:
    """The Nav2 wrapper (kind: ros_action, result-only mode) targets the
    ``mobile_base`` class tag, so any robot declaring it surfaces the
    skill — the rSkill is intentionally generic across all planar-base
    embodiments, not panda_mobile-specific."""
    caps = RobotCapabilities(embodiment_tags=["mobile_base"])
    palette = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=caps)
    ids = {s.rskill_id for s in palette.skills}
    assert "OpenRAL/rskill-nav2-navigate-to-pose" in ids, (
        f"expected wrapped-Nav2 rSkill in mobile_base palette, got {sorted(ids)!r}"
    )
    nav2 = next(s for s in palette.skills if s.rskill_id == "OpenRAL/rskill-nav2-navigate-to-pose")
    assert nav2.actions == (RSkillAction.NAVIGATE,)


def test_panda_mobile_robot_yaml_surfaces_wrapped_nav2_skill() -> None:
    """End-to-end: a robot's ACTUAL ``robots/<id>/robot.yaml`` capabilities
    must surface Nav2 once the robot declares the ``mobile_base`` class
    tag. Catches a regression where a future PR removes ``mobile_base``
    from ``robots/panda_mobile/robot.yaml`` and silently drops Nav2 from
    the deploy_sim palette."""
    from openral_core import RobotDescription

    description = RobotDescription.from_yaml(
        str(_REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml")
    )
    palette = build_tool_palette(
        installed_skills=_load_intree(),
        robot_capabilities=description.capabilities,
    )
    ids = {s.rskill_id for s in palette.skills}
    assert "OpenRAL/rskill-nav2-navigate-to-pose" in ids, (
        f"panda_mobile robot.yaml must declare 'mobile_base' so Nav2 surfaces; got {sorted(ids)!r}"
    )


def test_palette_does_not_distinguish_vla_from_ros_wrapper() -> None:
    """Both kinds surface in the same palette for matching embodiments.

    Catches a regression that adds a ``manifest.kind`` predicate to
    ``build_tool_palette``: any such change would silently drop one
    side or the other.
    """
    caps = RobotCapabilities(embodiment_tags=["franka_panda"])
    palette = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=caps)
    ids = {s.rskill_id for s in palette.skills}
    # MoveIt wrapper (kind: ros_action) AND any Franka VLA must both
    # appear. We don't pin which VLA — just that at least one VLA
    # surfaces alongside the wrapper.
    assert "OpenRAL/rskill-moveit-joints" in ids
    vla_ids = ids - {"OpenRAL/rskill-moveit-joints", "OpenRAL/rskill-nav2-navigate-to-pose"}
    assert vla_ids, (
        "expected at least one VLA in the Franka palette alongside the MoveIt wrapper; "
        f"got only wrapped skills: {sorted(ids)!r}"
    )
