"""ADR-0024 — schema-side cross-validators on ``RSkillManifest.kind``.

Pins the per-kind shape of the manifest so a regression that loosens any
of these rules surfaces here instead of as a silent VLA-on-wrapper /
wrapper-on-VLA bug downstream.

No mocks (CLAUDE.md §1.11) — every test constructs a real
:class:`~openral_core.RSkillManifest` against a real
:class:`~openral_core.RosIntegration` literal that matches the on-disk
shape of ``rskills/rskill-moveit-joints/rskill.yaml``.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from openral_core import (
    ActuatorRequirement,
    ControlMode,
    ControlModeSemantics,
    RosIntegration,
    RSkillAction,
    RSkillLatencyBudget,
    RSkillLicensePosture,
    RSkillManifest,
)
from pydantic import ValidationError

_VALID_ACTUATORS = [
    ActuatorRequirement(
        kind=ControlMode.JOINT_POSITION,
        control_mode_semantics=ControlModeSemantics(mode="absolute"),
    ),
]
_VALID_BUDGET = RSkillLatencyBudget(per_chunk_ms=100.0)
_VALID_ROS_INTEGRATION = RosIntegration(
    package="moveit_msgs",
    interface_type="MoveGroup",
    interface_name="/move_action",
    result_trajectory_field="planned_trajectory.joint_trajectory",
    default_goal_json=json.dumps({"request": {"group_name": "panda_arm"}}),
    ros_dependencies=["ros-jazzy-moveit"],
)


def _vla_kwargs(**overrides: object) -> dict[str, object]:
    """Build a minimum-valid ``kind: vla`` manifest kwargs dict."""
    kwargs: dict[str, object] = {
        "name": "openral/rskill-test-vla",
        "version": "0.1.0",
        "license": RSkillLicensePosture.APACHE_2_0,
        "role": "s1",
        "kind": "vla",
        "model_family": "act",
        "embodiment_tags": ["franka_panda"],
        "actuators_required": list(_VALID_ACTUATORS),
        "weights_uri": "hf://example/test",
        "chunk_size": 4,
        "latency_budget": _VALID_BUDGET,
        "description": "Test skill — pick a cube from the tabletop.",
        "actions": [RSkillAction.PICK],
    }
    kwargs.update(overrides)
    return kwargs


def _ros_action_kwargs(**overrides: object) -> dict[str, object]:
    """Build a minimum-valid ``kind: ros_action`` manifest kwargs dict."""
    kwargs: dict[str, object] = {
        "name": "openral/rskill-test-ros-action",
        "version": "0.1.0",
        "license": RSkillLicensePosture.APACHE_2_0,
        "role": "s1",
        "kind": "ros_action",
        "embodiment_tags": ["franka_panda"],
        "actuators_required": list(_VALID_ACTUATORS),
        "chunk_size": 1,
        "latency_budget": _VALID_BUDGET,
        "description": "Plan and execute via MoveIt's MoveGroup.",
        "actions": [RSkillAction.REACH],
        "ros_integration": _VALID_ROS_INTEGRATION,
    }
    kwargs.update(overrides)
    return kwargs


# ── kind == "vla" ───────────────────────────────────────────────────────────


def test_vla_kind_requires_model_family() -> None:
    with pytest.raises(ValidationError, match="model_family"):
        RSkillManifest(**_vla_kwargs(model_family=None))


def test_vla_kind_requires_weights_uri() -> None:
    with pytest.raises(ValidationError, match="weights_uri"):
        RSkillManifest(**_vla_kwargs(weights_uri=None))


def test_vla_kind_forbids_ros_integration() -> None:
    with pytest.raises(ValidationError, match="ros_integration"):
        RSkillManifest(**_vla_kwargs(ros_integration=_VALID_ROS_INTEGRATION))


def test_vla_kind_happy_path() -> None:
    m = RSkillManifest(**_vla_kwargs())
    assert m.kind == "vla"
    assert m.model_family == "act"
    assert m.weights_uri == "hf://example/test"
    assert m.ros_integration is None


# ── embodiment_tags: kind-aware presence ──────────────────────────────────────


def test_non_perception_kind_requires_embodiment_tag() -> None:
    """A vla / ros-wrapper manifest must still declare >=1 embodiment tag.

    Replaces the old unconditional ``min_length=1`` field constraint — the
    guarantee now holds for every actuating kind, enforced by
    ``_check_embodiment_tags_present``.
    """
    with pytest.raises(ValidationError, match="embodiment_tag"):
        RSkillManifest(**_vla_kwargs(embodiment_tags=[]))
    with pytest.raises(ValidationError, match="embodiment_tag"):
        RSkillManifest(**_ros_action_kwargs(embodiment_tags=[]))


def test_perception_kinds_allow_empty_embodiment_tags() -> None:
    """Detector / vlm rSkills are embodiment-agnostic — empty tags validate.

    Uses the real in-tree perception manifests (no synthetic placeholders,
    CLAUDE.md §1.11): they ship ``embodiment_tags: []`` and must load.
    """
    repo = pathlib.Path(__file__).resolve().parents[2]
    for name in ("rtdetr-coco-r18", "qwen35-4b-nf4"):
        m = RSkillManifest.from_yaml(str(repo / "rskills" / name / "rskill.yaml"))
        assert m.kind in {"detector", "vlm"}
        assert list(m.embodiment_tags) == []


# ── kind == "ros_action" ────────────────────────────────────────────────────


def test_ros_action_kind_requires_ros_integration() -> None:
    with pytest.raises(ValidationError, match="ros_integration"):
        RSkillManifest(**_ros_action_kwargs(ros_integration=None))


def test_ros_action_kind_forbids_model_family() -> None:
    with pytest.raises(ValidationError, match="model_family"):
        RSkillManifest(**_ros_action_kwargs(model_family="act"))


def test_ros_action_kind_forbids_weights_uri() -> None:
    with pytest.raises(ValidationError, match="weights_uri"):
        RSkillManifest(**_ros_action_kwargs(weights_uri="hf://example/test"))


def test_ros_action_kind_pins_chunk_size_to_one() -> None:
    # The schema constraint exists so the safety supervisor's row-0
    # check sees every commanded waypoint; loosening it would let
    # waypoints 1..N actuate unchecked. Pin it loudly.
    with pytest.raises(ValidationError, match=r"chunk_size=1"):
        RSkillManifest(**_ros_action_kwargs(chunk_size=5))


def test_ros_action_kind_happy_path() -> None:
    m = RSkillManifest(**_ros_action_kwargs())
    assert m.kind == "ros_action"
    assert m.model_family is None
    assert m.weights_uri is None
    assert m.ros_integration is not None
    assert m.ros_integration.package == "moveit_msgs"
    assert m.ros_integration.interface_type == "MoveGroup"
    assert m.chunk_size == 1


def test_ros_action_kind_result_only_mode_validates() -> None:
    # Nav2 shape: omit result_trajectory_field.
    ri = RosIntegration(
        package="nav2_msgs",
        interface_type="NavigateToPose",
        interface_name="/navigate_to_pose",
        result_trajectory_field=None,
        default_goal_json='{"pose": {"header": {"frame_id": "map"}}}',
    )
    m = RSkillManifest(
        **_ros_action_kwargs(
            name="openral/rskill-nav2-test",
            ros_integration=ri,
            actuators_required=[
                ActuatorRequirement(
                    kind=ControlMode.BODY_TWIST,
                    control_mode_semantics=ControlModeSemantics(mode="absolute"),
                ),
            ],
            actions=[RSkillAction.NAVIGATE],
        )
    )
    assert m.ros_integration is not None
    assert m.ros_integration.result_trajectory_field is None


# ── RosIntegration field-level validators ───────────────────────────────────


def test_ros_integration_interface_name_must_be_ros_path() -> None:
    with pytest.raises(ValidationError, match="ROS path"):
        RosIntegration(
            package="moveit_msgs",
            interface_type="MoveGroup",
            interface_name="move_action",  # missing leading /
            default_goal_json="{}",
        )


def test_ros_integration_default_goal_json_must_be_json_dict() -> None:
    with pytest.raises(ValidationError, match="not valid JSON"):
        RosIntegration(
            package="moveit_msgs",
            interface_type="MoveGroup",
            interface_name="/move_action",
            default_goal_json="{this is not json",
        )
    with pytest.raises(ValidationError, match="JSON object"):
        RosIntegration(
            package="moveit_msgs",
            interface_type="MoveGroup",
            interface_name="/move_action",
            default_goal_json='["not a dict"]',
        )


# ── Migration audit: every in-tree manifest declares kind ───────────────────


def test_every_intree_rskill_manifest_declares_kind() -> None:
    """ADR-0024 migration: no manifest may rely on a default for ``kind``.

    The schema field is required (no default) so this also doubles as
    proof that the migration script ran on every in-tree rskill. The
    test fails loud (with the offending paths) if any new manifest
    lands without ``kind:``.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    rskills_dir = repo_root / "rskills"
    missing: list[pathlib.Path] = []
    for yaml_path in sorted(rskills_dir.glob("*/rskill.yaml")):
        # Defer YAML parsing to RSkillManifest.from_yaml so any
        # validation failure surfaces with a useful pydantic message.
        manifest = RSkillManifest.from_yaml(str(yaml_path))
        assert manifest.kind in {
            "vla",
            "wam",
            "ros_action",
            "ros_service",
            "detector",
            "vlm",
            "reward",
            "playbook",
        }, f"{yaml_path}: unexpected kind={manifest.kind!r}"
    assert not missing
