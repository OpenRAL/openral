"""Unit tests for OpenArm tabletop camera defaults.

The openarm_robosuite scene composer used to carry module-level
constants for the "top" (a.k.a. "base") overview camera, baked to the
``mddoai/openarm_2026-05-14_clean`` dataset POV. ADR-0066 moved those
defaults onto the deploy scene composition so the environment owns its
arena and camera pose; the robot manifest describes only the robot.

CLAUDE.md §1.11: real schemas, real fixture under ``robots/openarm/``,
no mocks.
"""

from __future__ import annotations

import pytest
from openral_core import (
    ControlMode,
    DeployScene,
    EmbodimentKind,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SceneDefaults,
    TopCameraDefaults,
)
from pydantic import ValidationError

# ── Submodel schema round-trip ────────────────────────────────────────────────


def test_top_camera_defaults_round_trips_through_pydantic() -> None:
    """Construct → model_dump → re-validate yields an equal instance."""
    cam = TopCameraDefaults(
        pos=(0.20, 0.0, 0.95),
        target=(0.65, 0.0, 0.05),
        fovy=65.0,
    )
    reparsed = TopCameraDefaults.model_validate(cam.model_dump())
    assert reparsed == cam


def test_top_camera_defaults_rejects_invalid_fovy() -> None:
    """``fovy`` must lie in (0, 180); the schema's Field constraints fire."""
    with pytest.raises(ValidationError):
        TopCameraDefaults(pos=(0, 0, 0), target=(1, 0, 0), fovy=0.0)
    with pytest.raises(ValidationError):
        TopCameraDefaults(pos=(0, 0, 0), target=(1, 0, 0), fovy=180.0)


def test_scene_defaults_allows_missing_top_camera() -> None:
    """A robot may have a ``scene_defaults`` block with no ``top_camera``."""
    sd = SceneDefaults()
    assert sd.top_camera is None


def test_scene_defaults_forbids_unknown_fields() -> None:
    """The ``extra="forbid"`` config catches typos in the YAML."""
    with pytest.raises(ValidationError):
        SceneDefaults.model_validate({"top_kamera": {"pos": [0, 0, 0]}})


# ── OpenArm fixture ownership ─────────────────────────────────────────────────


def test_openarm_deploy_scene_loads_top_camera_defaults() -> None:
    """The in-tree OpenArm deploy scene carries the mddoai dataset POV."""
    scene = DeployScene.from_yaml("scenes/deploy/openarm_tabletop.yaml")
    assert scene.composition is not None
    params = scene.composition.params

    assert params["top_camera_pos"] == [0.20, 0.0, 0.95]
    assert params["top_camera_target"] == [0.65, 0.0, 0.05]
    assert params["top_camera_fovy"] == 65.0


def test_openarm_robot_yaml_does_not_own_scene_defaults() -> None:
    """ADR-0066 keeps scene camera defaults off the robot manifest."""
    desc = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    assert desc.scene_defaults is None


def test_openarm_robot_yaml_matches_in_code_constant() -> None:
    """``robots/openarm/robot.yaml`` and ``OPENARM_DESCRIPTION`` agree on defaults.

    The drift guard now asserts both omit scene defaults; scene composition is
    pinned by ``test_openarm_deploy_scene_loads_top_camera_defaults``.
    """
    pytest.importorskip("openral_hal")
    from openral_hal.openarm import OPENARM_DESCRIPTION

    yaml_desc = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    assert yaml_desc.scene_defaults == OPENARM_DESCRIPTION.scene_defaults


# ── Backwards compatibility ──────────────────────────────────────────────────


def test_robot_description_without_scene_defaults_still_loads() -> None:
    """Manifests omitting ``scene_defaults`` entirely stay valid.

    The default of ``None`` reproduces the historical behavior — the
    backend then falls back to its hard-coded openarm POV.
    """
    desc = RobotDescription(
        name="legacy_fixture",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name="j1",
                joint_type=JointType.REVOLUTE,
                parent_link="base_link",
                child_link="link_1",
            ),
        ],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION],
            embodiment_tags=["legacy"],
        ),
        safety=SafetyEnvelope(),
    )
    assert desc.scene_defaults is None
