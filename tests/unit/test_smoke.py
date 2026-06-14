"""Day-1 smoke test — the canonical pre-flight check.

This file's role is **deliberately** distinct from
``tests/unit/test_schemas_fuzz.py`` (hypothesis property tests) and
``tests/unit/test_rskill_manifest.py`` (RSkillManifest-specific schema
checks):

- It is the first test contributors run (``pytest tests/unit/test_smoke.py``)
  to verify the workspace installed correctly.
- It uses **only the public ``import openral_core as core`` surface**,
  catching regressions in the package re-exports without touching
  internal modules.
- It pins the package version string so a botched release bump is caught
  before tests/build/publish run.

Per the audit (``tests/README.md`` §4-A.7), this docstring documents the
file's role so future readers don't fold its assertions into the fuzz
suite by accident.
"""

from __future__ import annotations

import openral_core as core


def test_version_is_set() -> None:
    """Verify the package version is set."""
    assert core.__version__ == "0.1.0"


def test_can_construct_minimal_robot_description() -> None:
    """Verify that a minimal RobotDescription can be constructed and round-tripped."""
    desc = core.RobotDescription(
        name="smoke_robot",
        embodiment_kind=core.EmbodimentKind.MANIPULATOR,
        joints=[
            core.JointSpec(
                name="j1",
                joint_type=core.JointType.REVOLUTE,
                parent_link="base_link",
                child_link="link_1",
            )
        ],
        capabilities=core.RobotCapabilities(
            supported_control_modes=[core.ControlMode.JOINT_POSITION],
            embodiment_tags=["smoke"],
        ),
        safety=core.SafetyEnvelope(),
    )
    dumped = desc.model_dump()
    reloaded = core.RobotDescription.model_validate(dumped)
    assert reloaded.name == "smoke_robot"


def test_world_state_round_trip() -> None:
    """Verify WorldState can be serialized and deserialized."""
    ws = core.WorldState(
        stamp_ns=1_000_000_000,
        joint_state=core.JointState(
            name=["j1"],
            position=[0.0],
            stamp_ns=1_000_000_000,
        ),
    )
    dumped = ws.model_dump()
    reloaded = core.WorldState.model_validate(dumped)
    assert reloaded.stamp_ns == 1_000_000_000


def test_action_minimal() -> None:
    """Verify a minimal Action can be constructed."""
    action = core.Action(
        control_mode=core.ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    assert action.horizon == 1
    assert action.confidence == 1.0
