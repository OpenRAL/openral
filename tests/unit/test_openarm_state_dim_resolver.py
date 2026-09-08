"""Unit tests for ``_resolve_state_dim`` in the openarm_robosuite backend.

Backend derives observation/action width from the rSkill manifest's
``state_contract.dim`` / ``action_contract.dim`` (CLAUDE.md §6.4) instead of a
hardcoded constant tied to one checkpoint, falling back to the openarm
robot.yaml's joint count when no rSkill resolves.
"""

from __future__ import annotations

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.openarm_robosuite.env import _resolve_state_dim


def test_state_dim_falls_back_when_no_rskill_uri() -> None:
    """A non-rskill weights_uri leaves us on the fallback (robot joint count)."""
    assert _resolve_state_dim(weights_uri="mock://noop", fallback=16) == 16
    assert _resolve_state_dim(weights_uri="mock://noop", fallback=12) == 12


def test_state_dim_falls_back_when_uri_is_none() -> None:
    """``None`` weights_uri also drops to the fallback."""
    assert _resolve_state_dim(weights_uri=None, fallback=14) == 14


def test_state_dim_uses_fallback_when_rskill_unresolvable() -> None:
    """A bare reference that does not resolve drops to fallback.

    Network/missing-package errors are swallowed here so fixtures without HF Hub
    access stay green; the loader surfaces the real error when it later loads weights.
    """
    assert (
        _resolve_state_dim(
            weights_uri="this-rskill-does-not-exist-anywhere",
            fallback=16,
        )
        == 16
    )


def test_state_dim_from_rskill_state_contract(tmp_path) -> None:
    """A real on-disk rSkill manifest's ``state_contract.dim`` wins over fallback.

    Builds a minimal openarm rskill.yaml (16-D state == 16-D action, matching
    the OpenArm v2 16-joint embodiment) and resolves it via the loader's
    local-directory path — exercises the manifest-resolve branch without
    depending on any shipped checkpoint.
    """
    pytest.importorskip("openral_rskill")
    rskill_dir = tmp_path / "test-openarm-state-dim"
    rskill_dir.mkdir()
    (rskill_dir / "rskill.yaml").write_text(
        """
schema_version: "0.1"
name: "test/rskill-openarm-state-dim"
version: "0.0.1"
license: "apache-2.0"
role: "s1"
kind: vla
model_family: "act"
embodiment_tags: ["openarm"]
actions: ["pick"]
sensors_required:
  - modality: "rgb"
    vla_feature_key: "observation.images.base"
    min_width: 64
    min_height: 64
actuators_required:
  - kind: "joint_position"
    n_dof: 8
    vla_action_key: "action.joint.left"
    control_mode_semantics:
      mode: "absolute"
runtime: "pytorch"
weights_uri: "hf://test/rskill-openarm-state-dim"
state_contract:
  dim: 16
action_contract:
  dim: 16
chunk_size: 50
latency_budget:
  per_chunk_ms: 1000.0
paper_url: "https://example.com"
description: "fixture for state_contract.dim resolution"
""".strip()
    )
    assert _resolve_state_dim(weights_uri=str(rskill_dir), fallback=0) == 16


def test_state_dim_matches_openarm_robot_joint_count() -> None:
    """The robot manifest's joint count is the natural fallback.

    The OpenArm v2 robot.yaml declares 16 joints (7 arm + 1 gripper
    per side, 2 sides), so ``len(desc.joints)`` == 16 is the canonical
    fallback for rollouts that do not point at an rSkill.
    """
    desc = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    assert len(desc.joints) == 16
    assert _resolve_state_dim(weights_uri=None, fallback=len(desc.joints)) == 16


def test_state_dim_rejects_state_action_mismatch(tmp_path) -> None:
    """A rSkill that declares state_contract.dim != action_contract.dim
    is rejected — the openarm backend feeds the action vector through
    the observation.state slot, so they must agree.
    """
    pytest.importorskip("openral_rskill")
    rskill_dir = tmp_path / "test-mismatched-rskill"
    rskill_dir.mkdir()
    (rskill_dir / "rskill.yaml").write_text(
        """
schema_version: "0.1"
name: "test/rskill-state-action-mismatch"
version: "0.0.1"
license: "apache-2.0"
role: "s1"
kind: vla
model_family: "act"
embodiment_tags: ["openarm"]
actions: ["pick"]
sensors_required:
  - modality: "rgb"
    vla_feature_key: "observation.images.base"
    min_width: 64
    min_height: 64
actuators_required:
  - kind: "joint_position"
    n_dof: 8
    vla_action_key: "action.joint.left"
    control_mode_semantics:
      mode: "absolute"
runtime: "pytorch"
weights_uri: "hf://test/rskill-state-action-mismatch"
state_contract:
  dim: 16
action_contract:
  dim: 14
chunk_size: 50
latency_budget:
  per_chunk_ms: 1000.0
paper_url: "https://example.com"
description: "fixture for mismatched-dim rejection"
""".strip()
    )
    with pytest.raises(ROSConfigError, match=r"state_contract\.dim=16 but action_contract\.dim=14"):
        _resolve_state_dim(
            weights_uri=str(rskill_dir),
            fallback=16,
        )
