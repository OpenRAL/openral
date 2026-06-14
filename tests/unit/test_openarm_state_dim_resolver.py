"""Unit tests for ``_resolve_state_dim`` in the openarm_robosuite backend.

The openarm tabletop sim backend used to hardcode ``_OBS_STATE_DIM = 16``
at module scope, tying it to a single rSkill checkpoint. Per CLAUDE.md
§6.4, every rSkill that wants to participate in the dataset bridge
declares ``state_contract.dim`` and ``action_contract.dim`` — so the
backend now derives the action / observation width from the manifest
at backend init.

CLAUDE.md §1.11: real schemas, real rSkill manifests, no mocks. We
exercise the resolver against the in-tree
``rskills/pi05-openarm-vision-nf4/rskill.yaml`` fixture and against
the openarm robot.yaml's joint count as the fallback.
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

    Per the docstring, network / missing-package errors are swallowed
    so test fixtures without HF Hub access stay green; the downstream
    rSkill loader surfaces the real error when it later tries to
    actually load the policy weights.
    """
    assert (
        _resolve_state_dim(
            weights_uri="this-rskill-does-not-exist-anywhere",
            fallback=16,
        )
        == 16
    )


def test_state_dim_from_rskill_pi05_openarm_vision() -> None:
    """The in-tree pi05 OpenArm vision rskill declares dim=16."""
    pytest.importorskip("openral_rskill")
    # The rskill manifest is local — ``pi05-openarm-vision-nf4``
    # resolves via ``rskills/pi05-openarm-vision-nf4/rskill.yaml``.
    result = _resolve_state_dim(
        weights_uri="pi05-openarm-vision-nf4",
        fallback=0,
    )
    assert result == 16


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
    # Build a minimal rskill.yaml on disk with mismatched dims. We
    # reuse the loader's local-resolve path by pointing at the directory.
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
