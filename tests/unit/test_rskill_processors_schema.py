"""Schema-level tests for ``RSkillProcessors`` and ``ControlModeSemantics``.

Covers the additions made to close Gap 1, Gap 2, and Gap 3 of the rSkill
self-containment audit:

- :class:`RSkillProcessors` — per-file URIs for the lerobot
  ``PolicyProcessorPipeline`` preprocessor + postprocessor artefacts.
- :class:`ControlModeSemantics` — required nested block on every
  :class:`ActuatorRequirement` declaring absolute-vs-delta, gripper
  convention (when applicable), and reference frame (when cartesian).
- :class:`RSkillManifest` model_validator that forces modern lerobot
  families (``smolvla`` / ``pi05`` / ``xvla`` / ``diffusion`` / ``rldx``)
  to ship a ``processors`` block; only ``act`` may omit it.

No mocks: every test builds real Pydantic models (CLAUDE.md §1.11).
"""

from __future__ import annotations

import pytest
from openral_core import (
    ActuatorRequirement,
    ControlMode,
    ControlModeSemantics,
    RSkillAction,
    RSkillLatencyBudget,
    RSkillLicensePosture,
    RSkillManifest,
    RSkillProcessors,
    RSkillRuntime,
)
from pydantic import ValidationError

# ─── RSkillProcessors ────────────────────────────────────────────────────────


class TestRSkillProcessors:
    def test_distinct_uris_accepted(self) -> None:
        p = RSkillProcessors(
            preprocessor_uri="hf://lerobot/smolvla_libero/policy_preprocessor.json",
            postprocessor_uri="hf://lerobot/smolvla_libero/policy_postprocessor.json",
        )
        assert p.preprocessor_uri.endswith("preprocessor.json")
        assert p.postprocessor_uri.endswith("postprocessor.json")

    def test_identical_uris_rejected(self) -> None:
        with pytest.raises(ValidationError, match="different files"):
            RSkillProcessors(
                preprocessor_uri="hf://owner/repo/same.json",
                postprocessor_uri="hf://owner/repo/same.json",
            )

    def test_bare_repo_uri_without_file_tail_rejected(self) -> None:
        """``hf://owner/repo`` (the implicit-snapshot shape) is rejected."""
        with pytest.raises(ValidationError):
            RSkillProcessors(
                preprocessor_uri="hf://lerobot/smolvla_base",
                postprocessor_uri="hf://lerobot/smolvla_base/post.json",
            )

    def test_rskill_scheme_rejected(self) -> None:
        """``rskill://path/to/file.json`` is no longer an accepted processor URI scheme."""
        with pytest.raises(ValidationError):
            RSkillProcessors(
                preprocessor_uri="rskill://assets/preprocessor.json",
                postprocessor_uri="rskill://assets/postprocessor.json",
            )

    def test_revision_pin_accepted(self) -> None:
        """``hf://owner/repo@<rev>/file.json`` (revision-pinned form) parses."""
        p = RSkillProcessors(
            preprocessor_uri=("hf://lerobot/smolvla_libero@abc123def/policy_preprocessor.json"),
            postprocessor_uri=("hf://lerobot/smolvla_libero@abc123def/policy_postprocessor.json"),
        )
        assert "@abc123def" in p.preprocessor_uri

    def test_unknown_scheme_rejected(self) -> None:
        """Schemes other than hf:// (e.g. local:// or https://) are rejected."""
        with pytest.raises(ValidationError):
            RSkillProcessors(
                preprocessor_uri="https://example.com/preprocessor.json",
                postprocessor_uri="https://example.com/postprocessor.json",
            )


# ─── ControlModeSemantics + ActuatorRequirement cross-validation ────────────


class TestControlModeSemantics:
    def test_minimal_absolute_accepted(self) -> None:
        sem = ControlModeSemantics(mode="absolute")
        assert sem.mode == "absolute"
        assert sem.gripper_convention is None
        assert sem.reference_frame is None
        assert sem.joint_order is None

    def test_delta_mode_accepted(self) -> None:
        assert ControlModeSemantics(mode="delta").mode == "delta"

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ControlModeSemantics(mode="random")  # type: ignore[arg-type]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ControlModeSemantics.model_validate({"mode": "absolute", "foo": "bar"})


class TestActuatorRequirementSemanticsRules:
    def test_joint_position_with_minimal_semantics(self) -> None:
        a = ActuatorRequirement(
            kind=ControlMode.JOINT_POSITION,
            control_mode_semantics=ControlModeSemantics(mode="absolute"),
        )
        assert a.kind is ControlMode.JOINT_POSITION

    def test_gripper_kind_requires_convention(self) -> None:
        with pytest.raises(ValidationError, match="gripper_convention"):
            ActuatorRequirement(
                kind=ControlMode.GRIPPER_BINARY,
                control_mode_semantics=ControlModeSemantics(mode="absolute"),
            )

    @pytest.mark.parametrize(
        "convention",
        [
            "normalized_open_unit",
            "normalized_open_symmetric",
            "binary_close_one",
            "raw_joint_rad",
            "width_meters",
        ],
    )
    def test_every_gripper_convention_accepted(self, convention: str) -> None:
        a = ActuatorRequirement(
            kind=ControlMode.GRIPPER_POSITION,
            control_mode_semantics=ControlModeSemantics(
                mode="absolute",
                gripper_convention=convention,  # type: ignore[arg-type]
            ),
        )
        assert a.control_mode_semantics.gripper_convention == convention

    def test_non_gripper_with_convention_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not declare"):
            ActuatorRequirement(
                kind=ControlMode.JOINT_POSITION,
                control_mode_semantics=ControlModeSemantics(
                    mode="absolute", gripper_convention="raw_joint_rad"
                ),
            )

    @pytest.mark.parametrize(
        "cartesian_kind",
        [
            ControlMode.CARTESIAN_POSE,
            ControlMode.CARTESIAN_DELTA,
            ControlMode.CARTESIAN_TWIST,
        ],
    )
    def test_cartesian_kind_requires_reference_frame(self, cartesian_kind: ControlMode) -> None:
        with pytest.raises(ValidationError, match="reference_frame"):
            ActuatorRequirement(
                kind=cartesian_kind,
                control_mode_semantics=ControlModeSemantics(mode="absolute"),
            )

    def test_cartesian_kind_with_frame_accepted(self) -> None:
        a = ActuatorRequirement(
            kind=ControlMode.CARTESIAN_POSE,
            control_mode_semantics=ControlModeSemantics(
                mode="absolute", reference_frame="base_link"
            ),
        )
        assert a.control_mode_semantics.reference_frame == "base_link"

    def test_non_cartesian_with_reference_frame_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not declare"):
            ActuatorRequirement(
                kind=ControlMode.JOINT_POSITION,
                control_mode_semantics=ControlModeSemantics(
                    mode="absolute", reference_frame="world"
                ),
            )

    def test_joint_order_optional(self) -> None:
        """``joint_order`` is optional for canonical embodiments (filled from robot YAML)."""
        a = ActuatorRequirement(
            kind=ControlMode.JOINT_POSITION,
            control_mode_semantics=ControlModeSemantics(mode="absolute"),
        )
        assert a.control_mode_semantics.joint_order is None

    def test_joint_order_explicit_accepted(self) -> None:
        """``joint_order`` may be set explicitly (custom embodiments / multi-arm)."""
        a = ActuatorRequirement(
            kind=ControlMode.JOINT_POSITION,
            control_mode_semantics=ControlModeSemantics(
                mode="absolute", joint_order=["j1", "j2", "j3"]
            ),
        )
        assert a.control_mode_semantics.joint_order == ["j1", "j2", "j3"]


# ─── RSkillManifest.processors required-for-modern-families validator ───────


def _minimal_modern_manifest_kwargs(
    family: str = "smolvla",
) -> dict[str, object]:
    return {
        "name": "test/rskill-canary",
        "version": "0.1.0",
        "license": RSkillLicensePosture.APACHE_2_0,
        "role": "s1",
        "kind": "vla",
        "model_family": family,
        "embodiment_tags": ["so100_follower"],
        "runtime": RSkillRuntime.PYTORCH,
        "weights_uri": "hf://test/rskill-canary",
        "chunk_size": 16,
        "latency_budget": RSkillLatencyBudget(per_chunk_ms=100.0),
        "actuators_required": [
            ActuatorRequirement(
                kind=ControlMode.JOINT_POSITION,
                control_mode_semantics=ControlModeSemantics(mode="absolute"),
            )
        ],
        "description": "Canary rSkill fixture for the processors-schema test suite.",
        "actions": [RSkillAction.GENERALIST],
    }


def _processors() -> RSkillProcessors:
    return RSkillProcessors(
        preprocessor_uri="hf://test/rskill-canary/policy_preprocessor.json",
        postprocessor_uri="hf://test/rskill-canary/policy_postprocessor.json",
    )


class TestProcessorsRequiredForModernFamilies:
    @pytest.mark.parametrize(
        "family",
        ["smolvla", "pi05", "xvla", "diffusion", "rldx"],
    )
    def test_modern_family_without_processors_rejected(self, family: str) -> None:
        kwargs = _minimal_modern_manifest_kwargs(family)
        # rldx skips embodiment so100; relax so the manifest is otherwise valid.
        if family == "rldx":
            kwargs["embodiment_tags"] = ["franka_panda"]
        with pytest.raises(ValidationError, match="processors"):
            RSkillManifest(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "family",
        ["smolvla", "pi05", "xvla", "diffusion", "rldx"],
    )
    def test_modern_family_with_processors_accepted(self, family: str) -> None:
        kwargs = _minimal_modern_manifest_kwargs(family)
        if family == "rldx":
            kwargs["embodiment_tags"] = ["franka_panda"]
        m = RSkillManifest(**kwargs, processors=_processors())  # type: ignore[arg-type]
        assert m.processors is not None
        assert m.processors.preprocessor_uri.endswith("preprocessor.json")

    def test_act_family_without_processors_accepted_legacy_path(self) -> None:
        """The ``act`` family may omit ``processors`` to use the legacy path
        (norm stats inside ``model.safetensors``; used by ``rskills/act-aloha``).
        """
        kwargs = _minimal_modern_manifest_kwargs("act")
        kwargs["embodiment_tags"] = ["aloha"]
        m = RSkillManifest(**kwargs)  # type: ignore[arg-type]
        assert m.processors is None

    def test_act_family_with_processors_accepted_modern_path(self) -> None:
        """The ``act`` family may ALSO declare ``processors`` (modern path; used
        by ``rskills/act-libero`` whose upstream ships JSON sidecars)."""
        kwargs = _minimal_modern_manifest_kwargs("act")
        kwargs["embodiment_tags"] = ["aloha"]
        m = RSkillManifest(**kwargs, processors=_processors())  # type: ignore[arg-type]
        assert m.processors is not None
