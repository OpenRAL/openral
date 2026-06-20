"""Unit tests for the RSkillManifest schema (V1) and ``rskill.yaml`` loader.

Covers the rSkill *package* contract (CLAUDE.md §6.4 / RFC §1.4, §8.7) — the
on-disk descriptor distributed via HuggingFace Hub. Distinct from the
in-process ``Skill`` ABC (tested in ``test_skill.py``).

``schema_version`` stays at ``"0.1"`` deliberately: the schema has not
been published, so ADR-0013 extended the surface in place rather than
bumping. ADR-0013 added two symmetric guards on top of the initial
shape:

- ``actuators_required`` mirrors ``sensors_required`` on the output side
  (required, ``min_length=1``).
- ``"custom"`` is a tenth allowed embodiment tag; when present, the
  manifest MUST set ``embodiment_extra`` declaring the rig's sensor +
  actuator surface, and every actuator must have ``n_dof`` and
  ``vla_action_key`` set explicitly.

V1 already tightened: HF Hub regex on ``name`` / ``fallback_skill_id``,
SemVer on ``version``, ``hf://`` / ``local://`` discriminator on
``weights_uri``, closed Literal sets for ``embodiment_tags`` /
``model_family`` / ``benchmarks`` keys, required ``chunk_size``, and the
derived ``is_commercial_use_allowed`` property in place of the removed
free-field ``commercial_use_allowed``.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml
from openral_core import (
    ActuatorRequirement,
    ControlMode,
    ControlModeSemantics,
    EmbodimentExtra,
    QuantizationBackend,
    QuantizationConfig,
    QuantizationDtype,
    RSkillLatencyBudget,
    RSkillLicensePosture,
    RSkillManifest,
    RSkillRuntime,
)
from pydantic import ValidationError


def _minimal_manifest_dict() -> dict[str, object]:
    """Return a minimal valid manifest dict for tests.

    ADR-0013 extended the surface with actuators_required +
    embodiment_extra; the rSkill self-containment audit added
    ``control_mode_semantics`` (required per actuator, Gap 2) and a
    ``processors`` block (required for modern lerobot families, Gap 1+3).
    The version string stays at "0.1" because the schema has not been
    published yet.
    """
    return {
        "schema_version": "0.1",
        "name": "openral/rskill-pick-cube-so100",
        "version": "0.1.0",
        "license": "apache-2.0",
        "role": "s1",
        "kind": "vla",
        "model_family": "smolvla",
        "embodiment_tags": ["so100_follower"],
        "runtime": "pytorch",
        "weights_uri": "hf://lerobot/smolvla_base@main",
        "chunk_size": 16,
        "latency_budget": {"per_chunk_ms": 100.0},
        "actuators_required": [
            {
                "kind": "joint_position",
                "control_mode_semantics": {"mode": "absolute"},
            }
        ],
        "processors": {
            "preprocessor_uri": "hf://lerobot/smolvla_base/policy_preprocessor.json",
            "postprocessor_uri": "hf://lerobot/smolvla_base/policy_postprocessor.json",
        },
        "description": "Minimal V1 manifest fixture for the rSkill schema test suite.",
        "actions": ["generalist"],
    }


def _custom_embodiment_extra_dict() -> dict[str, object]:
    """Return a valid embodiment_extra block for the ``"custom"`` hatch."""
    return {
        "sensors": [
            {
                "modality": "rgb",
                "vla_feature_key": "observation.images.wrist",
                "min_width": 224,
                "min_height": 224,
            }
        ],
        "actuators": [
            {
                "kind": "joint_position",
                "n_dof": 6,
                "vla_action_key": "action.joints.arm",
                "control_mode_semantics": {"mode": "absolute"},
            }
        ],
    }


def _default_semantics() -> dict[str, str]:
    """Default control_mode_semantics for joint_position kind in tests."""
    return {"mode": "absolute"}


# ── Construction ─────────────────────────────────────────────────────────────


class TestRSkillManifestConstruction:
    def test_minimal_valid(self) -> None:
        m = RSkillManifest.model_validate(_minimal_manifest_dict())
        assert m.schema_version == "0.1"
        assert m.name == "openral/rskill-pick-cube-so100"
        assert m.role == "s1"
        assert m.license is RSkillLicensePosture.APACHE_2_0
        assert m.runtime is RSkillRuntime.PYTORCH
        assert m.model_family == "smolvla"
        assert m.embodiment_tags == ["so100_follower"]
        assert m.embodiment_extra is None
        assert m.chunk_size == 16
        assert m.benchmarks == {}
        assert m.is_commercial_use_allowed is True
        assert len(m.actuators_required) == 1
        assert m.actuators_required[0].kind is ControlMode.JOINT_POSITION
        # auto-fill is loader-side; on the schema they default to None.
        assert m.actuators_required[0].n_dof is None
        assert m.actuators_required[0].vla_action_key is None

    def test_missing_required_weights_uri_raises(self) -> None:
        d = _minimal_manifest_dict()
        del d["weights_uri"]
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_missing_required_latency_budget_raises(self) -> None:
        d = _minimal_manifest_dict()
        del d["latency_budget"]
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_missing_required_chunk_size_raises(self) -> None:
        d = _minimal_manifest_dict()
        del d["chunk_size"]
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_missing_required_model_family_raises(self) -> None:
        d = _minimal_manifest_dict()
        del d["model_family"]
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_missing_required_version_raises(self) -> None:
        d = _minimal_manifest_dict()
        del d["version"]
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_extra_fields_rejected(self) -> None:
        """extra='forbid' guards against silent typos in rskill.yaml."""
        d = _minimal_manifest_dict()
        d["unknwon_field"] = "oops"
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_removed_v0_fields_rejected(self) -> None:
        """V0 fields must not parse under V1 — surfaces stale manifests loudly."""
        for stale_field, value in [
            ("commercial_use_allowed", False),
            ("dispatch_target", "edge"),
            ("engine_uri", "hf://x/y/engine.plan"),
            ("signature", "STUBSIG=="),
            ("metadata", {"paper": "x"}),
        ]:
            d = _minimal_manifest_dict()
            d[stale_field] = value
            with pytest.raises(ValidationError):
                RSkillManifest.model_validate(d)

    def test_invalid_role_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["role"] = "s9"
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_invalid_schema_version_rejected(self) -> None:
        """Only ``"0.1"`` is accepted today; a future shape bumps post-1.0."""
        d = _minimal_manifest_dict()
        d["schema_version"] = "1"
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

        d["schema_version"] = "0"
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_quantization_default(self) -> None:
        m = RSkillManifest.model_validate(_minimal_manifest_dict())
        assert isinstance(m.quantization, QuantizationConfig)
        assert m.quantization.dtype is QuantizationDtype.FP32

    def test_custom_quantization(self) -> None:
        d = _minimal_manifest_dict()
        d["quantization"] = {"dtype": "int8", "backend": "tensorrt", "per_channel": True}
        m = RSkillManifest.model_validate(d)
        assert m.quantization.dtype is QuantizationDtype.INT8
        assert m.quantization.backend is QuantizationBackend.TENSORRT
        assert m.quantization.per_channel is True


# ── V1 validators ────────────────────────────────────────────────────────────


class TestSchemaVersion:
    def test_default_is_v0_1(self) -> None:
        d = _minimal_manifest_dict()
        del d["schema_version"]
        m = RSkillManifest.model_validate(d)
        assert m.schema_version == "0.1"


class TestNameRegex:
    @pytest.mark.parametrize(
        "name",
        [
            "owner/repo",
            "OpenRAL/rskill-pick-cube-so100",
            "user_42/skill.v2",
            "A-B/x.y_z-1",
        ],
    )
    def test_valid_names_accepted(self, name: str) -> None:
        d = _minimal_manifest_dict()
        d["name"] = name
        RSkillManifest.model_validate(d)

    @pytest.mark.parametrize(
        "name",
        [
            "no-slash",
            "/leading",
            "trailing/",
            "two/slashes/here",
            "owner/repo with space",
            "_leading_underscore/repo",
            "",
        ],
    )
    def test_invalid_names_rejected(self, name: str) -> None:
        d = _minimal_manifest_dict()
        d["name"] = name
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)


class TestSemVerVersion:
    @pytest.mark.parametrize(
        "version",
        ["0.1.0", "1.2.3", "10.20.30", "1.0.0-alpha", "1.0.0-rc.1", "1.0.0+build.7"],
    )
    def test_valid_semver_accepted(self, version: str) -> None:
        d = _minimal_manifest_dict()
        d["version"] = version
        RSkillManifest.model_validate(d)

    @pytest.mark.parametrize(
        "version",
        ["v1.0.0", "1.0", "1", "1.0.0.0", "1.0.0-", "abc", ""],
    )
    def test_invalid_versions_rejected(self, version: str) -> None:
        d = _minimal_manifest_dict()
        d["version"] = version
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)


class TestWeightsUri:
    @pytest.mark.parametrize(
        "uri",
        [
            "hf://owner/repo",
            "hf://owner/repo@main",
            "hf://owner/repo@abc1234",
            "local://rskills/diffusion-pusht",
            "local://./local/skill",
        ],
    )
    def test_valid_weights_uri_accepted(self, uri: str) -> None:
        d = _minimal_manifest_dict()
        d["weights_uri"] = uri
        RSkillManifest.model_validate(d)

    @pytest.mark.parametrize(
        "uri",
        [
            "http://example.com/weights",
            "https://example.com/weights",
            "s3://bucket/key",
            "owner/repo",  # missing scheme
            "hf://no-slash",
            "rskill://rskills/diffusion-pusht",  # rskill:// scheme removed
            "rskill://",  # empty path + removed scheme
            "file:///tmp/x",
        ],
    )
    def test_invalid_weights_uri_rejected(self, uri: str) -> None:
        d = _minimal_manifest_dict()
        d["weights_uri"] = uri
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)


class TestEmbodimentTags:
    def test_empty_list_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["embodiment_tags"] = []
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_off_list_tag_rejected(self) -> None:
        """Tags outside the canonical robots/ set must be rejected."""
        d = _minimal_manifest_dict()
        d["embodiment_tags"] = ["lerobot"]
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_multiple_canonical_tags_accepted(self) -> None:
        d = _minimal_manifest_dict()
        d["embodiment_tags"] = ["franka_panda", "sawyer"]
        m = RSkillManifest.model_validate(d)
        assert m.embodiment_tags == ["franka_panda", "sawyer"]


class TestModelFamily:
    @pytest.mark.parametrize(
        "fam",
        ["smolvla", "pi05", "xvla", "act", "diffusion", "rldx", "molmoact2", "gr00t", "openvla"],
    )
    def test_supported_families_accepted(self, fam: str) -> None:
        d = _minimal_manifest_dict()
        d["model_family"] = fam
        RSkillManifest.model_validate(d)

    # "groot" (single-zero typo) stays rejected — the canonical spelling is "gr00t".
    @pytest.mark.parametrize("fam", ["groot", "custom", "smolvla2", ""])
    def test_unsupported_family_rejected(self, fam: str) -> None:
        d = _minimal_manifest_dict()
        d["model_family"] = fam
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)


class TestChunkSize:
    def test_positive_chunk_size(self) -> None:
        d = _minimal_manifest_dict()
        d["chunk_size"] = 50
        m = RSkillManifest.model_validate(d)
        assert m.chunk_size == 50

    @pytest.mark.parametrize("size", [0, -1, -100])
    def test_non_positive_chunk_size_rejected(self, size: int) -> None:
        d = _minimal_manifest_dict()
        d["chunk_size"] = size
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)


class TestBenchmarks:
    def test_valid_keys_and_scores(self) -> None:
        d = _minimal_manifest_dict()
        d["benchmarks"] = {"libero_spatial": 0.8, "libero_10": 0.59}
        m = RSkillManifest.model_validate(d)
        assert m.benchmarks == {"libero_spatial": 0.8, "libero_10": 0.59}

    def test_unknown_benchmark_key_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["benchmarks"] = {"my_custom_suite": 0.5}
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    @pytest.mark.parametrize("score", [-0.01, 1.01, 2.0, -1.0])
    def test_score_out_of_range_rejected(self, score: float) -> None:
        d = _minimal_manifest_dict()
        d["benchmarks"] = {"pusht": score}
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_default_empty_dict(self) -> None:
        m = RSkillManifest.model_validate(_minimal_manifest_dict())
        assert m.benchmarks == {}


class TestMinVramGb:
    def test_keyed_by_quantization_dtype(self) -> None:
        d = _minimal_manifest_dict()
        d["min_vram_gb"] = {"fp32": 14.0, "bf16": 7.0}
        m = RSkillManifest.model_validate(d)
        assert m.min_vram_gb == {QuantizationDtype.FP32: 14.0, QuantizationDtype.BF16: 7.0}

    def test_unknown_dtype_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["min_vram_gb"] = {"sextupling": 99.0}
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_non_positive_vram_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["min_vram_gb"] = {"fp32": 0.0}
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_default_is_none(self) -> None:
        m = RSkillManifest.model_validate(_minimal_manifest_dict())
        assert m.min_vram_gb is None


class TestPaperAndSourceUrls:
    def test_paper_url_accepts_http_and_https(self) -> None:
        for u in ["http://arxiv.org/abs/2410.24164", "https://arxiv.org/abs/2410.24164"]:
            d = _minimal_manifest_dict()
            d["paper_url"] = u
            RSkillManifest.model_validate(d)

    def test_paper_url_rejects_non_http(self) -> None:
        d = _minimal_manifest_dict()
        d["paper_url"] = "ftp://example.com/x"
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_dataset_uri_requires_hf_scheme(self) -> None:
        d = _minimal_manifest_dict()
        d["dataset_uri"] = "https://huggingface.co/datasets/lerobot/libero"
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_dataset_uri_accepts_hf_uri(self) -> None:
        d = _minimal_manifest_dict()
        d["dataset_uri"] = "hf://lerobot/libero@v1"
        RSkillManifest.model_validate(d)

    def test_source_repo_same_regex_as_dataset(self) -> None:
        d = _minimal_manifest_dict()
        d["source_repo"] = "hf://lerobot/smolvla_base"
        RSkillManifest.model_validate(d)


class TestDescription:
    def test_short_description_accepted(self) -> None:
        d = _minimal_manifest_dict()
        d["description"] = "A small skill."
        RSkillManifest.model_validate(d)

    def test_500_char_description_accepted(self) -> None:
        d = _minimal_manifest_dict()
        d["description"] = "x" * 500
        RSkillManifest.model_validate(d)

    def test_over_500_char_description_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["description"] = "x" * 501
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)


class TestFallbackSkillId:
    def test_valid_hf_id_accepted(self) -> None:
        d = _minimal_manifest_dict()
        d["fallback_skill_id"] = "openral/rskill-smaller"
        RSkillManifest.model_validate(d)

    def test_malformed_fallback_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["fallback_skill_id"] = "no-slash"
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_self_reference_rejected(self) -> None:
        """A skill cannot list itself as its own fallback."""
        d = _minimal_manifest_dict()
        d["fallback_skill_id"] = d["name"]
        with pytest.raises(ValidationError, match="fallback_skill_id"):
            RSkillManifest.model_validate(d)


# ── Derived commercial-use posture ───────────────────────────────────────────


class TestIsCommercialUseAllowed:
    @pytest.mark.parametrize(
        "lic, expected",
        [
            (RSkillLicensePosture.APACHE_2_0, True),
            (RSkillLicensePosture.MIT, True),
            (RSkillLicensePosture.BSD, True),
            (RSkillLicensePosture.PERMISSIVE_RESEARCH, False),
            (RSkillLicensePosture.NVIDIA_NON_COMMERCIAL, False),
            # GR00T N1.7+ Open Model License permits commercial use.
            (RSkillLicensePosture.NVIDIA_OPEN_MODEL, True),
            (RSkillLicensePosture.PROPRIETARY, False),
            (RSkillLicensePosture.UNKNOWN, False),
        ],
    )
    def test_derivation(self, lic: RSkillLicensePosture, expected: bool) -> None:
        d = _minimal_manifest_dict()
        d["license"] = lic.value
        m = RSkillManifest.model_validate(d)
        assert m.is_commercial_use_allowed is expected


# ── Latency budget ───────────────────────────────────────────────────────────


class TestRSkillLatencyBudget:
    def test_per_chunk_ms_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            RSkillLatencyBudget(per_chunk_ms=0.0)
        with pytest.raises(ValidationError):
            RSkillLatencyBudget(per_chunk_ms=-1.0)

    def test_warmup_load_optional(self) -> None:
        b = RSkillLatencyBudget(per_chunk_ms=50.0)
        assert b.warmup_ms is None
        assert b.load_ms is None

    def test_warmup_must_be_positive_when_set(self) -> None:
        with pytest.raises(ValidationError):
            RSkillLatencyBudget(per_chunk_ms=50.0, warmup_ms=0.0)


# ── YAML round-trip ──────────────────────────────────────────────────────────


class TestRSkillManifestYAML:
    def test_from_yaml_loads_minimal(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "rskill.yaml"
        path.write_text(yaml.safe_dump(_minimal_manifest_dict()))
        m = RSkillManifest.from_yaml(str(path))
        assert m.name == "openral/rskill-pick-cube-so100"

    def test_from_yaml_missing_file_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError):
            RSkillManifest.from_yaml(str(tmp_path / "nope.yaml"))

    def test_from_yaml_invalid_content_raises(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("name: only-name\n")
        with pytest.raises(ValidationError):
            RSkillManifest.from_yaml(str(path))

    def test_json_roundtrip(self) -> None:
        m1 = RSkillManifest.model_validate(_minimal_manifest_dict())
        m2 = RSkillManifest.model_validate_json(m1.model_dump_json())
        assert m1 == m2


# ── actuators_required (ADR-0013) ────────────────────────────────────────────────────


class TestActuatorsRequired:
    def test_missing_actuators_required_rejected(self) -> None:
        """actuators_required is mandatory (min_length=1)."""
        d = _minimal_manifest_dict()
        del d["actuators_required"]
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_empty_actuators_required_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["actuators_required"] = []
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_multiple_actuators_accepted(self) -> None:
        d = _minimal_manifest_dict()
        d["actuators_required"] = [
            {
                "kind": "joint_position",
                "control_mode_semantics": {"mode": "absolute"},
            },
            {
                "kind": "gripper_binary",
                "control_mode_semantics": {
                    "mode": "absolute",
                    "gripper_convention": "binary_close_one",
                },
            },
        ]
        m = RSkillManifest.model_validate(d)
        assert [a.kind.value for a in m.actuators_required] == [
            "joint_position",
            "gripper_binary",
        ]

    def test_unknown_kind_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["actuators_required"] = [
            {"kind": "telekinesis", "control_mode_semantics": {"mode": "absolute"}}
        ]
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_explicit_n_dof_and_vla_action_key_accepted(self) -> None:
        """Manifest author may override the loader's auto-fill explicitly."""
        d = _minimal_manifest_dict()
        d["actuators_required"] = [
            {
                "kind": "joint_position",
                "n_dof": 6,
                "vla_action_key": "action.joints.arm",
                "control_mode_semantics": {"mode": "absolute"},
            }
        ]
        m = RSkillManifest.model_validate(d)
        assert m.actuators_required[0].n_dof == 6
        assert m.actuators_required[0].vla_action_key == "action.joints.arm"

    def test_actuator_requirement_extra_fields_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["actuators_required"] = [
            {
                "kind": "joint_position",
                "control_mode_semantics": {"mode": "absolute"},
                "foo": "bar",
            }
        ]
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)


# ── "custom" embodiment escape hatch (ADR-0013) ───────────────────────────


class TestCustomEmbodimentHatch:
    def test_custom_requires_embodiment_extra(self) -> None:
        d = _minimal_manifest_dict()
        d["embodiment_tags"] = ["custom"]
        # actuators_required must also carry n_dof + vla_action_key for "custom"
        d["actuators_required"] = [
            {
                "kind": "joint_position",
                "n_dof": 6,
                "vla_action_key": "action.joints.arm",
                "control_mode_semantics": {"mode": "absolute"},
            }
        ]
        with pytest.raises(ValidationError, match="embodiment_extra"):
            RSkillManifest.model_validate(d)

    def test_custom_with_embodiment_extra_accepted(self) -> None:
        d = _minimal_manifest_dict()
        d["embodiment_tags"] = ["custom"]
        d["actuators_required"] = [
            {
                "kind": "joint_position",
                "n_dof": 6,
                "vla_action_key": "action.joints.arm",
                "control_mode_semantics": {"mode": "absolute"},
            }
        ]
        d["embodiment_extra"] = _custom_embodiment_extra_dict()
        m = RSkillManifest.model_validate(d)
        assert m.embodiment_extra is not None
        assert len(m.embodiment_extra.sensors) == 1
        assert len(m.embodiment_extra.actuators) == 1

    def test_non_custom_with_embodiment_extra_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["embodiment_extra"] = _custom_embodiment_extra_dict()
        with pytest.raises(ValidationError, match="custom"):
            RSkillManifest.model_validate(d)

    def test_custom_with_missing_actuator_n_dof_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["embodiment_tags"] = ["custom"]
        d["embodiment_extra"] = _custom_embodiment_extra_dict()
        d["actuators_required"] = [
            {"kind": "joint_position", "control_mode_semantics": {"mode": "absolute"}}
        ]
        with pytest.raises(ValidationError, match="n_dof"):
            RSkillManifest.model_validate(d)

    def test_custom_with_missing_actuator_vla_action_key_rejected(self) -> None:
        d = _minimal_manifest_dict()
        d["embodiment_tags"] = ["custom"]
        d["embodiment_extra"] = _custom_embodiment_extra_dict()
        d["actuators_required"] = [
            {
                "kind": "joint_position",
                "n_dof": 6,
                "control_mode_semantics": {"mode": "absolute"},
            }
        ]
        with pytest.raises(ValidationError, match="vla_action_key"):
            RSkillManifest.model_validate(d)

    def test_embodiment_extra_requires_non_empty_sensors(self) -> None:
        extra = _custom_embodiment_extra_dict()
        extra["sensors"] = []
        with pytest.raises(ValidationError):
            EmbodimentExtra.model_validate(extra)

    def test_embodiment_extra_requires_non_empty_actuators(self) -> None:
        extra = _custom_embodiment_extra_dict()
        extra["actuators"] = []
        with pytest.raises(ValidationError):
            EmbodimentExtra.model_validate(extra)

    def test_embodiment_extra_extra_fields_rejected(self) -> None:
        extra = _custom_embodiment_extra_dict()
        extra["foo"] = "bar"
        with pytest.raises(ValidationError):
            EmbodimentExtra.model_validate(extra)


# ── ActuatorRequirement (standalone) ────────────────────────────────────────


class TestActuatorRequirement:
    @staticmethod
    def _abs_sem() -> ControlModeSemantics:
        return ControlModeSemantics(mode="absolute")

    def test_kind_only_is_valid(self) -> None:
        a = ActuatorRequirement(
            kind=ControlMode.JOINT_POSITION,
            control_mode_semantics=self._abs_sem(),
        )
        assert a.kind is ControlMode.JOINT_POSITION
        assert a.n_dof is None
        assert a.vla_action_key is None

    def test_non_positive_n_dof_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ActuatorRequirement(
                kind=ControlMode.JOINT_POSITION,
                n_dof=0,
                control_mode_semantics=self._abs_sem(),
            )
        with pytest.raises(ValidationError):
            ActuatorRequirement(
                kind=ControlMode.JOINT_POSITION,
                n_dof=-1,
                control_mode_semantics=self._abs_sem(),
            )

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ActuatorRequirement(
                kind="warp_drive",  # type: ignore[arg-type]
                control_mode_semantics=self._abs_sem(),
            )


# ── Real in-tree manifests parse against V1 ──────────────────────────────────


class TestInTreeManifests:
    """The 9 ``rskills/*/rskill.yaml`` files must all validate against V1.

    This is the migration safety net: if any manifest is missed during a
    schema bump, this test fails before CI.
    """

    def test_all_in_tree_manifests_parse(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        manifest_paths = sorted(repo_root.glob("rskills/*/rskill.yaml"))
        assert manifest_paths, (
            f"No skills/*/rskill.yaml manifests found under {repo_root}; "
            "the test is in the wrong place or the tree is missing skills."
        )
        for p in manifest_paths:
            RSkillManifest.from_yaml(str(p))


# ── Optional rSkill envelope (ADR-0018 §5 / ADR-0020) ────────────────────────


class TestRSkillEnvelope:
    """Optional ``envelope: SafetyEnvelope`` field carried by the manifest.

    Pre-existing manifests without ``envelope`` continue to parse — the field
    is optional and defaults to ``None``. When set, the C++ safety kernel
    (cpp/openral_safety_kernel/, ADR-0020) enforces the intersection of the
    skill envelope and the robot ceiling; the intersection algebra and the
    loosening-rejection live in :mod:`openral_safety.envelope_loader`, not
    here on the schema.
    """

    def test_envelope_defaults_to_none(self) -> None:
        m = RSkillManifest.model_validate(_minimal_manifest_dict())
        assert m.envelope is None

    def test_envelope_accepts_full_safety_envelope_dict(self) -> None:
        d = _minimal_manifest_dict()
        d["envelope"] = {
            "workspace_box_min_xyz": [-0.3, -0.3, 0.0],
            "workspace_box_max_xyz": [0.3, 0.3, 0.5],
            "max_ee_speed_m_s": 0.3,
            "max_ee_accel_m_s2": 1.0,
            "max_joint_speed_factor": 0.5,
            "max_force_n": 20.0,
            "max_torque_nm": 5.0,
            "deadman_required": True,
            "contact_force_threshold_n": 10.0,
        }
        m = RSkillManifest.model_validate(d)
        assert m.envelope is not None
        assert m.envelope.max_ee_speed_m_s == 0.3
        assert m.envelope.workspace_box_min_xyz == (-0.3, -0.3, 0.0)
        assert m.envelope.max_force_n == 20.0

    def test_envelope_partial_dict_uses_safety_envelope_defaults(self) -> None:
        d = _minimal_manifest_dict()
        d["envelope"] = {"max_force_n": 5.0}
        m = RSkillManifest.model_validate(d)
        assert m.envelope is not None
        assert m.envelope.max_force_n == 5.0
        # SafetyEnvelope defaults still apply for the unspecified fields.
        assert m.envelope.max_ee_speed_m_s == 0.5

    def test_envelope_round_trips_through_yaml(self) -> None:
        d = _minimal_manifest_dict()
        d["envelope"] = {"max_force_n": 7.5, "max_ee_speed_m_s": 0.2}
        m = RSkillManifest.model_validate(d)
        # Dump back to a dict, re-parse, and check the envelope survives.
        dumped = m.model_dump(mode="python", exclude_none=True)
        m2 = RSkillManifest.model_validate(dumped)
        assert m2.envelope is not None
        assert m2.envelope.max_force_n == 7.5
        assert m2.envelope.max_ee_speed_m_s == 0.2

    def test_extra_field_inside_envelope_rejected(self) -> None:
        # SafetyEnvelope is a plain BaseModel; verify a nonsensical extra
        # is rejected per Pydantic's default behavior on the nested model.
        d = _minimal_manifest_dict()
        d["envelope"] = {"max_force_n": 5.0, "garbage_field": 999.0}
        # SafetyEnvelope does not declare extra="forbid" today — the field
        # is silently ignored, matching how every other in-tree
        # RobotDescription.safety block is consumed. This test pins the
        # current behavior so a future tightening (extra="forbid") is an
        # explicit decision rather than an accident.
        m = RSkillManifest.model_validate(d)
        assert m.envelope is not None
        assert m.envelope.max_force_n == 5.0


# ── vlm kind (ADR-0047) ──────────────────────────────────────────────────────


def _vlm_manifest_dict() -> dict[str, object]:
    """Minimal valid manifest dict for kind='vlm'."""
    return {
        "schema_version": "0.1",
        "name": "OpenRAL/rskill-qwen35-4b-nf4",
        "version": "0.1.0",
        "license": "apache-2.0",
        "role": "s2",
        "kind": "vlm",
        "embodiment_tags": ["franka_panda"],
        "sensors_required": [{"modality": "rgb", "min_width": 336, "min_height": 336}],
        "actuators_required": [],
        "runtime": "pytorch",
        "weights_uri": "hf://Qwen/Qwen3.5-4B",
        "chunk_size": 1,
        "latency_budget": {"per_chunk_ms": 3000.0},
        "description": "Qwen3.5-4B NF4 scene VLM rSkill for robot scene understanding.",
        "actions": ["query"],
    }


class TestVlmKind:
    def test_minimal_vlm_valid(self) -> None:
        m = RSkillManifest.model_validate(_vlm_manifest_dict())
        assert m.kind == "vlm"
        assert m.role == "s2"
        assert m.actuators_required == []
        assert m.detector is None
        assert m.action_contract is None
        assert m.state_contract is None
        assert m.is_commercial_use_allowed is True

    def test_from_yaml_qwen35_rskill(self) -> None:
        import pathlib

        repo_root = pathlib.Path(__file__).resolve().parents[2]
        p = repo_root / "rskills" / "qwen35-4b-nf4" / "rskill.yaml"
        m = RSkillManifest.from_yaml(str(p))
        assert m.kind == "vlm"
        assert m.role == "s2"
        # weights_uri is the deployable pre-quantized NF4 checkpoint; source_repo
        # is the SHA-pinned upstream it was quantized from (provenance).
        assert m.weights_uri == "hf://OpenRAL/rskill-qwen35-4b-nf4"
        assert m.source_repo is not None and m.source_repo.startswith("hf://Qwen/Qwen3.5-4B@")

    def test_vlm_missing_weights_uri_raises(self) -> None:
        d = _vlm_manifest_dict()
        del d["weights_uri"]
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_vlm_with_actuators_raises(self) -> None:
        d = _vlm_manifest_dict()
        d["actuators_required"] = [
            {"kind": "joint_position", "control_mode_semantics": {"mode": "absolute"}}
        ]
        with pytest.raises(ValidationError, match="actuates nothing"):
            RSkillManifest.model_validate(d)

    def test_vlm_with_detector_block_raises(self) -> None:
        d = _vlm_manifest_dict()
        d["detector"] = {"labels": ["cup"], "input_size": [640, 640], "score_threshold": 0.5}
        with pytest.raises(ValidationError, match="detector"):
            RSkillManifest.model_validate(d)

    def test_vlm_with_action_contract_raises(self) -> None:
        d = _vlm_manifest_dict()
        d["action_contract"] = {"dim": 7}
        with pytest.raises(ValidationError, match="action_contract"):
            RSkillManifest.model_validate(d)

    def test_vlm_with_ros_integration_raises(self) -> None:
        d = _vlm_manifest_dict()
        d["ros_integration"] = {
            "action_type": "control_msgs/FollowJointTrajectory",
            "action_name": "/arm/follow_joint_trajectory",
        }
        with pytest.raises(ValidationError, match="ros_integration"):
            RSkillManifest.model_validate(d)

    def test_vlm_model_family_rejected(self) -> None:
        d = _vlm_manifest_dict()
        d["model_family"] = "qwen35"
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate(d)

    def test_vlm_query_action_accepted(self) -> None:
        from openral_core import RSkillAction

        m = RSkillManifest.model_validate(_vlm_manifest_dict())
        assert RSkillAction.QUERY in m.actions
