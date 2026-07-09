"""Tests for the per-file processor materialization helper.

Closes Gap 1 + Gap 3 of the rSkill self-containment audit at the
helper boundary. Verifies:

- :func:`parse_hf_file_uri` splits URIs of the various shapes the schema
  accepts (with / without revision pin; nested file paths).
- :func:`materialize_processor_dir` calls
  :func:`huggingface_hub.hf_hub_download` per URI with the exact
  ``(repo_id, filename, revision)`` triplet declared in the manifest's
  ``processors`` block — NOT ``snapshot_download``.
- The returned directory exposes the lerobot-canonical filenames
  (``policy_preprocessor.json`` / ``policy_postprocessor.json``) so the
  ``make_pre_post_processors`` factory consumes them transparently.

Mocks are scoped to the ``huggingface_hub`` network boundary
(CLAUDE.md §5.4 allows network-boundary mocks). Every other component
— Pydantic schemas, manifest loaders, the helper itself — runs for real.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

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
from openral_core.exceptions import ROSConfigError
from openral_rskill._vla_core import materialize_processor_dir, parse_hf_file_uri


def _modern_manifest(
    *,
    preprocessor_uri: str = "hf://lerobot/smolvla_libero/policy_preprocessor.json",
    postprocessor_uri: str = "hf://lerobot/smolvla_libero/policy_postprocessor.json",
    family: str = "smolvla",
    embodiment: str = "franka_panda",
) -> RSkillManifest:
    return RSkillManifest(
        name="test/rskill-modern",
        version="0.1.0",
        license=RSkillLicensePosture.APACHE_2_0,
        role="s1",
        kind="vla",
        model_family=family,  # type: ignore[arg-type]
        embodiment_tags=[embodiment],  # type: ignore[list-item]
        runtime=RSkillRuntime.PYTORCH,
        weights_uri="hf://lerobot/smolvla_libero",
        chunk_size=16,
        latency_budget=RSkillLatencyBudget(per_chunk_ms=150.0),
        actuators_required=[
            ActuatorRequirement(
                kind=ControlMode.JOINT_POSITION,
                control_mode_semantics=ControlModeSemantics(mode="absolute"),
            )
        ],
        processors=RSkillProcessors(
            preprocessor_uri=preprocessor_uri,
            postprocessor_uri=postprocessor_uri,
        ),
        description="Modern rSkill fixture for the processor-materialiser test suite.",
        actions=[RSkillAction.GENERALIST],
    )


# ─── parse_hf_file_uri ────────────────────────────────────────────────────


class TestParseHfFileUri:
    def test_simple_repo_plus_file(self) -> None:
        assert parse_hf_file_uri("hf://lerobot/smolvla_base/policy_preprocessor.json") == (
            "lerobot/smolvla_base",
            None,
            "policy_preprocessor.json",
        )

    def test_revision_pin(self) -> None:
        assert parse_hf_file_uri("hf://lerobot/smolvla_base@abc123/policy_preprocessor.json") == (
            "lerobot/smolvla_base",
            "abc123",
            "policy_preprocessor.json",
        )

    def test_nested_filename(self) -> None:
        """File-tail may include directories under the repo root."""
        assert parse_hf_file_uri("hf://owner/repo/subdir/nested/stats.safetensors") == (
            "owner/repo",
            None,
            "subdir/nested/stats.safetensors",
        )

    def test_missing_file_tail_rejected(self) -> None:
        with pytest.raises(ROSConfigError, match="missing a file tail"):
            parse_hf_file_uri("hf://lerobot/smolvla_base")

    def test_non_hf_scheme_rejected(self) -> None:
        with pytest.raises(ROSConfigError, match="hf://"):
            parse_hf_file_uri("local://bundled/file.json")


# ─── materialize_processor_dir ────────────────────────────────────────────


class TestMaterializeProcessorDir:
    def test_calls_hf_hub_download_per_uri_not_snapshot_download(self, tmp_path: Path) -> None:
        """Adapter must download EXACTLY the two files declared on the manifest."""
        manifest = _modern_manifest(
            preprocessor_uri="hf://lerobot/smolvla_libero/policy_preprocessor.json",
            postprocessor_uri="hf://lerobot/smolvla_libero/policy_postprocessor.json",
        )

        # Synthesize two fake downloads in tmp_path so the symlinks below
        # resolve to real files. (We're testing the call shape, not the
        # JSON content.)
        pre_real = tmp_path / "preprocessor.json"
        post_real = tmp_path / "postprocessor.json"
        pre_real.write_text("{}")
        post_real.write_text("{}")

        seen_calls: list[dict[str, object]] = []

        def _fake_download(
            *, repo_id: str, filename: str, revision: str | None = None, **kw: object
        ) -> str:
            seen_calls.append({"repo_id": repo_id, "filename": filename, "revision": revision})
            if filename.endswith("preprocessor.json"):
                return str(pre_real)
            return str(post_real)

        # Important: mock the symbol at huggingface_hub (where the helper
        # imports it from), NOT a re-export inside our module. Same call
        # shape as the production path.
        with patch("huggingface_hub.hf_hub_download", side_effect=_fake_download):
            staging = materialize_processor_dir(manifest)

        # Both URIs resolved into hf_hub_download calls; snapshot_download
        # is never reached. We assert exact call shape here so a regression
        # that re-introduces snapshot_download surfaces loudly.
        assert seen_calls == [
            {
                "repo_id": "lerobot/smolvla_libero",
                "filename": "policy_preprocessor.json",
                "revision": None,
            },
            {
                "repo_id": "lerobot/smolvla_libero",
                "filename": "policy_postprocessor.json",
                "revision": None,
            },
        ]

        # Staging dir exposes the lerobot-canonical filenames as symlinks.
        assert os.path.islink(os.path.join(staging, "policy_preprocessor.json"))
        assert os.path.islink(os.path.join(staging, "policy_postprocessor.json"))
        assert os.path.realpath(os.path.join(staging, "policy_preprocessor.json")) == str(pre_real)
        assert os.path.realpath(os.path.join(staging, "policy_postprocessor.json")) == str(
            post_real
        )

    def test_revision_pin_threaded_through(self, tmp_path: Path) -> None:
        """When the URI carries ``@<rev>``, that revision reaches hf_hub_download."""
        manifest = _modern_manifest(
            preprocessor_uri="hf://lerobot/smolvla_libero@deadbee/policy_preprocessor.json",
            postprocessor_uri="hf://lerobot/smolvla_libero@deadbee/policy_postprocessor.json",
        )

        fake_file = tmp_path / "f.json"
        fake_file.write_text("{}")
        seen: list[str | None] = []

        def _fake_download(
            *, repo_id: str, filename: str, revision: str | None = None, **kw: object
        ) -> str:
            seen.append(revision)
            return str(fake_file)

        with patch("huggingface_hub.hf_hub_download", side_effect=_fake_download):
            materialize_processor_dir(manifest)

        assert seen == ["deadbee", "deadbee"]

    def test_state_files_in_pipeline_json_are_materialized(self, tmp_path: Path) -> None:
        """Pipeline steps with ``state_file`` get their sibling .safetensors downloaded too.

        Reason: ``lerobot.processor.pipeline.PolicyProcessorPipeline.from_pretrained``
        walks every step in the loaded JSON. For steps that declare a
        ``state_file`` (normalizer / unnormalizer / tokenizer) it checks the
        passed directory first and, if the state file isn't there, falls back
        to ``hf_hub_download(repo_id=<dir>, ...)`` — which fails because the
        directory is a local path, not a repo id. So the helper must
        pre-materialize those state files into the same staging dir.
        """
        import json

        pre_state_filename = "policy_preprocessor_step_5_normalizer_processor.safetensors"
        post_state_filename = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

        manifest = _modern_manifest(
            preprocessor_uri="hf://lerobot/smolvla_libero/policy_preprocessor.json",
            postprocessor_uri="hf://lerobot/smolvla_libero/policy_postprocessor.json",
        )

        # Build two real JSON files that look like the lerobot pipeline shape:
        # preprocessor has one step with a state_file; postprocessor has one too.
        pre_json = tmp_path / "pre.json"
        pre_json.write_text(
            json.dumps(
                {
                    "name": "policy_preprocessor",
                    "steps": [
                        {"registry_name": "to_batch_processor", "config": {}},
                        {
                            "registry_name": "normalizer_processor",
                            "config": {},
                            "state_file": pre_state_filename,
                        },
                    ],
                }
            )
        )
        post_json = tmp_path / "post.json"
        post_json.write_text(
            json.dumps(
                {
                    "name": "policy_postprocessor",
                    "steps": [
                        {
                            "registry_name": "unnormalizer_processor",
                            "config": {},
                            "state_file": post_state_filename,
                        },
                    ],
                }
            )
        )
        # Two fake state-file blobs the JSONs point at.
        pre_state = tmp_path / "pre_state.safetensors"
        pre_state.write_bytes(b"\x00" * 16)
        post_state = tmp_path / "post_state.safetensors"
        post_state.write_bytes(b"\x00" * 16)

        seen_files: list[str] = []

        def _fake_download(
            *, repo_id: str, filename: str, revision: str | None = None, **kw: object
        ) -> str:
            seen_files.append(filename)
            if filename == "policy_preprocessor.json":
                return str(pre_json)
            if filename == "policy_postprocessor.json":
                return str(post_json)
            if filename == pre_state_filename:
                return str(pre_state)
            if filename == post_state_filename:
                return str(post_state)
            raise AssertionError(f"unexpected filename {filename!r}")

        with patch("huggingface_hub.hf_hub_download", side_effect=_fake_download):
            staging = materialize_processor_dir(manifest)

        # All four files were downloaded: two JSONs first (so the helper can
        # parse them), then their referenced state files.
        assert sorted(seen_files) == sorted(
            [
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                pre_state_filename,
                post_state_filename,
            ]
        )

        # Staging dir contains symlinks to every file under the names lerobot
        # expects (canonical JSONs + the verbatim state_file filenames from
        # the JSON, so lerobot's `base_path / state_filename` check hits).
        for fname in (
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            pre_state_filename,
            post_state_filename,
        ):
            link = os.path.join(staging, fname)
            assert os.path.islink(link), f"missing symlink {fname}"

    def test_legacy_manifest_without_processors_raises(self) -> None:
        """``manifest.processors is None`` is the legacy ACT path; helper must refuse."""
        # Build a legacy ACT manifest by hand — schema allows model_family=act
        # to omit processors.
        manifest = RSkillManifest(
            name="test/rskill-legacy-act",
            version="0.1.0",
            license=RSkillLicensePosture.MIT,
            role="s1",
            kind="vla",
            model_family="act",
            embodiment_tags=["aloha"],
            runtime=RSkillRuntime.PYTORCH,
            weights_uri="hf://lerobot/act_aloha_sim_transfer_cube_human",
            chunk_size=100,
            latency_budget=RSkillLatencyBudget(per_chunk_ms=25.0),
            actuators_required=[
                ActuatorRequirement(
                    kind=ControlMode.JOINT_POSITION,
                    control_mode_semantics=ControlModeSemantics(mode="absolute"),
                )
            ],
            description="Legacy ACT rSkill fixture (no processors block, norm stats baked in).",
            actions=[RSkillAction.GENERALIST],
        )
        assert manifest.processors is None
        with pytest.raises(ROSConfigError, match="processors"):
            materialize_processor_dir(manifest)
