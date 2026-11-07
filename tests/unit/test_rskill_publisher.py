"""Tests for ``tools/rskill_publisher.py``.

The publisher is a CLI script with no Python package wrapper; imported via
``importlib`` from its file path so the test exercises exactly the script
users invoke with ``uv run python tools/rskill_publisher.py``.

All HF Hub I/O (``HfApi.create_repo``, ``upload_folder``, ``model_info``) is
mocked — a network boundary double per CLAUDE.md §1.11. Only local
validation, dry-run, and token-resolution paths are exercised here; the
actual upload is covered by the manual smoke run the publisher's own
docstring documents.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Import the script as a module from its file path ────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PUBLISHER_PATH = _REPO_ROOT / "tools" / "rskill_publisher.py"


def _load_publisher_module() -> types.ModuleType:
    """Import ``tools/rskill_publisher.py`` as a fresh module per test.

    A fresh import is important because the script mutates ``sys.path`` at
    import time; loading it once at module level would leak that mutation
    across the suite.
    """
    spec = importlib.util.spec_from_file_location("_rskill_publisher_under_test", _PUBLISHER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def publisher() -> types.ModuleType:
    return _load_publisher_module()


# ── Manifest fixture ────────────────────────────────────────────────────────


# Name follows the enforced grammar rskill-<model>-<robot>-<task>-<quant>
# (hyphens separate segments; tokens use underscores):
# smolvla / so100_follower / tabletop / fp32 (default dtype).
_FIXTURE_NAME = "test/rskill-smolvla-so100-tabletop-fp32"
_VALID_MANIFEST_YAML = f"""\
name: {_FIXTURE_NAME}
version: 0.1.0
license: apache-2.0
role: s1
kind: vla
model_family: smolvla
embodiment_tags: [so100_follower]
scenes: [tabletop]
runtime: pytorch
weights_uri: hf://lerobot/smolvla_base
chunk_size: 16
latency_budget:
  per_chunk_ms: 100.0
actuators_required:
  - kind: joint_position
    control_mode_semantics: {{mode: absolute}}
processors:
  preprocessor_uri: hf://lerobot/smolvla_base/policy_preprocessor.json
  postprocessor_uri: hf://lerobot/smolvla_base/policy_postprocessor.json
paper_url: "https://arxiv.org/abs/2506.01844"
source_repo: "hf://lerobot/smolvla_base"
description: Publisher-test rSkill fixture — picks a cube on so100.
actions:
  - generalist
"""

_VALID_README = """\
# rskill-publisher-fixture

> **OpenRAL rSkill** — a publisher-test fixture wrapping SmolVLA.
> Real README, fully filled in for unit-test coverage.

## Upstream model

Wraps the upstream Apache-2.0 SmolVLA checkpoint at
`hf://lerobot/smolvla_base`. Paper: arxiv:2506.01844. Training data
covers ~1 700 LIBERO demos.

## Supported robots

| Robot | Embodiment tag |
| --- | --- |
| SO-100 follower | so100_follower |

## Sensors required

| Key | Modality |
| --- | --- |
| observation.images.camera1 | RGB |

## Manifest summary

See `rskill.yaml` for the full set of fields.

## License

Apache-2.0 to match the upstream weights.
"""


def _write_skill_dir(
    tmp_path: Path,
    manifest_text: str = _VALID_MANIFEST_YAML,
    *,
    readme: str | None = _VALID_README,
) -> Path:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "rskill.yaml").write_text(manifest_text, encoding="utf-8")
    if readme is not None:
        (skill_dir / "README.md").write_text(readme, encoding="utf-8")
    return skill_dir


# ── _resolve_token ──────────────────────────────────────────────────────────


def test_resolve_token_prefers_explicit_arg(
    publisher: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-env")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "from-hub-env")
    assert publisher._resolve_token("from-arg") == "from-arg"


def test_resolve_token_falls_back_to_hf_token_env(
    publisher: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-hf-token")
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    assert publisher._resolve_token(None) == "from-hf-token"


def test_resolve_token_falls_back_to_huggingface_hub_token_env(
    publisher: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "from-hub-env")
    assert publisher._resolve_token(None) == "from-hub-env"


def test_resolve_token_exits_1_when_missing(
    publisher: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        publisher._resolve_token(None)
    assert excinfo.value.code == 1


# ── _validate_manifest ───────────────────────────────────────────────────────


def test_validate_manifest_returns_parsed_rskill_manifest(
    publisher: types.ModuleType, tmp_path: Path
) -> None:
    skill_dir = _write_skill_dir(tmp_path)
    manifest = publisher._validate_manifest(skill_dir)
    assert manifest.name == _FIXTURE_NAME
    assert manifest.version == "0.1.0"
    assert manifest.latency_budget.per_chunk_ms == 100.0


def test_validate_manifest_exits_1_when_yaml_missing(
    publisher: types.ModuleType, tmp_path: Path
) -> None:
    skill_dir = tmp_path / "no-manifest"
    skill_dir.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        publisher._validate_manifest(skill_dir)
    assert excinfo.value.code == 1


def test_validate_manifest_exits_1_when_yaml_invalid(
    publisher: types.ModuleType, tmp_path: Path
) -> None:
    skill_dir = _write_skill_dir(tmp_path, manifest_text="name: missing-required-fields\n")
    with pytest.raises(SystemExit) as excinfo:
        publisher._validate_manifest(skill_dir)
    assert excinfo.value.code == 1


# ── main(): dry-run + error paths ───────────────────────────────────────────


def test_main_dry_run_prints_valid_and_returns(
    publisher: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir = _write_skill_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(skill_dir)])
    publisher.main()  # no --publish, no --bump-revision → dry run

    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert _FIXTURE_NAME in out
    assert "0.1.0" in out


def test_main_exits_1_when_skill_dir_does_not_exist(
    publisher: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(tmp_path / "nonexistent")])
    with pytest.raises(SystemExit) as excinfo:
        publisher.main()
    assert excinfo.value.code == 1


def test_main_exits_1_when_manifest_missing_in_real_directory(
    publisher: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir = tmp_path / "empty"
    skill_dir.mkdir()
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(skill_dir)])
    with pytest.raises(SystemExit) as excinfo:
        publisher.main()
    assert excinfo.value.code == 1


# ── _publish: privacy gate (security regression guard)


def test_publish_passes_private_true_to_create_repo(
    publisher: types.ModuleType,
    tmp_path: Path,
) -> None:
    skill_dir = _write_skill_dir(tmp_path)
    manifest = publisher._validate_manifest(skill_dir)

    fake_api = MagicMock()
    fake_api.create_repo.return_value = "https://huggingface.co/test/rskill-publisher-fixture"
    fake_api.model_info.return_value = MagicMock(private=True)

    with patch.dict(
        "sys.modules",
        {"huggingface_hub": types.SimpleNamespace(HfApi=lambda token=None: fake_api)},
    ):
        url = publisher._publish(skill_dir, manifest, token="fake-token")

    assert url == "https://huggingface.co/test/rskill-publisher-fixture"
    # Critical: ``private=True`` is passed to create_repo every time.
    create_kwargs = fake_api.create_repo.call_args.kwargs
    assert create_kwargs["private"] is True
    # Privacy gate fires: model_info must be called at least once for the verification.
    fake_api.model_info.assert_called()


def test_publish_aborts_when_repo_reports_not_private(
    publisher: types.ModuleType,
    tmp_path: Path,
) -> None:
    """If the API claims the repo is public, ``_publish`` MUST abort.

    Regression guard: if this stops aborting, an attacker (or a typo) could
    silently publish closed-source weights to a public repo (CLAUDE.md
    §1.9 ban on bundling closed-source weights without the license guard).
    """
    skill_dir = _write_skill_dir(tmp_path)
    manifest = publisher._validate_manifest(skill_dir)

    fake_api = MagicMock()
    fake_api.create_repo.return_value = "https://huggingface.co/test/rskill-publisher-fixture"
    fake_api.model_info.return_value = MagicMock(private=False)  # public!

    with (
        patch.dict(
            "sys.modules",
            {"huggingface_hub": types.SimpleNamespace(HfApi=lambda token=None: fake_api)},
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        publisher._publish(skill_dir, manifest, token="fake-token")

    assert excinfo.value.code == 1
    # Upload must NEVER happen when the privacy gate fails.
    fake_api.upload_folder.assert_not_called()


def test_publish_uses_ignore_patterns_for_secret_files(
    publisher: types.ModuleType,
) -> None:
    """The hardcoded ``_IGNORE_PATTERNS`` list must keep secrets / build artefacts off the Hub."""
    expected = {"*.pyc", "__pycache__", ".env", ".env.*", "*.key", "*.pem"}
    actual = set(publisher._IGNORE_PATTERNS)
    assert expected <= actual, f"Missing ignore patterns: {expected - actual}"


# ── --public flag + license gate (CLAUDE.md §1.9) ───────────────────────────


def test_public_visibility_error_allows_commercial(publisher: types.ModuleType) -> None:
    from openral_core.schemas import RSkillManifest

    m = RSkillManifest.from_yaml(  # apache-2.0 → commercial OK
        str(_REPO_ROOT / "rskills" / "rtdetr-coco-r18" / "rskill.yaml")
    )
    assert m.is_commercial_use_allowed
    assert publisher.public_visibility_error(m, public=True) is None
    assert publisher.public_visibility_error(m, public=False) is None


def test_public_visibility_error_refuses_non_commercial(publisher: types.ModuleType) -> None:
    from openral_core.schemas import RSkillManifest

    m = RSkillManifest.from_yaml(  # nvidia_non_commercial → must stay private
        str(_REPO_ROOT / "rskills" / "locateanything-3b-nf4" / "rskill.yaml")
    )
    assert not m.is_commercial_use_allowed
    err = publisher.public_visibility_error(m, public=True)
    assert err is not None and "non-commercial" in err
    # Private (no --public) is always fine.
    assert publisher.public_visibility_error(m, public=False) is None


def test_main_public_on_non_commercial_skill_exits_before_network(
    publisher: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A non-commercial manifest + --publish --public must abort at the license
    # gate (exit 1) without resolving a token or touching HfApi.
    nc_manifest = _VALID_MANIFEST_YAML.replace(
        "license: apache-2.0", "license: rlwrld_non_commercial"
    )
    skill_dir = _write_skill_dir(tmp_path, manifest_text=nc_manifest)
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(skill_dir), "--publish", "--public"])
    with pytest.raises(SystemExit) as excinfo:
        publisher.main()
    assert excinfo.value.code == 1


def test_publish_public_passes_private_false_and_runs_public_gate(
    publisher: types.ModuleType, tmp_path: Path
) -> None:
    skill_dir = _write_skill_dir(tmp_path)  # apache-2.0 fixture
    manifest = publisher._validate_manifest(skill_dir)

    fake_api = MagicMock()
    fake_api.create_repo.return_value = "https://huggingface.co/test/rskill-publisher-fixture"
    fake_api.model_info.return_value = MagicMock(private=False)  # repo is public

    with patch.dict(
        "sys.modules",
        {"huggingface_hub": types.SimpleNamespace(HfApi=lambda token=None: fake_api)},
    ):
        url = publisher._publish(skill_dir, manifest, token="fake-token", public=True)

    assert url == "https://huggingface.co/test/rskill-publisher-fixture"
    assert fake_api.create_repo.call_args.kwargs["private"] is False
    fake_api.upload_folder.assert_called_once()


def test_publish_public_aborts_when_repo_reports_private(
    publisher: types.ModuleType, tmp_path: Path
) -> None:
    # --public but the (reused) repo is private → abort, never upload.
    skill_dir = _write_skill_dir(tmp_path)
    manifest = publisher._validate_manifest(skill_dir)

    fake_api = MagicMock()
    fake_api.create_repo.return_value = "https://huggingface.co/test/rskill-publisher-fixture"
    fake_api.model_info.return_value = MagicMock(private=True)  # mismatch!

    with (
        patch.dict(
            "sys.modules",
            {"huggingface_hub": types.SimpleNamespace(HfApi=lambda token=None: fake_api)},
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        publisher._publish(skill_dir, manifest, token="fake-token", public=True)

    assert excinfo.value.code == 1
    fake_api.upload_folder.assert_not_called()


# ── Doc-validation gate (CLAUDE.md §3 rSkill packaging) ─────────────────────


def test_main_dry_run_exits_1_when_readme_missing(
    publisher: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run must surface the missing README and exit non-zero.

    Before the doc validator landed, a manifest-valid rSkill with no README
    would print "valid" and ship undocumented.
    """
    # Manifest exists; README does NOT.
    skill_dir = _write_skill_dir(tmp_path, readme=None)
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(skill_dir)])
    with pytest.raises(SystemExit) as excinfo:
        publisher.main()
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "README.md is required" in out
    assert "is NOT publish-ready" in out


def test_main_publish_aborts_when_readme_missing(
    publisher: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--publish`` must refuse upload when the README is missing — an upload
    that races past the doc validator would put an undocumented package on the Hub."""
    skill_dir = _write_skill_dir(tmp_path, readme=None)
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(skill_dir), "--publish"])

    # Gate must exit before touching huggingface_hub; fake_api has no upload_folder
    # configured so reaching it would fail loudly rather than silently pass.
    fake_api = MagicMock()
    with (
        patch.dict(
            "sys.modules",
            {"huggingface_hub": types.SimpleNamespace(HfApi=lambda token=None: fake_api)},
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        publisher.main()
    assert excinfo.value.code == 1
    fake_api.create_repo.assert_not_called()
    fake_api.upload_folder.assert_not_called()


def test_main_publish_aborts_when_description_is_template_default(
    publisher: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manifest description left at the scaffold default blocks publish."""
    bad_yaml = _VALID_MANIFEST_YAML.replace(
        "description: Publisher-test rSkill fixture — picks a cube on so100.",
        (
            'description: "Scaffold template generated by ral skill new. '
            'Edit this description, the manifest fields above, and the README.md."'
        ),
    )
    skill_dir = _write_skill_dir(tmp_path, manifest_text=bad_yaml)
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(skill_dir), "--publish"])

    fake_api = MagicMock()
    with (
        patch.dict(
            "sys.modules",
            {"huggingface_hub": types.SimpleNamespace(HfApi=lambda token=None: fake_api)},
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        publisher.main()
    assert excinfo.value.code == 1
    fake_api.create_repo.assert_not_called()
    fake_api.upload_folder.assert_not_called()


# ── Naming convention enforcement (rskill-<model>-<robot>-<task>-<quant>) ────


def test_enforce_repo_name_returns_manifest_when_compliant(
    publisher: types.ModuleType, tmp_path: Path
) -> None:
    skill_dir = _write_skill_dir(tmp_path)  # fixture name is already canonical
    manifest = publisher._validate_manifest(skill_dir)
    out = publisher._enforce_repo_name(skill_dir, manifest, fix_name=False)
    assert out.name == _FIXTURE_NAME


def test_main_dry_run_exits_1_on_noncompliant_name(
    publisher: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong name must fail the dry-run and print the suggested canonical name."""
    bad = _VALID_MANIFEST_YAML.replace(_FIXTURE_NAME, "test/rskill-wrong-name")
    skill_dir = _write_skill_dir(tmp_path, manifest_text=bad)
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(skill_dir)])
    with pytest.raises(SystemExit) as excinfo:
        publisher.main()
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    # The suggested name (derived from the manifest) is surfaced to the author.
    assert "test/rskill-smolvla-so100-" in out


def test_fix_name_rewrites_manifest_to_canonical(
    publisher: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--fix-name`` rewrites the on-disk name to a canonical one."""
    from openral_core import repo_name_is_canonical
    from openral_core.schemas import RSkillManifest

    bad = _VALID_MANIFEST_YAML.replace(_FIXTURE_NAME, "test/rskill-wrong-name")
    skill_dir = _write_skill_dir(tmp_path, manifest_text=bad)
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(skill_dir), "--fix-name"])
    publisher.main()  # dry-run + --fix-name → rewrites, then reports valid

    rewritten = RSkillManifest.from_yaml(str(skill_dir / "rskill.yaml"))
    assert repo_name_is_canonical(rewritten.name, kind="vla")
    assert rewritten.name.startswith("test/rskill-smolvla-so100-")
    # Only the top-level name changed; other content is intact.
    assert "model_family: smolvla" in (skill_dir / "rskill.yaml").read_text()


def test_enforce_repo_name_enforces_detector_kind(
    publisher: types.ModuleType, tmp_path: Path
) -> None:
    """Non-VLA kinds are NOT exempt: a non-canonical detector name hard-fails."""
    from openral_core.schemas import RSkillManifest

    # Force a deliberately non-canonical (hyphenated) name so the check fires
    # regardless of the fixture's current on-disk name.
    detector_yaml = str(_REPO_ROOT / "rskills" / "rtdetr-coco-r18" / "rskill.yaml")
    manifest = RSkillManifest.from_yaml(detector_yaml).model_copy(
        update={"name": "OpenRAL/rskill-rtdetr-coco-r18"}
    )
    with pytest.raises(SystemExit) as excinfo:
        publisher._enforce_repo_name(tmp_path, manifest, fix_name=False)
    assert excinfo.value.code == 1


def test_enforce_repo_name_accepts_canonical_playbook(
    publisher: types.ModuleType, tmp_path: Path
) -> None:
    """A playbook using the rskill-playbook-<name> grammar passes unchanged."""
    from openral_core.schemas import RSkillManifest

    playbook_yaml = str(_REPO_ROOT / "rskills" / "find-object" / "rskill.yaml")
    manifest = RSkillManifest.from_yaml(playbook_yaml).model_copy(
        update={"name": "OpenRAL/rskill-playbook-find_object"}
    )
    out = publisher._enforce_repo_name(tmp_path, manifest, fix_name=False)
    assert out.name == "OpenRAL/rskill-playbook-find_object"


# ── --gated (premium rSkills behind HF manual access approval) ──────────────


@pytest.mark.parametrize(
    "posture", ["apache-2.0", "mit", "bsd", "nvidia_open_model", "proprietary"]
)
def test_gated_visibility_error_allows_commercial_and_proprietary(
    publisher: types.ModuleType, tmp_path: Path, posture: str
) -> None:
    text = _VALID_MANIFEST_YAML.replace("license: apache-2.0", f"license: {posture}")
    manifest = publisher._validate_manifest(_write_skill_dir(tmp_path, manifest_text=text))
    assert publisher.gated_visibility_error(manifest, gated=True) is None


@pytest.mark.parametrize(
    "posture",
    [
        "permissive_research",
        "nvidia_non_commercial",
        "rlwrld_non_commercial",
        "cc-by-nc-sa-4.0",
        "unknown",
    ],
)
def test_gated_visibility_error_refuses_research_and_non_commercial(
    publisher: types.ModuleType, tmp_path: Path, posture: str
) -> None:
    text = _VALID_MANIFEST_YAML.replace("license: apache-2.0", f"license: {posture}")
    manifest = publisher._validate_manifest(_write_skill_dir(tmp_path, manifest_text=text))
    err = publisher.gated_visibility_error(manifest, gated=True)
    assert err is not None and "--gated refused" in err and posture in err
    assert publisher.gated_visibility_error(manifest, gated=False) is None


def test_main_gated_on_refused_posture_exits_before_network(
    publisher: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = _VALID_MANIFEST_YAML.replace("license: apache-2.0", "license: permissive_research")
    skill_dir = _write_skill_dir(tmp_path, manifest_text=text)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(skill_dir), "--publish", "--gated"])
    with (
        patch.object(publisher, "_publish") as publish,
        pytest.raises(SystemExit) as excinfo,
    ):
        publisher.main()
    assert excinfo.value.code == 1
    publish.assert_not_called()


def test_gated_and_public_are_mutually_exclusive(
    publisher: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill_dir = _write_skill_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(skill_dir), "--public", "--gated"])
    with pytest.raises(SystemExit) as excinfo:
        publisher.main()
    assert excinfo.value.code == 2  # argparse usage error


def _sibling(name: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(rfilename=name)


def _fake_api(infos: list[types.SimpleNamespace]) -> MagicMock:
    """A recording fake HfApi whose successive ``model_info`` calls return ``infos``."""
    fake_api = MagicMock()
    fake_api.create_repo.return_value = "https://huggingface.co/test/rskill-gated"
    fake_api.model_info.side_effect = infos
    return fake_api


def _gated_publish(publisher: types.ModuleType, tmp_path: Path, fake_api: MagicMock) -> None:
    """Run ``_publish(gated=True)`` against ``fake_api`` (the HF network boundary)."""
    skill_dir = _write_skill_dir(tmp_path)
    manifest = publisher._validate_manifest(skill_dir)
    with patch.dict(
        "sys.modules",
        {"huggingface_hub": types.SimpleNamespace(HfApi=lambda token=None: fake_api)},
    ):
        publisher._publish(skill_dir, manifest, token="fake-token", gated=True)


def test_publish_gated_sets_manual_gate_before_any_upload(
    publisher: types.ModuleType, tmp_path: Path
) -> None:
    fresh = types.SimpleNamespace(private=False, gated=False, siblings=[_sibling(".gitattributes")])
    gated = types.SimpleNamespace(private=False, gated="manual", siblings=[])
    api = _fake_api([fresh, gated])
    _gated_publish(publisher, tmp_path, api)

    assert api.create_repo.call_args.kwargs["private"] is False
    api.update_repo_settings.assert_called_once_with(
        _FIXTURE_NAME, gated="manual", repo_type="model"
    )
    order = [c[0] for c in api.method_calls]
    assert order[:4] == ["create_repo", "model_info", "update_repo_settings", "model_info"]
    assert order.index("update_repo_settings") < order.index("upload_folder")
    assert order.index("model_info", 2) < order.index("upload_folder")  # verified before upload


def test_publish_gated_republish_to_manual_repo_skips_settings_call(
    publisher: types.ModuleType, tmp_path: Path
) -> None:
    info = types.SimpleNamespace(
        private=False, gated="manual", siblings=[_sibling("rskill.yaml"), _sibling("README.md")]
    )
    api = _fake_api([info])
    _gated_publish(publisher, tmp_path, api)
    api.update_repo_settings.assert_not_called()
    api.upload_folder.assert_called_once()


@pytest.mark.parametrize(
    ("infos", "settings_called"),
    [
        # verification: the Hub did not apply manual gating
        (
            [
                types.SimpleNamespace(private=False, gated=False, siblings=[]),
                types.SimpleNamespace(private=False, gated="auto", siblings=[]),
            ],
            True,
        ),
        # an existing PRIVATE repo is refused, never changed
        ([types.SimpleNamespace(private=True, gated=False, siblings=[])], False),
        # an existing ungated repo that already holds files is refused, never changed
        (
            [
                types.SimpleNamespace(
                    private=False, gated=False, siblings=[_sibling("model.safetensors")]
                )
            ],
            False,
        ),
    ],
    ids=["gate-not-applied", "existing-private", "existing-ungated-with-files"],
)
def test_publish_gated_aborts_without_uploading(
    publisher: types.ModuleType,
    tmp_path: Path,
    infos: list[types.SimpleNamespace],
    settings_called: bool,
) -> None:
    api = _fake_api(infos)
    with pytest.raises(SystemExit) as excinfo:
        _gated_publish(publisher, tmp_path, api)
    assert excinfo.value.code == 1
    assert api.update_repo_settings.called is settings_called
    api.upload_folder.assert_not_called()
    api.upload_file.assert_not_called()
