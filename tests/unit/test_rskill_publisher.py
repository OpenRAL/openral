"""Smoke tests for ``tools/rskill_publisher.py``.

The publisher is a CLI script with no Python package wrapper; we import it
via :mod:`importlib` from its file path so the test exercises exactly the
script users invoke with ``uv run python tools/rskill_publisher.py``.

All HF Hub I/O (``HfApi.create_repo``, ``upload_folder``, ``model_info``)
is mocked.  The tests only exercise the local validation, dry-run, and
token-resolution paths — the actual upload behaviour is covered by the
manual smoke run that the publisher's docstring documents.

Coverage
--------
- ``_resolve_token`` — preference order: ``--token`` arg > ``HF_TOKEN`` env
  > ``HUGGINGFACE_HUB_TOKEN`` env > exit 1 with a hint.
- ``_validate_manifest`` — exits 1 on a missing or malformed manifest;
  returns the parsed :class:`RSkillManifest` on success.
- ``main`` — dry-run path (no ``--publish``) succeeds and emits the
  "valid" message on stdout.
- ``main`` — exits 1 when ``skill_dir`` is not a directory.
- ``main`` — exits 1 when the manifest is missing inside a real directory.
- ``_publish`` — creates the repo with ``private=True`` and runs the
  privacy gate; it MUST abort if the API reports the repo as public
  (a regression here is a security issue per CLAUDE.md §7.2 / §12).
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


# ── _publish: privacy gate (CLAUDE.md §7.2 / §12 — security regression guard)


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
    §7.2 ban on bundling closed-source weights without the license guard).
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


# ── --public flag + license gate (CLAUDE.md §9) ─────────────────────────────


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


# ── Doc-validation gate (CLAUDE.md §6.4) ────────────────────────────────────


def test_main_dry_run_exits_1_when_readme_missing(
    publisher: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run must surface the missing README and exit non-zero.

    Before the doc validator landed, dry-run only checked the manifest
    schema — a manifest-valid rSkill with no README would say "valid"
    and authors would then publish an undocumented package. The gate
    now blocks that path.
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
    """``--publish`` must refuse upload when the README is missing.

    Critical safety gate: an upload that races past the doc validator
    would put a non-documented package on the Hub.
    """
    skill_dir = _write_skill_dir(tmp_path, readme=None)
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    monkeypatch.setattr(sys, "argv", ["rskill_publisher", str(skill_dir), "--publish"])

    # Module-level network stub: if the publisher reaches HfApi, the
    # test fails the way we want — by raising AttributeError on
    # `upload_folder` since the fake never sets it. But the gate
    # should exit BEFORE we ever touch huggingface_hub.
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
