"""Unit tests for ``openral rskill install / list / search`` CLI commands.

Only the HF Hub network boundary is doubled (CLAUDE.md §1.11) — manifests resolve to
real in-tree ``rskills/`` fixtures and the real loader, license guard, and registry
code paths execute. Registry is isolated per test via tmp_path.

Coverage: ``rskill list`` empty/populated/--json; ``rskill install`` happy path
(Apache), non-commercial license guard block, proprietary + --yes skip-prompt,
--revision pinning every Hub call; ``rskill search`` recorded org listing + real
manifest fixtures.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

from huggingface_hub.errors import EntryNotFoundError
from openral_cli.main import app
from openral_rskill.loader import InstalledRSkillEntry, rSkill
from typer.testing import CliRunner

runner = CliRunner()


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_entry(
    repo_id: str = "test/rskill-alpha",
    installed_at: str = "2026-01-01T00:00:00+00:00",
) -> InstalledRSkillEntry:
    return InstalledRSkillEntry(
        repo_id=repo_id,
        version="0.2.0",
        revision=None,
        local_dir=f"/tmp/skills/{repo_id.replace('/', '-')}",
        manifest_path=f"/tmp/skills/{repo_id.replace('/', '-')}/rskill.yaml",
        license="apache-2.0",
        role="s1",
        kind="vla",
        embodiment_tags=["so100_follower"],
        installed_at=installed_at,
    )


# ── ral skill list ─────────────────────────────────────────────────────────────


class TestSkillList:
    """Tests for ``openral rskill list`` — unified in-tree + HF-Hub-installed table."""

    def test_intree_only_lists_repo_rskills(self, tmp_path: Path) -> None:
        """With an empty installed registry, the listing still shows in-tree rSkills."""
        reg = tmp_path / "rskills.json"
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = runner.invoke(app, ["rskill", "list"])
        assert result.exit_code == 0
        assert "in-tree" in result.output

    def test_installed_entries_appear_under_installed_source(self, tmp_path: Path) -> None:
        """Hub-installed entries appear with the ``installed`` source tag."""
        reg = tmp_path / "rskills.json"
        rSkill._register(_make_entry("test/rskill-alpha"), reg)
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = runner.invoke(app, ["rskill", "list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        installed = [row for row in data if row.get("source") == "installed"]
        assert installed, "no installed rskill in JSON output"
        assert any(row["repo_id"] == "test/rskill-alpha" for row in installed)

    def test_json_output_is_valid(self, tmp_path: Path) -> None:
        """``--json`` flag must emit a valid JSON array with both sources tagged."""
        reg = tmp_path / "rskills.json"
        rSkill._register(_make_entry("test/rskill-alpha"), reg)
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = runner.invoke(app, ["rskill", "list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        sources = {row["source"] for row in data}
        assert "installed" in sources
        # In-tree rSkills are auto-discovered from rskills/.
        assert "in-tree" in sources

    def test_corrupt_registry_exits_nonzero(self, tmp_path: Path) -> None:
        """Corrupt registry JSON must cause non-zero exit with error message."""
        reg = tmp_path / "rskills.json"
        reg.write_text("{{NOT JSON", encoding="utf-8")
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = runner.invoke(app, ["rskill", "list"])
        assert result.exit_code != 0

    def test_multiple_installed_entries_all_shown(self, tmp_path: Path) -> None:
        """All installed entries must appear in the table output."""
        reg = tmp_path / "rskills.json"
        rSkill._register(_make_entry("test/rskill-a"), reg)
        rSkill._register(
            _make_entry("test/rskill-b", installed_at="2026-06-01T00:00:00+00:00"), reg
        )
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = runner.invoke(app, ["rskill", "list", "--json"])
        data = json.loads(result.output)
        repo_ids = {row["repo_id"] for row in data if row.get("source") == "installed"}
        assert "test/rskill-a" in repo_ids
        assert "test/rskill-b" in repo_ids


# ── ral skill install ──────────────────────────────────────────────────────────


_RSKILLS_DIR = Path(__file__).resolve().parents[2] / "rskills"


class _FakeHub:
    """HF network boundary fake for the install path (CLAUDE.md §1.11).

    Resolves ``hf_hub_download`` / ``snapshot_download`` to **real in-tree
    ``rskills/<id>/`` fixture dirs** so the real ``RSkillManifest.from_yaml``,
    the real license/provenance guards, and the real registry write in
    ``rSkill.from_pretrained`` all execute. Only the network is doubled.
    """

    def __init__(self, repo_map: dict[str, Path], error: Exception | None = None) -> None:
        self.repo_map = repo_map
        self.error = error
        self.download_calls: list[dict[str, object]] = []
        self.snapshot_calls: list[dict[str, object]] = []

    def hf_hub_download(self, *, repo_id: str, filename: str, **kwargs: object) -> str:
        self.download_calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        if self.error is not None:
            raise self.error
        return str(self.repo_map[repo_id] / filename)

    def snapshot_download(self, *, repo_id: str, **kwargs: object) -> str:
        self.snapshot_calls.append({"repo_id": repo_id, **kwargs})
        if self.error is not None:
            raise self.error
        return str(self.repo_map[repo_id])


class TestSkillInstall:
    """Tests for ``openral rskill install``.

    Only the HF network boundary is faked (`_FakeHub`); manifest parsing,
    license/provenance guards, and registry writes are the real code paths,
    exercised against real in-tree ``rskills/`` fixtures.
    """

    APACHE_REPO = "OpenRAL/rskill-act-franka_panda-libero-fp32"
    NONCOMMERCIAL_REPO = "OpenRAL/rskill-locateanything_3b-any-general-nf4"

    def _invoke(
        self,
        hub: _FakeHub,
        tmp_path: Path,
        *args: str,
    ) -> CliRunner.Result:
        reg = tmp_path / "rskills.json"
        env = dict(os.environ)
        env.pop("OPENRAL_ALLOW_NONCOMMERCIAL", None)
        env.pop("OPENRAL_REQUIRE_SIGNED_SKILLS", None)
        with (
            patch("huggingface_hub.hf_hub_download", new=hub.hf_hub_download),
            patch("huggingface_hub.snapshot_download", new=hub.snapshot_download),
            patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg),
            patch.dict(os.environ, env, clear=True),
        ):
            return runner.invoke(app, ["rskill", "install", *args], catch_exceptions=False)

    def test_happy_path_apache_exits_zero(self, tmp_path: Path) -> None:
        """Apache-2.0 skill install must exit 0, register, and not prompt."""
        hub = _FakeHub({self.APACHE_REPO: _RSKILLS_DIR / "act-libero"})
        result = self._invoke(hub, tmp_path, self.APACHE_REPO)
        assert result.exit_code == 0, result.output
        assert "Installed" in result.output
        assert "apache-2.0" in result.output
        # The real registry write happened with the real manifest's fields.
        reg = json.loads((tmp_path / "rskills.json").read_text(encoding="utf-8"))
        (entry,) = [e for e in reg if e["repo_id"] == self.APACHE_REPO]
        assert entry["license"] == "apache-2.0"

    def test_noncommercial_license_guard_blocks_install(self, tmp_path: Path) -> None:
        """The real license guard must reject non-commercial weights by default."""
        hub = _FakeHub({self.NONCOMMERCIAL_REPO: _RSKILLS_DIR / "locateanything-3b-nf4"})
        # --yes passes the CLI confirm gate so the loader's guard is what rejects.
        result = self._invoke(hub, tmp_path, self.NONCOMMERCIAL_REPO, "--yes")
        assert result.exit_code != 0
        assert "Install failed" in result.output
        # Nothing was registered.
        assert not (tmp_path / "rskills.json").exists()

    def test_proprietary_with_yes_skips_prompt(self, tmp_path: Path) -> None:
        """--yes must bypass the confirmation prompt for proprietary licenses."""
        # Real fixture content with only the license posture flipped — the
        # proprietary path is warn-and-proceed in the loader (vendor review is
        # out-of-band), so install completes.
        src = (_RSKILLS_DIR / "act-libero" / "rskill.yaml").read_text(encoding="utf-8")
        skill_dir = tmp_path / "rskill-helix"
        skill_dir.mkdir()
        (skill_dir / "rskill.yaml").write_text(
            src.replace('license: "apache-2.0"', 'license: "proprietary"'), encoding="utf-8"
        )
        hub = _FakeHub({"test-vendor/rskill-helix": skill_dir})
        result = self._invoke(hub, tmp_path, "test-vendor/rskill-helix", "--yes")
        assert result.exit_code == 0, result.output
        assert "Installed" in result.output

    def test_revision_flag_reaches_hub_boundary(self, tmp_path: Path) -> None:
        """--revision <sha> must be forwarded to both Hub download calls."""
        hub = _FakeHub({self.APACHE_REPO: _RSKILLS_DIR / "act-libero"})
        result = self._invoke(hub, tmp_path, self.APACHE_REPO, "--revision", "deadbeef")
        assert result.exit_code == 0, result.output
        # CLI manifest fetch + loader manifest fetch + snapshot all pin it.
        assert all(c["revision"] == "deadbeef" for c in hub.download_calls)
        assert all(c["revision"] == "deadbeef" for c in hub.snapshot_calls)

    def test_no_revision_prints_tip(self, tmp_path: Path) -> None:
        """Installing without --revision must print a pinning tip."""
        hub = _FakeHub({self.APACHE_REPO: _RSKILLS_DIR / "act-libero"})
        result = self._invoke(hub, tmp_path, self.APACHE_REPO)
        assert result.exit_code == 0, result.output
        assert "Pin a revision" in result.output

    def test_manifest_fetch_error_exits_nonzero(self, tmp_path: Path) -> None:
        """Network error fetching rskill.yaml must exit non-zero."""
        hub = _FakeHub({}, error=RuntimeError("connection refused"))
        result = self._invoke(hub, tmp_path, self.APACHE_REPO)
        assert result.exit_code != 0
        assert "Failed to fetch manifest" in result.output

    def test_bare_name_suggests_org_without_network(self, tmp_path: Path) -> None:
        """An org-less id must fail fast with an OpenRAL/ suggestion and never hit HF."""
        hub = _FakeHub({}, error=AssertionError("must not download"))
        result = self._invoke(hub, tmp_path, "rskill-qwen35_4b-any-general-nf4")
        assert result.exit_code != 0
        # Suggests the canonical org-qualified id …
        assert "OpenRAL/rskill-qwen35_4b-any-general-nf4" in result.output
        # … and points at the discovery command.
        assert "rskill search" in result.output
        # The org-less guard short-circuits before any network call.
        assert hub.download_calls == []
        assert hub.snapshot_calls == []


# ── ral skill search ───────────────────────────────────────────────────────────


class TestSkillSearch:
    """Tests for ``openral rskill search``.

    The HF network boundary is the only thing doubled: a *recorded* set of
    ``OpenRAL/*`` repo ids stands in for
    ``HfApi.list_models(author="OpenRAL", filter="rskill")`` and each hit's
    manifest is resolved to a *real* in-tree ``rskills/<id>/rskill.yaml``
    fixture (CLAUDE.md §1.11 — recorded responses + real fixtures, no
    placeholders). Query/facet matching happens locally in
    ``openral_rskill.hub_search.search_hub_rskills``, so these tests exercise
    the real matching code, not a fake's approximation of it. The local
    rSkill registry (for the ``installed`` marker) is isolated per test via
    ``tmp_path``, following the same pattern as ``TestSkillList``.
    """

    REPO_ROOT: ClassVar[Path] = Path(__file__).resolve().parents[2]

    # A fourth id whose manifest is a copy of act-aloha's with a renamed
    # `name:` field — matches no real `rskills/<dir>` manifest, so it
    # exercises the `local=None` ("not available locally") branch without
    # inventing a placeholder fixture (CLAUDE.md §1.11).
    _NOT_IN_TREE_ID: ClassVar[str] = "OpenRAL/rskill-act-aloha-not-in-tree"

    # Has no rskill.yaml on the Hub and must be excluded from `hits` (but
    # named explicitly in the skip notice — CLAUDE.md §1.4).
    _BROKEN_ID: ClassVar[str] = "OpenRAL/rskill-broken-no-manifest"

    # Recorded OpenRAL org listing — mirrors real in-tree skills.
    _RECORDED_IDS: ClassVar[list[str]] = [
        "OpenRAL/rskill-act-aloha-aloha_transfer_cube-fp32",
        "OpenRAL/rskill-omdet_turbo-any-locator-fp16",
        _NOT_IN_TREE_ID,
        _BROKEN_ID,
    ]
    _MANIFEST_FIXTURES: ClassVar[dict[str, str]] = {
        "OpenRAL/rskill-act-aloha-aloha_transfer_cube-fp32": "rskills/act-aloha/rskill.yaml",
        "OpenRAL/rskill-omdet_turbo-any-locator-fp16": "rskills/omdet-turbo-locator/rskill.yaml",
    }

    def _run_search(
        self,
        *args: str,
        tmp_path: Path,
        recorded_ids: list[str] | None = None,
        capture: dict[str, object] | None = None,
    ) -> CliRunner.Result:
        recorded = recorded_ids if recorded_ids is not None else self._RECORDED_IDS
        cap = capture if capture is not None else {}

        not_in_tree_dir = tmp_path / "not-in-tree"
        not_in_tree_dir.mkdir(exist_ok=True)
        src = (self.REPO_ROOT / "rskills/act-aloha/rskill.yaml").read_text(encoding="utf-8")
        (not_in_tree_dir / "rskill.yaml").write_text(
            src.replace(
                'name: "OpenRAL/rskill-act-aloha-aloha_transfer_cube-fp32"',
                f'name: "{self._NOT_IN_TREE_ID}"',
            ),
            encoding="utf-8",
        )

        def fake_list_models(
            *, author: str, filter: str | None = None, limit: int | None = None, **_: object
        ) -> list[SimpleNamespace]:
            cap["author"] = author
            cap["filter"] = filter
            cap["limit"] = limit
            return [SimpleNamespace(id=i, tags=["OpenRAL", "rskill"]) for i in recorded]

        class _FakeHfApi:
            """Records the org listing query the way ``HfApi`` would answer it."""

            def list_models(
                self,
                *,
                author: str,
                filter: str | None = None,
                limit: int | None = None,
                **_: object,
            ) -> list[SimpleNamespace]:
                return fake_list_models(author=author, filter=filter, limit=limit)

        def fake_download(*, repo_id: str, filename: str, **_: object) -> str:
            if repo_id == self._NOT_IN_TREE_ID:
                return str(not_in_tree_dir / filename)
            rel = self._MANIFEST_FIXTURES.get(repo_id)
            if rel is None:
                raise EntryNotFoundError(f"no {filename} for {repo_id}")
            return str(self.REPO_ROOT / rel)

        reg = tmp_path / "rskills.json"
        with (
            patch("huggingface_hub.HfApi", return_value=_FakeHfApi()),
            patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
            patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg),
            # Widen the Rich console so long repo ids render unwrapped in the table.
            patch.dict(os.environ, {"COLUMNS": "200"}),
        ):
            return runner.invoke(app, ["rskill", "search", *args], catch_exceptions=False)

    def test_lists_valid_skills_with_install_hint(self, tmp_path: Path) -> None:
        """A query token present in both repo ids must surface both, plus the hint."""
        result = self._run_search("fp", tmp_path=tmp_path)
        assert result.exit_code == 0
        assert "rskill-act-aloha-aloha_transfer_cube-fp32" in result.output
        assert "rskill-omdet_turbo-any-locator-fp16" in result.output
        assert "rskill install" in result.output

    def test_searches_openral_org_by_tag_no_hub_search(self, tmp_path: Path) -> None:
        """The org is listed via the ``rskill`` tag; no Hub-side ``search=`` is sent."""
        cap: dict[str, object] = {}
        result = self._run_search(tmp_path=tmp_path, capture=cap)
        assert result.exit_code == 0
        assert cap["author"] == "OpenRAL"
        assert cap["filter"] == "rskill"
        assert cap["limit"] is None

    def test_query_matches_description_only_word(self, tmp_path: Path) -> None:
        """Multi-token query matching words that appear ONLY in the description.

        Neither "bimanual" nor "chunks" appears in the act-aloha repo id,
        manifest name, kind, role, or embodiment tags — only in its
        ``description`` — so this proves matching is not repo-id-only.
        """
        result = self._run_search("bimanual chunks", tmp_path=tmp_path)
        assert result.exit_code == 0
        assert "rskill-act-aloha-aloha_transfer_cube-fp32" in result.output
        assert "rskill-omdet_turbo-any-locator-fp16" not in result.output

    def test_unquoted_multi_word_query_matches_like_quoted(self, tmp_path: Path) -> None:
        """``search pick place`` (two argv tokens, no quotes) must behave the same
        as ``search "pick place"`` — QUERY is variadic and re-joined with spaces."""
        result = self._run_search("bimanual", "chunks", tmp_path=tmp_path)
        assert result.exit_code == 0
        assert "rskill-act-aloha-aloha_transfer_cube-fp32" in result.output
        assert "rskill-omdet_turbo-any-locator-fp16" not in result.output

    def test_manifestless_repo_excluded_and_skip_surfaced(self, tmp_path: Path) -> None:
        """The broken repo must not appear as a hit, but its id IS named in the
        skip notice (CLAUDE.md §1.4 — explicit, not just a bare count)."""
        result = self._run_search(tmp_path=tmp_path)
        assert result.exit_code == 0
        assert "skipped" in result.output.lower()
        assert self._BROKEN_ID in result.output

    def test_kind_filter_narrows_results(self, tmp_path: Path) -> None:
        result = self._run_search("--kind", "detector", tmp_path=tmp_path)
        assert result.exit_code == 0
        assert "rskill-omdet_turbo-any-locator-fp16" in result.output
        assert "rskill-act-aloha-aloha_transfer_cube-fp32" not in result.output

    def test_license_filter_narrows_results(self, tmp_path: Path) -> None:
        result = self._run_search("--license", "mit", tmp_path=tmp_path)
        assert result.exit_code == 0
        assert "rskill-act-aloha-aloha_transfer_cube-fp32" in result.output
        assert "rskill-omdet_turbo-any-locator-fp16" not in result.output

    def test_family_filter_narrows_results(self, tmp_path: Path) -> None:
        """``--family act`` must keep the ACT policy and drop the (family-less) detector."""
        result = self._run_search("--family", "act", tmp_path=tmp_path)
        assert result.exit_code == 0
        assert "rskill-act-aloha-aloha_transfer_cube-fp32" in result.output
        assert "rskill-omdet_turbo-any-locator-fp16" not in result.output

    def test_limit_caps_rows_after_sorting(self, tmp_path: Path) -> None:
        """``--limit 1`` must keep only the first hit in repo_id sort order."""
        result = self._run_search("--limit", "1", tmp_path=tmp_path)
        assert result.exit_code == 0
        assert "rskill-act-aloha-aloha_transfer_cube-fp32" in result.output
        assert "rskill-omdet_turbo-any-locator-fp16" not in result.output

    def test_default_limit_is_unlimited(self, tmp_path: Path) -> None:
        """The default (no ``--limit``) must show every match, not just the first 50."""
        result = self._run_search(tmp_path=tmp_path)
        assert result.exit_code == 0
        assert "rskill-act-aloha-aloha_transfer_cube-fp32" in result.output
        assert "rskill-omdet_turbo-any-locator-fp16" in result.output

    def test_results_sorted_by_repo_id(self, tmp_path: Path) -> None:
        """Rows must appear in ascending ``repo_id`` order regardless of Hub listing order."""
        result = self._run_search(tmp_path=tmp_path)
        assert result.exit_code == 0
        act_pos = result.output.index("rskill-act-aloha-aloha_transfer_cube-fp32")
        omdet_pos = result.output.index("rskill-omdet_turbo-any-locator-fp16")
        assert act_pos < omdet_pos

    def test_local_column_shows_in_tree_and_missing(self, tmp_path: Path) -> None:
        """The ``local`` column: ``in-tree`` for a real in-tree manifest, ``—`` otherwise."""
        recorded = [self._RECORDED_IDS[0], self._NOT_IN_TREE_ID]
        result = self._run_search("act-aloha", recorded_ids=recorded, tmp_path=tmp_path)
        assert result.exit_code == 0
        assert "in-tree" in result.output
        assert "—" in result.output

    def test_no_results_is_friendly(self, tmp_path: Path) -> None:
        result = self._run_search("nonesuch", recorded_ids=[], tmp_path=tmp_path)
        assert result.exit_code == 0
        assert "No rSkills" in result.output

    def test_json_output_is_valid(self, tmp_path: Path) -> None:
        result = self._run_search(
            "--json", recorded_ids=list(self._MANIFEST_FIXTURES), tmp_path=tmp_path
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        ids = {row["repo_id"] for row in payload}
        assert ids == {
            "OpenRAL/rskill-act-aloha-aloha_transfer_cube-fp32",
            "OpenRAL/rskill-omdet_turbo-any-locator-fp16",
        }
        for row in payload:
            assert {"repo_id", "kind", "role", "license", "local", "model_family"} <= row.keys()
        by_id = {row["repo_id"]: row for row in payload}
        act = by_id["OpenRAL/rskill-act-aloha-aloha_transfer_cube-fp32"]
        assert act["local"] == "in-tree"
        assert act["model_family"] == "act"

    def test_json_output_parseable_with_a_skipped_repo_present(self, tmp_path: Path) -> None:
        """A manifest-less repo in the recorded org listing must never leak into
        stdout as anything but valid JSON — no debug/skip line before or after
        the array (CLAUDE.md §1.4: structlog's default PrintLogger, unconfigured
        by the CLI, would otherwise write straight to stdout and break
        ``openral rskill search --json | jq``)."""
        recorded = [*self._MANIFEST_FIXTURES, self._BROKEN_ID]
        result = self._run_search("--json", recorded_ids=recorded, tmp_path=tmp_path)
        assert result.exit_code == 0
        payload = json.loads(result.output)  # raises if anything but the array hit stdout
        ids = {row["repo_id"] for row in payload}
        assert ids == {
            "OpenRAL/rskill-act-aloha-aloha_transfer_cube-fp32",
            "OpenRAL/rskill-omdet_turbo-any-locator-fp16",
        }
        assert self._BROKEN_ID not in ids
