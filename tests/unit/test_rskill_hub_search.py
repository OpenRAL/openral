"""Unit tests for ``openral_rskill.hub_search.search_hub_rskills``.

Only the HF Hub network boundary is doubled (CLAUDE.md §1.11): a recording
fake stands in for ``HfApi.list_models`` / ``hf_hub_download``, while every
matched manifest resolves to a *real* in-tree ``rskills/<dir>/rskill.yaml``
fixture so the real ``RSkillManifest`` validation, query matching, facet
filtering, and local-registry lookup all execute. The local rSkill registry
(for the ``installed`` marker) is isolated per test via ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from huggingface_hub.errors import EntryNotFoundError
from openral_rskill.hub_search import HubRSkillHit, search_hub_rskills
from openral_rskill.loader import rSkill
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ACT_ALOHA_ID = "OpenRAL/rskill-act-aloha-aloha_transfer_cube-fp32"
_OMDET_ID = "OpenRAL/rskill-omdet_turbo-any-locator-fp16"
_NOT_IN_TREE_ID = "OpenRAL/rskill-act-aloha-not-in-tree"
_BROKEN_ID = "OpenRAL/rskill-broken-no-manifest"

_MANIFEST_FIXTURES: dict[str, Path] = {
    _ACT_ALOHA_ID: _REPO_ROOT / "rskills/act-aloha/rskill.yaml",
    _OMDET_ID: _REPO_ROOT / "rskills/omdet-turbo-locator/rskill.yaml",
}


def _not_in_tree_manifest_path(tmp_path: Path) -> Path:
    """A copy of the act-aloha fixture with a renamed ``name:`` field.

    Matches no real ``rskills/<dir>`` manifest, so it exercises
    ``local=None`` without a placeholder fixture (CLAUDE.md §1.11).
    """
    src = _MANIFEST_FIXTURES[_ACT_ALOHA_ID].read_text(encoding="utf-8")
    out = tmp_path / "rskill.yaml"
    out.write_text(
        src.replace(f'name: "{_ACT_ALOHA_ID}"', f'name: "{_NOT_IN_TREE_ID}"'), encoding="utf-8"
    )
    return out


class _FakeHfApi:
    """Records the org listing query the way ``HfApi`` would answer it."""

    def __init__(self, recorded: list[str], calls: list[dict[str, object]]) -> None:
        self._recorded = recorded
        self._calls = calls

    def list_models(
        self, *, author: str, filter: str | None = None, limit: int | None = None, **_: object
    ) -> list[SimpleNamespace]:
        self._calls.append({"author": author, "filter": filter, "limit": limit})
        return [SimpleNamespace(id=i, tags=["OpenRAL", "rskill"]) for i in self._recorded]


def _patch_hub(
    recorded: list[str],
    manifest_paths: dict[str, Path],
    calls: list[dict[str, object]] | None = None,
) -> Any:
    """Patch the HF Hub boundary: org listing + per-repo manifest fetch."""
    calls = calls if calls is not None else []

    def fake_download(*, repo_id: str, filename: str, **_: object) -> str:
        path = manifest_paths.get(repo_id)
        if path is None:
            raise EntryNotFoundError(f"no {filename} for {repo_id}")
        return str(path)

    return (
        patch("huggingface_hub.HfApi", return_value=_FakeHfApi(recorded, calls)),
        patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
    )


class TestSearchHubRSkills:
    def test_lists_org_by_tag_no_hub_side_search(self, tmp_path: Path) -> None:
        """Exactly one ``list_models(author=org, filter="rskill")`` call, no ``search=``."""
        calls: list[dict[str, object]] = []
        p1, p2 = _patch_hub([_ACT_ALOHA_ID], _MANIFEST_FIXTURES, calls)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            search_hub_rskills(org="OpenRAL")
        assert calls == [{"author": "OpenRAL", "filter": "rskill", "limit": None}]

    def test_empty_query_matches_every_valid_manifest(self, tmp_path: Path) -> None:
        p1, p2 = _patch_hub([_ACT_ALOHA_ID, _OMDET_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        assert {h.repo_id for h in result.hits} == {_ACT_ALOHA_ID, _OMDET_ID}
        assert result.skipped == 0
        assert result.skipped_repo_ids == ()
        assert result.inspected == 2

    def test_hits_sorted_by_repo_id(self, tmp_path: Path) -> None:
        """Hub listing order (omdet before act) must not affect the sorted result."""
        p1, p2 = _patch_hub([_OMDET_ID, _ACT_ALOHA_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        assert [h.repo_id for h in result.hits] == [_ACT_ALOHA_ID, _OMDET_ID]

    def test_manifestless_repo_is_skipped_not_raised(self, tmp_path: Path) -> None:
        p1, p2 = _patch_hub([_ACT_ALOHA_ID, _BROKEN_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        assert [h.repo_id for h in result.hits] == [_ACT_ALOHA_ID]
        assert result.skipped == 1
        assert result.skipped_repo_ids == (_BROKEN_ID,)
        assert result.inspected == 2

    def test_malformed_yaml_is_skipped_not_raised(self, tmp_path: Path) -> None:
        """A repo whose ``rskill.yaml`` fails schema validation counts as skipped."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: not-a-valid-manifest\nlicense: apache-2.0\n", encoding="utf-8")
        paths = {**_MANIFEST_FIXTURES, "OpenRAL/rskill-bad-manifest": bad}
        p1, p2 = _patch_hub([_ACT_ALOHA_ID, "OpenRAL/rskill-bad-manifest"], paths)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        assert [h.repo_id for h in result.hits] == [_ACT_ALOHA_ID]
        assert result.skipped == 1
        assert result.skipped_repo_ids == ("OpenRAL/rskill-bad-manifest",)

    def test_skipped_repo_ids_are_sorted_and_count_matches(self, tmp_path: Path) -> None:
        """Multiple skipped repos → ``skipped_repo_ids`` sorted, ``skipped`` matching len()."""
        other_broken = "OpenRAL/rskill-also-broken"
        p1, p2 = _patch_hub([other_broken, _ACT_ALOHA_ID, _BROKEN_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        assert result.skipped_repo_ids == tuple(sorted([_BROKEN_ID, other_broken]))
        assert result.skipped == len(result.skipped_repo_ids) == 2

    def test_skip_is_never_logged(self, tmp_path: Path) -> None:
        """A skipped repo must not be logged at any level (CLAUDE.md §1.4 / no stdout
        pollution — the CLI never calls structlog.configure(), so any per-repo log
        record would write straight to stdout via structlog's default PrintLogger and
        corrupt ``openral rskill search --json | jq``)."""
        p1, p2 = _patch_hub([_ACT_ALOHA_ID, _BROKEN_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with (
            p1,
            p2,
            patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg),
            patch("openral_rskill.hub_search.log") as fake_log,
        ):
            search_hub_rskills()
        fake_log.debug.assert_not_called()
        fake_log.info.assert_not_called()

    def test_query_token_matches_description_only_word(self, tmp_path: Path) -> None:
        """ "bimanual"/"chunks" appear ONLY in act-aloha's description, not its id/fields."""
        p1, p2 = _patch_hub([_ACT_ALOHA_ID, _OMDET_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills("bimanual chunks")
        assert [h.repo_id for h in result.hits] == [_ACT_ALOHA_ID]

    def test_query_is_case_insensitive(self, tmp_path: Path) -> None:
        p1, p2 = _patch_hub([_ACT_ALOHA_ID, _OMDET_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills("BIMANUAL")
        assert [h.repo_id for h in result.hits] == [_ACT_ALOHA_ID]

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"kind": "detector"}, [_OMDET_ID]),
            ({"role": "s1"}, [_ACT_ALOHA_ID, _OMDET_ID]),
            ({"embodiment": "aloha"}, [_ACT_ALOHA_ID]),
            ({"license": "mit"}, [_ACT_ALOHA_ID]),
            ({"family": "act"}, [_ACT_ALOHA_ID]),
            ({"family": "smolvla"}, []),
        ],
    )
    def test_facet_filters(
        self, tmp_path: Path, kwargs: dict[str, str], expected: list[str]
    ) -> None:
        p1, p2 = _patch_hub([_ACT_ALOHA_ID, _OMDET_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills(**kwargs)
        assert sorted(h.repo_id for h in result.hits) == sorted(expected)

    def test_limit_caps_after_sorting(self, tmp_path: Path) -> None:
        p1, p2 = _patch_hub([_OMDET_ID, _ACT_ALOHA_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills(limit=1)
        assert [h.repo_id for h in result.hits] == [_ACT_ALOHA_ID]

    def test_limit_none_is_unlimited(self, tmp_path: Path) -> None:
        p1, p2 = _patch_hub([_ACT_ALOHA_ID, _OMDET_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills(limit=None)
        assert len(result.hits) == 2

    def test_local_marker_in_tree(self, tmp_path: Path) -> None:
        """A repo id whose manifest ``name`` matches a real in-tree manifest → "in-tree"."""
        p1, p2 = _patch_hub([_ACT_ALOHA_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        (hit,) = result.hits
        assert hit.local == "in-tree"

    def test_in_tree_marker_uses_hub_repo_id_not_stale_remote_name(self, tmp_path: Path) -> None:
        """A Hub repo id in-tree locally, whose fetched manifest has a DIFFERENT
        (stale) ``name`` field, must still read as in-tree — the ``local``
        marker keys off the Hub ``repo_id``, never the remote manifest's own
        (possibly out of date) ``name``.
        """
        stale = tmp_path / "stale.yaml"
        src = _MANIFEST_FIXTURES[_ACT_ALOHA_ID].read_text(encoding="utf-8")
        stale.write_text(
            src.replace(f'name: "{_ACT_ALOHA_ID}"', f'name: "{_NOT_IN_TREE_ID}"'),
            encoding="utf-8",
        )
        # Published AT the real in-tree repo id, but its own rskill.yaml still
        # declares the old name — simulating a rename that forgot to update it.
        paths = {_ACT_ALOHA_ID: stale}
        p1, p2 = _patch_hub([_ACT_ALOHA_ID], paths)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        (hit,) = result.hits
        assert hit.repo_id == _ACT_ALOHA_ID
        assert hit.manifest.name == _NOT_IN_TREE_ID
        assert hit.local == "in-tree"

    def test_manifest_name_alone_does_not_imply_in_tree(self, tmp_path: Path) -> None:
        """The converse: a manifest whose OWN ``name`` matches an in-tree skill,
        but whose Hub ``repo_id`` does not, must NOT read as in-tree.
        """
        mismatched_id = "OpenRAL/rskill-some-other-repo-id"
        paths = {mismatched_id: _MANIFEST_FIXTURES[_ACT_ALOHA_ID]}
        p1, p2 = _patch_hub([mismatched_id], paths)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        (hit,) = result.hits
        assert hit.repo_id == mismatched_id
        assert hit.manifest.name == _ACT_ALOHA_ID
        assert hit.local is None

    def test_local_marker_installed(self, tmp_path: Path) -> None:
        """A repo id registered in the local install registry → "installed"."""
        not_in_tree = _not_in_tree_manifest_path(tmp_path)
        paths = {**_MANIFEST_FIXTURES, _NOT_IN_TREE_ID: not_in_tree}
        reg = tmp_path / "rskills.json"
        reg.write_text(
            json.dumps(
                [
                    {
                        "repo_id": _NOT_IN_TREE_ID,
                        "version": "0.1.0",
                        "revision": None,
                        "local_dir": str(tmp_path),
                        "manifest_path": str(not_in_tree),
                        "license": "mit",
                        "role": "s1",
                        "embodiment_tags": ["aloha"],
                        "installed_at": "2026-01-01T00:00:00+00:00",
                    }
                ]
            ),
            encoding="utf-8",
        )
        p1, p2 = _patch_hub([_NOT_IN_TREE_ID], paths)
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        (hit,) = result.hits
        assert hit.local == "installed"

    def test_local_marker_none_when_neither(self, tmp_path: Path) -> None:
        not_in_tree = _not_in_tree_manifest_path(tmp_path)
        paths = {**_MANIFEST_FIXTURES, _NOT_IN_TREE_ID: not_in_tree}
        reg = tmp_path / "rskills.json"
        p1, p2 = _patch_hub([_NOT_IN_TREE_ID], paths)
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        (hit,) = result.hits
        assert hit.local is None

    def test_corrupt_registry_treated_as_empty_not_raised(self, tmp_path: Path) -> None:
        """A corrupt local registry must not fail the search (CLAUDE.md §1.1 — no silent crash)."""
        reg = tmp_path / "rskills.json"
        reg.write_text("{{NOT JSON", encoding="utf-8")
        p1, p2 = _patch_hub([_ACT_ALOHA_ID], _MANIFEST_FIXTURES)
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        (hit,) = result.hits
        # in-tree still wins over a broken installed-registry read.
        assert hit.local == "in-tree"

    def test_corrupt_registry_is_logged_not_silent(self, tmp_path: Path) -> None:
        """CLAUDE.md §1.4 (explicit beats implicit): a corrupt registry must warn, not
        just quietly disappear — the un-actionable-installed-marker case is real, but
        it must be visible to whoever runs `openral rskill search`."""
        reg = tmp_path / "rskills.json"
        reg.write_text("{{NOT JSON", encoding="utf-8")
        p1, p2 = _patch_hub([_ACT_ALOHA_ID], _MANIFEST_FIXTURES)
        with (
            p1,
            p2,
            patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg),
            patch("openral_rskill.hub_search.log") as fake_log,
        ):
            search_hub_rskills()
        fake_log.warning.assert_called_once()
        assert fake_log.warning.call_args.args[0] == "rskill.hub_search.registry_unreadable"

    def test_hub_tags_are_surfaced(self, tmp_path: Path) -> None:
        p1, p2 = _patch_hub([_ACT_ALOHA_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        (hit,) = result.hits
        assert hit.hub_tags == ("OpenRAL", "rskill")

    def test_hit_is_frozen(self, tmp_path: Path) -> None:
        p1, p2 = _patch_hub([_ACT_ALOHA_ID], _MANIFEST_FIXTURES)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = search_hub_rskills()
        (hit,) = result.hits
        assert isinstance(hit, HubRSkillHit)
        with pytest.raises(ValidationError):
            hit.local = "installed"  # type: ignore[misc]  # reason: mutation attempt on a frozen model

    def test_org_argument_is_forwarded(self, tmp_path: Path) -> None:
        calls: list[dict[str, object]] = []
        p1, p2 = _patch_hub([_ACT_ALOHA_ID], _MANIFEST_FIXTURES, calls)
        reg = tmp_path / "rskills.json"
        with p1, p2, patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            search_hub_rskills(org="SomeOtherOrg")
        assert calls[0]["author"] == "SomeOtherOrg"


class TestLoaderStillWorks:
    """Guards against accidentally breaking ``rSkill.list_installed`` while wiring
    the ``installed`` marker (regression companion to TestSearchHubRSkills)."""

    def test_list_installed_empty_with_isolated_registry(self, tmp_path: Path) -> None:
        reg = tmp_path / "rskills.json"
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            assert rSkill.list_installed() == []
