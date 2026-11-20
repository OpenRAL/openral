"""Deploy-start rSkill auto-install (``openral_cli.rskill_install``).

Only the HF network boundary is doubled (CLAUDE.md §1.11): downloads resolve to real
in-tree ``rskills/<id>/`` fixtures, so the real manifest parsing, licence guard and
registry write in ``rSkill.from_pretrained`` run.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError
from openral_cli.rskill_install import auto_install_rskills, install_listed_rskills

_RSKILLS = Path(__file__).resolve().parents[2] / "rskills"
APACHE = "OpenRAL/rskill-act-franka_panda-libero-fp32"
NONCOMMERCIAL = "OpenRAL/rskill-locateanything_3b-any-general-nf4"
GATED = "OpenRAL/rskill-act-so101_follower-restock-bf16"
_REPOS = {APACHE: _RSKILLS / "act-libero", NONCOMMERCIAL: _RSKILLS / "locateanything-3b-nf4"}


class _Hub:
    """The Hub at its network boundary: a repo map, a gated repo, or no Hub at all."""

    def __init__(self, *, offline: bool = False) -> None:
        self.offline = offline
        self.snapshots: list[str] = []

    def _check(self, repo_id: str) -> None:
        if self.offline:
            raise LocalEntryNotFoundError("Cannot reach huggingface.co (offline)")
        if repo_id == GATED:
            req = httpx.Request("GET", f"https://huggingface.co/{repo_id}")
            raise GatedRepoError("403 gated", response=httpx.Response(403, request=req))

    def hf_hub_download(self, *, repo_id: str, filename: str, **_: object) -> str:
        self._check(repo_id)
        return str(_REPOS[repo_id] / filename)

    def snapshot_download(self, *, repo_id: str, **_: object) -> str:
        self._check(repo_id)
        self.snapshots.append(repo_id)
        return str(_REPOS[repo_id])


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[Path]:
    reg = tmp_path / "rskills.json"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPENRAL_ALLOW_NONCOMMERCIAL", "OPENRAL_REQUIRE_SIGNED_SKILLS")
    }
    with (
        patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg),
        patch.dict(os.environ, env, clear=True),
    ):
        yield reg


def _run(hub: _Hub, ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    lines: list[str] = []
    with (
        patch("huggingface_hub.hf_hub_download", new=hub.hf_hub_download),
        patch("huggingface_hub.snapshot_download", new=hub.snapshot_download),
    ):
        installed, skipped = install_listed_rskills(ids, echo=lines.append)
    return installed, skipped, lines


def test_installs_a_missing_permissive_rskill(registry: Path) -> None:
    installed, skipped, lines = _run(_Hub(), [APACHE])
    assert (installed, skipped) == ([APACHE], [])
    assert [e["repo_id"] for e in json.loads(registry.read_text())] == [APACHE]
    assert lines[-1].startswith(f"rskill {APACHE}: installed v")


def test_skips_an_installed_rskill_without_downloading(registry: Path) -> None:
    _run(_Hub(), [APACHE])
    hub = _Hub()
    installed, skipped, lines = _run(hub, [APACHE.lower()])  # the list compares casefolded
    assert (installed, skipped, hub.snapshots) == ([], [], [])
    assert lines == [f"rskill {APACHE.lower()}: already installed"]


def test_skips_a_non_commercial_rskill_with_the_command(registry: Path) -> None:
    hub = _Hub()
    installed, skipped, lines = _run(hub, [NONCOMMERCIAL])
    assert (installed, skipped, hub.snapshots) == ([], [NONCOMMERCIAL], [])
    assert not registry.exists()
    assert "nvidia_non_commercial" in lines[-1]
    assert f"`openral rskill install {NONCOMMERCIAL}`" in lines[-1]
    assert "--yes" not in lines[-1] and "--non-commercial" not in lines[-1]


def test_a_gated_rskill_says_request_access_and_the_rest_install(registry: Path) -> None:
    installed, skipped, lines = _run(_Hub(), [GATED, APACHE])
    assert (installed, skipped) == ([APACHE], [GATED])
    assert any("request access on Hugging Face" in x and GATED in x for x in lines)


def test_an_unreachable_hub_never_raises(registry: Path) -> None:
    installed, skipped, lines = _run(_Hub(offline=True), [APACHE])
    assert (installed, skipped) == ([], [APACHE])
    assert "LocalEntryNotFoundError" in lines[-1]
    assert not registry.exists()


def test_auto_install_reads_the_list_and_obeys_the_off_switch(registry: Path) -> None:
    hub = _Hub()
    os.environ["OPENRAL_RSKILLS"] = f" {APACHE} , "
    os.environ["OPENRAL_RSKILL_AUTO_INSTALL"] = "0"
    with (
        patch("huggingface_hub.hf_hub_download", new=hub.hf_hub_download),
        patch("huggingface_hub.snapshot_download", new=hub.snapshot_download),
    ):
        auto_install_rskills(echo=print)
        assert hub.snapshots == []
        del os.environ["OPENRAL_RSKILL_AUTO_INSTALL"]
        auto_install_rskills(echo=print)
    assert hub.snapshots == [APACHE]
