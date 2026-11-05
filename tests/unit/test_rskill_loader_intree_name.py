"""``load_rskill_manifest`` resolves an ``<org>/rskill-<name>`` Hub id to its in-tree manifest.

In-tree directory names do not follow the repo id (``OpenRAL/rskill-smolvla-franka_panda-
libero_spatial-bf16`` lives in ``rskills/smolvla-libero``), so the loader matches the
manifest's ``name``. A malformed manifest elsewhere in the tree must neither break that
lookup nor print into the caller's output, and ``$OPENRAL_REPO_ROOT`` picks the tree.

Fixtures: real ``rskills/smolvla-libero`` + ``robots/franka_panda`` copied into a temporary
checkout. No network: every id asked for resolves in-tree or not at all.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from openral_rskill import loader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOLVLA_ID = "OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16"


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal OpenRAL checkout named by ``$OPENRAL_REPO_ROOT``, with one broken manifest."""
    shutil.copytree(_REPO_ROOT / "robots" / "franka_panda", tmp_path / "robots" / "franka_panda")
    shutil.copytree(
        _REPO_ROOT / "rskills" / "smolvla-libero", tmp_path / "rskills" / "smolvla-libero"
    )
    broken = tmp_path / "rskills" / "aaa-half-edited" / "rskill.yaml"
    broken.parent.mkdir(parents=True)
    broken.write_text('name: "OpenRAL/rskill-broken\n  kind: [vla\n', encoding="utf-8")
    monkeypatch.setenv(loader.REPO_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(loader, "_RSKILL_MANIFEST_CACHE", {})
    return tmp_path


def test_hub_id_resolves_in_tree_past_a_malformed_manifest(
    checkout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The broken manifest sorts first; the lookup still finds smolvla, silently."""
    manifest = loader.load_rskill_manifest(_SMOLVLA_ID)
    assert manifest.name == _SMOLVLA_ID
    assert manifest.weights_uri == "hf://lerobot/smolvla_libero"  # the in-tree config
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_name_index_skips_unknown_and_malformed(checkout: Path) -> None:
    """Unknown ids miss; the malformed file is never a match, never an error."""
    assert loader._intree_manifest_path_by_name("OpenRAL/rskill-not-in-tree") is None
    assert loader._intree_manifest_path_by_name("OpenRAL/rskill-broken") is None
    hit = loader._intree_manifest_path_by_name(_SMOLVLA_ID)
    assert hit == checkout.resolve() / "rskills" / "smolvla-libero" / "rskill.yaml"


def test_repo_root_override_requires_both_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An override without ``robots/`` + ``rskills/`` is ignored: the package checkout wins."""
    monkeypatch.setenv(loader.REPO_ROOT_ENV, str(tmp_path))
    assert loader.repo_root_override() is None
    assert loader.intree_repo_root() == _REPO_ROOT
    monkeypatch.delenv(loader.REPO_ROOT_ENV)
    assert loader.intree_repo_root() == _REPO_ROOT
