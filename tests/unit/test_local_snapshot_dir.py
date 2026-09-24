"""``local_snapshot_dir``: an installed rSkill snapshot is its own checkpoint directory.

``resolve_rskill_to_hf_with_revision`` returns the snapshot directory for a
skill installed with ``openral rskill install`` (``rskill.yaml`` next to
``model.safetensors``), and ``huggingface_hub.snapshot_download`` rejects a
filesystem path as a repo id — which is how the ACT LIBERO checkpoint failed
to build from such a snapshot (2026-09-23). Real ACT in-tree manifest + the
real cached Hub checkpoint; the Hub case skips when it is not cached.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from openral_rskill import local_snapshot_dir
from openral_rskill.loader import resolve_rskill_to_hf_with_revision

_REPO = Path(__file__).resolve().parents[2]
_ACT_MANIFEST = _REPO / "rskills" / "act-libero" / "rskill.yaml"


def test_a_directory_is_returned_as_itself(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"")
    # A pinned revision has nothing to add to a directory that IS the snapshot.
    assert local_snapshot_dir(str(snapshot), revision="deadbeef") == str(snapshot.resolve())


def test_an_installed_snapshot_round_trips_through_the_resolver(tmp_path: Path) -> None:
    """The exact shape ``openral rskill install`` leaves behind resolves to itself."""
    snapshot = tmp_path / "models--OpenRAL--rskill-act-libero" / "snapshots" / ("0" * 40)
    snapshot.mkdir(parents=True)
    shutil.copy(_ACT_MANIFEST, snapshot / "rskill.yaml")
    (snapshot / "model.safetensors").write_bytes(b"")
    repo_id, revision = resolve_rskill_to_hf_with_revision(str(snapshot))
    assert repo_id == str(snapshot.resolve())
    assert revision is None
    assert local_snapshot_dir(repo_id, revision=revision) == repo_id


def test_a_hub_id_snapshots_when_cached() -> None:
    pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        local = local_snapshot_dir("Deepkar/libero-test-act", local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip("Deepkar/libero-test-act is not in the local HF cache")
    assert (Path(local) / "model.safetensors").is_file()
