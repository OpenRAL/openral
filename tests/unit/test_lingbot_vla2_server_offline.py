"""Offline Qwen resolution in ``tools/_lingbot_vla2_server.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from huggingface_hub import constants

_PATH = Path(__file__).parents[2] / "tools" / "_lingbot_vla2_server.py"
_SPEC = importlib.util.spec_from_file_location("_lingbot_vla2_server_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_REPO = "Qwen/Qwen3-VL-4B-Instruct"
_SHA = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def test_offline_default_qwen_resolves_to_cached_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real huggingface_hub cache layout: refs/main -> snapshots/<sha>/.
    repo_dir = tmp_path / "models--Qwen--Qwen3-VL-4B-Instruct"
    snapshot = repo_dir / "snapshots" / _SHA
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(_SHA)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    for env in ("OPENRAL_QWEN3VL_PATH", "QWEN3VL_PATH"):
        monkeypatch.delenv(env, raising=False)

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    assert Path(_MODULE._resolve_qwen()) == snapshot

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    assert _MODULE._resolve_qwen() == _REPO
