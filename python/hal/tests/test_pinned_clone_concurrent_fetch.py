"""Concurrent callers of the pinned-clone asset fetchers must not corrupt the cache.

CI runs pytest partitions as parallel background processes against the shared
``$OPENRAL_CACHE_DIR`` (test-selective's "Run selected targets" step), and both
the ``python/core`` and ``python/hal`` partitions resolve ``openarm:`` asset
refs. The pre-fix fetchers cloned straight into the final cache path, so one
worker's ``rmtree`` + clone destroyed another's in-flight clone — observed on
a real runner as ``fatal: could not open .../tmp_pack_... for reading`` and
``fatal: Unable to read current working directory`` (openral PR #51's
select-and-test run). The fix stages each fetch in a unique directory and
atomically renames it into place, so the final path is never partial.

These tests drive the real fetchers against real local git repos (a
network-boundary substitution for the GitHub remotes, per CLAUDE.md §1.11) from
N barrier-synchronised threads. On the pre-fix code they fail: the concurrent
clones collide on the shared destination and raise ``ROSConfigError``.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from openral_hal import _anvil_openarm_v2_assets as anvil_mod
from openral_hal import _openarm_v2_assets as openarm_mod

_N_WORKERS = 6

_GIT_ID = ["-c", "user.email=ci@openral.test", "-c", "user.name=openral-ci"]


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *_GIT_ID, *args], cwd=cwd, check=True, capture_output=True)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _git("init", "-q", cwd=path)


def _commit_all(path: Path, message: str) -> str:
    _git("add", "-A", cwd=path)
    _git("commit", "-q", "-m", message, cwd=path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    )
    return head.stdout.strip()


@pytest.fixture
def file_protocol_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the fetchers' plain ``git submodule update`` clone file:// submodules.

    Git ≥2.38 refuses the file protocol for submodules by default (CVE-2022-39253);
    the config comes in via the environment so the code under test needs no flag.
    """
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.file.allow")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")


def _race(fetch: Callable[[], str]) -> set[str]:
    """Call ``fetch`` from N threads released simultaneously; return the result set."""
    barrier = threading.Barrier(_N_WORKERS)

    def worker() -> str:
        barrier.wait()
        return fetch()

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
        return {f.result() for f in [pool.submit(worker) for _ in range(_N_WORKERS)]}


def test_anvil_concurrent_fetch_is_race_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_protocol_allowed: None
) -> None:
    """N simultaneous first-time fetchers all get the same complete tree (incl. submodule)."""
    mesh_repo = tmp_path / "openarm_mujoco"
    _init_repo(mesh_repo)
    assets = mesh_repo / "v2" / "assets"
    assets.mkdir(parents=True)
    (assets / "link0.obj").write_text("v 0 0 0\n", encoding="utf-8")
    _commit_all(mesh_repo, "meshes")

    anvil_repo = tmp_path / "anvil-openarm-mujoco"
    _init_repo(anvil_repo)
    models = anvil_repo / "models"
    models.mkdir()
    (models / "anvil_openarm_bimanual.xml").write_text("<mujoco/>\n", encoding="utf-8")
    _git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        f"file://{mesh_repo}",
        "upstream/openarm_mujoco",
        cwd=anvil_repo,
    )
    sha = _commit_all(anvil_repo, "anvil variant")

    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(anvil_mod, "_ANVIL_REPO_URL", f"file://{anvil_repo}")
    monkeypatch.setattr(anvil_mod, "_ANVIL_PINNED_SHA", sha)

    results = _race(anvil_mod.ensure_anvil_openarm_v2_mjcf)

    assert len(results) == 1, f"divergent paths from concurrent fetches: {results}"
    mjcf = Path(results.pop())
    assert mjcf.is_file()
    meshes = mjcf.parent.parent / "upstream" / "openarm_mujoco" / "v2" / "assets"
    assert (meshes / "link0.obj").is_file(), "submodule tree incomplete after the race"
    # No staging leftovers, and the cached fast path returns the same tree.
    assert not list(mjcf.parent.parent.parent.glob(".staging-*"))
    assert anvil_mod.ensure_anvil_openarm_v2_mjcf() == str(mjcf)


def test_openarm_v2_concurrent_fetch_is_race_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same guarantee for the enactic v2 fetcher (no submodule step)."""
    repo = tmp_path / "openarm_mujoco"
    _init_repo(repo)
    mjcf_rel = Path("v2/openarm_mujoco_v2/openarm_v20_bimanual.xml")
    (repo / mjcf_rel).parent.mkdir(parents=True)
    (repo / mjcf_rel).write_text("<mujoco/>\n", encoding="utf-8")
    sha = _commit_all(repo, "v2")

    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(openarm_mod, "_OPENARM_REPO_URL", f"file://{repo}")
    monkeypatch.setattr(openarm_mod, "_OPENARM_V2_PINNED_SHA", sha)

    results = _race(openarm_mod.ensure_openarm_v2_mjcf)

    assert len(results) == 1, f"divergent paths from concurrent fetches: {results}"
    mjcf = Path(results.pop())
    assert mjcf.is_file()
    assert openarm_mod.ensure_openarm_v2_mjcf() == str(mjcf)
