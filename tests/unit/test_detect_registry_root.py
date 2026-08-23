"""``canonical_robot_path`` must not depend on the caller's directory.

Regression test. The workspace root was previously a hard-coded
``parents[5]``, which overshot the real root by one level, so the package-based
lookup never matched anything and every successful resolution came from the
CWD fallback. Running ``openral detect`` from anywhere but the repo root
therefore emitted an empty ``unknown_<host>`` scaffold — with zero joints and
no HAL — for a robot the probes had already positively identified. Nothing
raised; the operator got a plausible-looking file describing nothing.

These tests really change directory (``monkeypatch.chdir``) rather than
patching the resolver, because the directory the process is in *is* the thing
under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_detect.registry import (
    _PACKAGE_WORKSPACE_ROOT,
    _repo_root_candidates,
    canonical_robot_path,
)


class TestResolvesIndependentlyOfCwd:
    @pytest.mark.parametrize("slug", ["openarm", "so100", "so101_follower"])
    def test_resolves_from_an_unrelated_directory(
        self, slug: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from_repo = canonical_robot_path(slug)
        if from_repo is None:
            pytest.skip(f"no committed manifest for {slug!r}")
        monkeypatch.chdir(tmp_path)
        assert canonical_robot_path(slug) == from_repo

    def test_openarm_resolution_is_a_real_file(self) -> None:
        path = canonical_robot_path("openarm")
        assert path is not None
        assert path.is_file()
        assert path.parent.name == "openarm"


class TestWorkspaceRootDiscovery:
    def test_discovered_root_actually_holds_the_robots_tree(self) -> None:
        # The bug was an ancestor that looked plausible but had no robots/.
        assert _PACKAGE_WORKSPACE_ROOT is not None
        assert (_PACKAGE_WORKSPACE_ROOT / "robots").is_dir()
        assert (_PACKAGE_WORKSPACE_ROOT / "python").is_dir()

    def test_root_is_the_ancestor_containing_this_package(self) -> None:
        from openral_detect import registry

        assert _PACKAGE_WORKSPACE_ROOT is not None
        assert Path(registry.__file__).resolve().is_relative_to(_PACKAGE_WORKSPACE_ROOT)

    def test_cwd_is_read_per_call_not_frozen_at_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The old tuple captured Path.cwd() at import, so a later chdir could
        # not be seen. Reading it per call is what makes the fallback honest.
        monkeypatch.chdir(tmp_path)
        assert tmp_path.resolve() in [c.resolve() for c in _repo_root_candidates()]

    def test_unknown_slug_still_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert canonical_robot_path("definitely_not_a_robot") is None
