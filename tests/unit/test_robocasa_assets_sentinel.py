"""The RoboCasa asset sentinel vouches only for the install it was written for.

Regression: the sentinel lived in a host-global cache dir, so after one
install (venv / worktree / ``/tmp/rg1`` clone) fetched the GR1 assets, a
fresh clone with NO assets short-circuited and the env build died with
``FileNotFoundError: Object registry objaverse directory not found``.

The ``robocasa`` packages here are real on-disk package trees in the GR1
fork's layout; the download script is the process boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from openral_core.exceptions import ROSConfigError
from openral_sim._assets import _READY_SENTINEL, ensure_robocasa_assets


def _gr1_clone(root: Path, *, with_assets: bool) -> Path:
    pkg = root / "robocasa"
    (pkg / "scripts").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    # Downloader exits non-zero: reaching it proves no short-circuit happened.
    (pkg / "scripts" / "download_tabletop_assets.py").write_text("raise SystemExit(1)\n")
    if with_assets:
        for registry in ("objaverse", "sketchfab", "lightwheel"):
            (pkg / "models" / "assets" / "objects" / registry).mkdir(parents=True)
    return root


def test_sentinel_does_not_mask_a_fresh_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENRAL_ALLOW_ROBOCASA_ASSETS", "1")
    monkeypatch.delitem(sys.modules, "robocasa", raising=False)

    warm = _gr1_clone(tmp_path / "warm", with_assets=True)
    monkeypatch.syspath_prepend(str(warm))
    target = ensure_robocasa_assets()
    assert (target / _READY_SENTINEL).is_file()

    fresh = _gr1_clone(tmp_path / "fresh", with_assets=False)
    monkeypatch.syspath_prepend(str(fresh))
    with pytest.raises(ROSConfigError, match="GR1 asset download failed"):
        ensure_robocasa_assets()
