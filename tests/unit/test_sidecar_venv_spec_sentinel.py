"""Unit tests for ``ensure_pip_venv``'s spec-keyed provisioning sentinel.

Regression coverage for issue #89. The sentinel used to be an opaque ``ok``
marker, so a venv was "provisioned" forever regardless of *what* it had been
provisioned from: correcting a pin in ``_ISAAC_CUDA_DEPS`` could not repair an
existing venv, and the Isaac sidecar kept an ``nvidia-nvjitlink-cu12`` too old
for its own prebundled cusparse — Kit then hung past ``app ready`` until the
1200 s boot timeout expired. Keying the sentinel on a digest of the dependency
spec makes a corrected pin re-run the install.

Real venvs on the real ``uv`` (CLAUDE.md §1.11): the only thing supplied by the
test is the ``install`` callback, which is the caller's own extension point, not
a stand-in for a dependency. No package is installed — provisioning order is
what is under test, not resolution.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from openral_sim._sidecar_common import ensure_pip_venv, spec_marker

pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None, reason="ensure_pip_venv provisions through uv"
)


class _Recorder:
    """An ``install`` callback that records the specs it was asked to apply."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.spec = "first-spec==1.0"

    def __call__(self, uv: str, py: Path) -> None:
        self.calls.append(self.spec)


def _provision(home: Path, install: _Recorder, spec: tuple[str, ...] | None) -> Path:
    return ensure_pip_venv(
        label="spec-sentinel-test", home=home, python="3.12", install=install, spec=spec
    )


def test_unchanged_spec_reuses_the_venv(tmp_path: Path) -> None:
    install = _Recorder()
    first = _provision(tmp_path, install, ("torch==2.7.0", "nvidia-nvjitlink-cu12>=12.6,<13"))
    second = _provision(tmp_path, install, ("torch==2.7.0", "nvidia-nvjitlink-cu12>=12.6,<13"))
    assert first == second
    assert len(install.calls) == 1, "an unchanged spec must not reinstall"


def test_corrected_pin_repairs_an_existing_venv(tmp_path: Path) -> None:
    """The issue-#89 case: raising a floor must re-run the install."""
    install = _Recorder()
    _provision(tmp_path, install, ("torch==2.7.0", "nvidia-nvjitlink-cu12>=12.6,<13"))
    install.spec = "corrected"
    _provision(tmp_path, install, ("torch==2.7.0", "nvidia-nvjitlink-cu12>=12.8,<13"))
    assert install.calls == ["first-spec==1.0", "corrected"]
    # And the repair is not re-run on the next boot.
    _provision(tmp_path, install, ("torch==2.7.0", "nvidia-nvjitlink-cu12>=12.8,<13"))
    assert len(install.calls) == 2


def test_legacy_ok_sentinel_is_repaired_once(tmp_path: Path) -> None:
    """A venv provisioned before this change carries ``ok`` — repair, then settle."""
    install = _Recorder()
    spec = ("nvidia-nvjitlink-cu12>=12.8,<13",)
    _provision(tmp_path, install, spec)
    (tmp_path / ".venv" / ".deps-installed").write_text("ok\n")
    _provision(tmp_path, install, spec)
    assert len(install.calls) == 2
    _provision(tmp_path, install, spec)
    assert len(install.calls) == 2


def test_specless_callers_keep_the_opaque_marker(tmp_path: Path) -> None:
    """``spec=None`` (no stable spec to key on) preserves the old behavior."""
    install = _Recorder()
    _provision(tmp_path, install, None)
    assert (tmp_path / ".venv" / ".deps-installed").read_text() == "ok\n"
    _provision(tmp_path, install, None)
    assert len(install.calls) == 1


def test_interrupted_install_is_retried(tmp_path: Path) -> None:
    """No sentinel == the install never completed; it must run again."""
    install = _Recorder()
    _provision(tmp_path, install, ("depth-anything-3",))
    (tmp_path / ".venv" / ".deps-installed").unlink()
    _provision(tmp_path, install, ("depth-anything-3",))
    assert len(install.calls) == 2


def test_spec_marker_separates_adjacent_entries() -> None:
    """Concatenation must not let two different specs collide into one digest."""
    assert spec_marker(("ab", "c")) != spec_marker(("a", "bc"))
    assert spec_marker(None) == "ok\n"
