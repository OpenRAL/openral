"""Unit tests for the libero<->robocasa robosuite-conflict diagnostic.

robocasa's ``__init__`` imports ``PandaOmron`` / ``PandaMobile`` from robosuite,
which exist only in the robocasa-pinned robosuite fork. The LIBERO dependency
group installs an older robosuite that shadows it, so running a robocasa scene
after a libero sync fails with a cryptic ``cannot import name 'PandaOmron'``.
:func:`_robosuite_conflict_hint` turns that into an actionable resync error while
leaving a genuine ``No module named 'robocasa'`` (real absence) to the normal
``ensure_backend_deps`` install path.
"""

from __future__ import annotations

from openral_core.exceptions import ROSConfigError
from openral_sim.backends.robocasa import _robosuite_conflict_hint


def test_panda_omron_importerror_maps_to_resync_hint() -> None:
    exc = ImportError("cannot import name 'PandaOmron' from 'robosuite.models.robots'")
    hint = _robosuite_conflict_hint(exc)
    assert isinstance(hint, ROSConfigError)
    msg = str(hint)
    assert "version conflict" in msg
    assert "--group robocasa" in msg
    assert "PandaOmron" in msg


def test_panda_mobile_importerror_also_maps() -> None:
    exc = ImportError("cannot import name 'PandaMobile' from 'robosuite.models.robots'")
    assert _robosuite_conflict_hint(exc) is not None


def test_genuine_missing_robocasa_returns_none() -> None:
    # Real absence -> None so the caller re-raises and ensure_backend_deps installs.
    assert _robosuite_conflict_hint(ImportError("No module named 'robocasa'")) is None


def test_unrelated_importerror_returns_none() -> None:
    assert _robosuite_conflict_hint(ImportError("cannot import name 'foo' from 'bar'")) is None
