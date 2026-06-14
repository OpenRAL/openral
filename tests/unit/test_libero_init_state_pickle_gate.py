"""Tests for the LIBERO custom-BDDL init-state pickle gate (security audit C2).

``scene.backend_options.init_state_file`` is loaded via ``torch.load`` (pickle),
which executes arbitrary code from the file. The path comes from a possibly
shared/downloaded scene config, so the load is gated behind an explicit
acknowledgement, mirroring the other pickle sinks.
"""

from __future__ import annotations

import pathlib

import pytest
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.libero_custom_bddl import _load_init_states

torch = pytest.importorskip("torch")


def _write_init_state(tmp_path: pathlib.Path) -> pathlib.Path:
    """A real, trusted init-state checkpoint (torch.save of a small state array)."""
    p = tmp_path / "task.pruned_init"
    torch.save({"states": torch.zeros(2, 8)}, str(p))
    return p


def test_refused_without_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENRAL_ALLOW_UNSAFE_PICKLE", raising=False)
    path = _write_init_state(tmp_path)
    with pytest.raises(ROSConfigError, match="remote-code-execution"):
        _load_init_states(path)


def test_loads_with_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_ALLOW_UNSAFE_PICKLE", "1")
    path = _write_init_state(tmp_path)
    states = _load_init_states(path)
    assert "states" in states


def test_missing_file_raises(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_ALLOW_UNSAFE_PICKLE", "1")
    with pytest.raises(ROSConfigError, match="does not exist"):
        _load_init_states(tmp_path / "absent.pruned_init")
