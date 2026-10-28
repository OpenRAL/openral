"""LTTng tracing opt-in gate and JSON-fallback backend.

The userspace ``lttngust`` Python binding is a system dep, not installable
via uv on a generic CI runner, so these tests cover the gate semantics +
the JSON fallback (which IS exercisable everywhere). The actual LTTng
session subprocess wrappers are validated separately under
``tests/integration/`` (skipped when ``lttng`` is missing on PATH).
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import openral_observability.tracing_lttng as ttng
import pytest


@pytest.fixture(autouse=True)
def _reset_backend_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Backend resolution is cached at module scope; reset between tests."""
    monkeypatch.setattr(ttng, "_BACKEND_RESOLVED", False, raising=False)
    monkeypatch.setattr(ttng, "_BACKEND", None, raising=False)
    monkeypatch.setattr(ttng, "_WARNED_ONCE", False, raising=False)
    yield


def test_is_enabled_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ttng.ENV_TRACING_GATE, raising=False)
    assert ttng.is_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "On"])
def test_is_enabled_truthy_values(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ttng.ENV_TRACING_GATE, value)
    assert ttng.is_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_is_enabled_falsy_values(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ttng.ENV_TRACING_GATE, value)
    assert ttng.is_enabled() is False


def test_tracepoint_is_noop_when_gate_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the gate off the bracket runs without resolving any backend."""
    monkeypatch.delenv(ttng.ENV_TRACING_GATE, raising=False)
    counter = {"n": 0}
    with ttng.lttng_tracepoint(ttng.TP_RUNNER_TICK, tick_idx=1):
        counter["n"] += 1
    assert counter["n"] == 1
    # Backend remains unresolved → no allocation cost incurred.
    assert ttng._BACKEND is None


def test_tracepoint_falls_back_to_jsonl_when_lttngust_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the gate on and lttngust absent, events land in the JSONL fallback."""
    monkeypatch.setenv(ttng.ENV_TRACING_GATE, "1")
    monkeypatch.setenv(ttng.ENV_TRACING_FALLBACK_DIR, str(tmp_path))
    # Force the import path to fail so the backend selector picks the
    # JSON fallback even on hosts that happen to have lttngust.
    real_import = importlib.import_module

    def _fake_import(name: str, package: str | None = None) -> ModuleType:
        if name == "lttngust":
            raise ImportError("simulated missing lttngust for this test")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    # tracing_lttng uses the builtin ``import`` statement; patch builtins
    # to make the simulated failure visible to that path too.
    import builtins

    real_builtin_import = builtins.__import__

    def _fake_builtin_import(
        name: str,
        *args: Any,  # reason: forwards __import__'s mixed-type args verbatim
        **kwargs: Any,  # reason: forwards __import__'s mixed-type args verbatim
    ) -> ModuleType:
        if name == "lttngust":
            raise ImportError("simulated missing lttngust for this test")
        return real_builtin_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_builtin_import)

    with ttng.lttng_tracepoint(ttng.TP_HAL_READ_STATE, tick_idx=42, adapter="so100"):
        pass

    files = list(tmp_path.glob(f"openral-{os.getpid()}.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    # One begin + one end record.
    assert len(lines) == 2
    decoded = [json.loads(line) for line in lines]
    assert decoded[0]["name"] == ttng.TP_HAL_READ_STATE + "_begin"
    assert decoded[1]["name"] == ttng.TP_HAL_READ_STATE + "_end"
    # Attributes survive round-trip.
    assert decoded[0]["attrs"]["adapter"] == "so100"
    assert decoded[0]["attrs"]["tick_idx"] == 42


def test_json_fallback_default_dir_is_private(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default fallback dir is a fresh owner-only dir, never a shared /tmp path."""
    monkeypatch.delenv(ttng.ENV_TRACING_FALLBACK_DIR, raising=False)
    backend = ttng._JsonFallbackBackend.from_env()
    try:
        parent = backend._path.parent
        assert parent != Path("/tmp/openral-lttng-fallback")
        st = parent.stat()
        assert st.st_uid == os.getuid()
        assert st.st_mode & 0o077 == 0
    finally:
        backend._fp.close()
        shutil.rmtree(backend._path.parent)


def test_json_fallback_refuses_planted_symlink(tmp_path: Path) -> None:
    """A pre-planted symlink at the trace path is never followed."""
    victim = tmp_path / "victim.txt"
    victim.write_text("keep\n", encoding="utf-8")
    trace_path = tmp_path / f"openral-{os.getpid()}.jsonl"
    trace_path.symlink_to(victim)
    with pytest.raises(OSError):
        ttng._JsonFallbackBackend(trace_path)
    assert victim.read_text(encoding="utf-8") == "keep\n"


def test_lttng_session_helpers_error_cleanly_without_lttng_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """start_session / stop_session error message names the missing tool."""

    def _no_lttng(name: str) -> str | None:
        if name == "lttng":
            return None
        return f"/usr/bin/{name}"

    monkeypatch.setattr(shutil, "which", _no_lttng)
    with pytest.raises(ttng.LttngSessionError, match="lttng-tools not found on PATH"):
        ttng.start_session(name="openral", output_dir=tmp_path)
    with pytest.raises(ttng.LttngSessionError, match="lttng-tools not found on PATH"):
        ttng.stop_session(name="openral")


def test_json_fallback_refuses_preexisting_non_regular_file(tmp_path: Path) -> None:
    """A pre-created non-regular file (here a FIFO) in the fallback dir is refused."""
    trace_path = tmp_path / f"openral-{os.getpid()}.jsonl"
    os.mkfifo(trace_path)
    reader = os.open(trace_path, os.O_RDONLY | os.O_NONBLOCK)  # lets the O_WRONLY open return
    try:
        with pytest.raises(PermissionError, match="not a regular file owned by this user"):
            ttng._JsonFallbackBackend(trace_path)
    finally:
        os.close(reader)


def test_json_fallback_appends_to_own_preexisting_file(tmp_path: Path) -> None:
    """A regular file this user already owns is appended to, not refused."""
    trace_path = tmp_path / f"openral-{os.getpid()}.jsonl"
    trace_path.write_text('{"name": "earlier"}\n', encoding="utf-8")
    backend = ttng._JsonFallbackBackend(trace_path)
    try:
        backend.emit("openral:test", {"otel_trace_id": "abc"})
    finally:
        backend._fp.close()
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"name": "earlier"}
    assert json.loads(lines[1])["name"] == "openral:test"
