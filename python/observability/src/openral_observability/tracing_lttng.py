"""ros2_tracing (LTTng) opt-in for OpenRAL — ADR-0018 F9.

LTTng provides kernel-correlated, microsecond-resolution profiling that
OTel cannot — it captures scheduling decisions, syscalls, and DMA
events alongside our application tracepoints. It is **opt-in** because:

* The userspace tracer link adds dependencies (``lttng-ust``,
  ``babeltrace2``) and a daemon (``lttng-sessiond``) that most users
  don't want loaded by default.
* When the env var :data:`ENV_TRACING_GATE` (``OPENRAL_ROS2_TRACING``)
  is unset, every tracepoint in this module is a no-op — the lookup
  short-circuits before touching any LTTng symbol. Zero cost off.

Three public surfaces:

* :func:`is_enabled` — single source of truth for the gate.
* :func:`lttng_tracepoint` — context manager that fires an entry and an
  exit tracepoint around a code block. Used at the runner tick
  boundaries, the HAL command write / state read, and
  ``safety_node.validate``.
* :func:`start_session` / :func:`stop_session` / :func:`view_session` —
  thin ``lttng`` subprocess wrappers driven by ``openral profile session``.

The trace_id is exposed as an LTTng *context* (``--add-context=ip``
plus a custom string context for the active OTel trace_id) so a
``babeltrace2 --output-format=ctf-metadata`` rendering joins back to
the OTel timeline by the same key F7 uses.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from openral_observability.propagation import current_traceparent

__all__ = [
    "ENV_TRACING_GATE",
    "LttngSession",
    "LttngSessionError",
    "is_enabled",
    "lttng_tracepoint",
    "start_session",
    "stop_session",
    "view_session",
]

_LOG = logging.getLogger(__name__)

# When set to "1" / "true" / "yes" on the agent process, tracepoints
# emit to the LTTng userspace tracer (if available) and a CTF backup
# file. Any other value (or unset) means every tracepoint is a no-op.
ENV_TRACING_GATE: Final[str] = "OPENRAL_ROS2_TRACING"

# Optional output dir for the CTF fallback when ``lttngust`` is not
# installed. The fallback writes line-delimited JSON so a developer can
# pipe it through `jq` without a babeltrace2 install. The real LTTng
# path remains the canonical one and is preferred whenever available.
ENV_TRACING_FALLBACK_DIR: Final[str] = "OPENRAL_ROS2_TRACING_FALLBACK_DIR"


# Tracepoint base names — one constant per "event pair" the hot path
# emits. :func:`lttng_tracepoint` appends ``_begin`` / ``_end`` suffixes
# so each constant covers both sides of the bracket. The ``openral:``
# prefix matches ``tracetools``'s ROS 2 namespace so a future
# ``babeltrace2 --names=openral:*`` filter line is easy.
TP_RUNNER_TICK: Final[str] = "openral:runner_tick"
TP_HAL_READ_STATE: Final[str] = "openral:hal_read_state"
TP_HAL_SEND_ACTION: Final[str] = "openral:hal_send_action"
TP_SENSORS_READ_LATEST: Final[str] = "openral:sensors_read_latest"
TP_WORLD_STATE_SNAPSHOT: Final[str] = "openral:world_state_snapshot"
TP_SKILL_STEP: Final[str] = "openral:skill_step"
TP_ACTION_PUBLISH: Final[str] = "openral:action_publish"
TP_SAFETY_VALIDATE: Final[str] = "openral:safety_validate"

# W3C v0 traceparent has exactly four ``-``-separated parts:
# ``version-trace-span-flags``.
_TRACEPARENT_PART_COUNT: Final[int] = 4


class LttngSessionError(RuntimeError):
    """Raised when the ``lttng`` CLI is missing or a subprocess fails."""


@dataclass(frozen=True)
class LttngSession:
    """An active LTTng session as known to :func:`start_session`.

    Attributes:
        name: LTTng session name (e.g. ``openral``).
        output_dir: Directory the daemon writes the CTF trace into.
    """

    name: str
    output_dir: Path


# ── Gate + tracepoint API ──────────────────────────────────────────────────

_BACKEND_RESOLVED = False
_BACKEND: Any = None  # set once on first call to is_enabled()
_WARNED_ONCE = False


def is_enabled() -> bool:
    """Return True iff :data:`ENV_TRACING_GATE` is set to a truthy value.

    Truthy = ``1`` / ``true`` / ``yes`` (case-insensitive). The function
    resolves the backend lazily on first call and caches the result;
    flipping the env var mid-process therefore does not take effect
    until the process restarts. Cheap path is intentionally tighter
    than the env-var read — the inner ``_BACKEND_RESOLVED`` short-circuit
    is one attribute lookup once the answer is known.
    """
    raw = os.environ.get(ENV_TRACING_GATE, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_backend() -> Any:
    """Pick a tracer backend. Called once; cached in ``_BACKEND``.

    Order: ``lttngust`` (the upstream Python binding) → CTF JSON fallback.
    """
    global _BACKEND, _BACKEND_RESOLVED
    if _BACKEND_RESOLVED:
        return _BACKEND
    _BACKEND_RESOLVED = True
    if not is_enabled():
        _BACKEND = None
        return _BACKEND
    # Broad except: lttngust 2.7.1 (latest on PyPI as of 2026-05) calls
    # the long-removed ``time.clock()`` at import time on Python 3.8+,
    # so the failure mode is an ``AttributeError`` raised from inside
    # ``import lttngust`` itself rather than an ``ImportError``. Other
    # broken versions could fail in other ways; catching ``Exception``
    # at import + probe is the only path that keeps the hot loop alive
    # when the system tracer is misinstalled.
    try:
        import lttngust  # type: ignore[import-not-found] # reason: optional system dep

        backend = _LttngUstBackend(lttngust)
        backend.probe()
    except Exception as exc:  # reason: lttngust raises a moving target
        kind = exc.__class__.__name__
        _warn_fallback(f"lttngust unavailable ({kind}: {exc})")
        _BACKEND = _JsonFallbackBackend.from_env()
        return _BACKEND
    _BACKEND = backend
    return _BACKEND


def _warn_fallback(reason: str) -> None:
    """Emit the one-time fall-back-to-JSONL warning."""
    global _WARNED_ONCE
    if _WARNED_ONCE:
        return
    _LOG.warning(
        "%s=1 but %s; tracepoints will append to the JSON fallback under $%s "
        "(or /tmp/openral-lttng-fallback).",
        ENV_TRACING_GATE,
        reason,
        ENV_TRACING_FALLBACK_DIR,
    )
    _WARNED_ONCE = True


@contextlib.contextmanager
def lttng_tracepoint(name: str, **attrs: Any) -> Iterator[None]:
    """Fire an entry tracepoint, run the block, fire an exit tracepoint.

    When :data:`ENV_TRACING_GATE` is off this is a single env-var check
    plus a generator dance — measured at <300 ns on a 2024 laptop and
    safe to leave in the hot path. The trace_id from the active OTel
    span is attached automatically as ``otel_trace_id`` so a CTF reader
    can join back to the OTel timeline.

    Args:
        name: Tracepoint name; use one of the ``TP_*`` constants.
        **attrs: Arbitrary key/value attributes attached to the
            tracepoint. Keep small (≤16 keys, scalar values) — LTTng
            packs them into the CTF event payload.
    """
    backend = _resolve_backend()
    if backend is None:
        yield
        return
    traceparent = current_traceparent() or ""
    trace_id = ""
    if traceparent:
        # W3C v0: ``00-<trace>-<span>-<flags>``. Split robustly.
        parts = traceparent.split("-")
        if len(parts) == _TRACEPARENT_PART_COUNT:
            trace_id = parts[1]
    enriched = {"otel_trace_id": trace_id, **attrs}
    backend.emit(f"{name}_begin", enriched)
    try:
        yield
    finally:
        backend.emit(f"{name}_end", enriched)


class _LttngUstBackend:
    """Tracepoint backend that defers to ``lttngust.TraceLogger``.

    lttngust ships ``TraceLogger`` — a Python ``logging.Logger`` that
    proxies records into the userspace tracer. We use the
    ``openral:<tp_name>`` logger names so a session's
    ``--userspace --tracer openral:*`` filter captures everything.
    """

    def __init__(self, lttngust_mod: Any) -> None:
        self._mod = lttngust_mod
        self._loggers: dict[str, Any] = {}

    def probe(self) -> None:
        """Smoke-test that lttngust accepts a record before we commit.

        Trigger lttngust's lazy thread boot once so a broken install
        surfaces here (and falls back to JSONL) rather than from the
        first hot-path tracepoint. Sends one record on an
        ``openral:_probe`` logger, which is also a useful smoke event
        in the resulting CTF trace.
        """
        probe_logger = logging.getLogger("openral:_probe")
        probe_logger.info("openral_lttng_probe")

    def emit(self, name: str, attrs: dict[str, Any]) -> None:
        logger = self._loggers.get(name)
        if logger is None:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        # lttngust attaches the logger record's `args` as CTF fields when
        # they are passed as a dict. Stringify values so the tracer
        # doesn't choke on non-CTF-encodable types.
        safe = {k: _ctf_safe(v) for k, v in attrs.items()}
        logger.info("%s", name, extra={"openral_tp": safe})


class _JsonFallbackBackend:
    """Append tracepoints as line-delimited JSON to a fallback file.

    Used only when :data:`ENV_TRACING_GATE` is set but the ``lttngust``
    Python binding is not importable. The JSON shape mirrors the LTTng
    event header (``name``, ``ts_ns``, ``otel_trace_id``, ``attrs``) so
    downstream tooling can convert it to CTF later if needed.
    """

    def __init__(self, fallback_path: Path) -> None:
        self._path = fallback_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Open append-mode; one process == one file == no inter-process
        # lock needed for line-buffered writes on POSIX.
        # Long-lived backend handle; close on interpreter shutdown is best-effort.
        self._fp = self._path.open("a", encoding="utf-8")

    @classmethod
    def from_env(cls) -> _JsonFallbackBackend:
        # Opt-in profiling fallback only; users override via env when the
        # default /tmp directory is not appropriate.
        base = os.environ.get(ENV_TRACING_FALLBACK_DIR, "/tmp/openral-lttng-fallback")
        path = Path(base) / f"openral-{os.getpid()}.jsonl"
        return cls(path)

    def emit(self, name: str, attrs: dict[str, Any]) -> None:
        import json as _json
        import time

        record = {
            "name": name,
            "ts_ns": time.monotonic_ns(),
            "otel_trace_id": attrs.get("otel_trace_id", ""),
            "attrs": {k: _ctf_safe(v) for k, v in attrs.items() if k != "otel_trace_id"},
        }
        self._fp.write(_json.dumps(record, sort_keys=False) + "\n")
        # Tracepoints are rare relative to the ~30 Hz tick — flush
        # immediately so a `kill -9` does not lose the last few events.
        self._fp.flush()


def _ctf_safe(value: Any) -> Any:
    """Coerce ``value`` into something an LTTng/CTF encoder accepts."""
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_ctf_safe(v) for v in value]
    return str(value)


# ── lttng-cli wrappers driven by `openral profile session` ─────────────────────


def _require_lttng() -> str:
    exe = shutil.which("lttng")
    if exe is None:
        msg = (
            "lttng-tools not found on PATH; install `lttng-tools` (apt: lttng-tools, "
            "brew: lttng-tools) or run with OPENRAL_ROS2_TRACING=1 only on a host "
            "with the daemon available."
        )
        raise LttngSessionError(msg)
    return exe


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    # argv is composed from validated args; no shell involvement.
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        joined = " ".join(cmd)
        msg = f"{joined} failed (exit {completed.returncode}): {completed.stderr.strip()}"
        raise LttngSessionError(msg)
    return completed


def start_session(*, name: str, output_dir: Path) -> LttngSession:
    """Create + start an LTTng session that captures ``openral:*`` userspace events.

    Idempotent w.r.t. an existing session: if ``name`` already exists,
    it is destroyed first. The output directory is created on demand.
    """
    exe = _require_lttng()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Tear down any stale session of the same name. Ignore "session not
    # found" exit codes — destroying a missing session is fine.
    subprocess.run(  # reason: argv composed from validated args
        [exe, "destroy", name],
        check=False,
        capture_output=True,
        text=True,
    )
    _run([exe, "create", name, "--output", str(output_dir)])
    _run([exe, "enable-event", "--session", name, "--userspace", "openral:*"])
    # Capture the active span context so a CTF dump joins to OTel.
    _run([exe, "add-context", "--session", name, "--userspace", "--type=vpid"])
    _run([exe, "start", name])
    return LttngSession(name=name, output_dir=output_dir)


def stop_session(*, name: str) -> None:
    """Stop and destroy an LTTng session previously created via :func:`start_session`.

    Destroying flushes pending events to disk and is therefore part of
    the stop path — without it, an attached viewer would only see the
    most recent ring contents.
    """
    exe = _require_lttng()
    _run([exe, "stop", name])
    _run([exe, "destroy", name])


def view_session(*, output_dir: Path) -> None:
    """Print a brief CTF summary of the trace at ``output_dir``.

    Uses ``babeltrace2`` when available (the canonical viewer); falls
    back to listing the produced files when not. This is intentionally
    minimal — anything richer belongs in babeltrace2 itself or Trace
    Compass.
    """
    bt = shutil.which("babeltrace2")
    if bt is not None:
        completed = subprocess.run(  # reason: argv composed from validated arg
            [bt, str(output_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        print(completed.stdout, end="")
        if completed.returncode != 0:
            msg = f"babeltrace2 failed (exit {completed.returncode}): {completed.stderr.strip()}"
            raise LttngSessionError(msg)
        return
    # No babeltrace2; list the files so the user can hand the directory
    # to Trace Compass or `babeltrace2` later.
    if not output_dir.exists():
        msg = f"trace output dir does not exist: {output_dir}"
        raise LttngSessionError(msg)
    for entry in sorted(output_dir.rglob("*")):
        if entry.is_file():
            print(entry)
