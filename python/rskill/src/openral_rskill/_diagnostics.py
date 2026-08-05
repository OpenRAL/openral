"""Shared load-phase instrumentation for the rSkill / sim policy load path.

Internal module. Not part of the public ``openral_rskill`` surface.

`phase_timer(name, prefix=..., gpu_mb=...)` is the canonical seam every
VLA adapter's `_build_*` factory wraps each load phase with so the
operator can see exactly where a multi-second load is spending its time
— without it, phases like ``PI05Policy.from_pretrained`` (3.4 B-param
graph allocation) and ``materialize_processor_dir`` (HF Hub HEAD
requests for cached files) sit in opaque C/CUDA / network code for tens
of seconds with no log output at all.

The original implementation lived inline as ``_heartbeat`` in the pi05
adapter; it is generalised here so the smolvla / xvla / act adapters
can apply the same pattern without duplicating the threading + GPU
plumbing (CLAUDE.md §1.13).

Output shape per phase::

    <prefix>_<name>_start {**fields}
    <prefix>_<name>_heartbeat {elapsed_s, [gpu_mb], rss_mb, major_faults, **fields}
    <prefix>_<name>_heartbeat {...}                          # every interval_s
    ...
    <prefix>_<name>_done {elapsed_s, rss_mb, major_faults, **fields}

``rss_mb`` / ``major_faults`` (the latter counted from phase entry) are
Linux-only and simply absent elsewhere. They exist to attribute a load
phase that is slow while burning no CPU — the signature of page reclaim,
which no CPU-time metric can show.

Example:
-------
>>> from openral_rskill._diagnostics import phase_timer
>>> with phase_timer("load_weights", prefix="smolvla", repo="lerobot/smolvla_libero"):
...     pass  # heavy work goes here
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

__all__ = ["phase_timer"]

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _gpu_mb(*, no_import: bool = False) -> float | None:
    """Return current CUDA allocator usage in MB, or ``None`` if unavailable.

    Cheap: a single ``torch.cuda.memory_allocated()`` call — deliberately
    ``memory_allocated`` (live tensors), not ``memory_reserved`` (the caching
    allocator's pool): the question both callers ask is "did the weights
    actually go away", and reserved bytes stay put by design after a free.
    THE single GPU-memory probe — eviction accounting in
    ``rskill_runner_node`` and the phase-timer heartbeats read this same
    number, so their logs can never disagree about how much VRAM a swap
    freed.

    Args:
        no_import: When True, only consult an already-imported torch
            (``sys.modules``) — for callers on paths where importing
            torch just to answer would itself be the heavy operation.
            Default imports torch lazily, so a CPU-only host still
            answers ``None``.
    """
    if no_import:
        torch = sys.modules.get("torch")
        if torch is None:
            return None
    else:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return None
    try:
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_allocated()) / 1024 / 1024
    except Exception:
        return None


def _rss_majflt() -> tuple[float, int] | None:
    """Return ``(rss_mb, major_faults)`` for this process, or ``None``.

    Two small ``/proc`` reads, no psutil. Present on every heartbeat because
    a load phase that is neither CPU-bound nor I/O-bound is almost always
    stalled in page reclaim — a state that shows up nowhere in CPU
    accounting. A 2 GB model build on a host under memory pressure inflates
    wall-time with a rising ``major_faults`` and a flat ``elapsed_s``-per-MB,
    which is the signature this exists to capture. Non-Linux hosts get
    ``None`` and the fields are simply absent.
    """
    try:
        with open("/proc/self/statm") as f:  # reason: procfs, not a path op
            rss_pages = int(f.read().split()[1])
        with open("/proc/self/stat") as f:  # reason: procfs, not a path op
            majflt = int(f.read().rsplit(") ", 1)[1].split()[9])
    except (OSError, IndexError, ValueError):
        return None
    return rss_pages * _PAGE_SIZE / 1024 / 1024, majflt


class _SwitchIntervalGuard:
    """Depth-counted owner of the process-global GIL switch interval.

    The switch interval is PROCESS-GLOBAL, so its save/restore must have
    exactly one owner. Two overlapping ``phase_timer`` contexts on different
    threads would otherwise interleave their restores — A enters saving 5 ms,
    B enters saving A's 50 ms, A exits restoring 5 ms, B exits re-installing
    50 ms — leaving the deploy graph's camera pumps and HAL publisher at a
    50 ms switch interval for the rest of the session. The outermost
    acquire sets 0.05 and the matching outermost release restores the saved
    value; nested/concurrent phases ride along. (Deliberate module-level
    singleton: it guards a resource that is itself a process global.)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._prev = 0.005  # overwritten by the outermost holder on acquire

    def acquire(self) -> None:
        with self._lock:
            self._depth += 1
            if self._depth == 1:
                self._prev = sys.getswitchinterval()
                sys.setswitchinterval(0.05)

    def release(self) -> None:
        with self._lock:
            self._depth -= 1
            if self._depth == 0:
                sys.setswitchinterval(self._prev)


_switch_interval_guard = _SwitchIntervalGuard()


@contextmanager
def phase_timer(
    name: str,
    *,
    prefix: str = "phase",
    interval_s: float = 15.0,
    log: Any = None,  # noqa: ANN401  # reason: structlog BoundLogger has no public exported type
    gpu_mb: bool = False,
    **fields: Any,  # noqa: ANN401  # reason: structlog fields are untyped by design
) -> Iterator[None]:
    """Time a load phase and emit a heartbeat while it runs.

    Wraps a block of code that may spend many seconds inside opaque
    third-party code (HF Hub HEAD requests, ``Policy.from_pretrained``
    allocation, safetensors deserialisation, ``.to(device)`` transfer).
    Emits one structured log on entry, one on exit with the elapsed
    wall-time, and one heartbeat every ``interval_s`` seconds in between
    so the operator can distinguish a slow phase from a hang.

    Args:
        name: Short phase label (``"imports"``, ``"from_pretrained"``,
            ``"materialize_processor_dir"``). Combined with ``prefix``
            to form the event name.
        prefix: Adapter-specific prefix (``"pi05"``, ``"smolvla"``,
            ``"act"``). Lets the operator filter heartbeat traffic per
            policy family. Default ``"phase"`` is fine for one-off
            instrumentation outside a named adapter.
        interval_s: Heartbeat period in seconds. The default (15 s)
            matches the historical pi05 heartbeat cadence — large enough
            that fast phases never emit a heartbeat, small enough that
            the operator notices within one screen-refresh that the
            phase is still alive.
        log: Optional ``structlog.BoundLogger``. Defaults to a logger
            bound to this module so calling code does not have to
            allocate one per phase.
        gpu_mb: When True, every heartbeat carries the current CUDA
            allocator usage. Use for phases that move tensors to / from
            the GPU; skip for CPU-only phases to keep the log noise
            down.
        **fields: Extra structured fields attached to every emitted
            log event (typically ``repo=...``, ``dtype=...``,
            ``device=...``).

    Yields:
        Nothing; the caller's block runs between ``_start`` and
        ``_done``.

    Example:
        >>> with phase_timer(
        ...     "from_pretrained", prefix="smolvla", repo="lerobot/smolvla_libero", gpu_mb=False
        ... ):
        ...     pass  # SmolVLAPolicy.from_pretrained(...) goes here
    """
    # Resolve the logger lazily inside the function body — otherwise a
    # module-level ``structlog.get_logger(__name__)`` proxy can bind to
    # a stale configuration if the process called ``structlog.configure``
    # after this module was imported (matters in unit tests + in the
    # `tools/profile_policy_load.py` profiler that installs its own
    # capture processor before driving a load).
    logger = log if log is not None else structlog.get_logger(__name__)
    start = time.monotonic()
    stop_event = threading.Event()
    event_start = f"{prefix}_{name}_start"
    event_heartbeat = f"{prefix}_{name}_heartbeat"
    event_done = f"{prefix}_{name}_done"

    logger.info(event_start, **fields)

    # Baseline for the major-fault delta reported on every heartbeat and on
    # ``_done`` — see :func:`_rss_majflt`.
    baseline = _rss_majflt()
    majflt_0 = baseline[1] if baseline is not None else 0

    def _mem_fields() -> dict[str, Any]:
        snap = _rss_majflt()
        if snap is None:
            return {}
        return {"rss_mb": round(snap[0], 1), "major_faults": snap[1] - majflt_0}

    def _tick() -> None:
        while not stop_event.wait(interval_s):
            elapsed = time.monotonic() - start
            extra: dict[str, Any] = {"elapsed_s": round(elapsed, 1)}
            if gpu_mb:
                mb = _gpu_mb()
                if mb is not None:
                    extra["gpu_mb"] = round(mb, 1)
            logger.info(event_heartbeat, **extra, **_mem_fields(), **fields)

    thread = threading.Thread(target=_tick, daemon=True, name=f"{prefix}_{name}_heartbeat")
    thread.start()
    # GIL relief: load phases run inside the same process as the deploy
    # graph's high-rate threads (two 30 fps opencv camera readers + the HAL's
    # joint-state publisher in runtime_node). At the default 5 ms switch
    # interval those threads preempt the loading thread constantly and the
    # convoy effect starves it: measured live on an SO-101 deploy, the
    # SmolVLA import phase got ~12% of one core and a 6 s import stretched
    # past 15 minutes. Raising the interval to 50 ms for the phase lets the
    # loader run in long slices while every peer thread still gets the GIL
    # ~20x/s — cameras drop to a reduced rate for a few seconds and the HAL
    # publisher stays far inside the safety kernel's 1 s staleness deadline.
    # Load phases are rare, operator-initiated events; steady-state rates are
    # untouched. Restored in the same finally that stops the heartbeat —
    # depth-counted so overlapping phases (see _SwitchIntervalGuard) restore
    # exactly once, from the outermost saved value.
    _switch_interval_guard.acquire()
    try:
        yield
    finally:
        _switch_interval_guard.release()
        stop_event.set()
        thread.join(timeout=interval_s)
        logger.info(
            event_done,
            elapsed_s=round(time.monotonic() - start, 1),
            **_mem_fields(),
            **fields,
        )
