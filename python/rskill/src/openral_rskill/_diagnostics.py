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
    <prefix>_<name>_heartbeat {elapsed_s, [gpu_mb], **fields}    # every interval_s
    <prefix>_<name>_heartbeat {...}
    ...
    <prefix>_<name>_done {elapsed_s, **fields}

Example:
-------
>>> from openral_rskill._diagnostics import phase_timer
>>> with phase_timer("load_weights", prefix="smolvla", repo="lerobot/smolvla_libero"):
...     pass  # heavy work goes here
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

__all__ = ["phase_timer"]


def _gpu_mb() -> float | None:
    """Return current CUDA allocator usage in MB, or ``None`` if unavailable.

    Cheap: a single ``torch.cuda.memory_allocated()`` call. Imports
    torch lazily so a CPU-only host that wraps a phase with
    ``gpu_mb=True`` still works (returns ``None``).
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        return None


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

    def _tick() -> None:
        while not stop_event.wait(interval_s):
            elapsed = time.monotonic() - start
            extra: dict[str, Any] = {"elapsed_s": round(elapsed, 1)}
            if gpu_mb:
                mb = _gpu_mb()
                if mb is not None:
                    extra["gpu_mb"] = round(mb, 1)
            logger.info(event_heartbeat, **extra, **fields)

    thread = threading.Thread(target=_tick, daemon=True, name=f"{prefix}_{name}_heartbeat")
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=interval_s)
        logger.info(event_done, elapsed_s=round(time.monotonic() - start, 1), **fields)
