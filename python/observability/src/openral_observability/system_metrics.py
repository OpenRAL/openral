"""Background sampler for host / GPU / RAM gauges.

Starts a tiny daemon thread that samples ``psutil`` + ``pynvml`` every N
seconds and records into the OpenRAL meter so the dashboard's "System"
card shows live GPU/CPU/RAM utilisation. Idempotent: calling
:func:`start_system_metrics_collector` twice with the same interval is
a no-op.

Wired into the SDK lifecycle:
:func:`openral_observability.configure_observability` starts the
collector after the meter provider is installed, and
:func:`openral_observability.shutdown_observability` stops it before
draining the providers. Callers therefore do not need to invoke
``start_system_metrics_collector`` explicitly — it ships with every
observability-configured process.

``psutil`` and ``nvidia-ml-py`` (which provides ``pynvml``) are direct
dependencies of ``openral-observability`` — declared per CLAUDE.md §1.4
"explicit beats implicit", not relied on transitively. ``pynvml``
imports cleanly on hosts without an NVIDIA driver; ``nvmlInit()`` then
fails at runtime and the GPU path silently no-ops while CPU + RAM keep
flowing. If neither dependency is importable for some other reason,
``start_system_metrics_collector`` returns ``False`` and the
dashboard's System health card stays empty.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from openral_observability import metrics, semconv

__all__ = ["start_system_metrics_collector", "stop_system_metrics_collector"]

_LOG = logging.getLogger(__name__)

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None
_interval_s: float = 1.0


def start_system_metrics_collector(*, interval_s: float = 1.0) -> bool:
    """Start the background sampler. Returns ``True`` when it ran.

    ``False`` is returned when neither ``psutil`` nor ``pynvml`` could be
    imported — in that case the function is a quiet no-op and the
    dashboard's System card just stays empty.
    """
    global _thread, _stop_event, _interval_s

    cpu_ok, gpu_ok = _probe_availability()
    if not (cpu_ok or gpu_ok):
        _LOG.debug("system_metrics: neither psutil nor pynvml importable; collector skipped")
        return False

    with _lock:
        if _thread is not None and _thread.is_alive():
            _interval_s = interval_s
            return True
        _stop_event = threading.Event()
        _interval_s = interval_s
        _thread = threading.Thread(
            target=_run,
            args=(_stop_event,),
            name="openral-system-metrics",
            daemon=True,
        )
        _thread.start()
        return True


def stop_system_metrics_collector(*, timeout_s: float = 2.0) -> None:
    """Signal the collector to stop and join. Safe to call when not running."""
    global _thread, _stop_event
    with _lock:
        if _stop_event is not None:
            _stop_event.set()
        thread = _thread
    if thread is not None:
        thread.join(timeout=timeout_s)
    with _lock:
        _thread = None
        _stop_event = None


def _probe_availability() -> tuple[bool, bool]:
    cpu_ok = False
    try:
        # reason: psutil has no stubs in dev group; runtime-optional
        import psutil  # type: ignore[import-untyped]

        _ = psutil
    except ImportError:
        pass
    else:
        cpu_ok = True
    gpu_ok = False
    try:
        import pynvml  # type: ignore[import-untyped]  # reason: nvidia-ml-py ships no py.typed marker; runtime-safe via outer try/except for hosts that strip the dep

        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        gpu_ok = True
    except Exception:
        pass
    return cpu_ok, gpu_ok


def _set_abs(
    instrument: Any,
    prev: dict[str, float],
    key: str,
    value: float,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Record an absolute reading on an UpDownCounter via deltas.

    UpDownCounters accept *delta* arguments; we keep the previous
    absolute and emit ``value - prev`` so the cumulative on the wire
    matches the sampled absolute. The first sample for a key is emitted
    unconditionally — even when ``value == 0`` — so the SDK has an
    aggregator entry to export. Skipping it left the instrument
    untouched, the export round produced no data point, and the
    dashboard's System health card stayed empty until a non-zero delta
    finally fired.
    """
    if key not in prev:
        instrument.add(value, attributes=attrs or {})
        prev[key] = value
        return
    delta = value - prev[key]
    if delta != 0.0:
        instrument.add(delta, attributes=attrs or {})
    prev[key] = value


def _sample_once(
    psutil_mod: Any,
    pynvml_mod: Any,
    instruments: dict[str, Any],
    prev: dict[str, float],
) -> None:
    """One sampling tick: read host + GPU and emit deltas. Exceptions are swallowed."""
    if psutil_mod is not None:
        _set_abs(instruments["cpu_util"], prev, "cpu", float(psutil_mod.cpu_percent(interval=None)))
        vm = psutil_mod.virtual_memory()
        _set_abs(instruments["ram_used"], prev, "ram_used", vm.used / (1024 * 1024))
        _set_abs(instruments["ram_total"], prev, "ram_total", vm.total / (1024 * 1024))
    if pynvml_mod is None:
        return
    for i in range(pynvml_mod.nvmlDeviceGetCount()):
        handle = pynvml_mod.nvmlDeviceGetHandleByIndex(i)
        name = pynvml_mod.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        labels = {semconv.SYSTEM_GPU_INDEX: i, semconv.SYSTEM_GPU_NAME: name}
        util = pynvml_mod.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml_mod.nvmlDeviceGetMemoryInfo(handle)
        _set_abs(instruments["gpu_util"], prev, f"gpu_util.{i}", float(util.gpu), labels)
        _set_abs(
            instruments["gpu_mem_used"], prev, f"gpu_used.{i}", mem.used / (1024 * 1024), labels
        )
        _set_abs(
            instruments["gpu_mem_total"], prev, f"gpu_total.{i}", mem.total / (1024 * 1024), labels
        )


def _run(stop_event: threading.Event) -> None:
    cpu_ok, gpu_ok = _probe_availability()
    psutil_mod: Any = None
    pynvml_mod: Any = None
    if cpu_ok:
        import psutil as psutil_mod  # type: ignore[no-redef]
    if gpu_ok:
        import pynvml as pynvml_mod  # type: ignore[no-redef]

        try:
            pynvml_mod.nvmlInit()
        except Exception:
            pynvml_mod = None

    # Cache the meter instruments once — they're cheap thanks to
    # `_cached` but skipping the dict lookup keeps the hot loop tight.
    instruments = {
        "cpu_util": metrics.get_system_cpu_util_pct(),
        "ram_used": metrics.get_system_ram_used_mb(),
        "ram_total": metrics.get_system_ram_total_mb(),
        "gpu_mem_used": metrics.get_system_gpu_memory_used_mb(),
        "gpu_mem_total": metrics.get_system_gpu_memory_total_mb(),
        "gpu_util": metrics.get_system_gpu_util_pct(),
    }
    prev: dict[str, float] = {}

    while not stop_event.is_set():
        try:
            _sample_once(psutil_mod, pynvml_mod, instruments, prev)
        except Exception:
            _LOG.exception("system_metrics: sampler iteration failed; continuing")
        stop_event.wait(timeout=_interval_s)

    if pynvml_mod is not None:
        import contextlib

        with contextlib.suppress(Exception):
            pynvml_mod.nvmlShutdown()
