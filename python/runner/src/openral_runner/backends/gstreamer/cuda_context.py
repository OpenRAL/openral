"""Process-wide shared PyCUDA context (ADR-0010 PR I/3, ADR-0011).

A single CUDA context is allocated lazily on first call to
:func:`get_shared_cuda_context` and pushed onto the current thread.
``atexit`` detaches it on shutdown. The context is reused across the
GStreamer reader's NVMM→CUDA handoff, future Skill GPU consumers, and
any TensorRT engines that share the same address space — having one
context (instead of one per consumer) avoids the
``CUDA_ERROR_INVALID_CONTEXT`` failure mode when multiple PyCUDA users
each call ``cuda.init()`` at startup.

This module is **lazy in PyCUDA**: importing it is cheap (no
``import pycuda``); the first call to :func:`get_shared_cuda_context`
imports ``pycuda.driver`` and initialises the device. The reader's
CPU path therefore never pays the PyCUDA load cost.

Provenance: derived from work originally authored by Adrian Llopart
(adrianllopart@gmail.com) and re-licensed under Apache-2.0 for
openral with the author's explicit consent.
"""

from __future__ import annotations

import atexit
import os
import threading
from typing import TYPE_CHECKING, Any, Final

import structlog

if TYPE_CHECKING:
    import pycuda.driver as cuda

__all__ = [
    "cuda_context_state",
    "get_shared_cuda_context",
    "get_shared_cuda_device_index",
]

log = structlog.get_logger(__name__)

# Default GPU index to bind the shared context to. Multi-GPU hosts may want
# to override via the ``OPENRAL_CUDA_DEVICE_INDEX`` env var (read on first init).
_DEFAULT_DEVICE_INDEX: Final[int] = 0


class _SharedCudaContext:
    """Thread-safe singleton wrapping a single PyCUDA context.

    Instance lifecycle is module-level (see :data:`_singleton`). Tests
    that need to reset it should call :meth:`_reset_for_tests` rather
    than constructing a new instance — the atexit hook is registered
    on first init and is process-scoped.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._context: cuda.Context | None = None
        self._device: cuda.Device | None = None
        self._device_index: int = _DEFAULT_DEVICE_INDEX
        self._initialised = False
        self._atexit_registered = False

    def get(self) -> cuda.Context:
        """Return the shared context, creating it on first call."""
        with self._lock:
            if not self._initialised:
                self._initialise_locked()
            assert self._context is not None  # invariant after init
            return self._context

    def device_index(self) -> int:
        """Return the device index the shared context is bound to."""
        with self._lock:
            if not self._initialised:
                self._initialise_locked()
            return self._device_index

    def cleanup(self) -> None:
        """Detach the context. Idempotent; safe to call multiple times."""
        with self._lock:
            if not self._initialised or self._context is None:
                return
            try:
                self._context.detach()
            except Exception as exc:  # reason: PyCUDA cleanup raises a zoo
                log.warning(
                    "cuda_context.detach_failed",
                    error=str(exc),
                    note="context may have been detached already",
                )
            finally:
                self._initialised = False
                self._context = None
                self._device = None

    def _initialise_locked(self) -> None:
        """Initialise the device + context. Caller must hold ``_lock``."""
        try:
            import pycuda.driver as cuda  # noqa: PLC0415  # reason: PyCUDA optional
        except ImportError as exc:
            raise RuntimeError(
                "get_shared_cuda_context() called but PyCUDA is not installed. "
                "Install openral-runner[gstreamer-nvmm] (or pycuda directly)."
            ) from exc

        env_idx = os.getenv("OPENRAL_CUDA_DEVICE_INDEX")
        if env_idx is not None:
            try:
                self._device_index = int(env_idx)
            except ValueError as exc:
                raise RuntimeError(
                    f"OPENRAL_CUDA_DEVICE_INDEX={env_idx!r} is not an integer"
                ) from exc

        cuda.init()
        self._device = cuda.Device(self._device_index)
        self._context = self._device.make_context()
        self._context.push()
        self._initialised = True
        if not self._atexit_registered:
            atexit.register(self.cleanup)
            self._atexit_registered = True
        log.debug(
            "cuda_context.initialised",
            device_index=self._device_index,
            device_name=self._device.name(),
        )

    def _reset_for_tests(self) -> None:
        """Reset the singleton between unit tests. Internal use only."""
        self.cleanup()


_singleton = _SharedCudaContext()


def get_shared_cuda_context() -> Any:  # noqa: ANN401  # reason: pycuda.driver.Context — optional import
    """Return the process-wide shared PyCUDA context.

    First call creates the context on
    :envvar:`OPENRAL_CUDA_DEVICE_INDEX` (default ``0``); subsequent calls
    return the same object. The context is pushed onto the current
    thread on creation; PyCUDA expects callers to push/pop around
    operations that span threads.

    Raises:
        RuntimeError: When PyCUDA is not installed, or when
            ``OPENRAL_CUDA_DEVICE_INDEX`` is set to a non-integer.

    Example:
        >>> # Doctest exercised in tests/unit/test_gstreamer_cuda_context.py
        >>> # to avoid requiring CUDA at doctest time.
        >>> pass
    """
    return _singleton.get()


def get_shared_cuda_device_index() -> int:
    """Return the device index the shared context is bound to."""
    return _singleton.device_index()


def cuda_context_state() -> dict[str, object]:
    """Return a structured snapshot of the singleton for ``openral doctor``.

    Designed to be JSON-serialisable: includes whether the context is
    initialised, the bound device index, and (when initialised) the
    device name. Does **not** trigger initialisation.
    """
    with _singleton._lock:  # reason: doctor probe reads internal flag
        initialised = _singleton._initialised
        device_name: str | None = None
        if initialised and _singleton._device is not None:
            device_name = _singleton._device.name()
    return {
        "initialised": initialised,
        "device_index": _singleton._device_index,
        "device_name": device_name,
    }
