"""Runtime Protocol and NullRuntime — inference backend contract.

The ``Runtime`` Protocol defines the interface every inference backend must
satisfy.  Backends live in separate modules so that optional heavy dependencies
(``torch``, ``onnxruntime``, ``tensorrt``) are never imported unless the caller
explicitly requests them.

Public surface
--------------
- ``Runtime``: Structural protocol checked at runtime via ``isinstance``.
- ``NullRuntime``: Lightweight no-op implementation for testing lifecycle
  plumbing without model weights or ML frameworks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from openral_core.schemas import QuantizationConfig


@runtime_checkable
class Runtime(Protocol):
    """Structural protocol for skill inference backends.

    All methods are called from the ``Skill`` lifecycle hooks.  Implementations
    must be thread-safe if ``step()`` is called concurrently (e.g. async
    executor pattern as in SmolVLA).  Instances are created by
    backend-specific constructors (e.g. ``PyTorchRuntime(...)``,
    ``OnnxRuntime(...)``, ``NullRuntime()``); the protocol itself is not
    instantiable.

    Properties:
        is_loaded: ``True`` after :meth:`load` completes successfully.
        device: PyTorch-style device string, e.g. ``"cpu"`` or ``"cuda:0"``.
    """

    @property
    def is_loaded(self) -> bool:
        """True after load() completes successfully."""
        ...

    @property
    def device(self) -> str:
        """PyTorch-style device string."""
        ...

    def load(self, path: Path | str) -> None:
        """Load model weights from *path* into memory.

        Args:
            path: Local filesystem path or URL to weights file
                (``*.safetensors``, ``*.onnx``, ``*.pt``, etc.).

        Raises:
            ROSRuntimeError: If the file is missing, corrupt, or incompatible.
        """
        ...

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run one forward pass and return named outputs.

        Args:
            inputs: Named input tensors / arrays.  Keys and shapes depend on
                the model contract declared in ``rskill.yaml``.

        Returns:
            Named output tensors / arrays.

        Raises:
            ROSRuntimeError: If no model is loaded or the forward pass fails.
            ROSInferenceTimeout: If the backend exceeds its wall-clock budget.
        """
        ...

    def quantize(self, config: QuantizationConfig) -> None:
        """Apply post-load quantization.

        Args:
            config: Target dtype, backend, and optional calibration dataset.

        Raises:
            ROSRuntimeError: If the requested dtype/backend combination is
                unsupported, or if no model has been loaded yet.
        """
        ...

    def warmup(self, inputs: dict[str, Any]) -> None:
        """Run a dummy forward pass to amortize JIT/kernel-launch overhead.

        Args:
            inputs: Dummy inputs with the correct shapes (values are ignored).
        """
        ...

    def unload(self) -> None:
        """Release model weights and free device memory."""
        ...


class NullRuntime:
    """No-op inference backend for testing and development.

    Satisfies the ``Runtime`` Protocol without requiring any ML framework.
    ``infer()`` always returns an empty dict.  All state transitions are
    tracked faithfully so lifecycle tests can assert on ``is_loaded``.

    Args:
        device: Device string to report (default ``"cpu"``).

    Example:
        >>> rt = NullRuntime()
        >>> rt.is_loaded
        False
        >>> rt.load("any/path.pt")
        >>> rt.is_loaded
        True
        >>> rt.unload()
        >>> rt.is_loaded
        False
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialize with the given device string.

        Args:
            device: PyTorch-style device string (e.g. ``"cpu"``, ``"cuda:0"``).
        """
        self._device = device
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """True after load() has been called and before unload()."""
        return self._loaded

    @property
    def device(self) -> str:
        """PyTorch-style device string."""
        return self._device

    def load(self, path: Path | str) -> None:
        """Mark weights as loaded (no I/O performed).

        Args:
            path: Ignored; accepted for protocol compatibility.
        """
        self._loaded = True

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return an empty dict without computation.

        Args:
            inputs: Ignored; accepted for protocol compatibility.

        Returns:
            Always ``{}``.
        """
        return {}

    def quantize(self, config: QuantizationConfig) -> None:
        """Accept any quantization config silently (no-op).

        Args:
            config: Ignored; accepted for protocol compatibility.
        """

    def warmup(self, inputs: dict[str, Any]) -> None:
        """No-op warmup.

        Args:
            inputs: Ignored; accepted for protocol compatibility.
        """

    def unload(self) -> None:
        """Mark weights as unloaded (no memory freed)."""
        self._loaded = False
