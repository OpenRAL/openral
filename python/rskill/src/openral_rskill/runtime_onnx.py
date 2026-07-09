"""ONNX Runtime inference backend — ``Runtime`` implementation backed by ``onnxruntime``.

The ``onnxruntime`` import is **deferred to construction time** so this
module loads cleanly on hosts without the wheel — ``import openral_rskill``
must not fail just because the optional ONNX backend isn't installed.
The deferral also lets ``pytest --doctest-modules`` collect the file in
the curated doctest set (CLAUDE.md §5.4); see
``tests/unit/test_doctest_runner.py::DOCTEST_TARGETS``.

Public surface
--------------
- ``ONNXRuntime``: ``Runtime``-compatible backend for ``*.onnx`` model files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from openral_core.exceptions import ROSRuntimeError
from openral_core.schemas import QuantizationConfig

if TYPE_CHECKING:
    import onnxruntime as ort  # type: ignore[import-untyped]  # reason: onnxruntime ships no py.typed marker

log = structlog.get_logger()

_CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_CPU_PROVIDERS = ["CPUExecutionProvider"]


def _import_ort() -> Any:  # noqa: ANN401  # reason: onnxruntime ships no stubs
    """Import ``onnxruntime`` lazily, with a helpful error if missing.

    Raises:
        ROSRuntimeError: If the ``onnxruntime`` wheel is not installed; the
            message points to the install command rather than the bare
            ``ModuleNotFoundError`` that callers would otherwise see.
    """
    try:
        import onnxruntime as ort  # noqa: PLC0415  # reason: optional backend deferred to construction time
    except ImportError as exc:
        raise ROSRuntimeError(
            "ONNXRuntime: 'onnxruntime' is not installed. "
            "Install with: uv add onnxruntime --package openral-rskill"
        ) from exc
    return ort


class ONNXRuntime:
    """ONNX Runtime-based inference backend.

    Runs exported ``*.onnx`` models via ``onnxruntime.InferenceSession``.
    Automatically selects CUDA or CPU execution providers based on *device*.

    Quantization note
    -----------------
    ONNX quantization must be applied *before* loading, at export time, using
    ``onnxruntime.quantization.quantize_dynamic`` or ``quantize_static``.
    Calling :meth:`quantize` on an already-loaded session raises
    ``ROSRuntimeError``; use ``tools/quantize_onnx.py`` (planned) instead.

    Args:
        device: PyTorch-style device string.  Only ``"cpu"`` and ``"cuda:N"``
            are supported; ``"mps"`` is not supported by ONNX Runtime.

    Raises:
        ROSRuntimeError: Propagated from :meth:`load` and :meth:`infer`.

    Example:
        >>> rt = ONNXRuntime(device="cpu")
        >>> rt.is_loaded
        False
        >>> rt.device
        'cpu'
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialize and validate the device string.

        Args:
            device: PyTorch-style device string.  ``"mps"`` is rejected
                immediately since ONNX Runtime does not support Apple MPS.

        Raises:
            ROSRuntimeError: If *device* starts with ``"mps"``.
        """
        if device.startswith("mps"):
            raise ROSRuntimeError(
                "ONNXRuntime: Apple MPS is not supported by onnxruntime. "
                "Use MLXRuntime for Apple Silicon inference."
            )
        self._device = device
        self._session: ort.InferenceSession | None = None
        log.debug("onnx_runtime.created", device=device)

    @property
    def is_loaded(self) -> bool:
        """True after :meth:`load` completes successfully."""
        return self._session is not None

    @property
    def device(self) -> str:
        """PyTorch-style device string."""
        return self._device

    def load(self, path: Path | str) -> None:
        """Create an ``onnxruntime.InferenceSession`` from *path*.

        Args:
            path: Path to an ``*.onnx`` model file.

        Raises:
            ROSRuntimeError: If the file does not exist or the session cannot
                be created.
        """
        p = Path(path)
        if not p.exists():
            raise ROSRuntimeError(f"ONNXRuntime: model file not found at '{p}'.")
        ort = _import_ort()
        providers = _CUDA_PROVIDERS if self._device.startswith("cuda") else _CPU_PROVIDERS
        try:
            self._session = ort.InferenceSession(str(p), providers=providers)
        except Exception as exc:
            raise ROSRuntimeError(
                f"ONNXRuntime: failed to create InferenceSession for '{p}': {exc}"
            ) from exc
        log.info("onnx_runtime.loaded", path=str(p), providers=providers)

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run one ONNX forward pass.

        Input values that are not already ``numpy.ndarray`` are converted via
        ``numpy.asarray()``.

        Args:
            inputs: Named input arrays.  Keys must match the model's input
                names (check ``session.get_inputs()`` for the exact names).

        Returns:
            Named output arrays.

        Raises:
            ROSRuntimeError: If no session is loaded.
        """
        if self._session is None:
            raise ROSRuntimeError("ONNXRuntime: no model loaded; call load() first.")
        np_inputs = {
            k: v if isinstance(v, np.ndarray) else np.asarray(v) for k, v in inputs.items()
        }
        raw = self._session.run(None, np_inputs)
        output_names = [o.name for o in self._session.get_outputs()]
        return dict(zip(output_names, raw, strict=False))

    def quantize(self, config: QuantizationConfig) -> None:
        """Raise always — ONNX quantization is not applied at runtime.

        Args:
            config: Ignored; ONNX quantization must be applied at export time.

        Raises:
            ROSRuntimeError: Always; explains the correct offline workflow.
        """
        raise ROSRuntimeError(
            "ONNXRuntime: quantization must be applied offline before loading. "
            "Use 'onnxruntime.quantization.quantize_dynamic(model_path, output_path)' "
            "or the planned 'tools/quantize_onnx.py' utility, then pass the "
            "quantized *.onnx path to load()."
        )

    def warmup(self, inputs: dict[str, Any]) -> None:
        """Run one forward pass to prime the ONNX execution provider.

        Args:
            inputs: Dummy inputs with correct shapes (values are ignored by the
                provider's kernel-compilation step).

        Raises:
            ROSRuntimeError: Propagated from :meth:`infer`.
        """
        self.infer(inputs)
        log.debug("onnx_runtime.warmed_up", device=self._device)

    def unload(self) -> None:
        """Release the ``InferenceSession`` reference."""
        self._session = None
        log.info("onnx_runtime.unloaded", device=self._device)
