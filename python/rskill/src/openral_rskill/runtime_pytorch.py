"""PyTorch inference backend — ``Runtime`` implementation backed by ``torch``.

This module imports ``torch`` at the module level.  Import it only when PyTorch
is installed; otherwise, use ``NullRuntime`` or ``ONNXRuntime``.

Public surface
--------------
- ``PyTorchRuntime``: ``Runtime``-compatible backend for ``*.pt`` / ``*.safetensors``
  weight files executed with ``torch.inference_mode()``. Use
  :meth:`PyTorchRuntime.load_safetensors` (safe; no code execution) for new
  skills; :meth:`PyTorchRuntime.load` unpickles a full module and is gated
  behind ``OPENRAL_ALLOW_UNSAFE_PICKLE`` (security audit 2026-06, C2).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
import torch
from openral_core.exceptions import ROSRuntimeError
from openral_core.schemas import (
    QuantizationBackend,
    QuantizationConfig,
    QuantizationDtype,
)

log = structlog.get_logger()

# Loading a full pickled ``torch.nn.Module`` executes arbitrary code embedded in
# the checkpoint (``__reduce__``).  For HF-Hub-sourced rSkill weights this is a
# remote-code-execution sink, and rSkill signature verification is not yet
# implemented (ADR-0006).  The unsafe load is therefore refused unless the
# operator explicitly acknowledges the trust assumption via this env var.
# CLAUDE.md §1.4 (explicit beats implicit) / §6 (default to safer).
_ALLOW_UNSAFE_PICKLE_ENV = "OPENRAL_ALLOW_UNSAFE_PICKLE"


class PyTorchRuntime:
    """PyTorch-based inference runtime.

    Loads a ``torch.nn.Module`` from a ``.pt`` checkpoint (``torch.save`` /
    ``torch.load``) and runs inference with ``torch.inference_mode()``.

    Quantization support
    --------------------
    - ``INT8 + PYTORCH`` backend: ``torch.quantization.quantize_dynamic`` on
      ``torch.nn.Linear`` layers.
    - All other dtype/backend combinations raise ``ROSRuntimeError``; use the
      appropriate runtime adapter for TensorRT, GGUF, or MLX.

    Args:
        device: PyTorch device string, e.g. ``"cpu"``, ``"cuda:0"``, ``"mps"``.

    Raises:
        ROSRuntimeError: Propagated from :meth:`load`, :meth:`infer`,
            :meth:`quantize`.

    Example:
        >>> rt = PyTorchRuntime(device="cpu")
        >>> rt.is_loaded
        False
        >>> # rt.load("model.pt")  # requires a real checkpoint
        >>> rt.device
        'cpu'
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialize the runtime with the target device.

        Args:
            device: PyTorch-style device string (``"cpu"``, ``"cuda:0"``, ``"mps"``).
        """
        self._device = device
        self._model: torch.nn.Module | None = None
        log.debug("pytorch_runtime.created", device=device)

    @property
    def is_loaded(self) -> bool:
        """True after :meth:`load` completes successfully."""
        return self._model is not None

    @property
    def device(self) -> str:
        """PyTorch device string."""
        return self._device

    def load(self, path: Path | str) -> None:
        """Load a ``torch.nn.Module`` from *path*.

        The file must have been saved with ``torch.save(model, path)`` (full
        model, not state-dict only).  For state-dict loading, subclass and
        override :meth:`load`.

        Security
        --------
        Deserializing a full pickled module runs ``torch.load`` with
        ``weights_only=False``, which **executes arbitrary code embedded in the
        checkpoint**.  Because rSkill weights are fetched from third-party HF Hub
        repos and signature verification is not yet implemented (ADR-0006), the
        load is refused unless ``OPENRAL_ALLOW_UNSAFE_PICKLE=1`` is set to
        acknowledge that the checkpoint is trusted.  Prefer ``.safetensors``
        weights, which load without code execution.

        Args:
            path: Path to the ``.pt`` checkpoint.

        Raises:
            ROSRuntimeError: If the file does not exist, the unsafe-pickle
                acknowledgement env var is unset, the file cannot be loaded, or
                the object is not a ``torch.nn.Module``.
        """
        p = Path(path)
        if not p.exists():
            raise ROSRuntimeError(f"PyTorchRuntime: checkpoint not found at '{p}'.")
        if os.environ.get(_ALLOW_UNSAFE_PICKLE_ENV, "0") != "1":
            raise ROSRuntimeError(
                f"PyTorchRuntime: refusing to load '{p}'. Loading a pickled torch.nn.Module "
                "executes arbitrary code from the checkpoint (remote-code-execution risk for "
                "untrusted or unsigned weights). rSkill signature verification is not yet "
                "implemented (ADR-0006), so this is blocked by default. To load a TRUSTED "
                f"checkpoint, set: export {_ALLOW_UNSAFE_PICKLE_ENV}=1 . "
                "Prefer .safetensors weights, which load without code execution."
            )
        log.warning(
            "pytorch_runtime.unsafe_pickle_load",
            path=str(p),
            env=_ALLOW_UNSAFE_PICKLE_ENV,
            note="Loading a pickled module executes arbitrary code; ensure the checkpoint "
            "is from a trusted, verified source.",
        )
        try:
            obj = torch.load(str(p), map_location=self._device, weights_only=False)
        except Exception as exc:
            raise ROSRuntimeError(f"PyTorchRuntime: failed to load '{p}': {exc}") from exc
        if not isinstance(obj, torch.nn.Module):
            raise ROSRuntimeError(
                f"PyTorchRuntime: expected a torch.nn.Module but got {type(obj).__name__}. "
                "Save the full model (not a state dict) with torch.save(model, path)."
            )
        self._model = obj.to(self._device).eval()
        log.info("pytorch_runtime.loaded", path=str(p), device=self._device)

    def load_safetensors(
        self, path: Path | str, *, model: torch.nn.Module, strict: bool = True
    ) -> None:
        """Load a ``state_dict`` from a ``.safetensors`` file into *model*.

        This is the **safe** counterpart to :meth:`load`. A ``.safetensors`` file
        holds only tensors (a flat ``state_dict``) — never a pickled object — so
        loading it cannot execute code, and no ``OPENRAL_ALLOW_UNSAFE_PICKLE``
        acknowledgement is required. The caller supplies the architecture
        (*model*); its parameters are populated from the file. This is the
        recommended path for any new PyTorch-backed rSkill (security audit
        2026-06, C2): ship ``model.safetensors`` + construct the module from a
        known class instead of unpickling ``torch.save(model, ...)``.

        Args:
            path: Path to the ``.safetensors`` weights.
            model: A freshly constructed ``torch.nn.Module`` of the correct
                architecture to receive the weights.
            strict: Forwarded to ``model.load_state_dict``; when ``True`` (the
                default) every key must match exactly.

        Raises:
            ROSRuntimeError: If the file does not exist, cannot be parsed, or the
                ``state_dict`` does not fit *model* (when ``strict``).

        Example:
            >>> # rt = PyTorchRuntime(device="cpu")
            >>> # rt.load_safetensors("model.safetensors", model=MyPolicy())
        """
        p = Path(path)
        if not p.exists():
            raise ROSRuntimeError(f"PyTorchRuntime: safetensors file not found at '{p}'.")
        try:
            from safetensors.torch import load_file  # noqa: PLC0415
        except ImportError as exc:
            raise ROSRuntimeError(
                "PyTorchRuntime.load_safetensors requires 'safetensors'. "
                "It ships with torch/transformers; install the rskill extras."
            ) from exc
        try:
            state = load_file(str(p), device=self._device)
        except Exception as exc:
            raise ROSRuntimeError(
                f"PyTorchRuntime: failed to read safetensors '{p}': {exc}"
            ) from exc
        try:
            model.load_state_dict(state, strict=strict)
        except Exception as exc:
            raise ROSRuntimeError(
                f"PyTorchRuntime: state_dict from '{p}' does not fit the provided model: {exc}"
            ) from exc
        self._model = model.to(self._device).eval()
        log.info(
            "pytorch_runtime.loaded_safetensors", path=str(p), device=self._device, strict=strict
        )

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run one forward pass under ``torch.inference_mode()``.

        Args:
            inputs: Named input tensors.  Values that are not already
                ``torch.Tensor`` are converted via ``torch.as_tensor()``.

        Returns:
            Named output tensors.  If the model returns a single tensor, it is
            wrapped under the key ``"output"``.

        Raises:
            ROSRuntimeError: If no model is loaded.
        """
        if self._model is None:
            raise ROSRuntimeError("PyTorchRuntime: no model loaded; call load() first.")
        tensor_inputs = {
            k: v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for k, v in inputs.items()
        }
        with torch.inference_mode():
            output = self._model(tensor_inputs)
        if isinstance(output, dict):
            return output
        if isinstance(output, torch.Tensor):
            return {"output": output}
        return {"output": output}

    def quantize(self, config: QuantizationConfig) -> None:
        """Apply dynamic INT8 quantization to loaded Linear layers.

        Only ``dtype=INT8`` with ``backend=PYTORCH`` is supported in-process.
        All other combinations require export-time tooling.

        Args:
            config: Quantization specification.

        Raises:
            ROSRuntimeError: If no model is loaded, or the dtype/backend pair
                is unsupported.
        """
        if self._model is None:
            raise ROSRuntimeError("PyTorchRuntime: quantize() called before load().")
        if config.dtype is QuantizationDtype.INT8 and config.backend is QuantizationBackend.PYTORCH:
            # NOTE: torch.quantization.quantize_dynamic is deprecated in torch >=2.10.
            # Migration: torchao.quantization.quantize_(model, Int8DynActInt8WeightConfig())
            # See: https://github.com/pytorch/ao/issues/2259
            self._model = torch.quantization.quantize_dynamic(  # type: ignore[attr-defined]
                self._model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            log.info("pytorch_runtime.quantized", dtype="int8", backend="pytorch")
        else:
            raise ROSRuntimeError(
                f"PyTorchRuntime: unsupported quantization dtype={config.dtype.value!r} "
                f"backend={config.backend.value!r}. Use the matching runtime adapter "
                "(TensorRTRuntime, GGUFRuntime, MLXRuntime) or export offline."
            )

    def warmup(self, inputs: dict[str, Any]) -> None:
        """Run one forward pass to amortize JIT and CUDA kernel-launch overhead.

        Args:
            inputs: Dummy inputs with correct shapes (values are ignored).

        Raises:
            ROSRuntimeError: Propagated from :meth:`infer`.
        """
        self.infer(inputs)
        log.debug("pytorch_runtime.warmed_up", device=self._device)

    def unload(self) -> None:
        """Delete the model reference and optionally empty the CUDA cache."""
        self._model = None
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()
        log.info("pytorch_runtime.unloaded", device=self._device)
