"""TensorRT inference backend — builds a TRT engine from ONNX on first load.

Heavy dependencies (``tensorrt`` and ``cuda.bindings`` from ``cuda-python``)
are **deferred to load()/infer()** so ``import openral_rskill`` stays clean on
hosts without the ``tensorrt`` dependency group — the same deferral pattern as
``runtime_onnx.py``. The pure helpers below carry no heavy imports and are
unit-tested on every host.

rSkills ship ONNX (TRT engines are not portable across GPU arch / TRT version);
this backend builds and caches an engine per host, keyed on
``(rskill_id, "tensorrt-sm<cc>-trt<ver>", QuantizationConfig)`` via
:class:`openral_rskill.engine_cache.EngineCache`.

Public surface
--------------
- ``TensorRTRuntime``: ``Runtime``-compatible backend for ``*.onnx`` models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog
from openral_core.exceptions import ROSQuantizationError, ROSRuntimeError
from openral_core.schemas import QuantizationConfig

from openral_rskill.engine_cache import EngineCache

log = structlog.get_logger()

_DYNAMIC_DIM_MAX = 8
"""Upper bound (and opt point) for each dynamic input dim in the build profile.

A dynamic dim (``-1`` in the ONNX shape, e.g. RT-DETR's batch axis) is built
for ``[1, _DYNAMIC_DIM_MAX]`` with the optimization point at the max. Per-skill
profile ranges (from ``rskill.yaml``) are a future refinement.
"""


def _engine_cache_tag(compute_capability: tuple[int, int], trt_version: str) -> str:
    """Build the EngineCache ``backend`` discriminator for a host.

    Folds the GPU compute capability and the TensorRT version into the cache
    backend string so a serialized engine is never reused across a different
    GPU architecture or TRT version (where it would fail to deserialize or be
    silently wrong). Pure — no GPU or TRT import required.

    The ``sm{major}{minor}`` form (e.g. ``sm89``) is unambiguous because every
    CUDA compute-capability minor version in all known values is a single digit,
    so the concatenation never collides (``8.9`` -> ``89`` cannot clash with
    ``89.x``, which does not exist).

    Args:
        compute_capability: ``(major, minor)`` CUDA compute capability,
            e.g. ``(8, 9)`` for an Ada RTX 4070.
        trt_version: ``tensorrt.__version__`` string, e.g. ``"10.5.0"``.

    Returns:
        A tag like ``"tensorrt-sm89-trt10.5.0"``.

    Example:
        >>> _engine_cache_tag((8, 9), "10.5.0")
        'tensorrt-sm89-trt10.5.0'
    """
    major, minor = compute_capability
    return f"tensorrt-sm{major}{minor}-trt{trt_version}"


def _import_cudart() -> Any:  # noqa: ANN401  # reason: cuda-python ships no py.typed stubs
    """Import ``cuda.bindings.runtime`` (cudart) lazily with a helpful error.

    Raises:
        ROSRuntimeError: If the ``cuda-python`` wheel is not installed; the
            message points to ``uv sync --group tensorrt`` rather than the bare
            ``ImportError`` that callers would otherwise see.
    """
    try:
        from cuda.bindings import runtime as cudart  # noqa: PLC0415,I001  # reason: optional GPU dep deferred
    except ImportError as exc:
        raise ROSRuntimeError(
            "TensorRTRuntime: 'cuda-python' is not installed. "
            "Install with: uv sync --group tensorrt"
        ) from exc
    return cudart


def _import_trt() -> Any:  # noqa: ANN401  # reason: tensorrt ships no py.typed stubs
    """Import ``tensorrt`` lazily with a helpful error if the group is missing.

    Raises:
        ROSRuntimeError: If the ``tensorrt`` wheel is not installed; the message
            points to ``uv sync --group tensorrt`` rather than the bare
            ``ImportError`` that callers would otherwise see.
    """
    try:
        import tensorrt as trt  # noqa: PLC0415  # reason: optional GPU dep deferred to load()
    except ImportError as exc:
        raise ROSRuntimeError(
            "TensorRTRuntime: 'tensorrt' is not installed. Install with: uv sync --group tensorrt"
        ) from exc
    return trt


def _cuda_check(result: tuple[Any, ...], cudart: Any) -> tuple[Any, ...]:  # noqa: ANN401  # reason: cudart is an untyped cuda.bindings module object
    """Raise on a non-success cudart return tuple; return the trailing values.

    ``cuda.bindings.runtime`` calls return ``(err, *rest)``. This unwraps the
    error and surfaces it as a typed ``ROSRuntimeError`` rather than letting a
    bare enum propagate.

    Args:
        result: The ``(err, *rest)`` tuple returned by a ``cuda.bindings.runtime``
            call; ``result[0]`` is the ``cudaError_t`` status enum.
        cudart: The imported ``cuda.bindings.runtime`` module (from
            :func:`_import_cudart`), used to read ``cudaError_t.cudaSuccess`` and
            decode the error string.

    Returns:
        The trailing values ``result[1:]`` (the call's actual outputs) on success.

    Raises:
        ROSRuntimeError: If ``result[0]`` is not ``cudaSuccess``; the message
            carries the decoded CUDA error string.
    """
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        _, msg = cudart.cudaGetErrorString(err)
        decoded = msg.decode() if isinstance(msg, bytes) else msg
        raise ROSRuntimeError(f"TensorRTRuntime: CUDA error: {decoded}")
    return result[1:]


def _detect_compute_capability(device_index: int) -> tuple[int, int]:
    """Return the ``(major, minor)`` CUDA compute capability of *device_index*.

    Uses ``cuda.bindings.runtime.cudaGetDeviceProperties`` — available wherever
    ``cuda-python`` is installed, with no TensorRT dependency.

    Args:
        device_index: Zero-based CUDA device ordinal.

    Returns:
        ``(major, minor)``, e.g. ``(8, 9)`` for an Ada RTX 4070.

    Raises:
        ROSRuntimeError: If ``cuda-python`` is missing or the device query fails.
    """
    cudart = _import_cudart()
    (props,) = _cuda_check(cudart.cudaGetDeviceProperties(device_index), cudart)
    return int(props.major), int(props.minor)


class TensorRTRuntime:
    """TensorRT inference backend that builds its engine from ONNX on first load.

    Satisfies the :class:`openral_rskill.runtime.Runtime` Protocol. The rSkill
    ships ``model.onnx``; this backend builds a serialized TensorRT engine on
    the first :meth:`load`, caches it per host (keyed on GPU arch + TRT version
    + :class:`QuantizationConfig`), and deserializes the cached engine on
    subsequent loads.

    Quantization (fp16 / int8) is a **build-time** flag — pass the target
    :class:`QuantizationConfig` at construction. Calling :meth:`quantize` with a
    different config raises (the engine is already built for the construction
    config).

    Args:
        device: PyTorch-style CUDA device string (``"cuda:0"``). CPU/MPS are
            rejected — TensorRT is CUDA-only.
        rskill_id: HF Hub repo id or local path; the cache-key namespace.
        quantization: Build-time quantization config. Defaults to fp32.
        cache: Engine cache. Defaults to a process-default :class:`EngineCache`.

    Raises:
        ROSRuntimeError: On a non-CUDA device, missing deps, or build/infer
            failure.

    Example:
        >>> rt = TensorRTRuntime(device="cuda:0", rskill_id="openral/rskill-rtdetr-coco-r18")
        >>> rt.is_loaded
        False
        >>> rt.device
        'cuda:0'
    """

    def __init__(
        self,
        device: str = "cuda:0",
        *,
        rskill_id: str,
        quantization: QuantizationConfig | None = None,
        cache: EngineCache | None = None,
    ) -> None:
        """Validate the device and stash config; no heavy imports here."""
        if not device.startswith("cuda"):
            raise ROSRuntimeError(
                f"TensorRTRuntime: device must be CUDA (got {device!r}); "
                "TensorRT has no CPU/MPS backend."
            )
        self._device = device
        if ":" in device:
            suffix = device.split(":", 1)[1]
            try:
                self._device_index = int(suffix)
            except ValueError:
                raise ROSRuntimeError(
                    f"TensorRTRuntime: invalid device index in {device!r}; "
                    "expected 'cuda' or 'cuda:<N>'."
                ) from None
        else:
            self._device_index = 0
        self._rskill_id = rskill_id
        self._quant = quantization or QuantizationConfig()
        self._cache = cache or EngineCache()
        self._engine: Any | None = None
        self._context: Any | None = None
        self._trt: Any | None = None
        log.debug("tensorrt_runtime.created", device=device, rskill_id=rskill_id)

    @property
    def is_loaded(self) -> bool:
        """True after :meth:`load` builds/deserializes an engine + context."""
        return self._engine is not None and self._context is not None

    @property
    def device(self) -> str:
        """PyTorch-style CUDA device string."""
        return self._device

    def _backend_tag(self, trt: Any) -> str:  # noqa: ANN401  # reason: trt is an untyped tensorrt module object
        """Compute the arch/version cache discriminator for this host."""
        cc = _detect_compute_capability(self._device_index)
        return _engine_cache_tag(cc, trt.__version__)

    def _add_optimization_profile(self, builder: Any, network: Any, config: Any) -> None:  # noqa: ANN401  # reason: builder/network/config are untyped tensorrt objects
        """Add a build profile for any dynamic input dims; no-op if all static.

        Each ``-1`` (dynamic) input dim is given the range
        ``[1, _DYNAMIC_DIM_MAX]`` with the opt point at ``_DYNAMIC_DIM_MAX``;
        static dims keep their fixed value. Without this, building a network
        that has dynamic inputs fails (TRT requires at least one profile).
        """
        has_dynamic = any(
            -1 in tuple(network.get_input(i).shape) for i in range(network.num_inputs)
        )
        if not has_dynamic:
            return
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            dims = tuple(inp.shape)
            min_shape = tuple(1 if d == -1 else d for d in dims)
            opt_shape = tuple(_DYNAMIC_DIM_MAX if d == -1 else d for d in dims)
            profile.set_shape(inp.name, min_shape, opt_shape, opt_shape)
        config.add_optimization_profile(profile)

    def _build_serialized_engine(self, onnx_path: Path, trt: Any) -> bytes:  # noqa: ANN401  # reason: trt is an untyped tensorrt module object
        """Parse ONNX and build a serialized TRT engine honoring quantization."""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        # ``parse_from_file`` (not ``parse(bytes)``) so TRT resolves an
        # external-data companion (e.g. ``model.onnx.data``) relative to the
        # ONNX file's own directory. Large exports — including the RT-DETR
        # detector rSkills produced by ``tools/export_rtdetr_onnx.py`` — store
        # their weights in a sidecar ``*.data`` file; ``parser.parse(f.read())``
        # has no path context and fails to open it ("Failed to open file:
        # model.onnx.data").
        if not parser.parse_from_file(str(onnx_path)):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ROSQuantizationError(
                f"TensorRTRuntime: ONNX parse failed for '{onnx_path}': {errors}"
            )
        config = builder.create_builder_config()
        # Only fp16 / bf16 / int8 are wired; fp32 (and currently int4 /
        # fp4_nvfp4) fall through with no flag → an fp32 build. Wiring the
        # sub-8-bit dtypes is out of scope here.
        if self._quant.dtype == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self._quant.dtype == "bf16":
            config.set_flag(trt.BuilderFlag.BF16)
        elif self._quant.dtype == "int8":
            # No INT8 calibrator is wired yet, so TRT may warn and fall back to
            # higher precision per-layer (degraded, not failed). A real
            # calibrator must be added before an int8 rSkill ships.
            config.set_flag(trt.BuilderFlag.INT8)
        # A network with dynamic input dims (e.g. RT-DETR's dynamic batch) needs
        # an optimization profile or build fails. Each -1 dim spans [1, MAX]
        # with opt at MAX; static dims are fixed. Static-only networks add no
        # profile (a no-op profile is unnecessary).
        self._add_optimization_profile(builder, network, config)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise ROSQuantizationError(
                f"TensorRTRuntime: build_serialized_network returned None for '{onnx_path}' "
                f"(dtype={self._quant.dtype})."
            )
        return bytes(serialized)

    def serialized_engine(self, path: Path | str) -> bytes:
        """Return the serialized TRT engine for *path*, building + caching on a miss.

        The portable artifact a zero-copy executor (NVMM aggregator) deserializes
        in its own CUDA context. No execution context is created here. A cache hit
        returns the cached bytes without a build.

        Args:
            path: Path to the rSkill's ``*.onnx`` model file.

        Returns:
            The serialized TensorRT engine bytes.

        Raises:
            ROSRuntimeError: If the ONNX file is missing.
            ROSQuantizationError: If the ONNX->TRT build fails.
        """
        onnx_path = Path(path)
        if not onnx_path.exists():
            raise ROSRuntimeError(f"TensorRTRuntime: ONNX model not found at '{onnx_path}'.")
        trt = _import_trt()
        key = self._cache.cache_key(self._rskill_id, self._backend_tag(trt), self._quant)
        cached = self._cache.get(key)
        if cached is not None:
            log.info("tensorrt_runtime.cache_hit", rskill_id=self._rskill_id, key=key)
            return cached.read_bytes()
        engine_bytes = self._build_serialized_engine(onnx_path, trt)
        import tempfile  # noqa: PLC0415  # reason: only needed on the cold-build path

        with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as tf:
            tf.write(engine_bytes)
            tmp_engine = Path(tf.name)
        try:
            self._cache.put(key, tmp_engine)
        finally:
            tmp_engine.unlink(missing_ok=True)
        log.info(
            "tensorrt_runtime.built",
            rskill_id=self._rskill_id,
            key=key,
            bytes=len(engine_bytes),
        )
        return engine_bytes

    def load(self, path: Path | str) -> None:
        """Build (cache miss) or deserialize (cache hit) the engine for *path*.

        Args:
            path: Path to the rSkill's ``*.onnx`` model file.

        Raises:
            ROSRuntimeError: If the ONNX file is missing or engine
                deserialization / context creation fails.
            ROSQuantizationError: If the ONNX->TRT build fails.
        """
        onnx_path = Path(path)
        # Guard file existence before importing tensorrt so that the "not found"
        # error is raised even on hosts without the tensorrt wheel installed.
        if not onnx_path.exists():
            raise ROSRuntimeError(f"TensorRTRuntime: ONNX model not found at '{onnx_path}'.")
        # Explicitly release any prior engine/context before re-loading so a
        # second load() doesn't rely on GC latency to free device memory.
        if self.is_loaded:
            self.unload()
        trt = _import_trt()
        # Recompute the key so the deserialize guard can self-heal a corrupt
        # cache entry. serialized_engine() computes its own key internally; this
        # cold-path device query is cheap and keeps the two in lockstep.
        key = self._cache.cache_key(self._rskill_id, self._backend_tag(trt), self._quant)
        engine_bytes = self.serialized_engine(onnx_path)

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            self._cache.invalidate(key)
            raise ROSRuntimeError(
                f"TensorRTRuntime: failed to deserialize engine for '{self._rskill_id}' "
                f"(key={key}); cache entry invalidated."
            )
        context = engine.create_execution_context()
        if context is None:
            raise ROSRuntimeError(
                f"TensorRTRuntime: failed to create execution context for '{self._rskill_id}'."
            )
        # Assign trt only on success so a failed load() never leaves _trt set
        # with _engine=None (the no-engine guard in infer() stays coherent).
        self._engine = engine
        self._context = context
        self._trt = trt

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run one forward pass on host numpy inputs; return host numpy outputs.

        Inputs are uploaded to device buffers, the engine runs on a fresh
        stream, and outputs are copied back. This is the generic host path; the
        zero-copy NVMM device-pointer path (detector element) reuses
        ``self._engine`` / ``self._context`` directly and is added in a later PR.

        Args:
            inputs: Map of input-tensor name -> ``numpy.ndarray`` (dtype/shape
                must match the engine's binding).

        Returns:
            Map of output-tensor name -> ``numpy.ndarray``.

        Raises:
            ROSRuntimeError: If no engine is loaded or a CUDA/exec call fails.
        """
        if self._engine is None or self._context is None or self._trt is None:
            raise ROSRuntimeError("TensorRTRuntime: no engine loaded; call load() first.")
        trt = self._trt
        cudart = _import_cudart()
        engine, context = self._engine, self._context

        (stream,) = _cuda_check(cudart.cudaStreamCreate(), cudart)
        device_buffers: dict[str, int] = {}
        host_outputs: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
        try:
            # Pass 1 — inputs only: set the (possibly dynamic) input shapes and
            # upload host data. Output shapes are NOT resolved until every input
            # shape is set, so allocating outputs here would read -1 dims for a
            # tensor that iterates before the inputs (e.g. RT-DETR dynamic batch).
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                    continue
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                arr = np.ascontiguousarray(inputs[name], dtype=dtype)
                context.set_input_shape(name, tuple(arr.shape))
                (dptr,) = _cuda_check(cudart.cudaMalloc(arr.nbytes), cudart)
                # Register before the memcpy so a failing copy still frees the
                # buffer in the finally block (no leak window).
                device_buffers[name] = int(dptr)
                context.set_tensor_address(name, int(dptr))
                _cuda_check(
                    cudart.cudaMemcpyAsync(
                        int(dptr),
                        arr.ctypes.data,
                        arr.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        int(stream),
                    ),
                    cudart,
                )

            # All inputs are bound — output shapes are now resolvable. Guard
            # against an under-specified context before reading output dims.
            if not getattr(context, "all_binding_shapes_specified", True):
                raise ROSRuntimeError(
                    "TensorRTRuntime: not all binding shapes are specified after "
                    "setting inputs; cannot resolve dynamic output shapes."
                )

            # Pass 2 — outputs only: context.get_tensor_shape() now returns the
            # concrete (resolved) dims, so np.empty() gets a valid shape.
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    continue
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                shape = tuple(context.get_tensor_shape(name))
                out = np.empty(shape, dtype=dtype)
                (dptr,) = _cuda_check(cudart.cudaMalloc(out.nbytes), cudart)
                device_buffers[name] = int(dptr)
                host_outputs[name] = out
                context.set_tensor_address(name, int(dptr))

            if not context.execute_async_v3(int(stream)):
                raise ROSRuntimeError("TensorRTRuntime: execute_async_v3 returned False.")

            for name, out in host_outputs.items():
                _cuda_check(
                    cudart.cudaMemcpyAsync(
                        out.ctypes.data,
                        device_buffers[name],
                        out.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        int(stream),
                    ),
                    cudart,
                )
            _cuda_check(cudart.cudaStreamSynchronize(int(stream)), cudart)
            return host_outputs
        finally:
            for dptr in device_buffers.values():
                cudart.cudaFree(dptr)
            cudart.cudaStreamDestroy(int(stream))

    def quantize(self, config: QuantizationConfig) -> None:
        """Validate the quantization config (TRT quant is applied at build time).

        The engine was built for the config passed at construction. A matching
        config is a no-op; a different one raises — reconstruct with the target
        config to rebuild. Mirrors ``ONNXRuntime.quantize`` (export-time) intent.

        Args:
            config: Must equal the construction-time config.

        Raises:
            ROSRuntimeError: If *config* differs from the build-time config.
        """
        if config != self._quant:
            raise ROSRuntimeError(
                "TensorRTRuntime: quantization is applied at build time. "
                f"Engine built for {self._quant.dtype!r}; cannot re-quantize to "
                f"{config.dtype!r}. Reconstruct TensorRTRuntime with the target config."
            )

    def warmup(self, inputs: dict[str, Any]) -> None:
        """Run one dummy forward pass to amortize first-call allocation cost.

        Args:
            inputs: Inputs with correct dtype/shape; output values are discarded.

        Raises:
            ROSRuntimeError: Propagated from :meth:`infer`.
        """
        self.infer(inputs)
        log.debug("tensorrt_runtime.warmed_up", device=self._device)

    def unload(self) -> None:
        """Release the engine and execution context (frees device memory)."""
        self._context = None
        self._engine = None
        self._trt = None
        log.info("tensorrt_runtime.unloaded", device=self._device)
