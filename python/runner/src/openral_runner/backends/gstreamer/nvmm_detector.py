"""Clean-room NVMM zero-copy object detector (ADR-0037 PR5b).

Composes the engine bytes from
:meth:`~openral_rskill.runtime_tensorrt.TensorRTRuntime.serialized_engine`, the
:class:`~openral_runner.backends.gstreamer.trt_nvmm.TrtNvmmExecutor` (device-pointer
inference + RGBA->NCHW kernel), and the shared
:func:`~openral_runner.backends.gstreamer.objects_detector.postprocess_rtdetr`
decode. Consumes an :class:`~openral_runner.backends.gstreamer.nvbufsurface.NvBufSurfaceHandle`
(the GPU dataPtr of an NVMM frame) and emits :class:`~openral_core.ObjectsMetadata`
— the same output as the CPU tier, with no GPU->CPU copy.

Requires the ``tensorrt`` group (``cuda-python`` + ``tensorrt``) + ``nvrtc``;
the :class:`TrtNvmmExecutor` it wraps uses nvrtc + cuda-python (no pycuda), so
this tier deploys in the lean DeepStream ds-on image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from openral_core import ObjectsMetadata
from openral_core.exceptions import ROSConfigError

from openral_runner.backends.gstreamer.objects_detector import (
    identify_rtdetr_outputs,
    postprocess_rtdetr,
)
from openral_runner.backends.gstreamer.trt_nvmm import TrtNvmmExecutor

log = structlog.get_logger(__name__)

__all__ = ["NvmmObjectsDetector"]


class NvmmObjectsDetector:
    """Zero-copy NVMM object detector wrapping a TRT engine built from ONNX.

    Args:
        onnx_path: Path to the rSkill's ``*.onnx`` model.
        labels: Class-name list; index ``i`` ↔ class ``i``. Non-empty.
        model_id: Identifier embedded in emitted :class:`ObjectsMetadata`.
        input_size: ``(height, width)`` the NVMM branch scales frames to.
        score_threshold: Minimum sigmoid score, in ``[0, 1]``.
        device_index: CUDA device ordinal.
        quantization: Build-time TRT quantization (defaults to fp32; fp16 typical).

    Raises:
        ROSConfigError: On missing deps, bad ONNX/engine, or invalid args.

    Example:
        >>> # Exercised live in tests/unit/test_nvmm_detector.py;
        >>> # doctest skipped here because tensorrt / cuda-python are optional at doctest time.
        >>> pass
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        labels: list[str],
        model_id: str,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.5,
        device_index: int = 0,
        quantization: Any = None,  # noqa: ANN401  # reason: QuantizationConfig | None — avoid import at signature
    ) -> None:
        """Validate args, build the TRT engine via serialized_engine, allocate I/O buffers."""
        from openral_rskill.runtime_tensorrt import TensorRTRuntime  # noqa: PLC0415

        if not labels:
            raise ROSConfigError("NvmmObjectsDetector: labels list must be non-empty.")
        if not 0.0 <= score_threshold <= 1.0:
            raise ROSConfigError(
                f"NvmmObjectsDetector: score_threshold must be in [0, 1]; got {score_threshold!r}."
            )
        p = Path(onnx_path)
        if not p.exists():
            raise ROSConfigError(f"NvmmObjectsDetector: ONNX model not found at '{p}'.")

        runtime = TensorRTRuntime(
            device=f"cuda:{device_index}", rskill_id=model_id, quantization=quantization
        )
        engine_bytes = runtime.serialized_engine(p)
        self._executor = TrtNvmmExecutor(
            engine_bytes, input_size=input_size, device_index=device_index
        )
        try:
            self._logits_name, self._boxes_name = identify_rtdetr_outputs(
                self._executor.output_shapes()
            )
        except BaseException:
            # Free the executor's device buffers: a partial __init__ never
            # returns an instance, so the caller cannot call close() to reclaim
            # them (mirrors TrtNvmmExecutor's own __init__ guard).
            self._executor.close()
            raise
        self._labels = labels
        self._model_id = model_id
        self._score_threshold = score_threshold
        self.kind: str = "objects"
        log.debug("nvmm_detector.created", model_id=model_id, input_size=input_size)

    def detect_nvmm(self, handle: Any, sensor_id: str) -> ObjectsMetadata | None:  # noqa: ANN401  # reason: NvBufSurfaceHandle — avoid import at signature
        """Run zero-copy inference on an NVMM frame handle; return detections or ``None``.

        Boxes are reported in network-input pixel space (the branch scales frames
        to the network size), per the spec §11 convention.

        Args:
            handle: :class:`NvBufSurfaceHandle` (``gpu_ptr``/``width``/``height``/``pitch``).
            sensor_id: Sensor name forwarded to the emitted metadata.

        Returns:
            :class:`ObjectsMetadata`, or ``None`` if no detection passes threshold.
        """
        outputs = self._executor.infer_rgba_devptr(
            handle.gpu_ptr, width=handle.width, height=handle.height, pitch=handle.pitch
        )
        return postprocess_rtdetr(
            outputs[self._logits_name],
            outputs[self._boxes_name],
            labels=self._labels,
            model_id=self._model_id,
            sensor_id=sensor_id,
            score_threshold=self._score_threshold,
            frame_width=handle.width,
            frame_height=handle.height,
        )

    def close(self) -> None:
        """Release the executor's device buffers. Idempotent."""
        self._executor.close()
