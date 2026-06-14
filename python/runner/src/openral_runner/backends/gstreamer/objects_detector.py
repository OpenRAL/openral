"""CPU-tier object detector for the ADR-0037 perception event tee.

This module implements the **CPU (ONNXRuntime, system-memory BGR)** tier of
the ADR-0037 object detector. It implements the :class:`EventDetector` protocol
defined in :mod:`openral_runner.backends.gstreamer.perception_tee`, and so
plugs directly into the existing :class:`PerceptionEventPublisher` — pass an
:class:`ObjectsDetector` instance in the ``detectors`` list and it publishes
``openral_msgs/PromptStamped`` on ``/openral/perception/objects``.

The detector expects an RT-DETR / D-FINE-style ONNX export: two outputs named
arbitrarily, distinguished at run-time by shape — **logits** ``(1, N, C)``
pre-sigmoid and **boxes** ``(1, N, 4)`` cx/cy/w/h normalised ``[0, 1]``.

Tier selection
--------------
:func:`select_detector_tier` probes the local GStreamer plugin registry to
decide whether a DeepStream ``nvinfer`` element is available, then falls back
to the platform-detection logic from
:mod:`openral_runner.backends.gstreamer.pipeline`.  :func:`make_objects_detector`
dispatches the resolved tier:

* :attr:`DetectorTier.CPU_ONNX` → :class:`ObjectsDetector` (ONNXRuntime,
  system-memory BGR frames).
* :attr:`DetectorTier.NVMM_AGGREGATOR` →
  :class:`~openral_runner.backends.gstreamer.nvmm_detector.NvmmObjectsDetector`
  (clean-room zero-copy NVMM path, lazily imported so this module loads without
  pycuda / tensorrt).
* :attr:`DetectorTier.NVINFER` is the spike-gated DeepStream follow-up
  (ADR-0037 PR5b PR D) and raises a clear
  :exc:`~openral_core.exceptions.ROSConfigError`.

Lazy import
-----------
``onnxruntime`` is imported lazily inside :meth:`ObjectsDetector.__init__`
(mirroring the ``_import_ort`` pattern in
:mod:`openral_rskill.runtime_onnx`) so that ``import objects_detector``
succeeds on hosts where the wheel is not installed.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from openral_runner.backends.gstreamer.nvmm_detector import NvmmObjectsDetector
from openral_core import ObjectDetection2D, ObjectsMetadata
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

from openral_runner.backends.gstreamer.pipeline import (
    Platform,
    detect_platform,
    inspect_element_present,
)

__all__ = [
    "DetectorTier",
    "ObjectsDetector",
    "identify_rtdetr_outputs",
    "make_objects_detector",
    "postprocess_rtdetr",
    "select_detector_tier",
]

log = structlog.get_logger(__name__)


# ── Tier enum ──────────────────────────────────────────────────────────────────


class DetectorTier(str, Enum):
    """Execution tier for the ADR-0037 object detector.

    Attributes:
        CPU_ONNX: ONNXRuntime on system-memory BGR frames. Available on any
            host; implemented in this module.
        NVINFER: NVIDIA DeepStream ``nvinfer`` element. Present when the
            DeepStream GStreamer plugin registry is available on the host.
            **Not yet implemented** (ADR-0037 PR5b).
        NVMM_AGGREGATOR: Zero-copy NVMM aggregator for Jetson / Spark without
            DeepStream. **Not yet implemented** (ADR-0037 PR5b).
        VLM_SIDECAR: Out-of-process open-vocabulary VLM detector (e.g.
            LocateAnything-3B) reached over ZMQ. Selected for ``runtime:
            pytorch`` detector manifests; consumes the same system-memory BGR
            appsink branch as :attr:`CPU_ONNX` (ADR-0037 2026-06-09 amendment).
        ZEROSHOT_HF: In-process Transformers open-vocabulary detector
            (``AutoModelForZeroShotObjectDetection`` — e.g. OmDet-Turbo) run
            against a **fixed** class vocabulary, so it behaves as an unprompted
            large closed-vocabulary detector. Selected for manifests whose
            ``detector.engine`` is ``zeroshot_hf``; consumes the same
            system-memory BGR appsink branch as :attr:`CPU_ONNX` (ADR-0037
            2026-06-12 amendment).

    Example:
        >>> DetectorTier.CPU_ONNX.value
        'cpu_onnx'
        >>> DetectorTier.NVINFER.value
        'nvinfer'
        >>> DetectorTier.NVMM_AGGREGATOR.value
        'nvmm_aggregator'
        >>> DetectorTier.VLM_SIDECAR.value
        'vlm_sidecar'
        >>> DetectorTier.ZEROSHOT_HF.value
        'zeroshot_hf'
    """

    CPU_ONNX = "cpu_onnx"
    NVINFER = "nvinfer"
    NVMM_AGGREGATOR = "nvmm_aggregator"
    VLM_SIDECAR = "vlm_sidecar"
    ZEROSHOT_HF = "zeroshot_hf"


# ── Tier selection ─────────────────────────────────────────────────────────────


def select_detector_tier(platform: Platform | None = None) -> DetectorTier:
    """Probe the host and return the best available detector tier.

    Decision order (first match wins):

    1. If ``gst-inspect-1.0 nvinfer`` succeeds, DeepStream is present →
       :attr:`DetectorTier.NVINFER`.
    2. Else if the resolved platform is :attr:`Platform.TEGRA` →
       :attr:`DetectorTier.NVMM_AGGREGATOR` (NVMM without DeepStream).
    3. Otherwise → :attr:`DetectorTier.CPU_ONNX`.

    Note:
        The ``nvinfer`` probe always wins over any explicit ``platform``
        argument — if DeepStream is installed, ``NVINFER`` is returned even on
        ``Platform.CPU_ONLY``.  To force ``CPU_ONNX`` unconditionally, pass
        ``tier=DetectorTier.CPU_ONNX`` to :func:`make_objects_detector`.

    Args:
        platform: Override the platform detection.  ``None`` calls
            :func:`~openral_runner.backends.gstreamer.pipeline.detect_platform`.

    Returns:
        The selected :class:`DetectorTier`.

    Example:
        >>> # Safe on any host: DetectorTier.CPU_ONNX membership test
        >>> DetectorTier.CPU_ONNX in DetectorTier
        True
    """
    if inspect_element_present("nvinfer"):
        return DetectorTier.NVINFER
    resolved = platform if platform is not None else detect_platform()
    if resolved is Platform.TEGRA:
        return DetectorTier.NVMM_AGGREGATOR
    return DetectorTier.CPU_ONNX


# ── Lazy ORT import ────────────────────────────────────────────────────────────


def _import_ort() -> Any:  # noqa: ANN401  # reason: onnxruntime ships no stubs
    """Import ``onnxruntime`` lazily, with a helpful error if missing.

    Raises:
        ROSRuntimeError: If the ``onnxruntime`` wheel is not installed; the
            message points to the install command rather than the bare
            ``ModuleNotFoundError`` that callers would otherwise see.
    """
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]  # noqa: PLC0415  # reason: lazy import; onnxruntime ships no py.typed marker
    except ImportError as exc:
        raise ROSRuntimeError(
            "ObjectsDetector: 'onnxruntime' is not installed. "
            "Install with: uv add onnxruntime --package openral-runner"
        ) from exc
    return ort


# ── Pure-numpy nearest-neighbour resize ───────────────────────────────────────


def _nn_resize(arr: Any, out_h: int, out_w: int) -> Any:  # noqa: ANN401  # reason: numpy not typed
    """Nearest-neighbour image resize using only NumPy index arrays.

    Avoids a hard ``cv2`` dependency.  Quality is irrelevant for object
    detection pre-processing because the model is trained with the same
    bilinear-vs-nearest discrepancy tolerated at inference — NN resize is
    acceptable for CPU pre-processing budgets.

    Args:
        arr: HxWxC ``uint8`` array.
        out_h: Target height.
        out_w: Target width.

    Returns:
        ``out_h x out_w x C`` ``uint8`` array.
    """
    import numpy as np  # noqa: PLC0415  # reason: lazy — numpy not always available at import

    src_h, src_w = arr.shape[:2]
    row_idx = (np.arange(out_h) * src_h / out_h).astype(np.int32)
    col_idx = (np.arange(out_w) * src_w / out_w).astype(np.int32)
    return arr[np.ix_(row_idx, col_idx)]


# ── Shared RT-DETR decode helpers ─────────────────────────────────────────────


def identify_rtdetr_outputs(named_shapes: list[tuple[str, tuple[Any, ...]]]) -> tuple[str, str]:
    """Return ``(logits_name, boxes_name)`` from candidate output ``(name, shape)`` pairs.

    Among the 3-D outputs (``len(shape) == 3``): the one whose last dim is exactly
    ``4`` is boxes, the other is logits. If both (or neither) end in 4 — e.g.
    ``num_classes == 4`` — fall back to index order (0 = logits, 1 = boxes), the
    standard RT-DETR / D-FINE export convention.

    Args:
        named_shapes: ``(name, shape)`` for every model output. Non-3-D entries
            (e.g. an ``images`` passthrough) are ignored.

    Returns:
        ``(logits_name, boxes_name)``.

    Raises:
        ROSConfigError: If fewer than two 3-D outputs are present.

    Example:
        >>> identify_rtdetr_outputs([("l", (1, 300, 80)), ("b", (1, 300, 4))])
        ('l', 'b')
    """
    three_d = [(n, s) for (n, s) in named_shapes if s and len(s) == 3]  # noqa: PLR2004
    if len(three_d) < 2:  # noqa: PLR2004
        raise ROSConfigError(
            f"identify_rtdetr_outputs: expected >= 2 three-dimensional outputs "
            f"(logits + boxes); got {len(three_d)} among {named_shapes}."
        )
    last_dim_4 = [(n, s) for (n, s) in three_d if s[-1] == 4]  # noqa: PLR2004
    if len(last_dim_4) == 1:
        boxes_name = last_dim_4[0][0]
        logits_name = next(n for (n, s) in three_d if n != boxes_name)
    else:
        logits_name = three_d[0][0]
        boxes_name = three_d[1][0]
    return logits_name, boxes_name


def postprocess_rtdetr(
    logits: Any,  # noqa: ANN401  # reason: numpy not typed
    boxes: Any,  # noqa: ANN401  # reason: numpy not typed
    *,
    labels: list[str],
    model_id: str,
    sensor_id: str,
    score_threshold: float,
    frame_width: int,
    frame_height: int,
) -> ObjectsMetadata | None:
    """Decode RT-DETR / D-FINE raw outputs into :class:`ObjectsMetadata`.

    Tier-agnostic: CPU (ONNXRuntime), NVMM aggregator (TRT), and nvinfer
    (tensor-meta) all call this with already-identified ``logits`` / ``boxes``
    arrays, so the sigmoid → argmax → threshold → cxcywh→xyxy decode lives in one
    place (CLAUDE.md §13).

    Args:
        logits: ``(N, C)`` or ``(1, N, C)`` pre-sigmoid class scores.
        boxes: ``(N, 4)`` or ``(1, N, 4)`` normalised cxcywh in ``[0, 1]``.
        labels: Class-name list; index ``i`` ↔ class ``i``. Non-empty.
        model_id: Identifier embedded in the emitted metadata.
        sensor_id: Sensor name forwarded to the metadata.
        score_threshold: Minimum sigmoid score to keep a detection, in ``[0, 1]``.
        frame_width: Pixel width the normalised boxes scale to.
        frame_height: Pixel height the normalised boxes scale to.

    Returns:
        :class:`ObjectsMetadata` sorted by descending confidence, or ``None`` if
        no detection passes the threshold.

    Example:
        >>> import numpy as np
        >>> md = postprocess_rtdetr(
        ...     np.array([[[-9.0, 9.0]]], dtype=np.float32),
        ...     np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32),
        ...     labels=["bg", "car"],
        ...     model_id="m",
        ...     sensor_id="s",
        ...     score_threshold=0.5,
        ...     frame_width=100,
        ...     frame_height=100,
        ... )
        >>> md.detections[0].label
        'car'
    """
    import numpy as np  # noqa: PLC0415  # reason: lazy — numpy not always available at import

    logits2d = logits[0] if getattr(logits, "ndim", 2) == 3 else logits  # noqa: PLR2004
    boxes2d = boxes[0] if getattr(boxes, "ndim", 2) == 3 else boxes  # noqa: PLR2004
    scores = 1.0 / (1.0 + np.exp(-logits2d))  # (N, C)
    cls_ids = scores.argmax(axis=-1)  # (N,)
    query_scores = scores[np.arange(len(cls_ids)), cls_ids]  # (N,)

    dets: list[ObjectDetection2D] = []
    for i, (cls, score) in enumerate(zip(cls_ids, query_scores, strict=False)):
        if float(score) < score_threshold:
            continue
        cls_int = int(cls)
        if cls_int >= len(labels):
            log.warning(
                "objects_detector.label_out_of_range",
                cls_int=cls_int,
                num_labels=len(labels),
                model_id=model_id,
            )
            continue
        cx, cy, bw, bh = boxes2d[i]
        x0 = float(cx - bw / 2) * frame_width
        y0 = float(cy - bh / 2) * frame_height
        x1 = float(cx + bw / 2) * frame_width
        y1 = float(cy + bh / 2) * frame_height
        xi0 = max(0, min(frame_width, round(x0)))
        yi0 = max(0, min(frame_height, round(y0)))
        xi1 = max(0, min(frame_width, round(x1)))
        yi1 = max(0, min(frame_height, round(y1)))
        if xi1 <= xi0 or yi1 <= yi0:
            continue
        dets.append(
            ObjectDetection2D(
                label=labels[cls_int], confidence=float(score), bbox_xyxy=(xi0, yi0, xi1, yi1)
            )
        )
    if not dets:
        return None
    dets.sort(key=lambda d: d.confidence, reverse=True)
    return ObjectsMetadata(
        sensor_id=sensor_id,
        detections=dets,
        model_id=model_id,
        frame_width=frame_width,
        frame_height=frame_height,
    )


# ── Main detector class ────────────────────────────────────────────────────────


class ObjectsDetector:
    """CPU-tier object detector wrapping an RT-DETR / D-FINE ONNX model.

    Implements :class:`~openral_runner.backends.gstreamer.perception_tee.EventDetector`;
    plug it into
    :class:`~openral_runner.backends.gstreamer.perception_tee.PerceptionEventPublisher`
    to publish ``openral_msgs/PromptStamped`` on ``/openral/perception/objects``.

    The expected ONNX export signature is **two outputs** (in any order):

    * **logits** — shape ``(1, N, num_classes)`` — pre-sigmoid class scores.
    * **boxes** — shape ``(1, N, 4)`` — normalised cxcywh in ``[0, 1]``.

    The two outputs are identified at session-load time by shape: among all
    3-D outputs (ndim == 3), the output whose last dimension is exactly 4 is
    boxes; the other is logits.  When ``num_classes == 4`` both 3-D outputs
    have the same last dimension, so index order is used as a tiebreaker
    (index 0 = logits, index 1 = boxes), which matches the standard RT-DETR /
    D-FINE export convention.  Outputs with ndim != 3 (e.g. an ``images``
    passthrough) are ignored during identification.

    ``onnxruntime`` is lazy-imported at construction time so importing this
    module does not fail on hosts without the wheel.

    Args:
        onnx_path: Path to the ``*.onnx`` model file.
        labels: List of class-name strings.  Must be non-empty.  Index ``i``
            must correspond to class ``i`` in the model's output logits.
        model_id: Identifier embedded in every emitted :class:`ObjectsMetadata`
            (e.g. ``"rtdetr-coco-r18"``).
        input_size: ``(height, width)`` to resize frames to before inference.
            Default ``(640, 640)``.
        score_threshold: Minimum sigmoid score to keep a detection.
            Must be in ``[0, 1]``.  Default ``0.5``.
        device: PyTorch-style device string (``"cpu"`` or ``"cuda:N"``).
            Default ``"cpu"``.

    Raises:
        ROSConfigError: If the ONNX file does not exist, the score threshold is
            out of range, or the labels list is empty.
        ROSRuntimeError: If ``onnxruntime`` is not installed or fails to create
            the ``InferenceSession``.

    Example:
        >>> det = ObjectsDetector.__new__(ObjectsDetector)
        >>> det.kind
        Traceback (most recent call last):
            ...
        AttributeError: 'ObjectsDetector' object has no attribute 'kind'
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        labels: list[str],
        model_id: str,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Validate arguments, lazy-import ORT, create the InferenceSession."""
        p = Path(onnx_path)
        if not p.exists():
            raise ROSConfigError(f"ObjectsDetector: ONNX model file not found at '{p}'.")
        if not 0.0 <= score_threshold <= 1.0:
            raise ROSConfigError(
                f"ObjectsDetector: score_threshold must be in [0, 1]; got {score_threshold!r}."
            )
        if not labels:
            raise ROSConfigError("ObjectsDetector: labels list must be non-empty.")

        ort = _import_ort()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        try:
            # ``Any`` because onnxruntime ships no py.typed marker / stubs.
            self._session: Any = ort.InferenceSession(str(p), providers=providers)
        except Exception as exc:
            raise ROSRuntimeError(
                f"ObjectsDetector: failed to create InferenceSession for '{p}': {exc}"
            ) from exc

        self._input_name: str = self._session.get_inputs()[0].name

        outputs = self._session.get_outputs()
        self._logits_name, self._boxes_name = identify_rtdetr_outputs(
            [(o.name, tuple(o.shape)) for o in outputs]
        )

        self.kind: str = "objects"
        self._labels = labels
        self._model_id = model_id
        self._input_size = input_size  # (H, W)
        self._score_threshold = score_threshold
        self._device = device

        log.debug(
            "objects_detector.created",
            onnx_path=str(p),
            model_id=model_id,
            input_size=input_size,
            score_threshold=score_threshold,
            device=device,
        )

    # ── EventDetector protocol ─────────────────────────────────────────────────

    def detect(
        self,
        frame_bgr: bytes,
        width: int,
        height: int,
        sensor_id: str,
    ) -> ObjectsMetadata | None:
        """Run one detection pass on a BGR frame; return ``None`` if no objects pass threshold.

        Args:
            frame_bgr: Raw BGR bytes from the GStreamer appsink (``width * height * 3`` bytes).
            width: Frame width in pixels.
            height: Frame height in pixels.
            sensor_id: Sensor name forwarded to the emitted :class:`ObjectsMetadata`.

        Returns:
            :class:`~openral_core.ObjectsMetadata` with detections sorted by
            descending confidence, or ``None`` if zero detections survive the
            score threshold.
        """
        import numpy as np  # noqa: PLC0415  # reason: lazy — see module docstring

        try:
            arr = np.frombuffer(frame_bgr, dtype=np.uint8).reshape(height, width, 3)
        except ValueError:
            # Frame byte length doesn't match width*height*3 — caps mismatch upstream; drop.
            return None

        # Pre-process: BGR→RGB, nearest-neighbour resize, float32 /255, HWC→NCHW.
        rgb = arr[..., ::-1]  # BGR → RGB
        in_h, in_w = self._input_size
        if (height, width) != (in_h, in_w):
            rgb = _nn_resize(rgb, in_h, in_w)
        blob = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0  # HWC float32
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW

        # Inference.
        raw_outputs = self._session.run(None, {self._input_name: blob})
        # Map output names → arrays.
        output_names = [o.name for o in self._session.get_outputs()]
        out_map: dict[str, Any] = dict(zip(output_names, raw_outputs, strict=False))
        logits_arr = out_map[self._logits_name]  # (1, N, C)
        boxes_arr = out_map[self._boxes_name]  # (1, N, 4)

        # Post-process.
        return postprocess_rtdetr(
            logits_arr,
            boxes_arr,
            labels=self._labels,
            model_id=self._model_id,
            sensor_id=sensor_id,
            score_threshold=self._score_threshold,
            frame_width=width,
            frame_height=height,
        )

    def summarise(self, metadata: object) -> str:
        """Return a human-readable summary of an :class:`ObjectsMetadata` event.

        Args:
            metadata: Must be an :class:`~openral_core.ObjectsMetadata` instance.

        Returns:
            A one-line string such as ``"objects: 1x car, 1x person on head_rgb"``.

        Raises:
            TypeError: If ``metadata`` is not an :class:`ObjectsMetadata`.

        Example:
            >>> from openral_core import ObjectDetection2D, ObjectsMetadata
            >>> md = ObjectsMetadata(
            ...     sensor_id="head_rgb",
            ...     detections=[
            ...         ObjectDetection2D(label="car", confidence=0.9, bbox_xyxy=(0, 0, 10, 10))
            ...     ],
            ...     model_id="rtdetr-test",
            ...     frame_width=640,
            ...     frame_height=480,
            ... )
            >>> md.kind
            'objects'
        """
        if not isinstance(metadata, ObjectsMetadata):
            raise TypeError(
                "ObjectsDetector.summarise: expected ObjectsMetadata, "
                f"got {type(metadata).__name__!r}"
            )
        if not metadata.detections:
            return f"objects: none on {metadata.sensor_id}"
        counts = Counter(d.label for d in metadata.detections)
        parts = ", ".join(f"{n}x {label}" for label, n in sorted(counts.items()))
        return f"objects: {parts} on {metadata.sensor_id}"


# ── Factory ────────────────────────────────────────────────────────────────────


def make_objects_detector(
    onnx_path: str | Path,
    *,
    labels: list[str],
    model_id: str,
    tier: DetectorTier | None = None,
    **kwargs: Any,  # noqa: ANN401  # reason: forwarded to ObjectsDetector / NvmmObjectsDetector constructor
) -> ObjectsDetector | NvmmObjectsDetector:
    """Create an object detector for the resolved or requested tier.

    Args:
        onnx_path: Path to the ``*.onnx`` model file.
        labels: Class-name list (must match model output class count).
        model_id: Detector identifier embedded in emitted metadata.
        tier: Explicit tier override.  ``None`` calls
            :func:`select_detector_tier` to auto-select.  Pass
            ``DetectorTier.CPU_ONNX`` to force the CPU path regardless of
            platform.
        **kwargs: Extra keyword arguments forwarded to :class:`ObjectsDetector`
            (``input_size``, ``score_threshold``, ``device``) or
            :class:`~openral_runner.backends.gstreamer.nvmm_detector.NvmmObjectsDetector`
            (``input_size``, ``score_threshold``, ``device_index``,
            ``quantization``).

    Returns:
        An :class:`ObjectsDetector` for :attr:`DetectorTier.CPU_ONNX`, or a
        :class:`~openral_runner.backends.gstreamer.nvmm_detector.NvmmObjectsDetector`
        for :attr:`DetectorTier.NVMM_AGGREGATOR`.

    Raises:
        ROSConfigError: For :attr:`DetectorTier.NVINFER` — the DeepStream
            ``nvinfer`` tier is spike-gated (ADR-0037 PR5b PR D); pass
            ``tier=DetectorTier.NVMM_AGGREGATOR`` for the clean-room zero-copy
            path or ``tier=DetectorTier.CPU_ONNX`` for the CPU path.
        ROSConfigError: For any unrecognised tier value.

    Example:
        >>> # Membership tests are deterministic on any host:
        >>> DetectorTier.CPU_ONNX in DetectorTier
        True
    """
    if tier is None:
        tier = select_detector_tier()
    if tier is DetectorTier.CPU_ONNX:
        return ObjectsDetector(onnx_path, labels=labels, model_id=model_id, **kwargs)
    if tier is DetectorTier.NVMM_AGGREGATOR:
        from openral_runner.backends.gstreamer.nvmm_detector import (  # noqa: PLC0415
            NvmmObjectsDetector,
        )

        return NvmmObjectsDetector(onnx_path, labels=labels, model_id=model_id, **kwargs)
    if tier is DetectorTier.NVINFER:
        raise ROSConfigError(
            "ObjectsDetector: the 'nvinfer' tier is the spike-gated follow-up "
            "(ADR-0037 PR5b PR D). Pass tier=DetectorTier.NVMM_AGGREGATOR for the "
            "clean-room zero-copy path, or tier=DetectorTier.CPU_ONNX."
        )
    raise ROSConfigError(f"ObjectsDetector: unknown tier {tier!r}.")
