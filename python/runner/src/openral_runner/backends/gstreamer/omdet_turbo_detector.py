"""In-process open-vocabulary detector backed by a Transformers zero-shot model.

Unlike the LocateAnything sidecar (a heavy VLM pinned to ``transformers==4.57.1``,
hence out-of-process — ADR-0037 2026-06-09 amendment), a zero-shot detection
model such as ``omlab/omdet-turbo-swin-tiny-hf`` is a first-class
``transformers`` architecture (``AutoModelForZeroShotObjectDetection``) that
loads under the runtime's own ``transformers>=5``. It therefore runs **in
process** — no sidecar venv, no ZMQ — and is selected as
:attr:`~openral_runner.backends.gstreamer.objects_detector.DetectorTier.ZEROSHOT_HF`
for manifests whose ``detector.engine`` is ``zeroshot_hf``
(ADR-0037 2026-06-12 amendment).

**Both detector modes (ADR-0051), chosen by the rSkill manifest.** OmDet-Turbo
is open-vocabulary, so this one backend serves either invocation mode; which one
is intended is declared by the manifest's ``detector.mode``:

* ``continuous`` (e.g. ``rskills/omdet-turbo-indoor``) — the manifest's
  ``detector.labels`` is a **fixed** vocabulary detected every frame. Running a
  frozen large vocabulary makes it behave like a closed large-vocabulary
  detector — an unprompted background producer that populates the world object
  list with far more than the 80 COCO classes the RT-DETR rSkills cover.
* ``on_demand`` (e.g. ``rskills/omdet-turbo-locator``) — a prompted locator. The
  reasoner retargets it via :meth:`set_query` (the
  ``/openral/perception/detector_query`` topic) or asks one-shot via
  :meth:`detect_with_query` (the read-only ``locate_in_view`` service, ADR-0043).
  A lightweight, real-time alternative to the 3B LocateAnything VLM for simple
  "find X" queries.

``labels`` is the static default vocabulary either way.

It implements the same ``detect(frame_bgr, width, height, sensor_id) ->
ObjectsMetadata | None`` interface as
:class:`~openral_runner.backends.gstreamer.objects_detector.ObjectsDetector`,
so :class:`~openral_runner.backends.gstreamer.detector_runner.DetectorRunner`
drives it from the same system-memory BGR camera-tee branch as the CPU ONNX and
VLM-sidecar tiers.

``torch`` / ``transformers`` are imported lazily at first ``detect()`` so this
module imports cleanly on hosts without the wheels (mirroring the ONNXRuntime
lazy-import in :mod:`~openral_runner.backends.gstreamer.objects_detector`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openral_core import ObjectDetection2D, ObjectsMetadata
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

__all__ = [
    "OmDetTurboDetector",
    "build_objects_metadata_from_results",
    "query_to_classes",
]

# Degenerate-box guard: drop detections covering essentially the whole frame
# (a near-full-image box is never a useful manipulation target and is usually a
# background mis-fire). Mirrors the LocateAnything ``_MAX_AREA_FRAC`` intent.
_MAX_AREA_FRAC = 0.98


def build_objects_metadata_from_results(
    *,
    labels: Sequence[str],
    scores: Sequence[float],
    boxes_xyxy: Sequence[tuple[float, float, float, float]],
    width: int,
    height: int,
    model_id: str,
    sensor_id: str,
    score_threshold: float,
) -> ObjectsMetadata | None:
    """Build :class:`ObjectsMetadata` from decoded zero-shot detection results.

    Pure (no torch / transformers) so the conversion is unit-testable without a
    GPU or model load: callers pass already-decoded Python-native results (the
    post-processor's per-image ``text_labels`` / ``scores`` / pixel-space
    ``boxes``). Detections below ``score_threshold`` or with a degenerate /
    near-full-image box are dropped; survivors are clipped to frame bounds and
    sorted by descending confidence.

    Args:
        labels: Per-detection class label, parallel to ``scores`` / ``boxes_xyxy``.
        scores: Per-detection confidence in ``[0, 1]``.
        boxes_xyxy: Per-detection ``(x1, y1, x2, y2)`` in **pixel** coordinates.
        width: Frame width in pixels (clip bound).
        height: Frame height in pixels (clip bound).
        model_id: Identifier embedded in the emitted metadata.
        sensor_id: Sensor name forwarded to the metadata.
        score_threshold: Minimum confidence to keep a detection.

    Returns:
        :class:`ObjectsMetadata` sorted by descending confidence, or ``None`` if
        no detection survives.

    Example:
        >>> md = build_objects_metadata_from_results(
        ...     labels=["mug", "drawer"],
        ...     scores=[0.9, 0.2],
        ...     boxes_xyxy=[(10.0, 10.0, 50.0, 60.0), (0.0, 0.0, 5.0, 5.0)],
        ...     width=640,
        ...     height=480,
        ...     model_id="omdet-turbo-indoor",
        ...     sensor_id="front_cam",
        ...     score_threshold=0.3,
        ... )
        >>> [d.label for d in md.detections]
        ['mug']
    """
    if not (len(labels) == len(scores) == len(boxes_xyxy)):
        raise ROSConfigError(
            "build_objects_metadata_from_results: labels/scores/boxes length "
            f"mismatch ({len(labels)}/{len(scores)}/{len(boxes_xyxy)})."
        )
    frame_area = float(width * height) or 1.0
    dets: list[ObjectDetection2D] = []
    for label, score, (x1, y1, x2, y2) in zip(labels, scores, boxes_xyxy, strict=True):
        if float(score) < score_threshold:
            continue
        lo_x, hi_x = sorted((x1, x2))
        lo_y, hi_y = sorted((y1, y2))
        xi0 = max(0, min(width, round(lo_x)))
        yi0 = max(0, min(height, round(lo_y)))
        xi1 = max(0, min(width, round(hi_x)))
        yi1 = max(0, min(height, round(hi_y)))
        if xi1 <= xi0 or yi1 <= yi0:
            continue
        if ((xi1 - xi0) * (yi1 - yi0)) / frame_area >= _MAX_AREA_FRAC:
            continue
        dets.append(
            ObjectDetection2D(
                label=str(label), confidence=float(score), bbox_xyxy=(xi0, yi0, xi1, yi1)
            )
        )
    if not dets:
        return None
    dets.sort(key=lambda d: d.confidence, reverse=True)
    return ObjectsMetadata(
        sensor_id=sensor_id,
        detections=dets,
        model_id=model_id,
        frame_width=width,
        frame_height=height,
    )


def query_to_classes(query: str) -> list[str]:
    """Parse a free-text detector query into the OmDet class list.

    On-demand callers (the ``locate_in_view`` service, the
    ``/openral/perception/detector_query`` topic) send free text. OmDet-Turbo is
    a multi-label detector that takes a *list of classes*, so a multi-object
    query is split on commas or the ``</c>`` separator (the joiner
    ``LocateAnythingDetector`` uses), and a single phrase becomes a one-element
    list. Whitespace-only fragments are dropped.

    Args:
        query: Free-text query, e.g. ``"red mug"`` or ``"stapler, scissors"``.

    Returns:
        Non-empty class list.

    Raises:
        ROSConfigError: If ``query`` yields no non-empty class.

    Example:
        >>> query_to_classes("stapler, scissors")
        ['stapler', 'scissors']
        >>> query_to_classes("the red mug")
        ['the red mug']
    """
    parts = [c.strip() for chunk in query.split("</c>") for c in chunk.split(",")]
    classes = [c for c in parts if c]
    if not classes:
        raise ROSConfigError("OmDetTurboDetector: detection query must be non-empty")
    return classes


class OmDetTurboDetector:
    """In-process Transformers zero-shot detector over a fixed class vocabulary.

    Loads ``AutoProcessor`` + ``AutoModelForZeroShotObjectDetection`` from
    ``weights_source`` on first :meth:`detect`, moves the model to CUDA when
    available (else CPU), and runs the **fixed** ``labels`` vocabulary against
    every frame. The model + processor load is deferred so construction is cheap
    and side-effect-free — the dispatch path and unit tests can build the
    backend without the wheels or a GPU.

    Args:
        labels: Fixed class vocabulary to detect every frame (non-empty).
        model_id: Identifier embedded in every emitted :class:`ObjectsMetadata`.
        weights_source: HF repo id to load (e.g. ``omlab/omdet-turbo-swin-tiny-hf``).
        score_threshold: Minimum confidence to keep a detection.
        nms_threshold: IoU threshold for the post-processor's class-agnostic NMS.
        device: ``"auto"`` picks ``cuda`` when available else ``cpu``; or pass an
            explicit ``"cpu"`` / ``"cuda"`` / ``"cuda:N"``.

    Raises:
        ROSConfigError: If ``labels`` is empty.
    """

    def __init__(
        self,
        *,
        labels: list[str],
        model_id: str,
        weights_source: str,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        """Store config; the model/processor load is deferred to first detect()."""
        if not labels:
            raise ROSConfigError("OmDetTurboDetector requires at least one label")
        self.kind = "objects"
        self._labels = list(labels)
        self._model_id = model_id
        self._weights_source = weights_source
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._device_pref = device
        # Lazily populated on first detect(). ``Any`` because transformers/torch
        # ship partial / heavy stubs not worth importing under strict here.
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device: str = "cpu"

    def _ensure_ready(self) -> None:
        """Load torch + transformers and the model/processor on first use."""
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415 — lazy: keep torch off the module import path
            from transformers import (  # noqa: PLC0415
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError as exc:  # pragma: no cover — env-provisioning guard
            raise ROSRuntimeError(
                "OmDetTurboDetector needs 'torch' + 'transformers'. Install with: "
                "uv sync --group omdet (provides torch + transformers)."
            ) from exc

        self._torch = torch
        if self._device_pref == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._device_pref
        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]  # reason: transformers AutoProcessor.from_pretrained is unstubbed in 5.3.0
            self._weights_source
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self._weights_source)
        self._fit_timm_backbone_to_processor(model)
        self._model = model.to(self._device).eval()

    def _fit_timm_backbone_to_processor(self, model: Any) -> None:  # noqa: ANN401  # reason: transformers PreTrainedModel is dynamically typed
        """Rebuild a fixed-resolution timm backbone at the processor's input size.

        transformers ≥5.3 instantiates OmDet-Turbo's Swin via a timm backbone
        pinned to its pretrained 224² (window-attention masks precomputed for
        224), so the processor's native 640² input crashes with
        ``AssertionError: Input height (640) doesn't match model (224)``.
        Rebuild the timm backbone at the processor's size and transplant the
        weights — the window-attention params are resolution-independent (0
        missing / 0 unexpected), so the model runs in-process at full quality.
        No-op when the backbone is not a fixed-size timm features model or its
        patch-embed size already matches the processor.
        """
        try:
            import timm  # noqa: PLC0415 — optional; only timm-backbone models need this
        except ImportError:
            return
        bc = getattr(model.config, "backbone_config", None)
        backbone_name = getattr(bc, "backbone", None) if bc is not None else None
        if not backbone_name:
            return
        try:
            wrap = model.get_submodule("vision_backbone.vision_backbone")
            bb = wrap._backbone
            patch_embed = getattr(bb, "patch_embed", None)
        except AttributeError:
            return
        proc = getattr(self._processor, "image_processor", self._processor)
        size = getattr(proc, "size", None) or {}
        if patch_embed is None or "height" not in size or "width" not in size:
            return
        target = (int(size["height"]), int(size["width"]))
        if tuple(patch_embed.img_size) == target:
            return  # already the right size
        out_indices = tuple(getattr(bc, "out_indices", None) or (1, 2, 3))
        rebuilt = timm.create_model(
            backbone_name,
            pretrained=False,
            img_size=target,
            features_only=True,
            out_indices=out_indices,
        )
        rebuilt.load_state_dict(bb.state_dict(), strict=False)
        wrap._backbone = rebuilt  # moved to device by the caller's model.to()

    def set_query(self, text: str) -> None:
        """Override the persistent class vocabulary at runtime (on-demand hook).

        For ``mode: on_demand`` detectors (ADR-0051): the
        ``/openral/perception/detector_query`` topic retargets the continuous
        leg by replacing the class list. Parsed via :func:`query_to_classes`.

        Raises:
            ROSConfigError: If ``text`` yields no non-empty class.
        """
        self._labels = query_to_classes(text)

    def detect(
        self, frame_bgr: bytes, width: int, height: int, sensor_id: str
    ) -> ObjectsMetadata | None:
        """Run one detection pass over the current vocabulary on a raw BGR frame.

        Args:
            frame_bgr: Raw BGR bytes from the GStreamer appsink (``w*h*3`` bytes).
            width: Frame width in pixels.
            height: Frame height in pixels.
            sensor_id: Sensor name forwarded to the emitted metadata.

        Returns:
            :class:`ObjectsMetadata` with detections sorted by descending
            confidence, or ``None`` if no detection survives the threshold.
        """
        return self._detect_classes(frame_bgr, width, height, sensor_id, self._labels)

    def detect_with_query(
        self, frame_bgr: bytes, width: int, height: int, sensor_id: str, query: str
    ) -> ObjectsMetadata | None:
        """One-shot detect for ``query`` without mutating the persistent vocabulary.

        Backs the read-only ``locate_in_view`` service (ADR-0043): a reasoner
        query ("is X in view right now?") must not change what the continuous
        leg detects. ``query`` is parsed via :func:`query_to_classes`.

        Raises:
            ROSConfigError: If ``query`` yields no non-empty class.
        """
        return self._detect_classes(frame_bgr, width, height, sensor_id, query_to_classes(query))

    def _detect_classes(
        self,
        frame_bgr: bytes,
        width: int,
        height: int,
        sensor_id: str,
        classes: list[str],
    ) -> ObjectsMetadata | None:
        """Run OmDet over ``classes`` on one BGR frame; shared by detect()/detect_with_query()."""
        import numpy as np  # noqa: PLC0415 — lazy: keep numpy off the import path
        from PIL import Image  # noqa: PLC0415

        self._ensure_ready()
        try:
            arr = np.frombuffer(frame_bgr, dtype=np.uint8).reshape(height, width, 3)
        except ValueError:
            # Byte length doesn't match width*height*3 — caps mismatch upstream; drop.
            return None
        image = Image.fromarray(arr[:, :, ::-1], "RGB")  # BGR -> RGB

        inputs = self._processor(images=image, text=classes, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs,
            text_labels=classes,
            target_sizes=[(height, width)],
            threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
        )[0]

        return build_objects_metadata_from_results(
            labels=_result_labels(result),
            scores=[float(s) for s in result["scores"].tolist()],
            boxes_xyxy=[tuple(box) for box in result["boxes"].tolist()],
            width=width,
            height=height,
            model_id=self._model_id,
            sensor_id=sensor_id,
            score_threshold=self._score_threshold,
        )

    def close(self) -> None:
        """Release the model and free CUDA memory if we loaded onto the GPU."""
        had_cuda = self._model is not None and self._device.startswith("cuda")
        self._model = None
        self._processor = None
        if had_cuda and self._torch is not None:  # best-effort VRAM reclaim
            self._torch.cuda.empty_cache()


def _result_labels(result: dict[str, Any]) -> list[str]:
    """Extract per-detection labels across transformers post-processor versions.

    Newer ``post_process_grounded_object_detection`` returns ``text_labels``;
    older releases returned ``classes`` / ``labels``. Tolerate all three so the
    backend is not pinned to one transformers point release.
    """
    for key in ("text_labels", "classes", "labels"):
        if key in result:
            return [str(label) for label in result[key]]
    raise ROSRuntimeError(
        "OmDetTurboDetector: post-processor result has no label field "
        f"(expected one of text_labels/classes/labels); got keys {sorted(result)}."
    )
