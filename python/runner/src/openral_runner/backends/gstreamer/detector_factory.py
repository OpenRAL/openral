"""Manifest → detector-backend dispatch for ``kind: detector`` rSkills (ADR-0037).

A GStreamer-free seam (no ``gi`` import) so the runtime↔backend selection is
unit-testable without a live pipeline. :class:`~.detector_runner.DetectorRunner`
delegates construction here.

Dispatch keys on the manifest's ``detector.engine`` first (when set), then
falls back to the legacy ``runtime``-based selection:

* ``engine: zeroshot_hf`` → the in-process Transformers open-vocabulary detector
  (:class:`~.omdet_turbo_detector.OmDetTurboDetector`,
  :attr:`~.objects_detector.DetectorTier.ZEROSHOT_HF`) run over a fixed
  vocabulary. No ``onnx_path``; the model loads under the runtime's own
  ``transformers`` (ADR-0037 2026-06-12 amendment).
* ``runtime: onnx`` / ``tensorrt`` (or ``engine: rtdetr_onnx``) → an ONNX-backed
  detector (:class:`~.objects_detector.ObjectsDetector` CPU tier or
  :class:`~.nvmm_detector.NvmmObjectsDetector` NVMM tier), built from the
  caller-resolved ``onnx_path``.
* ``runtime: pytorch`` (or ``engine: vlm_sidecar``) → the out-of-process
  open-vocabulary VLM detector
  (:class:`~.locateanything_detector.LocateAnythingDetector`,
  :attr:`~.objects_detector.DetectorTier.VLM_SIDECAR`). No ``onnx_path`` is
  needed; the model runs in an isolated sidecar (ADR-0037 2026-06-09 amendment).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openral_core.exceptions import ROSConfigError
from openral_core.schemas import DetectorEngine, DetectorMode, RSkillManifest, RSkillRuntime

from openral_runner.backends.gstreamer.objects_detector import (
    DetectorTier,
    make_objects_detector,
    select_detector_tier,
)

__all__ = [
    "DetectorNodeWiring",
    "build_manifest_detector",
    "detector_node_wiring",
    "weights_source_from_manifest",
]


@dataclass(frozen=True)
class DetectorNodeWiring:
    """How the perception detector node wires a detector, given its mode (ADR-0051).

    A GStreamer/rclpy-free policy object so the node's wiring decision is
    unit-testable without ROS; ``RosImageObjectDetectorNode`` consumes it.

    Attributes:
        run_continuous_leg: The primary camera runs the continuous detect+publish
            leg (streams ``ObjectsMetadata`` into world state). ``True`` for
            ``continuous`` background producers.
        serve_on_demand: Expose the read-only ``locate_in_view`` service and
            subscribe the ``detector_query`` retarget topic. ``True`` for
            ``on_demand`` prompted locators.
    """

    run_continuous_leg: bool
    serve_on_demand: bool


def detector_node_wiring(mode: DetectorMode) -> DetectorNodeWiring:
    """Map a detector's invocation mode to its node wiring (ADR-0051).

    The two modes are mutually exclusive at the node:

    * ``continuous`` → a background producer: run the publish leg, do **not**
      serve queries (no ``locate_in_view`` service, no ``detector_query`` topic).
    * ``on_demand`` → a prompted locator: serve queries, do **not** publish
      continuously (frames are still cached so the service can answer about the
      current view).

    Args:
        mode: The manifest's :attr:`DetectorContract.mode`.

    Returns:
        The :class:`DetectorNodeWiring` policy for ``mode``.

    Example:
        >>> detector_node_wiring(DetectorMode.CONTINUOUS).run_continuous_leg
        True
        >>> detector_node_wiring(DetectorMode.ON_DEMAND).serve_on_demand
        True
    """
    if mode is DetectorMode.ON_DEMAND:
        return DetectorNodeWiring(run_continuous_leg=False, serve_on_demand=True)
    return DetectorNodeWiring(run_continuous_leg=True, serve_on_demand=False)


def weights_source_from_manifest(manifest: RSkillManifest) -> str:
    """Resolve the HF repo the model backend should load.

    Used by both the VLM-sidecar (``LocateAnythingDetector``) and the in-process
    zero-shot (``OmDetTurboDetector``) backends. Prefers ``source_repo`` (the
    upstream weights the rSkill wraps), falling back to ``weights_uri``. Strips
    the ``hf://`` scheme and any ``@revision`` suffix so the value is a bare
    ``org/name`` repo id for ``from_pretrained``.
    """
    raw = manifest.source_repo or manifest.weights_uri or "nvidia/LocateAnything-3B"
    return raw.removeprefix("hf://").split("@", 1)[0]


def build_manifest_detector(
    manifest: RSkillManifest,
    *,
    onnx_path: str | Path | None = None,
    tier: DetectorTier | None = None,
) -> tuple[Any, DetectorTier]:
    """Build the detector backend for a ``kind: detector`` manifest.

    Args:
        manifest: A validated manifest with ``kind == "detector"``.
        onnx_path: Caller-resolved ONNX weights path. Required for ``onnx`` /
            ``tensorrt`` runtimes; ignored for ``pytorch`` (VLM sidecar).
        tier: Explicit ONNX-tier override; ``None`` auto-selects. Ignored for
            the ``pytorch`` runtime.

    Returns:
        ``(detector, tier)`` — the backend instance and its resolved tier. The
        backend exposes ``detect(frame_bgr, width, height, sensor_id)``.

    Raises:
        ROSConfigError: If the manifest is not a detector, or an ONNX runtime is
            requested without an ``onnx_path``.
    """
    if manifest.kind != "detector" or manifest.detector is None:
        raise ROSConfigError(
            f"build_manifest_detector: manifest {manifest.name!r} is not a "
            f"kind:detector with a detector block (kind={manifest.kind!r})."
        )
    contract = manifest.detector

    # Explicit engine selector wins (disambiguates backends that share a
    # runtime: an in-process zero-shot detector is `runtime: pytorch` too).
    if contract.engine is DetectorEngine.ZEROSHOT_HF:
        # Lazy import: keeps torch/transformers off the path for ONNX-only runners.
        from openral_runner.backends.gstreamer.omdet_turbo_detector import (  # noqa: PLC0415
            OmDetTurboDetector,
        )

        zs = OmDetTurboDetector(
            labels=contract.labels,
            model_id=manifest.name,
            weights_source=weights_source_from_manifest(manifest),
            score_threshold=contract.score_threshold,
        )
        return zs, DetectorTier.ZEROSHOT_HF

    if manifest.runtime is RSkillRuntime.PYTORCH:
        # Lazy import: keeps zmq/numpy/PIL off the path for ONNX-only runners.
        from openral_runner.backends.gstreamer.locateanything_detector import (  # noqa: PLC0415
            LocateAnythingDetector,
        )

        vlm = LocateAnythingDetector(
            labels=contract.labels,
            model_id=manifest.name,
            weights_source=weights_source_from_manifest(manifest),
        )
        return vlm, DetectorTier.VLM_SIDECAR

    if onnx_path is None:
        raise ROSConfigError(
            f"build_manifest_detector: runtime {manifest.runtime.value!r} for "
            f"{manifest.name!r} needs a resolved onnx_path."
        )
    # DetectorContract.input_size is (width, height); the ONNX detector classes
    # take (net_h, net_w).
    net_w, net_h = contract.input_size
    resolved_tier = tier if tier is not None else select_detector_tier()
    onnx_detector = make_objects_detector(
        onnx_path,
        labels=contract.labels,
        model_id=manifest.name,
        tier=resolved_tier,
        input_size=(net_h, net_w),
        score_threshold=contract.score_threshold,
    )
    return onnx_detector, resolved_tier
