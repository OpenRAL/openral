"""Runtime glue (ADR-0037) that runs a ``kind: detector`` rSkill against a live pipeline.

Loads the :class:`~openral_core.schemas.DetectorContract` from the manifest, builds an
:class:`~openral_runner.backends.gstreamer.objects_detector.ObjectsDetector` (CPU tier)
or :class:`~openral_runner.backends.gstreamer.nvmm_detector.NvmmObjectsDetector` (NVMM
aggregator tier), attaches the appropriate branch to the bus tee via
:class:`~openral_runner.backends.gstreamer.tee_manager.TeeManager`,
and on each frame publishes the resulting :class:`~openral_core.ObjectsMetadata` to a
caller-supplied sink callback.

The branch is chosen by tier:

* :attr:`~objects_detector.DetectorTier.CPU_ONNX` → ``videoconvert !
  video/x-raw,format=BGR ! appsink`` — system-memory CPU path.
* :attr:`~objects_detector.DetectorTier.NVMM_AGGREGATOR` → ``<conv> !
  video/x-raw(memory:NVMM),format=RGBA ! appsink`` — zero-copy NVMM path, where
  ``<conv>`` is the platform's NVMM converter (``nvvideoconvert`` on DeepStream /
  ``nvvidconv`` on Tegra), resolved by
  :func:`~openral_runner.backends.gstreamer.pipeline.nvmm_convert_element`.

Importing this module requires the ``gstreamer`` optional-extra (``gi`` + GStreamer 1.0).
``gi`` is eagerly imported at load time — mirrors :mod:`tee_manager`'s eager ``gi`` +
``Gst.init`` so GStreamer process state is initialised before any later ``import rclpy``
in the same interpreter.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import gi
import structlog

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402  # gi requires a version-pin before import
from openral_core.exceptions import ROSConfigError, ROSRuntimeError  # noqa: E402
from openral_core.schemas import RSkillManifest  # noqa: E402

from openral_runner.backends.gstreamer.detector_factory import (  # noqa: E402
    build_manifest_detector,
)
from openral_runner.backends.gstreamer.objects_detector import DetectorTier  # noqa: E402
from openral_runner.backends.gstreamer.pipeline import (  # noqa: E402
    TEE_NAME,
    nvmm_convert_element,
)
from openral_runner.backends.gstreamer.tee_manager import BranchHandle, TeeManager  # noqa: E402

# Idempotent; mirrors tee_manager.py's eager init.
Gst.init(None)

log = structlog.get_logger(__name__)

__all__ = ["DetectorRunner"]


class DetectorRunner:
    """Runtime glue (ADR-0037) that wires a ``kind: detector`` rSkill to a live pipeline.

    Validates the manifest, builds an :class:`ObjectsDetector` (CPU tier) or
    :class:`~openral_runner.backends.gstreamer.nvmm_detector.NvmmObjectsDetector`
    (NVMM aggregator tier), attaches the appropriate branch to the named bus tee
    via :class:`TeeManager`, and fires ``on_detection`` for every non-``None``
    detection result on the GStreamer streaming thread.

    In production ``on_detection`` publishes to ROS; in tests it appends to a list.

    Args:
        pipeline: A running ``Gst.Pipeline`` containing the named tee.
        manifest: A validated :class:`~openral_core.schemas.RSkillManifest` with
            ``kind == "detector"`` and a non-``None`` ``.detector`` block.
        onnx_path: Resolved ONNX weights path.  Weights resolution from
            ``weights_uri`` is the loader's responsibility — caller supplies the
            ready path.
        sensor_id: Sensor / camera identifier embedded in emitted
            :class:`~openral_core.ObjectsMetadata`.
        on_detection: Callback invoked (on the streaming thread) for each
            non-``None`` :class:`~openral_core.ObjectsMetadata`.  In production
            this publishes to ROS; in tests pass ``collected.append``.
        tee_name: Name of the bus tee inside *pipeline*.  Defaults to
            :data:`~openral_runner.backends.gstreamer.pipeline.TEE_NAME`.
        tier: Execution tier override.  ``None`` calls
            :func:`~openral_runner.backends.gstreamer.objects_detector.select_detector_tier`
            to auto-select.  Pass ``DetectorTier.CPU_ONNX`` to force the CPU path.

    Raises:
        ROSConfigError: If ``manifest.kind != "detector"`` or ``manifest.detector``
            is ``None``.
        ROSConfigError: If the resolved tier is unsupported (e.g. ``NVINFER``).

    Example:
        >>> # Full live exercise in tests/unit/test_detector_runner_e2e.py
        >>> import contextlib
        >>> with contextlib.suppress(ImportError):
        ...     from openral_runner.backends.gstreamer.detector_runner import DetectorRunner
    """

    def __init__(
        self,
        pipeline: Any,  # noqa: ANN401  # reason: Gst.Pipeline — duck-typed to avoid gi type dependency at the signature
        manifest: RSkillManifest,
        *,
        onnx_path: str | Any | None = None,  # noqa: ANN401  # reason: str | pathlib.Path — caller-resolved; None for the VLM sidecar tier
        sensor_id: str,
        on_detection: Callable[[Any], None],
        tee_name: str = TEE_NAME,
        tier: DetectorTier | None = None,
    ) -> None:
        """Validate manifest, build detector, and prepare for ``start()``."""
        if manifest.kind != "detector":
            raise ROSConfigError(
                f"DetectorRunner: manifest {manifest.name!r} has kind={manifest.kind!r}; "
                "expected kind='detector'. Pass a kind:detector rSkill manifest."
            )
        if manifest.detector is None:
            raise ROSConfigError(
                f"DetectorRunner: manifest {manifest.name!r} has kind='detector' but "
                "no 'detector' block. The manifest is malformed."
            )

        contract = manifest.detector
        # DetectorContract.input_size is (width, height); the NVMM caps need
        # explicit width/height, so convert exactly once here. (The VLM sidecar
        # tier ignores it — LocateAnything resizes dynamically.)
        net_w, net_h = contract.input_size
        self._net_w, self._net_h = net_w, net_h

        # Dispatch onnx/tensorrt vs pytorch(VLM-sidecar) in a gi-free seam so it
        # stays unit-testable; both backends expose the same detect() interface.
        self._detector, self._tier = build_manifest_detector(
            manifest, onnx_path=onnx_path, tier=tier
        )

        self._pipeline = pipeline
        self._tee = TeeManager(pipeline, tee_name=tee_name)
        self._sensor_id = sensor_id
        self._on_detection = on_detection
        self._sink_name = f"{sensor_id}_det_sink"
        self._branch_name = f"detector_{sensor_id}"
        self._handle: BranchHandle | None = None
        self._appsink: Any = None  # reason: GstApp.AppSink — duck-typed
        self._signal_id: int | None = None

        log.debug(
            "detector_runner.created",
            manifest=manifest.name,
            sensor_id=sensor_id,
            tee_name=tee_name,
            tier=self._tier.value,
            score_threshold=contract.score_threshold,
            num_labels=len(contract.labels),
        )

    def start(self) -> None:
        """Attach the tier-appropriate branch to the live tee and connect the signal.

        For :attr:`~objects_detector.DetectorTier.CPU_ONNX`: builds a
        ``videoconvert ! video/x-raw,format=BGR ! appsink`` branch (system-memory CPU
        tier) and wires ``new-sample`` to :meth:`_on_sample_bgr`.

        For :attr:`~objects_detector.DetectorTier.NVMM_AGGREGATOR`: builds a
        ``<conv> ! video/x-raw(memory:NVMM),format=RGBA ! appsink`` branch
        (zero-copy NVMM tier), where ``<conv>`` is the platform's NVMM converter
        (``nvvideoconvert`` on DeepStream / ``nvvidconv`` on Tegra), resolved by
        :func:`~openral_runner.backends.gstreamer.pipeline.nvmm_convert_element`,
        and wires ``new-sample`` to :meth:`_on_sample_nvmm`.

        Raises:
            ROSConfigError: If the NVMM aggregator tier is selected but no NVMM
                colour-convert element is registered on the host.
            ROSRuntimeError: If the appsink element cannot be found after attach.
        """
        if self._tier is DetectorTier.NVMM_AGGREGATOR:
            conv = nvmm_convert_element()
            if conv is None:
                raise ROSConfigError(
                    "DetectorRunner: the NVMM aggregator tier needs an NVMM colour-convert "
                    "element ('nvvideoconvert' from DeepStream or 'nvvidconv' from Tegra/L4T); "
                    "neither is registered on this host."
                )
            sink_flags = "emit-signals=true max-buffers=1 drop=true sync=false"
            elements = (
                f"{conv} ! video/x-raw(memory:NVMM),format=RGBA,"
                f"width={self._net_w},height={self._net_h} ! "
                f"appsink name={self._sink_name} {sink_flags}"
            )
            handler = self._on_sample_nvmm
        else:  # CPU_ONNX
            sink_flags = "emit-signals=true max-buffers=1 drop=true sync=false"
            elements = (
                f"videoconvert ! video/x-raw,format=BGR ! "
                f"appsink name={self._sink_name} {sink_flags}"
            )
            handler = self._on_sample_bgr
        self._handle = self._tee.attach(elements, name=self._branch_name)

        # After adding the branch bin to the pipeline, look up the appsink by name.
        # get_by_name recurses into all child bins so it finds the appsink inside the branch bin.
        self._appsink = self._pipeline.get_by_name(self._sink_name)
        if self._appsink is None:
            # Roll back the branch before raising so the pipeline stays consistent.
            self._tee.detach(self._handle)
            self._handle = None
            raise ROSRuntimeError(
                f"DetectorRunner: appsink '{self._sink_name}' not found after attach; "
                "check that the branch bin was added to the pipeline successfully."
            )

        self._signal_id = self._appsink.connect("new-sample", handler)
        log.debug(
            "detector_runner.started",
            sensor_id=self._sensor_id,
            branch=self._branch_name,
            sink=self._sink_name,
            tier=self._tier.value,
        )

    def _on_sample_bgr(self, appsink: Any) -> int:  # noqa: ANN401  # reason: GstApp.AppSink — duck-typed
        """Pull a BGR sample, run the detector, and fire ``on_detection`` on a hit.

        Mirrors the ``_pull_bgr_sample`` pattern from
        :mod:`openral_runner.backends.gstreamer.perception_tee`:
        caps → structure, assert ``format=="BGR"``, get_int width/height,
        buffer.map(READ), ``bytes(map_info.data)``, unmap.

        Returns:
            ``int(Gst.FlowReturn.OK)`` always — the streaming thread must not
            see a Python exception.
        """
        ok_flow = int(Gst.FlowReturn.OK)

        sample = appsink.emit("pull-sample")
        if sample is None:  # pragma: no cover — EOS edge
            return ok_flow

        caps = sample.get_caps()
        structure = caps.get_structure(0) if caps is not None else None
        if structure is None:  # pragma: no cover
            return ok_flow

        gst_format = structure.get_string("format") or ""
        if gst_format != "BGR":
            log.warning(
                "detector_runner.unsupported_format",
                sensor_id=self._sensor_id,
                format=gst_format,
            )
            return ok_flow

        ok_w, width = structure.get_int("width")
        ok_h, height = structure.get_int("height")
        if not (ok_w and ok_h):  # pragma: no cover
            return ok_flow

        buffer = sample.get_buffer()
        ok_map, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok_map:  # pragma: no cover
            return ok_flow
        try:
            payload = bytes(map_info.data)
        finally:
            buffer.unmap(map_info)

        md = self._detector.detect(payload, int(width), int(height), self._sensor_id)
        if md is not None:
            try:
                self._on_detection(md)
            except Exception:  # reason: guard callback errors from the streaming thread
                log.warning(
                    "detector_runner.callback_error",
                    sensor_id=self._sensor_id,
                    exc_info=True,
                )

        return ok_flow

    def _on_sample_nvmm(self, appsink: Any) -> int:  # noqa: ANN401  # reason: GstApp.AppSink — duck-typed
        """Pull an NVMM RGBA sample, map it zero-copy, run the detector, fire on a hit.

        Mirrors the NVMM map path in :mod:`reader`. Returns ``int(Gst.FlowReturn.OK)``
        always — no Python exception may reach the GStreamer streaming thread.
        """
        import ctypes  # noqa: PLC0415  # reason: only the NVMM path needs ctypes

        from openral_runner.backends.gstreamer.nvbufsurface import wrap_buffer  # noqa: PLC0415

        ok_flow = int(Gst.FlowReturn.OK)
        sample = appsink.emit("pull-sample")
        if sample is None:  # pragma: no cover — EOS edge
            return ok_flow
        buffer = sample.get_buffer()
        ok_map, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok_map:  # pragma: no cover
            return ok_flow
        try:
            # NVMM buffers map read-only, so ctypes.from_buffer (which requires a
            # *writable* buffer) raises "underlying buffer is not writable". Copy
            # the small NvBufSurface descriptor instead — this copies only the
            # surface metadata struct, NOT the GPU frame (dataPtr still points at
            # device memory), so the frame stays zero-copy. ``struct_bytes`` must
            # outlive the wrap_buffer call: wrap_buffer derefs surface_list, whose
            # pointer indexes back into the still-mapped original buffer. Validated
            # against real DeepStream nvvideoconvert NVMM buffers in the ds-on
            # container (the host has no NVMM path to exercise this).
            struct_bytes = (ctypes.c_uint8 * map_info.size).from_buffer_copy(map_info.data)
            buffer_address = ctypes.cast(struct_bytes, ctypes.c_void_p).value
            if buffer_address is None:  # pragma: no cover — NULL base
                return ok_flow
            handle = wrap_buffer(buffer_address)
            md = self._detector.detect_nvmm(handle, self._sensor_id)
        except Exception:  # reason: guard the streaming thread from any unwrap/infer error
            log.warning(
                "detector_runner.nvmm_sample_error",
                sensor_id=self._sensor_id,
                exc_info=True,
            )
            return ok_flow
        finally:
            buffer.unmap(map_info)

        if md is not None:
            try:
                self._on_detection(md)
            except Exception:  # reason: guard callback errors from the streaming thread
                log.warning(
                    "detector_runner.callback_error",
                    sensor_id=self._sensor_id,
                    exc_info=True,
                )
        return ok_flow

    def stop(self) -> None:
        """Disconnect the signal and detach the branch from the live tee.

        Idempotent — calling ``stop()`` a second time is a no-op.
        Also closes the detector if it exposes a ``close()`` method (NVMM tier).
        """
        if self._signal_id is not None and self._appsink is not None:
            # Suppress: defensive cleanup — appsink may already be NULL on a torn-down pipeline.
            with contextlib.suppress(Exception):
                self._appsink.disconnect(self._signal_id)
            self._signal_id = None

        if self._handle is not None:
            self._tee.detach(self._handle)
            self._handle = None

        self._appsink = None

        if hasattr(self._detector, "close"):
            # reason: defensive — executor may already be freed
            with contextlib.suppress(Exception):
                self._detector.close()

        log.debug("detector_runner.stopped", sensor_id=self._sensor_id)
