"""GStreamer pipeline-string builder + platform detection.

This module is import-safe on hosts without GStreamer or PyGObject: it
does **not** import ``gi`` at module load. All it does is

1. Detect what kind of NVIDIA platform we are on
   (:class:`Platform`) by reading ``/etc/nv_tegra_release`` and
   probing ``gst-inspect-1.0`` for the presence of specific elements
   (``nvarguscamerasrc`` on Tegra, ``nvh264dec`` on desktop NVIDIA).
2. Build a GStreamer pipeline string from a typed
   :class:`PipelineSpec`, selecting the right source / decoder /
   format-converter elements for the detected platform.
3. Ensure the trailing ``appsink`` element carries a known name
   (default ``bh_sink``) so the reader can fetch it via
   ``Gst.Bin.get_by_name``.

The output is fed verbatim to ``Gst.parse_launch`` inside the reader
(commit #2 of ADR-0010 PR I). Keeping the builder a pure-Python
string transformer means every code path can be unit-tested on
stock Ubuntu without NVIDIA plugins or PyGObject.
"""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "LEAKY_BRANCH_QUEUE",
    "TEE_NAME",
    "PipelineSpec",
    "Platform",
    "Source",
    "build_pipeline_string",
    "detect_platform",
    "ensure_appsink_name",
    "inspect_element_present",
    "leaky_branch",
    "nvmm_convert_element",
]


# Default name attached to the trailing appsink so the reader can look it up
# via Gst.Bin.get_by_name without parsing the string back.
_DEFAULT_APPSINK_NAME: Final[str] = "bh_sink"

# Name of the per-camera ``tee`` that fans the decoded / GPU-uploaded frame to
# the policy leg and every optional leg. This tee is the **perception-bus
# attach point** (ADR-0037): the runtime ``TeeManager`` looks it up by name via
# ``Gst.Bin.get_by_name`` to request pads for reasoner-activated consumers (the
# object detector now, VLAs later) at runtime. Exported so that module can
# reference the same name the static builder emits.
TEE_NAME: Final[str] = "openral_cam_tee"

# Per-branch leaky queue. Every tee branch is prefixed with this so a slow or
# crashing consumer drops its own frames rather than backpressuring the policy
# leg (ADR-0018 §3 isolation invariant). Defined once and shared by the static
# builder (:func:`leaky_branch`) and the runtime ``TeeManager`` so a dynamically
# attached branch carries the identical isolation policy.
LEAKY_BRANCH_QUEUE: Final[str] = "queue leaky=downstream max-size-buffers=2"


def leaky_branch(elements: str, *, tee_name: str = TEE_NAME) -> str:
    """Return one ``tee`` branch: ``<tee>. ! <leaky queue> ! <elements>``.

    The single definition of the per-branch isolation policy (a
    ``leaky=downstream`` queue, ADR-0018 §3) so the static pipeline builder and
    the runtime ``TeeManager`` (ADR-0037) construct branches identically — a
    stalled consumer drops its own frames instead of stalling the policy leg.

    Args:
        elements: The branch body downstream of the leaky queue (e.g. an
            ``appsink``, or a ``nvvidconv ! ... ! appsink`` chain).
        tee_name: Name of the upstream ``tee`` to branch from. Defaults to
            :data:`TEE_NAME`.

    Returns:
        A pipeline-string fragment beginning ``<tee_name>. ! queue ...``.

    Example:
        >>> leaky_branch("appsink name=bh_sink", tee_name="t")
        't. ! queue leaky=downstream max-size-buffers=2 ! appsink name=bh_sink'
    """
    return f"{tee_name}. ! {LEAKY_BRANCH_QUEUE} ! {elements}"


# Path read to identify a Tegra host (Jetson / Spark). Present on every L4T
# image NVIDIA ships, absent on desktop Ubuntu.
_TEGRA_RELEASE_PATH: Final[Path] = Path("/etc/nv_tegra_release")

# Timeout for the gst-inspect-1.0 probe. The tool returns in well under a
# second on a warm host; the timeout exists to keep an unhealthy plugin
# registry from hanging tests.
_GST_INSPECT_TIMEOUT_S: Final[float] = 5.0


class Platform(str, Enum):
    """The kind of host the runner is executing on.

    ``TEGRA`` (Jetson / Spark) uses NVIDIA's L4T multimedia stack:
    ``nvarguscamerasrc`` for MIPI CSI, ``nvv4l2decoder`` for H.264 /
    HEVC, ``nvvidconv`` for NVMM-aware colour conversion. Buffers
    flow in ``video/x-raw(memory:NVMM)`` caps and can be lifted to
    CUDA zero-copy via ``libnvbufsurface.so``.

    ``NVIDIA_DESKTOP`` is x86_64 with an NVIDIA GPU and the
    ``gstreamer1.0-plugins-bad`` ``nvcodec`` family installed.
    The nvcodec plugin ships the H.264 / H.265 / JPEG / AV1 dec/enc
    family (``nvh264dec``, ``nvh264enc``, ``nvh265dec``, ``nvjpegdec``,
    …) and the ``cudaupload`` / ``cudadownload`` data-movement
    elements, but **not** ``nvvideoconvert`` in the open-core image.
    The pipeline builder (``_build_convert``) falls back to stock
    ``videoconvert`` on this branch; the GPU path covers decode/encode
    only. Note: the opt-in ``ds-on`` (DS9) image
    (``docker/inference/Dockerfile.x86``, ``WITH_DEEPSTREAM_STAGE=on``)
    **does** install ``nvvideoconvert`` + NVMM caps on x86; the detector
    NVMM aggregator tier (ADR-0037 PR5b) uses it via
    :func:`nvmm_convert_element`. The ``_build_convert`` / ``_build_caps``
    functions are for the main reader pipeline and remain DeepStream-free
    on open-core; the detector branch builds its own NVMM capsfilter
    independently.

    ``CPU_ONLY`` is the fallback: stock GStreamer plugins, no
    NVIDIA-specific elements. Frames flow in system memory; the
    NVMM path is unavailable.
    """

    TEGRA = "tegra"
    NVIDIA_DESKTOP = "nvidia_desktop"
    CPU_ONLY = "cpu_only"


class Source(str, Enum):
    """What kind of upstream feeds the pipeline.

    The :class:`Source` selects the head element of the pipeline; the
    :class:`Platform` selects which NVIDIA-accelerated variant of
    decoder / converter follows it.
    """

    USB = "usb"
    """``/dev/videoN`` v4l2 device — USB webcam or capture card."""

    CSI = "csi"
    """Jetson MIPI CSI camera via ``nvarguscamerasrc`` (Tegra only)."""

    RTSP = "rtsp"
    """Network camera over RTSP."""

    FILE = "file"
    """A media file on disk (mostly for offline replay / regression tests)."""

    TESTSRC = "testsrc"
    """``videotestsrc`` — synthetic frames; runs anywhere, no hardware."""


class PipelineSpec(BaseModel):
    """Validated description of a GStreamer ingest pipeline.

    A :class:`PipelineSpec` is a *structured* alternative to passing a
    raw GStreamer pipeline string through
    ``SensorReaderConfig.backend_params["pipeline"]``. The reader's
    factory accepts either form: when the YAML supplies ``pipeline``
    directly the string is passed through (with the appsink name
    ensured); when the YAML supplies ``source / device / ...`` we
    materialise a :class:`PipelineSpec` and build the string here.

    Args:
        source: The upstream element family (see :class:`Source`).
        device: Source-specific device locator. ``int`` for v4l2 device
            indices, ``str`` for paths (``/dev/video0``, ``/path/to/file.mp4``,
            ``rtsp://...``, etc.). Ignored when ``source == TESTSRC``.
        width: Negotiated frame width in pixels. ``None`` lets the source
            element pick its default.
        height: Negotiated frame height in pixels. ``None`` lets the source
            element pick its default.
        fps: Capture / framerate hint forwarded to the source.
        encoded: ``True`` when the upstream stream is compressed
            (H.264 / HEVC, e.g. typical USB-UVC H.264 or RTSP). Drives
            decoder selection.
        enable_nvmm: Hint to keep frames in ``memory:NVMM`` caps for
            NVMM→CUDA handoff. Honored only when the platform is
            :attr:`Platform.TEGRA` or :attr:`Platform.NVIDIA_DESKTOP`.
        enable_ros_tee: When ``True``, the builder inserts a ``tee`` so
            a second branch can feed a ROS 2 publisher (commit #4).
        enable_event_tee: When ``True``, the builder adds a third
            ``tee`` branch terminating in ``event_appsink_name``. The
            event branch lifts frames to system memory and rate-limits
            via ``videorate`` to :attr:`event_rate_hz`; the
            :class:`PerceptionEventPublisher` (ADR-0018 F6) runs
            detectors on its samples and publishes
            ``PromptStamped`` on ``/openral/perception/<kind>``. Policy
            and event legs share the
            :func:`openral_runner.backends.gstreamer.cuda_context.get_shared_cuda_context`
            singleton, per ADR-0011 §"Shared CUDA context".
        appsink_name: Name attached to the openral appsink for
            ``Gst.Bin.get_by_name``. Defaults to ``bh_sink``.
        ros_appsink_name: Name attached to the ROS-side appsink (when
            ``enable_ros_tee`` is ``True``). Defaults to ``ros_sink``.
        event_appsink_name: Name attached to the event-leg appsink (when
            ``enable_event_tee`` is ``True``). Defaults to
            ``event_sink``.
        event_rate_hz: Maximum framerate on the event branch in Hz. The
            event leg is for vision events (motion, scene change,
            objects, OCR) and runs at a fraction of the policy rate so
            CPU detectors keep up. Defaults to 5 Hz, matching the
            reasoner tick rate.
        max_buffers: ``max-buffers`` property on the appsink. The reader
            wants ``1`` (latest-only) but tests with finite
            ``num-buffers`` may set higher to avoid drop messages.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Source
    device: int | str | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    fps: int = Field(default=30, gt=0)
    encoded: bool = False
    enable_nvmm: bool = True
    enable_ros_tee: bool = False
    enable_event_tee: bool = False
    appsink_name: str = _DEFAULT_APPSINK_NAME
    ros_appsink_name: str = "ros_sink"
    event_appsink_name: str = "event_sink"
    event_rate_hz: float = Field(default=5.0, gt=0)
    max_buffers: int = Field(default=1, gt=0)

    @field_validator("appsink_name", "ros_appsink_name", "event_appsink_name")
    @classmethod
    def _validate_element_name(cls, value: str) -> str:
        """GStreamer element names are restricted; reject obviously bad ones."""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*", value):
            raise ValueError(
                f"appsink name {value!r} is not a valid GStreamer element name; "
                "expected /[A-Za-z_][A-Za-z0-9_-]*/"
            )
        return value


@functools.lru_cache(maxsize=1)
def detect_platform() -> Platform:
    """Detect which NVIDIA platform we are on.

    Order:

    1. If ``/etc/nv_tegra_release`` exists, this is a Tegra host
       (Jetson Nano / NX / AGX / Thor / Spark). Return
       :attr:`Platform.TEGRA`.
    2. Else, if ``gst-inspect-1.0`` reports the ``nvh264dec`` element
       present, this is a desktop NVIDIA host with the ``nvcodec``
       plugin family installed. Return :attr:`Platform.NVIDIA_DESKTOP`.
    3. Otherwise return :attr:`Platform.CPU_ONLY`.

    The result is cached for the lifetime of the Python process via
    :func:`functools.lru_cache`; platform never changes mid-run.

    Returns:
        The detected :class:`Platform`.

    Example:
        >>> detect_platform() in {Platform.TEGRA, Platform.NVIDIA_DESKTOP, Platform.CPU_ONLY}
        True
    """
    if _TEGRA_RELEASE_PATH.exists():
        return Platform.TEGRA
    if inspect_element_present("nvh264dec"):
        return Platform.NVIDIA_DESKTOP
    return Platform.CPU_ONLY


def inspect_element_present(element_name: str) -> bool:
    """Return ``True`` when ``gst-inspect-1.0 <element_name>`` succeeds.

    Used by :func:`detect_platform` to probe for ``nvh264dec``,
    ``nvarguscamerasrc``, etc. Falls back to ``False`` when
    ``gst-inspect-1.0`` is not on ``$PATH`` (i.e. GStreamer not
    installed at all).

    Args:
        element_name: GStreamer element factory name.

    Returns:
        ``True`` iff the element is registered with the local GStreamer
        plugin registry.
    """
    if shutil.which("gst-inspect-1.0") is None:
        return False
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", "--exists", element_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_GST_INSPECT_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def nvmm_convert_element() -> str | None:
    """Return the NVMM-aware colour-convert element registered on this host, or ``None``.

    DeepStream (the x86 ``ds-on`` image) provides ``nvvideoconvert``; the Tegra/L4T
    multimedia stack provides ``nvvidconv``. Prefer ``nvvideoconvert`` when present
    (the detector's NVMM aggregator tier is validated in the DeepStream container),
    else ``nvvidconv``. Returns ``None`` when neither is registered — the host has no
    NVMM colour-convert path.

    Example:
        >>> nvmm_convert_element() in {"nvvideoconvert", "nvvidconv", None}
        True
    """
    if inspect_element_present("nvvideoconvert"):
        return "nvvideoconvert"
    if inspect_element_present("nvvidconv"):
        return "nvvidconv"
    return None


def ensure_appsink_name(pipeline: str, name: str = _DEFAULT_APPSINK_NAME) -> str:
    """Inject ``name=<name>`` into the trailing ``appsink`` element if absent.

    The reader (commit #2) looks the appsink up by name via
    ``Gst.Bin.get_by_name``. Users who pass a raw pipeline string in
    YAML often forget to name the appsink; this function rewrites the
    string to add ``name=bh_sink`` so the reader can find it.

    If the pipeline already names its trailing appsink (e.g.
    ``... ! appsink name=my_sink``), the string is returned unchanged
    — the caller is expected to pass the same name to
    :class:`GStreamerSensorReader` via ``appsink_name``.

    Args:
        pipeline: GStreamer pipeline string. Must end in an ``appsink``
            element (case-sensitive). A trailing ``!`` separator is
            allowed but not required.
        name: The name to set on the appsink. Validated by
            :class:`PipelineSpec`.

    Returns:
        The (possibly rewritten) pipeline string.

    Raises:
        ValueError: When the pipeline does not appear to terminate in
            an ``appsink`` element.

    Example:
        >>> ensure_appsink_name("videotestsrc ! videoconvert ! appsink")
        'videotestsrc ! videoconvert ! appsink name=bh_sink'
        >>> ensure_appsink_name("v4l2src ! appsink name=already_named")
        'v4l2src ! appsink name=already_named'
    """
    pipeline = pipeline.strip()
    # Match the last appsink element. ``appsink`` may be followed by
    # whitespace + optional properties, terminated either by end of
    # string or another ``!``. We only touch the *trailing* appsink so
    # an intermediate ``! tee ! ... ! appsink name=ros_sink t. ! ... !
    # appsink`` is handled correctly: the second match is the ral sink.
    match = re.search(r"\bappsink\b([^!]*)$", pipeline)
    if match is None:
        raise ValueError(
            f"Pipeline string must terminate in an 'appsink' element; got: {pipeline!r}"
        )
    tail = match.group(1)
    if re.search(r"\bname\s*=", tail):
        return pipeline
    suffix = f" name={name}"
    insert_at = match.end(1)
    return pipeline[:insert_at] + suffix + pipeline[insert_at:]


def build_pipeline_string(spec: PipelineSpec, platform: Platform | None = None) -> str:
    """Materialise a GStreamer pipeline string from a :class:`PipelineSpec`.

    Element selection is platform-aware:

    * Source: ``v4l2src`` for USB on every platform; ``nvarguscamerasrc``
      for CSI on Tegra (raises on non-Tegra); ``rtspsrc`` for RTSP;
      ``filesrc`` for file; ``videotestsrc`` for synthetic.
    * Decode (when ``spec.encoded`` is True): ``nvv4l2decoder`` on Tegra,
      ``nvh264dec`` on desktop NVIDIA, ``avdec_h264`` on CPU-only.
    * Colour convert: ``nvvidconv`` on Tegra; ``videoconvert`` on
      desktop NVIDIA (the open-core image does not bundle NVIDIA
      DeepStream, so ``nvvideoconvert`` is not used here — H.264 dec
      stays on the GPU but colour conversion runs on the CPU; the
      opt-in ``ds-on`` image does provide ``nvvideoconvert`` on x86
      but that is used only by the detector NVMM branch, not this
      builder); and ``videoconvert`` on CPU-only.
    * Memory: ``video/x-raw(memory:NVMM)`` caps when
      ``spec.enable_nvmm`` AND the platform supports it; ``video/x-raw``
      (system memory) otherwise.
    * ROS tee: when ``spec.enable_ros_tee``, a second branch is inserted
      that lifts NVMM frames to system memory before the ROS-side appsink.

    Args:
        spec: The validated :class:`PipelineSpec`.
        platform: Override platform detection. Used by tests to force
            CPU-only builds on a Tegra host or vice versa. ``None`` ⇒
            call :func:`detect_platform`.

    Returns:
        A GStreamer pipeline string. Always terminates in an ``appsink``
        named per :attr:`PipelineSpec.appsink_name`.

    Raises:
        ValueError: When the requested ``source`` is incompatible with
            the platform (e.g. ``CSI`` on ``CPU_ONLY``).

    Example:
        >>> spec = PipelineSpec(source=Source.TESTSRC, fps=30, width=320, height=240)
        >>> out = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
        >>> out.startswith("videotestsrc is-live=true")
        True
        >>> "videoconvert" in out and "appsink name=bh_sink" in out
        True
    """
    if platform is None:
        platform = detect_platform()

    head = _build_head(spec, platform)
    decode = _build_decode(spec, platform)
    convert = _build_convert(spec, platform)
    caps = _build_caps(spec, platform)
    appsink = _build_appsink(spec.appsink_name, max_buffers=spec.max_buffers)

    parts: list[str] = [head]
    if decode:
        parts.append(decode)
    if convert:
        parts.append(convert)
    if caps:
        parts.append(caps)

    if spec.enable_ros_tee or spec.enable_event_tee:
        # Fan the frame to the policy leg + every optional leg off a single
        # named tee — the perception-bus attach point (ADR-0037) the runtime
        # TeeManager later requests pads on. Each leg is wrapped by
        # ``leaky_branch`` so a slow / crashing consumer drops its own frames
        # rather than backpressuring the policy leg (ADR-0018 §3).
        branches: list[str] = [leaky_branch(appsink)]
        if spec.enable_ros_tee:
            branches.append(leaky_branch(_build_ros_tee_branch(spec, platform)))
        if spec.enable_event_tee:
            branches.append(leaky_branch(_build_event_tee_branch(spec, platform)))
        return " ! ".join(parts) + f" ! tee name={TEE_NAME}  " + "  ".join(branches)

    parts.append(appsink)
    return " ! ".join(parts)


# ── Element builders ───────────────────────────────────────────────────────────


def _build_head(spec: PipelineSpec, platform: Platform) -> str:
    """Return the source element + its properties."""
    if spec.source is Source.TESTSRC:
        return "videotestsrc is-live=true"
    if spec.source is Source.USB:
        device = _coerce_device_str(spec.device, default="/dev/video0")
        return f"v4l2src device={device}"
    if spec.source is Source.CSI:
        if platform is not Platform.TEGRA:
            raise ValueError(
                f"Source.CSI requires Platform.TEGRA, got {platform}. "
                "MIPI CSI cameras are only exposed via nvarguscamerasrc on Jetson / Spark."
            )
        # nvarguscamerasrc takes a bare integer sensor-id (0, 1, …), not a /dev path.
        sensor_id = str(spec.device) if spec.device is not None else "0"
        return f"nvarguscamerasrc sensor-id={sensor_id}"
    if spec.source is Source.RTSP:
        location = _coerce_device_str(spec.device, default=None)
        if location is None:
            raise ValueError("Source.RTSP requires backend_params.device=<rtsp url>")
        # latency=0 to avoid the default 200ms jitter buffer when feeding a
        # real-time policy; the runner's deadline guard catches stalls.
        return f"rtspsrc location={location} latency=0"
    if spec.source is Source.FILE:
        location = _coerce_device_str(spec.device, default=None)
        if location is None:
            raise ValueError("Source.FILE requires backend_params.device=<file path>")
        return f"filesrc location={location}"
    raise ValueError(f"Unhandled Source: {spec.source}")  # pragma: no cover — enum-exhaustive


def _build_decode(spec: PipelineSpec, platform: Platform) -> str:
    """Return the decoder + parser when the upstream stream is encoded.

    File source defers to ``decodebin`` (auto-picks decoder); RTSP and
    encoded USB share a per-platform H.264 decoder lookup, with RTSP
    prefixing the depay element.
    """
    if spec.source is Source.FILE:
        return "decodebin"
    h264_decoder = {
        Platform.TEGRA: "nvv4l2decoder",
        Platform.NVIDIA_DESKTOP: "nvh264dec",
        Platform.CPU_ONLY: "avdec_h264",
    }[platform]
    if spec.source is Source.RTSP:
        return f"rtph264depay ! h264parse ! {h264_decoder}"
    if not spec.encoded:
        return ""
    # USB cameras that deliver an H.264 stream — rare in practice but supported
    # by some industrial cams. Mirrors the RTSP branch sans depay.
    return f"h264parse ! {h264_decoder}"


def _build_convert(spec: PipelineSpec, platform: Platform) -> str:
    """Return the colour-conversion element appropriate for the platform.

    ``nvvidconv`` exists on Tegra (L4T multimedia stack, NVMM-aware).
    ``nvvideoconvert`` is a NVIDIA DeepStream element — it is **not**
    in the open-source ``gstreamer1.0-plugins-bad`` ``nvcodec`` plugin
    family. ADR-0010 Amendment 2026-05-12 rejected bundling DeepStream
    into open-core; the corollary is that ``Platform.NVIDIA_DESKTOP``
    falls back to stock ``videoconvert`` (CPU) for the main reader
    pipeline. The H.264 / H.265 dec/enc steps still run on the GPU via
    ``nvh264dec`` / ``nvh264enc``; only the colour-space convert step
    runs on the CPU.

    Note: the opt-in ``ds-on`` (DS9) image
    (``docker/inference/Dockerfile.x86``, ``WITH_DEEPSTREAM_STAGE=on``)
    **does** provide ``nvvideoconvert`` + NVMM caps on x86. The detector
    NVMM aggregator tier (ADR-0037 PR5b) uses it via the separate
    :func:`nvmm_convert_element` resolver, which is called by
    :class:`~openral_runner.backends.gstreamer.detector_runner.DetectorRunner`
    when building the NVMM branch — not by this function. The main
    reader pipeline (this function) remains DeepStream-free on open-core.
    """
    if platform is Platform.TEGRA:
        return "nvvidconv"
    return "videoconvert"


def _build_caps(spec: PipelineSpec, platform: Platform) -> str:
    """Return a capsfilter element with negotiated format / size / framerate.

    The format is pinned to ``BGR`` on the CPU path so the appsink callback
    sees a known per-pixel encoding (the CPU branch of the reader rejects
    NV12 / I420). On NVMM paths the format defaults to ``NV12`` which is
    what ``nvvidconv`` outputs by default on Tegra and what the
    NvBufSurface ctypes wrapper expects.

    NVMM caps are only emitted for ``Platform.TEGRA`` on the main
    reader pipeline. On ``Platform.NVIDIA_DESKTOP`` the open-core image
    does not bundle DeepStream, so no element in the standard reader
    pipeline produces ``memory:NVMM`` caps after ``videoconvert``;
    claiming NVMM there would fail caps negotiation. (The opt-in
    ``ds-on`` (DS9) image does provide NVMM on x86, but the detector
    NVMM aggregator tier (ADR-0037 PR5b) builds its own NVMM branch via
    :func:`nvmm_convert_element` independently of this function.)
    """
    use_nvmm = spec.enable_nvmm and platform is Platform.TEGRA
    raw = "video/x-raw(memory:NVMM)" if use_nvmm else "video/x-raw"
    fmt = "NV12" if use_nvmm else "BGR"
    fields: list[str] = [f"format={fmt}"]
    if spec.width is not None:
        fields.append(f"width={spec.width}")
    if spec.height is not None:
        fields.append(f"height={spec.height}")
    fields.append(f"framerate={spec.fps}/1")
    return f"{raw},{','.join(fields)}"


def _build_appsink(name: str, *, max_buffers: int) -> str:
    """Return the openral appsink with the standard reader properties.

    ``emit-signals=true`` enables the ``new-sample`` signal the reader
    connects to. ``drop=true`` + ``max-buffers=1`` (configurable) keeps
    only the latest frame in the sink — older frames are discarded as
    soon as a newer one arrives, which is exactly the latest-only
    contract :meth:`SensorReader.read_latest` needs.

    ``sync=false`` lets the appsink ingest as fast as upstream
    delivers, regardless of clock — we use monotonic timestamps from
    the buffer PTS rather than realtime sync.
    """
    return f"appsink name={name} emit-signals=true max-buffers={max_buffers} drop=true sync=false"


def _build_ros_tee_branch(spec: PipelineSpec, platform: Platform) -> str:
    """Return the secondary branch fed into the ROS publisher (commit #4).

    The ROS path **always** lifts frames to system memory before the
    appsink so that the publisher and the ral appsink don't share
    NVMM buffer ownership.
    """
    convert = "nvvidconv" if platform is Platform.TEGRA else "videoconvert"
    # Force system memory caps on the ROS side regardless of upstream.
    return (
        f"{convert} ! video/x-raw,format=BGR ! "
        f"appsink name={spec.ros_appsink_name} emit-signals=true "
        f"max-buffers=1 drop=true sync=false"
    )


def _build_event_tee_branch(spec: PipelineSpec, platform: Platform) -> str:
    """Return the perception/event branch fed into the event detector (ADR-0018 F6).

    The event branch is the third leg of the per-camera ``tee`` (after the
    policy NVMM/CUDA leg and the ROS observability leg). It always lifts
    frames to system memory because Python-level event detectors
    (``cv2``, ``tflite``) consume system-memory buffers; on Jetson the
    in-pipeline ``nvinfer`` element runs upstream of this branch via a
    downstream patch and only the post-inference metadata reaches the
    appsink. A ``videorate`` cap pins the leg to
    :attr:`PipelineSpec.event_rate_hz` so a 30 Hz policy leg coexists
    with a 5 Hz detector loop.
    """
    convert = "nvvidconv" if platform is Platform.TEGRA else "videoconvert"
    # event_rate_hz is float; render as Gst fraction (rate/1) by rounding to int.
    # ``videorate`` then drops or duplicates frames as needed to honour the cap.
    rate = max(1, round(spec.event_rate_hz))
    return (
        f"{convert} ! videorate ! "
        f"video/x-raw,format=BGR,framerate={rate}/1 ! "
        f"appsink name={spec.event_appsink_name} emit-signals=true "
        f"max-buffers=1 drop=true sync=false"
    )


def _coerce_device_str(device: int | str | None, *, default: str | None) -> str | None:
    """Convert YAML-typed device value to a string for the pipeline."""
    if device is None:
        return default
    if isinstance(device, int):
        return f"/dev/video{device}"
    return str(device)
