"""Producer-side helpers for recording rich span attributes on OpenRAL hot-path spans.

Each function takes an open span (from ``opentelemetry.trace``) plus a
typed payload and writes the attribute set the live dashboard expects.
Centralising the encoding here keeps the wire format consistent across
HAL adapters, world-state aggregators, and sensor readers — and lets us
evolve the schema (rounding, list-truncation, thumbnail size) in one
place.

All helpers are safe to call on a no-op span (the default before
:func:`configure_observability` runs); they're additive, never raise on
missing optional fields, and silently truncate over-long lists so a
24-DoF arm doesn't blow up the span size.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from openral_observability import semconv

if TYPE_CHECKING:
    from opentelemetry.trace import Span

__all__ = [
    "encode_frame_thumbnail",
    "encode_rgb_thumbnail",
    "modality_for_encoding",
    "record_action",
    "record_ee_poses",
    "record_joint_state",
    "record_sensor_frame_attrs",
]


_MODALITY_BY_ENCODING: dict[str, str] = {
    "bgr8": "rgb",
    "rgb8": "rgb",
    "mono8": "mono",
    "depth16": "depth",
    "jpeg": "rgb",
    "png": "rgb",
    "cuda_nv12": "rgb",
    "raw": "raw",
}


def modality_for_encoding(encoding: object) -> str:
    """Map a :class:`openral_core.FrameEncoding` (or its string value) to a modality label.

    The dashboard's Perception card groups frames by modality
    (``rgb`` / ``mono`` / ``depth`` / ``raw``) — keep the mapping
    centralised here so HAL nodes and the HardwareRunner produce the
    same label for a given encoding.
    """
    value = getattr(encoding, "value", encoding)
    return _MODALITY_BY_ENCODING.get(str(value), "unknown")


# Cap the number of joints / pose components we send so a future 100-DoF
# humanoid doesn't push past OTLP's per-attribute size limits. Spans
# stay readable in Jaeger and the dashboard ring stays bounded.
_MAX_JOINTS = 64
_MAX_EE_FRAMES = 8
# Thumbnail target — capped at VGA so the dashboard shows native-resolution
# frames (the largest rskill camera contract is 640x480; PIL.thumbnail only
# ever shrinks, so this never upscales). Emitted at a throttled rate
# (HardwareRunner.thumbnail_hz, default 25 Hz), not faster than tick rate — so q90 stays
# cheap over OTLP even at 2-3 cameras on localhost.
_THUMB_MAX_WIDTH = 640
_THUMB_MAX_HEIGHT = 480
_THUMB_JPEG_QUALITY = 90


def _r3(values: Iterable[float]) -> list[float]:
    """Round to 3 decimals to keep the attribute payload compact."""
    return [round(float(v), 3) for v in values]


def record_joint_state(
    span: Span,
    *,
    names: list[str] | None,
    positions: list[float] | None,
    velocities: list[float] | None = None,
    efforts: list[float] | None = None,
    position_limits: list[tuple[float, float] | None] | None = None,
    velocity_limits: list[float | None] | None = None,
    effort_limits: list[float | None] | None = None,
    stamp_ns: int | None = None,
) -> None:
    """Attach per-joint robot-state attributes to a ``hal.read_state`` span.

    Lists are truncated to :data:`_MAX_JOINTS` and rounded to 3 decimals
    (~1 mrad on revolute joints — plenty for a debug pane). Limits are
    pulled from :class:`openral_core.JointSpec`; pass ``None`` per joint
    when a robot exposes a free axis.
    """
    if names is not None:
        span.set_attribute(semconv.HAL_JOINT_NAMES, list(names[:_MAX_JOINTS]))
    if positions is not None:
        span.set_attribute(semconv.HAL_JOINT_POSITIONS, _r3(positions[:_MAX_JOINTS]))
    if velocities is not None:
        span.set_attribute(semconv.HAL_JOINT_VELOCITIES, _r3(velocities[:_MAX_JOINTS]))
    if efforts is not None:
        span.set_attribute(semconv.HAL_JOINT_EFFORTS, _r3(efforts[:_MAX_JOINTS]))
    if position_limits is not None:
        limits = list(position_limits[:_MAX_JOINTS])
        lo = [round(float(lim[0]) if lim else float("-inf"), 3) for lim in limits]
        hi = [round(float(lim[1]) if lim else float("inf"), 3) for lim in limits]
        # OTLP doesn't carry ±inf cleanly; clamp to ±1e6.
        span.set_attribute(semconv.HAL_JOINT_POSITION_LIMITS_LO, [max(v, -1e6) for v in lo])
        span.set_attribute(semconv.HAL_JOINT_POSITION_LIMITS_HI, [min(v, 1e6) for v in hi])
    if velocity_limits is not None:
        span.set_attribute(
            semconv.HAL_JOINT_VELOCITY_LIMITS,
            [round(float(v), 3) if v is not None else 0.0 for v in velocity_limits[:_MAX_JOINTS]],
        )
    if effort_limits is not None:
        span.set_attribute(
            semconv.HAL_JOINT_EFFORT_LIMITS,
            [round(float(v), 3) if v is not None else 0.0 for v in effort_limits[:_MAX_JOINTS]],
        )
    if stamp_ns is not None:
        span.set_attribute(semconv.HAL_JOINT_STAMP_NS, int(stamp_ns))


def record_action(
    span: Span,
    *,
    next_row: list[float] | None,
    dim: int | None = None,
    horizon: int | None = None,
    applied: bool | None = None,
    gripper_position: float | None = None,
    gripper_force_n: float | None = None,
) -> None:
    """Attach commanded-action attributes to a ``hal.send_action`` span.

    ``next_row`` is the row of the action chunk that the runner is about
    to apply on this tick. We only record one row (not the full
    ``horizon * dim`` chunk) so the dashboard's command-vs-reality
    overlay stays cheap.
    """
    if next_row is not None:
        span.set_attribute(semconv.HAL_ACTION_NEXT, _r3(next_row[:_MAX_JOINTS]))
    if dim is not None:
        span.set_attribute(semconv.HAL_ACTION_DIM, int(dim))
    if horizon is not None:
        span.set_attribute(semconv.HAL_ACTION_HORIZON, int(horizon))
    if applied is not None:
        span.set_attribute(semconv.HAL_ACTION_APPLIED, bool(applied))
    if gripper_position is not None:
        span.set_attribute(semconv.HAL_GRIPPER_POSITION, round(float(gripper_position), 3))
    if gripper_force_n is not None:
        span.set_attribute(semconv.HAL_GRIPPER_FORCE_N, round(float(gripper_force_n), 3))


def record_ee_poses(span: Span, ee_poses: Any) -> None:
    """Attach end-effector poses to a ``world_state.snapshot`` span.

    Accepts a mapping of ``ee_name → Pose6D``-like object (any object
    that yields ``xyz`` as a 3-tuple and ``quat_xyzw`` as a 4-tuple —
    matches :class:`openral_core.Pose6D`). Poses are flattened as
    ``openral.hal.ee.pose.<name>`` → ``[x, y, z, qx, qy, qz, qw]``.
    """
    if not ee_poses:
        return
    names: list[str] = []
    for name, pose in list(ee_poses.items())[:_MAX_EE_FRAMES]:
        try:
            tx, ty, tz = pose.xyz
            qx, qy, qz, qw = pose.quat_xyzw
        except (AttributeError, ValueError, TypeError):
            continue
        names.append(str(name))
        span.set_attribute(
            f"{semconv.HAL_EE_POSE_PREFIX}.{name}",
            _r3([tx, ty, tz, qx, qy, qz, qw]),
        )
    if names:
        span.set_attribute(semconv.HAL_EE_NAMES, names)


def record_sensor_frame_attrs(
    span: Span,
    *,
    modality: str | None = None,
    encoding: str | None = None,
    width: int | None = None,
    height: int | None = None,
    channels: int | None = None,
    age_ms: float | None = None,
    thumbnail_bytes: bytes | None = None,
    thumbnail_already_encoded_b64: bool = False,
) -> None:
    """Attach sensor-frame attributes to a ``sensors.read_latest`` span.

    Pass ``thumbnail_bytes`` at the throttled dashboard rate
    (``HardwareRunner.thumbnail_hz``, default 25 Hz per camera); OTLP
    attributes are a preview channel, not a 30 fps video transport. When
    set, the value is base64-encoded inline; downstream consumers
    (including :mod:`openral_observability.dashboard`) decode it for
    display.
    """
    if modality is not None:
        span.set_attribute(semconv.SENSORS_MODALITY, str(modality))
    if encoding is not None:
        span.set_attribute(semconv.SENSORS_ENCODING, str(encoding))
    if width is not None:
        span.set_attribute(semconv.SENSORS_WIDTH, int(width))
    if height is not None:
        span.set_attribute(semconv.SENSORS_HEIGHT, int(height))
    if channels is not None:
        span.set_attribute(semconv.SENSORS_CHANNELS, int(channels))
    if age_ms is not None:
        span.set_attribute(semconv.SENSORS_AGE_MS, round(float(age_ms), 3))
    if thumbnail_bytes is not None:
        if thumbnail_already_encoded_b64:
            encoded = thumbnail_bytes.decode("ascii")
        else:
            encoded = base64.b64encode(thumbnail_bytes).decode("ascii")
        span.set_attribute(semconv.SENSORS_THUMBNAIL_JPEG_B64, encoded)


def encode_rgb_thumbnail(rgb: Any) -> bytes | None:
    """Encode an HWC uint8 RGB ndarray as a small JPEG suitable for OTLP.

    Returns ``None`` when Pillow isn't importable so producers can keep
    the call site unconditional without paying for an ImportError on
    headless test runners. Resizes to fit within
    ``_THUMB_MAX_WIDTH * _THUMB_MAX_HEIGHT`` preserving aspect ratio;
    encodes at JPEG quality ``_THUMB_JPEG_QUALITY`` (~60).
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.fromarray(rgb)
    except Exception:
        return None
    img.thumbnail((_THUMB_MAX_WIDTH, _THUMB_MAX_HEIGHT))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_THUMB_JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def encode_frame_thumbnail(frame: Any) -> bytes | None:
    """Encode a :class:`openral_core.SensorFrame` as a small JPEG thumbnail.

    Handles the encodings the dashboard knows how to render:

    * ``JPEG`` / ``PNG`` — decoded then re-encoded at the smaller size.
    * ``RGB8`` / ``BGR8`` — interpreted as ``H*W*C`` raw bytes.
    * ``MONO8`` — interpreted as grayscale, converted to RGB.

    Returns ``None`` for encodings the dashboard can't render
    (``DEPTH16``, ``CUDA_NV12``, ``RAW``) or when ``frame.data`` is
    empty (the frame carries a ``topic`` ref or a GPU ``handle``
    instead of inline pixels). Also returns ``None`` if Pillow isn't
    importable — the call site stays unconditional and gracefully
    skips the thumbnail attribute.

    The whole encode pipeline runs in a few ms per frame at typical
    sensor resolutions. The runner calls it at the throttled
    ``thumbnail_hz`` rate (not every tick), so the per-tick cost is
    amortised across cameras and the tick budget is unaffected.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    data = getattr(frame, "data", None)
    if not data:
        return None
    encoding = str(getattr(frame.encoding, "value", frame.encoding))
    width = int(getattr(frame, "width", 0) or 0)
    height = int(getattr(frame, "height", 0) or 0)
    img: Image.Image
    try:
        if encoding in ("jpeg", "png"):
            img = Image.open(io.BytesIO(data))
        elif encoding == "rgb8":
            img = Image.frombytes("RGB", (width, height), data)
        elif encoding == "bgr8":
            img = Image.frombytes("RGB", (width, height), data)
            b, g, r = img.split()
            img = Image.merge("RGB", (r, g, b))
        elif encoding == "mono8":
            img = Image.frombytes("L", (width, height), data).convert("RGB")
        else:
            # depth16 / cuda_nv12 / raw — not renderable as a colour thumb here.
            return None
    except Exception:
        return None
    img.thumbnail((_THUMB_MAX_WIDTH, _THUMB_MAX_HEIGHT))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_THUMB_JPEG_QUALITY, optimize=True)
    return buf.getvalue()
