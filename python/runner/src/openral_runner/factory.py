"""Typed registries that resolve a deployment's free axes.

Two dict registries map a config id to its factory, each rejecting unknown ids
with a typed ``ROSConfigError``:

* ``SKILL_REGISTRY`` — :attr:`VLASpec.id` → :class:`Skill` factory. Today only
  ``gpu_passthrough`` (a no-op rSkill for plumbing verification).
* ``SENSOR_BACKEND_REGISTRY`` — :attr:`SensorReaderConfig.backend` →
  :class:`SensorReader` factory (``opencv_thread`` / ``gstreamer``).

Adding skills / backends is additive — append to the dict. The rSkill that
drives a real deployment is selected at runtime by the reasoner from the
installed registry (``rskills/``), not pinned in the deploy config.
"""

from __future__ import annotations

from collections.abc import Callable

import structlog
from openral_core import (
    SensorReaderConfig,
)
from openral_core.exceptions import ROSConfigError
from openral_rskill.base import rSkillBase
from openral_rskill.gpu_passthrough import GpuPassthroughSkill

from openral_runner.backends import OpenCVThreadSensorReader
from openral_runner.backends.gstreamer.pipeline import PipelineSpec, Source
from openral_runner.sensor_reader import SensorReader

__all__ = [
    "SENSOR_BACKEND_REGISTRY",
    "SKILL_REGISTRY",
]

log = structlog.get_logger(__name__)


def _to_int(value: object, *, field: str, sensor_id: str) -> int:
    """Coerce a YAML-typed value (int / str / float) to ``int`` or raise."""
    if isinstance(value, bool):  # bool is a subclass of int; reject explicitly
        raise ROSConfigError(
            f"SensorReaderConfig({sensor_id!r}).backend_params.{field} must be "
            f"an integer, not a bool"
        )
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ROSConfigError(
                f"SensorReaderConfig({sensor_id!r}).backend_params.{field}="
                f"{value!r} is not coercible to int"
            ) from exc
    raise ROSConfigError(
        f"SensorReaderConfig({sensor_id!r}).backend_params.{field} has "
        f"unsupported type {type(value).__name__}"
    )


def _make_gpu_passthrough_skill(extra: dict[str, object]) -> rSkillBase:
    """Build a :class:`GpuPassthroughSkill` from ``vla.extra`` overrides.

    Recognised ``extra`` keys:
        ``sensor_id`` (str, default ``"wrist_rgb"``): which
            ``WorldState.image_frames`` slot to read.
        ``n_joints`` (int, default 6), ``horizon`` (int, default 1):
            zero-action chunk shape.
        ``device`` (str, default ``"cuda"``): torch device. The skill
            raises at configure() if cuda is requested and unavailable.
    """
    sensor_id = str(extra.get("sensor_id", "wrist_rgb"))
    n_joints = _to_int(extra.get("n_joints", 6), field="n_joints", sensor_id="gpu_passthrough")
    horizon = _to_int(extra.get("horizon", 1), field="horizon", sensor_id="gpu_passthrough")
    device = str(extra.get("device", "cuda"))
    return GpuPassthroughSkill(
        sensor_id=sensor_id,
        n_joints=n_joints,
        horizon=horizon,
        device=device,
    )


SKILL_REGISTRY: dict[str, Callable[[dict[str, object]], rSkillBase]] = {
    "gpu_passthrough": _make_gpu_passthrough_skill,
}
"""Registry of Skill factories. Keyed by :attr:`VLASpec.id`."""


def _make_opencv_thread_reader(cfg: SensorReaderConfig) -> SensorReader:
    """Build an :class:`OpenCVThreadSensorReader` from a :class:`SensorReaderConfig`."""
    params = cfg.backend_params
    device_param = params.get("device")
    if device_param is None:
        raise ROSConfigError(
            f"SensorReaderConfig({cfg.sensor_id!r}, backend=opencv_thread) "
            f"requires backend_params.device (int camera index or str path)."
        )
    # ``device`` is JSON-typed (str | int); ``int`` for /dev/videoN, ``str`` for paths.
    if isinstance(device_param, bool):
        raise ROSConfigError(
            f"SensorReaderConfig({cfg.sensor_id!r}).backend_params.device "
            f"must be an int or str, not a bool"
        )
    device: int | str = device_param if isinstance(device_param, int) else str(device_param)
    fps_param = params.get("fps", 30)
    fps = _to_int(fps_param, field="fps", sensor_id=cfg.sensor_id)
    width_param = params.get("width")
    height_param = params.get("height")
    width = (
        _to_int(width_param, field="width", sensor_id=cfg.sensor_id)
        if width_param is not None
        else None
    )
    height = (
        _to_int(height_param, field="height", sensor_id=cfg.sensor_id)
        if height_param is not None
        else None
    )
    return OpenCVThreadSensorReader(
        sensor_id=cfg.sensor_id,
        device=device,
        fps=fps,
        width=width,
        height=height,
        default_max_age_ms=cfg.max_age_ms,
    )


def _make_gstreamer_reader(cfg: SensorReaderConfig) -> SensorReader:
    """Build a :class:`GStreamerSensorReader` from a :class:`SensorReaderConfig`.

    The YAML may supply either a fully-formed ``pipeline`` string or a
    structured ``source / device / width / height / fps`` description that the
    pipeline builder materialises. Exactly one of these forms must be provided.
    """
    params = cfg.backend_params
    pipeline_param = params.get("pipeline")
    source_param = params.get("source")
    if (pipeline_param is None) == (source_param is None):
        raise ROSConfigError(
            f"SensorReaderConfig({cfg.sensor_id!r}, backend=gstreamer) requires "
            "exactly one of backend_params.pipeline (full GStreamer string) or "
            "backend_params.source (usb / csi / rtsp / file / testsrc); "
            f"got pipeline={'set' if pipeline_param else 'None'}, "
            f"source={'set' if source_param else 'None'}"
        )
    try:
        from openral_runner.backends.gstreamer import GStreamerSensorReader
    except ImportError as exc:
        raise ROSConfigError(
            f"SensorReaderConfig({cfg.sensor_id!r}, backend=gstreamer) requires the "
            "'gstreamer' optional-extra (pip install openral-runner[gstreamer]) "
            f"— gi import failed: {exc}"
        ) from exc

    ros_topic = cfg.publish_topic if cfg.publish_to_ros else None
    ros_rate = cfg.publish_rate_hz if cfg.publish_to_ros else None
    if cfg.publish_to_ros and not ros_topic:
        raise ROSConfigError(
            f"SensorReaderConfig({cfg.sensor_id!r}): publish_to_ros=True requires "
            "publish_topic (absolute ROS topic, e.g. /cameras/wrist_rgb/image_raw)."
        )

    if pipeline_param is not None:
        if not isinstance(pipeline_param, str):
            raise ROSConfigError(
                f"SensorReaderConfig({cfg.sensor_id!r}).backend_params.pipeline "
                f"must be a string, got {type(pipeline_param).__name__}"
            )
        return GStreamerSensorReader(
            sensor_id=cfg.sensor_id,
            pipeline=pipeline_param,
            ros_topic=ros_topic,
            ros_rate_hz=ros_rate,
            default_max_age_ms=cfg.max_age_ms,
        )

    spec = _gstreamer_spec_from_params(cfg, source_param)
    return GStreamerSensorReader(
        sensor_id=cfg.sensor_id,
        spec=spec,
        ros_topic=ros_topic,
        ros_rate_hz=ros_rate,
        default_max_age_ms=cfg.max_age_ms,
    )


def _gstreamer_spec_from_params(
    cfg: SensorReaderConfig,
    source_param: object,
) -> PipelineSpec:
    """Materialise a :class:`PipelineSpec` from a GStreamer SensorReaderConfig."""
    if not isinstance(source_param, str):
        raise ROSConfigError(
            f"SensorReaderConfig({cfg.sensor_id!r}).backend_params.source must be "
            f"a string (usb / csi / rtsp / file / testsrc), got {type(source_param).__name__}"
        )
    try:
        source = Source(source_param)
    except ValueError as exc:
        raise ROSConfigError(
            f"SensorReaderConfig({cfg.sensor_id!r}).backend_params.source="
            f"{source_param!r} is not a valid Source; "
            f"valid: {[s.value for s in Source]}"
        ) from exc
    params = cfg.backend_params
    spec_kwargs: dict[str, object] = {"source": source}
    _copy_if_present(spec_kwargs, params, "device")
    _copy_int_if_present(spec_kwargs, params, "width", cfg.sensor_id)
    _copy_int_if_present(spec_kwargs, params, "height", cfg.sensor_id)
    _copy_int_if_present(spec_kwargs, params, "fps", cfg.sensor_id)
    _copy_bool_if_present(spec_kwargs, params, "encoded")
    _copy_bool_if_present(spec_kwargs, params, "enable_nvmm")
    if cfg.publish_to_ros:
        spec_kwargs["enable_ros_tee"] = True
    try:
        return PipelineSpec(**spec_kwargs)
    except ValueError as exc:
        raise ROSConfigError(
            f"SensorReaderConfig({cfg.sensor_id!r}): invalid GStreamer spec: {exc}"
        ) from exc


def _copy_if_present(dst: dict[str, object], src: dict[str, object], key: str) -> None:
    """Copy ``src[key]`` into ``dst`` verbatim when present."""
    if (value := src.get(key)) is not None:
        dst[key] = value


def _copy_int_if_present(
    dst: dict[str, object], src: dict[str, object], key: str, sensor_id: str
) -> None:
    """Coerce ``src[key]`` to int and copy into ``dst`` when present."""
    if (value := src.get(key)) is not None:
        dst[key] = _to_int(value, field=key, sensor_id=sensor_id)


def _copy_bool_if_present(dst: dict[str, object], src: dict[str, object], key: str) -> None:
    """Coerce ``src[key]`` to bool and copy into ``dst`` when present."""
    if (value := src.get(key)) is not None:
        dst[key] = bool(value)


SENSOR_BACKEND_REGISTRY: dict[str, Callable[[SensorReaderConfig], SensorReader]] = {
    "opencv_thread": _make_opencv_thread_reader,
    "gstreamer": _make_gstreamer_reader,
}
"""Registry of SensorReader factories. Keyed by :attr:`SensorReaderConfig.backend`."""
