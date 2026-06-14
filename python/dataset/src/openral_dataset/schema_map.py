"""``RobotDescription`` → LeRobot v3 ``features`` dict mapping.

A LeRobot v3.0 dataset writer needs a ``features`` schema dict at
construction time describing every column in the on-disk Parquet +
the codec/shape for every video stream. Building that dict by hand is
error-prone: the state vector shape comes from
``RobotDescription.observation_spec.state_shape``, the action dim from
``RobotDescription.action_spec.dim``, and every per-camera video key
from the ``SensorSpec.vla_feature_key`` of every sensor whose modality
is an image / depth stream.

This module exposes one pure function — :func:`features_from_robot` —
that does that mapping. It has no I/O and no lerobot dependency, which
keeps :class:`openral_dataset.LeRobotDatasetSink` testable on hosts
without lerobot installed (the sink lazy-imports lerobot; this
function does not import lerobot at all).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from openral_core import RobotDescription

__all__ = ["FeatureSpec", "features_from_robot"]


# LeRobot v3 image modality string. The string is part of the on-disk
# ``meta/info.json`` schema and must match what lerobot writes — kept
# as a module-level constant so a future lerobot rename surfaces in
# exactly one place.
_LEROBOT_IMAGE_DTYPE: Final[str] = "video"
_LEROBOT_STATE_DTYPE: Final[str] = "float32"
_LEROBOT_ACTION_DTYPE: Final[str] = "float32"

# Sensor modalities that map to a LeRobot v3 ``video`` feature. RGB and
# depth both ride a video stream; non-image sensors (lidar, force-torque)
# are intentionally skipped — they have no v3 representation today.
_IMAGE_MODALITIES: Final[frozenset[str]] = frozenset({"rgb", "depth", "rgbd", "thermal", "ir"})


@dataclass(frozen=True)
class FeatureSpec:
    """One LeRobot v3 feature entry.

    Attributes:
        key: The dotted feature name (e.g. ``"observation.state"``,
            ``"observation.images.wrist"``, ``"action"``).
        dtype: One of ``"float32"`` / ``"int64"`` / ``"bool"`` / ``"video"``
            (LeRobot's closed set).
        shape: Feature shape; ``()`` for scalars, ``(N,)`` for vectors,
            ``(H, W, 3)`` for images.
    """

    key: str
    dtype: str
    shape: tuple[int, ...]


def features_from_robot(
    robot: RobotDescription,
    *,
    fps: float,
    state_shape_override: tuple[int, ...] | None = None,
    action_dim_override: int | None = None,
    camera_shape_override: tuple[int, int] | None = None,
) -> dict[str, FeatureSpec]:
    """Build the LeRobot v3 ``features`` dict for ``robot``.

    Args:
        robot: Normative robot description. ``observation_spec.state_shape``
            and ``action_spec.dim`` are consulted when present; when absent
            or empty (the typical pre-spec robot in the catalogue per
            ADR-0007 — Franka, GR1, Sawyer, etc., where the sim-specific
            contract lives on the rSkill manifest or the scene adapter),
            the caller MUST supply ``state_shape_override`` /
            ``action_dim_override``. Sensors with image modalities and a
            ``vla_feature_key`` always contribute per-camera video features.
        fps: Recording cadence in Hz; carried alongside the feature dict
            into ``meta/info.json["fps"]`` by the sink. Validated here
            so the sink doesn't have to repeat it.
        state_shape_override: Shape of the proprioception vector when
            the robot's ``observation_spec`` is missing or empty.
            Resolved from the first frame by the sink in the typical
            sim path (see :class:`LeRobotDatasetSink._create_dataset`).
        action_dim_override: Dimensionality of the action vector when
            the robot's ``action_spec`` is missing. Same fallback story
            as ``state_shape_override``.
        camera_shape_override: Uniform ``(height, width)`` override for
            every camera-bearing sensor's feature shape. Used by the
            sim CLI to thread ``SceneSpec.observation_height/width``
            (sim renders all cameras at one scene-level resolution
            that often differs from the physical sensor's intrinsics).
            When ``None``, each camera's shape comes from its
            ``SensorSpec.intrinsics.{width,height}``.

    Returns:
        Mapping from feature name (e.g. ``"observation.state"``) to
        :class:`FeatureSpec`. Always contains the canonical
        bookkeeping features (``next.reward``, ``next.done``,
        ``next.success``, ``next.terminated``, ``next.truncated``) so
        sinks can write them unconditionally.

    Raises:
        ValueError: If ``fps <= 0`` or neither the robot's spec nor an
            override provides a usable state/action shape.

    Example:
        >>> from openral_core import RobotDescription
        >>> robot = RobotDescription.from_yaml("robots/so100_follower/robot.yaml")  # doctest: +SKIP
        >>> feats = features_from_robot(robot, fps=30.0)  # doctest: +SKIP
        >>> assert "observation.state" in feats  # doctest: +SKIP
        >>> assert "action" in feats  # doctest: +SKIP
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive; got {fps!r}")

    # Resolve state shape: explicit override > robot.observation_spec > error.
    if state_shape_override is not None:
        state_shape = tuple(int(d) for d in state_shape_override)
    elif robot.observation_spec is not None and robot.observation_spec.state_shape:
        state_shape = tuple(robot.observation_spec.state_shape)
    else:
        raise ValueError(
            f"robot {robot.name!r} has no observation_spec.state_shape; "
            "either declare one on the manifest or pass state_shape_override "
            "(the sink does this automatically from the first frame)"
        )
    if not state_shape or any(d <= 0 for d in state_shape):
        raise ValueError(
            f"robot {robot.name!r} resolved state_shape={state_shape!r}; "
            "every dimension must be > 0"
        )

    # Resolve action dim: explicit override > robot.action_spec > error.
    if action_dim_override is not None:
        action_dim = int(action_dim_override)
    elif robot.action_spec is not None and (robot.action_spec.dim or 0) > 0:
        action_dim = robot.action_spec.dim
    else:
        raise ValueError(
            f"robot {robot.name!r} has no action_spec.dim; "
            "either declare one on the manifest or pass action_dim_override"
        )
    if action_dim <= 0:
        raise ValueError(f"robot {robot.name!r} resolved action_dim={action_dim}; must be > 0")

    feats: dict[str, FeatureSpec] = {
        "observation.state": FeatureSpec(
            key="observation.state",
            dtype=_LEROBOT_STATE_DTYPE,
            shape=state_shape,
        ),
        "action": FeatureSpec(
            key="action",
            dtype=_LEROBOT_ACTION_DTYPE,
            shape=(action_dim,),
        ),
    }

    # Per-camera video features. Only sensors with both an image modality
    # AND a vla_feature_key contribute — sensors without a feature key
    # are not addressable by name in the v3 row.
    #
    # ADR-0019 / ADR-0007: per-camera shape resolution order:
    #   1. ``camera_shape_override`` (sim CLI passes
    #      ``SceneSpec.observation_height/width`` here — sim renders all
    #      cameras at a single scene-level resolution that differs from
    #      the physical sensor's intrinsics).
    #   2. ``SensorSpec.intrinsics.{width,height}`` (hardware path; the
    #      physical sensor's native resolution).
    # Channel count derives from modality: depth/ir/thermal → 1, rgb/rgbd → 3.
    for sensor in robot.sensors:
        if sensor.modality not in _IMAGE_MODALITIES:
            continue
        if sensor.vla_feature_key is None:
            continue
        if camera_shape_override is not None:
            height, width = (int(d) for d in camera_shape_override)
        else:
            if sensor.intrinsics is None:
                raise ValueError(
                    f"robot {robot.name!r} sensor {sensor.name!r} declares "
                    f"vla_feature_key={sensor.vla_feature_key!r} but has no "
                    "intrinsics; ADR-0019 requires intrinsics.{width,height} "
                    "for every bridge-bindable camera sensor (or a "
                    "camera_shape_override from the sim scene config)"
                )
            height = int(sensor.intrinsics.height)
            width = int(sensor.intrinsics.width)
        if height <= 0 or width <= 0:
            raise ValueError(
                f"robot {robot.name!r} sensor {sensor.name!r} resolved "
                f"camera shape ({width}x{height}); both dims must be > 0"
            )
        channels = 1 if sensor.modality in {"depth", "ir", "thermal"} else 3
        feats[sensor.vla_feature_key] = FeatureSpec(
            key=sensor.vla_feature_key,
            dtype=_LEROBOT_IMAGE_DTYPE,
            shape=(height, width, channels),
        )

    # Canonical bookkeeping features. LeRobot v3 writes these as columns
    # alongside the per-frame data; v3 rejects a scalar ``()`` shape so
    # they all carry shape ``(1,)`` and frames provide a 1-element
    # ndarray instead of a Python bool / float.
    feats["next.reward"] = FeatureSpec(key="next.reward", dtype="float32", shape=(1,))
    feats["next.done"] = FeatureSpec(key="next.done", dtype="bool", shape=(1,))
    feats["next.success"] = FeatureSpec(key="next.success", dtype="bool", shape=(1,))
    feats["next.terminated"] = FeatureSpec(key="next.terminated", dtype="bool", shape=(1,))
    feats["next.truncated"] = FeatureSpec(key="next.truncated", dtype="bool", shape=(1,))

    return feats
