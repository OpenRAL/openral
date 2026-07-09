"""Helpers that resolve sensor catalog ids into ``SensorSpec`` / ``SensorBundle``.

Used by the per-robot ``*_with_sensors(...)`` factories in the HAL package to
embed catalog-built sensor specs into a deep copy of the canonical
``RobotDescription``.

The resolution path is option (a) of issue #23 — embed the materialised
``SensorSpec`` / ``SensorBundle`` directly.  The provenance-preserving
``catalog:`` reference field is deferred to a separate v0.3 schema ADR.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openral_core.schemas import RobotDescription, SensorBundle, SensorSpec
from openral_sensors import CATALOG

__all__ = ["SensorRequest", "with_sensors"]


SensorRequest = str | tuple[str, Mapping[str, Any]]
"""Either a bare catalog id (``"intel/realsense_d435"``) or an
``(id, kwargs)`` pair forwarding kwargs to the catalog factory."""


def with_sensors(
    description: RobotDescription,
    requests: Sequence[SensorRequest],
    *,
    default_parent_frame: str | None = None,
) -> RobotDescription:
    """Return a deep copy of ``description`` with catalog-resolved sensors attached.

    Args:
        description: Canonical ``RobotDescription`` (sensors/bundles empty).
        requests: Sequence of catalog requests.  Each entry is either a bare
            id (``"intel/realsense_d435"``) — in which case the factory is
            called with ``parent_frame=default_parent_frame`` — or an
            ``(id, kwargs)`` pair, in which case ``kwargs`` is forwarded
            verbatim to the catalog factory and overrides the default.
        default_parent_frame: tf2 frame to pass as ``parent_frame`` when a
            request does not specify one.  Defaults to
            ``description.base_frame``.

    Returns:
        A new ``RobotDescription`` (Pydantic deep-copy) with the resolved
        sensors and bundles appended to the existing fields.

    Raises:
        KeyError: If a catalog id is unknown.

    Example:
        >>> from openral_hal.so100_follower import SO100_DESCRIPTION
        >>> from openral_hal._sensor_wiring import with_sensors
        >>> desc = with_sensors(SO100_DESCRIPTION, ["logitech/c920"])
        >>> desc.sensors[0].vendor
        'Logitech'
    """
    parent = default_parent_frame or description.base_frame

    new_sensors: list[SensorSpec] = list(description.sensors)
    new_bundles: list[SensorBundle] = list(description.sensor_bundles)

    for request in requests:
        if isinstance(request, str):
            sensor_id, kwargs = request, {"parent_frame": parent}
        else:
            sensor_id, raw = request
            kwargs = {"parent_frame": parent, **dict(raw)}

        built = CATALOG.build(sensor_id, **kwargs)
        if isinstance(built, SensorBundle):
            new_bundles.append(built)
        else:
            new_sensors.append(built)

    return description.model_copy(
        update={"sensors": new_sensors, "sensor_bundles": new_bundles},
        deep=True,
    )
