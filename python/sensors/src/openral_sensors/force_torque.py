"""6-axis force/torque wrist sensor — Robotiq FT 300-S.

Returns a ``SensorSpec`` with ``modality=FORCE_TORQUE`` and ``n_axes=6``.
Bandwidth and full-scale ranges live in ``metadata`` until a schema field is
added.

Example:
    >>> from openral_sensors.force_torque import robotiq_ft300s_spec
    >>> spec = robotiq_ft300s_spec(name="wrist_ft", parent_frame="ee_link")
    >>> spec.modality, spec.n_axes
    ('force_torque', 6)
"""

from __future__ import annotations

from openral_core.schemas import SensorModality, SensorSpec

from openral_sensors.catalog import CATALOG, SensorCatalogEntry

__all__ = [
    "robotiq_ft300s_spec",
]


def robotiq_ft300s_spec(
    name: str = "wrist_ft",
    parent_frame: str = "ee_link",
    rate_hz: float = 100.0,
) -> SensorSpec:
    """Build a ``SensorSpec`` for a Robotiq FT 300-S 6-axis F/T sensor."""
    return SensorSpec(
        name=name,
        modality=SensorModality.FORCE_TORQUE,
        frame_id=f"{name}_frame",
        parent_frame=parent_frame,
        rate_hz=rate_hz,
        n_axes=6,
        ros2_topic=f"/{name}/wrench",
        ros2_msg_type="geometry_msgs/WrenchStamped",
        vendor="Robotiq",
        model="FT 300-S",
        driver_pkg="robotiq_ft_sensor",
        metadata={
            "bandwidth_hz": 100.0,
            "fx_fy_max_n": 300.0,
            "fz_max_n": 300.0,
            "tx_ty_tz_max_nm": 30.0,
        },
    )


CATALOG.register_many(
    [
        SensorCatalogEntry(
            id="robotiq/ft_300s",
            vendor="robotiq",
            model="ft_300s",
            kind="sensor",
            factory=robotiq_ft300s_spec,
            modalities=(SensorModality.FORCE_TORQUE,),
            description="Robotiq FT 300-S — 6-axis F/T, 100 Hz; UR-native.",
        ),
    ]
)
