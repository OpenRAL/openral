"""Tests for catalog-backed sensor wiring on canonical RobotDescriptions (issue #23).

Covers the four canonical robot descriptions plus the shared
``with_sensors`` helper.
"""

from __future__ import annotations

import pytest
from openral_core.schemas import RobotDescription, SensorBundle, SensorSpec
from openral_hal import (
    FRANKA_PANDA_DESCRIPTION,
    SO100_DESCRIPTION,
    UR5e_DESCRIPTION,
    UR10e_DESCRIPTION,
    franka_panda_with_sensors,
    so100_with_sensors,
    ur5e_with_sensors,
    ur10e_with_sensors,
)
from openral_hal._sensor_wiring import with_sensors
from openral_sensors import CATALOG

# ── Canonical descriptions remain empty (no behaviour change) ─────────────────


@pytest.mark.parametrize(
    "description",
    [SO100_DESCRIPTION, FRANKA_PANDA_DESCRIPTION, UR5e_DESCRIPTION, UR10e_DESCRIPTION],
)
def test_canonical_description_has_no_sensors(description: RobotDescription) -> None:
    assert description.sensors == []
    assert description.sensor_bundles == []


# ── SO-100 default loadout ────────────────────────────────────────────────────


class TestSO100WithSensors:
    def test_default_is_logitech_c920(self) -> None:
        desc = so100_with_sensors()
        assert len(desc.sensors) == 1
        assert desc.sensors[0].vendor == "Logitech"
        assert desc.sensors[0].model == "C920"

    def test_parent_frame_uses_base_frame(self) -> None:
        desc = so100_with_sensors()
        assert desc.sensors[0].parent_frame == SO100_DESCRIPTION.base_frame

    def test_explicit_empty_list(self) -> None:
        desc = so100_with_sensors([])
        assert desc.sensors == []
        assert desc.sensor_bundles == []

    def test_does_not_mutate_canonical(self) -> None:
        _ = so100_with_sensors()
        assert SO100_DESCRIPTION.sensors == []


@pytest.mark.parametrize(
    "manifest_path, expected_catalog_ids",
    [
        ("robots/so100_follower/robot.yaml", [None, "generic/usb_uvc_rgb"]),
        ("robots/so101_follower/robot.yaml", [None, "generic/usb_uvc_rgb"]),
    ],
)
def test_robot_manifest_sensors_carry_catalog_provenance(
    manifest_path: str, expected_catalog_ids: list[str]
) -> None:
    desc = RobotDescription.from_yaml(manifest_path)

    assert [sensor.catalog_id for sensor in desc.sensors] == expected_catalog_ids
    for sensor in desc.sensors:
        if sensor.catalog_id is not None:
            assert sensor.catalog_id in CATALOG


# ── Franka Panda default loadout ──────────────────────────────────────────────


class TestFrankaWithSensors:
    def test_default_is_realsense_d435i(self) -> None:
        desc = franka_panda_with_sensors()
        assert len(desc.sensor_bundles) == 1
        bundle = desc.sensor_bundles[0]
        assert bundle.sensors[0].vendor == "Intel"
        assert "D435i" in bundle.sensors[0].model

    def test_parent_frame(self) -> None:
        desc = franka_panda_with_sensors()
        for s in desc.sensor_bundles[0].sensors:
            assert s.parent_frame == FRANKA_PANDA_DESCRIPTION.base_frame


# ── UR loadouts ───────────────────────────────────────────────────────────────


class TestURWithSensors:
    def test_ur5e_default(self) -> None:
        desc = ur5e_with_sensors()
        # D415 bundle + Robotiq FT spec
        assert len(desc.sensor_bundles) == 1
        assert desc.sensor_bundles[0].sensors[0].vendor == "Intel"
        assert any(s.vendor == "Robotiq" for s in desc.sensors)

    def test_ur10e_default(self) -> None:
        desc = ur10e_with_sensors()
        assert len(desc.sensor_bundles) == 1
        assert "D435" in desc.sensor_bundles[0].sensors[0].model
        assert any(s.vendor == "Robotiq" for s in desc.sensors)


# ── with_sensors helper ───────────────────────────────────────────────────────


class TestWithSensorsHelper:
    def test_kwargs_forwarded_to_factory(self) -> None:
        desc = with_sensors(
            SO100_DESCRIPTION,
            [("logitech/c920", {"name": "scene_cam", "rate_hz": 60.0})],
        )
        sensor = desc.sensors[0]
        assert sensor.name == "scene_cam"
        assert sensor.rate_hz == 60.0

    def test_default_parent_frame_override(self) -> None:
        desc = with_sensors(
            SO100_DESCRIPTION, ["logitech/c920"], default_parent_frame="custom_frame"
        )
        assert desc.sensors[0].parent_frame == "custom_frame"

    def test_unknown_id_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown sensor id"):
            with_sensors(SO100_DESCRIPTION, ["nonexistent/foo"])

    def test_returns_distinct_pydantic_instance(self) -> None:
        desc = with_sensors(SO100_DESCRIPTION, ["logitech/c920"])
        assert isinstance(desc, RobotDescription)
        assert desc is not SO100_DESCRIPTION

    def test_appends_to_existing(self) -> None:
        first = with_sensors(SO100_DESCRIPTION, ["logitech/c920"])
        second = with_sensors(first, ["intel/realsense_d415"])
        assert len(second.sensors) == 1
        assert len(second.sensor_bundles) == 1
        assert isinstance(second.sensors[0], SensorSpec)
        assert isinstance(second.sensor_bundles[0], SensorBundle)
