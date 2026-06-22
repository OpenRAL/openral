"""Unit tests for the SensorCatalog registry and per-vendor factory modules.

Covers:
- SensorCatalog API (register/get/filter/build).
- Every registered vendor factory: schema round-trip + parent-frame propagation.
- openral sensor list / show CLI verbs.
"""

from __future__ import annotations

import pytest
from openral_cli.main import app
from openral_core.schemas import (
    SensorBundle,
    SensorModality,
    SensorSpec,
)
from openral_sensors import CATALOG, SensorCatalog, SensorCatalogEntry
from openral_sensors.force_torque import robotiq_ft300s_spec
from openral_sensors.realsense import (
    realsense_d435i_bundle,
)
from openral_sensors.usb_uvc import generic_uvc_rgb_spec, logitech_c920_spec
from typer.testing import CliRunner

runner = CliRunner()


# ── SensorCatalog API ─────────────────────────────────────────────────────────


def _dummy_factory() -> SensorSpec:
    return SensorSpec(
        name="x",
        modality=SensorModality.RGB,
        frame_id="x",
        rate_hz=30.0,
    )


def _dummy_entry(sensor_id: str = "acme/cam") -> SensorCatalogEntry:
    return SensorCatalogEntry(
        id=sensor_id,
        vendor="acme",
        model="cam",
        kind="sensor",
        factory=_dummy_factory,
        modalities=(SensorModality.RGB,),
        description="Test sensor.",
    )


class TestSensorCatalog:
    def test_register_and_get(self) -> None:
        cat = SensorCatalog()
        entry = cat.register(_dummy_entry())
        assert cat.get("acme/cam") is entry

    def test_contains(self) -> None:
        cat = SensorCatalog()
        cat.register(_dummy_entry())
        assert "acme/cam" in cat
        assert "missing/x" not in cat
        assert 123 not in cat  # type: ignore[operator]

    def test_len(self) -> None:
        cat = SensorCatalog()
        assert len(cat) == 0
        cat.register(_dummy_entry())
        assert len(cat) == 1

    def test_register_duplicate_raises(self) -> None:
        cat = SensorCatalog()
        cat.register(_dummy_entry())
        with pytest.raises(KeyError, match="already has an entry"):
            cat.register(_dummy_entry())

    def test_register_duplicate_replace(self) -> None:
        cat = SensorCatalog()
        cat.register(_dummy_entry())
        cat.register(_dummy_entry(), replace=True)  # no raise
        assert len(cat) == 1

    def test_get_unknown_raises(self) -> None:
        cat = SensorCatalog()
        with pytest.raises(KeyError, match="Unknown sensor id"):
            cat.get("missing/x")

    def test_unregister_idempotent(self) -> None:
        cat = SensorCatalog()
        cat.register(_dummy_entry())
        cat.unregister("acme/cam")
        cat.unregister("acme/cam")  # no raise
        assert len(cat) == 0

    def test_list_ids_sorted(self) -> None:
        cat = SensorCatalog()
        cat.register(_dummy_entry("z/late"))
        cat.register(_dummy_entry("a/early"))
        assert cat.list_ids() == ["a/early", "z/late"]

    def test_filter_by_vendor(self) -> None:
        cat = SensorCatalog()
        cat.register(_dummy_entry("acme/cam"))
        cat.register(
            SensorCatalogEntry(
                id="other/cam",
                vendor="other",
                model="cam",
                kind="sensor",
                factory=_dummy_factory,
                modalities=(SensorModality.RGB,),
            )
        )
        results = cat.filter(vendor="acme")
        assert len(results) == 1
        assert results[0].vendor == "acme"

    def test_filter_by_modality(self) -> None:
        cat = SensorCatalog()
        cat.register(_dummy_entry())
        assert cat.filter(modality=SensorModality.RGB)
        assert not cat.filter(modality=SensorModality.LIDAR_2D)

    def test_filter_by_kind(self) -> None:
        cat = SensorCatalog()
        cat.register(_dummy_entry())
        assert cat.filter(kind="sensor")
        assert not cat.filter(kind="bundle")

    def test_build_calls_factory(self) -> None:
        cat = SensorCatalog()
        cat.register(_dummy_entry())
        spec = cat.build("acme/cam")
        assert isinstance(spec, SensorSpec)
        assert spec.modality == SensorModality.RGB

    def test_build_passes_kwargs(self) -> None:
        cat = SensorCatalog()
        cat.register(
            SensorCatalogEntry(
                id="acme/cam",
                vendor="acme",
                model="cam",
                kind="sensor",
                factory=lambda name="x", parent_frame="p": SensorSpec(
                    name=name,
                    modality=SensorModality.RGB,
                    frame_id=name,
                    parent_frame=parent_frame,
                    rate_hz=30.0,
                ),
                modalities=(SensorModality.RGB,),
            )
        )
        spec = cat.build("acme/cam", name="head", parent_frame="head_link")
        assert isinstance(spec, SensorSpec)
        assert spec.name == "head"
        assert spec.parent_frame == "head_link"


# ── Global CATALOG is populated on import ─────────────────────────────────────


class TestGlobalCatalog:
    def test_has_entries(self) -> None:
        # Catalog IDs that are wired into a HAL adapter or sim scene.
        assert len(CATALOG) == 7

    def test_realsense_present(self) -> None:
        for sid in [
            "intel/realsense_d435",
            "intel/realsense_d435i",
            "intel/realsense_d415",
        ]:
            assert sid in CATALOG, f"missing {sid!r}"

    def test_per_vendor_present(self) -> None:
        for sid in [
            "intel/realsense_d435",
            "intel/realsense_d435i",
            "intel/realsense_d415",
            "generic/usb_uvc_rgb",
            "logitech/c920",
            "luxonis/oak_d_pro",
            "robotiq/ft_300s",
        ]:
            assert sid in CATALOG, f"missing {sid!r}"

    def test_ids_are_lowercase_slug(self) -> None:
        for sid in CATALOG.list_ids():
            assert sid == sid.lower(), f"id {sid!r} is not lowercase"
            assert "/" in sid, f"id {sid!r} missing vendor/model separator"
            assert " " not in sid, f"id {sid!r} contains whitespace"

    def test_every_entry_factory_round_trips(self) -> None:
        """Every catalog factory must produce a valid Pydantic model."""
        for entry in CATALOG.entries():
            obj = entry.factory()
            assert isinstance(obj, (SensorSpec, SensorBundle))
            # Round-trip through JSON to confirm schema validity.
            obj.model_validate_json(obj.model_dump_json())

    def test_kind_matches_factory_output(self) -> None:
        for entry in CATALOG.entries():
            obj = entry.factory()
            if entry.kind == "sensor":
                assert isinstance(obj, SensorSpec)
            else:
                assert isinstance(obj, SensorBundle)


# ── Per-vendor factory smoke tests ────────────────────────────────────────────


class TestRealsenseExtensions:
    def test_d435i_aliases_d435_with_renamed_model(self) -> None:
        b = realsense_d435i_bundle(name="wrist")
        assert all(s.model == "RealSense D435i" for s in b.sensors)
        assert {s.modality for s in b.sensors} == {
            SensorModality.RGB.value,
            SensorModality.DEPTH.value,
            SensorModality.IMU.value,
        }


class TestUsbUvc:
    def test_generic_uvc_catalog_id(self) -> None:
        s = generic_uvc_rgb_spec(name="wrist")
        assert s.catalog_id == "generic/usb_uvc_rgb"
        assert s.name == "wrist"
        assert s.intrinsics is not None

    def test_c920_default_resolution(self) -> None:
        s = logitech_c920_spec()
        assert s.intrinsics is not None
        assert s.intrinsics.width == 1920
        assert s.intrinsics.height == 1080

    def test_c920_fov(self) -> None:
        s = logitech_c920_spec()
        assert s.fov_h_deg == pytest.approx(78.0)

    def test_intrinsics_consistent_with_fov(self) -> None:
        """fx ≈ width / (2 * tan(hFOV/2))."""
        import math

        s = logitech_c920_spec()
        assert s.intrinsics is not None
        expected_fx = s.intrinsics.width / (2.0 * math.tan(math.radians(78.0 / 2.0)))
        assert s.intrinsics.fx == pytest.approx(expected_fx, rel=1e-3)


class TestForceTorque:
    def test_n_axes_6(self) -> None:
        assert robotiq_ft300s_spec().n_axes == 6

    def test_modality_force_torque(self) -> None:
        assert robotiq_ft300s_spec().modality == SensorModality.FORCE_TORQUE.value

    def test_ros2_msg_wrenchstamped(self) -> None:
        assert robotiq_ft300s_spec().ros2_msg_type == "geometry_msgs/WrenchStamped"

    def test_bandwidth_in_metadata(self) -> None:
        assert robotiq_ft300s_spec().metadata["bandwidth_hz"] == 100.0


# ── openral sensor CLI ─────────────────────────────────────────────────────────────


class TestSensorCli:
    def test_list_default(self) -> None:
        result = runner.invoke(app, ["sensor", "list"])
        assert result.exit_code == 0
        assert "intel/realsense_d435" in result.output

    def test_list_filter_vendor(self) -> None:
        result = runner.invoke(app, ["sensor", "list", "--vendor", "intel"])
        assert result.exit_code == 0
        assert "intel/realsense_d435" in result.output
        assert "logitech/c920" not in result.output

    def test_list_filter_modality_empty(self) -> None:
        # No catalog entries claim lidar_2d after the cleanup.
        result = runner.invoke(app, ["sensor", "list", "--modality", "lidar_2d"])
        assert result.exit_code == 0
        assert "No sensors match" in result.output

    def test_list_filter_kind_bundle(self) -> None:
        result = runner.invoke(app, ["sensor", "list", "--kind", "bundle"])
        assert result.exit_code == 0
        assert "intel/realsense_d435" in result.output
        assert "logitech/c920" not in result.output

    def test_list_invalid_modality_exits_1(self) -> None:
        result = runner.invoke(app, ["sensor", "list", "--modality", "bogus"])
        assert result.exit_code == 1
        assert "Unknown" in result.output

    def test_list_invalid_kind_exits_1(self) -> None:
        result = runner.invoke(app, ["sensor", "list", "--kind", "bogus"])
        assert result.exit_code == 1

    def test_list_json_output(self) -> None:
        import json

        result = runner.invoke(app, ["sensor", "list", "--vendor", "intel", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert all("id" in row and "modalities" in row for row in data)

    def test_list_no_match_message(self) -> None:
        result = runner.invoke(app, ["sensor", "list", "--vendor", "no_such_vendor"])
        assert result.exit_code == 0
        assert "No sensors match" in result.output

    def test_show_known(self) -> None:
        result = runner.invoke(app, ["sensor", "show", "robotiq/ft_300s"])
        assert result.exit_code == 0
        assert "ft_300s" in result.output
        assert "force_torque" in result.output

    def test_show_unknown_exits_1(self) -> None:
        result = runner.invoke(app, ["sensor", "show", "no_such/sensor"])
        assert result.exit_code == 1
        assert "Unknown sensor id" in result.output

    def test_show_json_parses(self) -> None:
        import json

        result = runner.invoke(
            app, ["sensor", "show", "robotiq/ft_300s", "--json", "--name", "wrist"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["name"] == "wrist"
        assert payload["modality"] == "force_torque"

    def test_show_passes_name_and_parent_frame(self) -> None:
        result = runner.invoke(
            app,
            [
                "sensor",
                "show",
                "intel/realsense_d435",
                "--name",
                "head",
                "--parent-frame",
                "head_link",
            ],
        )
        assert result.exit_code == 0
        assert "head" in result.output
        assert "head_link" in result.output

    def test_help_lists_sensor(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "sensor" in result.output
