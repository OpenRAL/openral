"""Maximum-size contradiction checks using the external USD tutorial catalog."""

from pathlib import Path

import pytest
from openral_core import (
    CatalogMeasurementCalibration,
    DeployScene,
    PhysicsBackend,
    SceneSpec,
    USDObjectCatalog,
    USDObjectCatalogRef,
    load_usd_object_catalog,
)
from openral_world_state.object_tracker.catalog import CatalogDeclarations
from pydantic import ValidationError


@pytest.fixture
def catalog() -> USDObjectCatalog:
    pytest.importorskip("pxr.Usd", reason="requires scene-usd optional dependency")
    source = (
        Path(__file__).resolve().parents[2]
        / "docs/tutorials/deploy/usd-object-catalog-example.usda"
    )
    result = load_usd_object_catalog(
        DeployScene(
            scene=SceneSpec(id="usd-catalog-tutorial", backend=PhysicsBackend.ISAACSIM),
            object_catalog=USDObjectCatalogRef(
                usd=str(source), prim_path="/Cell/Catalog", expected_revision="tutorial-v1"
            ),
        ),
        scene_path=source.with_suffix(".yaml"),
    )
    assert result is not None
    # These are fixture-specific synthetic sensor bounds, not deploy calibration.
    return USDObjectCatalog.model_validate(
        result.model_dump()
        | {
            "measurement_calibration": {
                "point_error_bound_m": 0.005,
                "calibration_reference": "tutorial-recorded-depth-bound",
                "sensor_ids": ["head_depth"],
                "max_age_s": 0.2,
            }
        }
    )


def test_partial_compatible_observations_cannot_clear_mismatch(catalog: USDObjectCatalog) -> None:
    declarations = CatalogDeclarations(catalog)
    assert (
        declarations.associate("orange@0", 0, " ORANGE ", observed_label_count=1)
        == "catalog_unverified"
    )
    assert (
        declarations.observe_span("orange@0", 0, 0.5, camera="head_depth", stamp_ns=10, now_ns=10)
        == "catalog_dimension_mismatch"
    )
    assert declarations.blocked("orange@0", 0)
    assert "span_m=0.5" in declarations.diagnostics()["orange@0"]
    assert "limit_m=0.09" in declarations.diagnostics()["orange@0"]
    assert (
        declarations.observe_span("orange@0", 0, 0.1, camera="head_depth", stamp_ns=11, now_ns=11)
        == "catalog_dimension_mismatch"
    )
    assert (
        declarations.observe_span("orange@0", 0, 0.1, camera="head_depth", stamp_ns=11, now_ns=11)
        == "catalog_stale_observation"
    )
    assert declarations.blocked("orange@0", 0)
    assert (
        declarations.observe_span("orange@0", 0, 0.1, camera="head_depth", stamp_ns=12, now_ns=12)
        == "catalog_dimension_mismatch"
    )
    assert declarations.blocked("orange@0", 0)


def test_missing_calibration_and_unsupported_or_stale_sensor_are_unknown(
    catalog: USDObjectCatalog,
) -> None:
    declarations = CatalogDeclarations(catalog)
    declarations.associate("orange@0", 0, "orange", observed_label_count=1)
    for camera, stamp, now, reason in [
        ("wrist_depth", 10, 10, "catalog_unsupported_sensor"),
        ("head_depth", 0, 0, "catalog_stale_observation"),
        ("head_depth", 11, 10, "catalog_stale_observation"),
        ("head_depth", 10, 300_000_010, "catalog_stale_observation"),
    ]:
        assert (
            declarations.observe_span("orange@0", 0, 1.0, camera=camera, stamp_ns=stamp, now_ns=now)
            == reason
        )
        assert not declarations.blocked("orange@0", 0)
    uncalibrated = CatalogDeclarations(catalog.model_copy(update={"measurement_calibration": None}))
    uncalibrated.associate("orange@0", 0, "orange", observed_label_count=1)
    assert (
        uncalibrated.observe_span("orange@0", 0, 1.0, camera="head_depth", stamp_ns=10, now_ns=10)
        == "catalog_uncalibrated"
    )


def test_ambiguity_suspends_checks_but_keeps_mismatch(catalog: USDObjectCatalog) -> None:
    declarations = CatalogDeclarations(catalog)
    declarations.associate("orange@0", 0, "orange", observed_label_count=1)
    declarations.observe_span("orange@0", 0, 0.5, camera="head_depth", stamp_ns=10, now_ns=10)
    assert (
        declarations.associate("orange@0", 0, "orange", observed_label_count=2)
        == "catalog_ambiguous"
    )
    assert (
        declarations.observe_span("orange@0", 0, 0.1, camera="head_depth", stamp_ns=11, now_ns=11)
        == "catalog_unassociated"
    )
    assert declarations.blocked("orange@0", 0)
    assert (
        declarations.associate("orange@0", 0, "orange", observed_label_count=1)
        == "catalog_dimension_mismatch"
    )
    assert (
        declarations.associate("orange@0", 1, "orange", observed_label_count=1)
        == "catalog_dimension_mismatch"
    )
    assert declarations.blocked("orange@0", 1)
    assert not declarations.blocked("orange@0", 0)
    assert (
        declarations.associate("orange@0", 0, "orange", observed_label_count=1)
        == "catalog_stale_generation"
    )
    declarations.retain(set())
    assert declarations.diagnostics() == {}


def test_nominal_dimensions_never_disprove_partial_surface(catalog: USDObjectCatalog) -> None:
    data = catalog.model_dump()
    data["objects"][0]["shape"]["dimension_semantics"] = "nominal"
    declarations = CatalogDeclarations(USDObjectCatalog.model_validate(data))
    declarations.associate("orange@0", 0, "orange", observed_label_count=1)
    assert (
        declarations.observe_span("orange@0", 0, 1.0, camera="head_depth", stamp_ns=10, now_ns=10)
        == "catalog_no_maximum"
    )
    assert not declarations.blocked("orange@0", 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("point_error_bound_m", float("nan")),
        ("max_age_s", float("inf")),
        ("sensor_ids", ["head_depth", "head_depth"]),
        ("calibration_reference", " "),
    ],
)
def test_calibration_requires_explicit_finite_bound(field: str, value: object) -> None:
    data = {
        "point_error_bound_m": 0.005,
        "calibration_reference": "tutorial-depth-bound",
        "sensor_ids": ["head_depth"],
        "max_age_s": 0.2,
    }
    with pytest.raises(ValidationError):
        CatalogMeasurementCalibration.model_validate(data | {field: value})


def test_duplicate_declared_labels_are_not_identity_proof(catalog: USDObjectCatalog) -> None:
    data = catalog.model_dump()
    duplicate = dict(data["objects"][0])
    duplicate.update(name="orange_second", prim_path="/Cell/OrangeSecond")
    data["objects"] = [data["objects"][0], duplicate]
    declarations = CatalogDeclarations(USDObjectCatalog.model_validate(data))
    assert (
        declarations.associate("orange@0", 0, "orange", observed_label_count=1)
        == "catalog_ambiguous"
    )
    assert (
        declarations.observe_span("orange@0", 0, 1.0, camera="head_depth", stamp_ns=10, now_ns=10)
        == "catalog_unassociated"
    )


def test_calibrated_uncertainty_boundary_is_inclusive(catalog: USDObjectCatalog) -> None:
    declarations = CatalogDeclarations(catalog)
    declarations.associate("orange@0", 0, "orange", observed_label_count=1)
    threshold = 2 * 0.04 + 2 * 0.005
    assert (
        declarations.observe_span(
            "orange@0", 0, threshold, camera="head_depth", stamp_ns=10, now_ns=10
        )
        == "catalog_span_compatible"
    )
    assert (
        declarations.observe_span(
            "orange@0", 0, threshold + 0.001, camera="head_depth", stamp_ns=11, now_ns=11
        )
        == "catalog_dimension_mismatch"
    )
