"""Real tracker observations gate queries on calibrated declared-size mismatch."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import openral_core.usd_catalog as usd_catalog_module
import pytest
from openral_core import (
    DeployScene,
    ObjectQuery,
    ObjectQueryStatus,
    ObjectRole,
    PhysicsBackend,
    ROSConfigError,
    SceneSpec,
    TrackStatus,
    USDObjectCatalog,
    USDObjectCatalogRef,
    load_usd_object_catalog,
)
from openral_world_state.object_tracker import CameraFrame, ObjectTracker, TrackerParams
from openral_world_state.object_tracker.catalog import CatalogDeclarations
from openral_world_state.object_tracker.grounding import QueryEscalation

from tests.unit._tracker_scenes import grid_from_depth, render, so101_top, sphere, step

_FAKES = Path(__file__).resolve().parent / "fakes"


def _catalog_scene(source: Path, *, reader_python: str | None = None) -> DeployScene:
    return DeployScene(
        scene=SceneSpec(id="usd-orange-tutorial", backend=PhysicsBackend.ISAACSIM),
        object_catalog=USDObjectCatalogRef(
            usd=str(source),
            prim_path="/Cell/Catalog",
            expected_revision="tutorial-v1",
            reader_python=reader_python,
        ),
    )


def test_usd_catalog_external_reader_uses_real_standalone_usd_python() -> None:
    pytest.importorskip("pxr.Usd", reason="requires scene-usd dependency")
    source = (
        Path(__file__).resolve().parents[2]
        / "docs/tutorials/deploy/usd-object-catalog-example.usda"
    )
    inline = load_usd_object_catalog(_catalog_scene(source), scene_path=source.with_suffix(".yaml"))
    external = load_usd_object_catalog(
        _catalog_scene(source, reader_python=sys.executable),
        scene_path=source.with_suffix(".yaml"),
    )
    assert inline is not None
    assert external == inline


def test_usd_catalog_external_reader_rejects_missing_executable() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "docs/tutorials/deploy/usd-object-catalog-example.usda"
    )
    with pytest.raises(ROSConfigError, match="must name an executable file"):
        load_usd_object_catalog(
            _catalog_scene(source, reader_python="/missing/usd-python"),
            scene_path=source.with_suffix(".yaml"),
        )


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-specific")
def test_usd_catalog_external_reader_timeout_kills_wrapper_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "docs/tutorials/deploy/usd-object-catalog-example.usda"
    )
    started = tmp_path / "child-started"
    survived = tmp_path / "child-survived"
    monkeypatch.setenv("OPENRAL_USD_READER_CHILD_STARTED", str(started))
    monkeypatch.setenv("OPENRAL_USD_READER_CHILD_SURVIVED", str(survived))
    monkeypatch.setattr(usd_catalog_module, "_READER_TIMEOUT_S", 1.0)
    with pytest.raises(ROSConfigError, match="timed out"):
        load_usd_object_catalog(
            _catalog_scene(source, reader_python=str(_FAKES / "usd_catalog_timeout_wrapper.py")),
            scene_path=source.with_suffix(".yaml"),
        )
    time.sleep(1.2)
    assert started.exists()
    assert not survived.exists()


def test_usd_catalog_external_reader_rejects_bad_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "docs/tutorials/deploy/usd-object-catalog-example.usda"
    )
    monkeypatch.setattr(
        usd_catalog_module, "_USD_READER_HELPER", _FAKES / "usd_catalog_bad_protocol.py"
    )
    with pytest.raises(ROSConfigError, match="expected a JSON object"):
        load_usd_object_catalog(
            _catalog_scene(source, reader_python=sys.executable),
            scene_path=source.with_suffix(".yaml"),
        )


@pytest.mark.parametrize(
    "mode", ["maximum", "no_calibration", "nominal", "duplicate", "undeclared"]
)
def test_actual_depth_observations_only_block_calibrated_unique_maximum(mode: str) -> None:
    pytest.importorskip("pxr.Usd", reason="requires scene-usd dependency")
    source = (
        Path(__file__).resolve().parents[2]
        / "docs/tutorials/deploy/usd-object-catalog-example.usda"
    )
    catalog = load_usd_object_catalog(
        DeployScene(
            scene=SceneSpec(id="usd-orange-tutorial", backend=PhysicsBackend.ISAACSIM),
            object_catalog=USDObjectCatalogRef(
                usd=str(source), prim_path="/Cell/Catalog", expected_revision="tutorial-v1"
            ),
        ),
        scene_path=source.with_suffix(".yaml"),
    )
    assert catalog is not None
    rig = so101_top()
    data = catalog.model_dump()
    # Fixture bounds describe exact analytic depth; they are not physical calibration.
    data["measurement_calibration"] = {
        "point_error_bound_m": 0.001,
        "calibration_reference": "analytic-sphere-raycast",
        "sensor_ids": [rig.name],
        "max_age_s": 0.1,
    }
    # The rendered orange is deliberately much larger than this declared maximum.
    data["objects"][0]["shape"]["shape"]["radius_m"] = 0.005
    if mode == "no_calibration":
        data["measurement_calibration"] = None
    elif mode == "nominal":
        data["objects"][0]["shape"]["dimension_semantics"] = "nominal"
    elif mode == "duplicate":
        second = dict(data["objects"][0])
        second.update(name="orange_second", prim_path="/Cell/OrangeSecond")
        data["objects"] = [data["objects"][0], second]
    tracker = ObjectTracker(rig.frame, TrackerParams(search_start_s=5.0, search_max_s=60.0))
    if mode != "undeclared":
        tracker.catalog_declarations = CatalogDeclarations(USDObjectCatalog.model_validate(data))
    tracker.open_query(
        "orange-goal",
        "orange-query",
        ObjectQuery(label="orange", count="one", role=ObjectRole.TARGET),
        "obj:orange@0",
        stamp_ns=0,
    )
    orange = sphere((*rig.centre_xy, rig.support_z + 0.04), 0.04)
    events: list[QueryEscalation] = []
    wall = 0.0
    for index in range(3):
        output, wall = step(
            tracker, rig, [orange], stamp_ns=(index + 1) * 500_000_000, wall_s=wall + 1
        )
        events.extend(output.escalations)
    query = tracker.queries.get("orange-goal", "orange-query")
    assert query is not None
    assert output.tracked.objects
    if mode == "maximum":
        assert query.status is ObjectQueryStatus.UNRESOLVED
        assert [event.kind for event in events] == ["unresolved"]
        assert any(
            observation.reason.startswith("catalog_dimension_mismatch")
            for observation in query.observations
        )
        assert not tracker.queries.ready("orange-goal")

        # A later lifecycle generation on a grid-only feed must still see the latch.
        track = tracker.store.all()[0]
        tracker.store.replace(
            track.model_copy(update={"status": TrackStatus.CONFIRMED, "epoch": track.epoch + 1})
        )
        depth, _ = render([orange], rig)
        stamp_ns = 2_000_000_000
        frame = CameraFrame(
            rig.name, stamp_ns, depth, rig.k, rig.t, np.zeros(depth.shape, dtype=np.uint8)
        )
        resumed = tracker.feed(
            frame,
            grid=grid_from_depth(depth, rig),
            grid_stamp_ns=stamp_ns,
            wall_s=wall + 1.0,
        )
        [reconfirmed] = resumed.tracked.objects
        assert reconfirmed.epoch == track.epoch + 1
        assert reconfirmed.status is TrackStatus.CONFIRMED
        assert query.status is ObjectQueryStatus.UNRESOLVED
        assert not tracker.queries.ready("orange-goal")
        assert any(
            observation.reason.startswith("catalog_dimension_mismatch")
            for observation in query.observations
        )
    else:
        assert query.status is ObjectQueryStatus.RESOLVED
        assert tracker.queries.ready("orange-goal")
        assert events == []
        assert query.observations == []
