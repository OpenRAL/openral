"""Standalone OpenUSD loading for DeployScene declaration metadata."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from openral_core import (
    DeployScene,
    ObjectSurfaceModelConfig,
    PhysicsBackend,
    ROSConfigError,
    SceneSpec,
    USDObjectCatalog,
    USDObjectCatalogRef,
    load_usd_object_catalog,
)

_EXAMPLE_USD = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "tutorials"
    / "deploy"
    / "usd-object-catalog-example.usda"
)


def _scene(usd: str, expected_revision: str = "tutorial-v1") -> DeployScene:
    return DeployScene(
        scene=SceneSpec(id="usd-catalog-test", backend=PhysicsBackend.ISAACSIM),
        object_catalog=USDObjectCatalogRef(
            usd=usd,
            prim_path="/Cell/Catalog",
            expected_revision=expected_revision,
        ),
    )


def test_load_catalog_from_real_standalone_usd(tmp_path: Path) -> None:
    pytest.importorskip("pxr.Usd", reason="install optional scene-usd group")
    asset = tmp_path / "objects.usda"
    shutil.copyfile(_EXAMPLE_USD, asset)
    catalog = load_usd_object_catalog(_scene("objects.usda"), scene_path=tmp_path / "scene.yaml")

    assert catalog is not None
    assert catalog.source_revision == "tutorial-v1"
    assert catalog.units == "m"
    assert catalog.stage_frame_id == "world"
    assert catalog.objects[0].shape is not None
    assert catalog.objects[0].shape.dimension_semantics == "maximum"
    assert catalog.objects[0].workspace_ids == ("bench_top",)
    assert catalog.workspaces[0].pose_in_stage.xyz_m == (0.6, 0.0, 0.75)


def test_catalog_revision_must_match_deploy_scene(tmp_path: Path) -> None:
    pytest.importorskip("pxr.Usd", reason="install optional scene-usd group")
    shutil.copyfile(_EXAMPLE_USD, tmp_path / "objects.usda")

    with pytest.raises(ROSConfigError, match="does not match"):
        load_usd_object_catalog(
            _scene("objects.usda", expected_revision="stale-revision"),
            scene_path=tmp_path / "scene.yaml",
        )


def test_catalog_refuses_remote_file_uri(tmp_path: Path) -> None:
    with pytest.raises(ROSConfigError, match="remote file host"):
        load_usd_object_catalog(
            _scene("file://asset-host/share/catalog.usda"),
            scene_path=tmp_path / "scene.yaml",
        )


def test_catalog_can_be_authored_in_usd_root_layer_custom_data(tmp_path: Path) -> None:
    pytest.importorskip("pxr.Usd", reason="install optional scene-usd group")
    from pxr import Usd

    asset = tmp_path / "objects.usda"
    shutil.copyfile(_EXAMPLE_USD, asset)
    stage = Usd.Stage.Open(str(asset))
    assert stage is not None
    catalog_json = stage.GetPrimAtPath("/Cell/Catalog").GetCustomDataByKey(
        "openral:object_catalog_json"
    )
    stage.GetRootLayer().customLayerData = {"openral": {"object_catalog_json": catalog_json}}
    stage.GetRootLayer().Save()

    catalog = load_usd_object_catalog(
        DeployScene(
            scene=SceneSpec(id="usd-catalog-root-test", backend=PhysicsBackend.ISAACSIM),
            object_catalog=USDObjectCatalogRef(
                usd="objects.usda", prim_path="/", expected_revision="tutorial-v1"
            ),
        ),
        scene_path=tmp_path / "scene.yaml",
    )
    assert catalog is not None
    assert catalog.source_revision == "tutorial-v1"


def test_loader_does_not_open_catalog_inside_an_unloaded_payload(tmp_path: Path) -> None:
    pytest.importorskip("pxr.Usd", reason="install optional scene-usd group")
    from pxr import Usd

    payload_path = tmp_path / "payload.usda"
    payload_stage = Usd.Stage.CreateNew(str(payload_path))
    payload_stage.DefinePrim("/Asset", "Xform")
    payload_stage.DefinePrim("/Asset/Catalog", "Xform").SetCustomDataByKey(
        "openral:object_catalog_json",
        '{"schema_version":1,"source_revision":"tutorial-v1","units":"m",'
        '"stage_frame_id":"world","objects":[{"name":"box","prim_path":"/Cell/Box",'
        '"label":"box"}]}',
    )
    payload_stage.DefinePrim("/Asset/Box", "Xform")
    payload_stage.GetRootLayer().Save()

    root_path = tmp_path / "root.usda"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.DefinePrim("/Cell", "Xform").GetPayloads().AddPayload(str(payload_path), "/Asset")
    root_stage.GetRootLayer().Save()

    with pytest.raises(ROSConfigError, match="missing or unloaded"):
        load_usd_object_catalog(
            DeployScene(
                scene=SceneSpec(id="usd-payload-test", backend=PhysicsBackend.ISAACSIM),
                object_catalog=USDObjectCatalogRef(
                    usd="root.usda", prim_path="/Cell/Catalog", expected_revision="tutorial-v1"
                ),
            ),
            scene_path=tmp_path / "scene.yaml",
        )


def test_catalog_schema_rejects_duplicate_names_and_unbounded_geometry() -> None:
    base = {
        "schema_version": 1,
        "source_revision": "r1",
        "units": "m",
        "stage_frame_id": "world",
        "objects": [
            {"name": "same", "prim_path": "/Cell/A", "label": "a"},
            {"name": "same", "prim_path": "/Cell/B", "label": "b"},
        ],
    }
    with pytest.raises(ValueError, match="names and prim paths must be unique"):
        USDObjectCatalog.model_validate(base)

    with pytest.raises(ValueError):
        USDObjectCatalog.model_validate(
            {
                **base,
                "objects": [
                    {
                        "name": "large",
                        "prim_path": "/Cell/Large",
                        "label": "large",
                        "shape": {
                            "shape": {"shape": "box", "half_extents_m": [2.1, 0.1, 0.1]},
                            "dimension_semantics": "maximum",
                        },
                    }
                ],
            }
        )


def test_surface_model_bounds_are_required_and_finite() -> None:
    with pytest.raises(ValueError):
        ObjectSurfaceModelConfig(
            voxel_m=0.01,
            max_points=2048,
            max_voxels=4096,
            max_iterations=30,
            max_correspondence_m=float("inf"),
            max_rmse_m=0.02,
            min_overlap=0.1,
        )
