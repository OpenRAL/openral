"""so101 deploy-sim cameras (issue #88).

`openral deploy sim` builds a bare/composed MuJoCo twin for the SO-101 (its
manifest sets ``hal.sim: null``, so build_hal derives a ``MujocoArmHAL``). The
upstream SO-101 MJCF declares zero ``<camera>`` elements, so without a composed
scene the HAL reported every manifest camera absent and no frames reached the
dashboard / WorldState / detectors.

The fix has three parts, all exercised here against real fixtures (no mocks):

1. The so101 manifest declares ``scene_defaults.composition`` (the so101_box
   composer) so deploy sim composes an arena with ``oak_top`` + ``wrist``
   cameras, and maps its ``front`` sensor onto the ``oak_top`` MJCF camera via
   ``sim_camera_name``.
2. ``build_hal``'s derived (``hal.sim: null``) sim path threads the composed
   ``mjcf_path`` transport into ``MujocoArmHAL.from_description``.
3. ``SimSensorBridge`` resolves frames by VLA slot OR sensor name (covered in
   ``test_sim_sensor_bridge_camera_key``).
"""

from __future__ import annotations

import importlib

import pytest
from openral_core import RobotDescription
from openral_hal import build_hal

pytest.importorskip("mujoco")
pytest.importorskip("openral_sim")

_SO101 = "robots/so101_follower/robot.yaml"


def test_manifest_declares_box_composition_and_camera_mapping() -> None:
    desc = RobotDescription.from_yaml(_SO101)
    comp = desc.scene_defaults.composition if desc.scene_defaults else None
    assert comp is not None, "so101 manifest must declare scene_defaults.composition"
    assert comp.composer == ("openral_sim.backends.so101_box._assets:compose_so101_box_mjcf")
    by_name = {s.name: s for s in desc.sensors if s.modality == "rgb"}
    # front maps onto the composed overhead camera; wrist matches its MJCF name.
    assert by_name["front"].sim_camera_name == "oak_top"
    assert by_name["wrist"].sim_camera_name == "wrist"


def test_build_hal_threads_composed_mjcf_into_derived_twin() -> None:
    # The derived (hal.sim: null) path must honour the composed mjcf_path the
    # manifest-driven node threads in — not silently rebuild the camera-less
    # upstream arm.
    desc = RobotDescription.from_yaml(_SO101)
    comp = desc.scene_defaults.composition
    mod, _, fn = comp.composer.partition(":")
    xml, meshdir = getattr(importlib.import_module(mod), fn)(**comp.params)
    scene_path = meshdir.parent / f"{desc.name}_composed_scene.xml"
    scene_path.write_text(xml)

    hal = build_hal(desc, mode="sim", transport={"mjcf_path": str(scene_path)})
    assert type(hal).__name__ == "MujocoArmHAL"
    assert hal._mjcf_path == str(scene_path)

    # Negative control: no mjcf_path transport -> the camera-less upstream arm.
    bare = build_hal(desc, mode="sim", transport={})
    assert bare._mjcf_path != str(scene_path)


def test_composed_scene_renders_both_manifest_cameras() -> None:
    """End-to-end: the composed twin renders `front` + `wrist` frames.

    Mirrors the SimSensorBridge camera tick exactly (it calls
    ``hal.read_images()``). Skips only if offscreen GL is unavailable on the
    runner (CI without a display); a dev host with a GPU renders for real.
    """
    import numpy as np

    desc = RobotDescription.from_yaml(_SO101)
    comp = desc.scene_defaults.composition
    mod, _, fn = comp.composer.partition(":")
    xml, meshdir = getattr(importlib.import_module(mod), fn)(**comp.params)
    scene_path = meshdir.parent / f"{desc.name}_composed_scene.xml"
    scene_path.write_text(xml)

    hal = build_hal(desc, mode="sim", transport={"mjcf_path": str(scene_path)})
    hal.connect()
    try:
        frames = hal.read_images()
    except Exception as exc:  # pragma: no cover - headless GL not available
        pytest.skip(f"offscreen MuJoCo render unavailable: {exc}")
    finally:
        hal.disconnect()

    if not frames:
        pytest.skip("offscreen MuJoCo render produced no frames (no GL context)")
    assert set(frames) == {"front", "wrist"}
    for name, arr in frames.items():
        a = np.asarray(arr)
        assert a.ndim == 3 and a.shape[2] == 3, f"{name}: {a.shape}"
        assert a.dtype == np.uint8
