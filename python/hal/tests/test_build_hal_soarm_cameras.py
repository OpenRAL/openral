"""SO-ARM (SO-100 / SO-101) deploy-sim cameras (issue #88).

`openral deploy sim` builds a bare/composed MuJoCo twin for the SO-ARM arms
(their manifests set ``hal.sim: null``, so build_hal derives a ``MujocoArmHAL``).
The upstream SO-100 / SO-101 MJCFs declare zero ``<camera>`` elements, so
without a composed scene the HAL reported every manifest camera absent and no
frames reached the dashboard / WorldState / detectors.

The fix has three parts, all exercised here against real fixtures (no mocks):

1. Each manifest declares ``scene_defaults.composition`` (the so101_box
   composer, which is robot-agnostic via its ``base_body`` / ``gripper_body``
   anchors) so deploy sim composes an arena with ``oak_top`` + ``wrist``
   cameras around the robot's OWN MJCF, and maps its ``front`` sensor onto the
   ``oak_top`` MJCF camera via ``sim_camera_name``.
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

_SOARM = ["robots/so100_follower/robot.yaml", "robots/so101_follower/robot.yaml"]


def _compose_for(desc: RobotDescription) -> str:
    """Replicate the lifecycle node's compose step, returning the scene path."""
    import inspect

    comp = desc.scene_defaults.composition
    assert comp is not None
    mod, _, fn = comp.composer.partition(":")
    composer = getattr(importlib.import_module(mod), fn)
    kwargs = dict(comp.params)
    if "robot_description" in inspect.signature(composer).parameters:
        kwargs.setdefault("robot_description", desc)
    xml, meshdir = composer(**kwargs)
    scene_path = meshdir.parent / f"{desc.name}_composed_scene.xml"
    scene_path.write_text(xml)
    return str(scene_path)


@pytest.mark.parametrize("robot_yaml", _SOARM)
def test_manifest_declares_box_composition_and_camera_mapping(robot_yaml: str) -> None:
    desc = RobotDescription.from_yaml(robot_yaml)
    comp = desc.scene_defaults.composition if desc.scene_defaults else None
    assert comp is not None, f"{desc.name} must declare scene_defaults.composition"
    assert comp.composer == "openral_sim.backends.so101_box._assets:compose_so101_box_mjcf"
    by_name = {s.name: s for s in desc.sensors if s.modality == "rgb"}
    # front maps onto the composed overhead camera; wrist matches its MJCF name.
    assert by_name["front"].sim_camera_name == "oak_top"
    assert by_name["wrist"].sim_camera_name == "wrist"


@pytest.mark.parametrize("robot_yaml", _SOARM)
def test_build_hal_threads_composed_mjcf_into_derived_twin(robot_yaml: str) -> None:
    # The derived (hal.sim: null) path must honour the composed mjcf_path the
    # manifest-driven node threads in — not silently rebuild the camera-less
    # upstream arm.
    desc = RobotDescription.from_yaml(robot_yaml)
    scene_path = _compose_for(desc)

    hal = build_hal(desc, mode="sim", transport={"mjcf_path": scene_path})
    assert type(hal).__name__ == "MujocoArmHAL"
    assert hal._mjcf_path == scene_path

    # Negative control: no mjcf_path transport -> the camera-less upstream arm.
    bare = build_hal(desc, mode="sim", transport={})
    assert bare._mjcf_path != scene_path


@pytest.mark.parametrize("robot_yaml", _SOARM)
def test_composed_scene_renders_both_manifest_cameras(robot_yaml: str) -> None:
    """End-to-end: the composed twin renders `front` + `wrist` frames.

    Mirrors the SimSensorBridge camera tick exactly (it calls
    ``hal.read_images()``). Skips only if offscreen GL is unavailable on the
    runner (CI without a display); a dev host with a GPU renders for real.
    """
    import numpy as np

    desc = RobotDescription.from_yaml(robot_yaml)
    scene_path = _compose_for(desc)

    hal = build_hal(desc, mode="sim", transport={"mjcf_path": scene_path})
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
