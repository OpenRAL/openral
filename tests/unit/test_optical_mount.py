# SPDX-License-Identifier: Apache-2.0
"""``optical_mount``: one rule for the frame a camera's pixels are in, and who publishes it.

Issue #641 (the #627 class of bug): the object lift looks a camera up in
``image_frame_id``; the DA3 resolver names a cloud frame; the sim bridge and the real
deploy launch publish the REP-103 optical child. All four read ``optical_mount``.
``MOUNTS`` pins that rule's answer on every in-tree manifest by hand, so a change to
the rule shows here; the other tests hold each consumer and publisher to it. The
launch's half is in ``test_deploy_e2e_camera_wiring.py`` (it needs the ROS launch
modules), the bridge's on a real node in ``test_isaac_sim_sidecar_wire.py``.

Real fixtures (CLAUDE.md §1.11): every ``robots/*/robot.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.depth_extrinsic import (
    depth_extrinsic_problems,
    image_frame_id,
    optical_mount,
    resolve_estimated_depth_camera,
)
from openral_core.schemas import SensorSpec
from openral_hal.sim_sensor_bridge import _static_optical_mount

_REPO = Path(__file__).resolve().parents[2]
ROBOTS = sorted(p.parent.name for p in (_REPO / "robots").glob("*/robot.yaml"))
#: Every in-tree RGB camera that is placed through a mount, written out by hand:
#: ``{robot: {camera: (mount sensor, mount frame, optical frame)}}``. Every other RGB
#: camera keeps its own ``frame_id`` (a robot link or a sim bridge's live optical frame).
MOUNTS: dict[str, dict[str, tuple[str, str, str]]] = {
    "makermods_metal": {"top": ("top", "top_camera", "top_camera_optical_frame")},
    "makermods_metal_station": {"top": ("top", "top_camera", "top_camera_optical_frame")},
    "openamrobot_v2": {"top": ("head_zed", "zed_camera_link", "zed_camera_link_optical_frame")},
    "openarm": {"top": ("head_zed", "zed_camera_link", "zed_camera_link_optical_frame")},
}


def lift_sensors(robot: str) -> list[SensorSpec]:
    """The sensors the object lift resolves cameras over: the manifest's and its bundles'."""
    desc = RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml"))
    return [*desc.sensors, *(s for b in desc.sensor_bundles for s in b.sensors)]


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_rule_on_every_manifest(robot: str) -> None:
    """``optical_mount`` and ``image_frame_id`` against the hand-written table."""
    sensors = lift_sensors(robot)
    want = MOUNTS.get(robot, {})
    assert set(want) <= {s.name for s in sensors}
    for spec in sensors:
        if spec.modality != "rgb":
            continue
        placed = optical_mount(spec, sensors)
        if spec.name not in want:
            assert placed is None, spec.name
            assert image_frame_id(spec, sensors) == spec.frame_id
            continue
        assert placed is not None, spec.name
        assert (placed.sensor.name, placed.frame, placed.optical) == want[spec.name]
        assert image_frame_id(spec, sensors) == want[spec.name][2]


@pytest.mark.parametrize("robot", sorted(MOUNTS))
def test_the_sim_bridge_latches_the_frame_the_lift_reads(robot: str) -> None:
    """Over the bridge's own sensor set (the manifest's): on MuJoCo it latches each
    mounted camera's optical child; on a backend that places depth cameras itself
    (Isaac) it leaves a depth sensor's mount to the depth leg's live TF."""
    sensors = RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml")).sensors
    for name, (mount, frame, optical) in MOUNTS[robot].items():
        spec = next(s for s in sensors if s.name == name)
        latched = _static_optical_mount(spec, sensors, backend_places_depth=False)
        assert latched is not None and latched[1:] == (frame, optical)
        mount_is_depth = next(s for s in sensors if s.name == mount).is_depth_camera
        isaac = _static_optical_mount(spec, sensors, backend_places_depth=True)
        assert (isaac is None) == mount_is_depth
        assert isaac is None or isaac[1:] == (frame, optical)


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("hal_mode", ["sim", "real"])
def test_the_da3_cloud_frame_is_the_image_frame(robot: str, hal_mode: str) -> None:
    """Wherever the DA3 resolver places a camera, its cloud is in the frame the camera's
    pixels are in, hung under the mount of the hand-written table."""
    sensors = lift_sensors(robot)
    for spec in sensors:
        if spec.modality != "rgb":
            continue
        da3 = resolve_estimated_depth_camera(sensors, spec.name, hal_mode=hal_mode)
        want = MOUNTS.get(robot, {}).get(spec.name)
        if want is None:
            # No mount: placed only in sim, by the bridge's live optical frame.
            assert da3.optical_parent is None
            assert da3.cloud_frame in ("", spec.frame_id)
            assert hal_mode == "sim" or not da3.cloud_frame
        else:
            assert (da3.mount_sensor, da3.optical_parent, da3.cloud_frame) == want


def test_a_shared_mount_gets_its_optical_child_in_sim_without_a_depth_sensor() -> None:
    """The behaviour #641 changes: the sim bridge used to ignore ``shares_mount_with``.

    OpenArm ``top`` rides ``head_zed``'s mount, so its pixels are in
    ``zed_camera_link_optical_frame``. The bridge latched that frame only for the depth
    sensor itself; with an RGB-only head camera on that mount nothing published it and
    the lift dropped every ``top`` detection. The RGB camera now gets its mount's child
    itself, on every backend.
    """
    arm = lift_sensors("openarm")
    top = next(s for s in arm if s.name == "top")
    assert top.shares_mount_with == "head_zed" and top.frame_id == "world"
    rgb_only = [
        s.model_copy(update={"modality": "rgb"}) if s.name == "head_zed" else s for s in arm
    ]
    for backend_places_depth in (False, True):
        placed = _static_optical_mount(top, rgb_only, backend_places_depth=backend_places_depth)
        assert placed is not None
        assert placed[1:] == ("zed_camera_link", "zed_camera_link_optical_frame")


def test_the_depth_leg_latches_a_sensors_own_mount() -> None:
    """The depth leg stamps a raster in the sensor's own optical frame, so it resolves
    the sensor alone (no ``sensors`` to share a mount with)."""
    arm = lift_sensors("openarm")
    zed = next(s for s in arm if s.name == "head_zed")
    placed = _static_optical_mount(zed, (), backend_places_depth=False)
    assert placed is not None and placed.sensor is zed
    assert placed[1:] == ("zed_camera_link", "zed_camera_link_optical_frame")
    assert _static_optical_mount(zed, (), backend_places_depth=True) is None  # Isaac: live
    front = next(s for s in lift_sensors("panda_mobile") if s.name == "front_depth")
    assert _static_optical_mount(front, (), backend_places_depth=False) is None  # optical


def test_the_extrinsic_gate_checks_the_mount_optical_mount_names() -> None:
    """``depth_extrinsic_problems`` checks an estimated-depth camera's mount sensor:
    the one ``optical_mount`` resolves, so the gate and the resolver agree."""
    desc = RobotDescription.from_yaml(str(_REPO / "robots" / "openarm" / "robot.yaml"))
    nominal = [
        s.model_copy(update={"static_transform_calibrated": False, "modality": "rgb"})
        if s.name == "head_zed"
        else s
        for s in desc.sensors
    ]
    rgb_zed = desc.model_copy(update={"sensors": nominal})
    assert depth_extrinsic_problems(rgb_zed, []) == []  # no cloud source left
    (problem,) = depth_extrinsic_problems(rgb_zed, [], estimated_camera="top")
    assert problem.startswith("head_zed: mount is the manifest's nominal value")
    da3 = resolve_estimated_depth_camera(nominal, "top", hal_mode="sim")
    assert da3.mount_sensor == "head_zed"
