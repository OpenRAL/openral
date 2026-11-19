# SPDX-License-Identifier: Apache-2.0
"""``optical_mount``: one rule for the frame a camera's pixels are in, and who publishes it.

Issue #641 (the #627 class of bug): the object lift looks a camera up in
``image_frame_id``; the DA3 resolver names a cloud frame; the sim bridge and the real
deploy launch publish the REP-103 optical child. All four read ``optical_mount``, and
these tests hold, on every in-tree manifest, that a frame a consumer names is one a
publisher emits. The launch's half is in ``test_deploy_e2e_camera_wiring.py`` (it needs
the ROS launch modules).

Real fixtures (CLAUDE.md §1.11): every ``robots/*/robot.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.depth_extrinsic import (
    image_frame_id,
    optical_mount,
    resolve_estimated_depth_camera,
)
from openral_core.schemas import SensorSpec
from openral_hal.depth_cloud import depth_optical_frame_id, is_depth_sensor
from openral_hal.sim_sensor_bridge import _static_optical_mount

_REPO = Path(__file__).resolve().parents[2]
ROBOTS = sorted(p.parent.name for p in (_REPO / "robots").glob("*/robot.yaml"))
#: The sim bridge's two depth paths: MuJoCo ray-casts, Isaac surfaces depth in obs.
BACKENDS = {"mujoco": (True, False), "isaac": (False, True)}


def lift_sensors(robot: str) -> list[SensorSpec]:
    """The sensors the object lift resolves cameras over: the manifest's and its bundles'."""
    desc = RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml"))
    return [*desc.sensors, *(s for b in desc.sensor_bundles for s in b.sensors)]


def test_every_robot_is_covered() -> None:
    assert {"makermods_metal", "openarm", "openamrobot_v2", "panda_mobile"} <= set(ROBOTS)


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_the_sim_bridge_publishes_every_frame_the_lift_reads(robot: str, backend: str) -> None:
    """Each RGB camera's ``image_frame_id`` is its own ``frame_id``, a mount frame that
    is already optical, or an optical child the bridge puts on TF: latched under the
    mount, or broadcast live by the depth leg. Never both: a frame has one parent."""
    mujoco, obs_depth = BACKENDS[backend]
    sensors = lift_sensors(robot)
    depth = [s for s in sensors if is_depth_sensor(s)]
    # What SimSensorBridge._setup_cameras + _setup_depth latch on /tf_static ...
    static: dict[str, str] = {}
    for spec in (*(s for s in sensors if s.modality == "rgb"), *depth):
        placed = _static_optical_mount(spec, sensors, mujoco=mujoco, obs_depth=obs_depth)
        if placed is not None:
            assert static.setdefault(placed.optical, placed.frame) == placed.frame
    # ... and what the depth leg broadcasts live (every depth optical frame not latched).
    live = {depth_optical_frame_id(s) for s in depth} - set(static)
    for spec in sensors:
        if spec.modality != "rgb":
            continue
        frame = image_frame_id(spec, sensors)
        placed = optical_mount(spec, sensors)
        if placed is None:
            assert frame == spec.frame_id
        elif placed.optical != placed.frame:
            assert frame in static or frame in live, f"{spec.name}: nothing publishes {frame}"
            assert not (frame in static and frame in live)


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("hal_mode", ["sim", "real"])
def test_the_da3_cloud_frame_is_the_image_frame(robot: str, hal_mode: str) -> None:
    """Wherever the DA3 resolver places a camera, its cloud is in the frame the camera's
    pixels are in, hung under the mount ``optical_mount`` names."""
    sensors = lift_sensors(robot)
    for spec in sensors:
        if spec.modality != "rgb":
            continue
        mount = resolve_estimated_depth_camera(sensors, spec.name, hal_mode=hal_mode)
        if not mount.cloud_frame:
            continue
        assert mount.cloud_frame == image_frame_id(spec, sensors)
        placed = optical_mount(spec, sensors)
        if mount.optical_parent is not None:
            assert placed is not None
            assert (placed.frame, placed.optical) == (mount.optical_parent, mount.cloud_frame)
            assert placed.sensor.name == mount.mount_sensor


def test_a_shared_mount_gets_its_optical_child_in_sim_without_a_depth_leg() -> None:
    """The behaviour #641 changes: the sim bridge used to ignore ``shares_mount_with``.

    OpenArm ``top`` rides ``head_zed``'s mount, so its pixels are in
    ``zed_camera_link_optical_frame``. The bridge latched that frame only from the depth
    leg (``head_zed`` itself); with the ZED's depth off (an RGB-only head camera) nothing
    published it and the lift dropped every ``top`` detection. The RGB camera now gets
    its mount's child itself, on every backend, and the depth leg still owns the frame
    where it broadcasts it live (Isaac).
    """
    arm = lift_sensors("openarm")
    top = next(s for s in arm if s.name == "top")
    assert top.shares_mount_with == "head_zed" and top.frame_id == "world"
    want = ("zed_camera_link", "zed_camera_link_optical_frame")
    assert image_frame_id(top, arm) == want[1]

    rgb_only = [
        s.model_copy(update={"modality": "rgb"}) if s.name == "head_zed" else s for s in arm
    ]
    for mujoco, obs_depth in BACKENDS.values():
        placed = _static_optical_mount(top, rgb_only, mujoco=mujoco, obs_depth=obs_depth)
        assert placed is not None and placed[1:] == want

    # With the depth leg: latched on MuJoCo (the mount is declared), live on Isaac.
    assert _static_optical_mount(top, arm, mujoco=True, obs_depth=False)[1:] == want
    assert _static_optical_mount(top, arm, mujoco=False, obs_depth=True) is None


def test_an_own_mount_camera_gets_its_child_and_a_link_camera_none() -> None:
    """MakerMods ``top`` (own body-convention mount) gets ``top_camera_optical_frame`` on
    every backend; panda_mobile's ``wrist`` (a robot link, no mount) keeps its frame."""
    metal = lift_sensors("makermods_metal")
    top = next(s for s in metal if s.name == "top")
    for mujoco, obs_depth in BACKENDS.values():
        placed = _static_optical_mount(top, metal, mujoco=mujoco, obs_depth=obs_depth)
        assert placed is not None
        assert placed[1:] == ("top_camera", "top_camera_optical_frame")
        assert placed.sensor is top

    mobile = lift_sensors("panda_mobile")
    wrist = next(s for s in mobile if s.name == "wrist")
    assert optical_mount(wrist, mobile) is None
    assert image_frame_id(wrist, mobile) == wrist.frame_id
    assert _static_optical_mount(wrist, mobile, mujoco=True, obs_depth=False) is None
