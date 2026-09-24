# SPDX-License-Identifier: Apache-2.0
"""What `--foxglove` has to put on the graph for the panels to fill.

Two things were missing on the real OpenArm cell, both invisible from inside the
graph — every node healthy, every panel empty:

* **The Bucket-2 converter was never spawned.** The layout's voxel panels read
  `/openral/world_voxels_cloud`, a `sensor_msgs` re-publication of the custom
  `openral_msgs/OccupancyVoxels`. Foxglove renders the standard type natively
  and the custom one not at all, and nothing else in the graph produces it, so
  those panels could never fill on any deploy.

* **Camera slots were a guess.** They are sensor names from the robot manifest
  and the deploy scene, and they differ per robot. A declared RGB sensor only
  gets a reader — and therefore a topic — when it carries a `deploy_binding`;
  `robots/openarm` declares a `top` camera for sim with none, so on the real
  cell `/openral/cameras/top/image` has zero publishers and its panel reads
  "Image topic does not exist", which is indistinguishable from a dead camera.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LAUNCH = _ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
_MANIFEST = _ROOT / "robots" / "openarm" / "robot.yaml"

# A synthetic real-cell binding for the tests below: the manifest's `top` is a MuJoCo render
# with no `deploy_binding`, and no in-tree scene currently binds OpenArm's cameras for a real
# deploy — so this binds all three rig cameras directly, the same shape a real deploy scene
# would carry, without depending on one being committed. Host binding ONLY: each name is a
# robot-manifest sensor, so its frame, mount and intrinsics stay the manifest's
# (`check_scene_sensor_overrides`); `frame_id` is required and restated verbatim.
_SCENE_YAML = {
    "scene": {"id": "test_cell"},
    "robot_id": "openarm",
    "sensors": [
        {
            "name": "top",
            "modality": "rgb",
            "frame_id": "world",
            "rate_hz": 30.0,
            "encoding": "bgr8",
            "deploy_binding": {
                "backend": "ros2_image",
                "backend_params": {"topic": "/zed/zed_node/rgb/color/rect/image"},
            },
        },
        {
            "name": "wrist_left",
            "modality": "rgb",
            "frame_id": "openarm_left_ee_base_link",
            "rate_hz": 30.0,
            "encoding": "bgr8",
            "deploy_binding": {
                "backend": "opencv_thread",
                "backend_params": {"device": "/dev/camera_wrist_left"},
            },
        },
        {
            "name": "wrist_right",
            "modality": "rgb",
            "frame_id": "openarm_right_ee_base_link",
            "rate_hz": 30.0,
            "encoding": "bgr8",
            "deploy_binding": {
                "backend": "opencv_thread",
                "backend_params": {"device": "/dev/camera_wrist_right"},
            },
        },
    ],
}


@pytest.fixture
def _scene(tmp_path: pathlib.Path) -> pathlib.Path:
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text(yaml.safe_dump(_SCENE_YAML), encoding="utf-8")
    return scene_path


def _rgb_sensors(doc: dict[str, object]) -> list[dict[str, object]]:
    sensors = doc.get("sensors") or []
    assert isinstance(sensors, list)
    return [s for s in sensors if isinstance(s, dict) and s.get("modality") == "rgb"]


def test_the_deploy_launch_spawns_the_bucket2_converter() -> None:
    """Without it, two panels in the shipped layout can never fill."""
    text = _LAUNCH.read_text(encoding="utf-8")
    assert 'executable="bucket2_markers"' in text, (
        "deploy_e2e.launch.py must spawn openral_foxglove_bringup's bucket2_markers; "
        "nothing else republishes the openral_msgs world types as Foxglove-native ones"
    )


def test_every_rgb_camera_the_real_deploy_declares_is_also_bound(_scene: pathlib.Path) -> None:
    """A declared-but-unbound RGB slot is a panel that can never fill.

    The manifest's `top` is the SIM overhead camera — a MuJoCo render with no
    `deploy_binding` — so on a real deploy `/openral/cameras/top/image` would
    have zero publishers while the bridge still advertised the channel (its
    allowlist is the pattern `/openral/cameras/.*/image`). In the viewer that
    is indistinguishable from a dead camera. A real deploy scene must override
    `top` with a bound camera, so every slot it surfaces has a publisher
    behind it.
    """
    scene = yaml.safe_load(_scene.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))

    bound = {s["name"] for s in _rgb_sensors(scene) if s.get("deploy_binding")}
    bound |= {s["name"] for s in _rgb_sensors(manifest) if s.get("deploy_binding")}
    declared = {s["name"] for s in _rgb_sensors(manifest)} | {
        s["name"] for s in _rgb_sensors(scene)
    }

    assert {"top", "wrist_left", "wrist_right"} <= bound, (
        f"expected all three rig cameras to be bound: {bound}"
    )
    assert declared - bound == set(), (
        f"{sorted(declared - bound)} are declared but never bound, so their panels "
        "would advertise a channel with no publisher — the failure that reads as a "
        "dead camera. Bind them in the scene or drop the declaration."
    )


def test_the_real_top_camera_keeps_the_sim_slot(
    _scene: pathlib.Path,
) -> None:
    """Sim and real feed the policy the same slot, with no scene remap.

    The manifest's `top` carries `observation.images.top` in sim, and a real
    deploy scene binds only the hardware. It must not restate the key:
    `merge_deploy_sensors` keeps the manifest's, so the checkpoint sees the real
    camera on the slot it saw the MuJoCo render on. A checkpoint trained on a
    different name maps it with `image_preprocessing.aliases`, not a scene edit.
    Nor may it restate geometry: a camera with a different mount or intrinsics
    is a different manifest entry or a workcell camera with its own name.
    """
    scene = yaml.safe_load(_scene.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))

    scene_top = next(s for s in _rgb_sensors(scene) if s["name"] == "top")
    manifest_top = next(s for s in _rgb_sensors(manifest) if s["name"] == "top")

    assert manifest_top["vla_feature_key"] == "observation.images.top"
    assert "vla_feature_key" not in scene_top

    # And through the real merge the sensor leg uses, not just the YAML.
    from openral_core import DeployScene, RobotDescription
    from openral_rskill_ros.sensor_leg import merge_deploy_sensors

    description = RobotDescription.from_yaml(str(_MANIFEST))
    merged_top = next(
        sensor
        for sensor in merge_deploy_sensors(
            description.sensors, DeployScene.from_yaml(str(_scene)).sensors
        )
        if sensor.name == "top"
    )
    robot_top = next(s for s in description.sensors if s.name == "top")
    assert merged_top.vla_feature_key == "observation.images.top"
    assert merged_top.deploy_binding is not None
    # Geometry is the manifest's: the scene binds, the robot describes.
    assert merged_top.frame_id == robot_top.frame_id
    assert merged_top.intrinsics == robot_top.intrinsics
    assert merged_top.parent_frame == robot_top.parent_frame


def test_the_launch_generates_its_layout_from_the_bound_cameras() -> None:
    """The layout must come from the deploy, not from a fixed default."""
    text = _LAUNCH.read_text(encoding="utf-8")
    call = text.split("_write_foxglove_layout(", 2)[-1]
    assert call.lstrip().startswith("bound_rgb_camera_names"), (
        "the layout must be generated from the deploy-bound cameras; a hardcoded "
        "default cannot know the scene"
    )
    assert 'getattr(s, "deploy_binding", None) is not None' in text


def test_the_shipped_default_matches_how_manifests_spell_wrist_cameras() -> None:
    """`left_wrist` matched no camera on any robot; `wrist_left` is the spelling."""
    layout = pytest.importorskip("openral_foxglove_bringup.layout")
    assert "wrist_left" in layout.DEFAULT_CAMERAS
    assert "left_wrist" not in layout.DEFAULT_CAMERAS


def test_the_launch_gives_the_layout_the_robots_own_base_frame() -> None:
    """A 3D panel following a frame TF never broadcasts renders nothing at all.

    Not "no point cloud" — the whole scene: no robot model, no octomap voxels,
    no Bucket-2 markers. `build_layout`'s default is the ROS-conventional
    `base_link`, and OpenArm's root is `openarm_base`, so the generated layout
    opened onto an empty 3D view while every topic underneath it was
    publishing. Observed on the cell with the ZED cloud live at 3.3 Hz.
    """
    text = _LAUNCH.read_text(encoding="utf-8")
    assert "follow_frame=base_frame" in text, (
        "the generated layout must follow the robot's own base frame, not the "
        "library's base_link default"
    )
    assert "description.base_frame" in text, (
        "the base frame must come from the robot manifest, which is the only thing that knows it"
    )


def test_both_3d_panels_follow_the_same_frame() -> None:
    """The Bucket-2 panel had `base_link` hardcoded past the parameter.

    Threading `follow_frame` only into the hero panel would have left the
    voxel panel — the one whose entire job is to show the world model
    — blank on exactly the robots the fix was for.
    """
    layout = pytest.importorskip("openral_foxglove_bringup.layout")
    built = layout.build_layout(["top"], follow_frame="openarm_base")
    panels = {
        panel_id: cfg["followTf"]
        for panel_id, cfg in built["configById"].items()
        if isinstance(cfg, dict) and "followTf" in cfg
    }
    robot_relative = {k: v for k, v in panels.items() if v != "map"}
    assert robot_relative, f"no robot-relative 3D panel found in {panels}"
    assert set(robot_relative.values()) == {"openarm_base"}, (
        f"every robot-relative panel must follow the requested frame: {panels}"
    )
