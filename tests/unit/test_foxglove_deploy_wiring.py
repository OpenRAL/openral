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
  `robots/openarm` once declared its `top` camera for sim with none, so on the
  real cell `/openral/cameras/top/image` had zero publishers and its panel read
  "Image topic does not exist", which is indistinguishable from a dead camera.
  The manifest now binds every OpenArm camera itself (a deploy scene never
  touches a robot camera).
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LAUNCH = _ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
_MANIFEST = _ROOT / "robots" / "openarm" / "robot.yaml"

# The committed real OpenArm cell. It has no `sensors:` block: every camera on that cell is a
# robot camera, bound in robots/openarm/robot.yaml (a deploy scene never touches one).
_SCENE = _ROOT / "scenes" / "deploy" / "openarm_bench.yaml"


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


def test_every_rgb_camera_the_real_deploy_declares_is_also_bound() -> None:
    """A declared-but-unbound RGB slot is a panel that can never fill.

    On a real deploy an unbound RGB sensor has zero publishers while the bridge
    still advertises its channel (the allowlist is the pattern
    `/openral/cameras/.*/image`) — in the viewer, indistinguishable from a dead
    camera. So every RGB slot the real OpenArm cell surfaces (manifest plus scene)
    must carry a binding.
    """
    scene = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))

    declared_rgb = _rgb_sensors(manifest) + _rgb_sensors(scene)
    bound = {s["name"] for s in declared_rgb if s.get("deploy_binding")}
    declared = {s["name"] for s in declared_rgb}

    assert {"top", "wrist_left", "wrist_right"} <= bound, (
        f"expected all three rig cameras to be bound: {bound}"
    )
    assert declared - bound == set(), (
        f"{sorted(declared - bound)} are declared but never bound, so their panels "
        "would advertise a channel with no publisher — the failure that reads as a "
        "dead camera. Bind them in robot.yaml (or, for a workcell camera, the scene)."
    )


def test_the_real_top_camera_keeps_the_sim_slot() -> None:
    """Sim and real feed the policy the same slot, with no scene remap.

    The manifest's `top` carries `observation.images.top` in sim and, through its
    own `deploy_binding`, on the real cell. The scene never touches it, so the
    checkpoint sees the real camera on the slot it saw the MuJoCo render on. A
    checkpoint trained on a different name maps it with `image_preprocessing.aliases`
    (the restock π0.5 maps `top` onto its `context` input), not a scene edit.
    """
    from openral_core import DeployScene, RobotDescription, merge_deploy_sensors

    scene = DeployScene.from_yaml(str(_SCENE))
    description = RobotDescription.from_yaml(str(_MANIFEST))

    assert "top" not in {s.name for s in scene.sensors}
    merged_top = next(
        sensor
        for sensor in merge_deploy_sensors(description.sensors, scene.sensors)
        if sensor.name == "top"
    )
    robot_top = next(s for s in description.sensors if s.name == "top")
    assert merged_top == robot_top
    assert merged_top.vla_feature_key == "observation.images.top"
    assert merged_top.deploy_binding is not None


def test_a_scene_that_rebinds_the_top_camera_is_refused(tmp_path: pathlib.Path) -> None:
    """A deploy scene never touches a robot camera, not even to bind it to a host."""
    from openral_core import DeployScene, RobotDescription, merge_deploy_sensors
    from openral_core.exceptions import ROSConfigError

    doc = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    doc["sensors"] = [
        {
            "name": "top",
            "modality": "rgb",
            "frame_id": "world",
            "rate_hz": 30.0,
            "deploy_binding": {
                "backend": "ros2_image",
                "backend_params": {"topic": "/zed/zed_node/rgb/color/rect/image"},
            },
        }
    ]
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text(yaml.safe_dump(doc), encoding="utf-8")

    description = RobotDescription.from_yaml(str(_MANIFEST))
    with pytest.raises(ROSConfigError, match=r"'top'.*defined by the robot manifest"):
        merge_deploy_sensors(description.sensors, DeployScene.from_yaml(str(scene_path)).sensors)


def test_the_launch_generates_its_layout_from_the_bound_cameras() -> None:
    """The layout must come from the deploy, not from a fixed default."""
    text = _LAUNCH.read_text(encoding="utf-8")
    call = text.split("_write_foxglove_layout(", 2)[-1]
    assert call.lstrip().startswith("bound_rgb_camera_names"), (
        "the layout must be generated from the deploy-bound cameras; a hardcoded "
        "default cannot know the scene"
    )
    # The "bound" filter is the shared openral_core rule (real: merged manifest+scene
    # sensors with a deploy_binding), exercised on real manifests in
    # test_deploy_e2e_camera_wiring.py::test_real_deploy_picks_a_bound_camera.
    assert "publishing = publishing_sensors(description.sensors, scene_sensors, hal_mode)" in text
    assert '[s.name for s in publishing if s.modality == "rgb"]' in text


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
