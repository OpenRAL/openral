# SPDX-License-Identifier: Apache-2.0
"""What `--foxglove` has to put on the graph for the panels to fill.

Two things were missing on the real OpenArm cell, both invisible from inside the
graph — every node healthy, every panel empty:

* **The Bucket-2 converter was never spawned.** The layout's collision and voxel
  panels read `/openral/world_collisions_markers` and
  `/openral/world_voxels_cloud`, which are `visualization_msgs` / `sensor_msgs`
  re-publications of the custom `openral_msgs` world types. Foxglove renders the
  standard types natively and the custom ones not at all, and nothing else in
  the graph produces them, so those panels could never fill on any deploy.

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
_SCENE = _ROOT / "scenes" / "deploy" / "openarm_restock_shelf.yaml"
_MANIFEST = _ROOT / "robots" / "openarm" / "robot.yaml"


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

    The manifest's `top` is the SIM overhead camera — a MuJoCo render with no
    `deploy_binding` — so on the real cell `/openral/cameras/top/image` had
    zero publishers while the bridge still advertised the channel (its
    allowlist is the pattern `/openral/cameras/.*/image`). In the viewer that
    is indistinguishable from a dead camera. The scene now overrides `top`
    with the ZED head camera, so every slot this deploy surfaces has a
    publisher behind it.
    """
    scene = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
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


def test_the_overridden_top_slot_keeps_the_feature_key_the_policy_was_trained_on() -> None:
    """Renaming the slot must not rename the policy's input.

    The scene's `top` overrides the manifest's `top` field-wise, and the
    manifest's is the sim overhead camera carrying
    `observation.images.base`. The policy reads by `vla_feature_key`, not by
    sensor name, and was trained with the ZED on
    `observation.images.context` — so letting the manifest's key survive the
    merge would hand it an OOD base stream and an empty context stream, with
    every node healthy and no error anywhere.
    """
    scene = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))

    scene_top = next(s for s in _rgb_sensors(scene) if s["name"] == "top")
    manifest_top = next(s for s in _rgb_sensors(manifest) if s["name"] == "top")

    assert scene_top["vla_feature_key"] == "observation.images.context"
    assert manifest_top["vla_feature_key"] == "observation.images.base", (
        "fixture moved: the sim `top` no longer carries the key this override has to shadow"
    )
    # `merge_deploy_sensors` copies only the fields the scene explicitly sets,
    # so anything the sim entry declares and the scene omits survives into the
    # real deploy — sim intrinsics on a ZED, for instance.
    for field in ("frame_id", "rate_hz", "intrinsics", "encoding", "vendor", "model"):
        assert field in scene_top, (
            f"the sim `top` sets {field!r}; the override must restate it or the "
            "MuJoCo value silently describes the ZED"
        )


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
    collision/voxel panel — the one whose entire job is to show the world model
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
