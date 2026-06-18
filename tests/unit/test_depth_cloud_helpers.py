# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the reusable depth-cloud HAL helpers (ADR-0030).

These are the robot-agnostic pieces a deploy-sim HAL node uses to turn a
depth ``SensorSpec`` into a ``sensor_msgs/PointCloud2`` for octomap_server:

* `is_depth_sensor` / `mjcf_camera_name` / `depth_synth_kwargs` — pure
  SensorSpec adapters (no ROS, no MuJoCo).
* `camera_optical_tf_to_base` — the live camera-optical-frame → base TF,
  ray-cast-free, against a real `mujoco.MjModel`.
* `pointcloud2_from_points_xyz` — packs an (N,3) array into PointCloud2
  (needs `sensor_msgs`; skipped when ROS isn't on the path).
"""

from __future__ import annotations

import numpy as np
import pytest
from openral_core.schemas import IntrinsicsPinhole, SensorSpec
from openral_hal.depth_cloud import (
    camera_optical_tf_to_base,
    depth_synth_kwargs,
    is_depth_sensor,
    mjcf_camera_name,
)


def _depth_spec() -> SensorSpec:
    return SensorSpec(
        name="front_depth",
        modality="depth",
        frame_id="front_depth_optical_frame",
        rate_hz=10.0,
        intrinsics=IntrinsicsPinhole(width=64, height=48, fx=40.0, fy=40.0, cx=32.0, cy=24.0),
        range_min_m=0.2,
        range_max_m=4.0,
        metadata={"mjcf_camera": "robot0_agentview_left"},
    )


def test_is_depth_sensor_discriminates_modality() -> None:
    assert is_depth_sensor(_depth_spec()) is True
    rgb = SensorSpec(name="cam", modality="rgb", frame_id="f", rate_hz=20.0)
    assert is_depth_sensor(rgb) is False
    # depth modality but no intrinsics → cannot back-project → not usable.
    no_intr = SensorSpec(name="d", modality="depth", frame_id="f", rate_hz=10.0)
    assert is_depth_sensor(no_intr) is False


def test_mjcf_camera_name_prefers_metadata_then_falls_back_to_name() -> None:
    assert mjcf_camera_name(_depth_spec()) == "robot0_agentview_left"
    bare = SensorSpec(
        name="head_depth",
        modality="depth",
        frame_id="f",
        rate_hz=10.0,
        intrinsics=IntrinsicsPinhole(width=8, height=8, fx=4.0, fy=4.0, cx=4.0, cy=4.0),
    )
    assert mjcf_camera_name(bare) == "head_depth"


def test_depth_synth_kwargs_extracts_intrinsics_and_ranges() -> None:
    kw = depth_synth_kwargs(_depth_spec(), max_range_default=8.0)
    assert kw["camera_name"] == "robot0_agentview_left"
    assert kw["width"] == 64
    assert kw["height"] == 48
    assert kw["fx"] == 40.0
    assert kw["cy"] == 24.0
    assert kw["min_range_m"] == 0.2
    assert kw["max_range_m"] == 4.0  # from range_max_m, not the default


def test_depth_synth_kwargs_uses_default_when_range_absent() -> None:
    spec = SensorSpec(
        name="d",
        modality="depth",
        frame_id="f",
        rate_hz=10.0,
        intrinsics=IntrinsicsPinhole(width=8, height=8, fx=4.0, fy=4.0, cx=4.0, cy=4.0),
    )
    kw = depth_synth_kwargs(spec, max_range_default=5.0)
    assert kw["max_range_m"] == 5.0
    assert kw["min_range_m"] == 0.0


# ── live camera-optical → base TF (real MuJoCo, no GL) ────────────────────

_TF_MJCF = """
<mujoco model="cam_tf_test">
  <worldbody>
    <body name="base" pos="1 0 0">
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
    <camera name="depth0" pos="0 0 0"/>
  </worldbody>
</mujoco>
"""


def test_camera_optical_tf_to_base_translation_and_orientation() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_TF_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    xyz, quat_xyzw = camera_optical_tf_to_base(
        model=model, data=data, camera_name="depth0", base_body_name="base"
    )
    # Camera at world origin, base body at (1,0,0): optical origin expressed
    # in the base frame is (-1, 0, 0).
    assert np.allclose(xyz, [-1.0, 0.0, 0.0], atol=1e-6)
    # Default MuJoCo camera looks down world -Z; optical→base rotation is a
    # 180° flip about x (REP-103 optical y/z vs MuJoCo cam y/z) → quat
    # (x=1, y=0, z=0, w=0).
    q = np.asarray(quat_xyzw, dtype=float)
    q *= np.sign(q[0]) or 1.0  # fix sign ambiguity for the assert
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


# ── base body resolution (real MuJoCo, no GL) ─────────────────────────────


def test_resolve_base_body_name_prefers_existing_candidates() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import resolve_base_body_name

    # Fixed-arm robosuite naming: only ``robot0_base`` exists.
    robosuite = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='robot0_base'>"
        "<geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
    )
    assert resolve_base_body_name(robosuite) == "robot0_base"

    # Synthetic twin: bare ``base`` wins over the later candidates.
    synthetic = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        "<body name='base'><geom type='box' size='.1 .1 .1'/></body>"
        "<body name='robot0_base'><geom type='box' size='.1 .1 .1'/></body>"
        "</worldbody></mujoco>"
    )
    assert resolve_base_body_name(synthetic) == "base"


def test_resolve_base_body_name_prefers_mobile_base_over_robot0_placeholder() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import resolve_base_body_name

    # RoboCasa mobile-manipulator naming: both bodies exist, but robot0_base is
    # a placeholder mount left at a fixed offset; mobilebase0_base is the real
    # base. The mobile base must win (else the viewer frames empty space).
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        "<body name='robot0_base' pos='10 10 0'><geom type='box' size='.1 .1 .1'/></body>"
        "<body name='mobilebase0_base' pos='0 0 0'><geom type='box' size='.1 .1 .1'/></body>"
        "</worldbody></mujoco>"
    )
    assert resolve_base_body_name(model) == "mobilebase0_base"


def test_resolve_base_body_name_returns_none_when_absent() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import resolve_base_body_name

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='counter_top'>"
        "<geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
    )
    assert resolve_base_body_name(model) is None


# ── robosuite collision-geom hiding (real MuJoCo, no GL) ──────────────────


def test_apply_robosuite_visual_geomgroups_hides_collision_for_robosuite() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import apply_robosuite_visual_geomgroups

    # Robosuite signature: a robot0_-prefixed body → hide group 0, show group 1.
    robosuite = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='robot0_base'>"
        "<geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
    )
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 1
    acted = apply_robosuite_visual_geomgroups(opt, robosuite)
    assert acted is True
    assert opt.geomgroup[0] == 0  # collision hidden
    assert opt.geomgroup[1] == 1  # visual shown


def test_apply_robosuite_visual_geomgroups_detects_via_agentview_camera() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import apply_robosuite_visual_geomgroups

    # Custom robosuite composition without a robot0_ body, but with an agentview
    # camera (the robosuite signature) → still hide collision.
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><camera name='agentview'/>"
        "<body name='openarm_root'><geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
    )
    opt = mujoco.MjvOption()
    assert apply_robosuite_visual_geomgroups(opt, model) is True
    assert opt.geomgroup[0] == 0


def test_apply_robosuite_visual_geomgroups_noop_for_dm_control() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import apply_robosuite_visual_geomgroups

    # No robosuite naming (à la gym-aloha, whose VISUAL geoms live in group 0):
    # must NOT touch geomgroup, else those visuals get blanked.
    dm = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='vx300s_left/base_link'>"
        "<geom type='box' size='.1 .1 .1' group='0'/></body></worldbody></mujoco>"
    )
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 1
    acted = apply_robosuite_visual_geomgroups(opt, dm)
    assert acted is False
    assert opt.geomgroup[0] == 1  # untouched → group-0 visuals stay visible


# ── preferred viewer camera selection (real MuJoCo, no GL) ────────────────


def test_preferred_viewer_camera_prefers_agentview() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import preferred_viewer_camera_id

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        "<camera name='robot0_eye_in_hand'/>"
        "<camera name='robot0_agentview_left'/>"
        "<camera name='robot0_agentview_right'/>"
        "<body name='b'><geom type='box' size='.1 .1 .1'/></body>"
        "</worldbody></mujoco>"
    )
    cid = preferred_viewer_camera_id(model)
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid) == "robot0_agentview_left"


def test_preferred_viewer_camera_picks_top_over_front_close_for_aloha() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import preferred_viewer_camera_id

    # gym-aloha's real camera set: the top-down overview must win over the
    # ``front_close`` zoom (``top`` is ranked above bare ``front``).
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        "<camera name='left_pillar'/><camera name='right_pillar'/>"
        "<camera name='top'/><camera name='angle'/><camera name='front_close'/>"
        "<camera name='left_wrist'/><camera name='right_wrist'/>"
        "<body name='b'><geom type='box' size='.1 .1 .1'/></body>"
        "</worldbody></mujoco>"
    )
    cid = preferred_viewer_camera_id(model)
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid) == "top"


def test_preferred_viewer_camera_falls_back_to_first_then_none() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import preferred_viewer_camera_id

    # No overview cam → first authored camera (e.g. a wrist cam) still beats free orbit.
    wrist_only = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><camera name='robot0_eye_in_hand'/>"
        "<body name='b'><geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
    )
    assert preferred_viewer_camera_id(wrist_only) == 0

    # No cameras at all → -1, so the caller uses the base-aligned free camera.
    no_cam = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='b'>"
        "<geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
    )
    assert preferred_viewer_camera_id(no_cam) == -1


# ── base-aligned free camera (real MuJoCo, no GL) ─────────────────────────

_CAM_MJCF = """
<mujoco model="base_cam_test">
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

# A base translated to (2, 3, 0) and yawed 90° about world +Z.
_CAM_MJCF_YAWED = """
<mujoco model="base_cam_yaw_test">
  <worldbody>
    <body name="base" pos="2 3 0" euler="0 0 90">
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_base_aligned_free_camera_centres_and_aligns_at_origin() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import base_aligned_free_camera

    model = mujoco.MjModel.from_xml_string(_CAM_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lookat, distance, azimuth, elevation = base_aligned_free_camera(
        model=model, data=data, base_body_name="base"
    )
    # Base at the world origin, identity orientation → lookat is the origin,
    # azimuth is the bare offset (no yaw), distance scales with model extent.
    assert np.allclose(lookat, [0.0, 0.0, 0.0], atol=1e-6)
    assert azimuth == pytest.approx(135.0)
    assert elevation == pytest.approx(-25.0)
    assert distance == pytest.approx(float(model.stat.extent) * 2.0)  # small scene < cap


def test_base_aligned_free_camera_tracks_position_and_yaw() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import base_aligned_free_camera

    model = mujoco.MjModel.from_xml_string(_CAM_MJCF_YAWED)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lookat, _distance, azimuth, _elevation = base_aligned_free_camera(
        model=model, data=data, base_body_name="base", azimuth_offset_deg=135.0
    )
    # lookat follows the base origin; azimuth is offset by the base's 90° yaw so
    # the framing stays identical relative to the robot's own forward axis.
    assert np.allclose(lookat, [2.0, 3.0, 0.0], atol=1e-6)
    assert azimuth == pytest.approx(225.0, abs=1e-4)


def test_base_aligned_free_camera_caps_distance_for_large_scenes() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import base_aligned_free_camera

    # A ~20 m-wide composed scene (à la a RoboCasa kitchen): the base body sits
    # at one end while model.stat.extent reflects the whole room.
    big = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        "<body name='base' pos='0 -5 0'><geom type='box' size='.1 .1 .1'/></body>"
        "<geom name='far_wall' type='box' pos='10 10 0' size='0.1 5 3'/>"
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(big)
    mujoco.mj_forward(big, data)
    assert float(big.stat.extent) * 2.0 > 3.5  # extent-based distance would blow past the cap
    _lookat, distance, _az, _el = base_aligned_free_camera(
        model=big, data=data, base_body_name="base", max_distance_m=3.5
    )
    # Capped so the robot stays screen-filling instead of a speck 40 m away.
    assert distance == pytest.approx(3.5)


def test_base_aligned_free_camera_falls_back_when_base_absent() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import base_aligned_free_camera

    model = mujoco.MjModel.from_xml_string(_CAM_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Unknown / None base body → centre on the model, no yaw offset, still safe.
    lookat, distance, azimuth, _elevation = base_aligned_free_camera(
        model=model, data=data, base_body_name="does_not_exist"
    )
    assert np.allclose(lookat, np.asarray(model.stat.center), atol=1e-6)
    assert azimuth == pytest.approx(135.0)
    assert distance > 0.0

    none_lookat, _d, none_az, _e = base_aligned_free_camera(
        model=model, data=data, base_body_name=None
    )
    assert np.allclose(none_lookat, np.asarray(model.stat.center), atol=1e-6)
    assert none_az == pytest.approx(135.0)


def test_robot_self_body_ids_matches_prefixes() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_hal.depth_cloud import robot_self_body_ids

    xml = """
    <mujoco model="self_bodies">
      <worldbody>
        <body name="robot0_link1"><geom type="box" size="0.1 0.1 0.1"/></body>
        <body name="mobilebase0_base"><geom type="box" size="0.1 0.1 0.1"/></body>
        <body name="counter_top"><geom type="box" size="0.1 0.1 0.1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    # sim_joint_names like robosuite/robocasa emits.
    ids = robot_self_body_ids(model, ["robot0_joint1", "mobilebase0_joint_mobile_forward"])
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in ids}
    assert "robot0_link1" in names
    assert "mobilebase0_base" in names
    assert "counter_top" not in names  # kitchen fixture stays in the world map
    assert robot_self_body_ids(model, []) == frozenset()


def test_camera_optical_tf_unknown_camera_raises() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_core.exceptions import ROSConfigError

    model = mujoco.MjModel.from_xml_string(_TF_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.raises(ROSConfigError):
        camera_optical_tf_to_base(
            model=model, data=data, camera_name="missing", base_body_name="base"
        )


# ── PointCloud2 packing (needs sensor_msgs) ──────────────────────────────


def test_pointcloud2_from_points_xyz_packs_fields() -> None:
    pytest.importorskip("sensor_msgs")
    from openral_hal.depth_cloud import pointcloud2_from_points_xyz

    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    msg = pointcloud2_from_points_xyz(points, frame_id="cam_optical", stamp=None)
    assert msg.header.frame_id == "cam_optical"
    assert msg.height == 1
    assert msg.width == 2
    assert msg.point_step == 12  # 3 × float32
    assert msg.row_step == 24
    assert msg.is_dense is True
    assert [f.name for f in msg.fields] == ["x", "y", "z"]
    # Round-trip the raw buffer back to floats.
    flat = np.frombuffer(bytes(msg.data), dtype=np.float32)
    assert np.allclose(flat, points.ravel())


# ── full deploy-sim chain: SensorSpec → synth → PointCloud2 ───────────────

_CHAIN_MJCF = """
<mujoco model="depth_chain_test">
  <worldbody>
    <camera name="robot0_agentview_left" pos="0 0 0"/>
    <geom name="wall" type="box" pos="0 0 -1.5" size="5 5 0.1"/>
  </worldbody>
</mujoco>
"""


def test_sensorspec_to_pointcloud2_end_to_end() -> None:
    """Exactly what `_publish_depth_clouds` runs, minus the rclpy node."""
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("sensor_msgs")
    from openral_hal.depth_cloud import pointcloud2_from_points_xyz
    from openral_sim.backends.depth_camera import synthesize_depth_pointcloud

    model = mujoco.MjModel.from_xml_string(_CHAIN_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    spec = _depth_spec()  # metadata.mjcf_camera == "robot0_agentview_left"
    kwargs = depth_synth_kwargs(spec, max_range_default=8.0)
    points = synthesize_depth_pointcloud(model=model, data=data, stride=4, **kwargs)
    cloud = pointcloud2_from_points_xyz(points, frame_id=spec.frame_id, stamp=None)

    assert cloud.header.frame_id == "front_depth_optical_frame"
    assert cloud.width > 0  # the wall is in range → a non-empty cloud
    assert cloud.width == points.shape[0]
    # Wall near face at 1.4 m, within the spec's [0.2, 4.0] range.
    assert np.allclose(points[:, 2], 1.4, atol=1e-2)
