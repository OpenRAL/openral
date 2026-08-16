# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the simulated depth-camera → point-cloud synth.

`synthesize_depth_pointcloud` casts one `mj_ray` per
(strided) pixel through a pinhole model and returns the hit points in the
camera *optical* frame (REP-103: +x right, +y down, +z forward). It is the
3-D analogue of `synthesize_laser_scan_2d` and is robot-agnostic: any named
MJCF camera in any robot's scene works.

Tests build a real `mujoco.MjModel` (a camera facing a known-distance wall)
— no mocks, no GL context — and assert the returned cloud's geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# The synth itself defers `mujoco`, so it imports without it; the cast
# functions need a real model, hence the module-level importorskip.
from openral_sim.backends.depth_camera import (
    synthesize_depth_image,
    synthesize_depth_pointcloud,
)

mujoco = pytest.importorskip("mujoco")

# A camera at the world origin (default orientation → looks down world -Z,
# +x right, +y up) facing a large fronto-parallel wall whose near face sits
# at z = -1.9 m. Every ray that hits the wall must report an optical-frame
# z of ~1.9 m regardless of pixel (the wall is perpendicular to the optical
# axis).
_WALL_Z = -2.0
_WALL_HALF_THICK = 0.1
_EXPECTED_DEPTH_M = -(_WALL_Z + _WALL_HALF_THICK)  # 1.9 m to the near face

_DEPTH_CAM_MJCF = f"""
<mujoco model="depth_cam_test">
  <worldbody>
    <camera name="depth0" pos="0 0 0"/>
    <geom name="wall" type="box" pos="0 0 {_WALL_Z}" size="5 5 {_WALL_HALF_THICK}"/>
  </worldbody>
</mujoco>
"""

# Small pinhole — fast to ray-cast, still exercises off-axis pixels.
_W, _H = 32, 24
_FX = _FY = 20.0
_CX, _CY = 16.0, 12.0


def _model_data() -> tuple[object, object]:
    model = mujoco.MjModel.from_xml_string(_DEPTH_CAM_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_depth_cloud_recovers_fronto_parallel_wall() -> None:
    model, data = _model_data()
    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
    )
    # Every pixel sees the wall, so the cloud is full (W*H points).
    assert points.shape == (_W * _H, 3)
    assert points.dtype == np.float32
    # Fronto-parallel wall → constant optical-frame depth across the image.
    assert np.allclose(points[:, 2], _EXPECTED_DEPTH_M, atol=1e-3)
    # Optical frame: +x right, +y down → both signs present, centred on 0.
    assert points[:, 0].min() < 0.0 < points[:, 0].max()
    assert points[:, 1].min() < 0.0 < points[:, 1].max()
    assert abs(float(points[:, 0].mean())) < 0.05
    assert abs(float(points[:, 1].mean())) < 0.05


def test_depth_cloud_back_projection_matches_pinhole() -> None:
    """A corner pixel back-projects to x=(u-cx)/fx * z, y=(v-cy)/fy * z."""
    model, data = _model_data()
    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
    )
    z = _EXPECTED_DEPTH_M
    # The (u=0, v=0) pixel is the first row-major ray.
    expected_x = (0.0 - _CX) / _FX * z
    expected_y = (0.0 - _CY) / _FY * z
    assert points[0, 0] == pytest.approx(expected_x, abs=2e-3)
    assert points[0, 1] == pytest.approx(expected_y, abs=2e-3)
    assert points[0, 2] == pytest.approx(z, abs=2e-3)


def test_depth_cloud_respects_max_range() -> None:
    model, data = _model_data()
    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=1.0,  # wall is at 1.9 m → out of range → no points
    )
    assert points.shape == (0, 3)


def test_depth_cloud_respects_min_range() -> None:
    model, data = _model_data()
    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
        min_range_m=3.0,  # wall closer than 3 m → dropped
    )
    assert points.shape == (0, 3)


def test_depth_cloud_stride_subsamples() -> None:
    model, data = _model_data()
    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
        stride=2,
    )
    expected = math.ceil(_W / 2) * math.ceil(_H / 2)
    assert points.shape == (expected, 3)


_SELF_FILTER_MJCF = """
<mujoco model="depth_self_filter">
  <worldbody>
    <camera name="depth0" pos="0 0 0"/>
    <body name="robot_arm" pos="0 0 -1.0">
      <geom name="arm" type="box" size="0.3 0.3 0.05"/>
    </body>
    <geom name="wall" type="box" pos="0 0 -2.0" size="5 5 0.1"/>
  </worldbody>
</mujoco>
"""

_SELF_FILTER_NO_BACKGROUND_MJCF = """
<mujoco model="depth_self_filter_no_background">
  <worldbody>
    <camera name="depth0" pos="0 0 0"/>
    <body name="robot_arm" pos="0 0 -1.0">
      <geom name="arm" type="box" size="0.3 0.3 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_depth_cloud_excludes_robot_body_hits() -> None:
    """exclude_body_ids drops hits on the robot's own bodies (self-filter).

    The camera sees a near 'robot_arm' box (z=-1) in front of the far wall
    (z=-2). Without exclusion the nearest hit is the arm (~1 m); excluding the
    arm body, every ray passes through to the wall (~1.9 m).
    """
    model = mujoco.MjModel.from_xml_string(_SELF_FILTER_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_arm")

    common = dict(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
    )
    without = synthesize_depth_pointcloud(**common)
    withf = synthesize_depth_pointcloud(**common, exclude_body_ids=frozenset({arm_body}))
    # The arm box is the closest thing in view, so without the filter the
    # nearest return is the arm near-face at ~0.95 m...
    assert float(without[:, 2].min()) == pytest.approx(0.95, abs=0.05)
    # ...with the arm excluded every ray passes through to the wall (~1.9 m),
    # and no return is closer than the wall.
    assert float(withf[:, 2].min()) == pytest.approx(1.9, abs=0.05)
    assert withf.shape == (_W * _H, 3)
    arm_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arm")
    assert int(model.geom_group[arm_geom]) == 0


def test_depth_cloud_emits_max_range_clearing_rays_behind_transparent_body() -> None:
    model = mujoco.MjModel.from_xml_string(_SELF_FILTER_NO_BACKGROUND_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_arm")

    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=5.0,
        exclude_body_ids=frozenset({arm_body}),
    )

    assert points.shape[0] > 0
    assert np.linalg.norm(points, axis=1).max() == pytest.approx(5.0, abs=1e-5)


def test_depth_cloud_unknown_camera_raises() -> None:
    from openral_core.exceptions import ROSConfigError

    model, data = _model_data()
    with pytest.raises(ROSConfigError, match="nonexistent"):
        synthesize_depth_pointcloud(
            model=model,
            data=data,
            camera_name="nonexistent",
            width=_W,
            height=_H,
            fx=_FX,
            fy=_FY,
            cx=_CX,
            cy=_CY,
            max_range_m=10.0,
        )


# ── synthesize_depth_image (dense raster for nvblox) ──────────────────────────


def test_depth_image_dense_fronto_parallel_wall() -> None:
    """The image is dense ``(H, W)`` and a fronto-parallel wall reads constant Z."""
    model, data = _model_data()
    depth = synthesize_depth_image(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
    )
    # Dense raster: (rows, cols) = (height, width) — every pixel kept.
    assert depth.shape == (_H, _W)
    assert depth.dtype == np.float32
    # Perpendicular optical-Z is constant across a fronto-parallel wall, and
    # equals the cloud synth's z-column (1.9 m to the near face).
    assert np.allclose(depth, _EXPECTED_DEPTH_M, atol=1e-3)


def test_depth_image_zero_for_misses() -> None:
    """Out-of-range / no-return pixels read exactly ``0.0`` (nvblox's sentinel)."""
    model, data = _model_data()
    depth = synthesize_depth_image(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=1.0,  # wall at 1.9 m → every ray out of range
    )
    assert depth.shape == (_H, _W)
    assert np.count_nonzero(depth) == 0


def test_depth_image_stride_scales_raster() -> None:
    """``stride=2`` rasterises ``ceil(H/2) x ceil(W/2)`` (CameraInfo scales to match)."""
    model, data = _model_data()
    depth = synthesize_depth_image(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
        stride=2,
    )
    assert depth.shape == (math.ceil(_H / 2), math.ceil(_W / 2))
    assert np.allclose(depth, _EXPECTED_DEPTH_M, atol=1e-3)


def test_depth_image_matches_cloud_z_column() -> None:
    """The dense image's nonzero pixels equal the cloud synth's optical-Z, in order.

    Both synths share ``_cast_depth_rays``; on a wall every ray hits, so the
    raster flattened row-major must equal the cloud's ``z`` column exactly.
    """
    model, data = _model_data()
    common = dict(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
    )
    cloud = synthesize_depth_pointcloud(**common)  # (W*H, 3), full wall
    depth = synthesize_depth_image(**common)  # (H, W)
    assert np.allclose(depth.reshape(-1), cloud[:, 2], atol=1e-5)


# ── Visual-only geometry must not be seen through (mj_multiRay regression) ──

# A RoboCasa counter in miniature: a cabinet body whose collision carcass tops
# out at world z 0.890, and a separate counter body carrying the countertop as
# a VISUAL-only slab (contype=0 conaffinity=0) whose top face is at 0.920, plus
# one small collision bracket off to the side. The bracket is what gives the
# counter body a BVH; MuJoCo builds that BVH from collision geoms only, so the
# body's bounding sphere hugs the bracket and excludes the slab entirely.
#
# `mj_multiRay` culls whole bodies against that sphere, so it used to skip the
# counter body and report the cabinet carcass beneath it — the counter surface
# read 30 mm too low, in the direction that makes the world-collision voxel
# grid think there is free space where a solid countertop is. `mj_ray` (and
# MuJoCo's own GL depth renderer, checked by hand on this scene) reports 0.920.
_CARCASS_TOP_Z = 0.890
_SLAB_TOP_Z = 0.920
_COUNTER_CAM_Z = 2.0

_COUNTER_MJCF = f"""
<mujoco model="visual_countertop">
  <worldbody>
    <camera name="depth0" pos="0 0 {_COUNTER_CAM_Z}"/>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="cabinet" pos="0 0 0">
      <geom name="carcass" type="box" size="0.5 0.5 0.445" pos="0 0 0.445"/>
    </body>
    <body name="counter_top" pos="0 0 0">
      <geom name="slab" type="box" size="0.6 0.6 0.015" pos="0 0 0.905"
            contype="0" conaffinity="0"/>
      <geom name="bracket" type="box" size="0.03 0.03 0.03" pos="2.0 0 0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


def _counter_model_data() -> tuple[object, object]:
    model = mujoco.MjModel.from_xml_string(_COUNTER_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_depth_cloud_hits_a_visual_only_surface_not_the_solid_behind_it() -> None:
    """A visual-only countertop is the surface, not the collision carcass under it."""
    model, data = _counter_model_data()
    # Narrow field of view so every ray lands on the slab (half-extent 0.6 m,
    # ~1.08 m below the camera).
    width = height = 9
    fx = fy = 40.0
    depth = synthesize_depth_image(
        model=model,
        data=data,
        camera_name="depth0",
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=width / 2.0,
        cy=height / 2.0,
        max_range_m=10.0,
    )

    expected = _COUNTER_CAM_Z - _SLAB_TOP_Z
    buried = _COUNTER_CAM_Z - _CARCASS_TOP_Z
    assert np.allclose(depth, expected, atol=1e-3), (
        f"every ray must stop on the visual countertop at optical-Z {expected:.3f} m; "
        f"{buried:.3f} m means the cast saw through it to the carcass beneath"
    )


def test_depth_cloud_sees_a_visual_geom_with_nothing_behind_it() -> None:
    """The same cull also loses the surface entirely when no solid backs it.

    Aimed past the cabinet at slab-only overhang, the wrong answer is not a
    30 mm error but the floor 0.92 m further away — a hole punched straight
    through a solid surface in the voxel map.
    """
    model, data = _counter_model_data()
    # Aim at x ≈ 0.55 m: past the carcass (half-extent 0.5) but still on the
    # slab (half-extent 0.6). Optical +x right → shift the principal point.
    width = height = 3
    fx = fy = 40.0
    depth = synthesize_depth_image(
        model=model,
        data=data,
        camera_name="depth0",
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=width / 2.0 - 0.55 * fx / (_COUNTER_CAM_Z - _SLAB_TOP_Z),
        cy=height / 2.0,
        max_range_m=10.0,
    )

    assert np.allclose(depth, _COUNTER_CAM_Z - _SLAB_TOP_Z, atol=5e-3), (
        "the overhanging countertop must still be the nearest surface; "
        f"{_COUNTER_CAM_Z:.3f} m is the floor seen through it"
    )


# ── the raster path must carry the self-filter's clearing rays too ────────────
#
# Since the one-cast refactor the deploy-sim bridge builds octomap_server's
# cloud from the RASTER (`synthesize_depth_image` → `points_from_depth_grid`),
# not from `synthesize_depth_pointcloud`. The raster's `0.0` sentinel cannot
# distinguish "no return" from "the only return was the robot's own body", so
# every clearing ray the self-filter produces vanished on the path that feeds
# the safety kernel: OctoMap got NO ray at all for the pixels covering the
# arm, and any cell in the robot's own silhouette kept whatever occupancy it
# held, forever. `synthesize_depth_frame` returns the mask alongside the
# raster so both outputs stay correct off one cast.


def _clearing_frame_kwargs(model: object, data: object) -> dict[str, object]:
    arm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_arm")
    return dict(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=5.0,
        exclude_body_ids=frozenset({int(arm_body)}),
    )


def test_depth_frame_marks_the_self_filtered_rays_the_raster_cannot_express() -> None:
    """The mask is exactly the pixels that read 0.0 *because* they hit the robot."""
    from openral_sim.backends.depth_camera import synthesize_depth_frame

    model = mujoco.MjModel.from_xml_string(_SELF_FILTER_NO_BACKGROUND_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    depth, clearing = synthesize_depth_frame(**_clearing_frame_kwargs(model, data))

    assert depth.shape == clearing.shape == (_H, _W)
    assert clearing.any(), "the arm fills part of the view; those rays are clearing rays"
    # Disjoint by construction: a cleared pixel never also carries a depth.
    assert not np.any((depth > 0.0) & clearing)
    # …and the raster itself is unchanged, so nvblox still sees "no measurement".
    assert np.array_equal(depth, synthesize_depth_image(**_clearing_frame_kwargs(model, data)))


def test_bridge_cloud_from_the_raster_equals_the_cloud_synth() -> None:
    """One cast, two outputs, zero drift: the published cloud loses nothing.

    This is the regression that let a stale voxel sit inside the arm's own
    silhouette and never be cleared — the bridge's cloud silently dropped
    every clearing ray the cloud synth emits.
    """
    from openral_hal.depth_cloud import points_from_depth_grid
    from openral_sim.backends.depth_camera import synthesize_depth_frame

    model = mujoco.MjModel.from_xml_string(_SELF_FILTER_NO_BACKGROUND_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    expected = synthesize_depth_pointcloud(**_clearing_frame_kwargs(model, data))
    depth, clearing = synthesize_depth_frame(**_clearing_frame_kwargs(model, data))

    published = points_from_depth_grid(
        depth, fx=_FX, fy=_FY, cx=_CX, cy=_CY, clearing=clearing, max_range_m=5.0
    )
    # Without the mask — the pre-fix bridge — every clearing endpoint is lost.
    lossy = points_from_depth_grid(depth, fx=_FX, fy=_FY, cx=_CX, cy=_CY)
    assert len(lossy) < len(expected), "fixture no longer exercises the clearing rays"

    assert published.shape == expected.shape
    assert np.allclose(published, expected, atol=1e-5)
    assert np.linalg.norm(published, axis=1).max() == pytest.approx(5.0, abs=1e-5)


def test_bridge_cloud_keeps_clearing_rays_alongside_real_returns() -> None:
    """A scene with a wall behind the arm keeps both kinds of ray in ray order."""
    from openral_hal.depth_cloud import points_from_depth_grid
    from openral_sim.backends.depth_camera import synthesize_depth_frame

    # Arm in front, wall only across the LEFT half of the view, so some rays
    # through the arm find a surface behind and some find nothing.
    xml = """
    <mujoco model="depth_self_filter_partial_background">
      <worldbody>
        <camera name="depth0" pos="0 0 0"/>
        <body name="robot_arm" pos="0 0 -1.0">
          <geom name="arm" type="box" size="0.3 0.3 0.05"/>
        </body>
        <geom name="wall" type="box" pos="-1.0 0 -2.0" size="0.6 5 0.1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    expected = synthesize_depth_pointcloud(**_clearing_frame_kwargs(model, data))
    depth, clearing = synthesize_depth_frame(**_clearing_frame_kwargs(model, data))
    assert (depth > 0.0).any() and clearing.any(), "fixture must exercise both kinds"

    published = points_from_depth_grid(
        depth, fx=_FX, fy=_FY, cx=_CX, cy=_CY, clearing=clearing, max_range_m=5.0
    )
    assert published.shape == expected.shape
    assert np.allclose(published, expected, atol=1e-5)


def test_points_from_depth_grid_refuses_a_clearing_mask_it_cannot_place() -> None:
    """A clearing ray with no range is a silent hole; fail loudly instead."""
    from openral_core.exceptions import ROSConfigError
    from openral_hal.depth_cloud import points_from_depth_grid

    grid = np.array([[0.0, 2.0]], dtype=np.float32)
    with pytest.raises(ROSConfigError, match="max_range_m"):
        points_from_depth_grid(
            grid, fx=4.0, fy=4.0, cx=1.0, cy=0.5, clearing=np.array([[True, False]])
        )
    with pytest.raises(ROSConfigError, match="does not match"):
        points_from_depth_grid(
            grid,
            fx=4.0,
            fy=4.0,
            cx=1.0,
            cy=0.5,
            clearing=np.array([[True], [False]]),
            max_range_m=4.0,
        )
