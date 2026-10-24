"""Actual PCL registration on public-scene object surfaces, never mocked transforms."""

from __future__ import annotations

import numpy as np
import pytest
from openral_core import ObjectQuery, ObjectRole, ObjectSurfaceModelConfig
from openral_core.exceptions import ROSConfigError
from openral_world_state.object_tracker import ObjectTracker
from openral_world_state.object_tracker.surface_model import ObjectSurfaceModels

from tests.unit._tracker_scenes import RIGS, box, sphere, step, surface_samples

pytest.importorskip(
    "_openral_pcl_registration", reason="build openral_world_state native PCL module"
)
spatial = pytest.importorskip("scipy.spatial", reason="surface-distance evaluation")


def _config(**updates: object) -> ObjectSurfaceModelConfig:
    return ObjectSurfaceModelConfig.model_validate(
        {
            "voxel_m": 0.003,
            "max_points": 2048,
            "max_voxels": 4096,
            "max_iterations": 50,
            "max_correspondence_m": 0.03,
            "max_rmse_m": 0.003,
            "min_overlap": 0.9,
            **updates,
        }
    )


def _surface() -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg] # reason: test arrays
    rig = RIGS["so101_top"]()
    x, y = rig.centre_xy
    center = (x, y, rig.support_z + 0.025)
    points = surface_samples(box(center, (0.05, 0.035, 0.025), 0.0))
    pose = np.eye(4)
    pose[:3, 3] = center
    return points, pose


def test_motion_relocates_the_persistent_surface_without_a_world_trail() -> None:
    points, pose = _surface()
    models = ObjectSurfaceModels(_config())
    assert models.observe("trk:1", 0, points, pose, 10) == "accepted"
    shift = np.array((0.006, -0.004, 0.003))
    assert models.observe("trk:1", 0, points + shift, pose, 20) == "accepted"
    world = models.world_points("trk:1")
    distance, _ = spatial.cKDTree(points + shift).query(world)
    assert distance.max() < 0.003
    assert "stamp_ns=20" in models.diagnostics()["trk:1"]


def test_partial_observation_does_not_carve_previous_surfaces_free() -> None:
    points, pose = _surface()
    models = ObjectSurfaceModels(_config())
    models.observe("trk:1", 0, points, pose, 10)
    before = models.world_points("trk:1")
    partial = points[points[:, 0] >= pose[0, 3] - 0.01]
    assert models.observe("trk:1", 0, partial, pose, 20) == "accepted"
    assert len(models.world_points("trk:1")) >= len(before)


def test_rigid_rotation_updates_the_object_pose() -> None:
    points, pose = _surface()
    models = ObjectSurfaceModels(_config())
    models.observe("trk:1", 0, points, pose, 10)
    angle = 0.04
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    moved = (points - pose[:3, 3]) @ rotation.T + pose[:3, 3] + (0.002, 0, 0)
    assert models.observe("trk:1", 0, moved, pose, 20) == "accepted"
    distances, _ = spatial.cKDTree(moved).query(models.world_points("trk:1"))
    assert distances.max() < 0.003


def test_bad_registration_and_old_samples_do_not_change_pose_or_model() -> None:
    points, pose = _surface()
    models = ObjectSurfaceModels(_config())
    models.observe("trk:1", 0, points, pose, 10)
    before = models.world_points("trk:1").copy()
    assert models.observe("trk:1", 0, points + 2.0, pose, 20) == "registration_rejected"
    np.testing.assert_array_equal(models.world_points("trk:1"), before)
    assert "stamp_ns=10" in models.diagnostics()["trk:1"]
    assert models.observe("trk:1", 0, points, pose, 9) == "stale"
    assert models.observe("trk:1", 0, points, pose, 10) == "stale"


def test_generation_reset_capacity_and_track_retirement_are_bounded() -> None:
    points, pose = _surface()
    models = ObjectSurfaceModels(_config(max_voxels=16))
    assert models.observe("trk:1", 0, points, pose, 10) == "model_capacity"
    assert len(models.world_points("trk:1")) == 16
    # New lifecycle cannot inherit a previous instance's shape or frame.
    pose2 = pose.copy()
    pose2[:3, 3] += 1.0
    assert models.observe("trk:1", 1, points + 1.0, pose2, 20) == "model_capacity"
    assert models.observe("trk:1", 0, points, pose, 30) == "stale"
    models.retain(set())
    assert models.diagnostics() == {}


def test_small_point_budget_keeps_the_native_registration_minimum() -> None:
    points, pose = _surface()
    models = ObjectSurfaceModels(_config(max_points=4))
    assert len(models._sample(points[:9])) == 4
    assert models.observe("trk:1", 0, points[:9], pose, 10) == "accepted"
    assert len(models.world_points("trk:1")) >= 4
    assert models.observe("trk:1", 0, points[:9], pose, 20) == "accepted"


def test_an_undersized_voxel_target_reseeds_from_the_current_tracker_pose() -> None:
    points, pose = _surface()
    models = ObjectSurfaceModels(_config(voxel_m=0.01))
    centre = pose[:3, 3]
    clustered = centre + np.array(
        [
            (0.0010, 0.0010, 0.0010),
            (0.0011, 0.0010, 0.0010),
            (0.0010, 0.0011, 0.0010),
            (0.0010, 0.0010, 0.0011),
            (0.0012, 0.0011, 0.0010),
            (0.0010, 0.0012, 0.0011),
            (0.0011, 0.0010, 0.0012),
            (0.0012, 0.0012, 0.0012),
            (0.0013, 0.0011, 0.0012),
        ]
    )
    assert models.observe("trk:1", 0, clustered, pose, 10) == "accepted"
    assert len(models.world_points("trk:1")) < 4
    assert models.observe("trk:1", 0, points, pose, 20) == "model_reseeded"
    assert len(models.world_points("trk:1")) >= 4
    assert "stamp_ns=20" in models.diagnostics()["trk:1"]


def test_nonrigid_or_invalid_inputs_are_rejected_at_the_boundary() -> None:
    points, pose = _surface()
    models = ObjectSurfaceModels(_config())
    pose[0, 0] = 2.0
    with pytest.raises(ROSConfigError, match="rigid"):
        models.observe("trk:1", 0, points, pose, 10)
    pose[0, 0] = 1.0
    points[0, 0] = float("nan")
    with pytest.raises(ROSConfigError, match="finite"):
        models.observe("trk:1", 0, points, pose, 10)


def test_out_of_range_voxels_do_not_commit_a_registered_pose() -> None:
    points, pose = _surface()
    models = ObjectSurfaceModels(_config())
    models.observe("trk:1", 0, points, pose, 10)
    before = models.world_points("trk:1").copy()
    models.config = _config(voxel_m=1e-300)
    with pytest.raises(ROSConfigError, match="voxel indices"):
        models.observe("trk:1", 0, points + 0.001, pose, 20)
    np.testing.assert_array_equal(models.world_points("trk:1"), before)
    assert "stamp_ns=10" in models.diagnostics()["trk:1"]


def test_native_boundary_validates_buffers_and_pose_without_numpy_abi_assumptions() -> None:
    import _openral_pcl_registration as registration  # type: ignore[import-not-found]  # reason: no stub

    points, pose = _surface()
    for bad in (points.astype(np.float32), points[:, :2], points[::2]):
        with pytest.raises(ValueError, match="contiguous float64"):
            registration.align(bad, points, pose, 0.03, 20)
    unaligned = np.frombuffer(b"x" + points.tobytes(), dtype=np.float64, offset=1).reshape(-1, 3)
    assert not unaligned.flags.aligned
    assert registration.align(unaligned, points, np.eye(4), 0.03, 20)[1]
    pose[0, 0] = -1.0
    with pytest.raises(ValueError, match="rigid"):
        registration.align(points, points, pose, 0.03, 20)


def test_tracker_pilot_preserves_the_existing_containing_geometry() -> None:
    rig = RIGS["so101_top"]()
    plain, pilot = ObjectTracker(rig.frame), ObjectTracker(rig.frame)
    pilot.surface_models = ObjectSurfaceModels(_config())
    for tracker in (plain, pilot):
        tracker.open_query(
            "g",
            "q0",
            ObjectQuery(label="orange", count="one", role=ObjectRole.TARGET),
            "obj:orange@0",
            stamp_ns=0,
        )
    for index, dx in enumerate((0.0, 0.004), start=1):
        x, y = rig.centre_xy
        ball = sphere((x + dx, y, rig.support_z + 0.028), 0.028)
        for tracker in (plain, pilot):
            step(tracker, rig, [ball], stamp_ns=index * 1_000_000_000, wall_s=index * 10.0)
        assert plain.store.snapshot(index).model_dump() == pilot.store.snapshot(index).model_dump()
    assert pilot.surface_models.diagnostics()


@pytest.mark.parametrize("rig_name", tuple(RIGS))
def test_owned_cells_clear_only_from_independent_free_depth(rig_name: str) -> None:
    from tests.unit._tracker_scenes import render

    rig = RIGS[rig_name]()
    x, y = rig.centre_xy
    centre = (x, y, rig.support_z + 0.035)
    points = surface_samples(sphere(centre, 0.025))
    pose = np.eye(4)
    pose[:3, 3] = centre
    models = ObjectSurfaceModels(_config(max_points=8192))
    models.observe("trk:1", 0, points, pose, 10)
    models.observe("trk:2", 0, points, pose, 10)
    neighbour = models.world_points("trk:2").copy()
    depth, _ = render([], rig)
    before = len(models.world_points("trk:1"))
    removed = models.clear_observed_free(
        "trk:1",
        0,
        10,
        depth,
        np.zeros(depth.shape, np.uint8),
        rig.k,
        rig.t,
        now_ns=10,
        max_age_ns=5,
        depth_error_m=0.001,
        pose_error_m=0.001,
        pixel_error=1,
        min_depth_m=0.05,
        max_depth_m=5.0,
    )
    assert removed > 0
    assert len(models.world_points("trk:1")) == before - removed
    np.testing.assert_array_equal(models.world_points("trk:2"), neighbour)
    # Repeating a frame cannot reset the bounded pixel budget.
    assert (
        models.clear_observed_free(
            "trk:1",
            0,
            10,
            depth,
            np.zeros(depth.shape, np.uint8),
            rig.k,
            rig.t,
            now_ns=10,
            max_age_ns=5,
            depth_error_m=0.001,
            pose_error_m=0.001,
            pixel_error=1,
            min_depth_m=0.05,
            max_depth_m=5.0,
        )
        == 0
    )
    models.retain({"trk:2"})
    np.testing.assert_array_equal(models.world_points("trk:2"), neighbour)
    with pytest.raises(KeyError):
        models.world_points("trk:1")


@pytest.mark.parametrize(
    "failure",
    (
        "missing",
        "nan",
        "finite_invalid",
        "occluded",
        "robot",
        "stale",
        "future",
        "epoch",
        "old_pose",
        "mask_missing",
        "offimage",
        "uncertainty",
    ),
)
def test_unknown_or_unsynchronised_free_evidence_keeps_owned_cells(failure: str) -> None:
    from tests.unit._tracker_scenes import render

    points, pose = _surface()
    rig = RIGS["so101_top"]()
    models = ObjectSurfaceModels(_config(max_points=8192))
    models.observe("trk:1", 0, points, pose, 10)
    before = models.world_points("trk:1").copy()
    depth, _ = render([], rig)
    mask = np.zeros(depth.shape, np.uint8)
    if failure == "missing":
        depth[:] = 0
    elif failure == "nan":
        depth[:] = np.nan
    elif failure == "finite_invalid":
        depth[:] = 999.0
    elif failure == "occluded":
        depth[:] = 0.01
    elif failure == "robot":
        mask[:] = 128
    camera_pose = rig.t.copy()
    if failure == "offimage":
        camera_pose[:3, 3] += (100, 100, 100)
    removed = models.clear_observed_free(
        "trk:1",
        1 if failure == "epoch" else 0,
        9 if failure == "old_pose" else 10,
        depth,
        None if failure == "mask_missing" else mask,
        rig.k,
        camera_pose,
        now_ns=20 if failure == "stale" else 9 if failure == "future" else 10,
        max_age_ns=5,
        depth_error_m=0.001,
        pose_error_m=10.0 if failure == "uncertainty" else 0.001,
        pixel_error=1,
        min_depth_m=0.05,
        max_depth_m=5.0,
    )
    assert removed == 0
    np.testing.assert_array_equal(models.world_points("trk:1"), before)


def test_explicit_uncertainty_and_pixel_budget_cannot_license_extra_clearing() -> None:
    from tests.unit._tracker_scenes import render

    points, pose = _surface()
    rig = RIGS["so101_top"]()
    depth, _ = render([], rig)
    mask = np.zeros(depth.shape, np.uint8)
    models = ObjectSurfaceModels(_config(max_points=4))
    models.observe("trk:1", 0, points, pose, 10)
    before = models.world_points("trk:1").copy()
    assert (
        models.clear_observed_free(
            "trk:1",
            0,
            10,
            depth,
            mask,
            rig.k,
            rig.t,
            now_ns=10,
            max_age_ns=5,
            depth_error_m=0.001,
            pose_error_m=0.001,
            pixel_error=1,
            min_depth_m=0.05,
            max_depth_m=5.0,
        )
        == 0
    )
    np.testing.assert_array_equal(models.world_points("trk:1"), before)
    with pytest.raises(ROSConfigError, match="error bounds"):
        models.clear_observed_free(
            "trk:1",
            0,
            10,
            depth,
            mask,
            rig.k,
            rig.t,
            now_ns=10,
            max_age_ns=5,
            depth_error_m=-1,
            pose_error_m=0.001,
            pixel_error=1,
            min_depth_m=0.05,
            max_depth_m=5.0,
        )
