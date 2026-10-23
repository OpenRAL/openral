"""Surface retirement through real scene depth, native PCL and tracker refreshes."""

from __future__ import annotations

import numpy as np
import pytest
from openral_core import ObjectQuery, ObjectRole, ObjectSurfaceModelConfig
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_world_state.object_tracker import ObjectTracker
from openral_world_state.object_tracker.surface_model import ObjectSurfaceModels

from tests.unit._tracker_scenes import RIGS, sphere, step

pytest.importorskip("_openral_pcl_registration", reason="build native PCL registration module")


@pytest.mark.parametrize("rig_name", tuple(RIGS))
def test_tracker_clears_owned_surface_only_with_explicit_sensor_calibration(rig_name: str) -> None:
    rig = RIGS[rig_name]()
    x, y = rig.centre_xy
    centre = np.array((x, y, rig.support_z + 0.028))
    ball = sphere(tuple(centre), 0.028)
    cleared: list[int] = []
    for calibrated in (False, True):
        config = ObjectSurfaceModelConfig(
            voxel_m=0.003,
            max_points=8192,
            max_voxels=8192,
            max_iterations=50,
            max_correspondence_m=0.03,
            max_rmse_m=0.003,
            min_overlap=0.9,
            free_space_calibration=(
                {
                    "calibration_ref": f"analytic-scene-raycast:{rig_name}",
                    "sensor_ids": [rig.name],
                    "depth_error_m": 0.001,
                    "pose_error_m": 0.001,
                    "pixel_error": 1,
                    "max_age_s": 0.5,
                }
                if calibrated
                else None
            ),
        )
        tracker = ObjectTracker(rig.frame)
        tracker.surface_models = models = ObjectSurfaceModels(config)
        tracker.open_query(
            "g",
            "q0",
            ObjectQuery(label="orange", count="one", role=ObjectRole.TARGET),
            "obj:orange@0",
            stamp_ns=0,
        )
        step(tracker, rig, [ball], stamp_ns=1_000_000_000, wall_s=10)
        track = tracker.store.all()[0]
        points = models.world_points(track.object_id)
        pose = homogeneous_from_quat_xyz(track.box.pose.xyz, track.box.pose.quat_xyzw)
        # A previously accumulated stray surface in front of the real object.
        # A new full depth image independently sees past this owned voxel.
        ray = centre - rig.t[:3, 3]
        ghost = centre - ray / np.linalg.norm(ray) * 0.06
        result = models.observe(
            track.object_id, track.epoch, np.vstack((points, ghost)), pose, 1_500_000_000
        )
        assert result == "accepted"
        step(tracker, rig, [ball], stamp_ns=2_000_000_000, wall_s=20)
        diagnostic = models.diagnostics()[track.object_id]
        assert "stamp_ns=2000000000" in diagnostic
        cleared.append(int(diagnostic.split("cleared_cells=")[1]))
    assert cleared[0] == 0
    assert cleared[1] > 0
