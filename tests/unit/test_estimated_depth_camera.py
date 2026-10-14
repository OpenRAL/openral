# SPDX-License-Identifier: Apache-2.0
"""DA3 metric depth from an RGB camera feeds the world-collision map (sim and real).

``DeployRuntime.estimated_depth_camera`` names the camera; the deploy resolves its cloud
as octomap's input, refuses a camera whose cloud could not be placed, and on a real
deploy requires its mount like any cloud source. ``openral_perception_ros.metric_depth``
turns DA3METRIC's canonical output into metres through the camera's own focal.

Real manifests and scenes (CLAUDE.md §1.11); the provider node itself runs in
``tests/integration/test_estimated_depth_provider_live.py`` against a fake sidecar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openral_core import DeployRuntime, RobotDescription
from openral_core.depth_extrinsic import (
    depth_extrinsic_problems,
    estimated_depth_camera_problems,
)
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import deploy_cloud_topic
from openral_perception_ros.metric_depth import (
    DA3_CANONICAL_FOCAL_PX,
    da3_metric_depth,
    points_from_depth,
    scale_intrinsics,
)
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[2]
_VSLAM = _REPO / "robots" / "panda_mobile_vslam" / "robot.yaml"


class TestRuntimeField:
    def test_off_by_default(self) -> None:
        assert DeployRuntime().estimated_depth_camera is None

    @pytest.mark.parametrize(
        ("fields", "match"),
        [
            ({"estimated_depth_camera": ""}, "non-empty"),
            (
                {"estimated_depth_camera": "top", "octomap_cloud_topic": "/zed/points"},
                "set one",
            ),
            (
                {"estimated_depth_camera": "shoulder_left", "slam_mono_camera": "shoulder_left"},
                "twice",
            ),
        ],
    )
    def test_ambiguous_choices_are_refused(self, fields: dict[str, str], match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            DeployRuntime(**fields)  # type: ignore[arg-type]

    def test_its_cloud_is_the_map_input_sim_and_real(self) -> None:
        sensors = RobotDescription.from_yaml(str(_VSLAM)).sensors
        for mode in ("sim", "real"):
            topic = deploy_cloud_topic(
                sensors, pinned=None, hal_mode=mode, estimated_camera="shoulder_right"
            )
            assert topic == "/openral/cameras/shoulder_right/points"


class TestCameraChecks:
    def _sensors(self) -> list:  # type: ignore[type-arg]
        return list(RobotDescription.from_yaml(str(_VSLAM)).sensors)

    def test_an_optical_rgb_camera_with_intrinsics_is_accepted(self) -> None:
        assert estimated_depth_camera_problems(self._sensors(), "shoulder_right") == []

    @pytest.mark.parametrize(
        ("camera", "match"),
        [
            ("front_depth", "depth sensor"),
            ("wrist", "not an \\*_optical_frame"),
            ("nope", "no such sensor"),
        ],
    )
    def test_a_camera_whose_cloud_cannot_be_placed_is_refused(
        self, camera: str, match: str
    ) -> None:
        import re

        problems = estimated_depth_camera_problems(self._sensors(), camera)
        assert problems and re.search(match, problems[0])

    def test_a_top_camera_framed_in_world_is_refused(self) -> None:
        """SO-101's ``top`` is declared in ``world``: no TF places a cloud there."""
        so101 = RobotDescription.from_yaml(str(_REPO / "robots/so101_follower/robot.yaml"))
        (problem,) = estimated_depth_camera_problems(so101.sensors, "top")
        assert "'world'" in problem

    def test_a_real_deploy_needs_its_mount_like_any_cloud_source(self) -> None:
        desc = RobotDescription.from_yaml(str(_VSLAM))
        without = depth_extrinsic_problems(desc, [])
        with_cam = depth_extrinsic_problems(desc, [], estimated_camera="shoulder_right")
        assert not any(p.startswith("shoulder_right:") for p in without)
        assert any(p.startswith("shoulder_right:") for p in with_cam)


class TestMetricDepth:
    def test_a_point_survives_the_round_trip_through_da3s_raster(self) -> None:
        """Camera K at 640x480, DA3 raster 504x378: the cloud lands where the point is."""
        k_cam = (520.0, 515.0, 322.0, 241.0)
        point = np.array([0.21, -0.12, 1.4])
        dw, dh = 504, 378
        fx, fy, cx, cy = scale_intrinsics(k_cam, (640, 480), (dw, dh))
        col = round(fx * point[0] / point[2] + cx)
        row = round(fy * point[1] / point[2] + cy)
        canonical = np.zeros((dh, dw), dtype=np.float32)
        # What DA3METRIC predicts for a surface at depth z, seen with focal f at its raster.
        canonical[row, col] = point[2] * DA3_CANONICAL_FOCAL_PX / (0.5 * (fx + fy))
        depth = da3_metric_depth(canonical, fx=fx, fy=fy)
        (got,) = points_from_depth(depth, fx=fx, fy=fy, cx=cx, cy=cy, max_range_m=3.0)
        # Within one raster pixel of the true point at that range.
        assert np.allclose(got, point, atol=point[2] / fx)

    def test_far_and_invalid_returns_are_dropped(self) -> None:
        depth = da3_metric_depth(np.array([[300.0, np.nan, 3000.0]]), fx=300.0, fy=300.0)
        assert depth.tolist() == [[300.0, 0.0, 3000.0]]
        pts = points_from_depth(depth / 100.0, fx=1.0, fy=1.0, cx=0.0, cy=0.0, max_range_m=5.0)
        assert pts[:, 2].tolist() == [3.0]


class TestDeployResolution:
    def _scene(self, tmp_path: Path, camera: str) -> Path:
        text = (_REPO / "scenes" / "deploy" / "robocasa_vslam_mono.yaml").read_text()
        text = text.replace(
            "\n  enable_octomap: false\n",
            f"\n  enable_octomap: true\n  estimated_depth_camera: {camera}\n",
        )
        out = tmp_path / "estimated.yaml"
        out.write_text(text)
        return out

    def _resolve(self, scene: Path) -> object:
        from openral_cli.deploy_sim import resolve_launch_invocation

        return resolve_launch_invocation(
            config=scene,
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_param_overrides=None,
        )

    def test_the_scene_field_reaches_the_launch(self, tmp_path: Path) -> None:
        inv = self._resolve(self._scene(tmp_path, "shoulder_right"))
        assert inv.estimated_depth_camera == "shoulder_right"  # type: ignore[attr-defined]
        assert "estimated_depth_camera:=shoulder_right" in " ".join(inv.argv_template)  # type: ignore[attr-defined]

    def test_an_unplaceable_camera_is_refused_before_launch(self, tmp_path: Path) -> None:
        with pytest.raises(ROSConfigError, match="optical_frame"):
            self._resolve(self._scene(tmp_path, "wrist"))
