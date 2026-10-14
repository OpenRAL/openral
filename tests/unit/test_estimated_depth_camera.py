# SPDX-License-Identifier: Apache-2.0
"""DA3 metric depth from an RGB camera feeds the world-collision map (sim and real).

``DeployRuntime.estimated_depth_camera`` names the camera; the deploy resolves its cloud
as octomap's input, refuses a camera whose cloud could not be placed, and on a real
deploy requires its mount like any cloud source. Any RGB camera qualifies (top, context,
wrist) once a declared mount ties it to the robot — unless the deploy has a measured cloud
source (a depth camera, a 3D lidar): an estimate never replaces a measurement (Safety-WG,
hazard HZ-0382-6). ``openral_perception_ros.metric_depth``
turns DA3METRIC's canonical output into metres through the camera's own focal.

Real manifests and scenes (CLAUDE.md §1.11); the provider node itself runs in
``tests/integration/test_estimated_depth_provider_live.py`` against a fake sidecar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from openral_core import (
    DeployRuntime,
    RobotDescription,
    apply_sensor_overlays,
    load_robot_unit,
)
from openral_core.depth_extrinsic import (
    depth_extrinsic_problems,
    resolve_estimated_depth_camera,
)
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    SensorDeployBinding,
    SensorReaderBackend,
    SensorSpec,
    deploy_cloud_topic,
)
from openral_perception_ros.metric_depth import (
    DA3_CANONICAL_FOCAL_PX,
    camera_info_problem,
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


class TestCameraMount:
    """Any RGB camera tied to the robot by a declared mount; sim adds the bridge's live TF."""

    def _vslam(self) -> list:  # type: ignore[type-arg]
        """panda_mobile_vslam without ``front_depth``: the mount rules apply only then."""
        return [s for s in RobotDescription.from_yaml(str(_VSLAM)).sensors if not s.is_cloud_source]

    def test_a_sim_optical_camera_is_placed_by_the_bridge(self) -> None:
        mount = resolve_estimated_depth_camera(self._vslam(), "shoulder_right", hal_mode="sim")
        assert mount == ("shoulder_right_optical_frame", None, "shoulder_right", [])

    def test_the_same_camera_has_no_mount_on_a_real_deploy(self) -> None:
        mount = resolve_estimated_depth_camera(self._vslam(), "shoulder_right", hal_mode="real")
        assert any("no mount ties it to the robot" in p for p in mount.problems), mount.problems

    @pytest.mark.parametrize(
        ("robot", "camera", "measured"),
        [
            ("panda_mobile_vslam", "shoulder_right", "front_depth"),
            ("openarm", "top", "head_zed"),  # top is the ZED's colour; its depth is measured
        ],
    )
    def test_a_measured_cloud_source_refuses_the_estimate(
        self, robot: str, camera: str, measured: str
    ) -> None:
        """An estimate never replaces a measurement (hazard HZ-0382-6), in sim and real."""
        sensors = RobotDescription.from_yaml(str(_REPO / f"robots/{robot}/robot.yaml")).sensors
        for mode in ("sim", "real"):
            problems = resolve_estimated_depth_camera(sensors, camera, hal_mode=mode).problems
            assert problems and problems[0].startswith(f"{camera}: REFUSED"), problems
            assert f"robot sensor {measured!r}" in problems[0], problems
            assert "no override" in problems[0], problems

    def test_a_scene_mounted_measured_sensor_refuses_it_too_and_is_named_as_such(self) -> None:
        robot = RobotDescription.from_yaml(str(_VSLAM)).sensors
        rgb_only = [s for s in robot if not s.is_cloud_source]
        lidar = SensorSpec(
            name="cell_lidar",
            modality="point_cloud",
            frame_id="cell_lidar",
            parent_frame="base_link",
            static_transform_xyz_rpy=(1.0, 0.0, 1.5, 0.0, 0.0, 0.0),
            rate_hz=10.0,
        )
        (problem,) = resolve_estimated_depth_camera(
            [*rgb_only, lidar],
            "shoulder_right",
            hal_mode="sim",
            scene_sensor_names={"cell_lidar"},
        ).problems
        assert "scene sensor 'cell_lidar' (point_cloud)" in problem

    def test_a_camera_sharing_a_non_cloud_mount_is_placed_in_sim_and_real(self) -> None:
        """OpenArm's ``top`` rides ``head_zed``'s mount; with that ZED wired colour-only
        (no depth stream, so no measured cloud) the estimate is placed on its mount."""
        robot = str(_REPO / "robots/openarm/robot.yaml")
        # Real: Thor's unit overlay gives `top` its ros2_image binding (the driver's K).
        thor = apply_sensor_overlays(
            RobotDescription.from_yaml(robot).sensors, load_robot_unit(robot, "thor").sensors
        )
        for mode in ("sim", "real"):
            arm = [
                s.model_copy(update={"modality": "rgb"}) if s.name == "head_zed" else s
                for s in (thor if mode == "real" else RobotDescription.from_yaml(robot).sensors)
            ]
            mount = resolve_estimated_depth_camera(arm, "top", hal_mode=mode)
            assert mount == ("zed_camera_link_optical_frame", "zed_camera_link", "head_zed", [])

    def test_a_real_camera_without_a_driver_calibration_is_refused(self) -> None:
        """A real deploy takes DA3's focal from the driver's CameraInfo, never the manifest's
        K (OpenArm's `top` K is a sim stand-in): no ros2_image binding, no estimate."""
        arm = [
            s.model_copy(update={"modality": "rgb"}) if s.name == "head_zed" else s
            for s in RobotDescription.from_yaml(str(_REPO / "robots/openarm/robot.yaml")).sensors
        ]
        assert resolve_estimated_depth_camera(arm, "top", hal_mode="sim").problems == []
        (problem,) = resolve_estimated_depth_camera(arm, "top", hal_mode="real").problems
        assert "has no deploy_binding" in problem
        assert "ros2_image deploy_binding" in problem
        opencv = [
            s.model_copy(
                update={
                    "deploy_binding": SensorDeployBinding(backend=SensorReaderBackend.OPENCV_THREAD)
                }
            )
            if s.name == "top"
            else s
            for s in arm
        ]
        (problem,) = resolve_estimated_depth_camera(opencv, "top", hal_mode="real").problems
        assert "has a opencv_thread binding" in problem

    @pytest.mark.parametrize(
        ("robot", "camera", "match"),
        [
            ("panda_mobile_vslam", "front_depth", "depth sensor"),
            ("panda_mobile_vslam", "wrist", "no mount ties"),  # panda_hand is not the camera
            ("panda_mobile_vslam", "nope", "no such sensor"),
            ("so101_follower", "top", "no mount ties"),  # declared in `world`
            ("so101_follower", "wrist", "no mount ties"),  # parent gripper, no offset
        ],
    )
    def test_a_camera_nobody_placed_is_refused(self, robot: str, camera: str, match: str) -> None:
        sensors = RobotDescription.from_yaml(str(_REPO / f"robots/{robot}/robot.yaml")).sensors
        for mode in ("sim", "real"):
            problems = resolve_estimated_depth_camera(sensors, camera, hal_mode=mode).problems
            assert any(match in p for p in problems), (mode, problems)

    def test_a_real_deploy_checks_the_mount_the_camera_rides(self) -> None:
        robot = str(_REPO / "robots/openarm/robot.yaml")
        desc = RobotDescription.from_yaml(robot)
        nominal = depth_extrinsic_problems(desc, [], estimated_camera="top")
        assert [p.split(":")[0] for p in nominal] == ["head_zed"]
        thor = load_robot_unit(robot, "thor").sensors
        calibrated = desc.model_copy(update={"sensors": apply_sensor_overlays(desc.sensors, thor)})
        assert depth_extrinsic_problems(calibrated, thor, estimated_camera="top") == []


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

    @pytest.mark.parametrize(
        ("k", "info_wh", "image_wh", "match"),
        [
            # A driver's full-resolution CameraInfo for a downscaled republish: usable.
            ((1498.0, 1498.0, 959.5, 539.5), (1920, 1080), (960, 540), None),
            ((1498.0, 1498.0, 959.5, 539.5), (1920, 1080), (640, 361), None),  # rounding
            ((500.0, 500.0, 319.5, 239.5), (640, 360), (640, 480), "is not a resize"),
            ((0.0, 0.0, 0.0, 0.0), (640, 480), (640, 480), "not a positive finite"),
            ((500.0, 0.0, 319.5, 239.5), (640, 480), (640, 480), "not a positive finite"),
            ((float("nan"), 500.0, 319.5, 239.5), (640, 480), (640, 480), "not a positive"),
            ((500.0, 500.0, 319.5, 239.5), (0, 0), (640, 480), "empty size"),
        ],
    )
    def test_a_camera_info_that_cannot_scale_the_image_is_named(
        self,
        k: tuple[float, float, float, float],
        info_wh: tuple[int, int],
        image_wh: tuple[int, int],
        match: str | None,
    ) -> None:
        problem = camera_info_problem(k, info_wh, image_wh)
        if match is None:
            assert problem is None
        else:
            assert problem is not None and match in problem


class TestDeployResolution:
    """The shipped mono scene's robot, as shipped (``front_depth``) and without it."""

    def _rgb_only_robot(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """panda_mobile_vslam minus ``front_depth``, via ``$OPENRAL_ROBOTS_DIR``."""
        src = _VSLAM.parent
        out = tmp_path / "robots" / src.name
        out.mkdir(parents=True)
        manifest = yaml.safe_load(_VSLAM.read_text())
        manifest["sensors"] = [s for s in manifest["sensors"] if s["name"] != "front_depth"]
        (out / "robot.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        for f in src.iterdir():
            if f.name != "robot.yaml":
                (out / f.name).write_bytes(f.read_bytes())
        monkeypatch.setenv("OPENRAL_ROBOTS_DIR", str(tmp_path / "robots"))

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

    def test_the_shipped_robots_depth_camera_refuses_it_before_launch(self, tmp_path: Path) -> None:
        with pytest.raises(ROSConfigError, match=r"REFUSED.*robot sensor 'front_depth' \(depth\)"):
            self._resolve(self._scene(tmp_path, "shoulder_right"))

    def test_the_scene_field_reaches_the_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._rgb_only_robot(tmp_path, monkeypatch)
        inv = self._resolve(self._scene(tmp_path, "shoulder_right"))
        assert inv.estimated_depth_camera == "shoulder_right"  # type: ignore[attr-defined]
        assert "estimated_depth_camera:=shoulder_right" in " ".join(inv.argv_template)  # type: ignore[attr-defined]

    def test_an_unplaceable_camera_is_refused_before_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._rgb_only_robot(tmp_path, monkeypatch)
        with pytest.raises(ROSConfigError, match="no mount ties"):
            self._resolve(self._scene(tmp_path, "wrist"))
