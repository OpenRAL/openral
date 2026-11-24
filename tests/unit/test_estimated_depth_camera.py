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
    RSkillManifest,
)
from openral_core.depth_extrinsic import (
    auto_estimated_depth_camera,
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
    camera_info_problem,
    metres_from_output,
    points_from_depth,
    scale_intrinsics,
)
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[2]
_DA3 = RSkillManifest.from_yaml(
    str(_REPO / "rskills/rskill-da3metric_large-any-metric_depth-bf16/rskill.yaml")
).depth_estimator
assert _DA3 is not None and _DA3.canonical_focal_px is not None
_DA3_FOCAL = _DA3.canonical_focal_px
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
            DeployRuntime(**fields)

    def test_a_depth_rskill_needs_a_camera_or_octomap_to_pick_one(self) -> None:
        skill = "rskill-depth_pro2_s-any-metric_depth-bf16"
        with pytest.raises(ValidationError, match="needs estimated_depth_camera"):
            DeployRuntime(estimated_depth_rskill=skill)
        assert DeployRuntime(estimated_depth_rskill=skill, enable_octomap=True).enable_octomap

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
        # Real: the manifest gives `top` its ros2_image binding (the driver's K).
        sensors = RobotDescription.from_yaml(robot).sensors
        for mode in ("sim", "real"):
            arm = [
                s.model_copy(update={"modality": "rgb"}) if s.name == "head_zed" else s
                for s in sensors
            ]
            mount = resolve_estimated_depth_camera(arm, "top", hal_mode=mode)
            assert mount == ("zed_camera_link_optical_frame", "zed_camera_link", "head_zed", [])

    def _arm_rgb_only(self) -> list:  # type: ignore[type-arg]
        """OpenArm with ``head_zed`` wired colour-only: ``top`` rides its mount, no cloud."""
        return [
            s.model_copy(update={"modality": "rgb"}) if s.name == "head_zed" else s
            for s in RobotDescription.from_yaml(str(_REPO / "robots/openarm/robot.yaml")).sensors
        ]

    def _bind(self, sensors: list, binding: SensorDeployBinding) -> list:  # type: ignore[type-arg]
        return [
            s.model_copy(update={"deploy_binding": binding}) if s.name == "top" else s
            for s in sensors
        ]

    def test_a_real_camera_needs_a_binding_to_publish(self) -> None:
        arm = [
            s.model_copy(update={"deploy_binding": None}) if s.name == "top" else s
            for s in self._arm_rgb_only()
        ]
        assert resolve_estimated_depth_camera(arm, "top", hal_mode="sim").problems == []
        (problem,) = resolve_estimated_depth_camera(arm, "top", hal_mode="real").problems
        assert "no deploy_binding" in problem

    def test_a_real_camera_needs_a_measured_calibration(self) -> None:
        """DA3's metres scale with the focal: OpenArm's `top` K is a sim stand-in, so the
        manifest's nominal intrinsics are refused; the driver's CameraInfo (ros2_image) or
        measured intrinsics (a unit overlay's, or intrinsics_calibrated) are accepted."""
        opencv = self._bind(
            self._arm_rgb_only(), SensorDeployBinding(backend=SensorReaderBackend.OPENCV_THREAD)
        )
        (problem,) = resolve_estimated_depth_camera(opencv, "top", hal_mode="real").problems
        assert "manifest's nominal value" in problem and "opencv_thread" in problem
        calibrated = resolve_estimated_depth_camera(
            opencv, "top", hal_mode="real", calibrated_intrinsics={"top"}
        )
        assert calibrated.problems == []
        flagged = [
            s.model_copy(update={"intrinsics_calibrated": True}) if s.name == "top" else s
            for s in opencv
        ]
        assert resolve_estimated_depth_camera(flagged, "top", hal_mode="real").problems == []
        driver = self._bind(
            self._arm_rgb_only(),
            SensorDeployBinding(
                backend=SensorReaderBackend.ROS2_IMAGE, backend_params={"topic": "/cam/image_raw"}
            ),
        )
        assert resolve_estimated_depth_camera(driver, "top", hal_mode="real").problems == []

    def test_the_rectified_declaration_must_be_a_bool(self) -> None:
        bad = self._bind(
            self._arm_rgb_only(),
            SensorDeployBinding(
                backend=SensorReaderBackend.ROS2_IMAGE,
                backend_params={"topic": "/cam/image_rect", "rectified": "yes"},
            ),
        )
        (problem,) = resolve_estimated_depth_camera(bad, "top", hal_mode="real").problems
        assert "rectified must be true or false" in problem

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
        calibrated = RobotDescription.from_yaml(robot)
        nominal = calibrated.model_copy(
            update={
                "sensors": [
                    s.model_copy(update={"static_transform_calibrated": False})
                    for s in calibrated.sensors
                ]
            }
        )
        problems = depth_extrinsic_problems(nominal, [], estimated_camera="top")
        assert [p.split(":")[0] for p in problems] == ["head_zed"]
        assert depth_extrinsic_problems(calibrated, [], estimated_camera="top") == []

    def test_a_scene_camera_sharing_a_robot_mount_is_checked_at_that_mount(self) -> None:
        robot = str(_REPO / "robots/openarm/robot.yaml")
        desc = RobotDescription.from_yaml(robot)
        top = next(s for s in desc.sensors if s.name == "top")
        bench = top.model_copy(
            update={
                "name": "bench_cam",
                "shares_mount_with": "head_zed",
                "parent_frame": None,
                "static_transform_xyz_rpy": None,
            }
        )
        assert depth_extrinsic_problems(desc, [], [bench], estimated_camera="bench_cam") == []


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
        canonical[row, col] = point[2] * _DA3_FOCAL / (0.5 * (fx + fy))
        depth = metres_from_output(canonical, _DA3, fx=fx, fy=fy)
        (got,) = points_from_depth(depth, fx=fx, fy=fy, cx=cx, cy=cy, max_range_m=3.0)
        # Within one raster pixel of the true point at that range.
        assert np.allclose(got, point, atol=point[2] / fx)

    def test_depth_pro2_metres_pass_through_and_sky_is_dropped(self) -> None:
        """A ``metric`` skill (Depth Pro 2) is already metres; infinite sky is no return."""
        pro = RSkillManifest.from_yaml(
            str(_REPO / "rskills/rskill-depth_pro2_s-any-metric_depth-bf16/rskill.yaml")
        ).depth_estimator
        assert pro is not None
        raw = np.array([[1.25, np.inf, -0.1]], dtype=np.float32)
        assert metres_from_output(raw, pro, fx=900.0, fy=901.0).tolist() == [[1.25, 0.0, 0.0]]

    def test_a_relative_skill_has_no_metric_reading(self) -> None:
        """DA3-SMALL (``relative``) never becomes metres for the collision map."""
        small = RSkillManifest.from_yaml(
            str(_REPO / "rskills/rskill-da3_small-any-depth-bf16/rskill.yaml")
        ).depth_estimator
        assert small is not None
        with pytest.raises(ROSConfigError, match="no metric reading"):
            metres_from_output(np.ones((2, 2)), small, fx=1.0, fy=1.0)

    @pytest.mark.parametrize(
        ("skill", "metric", "refusal"),
        [
            ("rskill-da3metric_large-any-metric_depth-bf16", True, None),
            ("rskill-depth_pro2_s-any-metric_depth-bf16", True, None),
            ("rskill-da3_small-any-depth-bf16", False, None),
            ("rskill-da3_small-any-depth-bf16", True, "relative depth"),
            ("rskill-depth_pro2_s-any-metric_depth-bf16", False, "predicts no camera"),
            ("rskill-sam2_1-any-grasped_object_mask-bf16", True, "not a depth_estimator"),
        ],
    )
    def test_the_provider_accepts_only_a_skill_that_fits_its_mode(
        self, skill: str, metric: bool, refusal: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Metric mode needs metres; the SLAM mode needs the model's own camera."""
        from openral_rskill.depth_estimator import depth_skill

        monkeypatch.setenv("OPENRAL_ALLOW_NONCOMMERCIAL", "1")  # Depth Pro 2 is apple-amlr
        manifest = str(_REPO / "rskills" / skill / "rskill.yaml")
        if refusal is None:
            _, repo = depth_skill(manifest, metric=metric)
            assert "/" in repo and "@" not in repo
        else:
            with pytest.raises(ROSConfigError, match=refusal):
                depth_skill(manifest, metric=metric)

    def test_research_only_depth_pro2_is_refused_without_the_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The provider loads through the license gate: no world map from apple-amlr weights."""
        from openral_rskill.depth_estimator import depth_skill

        monkeypatch.delenv("OPENRAL_ALLOW_NONCOMMERCIAL", raising=False)
        manifest = _REPO / "rskills/rskill-depth_pro2_s-any-metric_depth-bf16/rskill.yaml"
        with pytest.raises(ROSConfigError, match="OPENRAL_ALLOW_NONCOMMERCIAL"):
            depth_skill(manifest, metric=True)

    def test_far_and_invalid_returns_are_dropped(self) -> None:
        depth = metres_from_output(np.array([[300.0, np.nan, 3000.0]]), _DA3, fx=300.0, fy=300.0)
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
            ((500.0, 500.0, float("nan"), 239.5), (640, 480), (640, 480), "principal point"),
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


def _sensors(robot: str) -> list[SensorSpec]:
    return RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml")).sensors


class TestAutoPick:
    """``enable_octomap: true`` with no cloud picks the one RGB camera that qualifies."""

    @pytest.mark.parametrize("robot", ["widowx", "makermods_metal"])
    def test_the_one_mounted_rgb_camera_is_picked(self, robot: str) -> None:
        # makermods_metal's wrist has no mount, so only top qualifies.
        assert auto_estimated_depth_camera(_sensors(robot), hal_mode="sim") == "top"

    def test_a_measured_sensor_refuses_every_estimate(self) -> None:
        # Never reached by a deploy in sim (front_depth's cloud is mapped instead, see
        # TestDeployAutoPick); on real with nothing pinned, the refusal names the sensor.
        with pytest.raises(ROSConfigError, match=r"no RGB camera qualifies(.|\n)*'front_depth'"):
            auto_estimated_depth_camera(_sensors("panda_mobile"), hal_mode="real")

    def test_unmounted_cameras_are_refused_naming_each_problem(self) -> None:
        with pytest.raises(ROSConfigError) as exc:
            auto_estimated_depth_camera(_sensors("so101_follower"), hal_mode="sim")
        assert "top: no mount ties it" in str(exc.value)
        assert "wrist: no mount ties it" in str(exc.value)

    def test_several_qualifying_cameras_are_named_never_first_wins(self) -> None:
        rgb_only = [
            s for s in RobotDescription.from_yaml(str(_VSLAM)).sensors if s.modality == "rgb"
        ]
        with pytest.raises(ROSConfigError, match="shoulder_left, shoulder_right could each"):
            auto_estimated_depth_camera(rgb_only, hal_mode="sim")


class TestDeployAutoPick:
    """``openral deploy sim`` on the shipped WidowX twin, octomap toggled in the scene."""

    def _resolve(self, tmp_path: Path, runtime: str) -> object:
        from openral_cli.deploy_sim import resolve_launch_invocation

        text = (_REPO / "scenes" / "deploy" / "widowx_twin.yaml").read_text()
        assert "\n  enable_octomap: false\n" in text
        scene = tmp_path / "widowx.yaml"
        scene.write_text(text.replace("\n  enable_octomap: false\n", runtime))
        return resolve_launch_invocation(
            config=scene,
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_param_overrides=None,
        )

    def test_explicit_octomap_without_a_cloud_estimates_from_top(self, tmp_path: Path) -> None:
        inv = self._resolve(tmp_path, "\n  enable_octomap: true\n")
        assert inv.estimated_depth_camera == "top"  # type: ignore[attr-defined]
        argv = inv.argv_template  # type: ignore[attr-defined]
        assert "estimated_depth_camera:=top" in argv and "enable_octomap:=true" in argv
        # The default skill, checked before launch and forwarded as an absolute manifest.
        da3m = _REPO / "rskills/rskill-da3metric_large-any-metric_depth-bf16/rskill.yaml"
        assert f"estimated_depth_rskill:={da3m}" in argv

    def test_the_scenes_depth_rskill_comes_with_the_pick(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRAL_ALLOW_NONCOMMERCIAL", "1")  # Depth Pro 2 is research-only
        dp2 = "rskills/rskill-depth_pro2_s-any-metric_depth-bf16/rskill.yaml"  # repo-relative
        inv = self._resolve(
            tmp_path, f"\n  enable_octomap: true\n  estimated_depth_rskill: {dp2}\n"
        )
        assert f"estimated_depth_rskill:={_REPO / dp2}" in inv.argv_template  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        ("skill", "refusal"),
        [
            ("rskill-da3_small-any-depth-bf16", "predicts relative depth"),
            ("rskill-sam2_1-any-grasped_object_mask-bf16", "not a depth_estimator"),
            ("rskill-depth_pro2_s-any-metric_depth-bf16", "OPENRAL_ALLOW_NONCOMMERCIAL"),
            ("rskill-no-such-depth", "no manifest at"),
        ],
    )
    def test_an_unusable_depth_rskill_is_refused_before_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skill: str, refusal: str
    ) -> None:
        """Not inside the running graph, where the refusal would leave the robot halted."""
        monkeypatch.delenv("OPENRAL_ALLOW_NONCOMMERCIAL", raising=False)
        runtime = f"\n  enable_octomap: true\n  estimated_depth_rskill: {skill}\n"
        with pytest.raises(ROSConfigError, match=refusal):
            self._resolve(tmp_path, runtime)

    def test_an_explicit_camera_is_kept(self, tmp_path: Path) -> None:
        inv = self._resolve(tmp_path, "\n  enable_octomap: true\n  estimated_depth_camera: top\n")
        assert inv.estimated_depth_camera == "top"  # type: ignore[attr-defined]

    def test_unset_octomap_stays_off_an_estimate_never_enables_it(self, tmp_path: Path) -> None:
        inv = self._resolve(tmp_path, "\n")
        assert inv.estimated_depth_camera is None  # type: ignore[attr-defined]
        argv = inv.argv_template  # type: ignore[attr-defined]
        assert "enable_octomap:=false" in argv
        assert not [a for a in argv if a.startswith("estimated_depth_camera:=")]

    def test_a_measured_depth_sensor_is_mapped_not_estimated(self, tmp_path: Path) -> None:
        text = (_REPO / "scenes" / "deploy" / "robocasa_vslam_mono.yaml").read_text()
        scene = tmp_path / "mono.yaml"
        scene.write_text(text.replace("\n  enable_octomap: false\n", "\n  enable_octomap: true\n"))
        from openral_cli.deploy_sim import resolve_launch_invocation

        inv = resolve_launch_invocation(
            config=scene,
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_param_overrides=None,
        )
        assert inv.estimated_depth_camera is None
        assert inv.octomap_cloud_topic is None
        assert not [a for a in inv.argv_template if a.startswith("estimated_depth_camera:=")]


def test_the_depth_pro2_launcher_refuses_research_weights_without_the_opt_in() -> None:
    """An operator starting ``tools/depth_pro2_sidecar.py`` directly hits the license gate."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "OPENRAL_ALLOW_NONCOMMERCIAL"}
    tool = _REPO / "tools/depth_pro2_sidecar.py"
    proc = subprocess.run(
        [sys.executable, str(tool), "--model", "apple/DepthPro2-S", "--port", "5799"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode != 0
    assert "OPENRAL_ALLOW_NONCOMMERCIAL=1" in proc.stderr
