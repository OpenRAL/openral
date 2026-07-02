"""Unit tests for openral_sensors — RealSense bundle factories, launch
generator, node params mapping, and calibrate_camera_cmd helper.

All tests run without a live ROS 2 installation or physical camera.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
from openral_cli.main import app
from openral_core.schemas import SensorBundle, SensorModality, SensorSpec
from openral_sensors.realsense import (
    bundle_to_node_params,
    calibrate_camera_cmd,
    generate_launch_py,
    realsense_d435_bundle,
)
from typer.testing import CliRunner

runner = CliRunner()


# ── D435 bundle factory ───────────────────────────────────────────────────────


class TestD435Bundle:
    def test_returns_sensor_bundle(self) -> None:
        bundle = realsense_d435_bundle()
        assert isinstance(bundle, SensorBundle)

    def test_has_three_sensors(self) -> None:
        bundle = realsense_d435_bundle()
        assert len(bundle.sensors) == 3

    def test_sensor_modalities(self) -> None:
        bundle = realsense_d435_bundle()
        modalities = [s.modality for s in bundle.sensors]
        assert SensorModality.RGB in modalities
        assert SensorModality.DEPTH in modalities
        assert SensorModality.IMU in modalities

    def test_bundle_name_prefix(self) -> None:
        bundle = realsense_d435_bundle(name="wrist")
        assert bundle.bundle_name == "wrist"
        assert all(s.name.startswith("wrist") for s in bundle.sensors)

    def test_topic_prefix(self) -> None:
        bundle = realsense_d435_bundle(name="head")
        rgb = next(s for s in bundle.sensors if s.modality == SensorModality.RGB)
        assert rgb.ros2_topic.startswith("/head/")

    def test_frame_ids_use_name(self) -> None:
        bundle = realsense_d435_bundle(name="cam")
        for sensor in bundle.sensors:
            assert "cam" in sensor.frame_id

    def test_rgb_encoding(self) -> None:
        bundle = realsense_d435_bundle()
        rgb = next(s for s in bundle.sensors if s.modality == SensorModality.RGB)
        assert rgb.encoding == "rgb8"

    def test_depth_encoding(self) -> None:
        bundle = realsense_d435_bundle()
        depth = next(s for s in bundle.sensors if s.modality == SensorModality.DEPTH)
        assert depth.encoding == "16UC1"

    def test_depth_range(self) -> None:
        bundle = realsense_d435_bundle()
        depth = next(s for s in bundle.sensors if s.modality == SensorModality.DEPTH)
        assert depth.range_min_m == pytest.approx(0.1)
        assert depth.range_max_m == pytest.approx(10.0)

    def test_imu_rate(self) -> None:
        bundle = realsense_d435_bundle(imu_rate_hz=200.0)
        imu = next(s for s in bundle.sensors if s.modality == SensorModality.IMU)
        assert imu.rate_hz == pytest.approx(200.0)

    def test_sync_is_hardware(self) -> None:
        assert realsense_d435_bundle().sync == "hardware"

    def test_serial_no_stored_in_metadata(self) -> None:
        bundle = realsense_d435_bundle(serial_no="123ABC")
        rgb = next(s for s in bundle.sensors if s.modality == SensorModality.RGB)
        assert rgb.metadata["serial_no"] == "123ABC"

    def test_empty_serial_no_omits_metadata(self) -> None:
        bundle = realsense_d435_bundle()
        rgb = next(s for s in bundle.sensors if s.modality == SensorModality.RGB)
        assert "serial_no" not in rgb.metadata

    def test_model_is_d435(self) -> None:
        bundle = realsense_d435_bundle()
        for sensor in bundle.sensors:
            assert sensor.model == "RealSense D435"

    def test_driver_pkg(self) -> None:
        bundle = realsense_d435_bundle()
        for sensor in bundle.sensors:
            assert sensor.driver_pkg == "realsense2_camera"

    def test_parent_frame_propagated(self) -> None:
        bundle = realsense_d435_bundle(parent_frame="ee_link")
        for sensor in bundle.sensors:
            assert sensor.parent_frame == "ee_link"

    def test_intrinsics_present_on_rgb(self) -> None:
        bundle = realsense_d435_bundle()
        rgb = next(s for s in bundle.sensors if s.modality == SensorModality.RGB)
        assert rgb.intrinsics is not None
        assert rgb.intrinsics.width == 640
        assert rgb.intrinsics.height == 480


# ── bundle_to_node_params ─────────────────────────────────────────────────────


class TestBundleToNodeParams:
    def test_returns_dict(self) -> None:
        params = bundle_to_node_params(realsense_d435_bundle())
        assert isinstance(params, dict)

    def test_camera_name_matches_bundle(self) -> None:
        params = bundle_to_node_params(realsense_d435_bundle(name="head"))
        assert params["camera_name"] == "head"

    def test_camera_namespace(self) -> None:
        params = bundle_to_node_params(realsense_d435_bundle(name="wrist"))
        assert params["camera_namespace"] == "/wrist"

    def test_serial_no_from_arg(self) -> None:
        params = bundle_to_node_params(realsense_d435_bundle(), serial_no="XYZ")
        assert params["serial_no"] == "XYZ"

    def test_serial_no_from_metadata(self) -> None:
        params = bundle_to_node_params(realsense_d435_bundle(serial_no="META"))
        assert params["serial_no"] == "META"

    def test_serial_no_arg_overrides_metadata(self) -> None:
        params = bundle_to_node_params(realsense_d435_bundle(serial_no="META"), serial_no="ARG")
        assert params["serial_no"] == "ARG"

    def test_rgb_profile_640x480x30(self) -> None:
        params = bundle_to_node_params(realsense_d435_bundle())
        assert params["rgb_camera.color_profile"] == "640x480x30"

    def test_depth_profile_640x480x30(self) -> None:
        params = bundle_to_node_params(realsense_d435_bundle())
        assert params["depth_module.depth_profile"] == "640x480x30"

    def test_enable_color_true(self) -> None:
        assert bundle_to_node_params(realsense_d435_bundle())["enable_color"] is True

    def test_enable_depth_true(self) -> None:
        assert bundle_to_node_params(realsense_d435_bundle())["enable_depth"] is True

    def test_enable_imu(self) -> None:
        params = bundle_to_node_params(realsense_d435_bundle())
        assert params["enable_gyro"] is True
        assert params["enable_accel"] is True

    def test_no_rgb_sensor_raises(self) -> None:
        depth_only = SensorBundle(
            bundle_name="depth_only",
            sensors=[
                SensorSpec(
                    name="d",
                    modality=SensorModality.DEPTH,
                    frame_id="d_frame",
                    rate_hz=30.0,
                    ros2_topic="/d/depth",
                    ros2_msg_type="sensor_msgs/Image",
                )
            ],
        )
        with pytest.raises(ValueError, match="no RGB sensor"):
            bundle_to_node_params(depth_only)


# ── generate_launch_py ────────────────────────────────────────────────────────


class TestGenerateLaunchPy:
    def test_returns_string(self) -> None:
        src = generate_launch_py(realsense_d435_bundle())
        assert isinstance(src, str)

    def test_starts_with_header_comment(self) -> None:
        src = generate_launch_py(realsense_d435_bundle())
        assert src.startswith("# Generated by OpenRAL")

    def test_contains_camera_name(self) -> None:
        src = generate_launch_py(realsense_d435_bundle(name="head"))
        assert "head" in src

    def test_imports_launch(self) -> None:
        src = generate_launch_py(realsense_d435_bundle())
        assert "from launch import LaunchDescription" in src

    def test_uses_realsense2_camera_package(self) -> None:
        src = generate_launch_py(realsense_d435_bundle())
        assert "realsense2_camera" in src

    def test_is_valid_python(self) -> None:
        """The generated file must parse as valid Python."""
        src = generate_launch_py(realsense_d435_bundle())
        ast.parse(src)  # raises SyntaxError if invalid

    def test_generate_launch_description_function_present(self) -> None:
        src = generate_launch_py(realsense_d435_bundle())
        assert "def generate_launch_description" in src

    def test_serial_no_in_output(self) -> None:
        src = generate_launch_py(realsense_d435_bundle(), serial_no="SN999")
        assert "SN999" in src


# ── calibrate_camera_cmd ──────────────────────────────────────────────────────


class TestCalibrateCameraCmd:
    def _rgb_spec(
        self, name: str = "head_color", topic: str = "/head/color/image_raw"
    ) -> SensorSpec:
        return SensorSpec(
            name=name,
            modality=SensorModality.RGB,
            frame_id=f"{name}_optical_frame",
            rate_hz=30.0,
            ros2_topic=topic,
            ros2_msg_type="sensor_msgs/Image",
        )

    def test_returns_list(self) -> None:
        cmd = calibrate_camera_cmd(self._rgb_spec())
        assert isinstance(cmd, list)

    def test_starts_with_ros2(self) -> None:
        cmd = calibrate_camera_cmd(self._rgb_spec())
        assert cmd[0] == "ros2"

    def test_contains_size_arg(self) -> None:
        cmd = calibrate_camera_cmd(self._rgb_spec(), chessboard_cols=8, chessboard_rows=6)
        idx = cmd.index("--size")
        assert cmd[idx + 1] == "8x6"

    def test_contains_square_arg(self) -> None:
        cmd = calibrate_camera_cmd(self._rgb_spec(), square_size_m=0.03)
        idx = cmd.index("--square")
        assert cmd[idx + 1] == "0.03"

    def test_image_remap(self) -> None:
        cmd = calibrate_camera_cmd(self._rgb_spec(topic="/cam/color/image_raw"))
        remaps = " ".join(cmd)
        assert "image:=/cam/color/image_raw" in remaps

    def test_camera_info_remap_derived(self) -> None:
        cmd = calibrate_camera_cmd(self._rgb_spec(topic="/cam/color/image_raw"))
        remaps = " ".join(cmd)
        assert "camera_info:=/cam/color/camera_info" in remaps

    def test_non_rgb_raises(self) -> None:
        depth_spec = SensorSpec(
            name="depth",
            modality=SensorModality.DEPTH,
            frame_id="d_frame",
            rate_hz=30.0,
            ros2_topic="/cam/depth/image_rect_raw",
            ros2_msg_type="sensor_msgs/Image",
        )
        with pytest.raises(ValueError, match="rgb"):
            calibrate_camera_cmd(depth_spec)


# ── openral calibrate camera CLI ───────────────────────────────────────────────────


class TestCalibrateCameraCLI:
    def test_dry_run_prints_command(self) -> None:
        result = runner.invoke(
            app,
            ["calibrate", "camera", "--sensor", "head_color", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "ros2 run camera_calibration cameracalibrator" in result.output
        # Topic derivation: image: /head_color/image_raw, info: /head_color/camera_info
        assert "/head_color/image_raw" in result.output
        assert "/head_color/camera_info" in result.output

    def test_dry_run_custom_chessboard(self) -> None:
        result = runner.invoke(
            app,
            [
                "calibrate",
                "camera",
                "--sensor",
                "head_color",
                "--chessboard-size",
                "9x7",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "9x7" in result.output

    def test_dry_run_custom_topic(self) -> None:
        result = runner.invoke(
            app,
            [
                "calibrate",
                "camera",
                "--sensor",
                "wrist_color",
                "--topic",
                "/wrist/color/image_raw",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "/wrist/color/image_raw" in result.output

    def test_invalid_chessboard_size_exits_1(self) -> None:
        result = runner.invoke(
            app,
            ["calibrate", "camera", "--sensor", "cam", "--chessboard-size", "bad", "--dry-run"],
        )
        assert result.exit_code == 1

    def test_no_ros2_bin_exits_1(self) -> None:
        with patch("openral_cli.main.shutil.which", return_value=None):
            result = runner.invoke(
                app,
                ["calibrate", "camera", "--sensor", "head_color"],
            )
        assert result.exit_code == 1
        assert "ros2 not found" in result.output

    def test_calibrator_failure_propagates_exit_code(self) -> None:
        with (
            patch("openral_cli.main.shutil.which", return_value="/usr/bin/ros2"),
            patch(
                "openral_cli.main.subprocess.run",
                return_value=type("R", (), {"returncode": 2})(),
            ) as run_patch,
        ):
            result = runner.invoke(
                app,
                ["calibrate", "camera", "--sensor", "head_color"],
            )
        assert result.exit_code == 2
        # Command is built deterministically; spot-check key fragments.
        cmd = run_patch.call_args.args[0]
        assert cmd[:4] == ["ros2", "run", "camera_calibration", "cameracalibrator"]
        assert "image:=/head_color/image_raw" in cmd

    def test_calibrator_success_exits_0(self) -> None:
        with (
            patch("openral_cli.main.shutil.which", return_value="/usr/bin/ros2"),
            patch(
                "openral_cli.main.subprocess.run",
                return_value=type("R", (), {"returncode": 0})(),
            ),
        ):
            result = runner.invoke(
                app,
                ["calibrate", "camera", "--sensor", "head_color"],
            )
        assert result.exit_code == 0

    def test_help_text_mentions_calibrate(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "calibrate" in result.output
