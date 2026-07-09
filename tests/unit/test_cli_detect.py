"""Unit tests for the ``openral detect`` and ``ral skill check`` CLI commands.

Hermetic — every probe is exercised against a clean container, no
hardware required.  Larger end-to-end coverage lives in
``test_detect_probes_no_hardware.py`` / ``test_detect_assemble.py`` /
``test_detect_compatibility.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml
from openral_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


class TestBhDetect:
    def test_detect_no_write_prints_summary(self) -> None:
        result = runner.invoke(
            app,
            ["detect", "--no-write", "--include", "network", "--dds-timeout", "0"],
        )
        assert result.exit_code == 0, result.output
        assert "openral detect" in result.output
        # --no-write prints the assembled yaml to stdout.
        assert "name:" in result.output

    def test_no_write_with_deployment_warns(self, tmp_path: Path) -> None:
        deploy = tmp_path / "workcell.yaml"
        result = runner.invoke(
            app,
            [
                "detect",
                "--no-write",
                "--deployment",
                str(deploy),
                "--robot",
                "so101",
                "--include",
                "network",
                "--dds-timeout",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "--deployment ignored under --no-write" in result.output
        assert not deploy.exists()

    def test_detect_writes_full_robot_yaml(self, tmp_path: Path) -> None:
        out = tmp_path / "robot.yaml"
        result = runner.invoke(
            app,
            [
                "detect",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
            input="\nn\n",
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        data = yaml.safe_load(out.read_text())
        # Must be a complete RobotDescription, not the legacy stub.
        assert "name" in data
        assert "capabilities" in data
        assert "embodiment_kind" in data
        assert "safety" in data

    def test_detect_robot_override_forces_so101(self, tmp_path: Path) -> None:
        # No SO-101 hardware attached; the --robot override pins the manifest
        # regardless of what USB/network probing finds.
        out = tmp_path / "robot.yaml"
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "so101",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
            input="\nn\n",
        )
        assert result.exit_code == 0, result.output
        data = yaml.safe_load(out.read_text())
        assert data["name"] == "so101_follower"

    def test_detect_bad_robot_override_exits_1(self) -> None:
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "not_a_robot",
                "--no-write",
                "--include",
                "network",
                "--dds-timeout",
                "0",
            ],
        )
        assert result.exit_code == 1
        assert "no committed" in result.output

    def test_detect_deployment_scaffolds_deploy_scene(self, tmp_path: Path) -> None:
        out = tmp_path / "robot.yaml"
        deploy = tmp_path / "workcell.yaml"
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "so101",
                "--output",
                str(out),
                "--deployment",
                str(deploy),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
            input="\nn\n",
        )
        assert result.exit_code == 0, result.output
        assert deploy.exists()
        # The scaffold loads back as a valid DeployScene.
        from openral_core import DeployScene

        scene = DeployScene.from_yaml(str(deploy))
        assert scene.robot_id == "so101_follower"
        assert scene.scene.id == "so101_follower_workcell"
        # safety unset → the robot manifest's envelope applies as-is.
        assert scene.safety is None
        assert scene.sensors == []
        banner = deploy.read_text()
        assert "review before" in banner
        assert "reasoner selects it at runtime" in banner

    def test_scaffold_notifies_calibration_needed(self, tmp_path: Path) -> None:
        """A serial (lerobot Feetech) HAL with no calibration on disk prompts
        the operator to link/generate one, pointing at the lerobot docs."""
        deploy = tmp_path / "workcell.yaml"
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "so101",
                "--output",
                str(tmp_path / "robot.yaml"),
                "--deployment",
                str(deploy),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
            input="\nn\n",
        )
        assert result.exit_code == 0, result.output
        assert "Calibration required" in result.output
        assert "so101#calibrate" in result.output

    def test_scaffold_calibration_notice_suppressed_when_present(self, tmp_path: Path) -> None:
        """When a calibration file already sits in the scene's calibration dir,
        the notice is not shown."""
        deploy = tmp_path / "workcell.yaml"
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        (cal_dir / "so101_follower.json").write_text("{}", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "so101",
                "--output",
                str(tmp_path / "robot.yaml"),
                "--deployment",
                str(deploy),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
            input="\nn\n",
        )
        assert result.exit_code == 0, result.output
        assert "Calibration required" not in result.output

    def test_detect_scene_uses_detected_serial_port(self, tmp_path: Path) -> None:
        """The scaffolded scene's HAL port comes from the USB probe, not the
        canonical manifest's stale default (so101 ships /dev/ttyUSB0)."""
        from openral_detect.report import UsbDeviceRecord, UsbMatchRecord, UsbProbeResult

        out = tmp_path / "robot.yaml"
        deploy = tmp_path / "workcell.yaml"
        dev = UsbDeviceRecord(
            port="/dev/ttyACM0", vid=6790, pid=21971, description="USB_Single_Serial"
        )
        match = UsbMatchRecord(
            device=dev,
            chip="CH343",
            driver_hint="Feetech serial bus",
            embodiment_tag="so101_follower",
            bh_robot_type="so101",
        )
        usb_result = UsbProbeResult(devices=[dev], matches=[match])
        with patch("openral_detect.detect.probe_usb", return_value=usb_result):
            result = runner.invoke(
                app,
                [
                    "detect",
                    "--robot",
                    "so101",
                    "--output",
                    str(out),
                    "--deployment",
                    str(deploy),
                    "--include",
                    "network,usb",
                    "--dds-timeout",
                    "0",
                    "--yes",
                ],
                input="rig\nn\n",
            )
        assert result.exit_code == 0, result.output
        from openral_core import DeployScene

        scene = DeployScene.from_yaml(str(deploy))
        assert scene.hal is not None
        assert scene.hal.defaults["port"] == "/dev/ttyACM0"

    def test_detect_builds_custom_robot_yaml_split(self, tmp_path: Path) -> None:
        """Robot cam → custom robot.yaml sensor (canonical spec reused);
        workspace cam → DeployScene only; unbound canonical sensors dropped."""
        from openral_detect.report import V4l2CameraInfo

        out = tmp_path / "robot.yaml"
        deploy = tmp_path / "workcell.yaml"
        cams = [
            V4l2CameraInfo(device_path="/dev/video7", name="fake wrist cam"),
            V4l2CameraInfo(device_path="/dev/video8", name="fake overhead cam"),
        ]
        # input order: robot-name prompt, then per-camera answers, then
        # joint/safety gate (Task 4 adds it; here answer "n").
        with patch("openral_detect.detect.probe_v4l2_cameras", return_value=cams):
            result = runner.invoke(
                app,
                [
                    "detect",
                    "--robot",
                    "so101",
                    "--output",
                    str(out),
                    "--deployment",
                    str(deploy),
                    "--include",
                    "network,cameras_v4l2",
                    "--dds-timeout",
                    "0",
                    "--yes",
                ],
                input="my_so101_bench\nwrist\nw:overhead\nn\n",
            )
        assert result.exit_code == 0, result.output

        from openral_core import DeployScene, RobotDescription

        desc = RobotDescription.from_yaml(str(out))
        assert desc.name == "my_so101_bench"
        # Custom robot.yaml has ONLY the bound robot cam, reusing the canonical spec.
        assert [s.name for s in desc.sensors] == ["wrist"]
        wrist = desc.sensors[0]
        assert wrist.frame_id == "wrist_camera"  # canonical spec reused verbatim
        assert wrist.vla_feature_key == "observation.images.camera2"
        # 'top' was never bound → dropped from the custom manifest.
        assert "top" not in [s.name for s in desc.sensors]

        scene = DeployScene.from_yaml(str(deploy))
        by_name = {s.name: s for s in scene.sensors}
        assert set(by_name) == {"overhead"}  # workspace cam ONLY in the scene
        assert by_name["overhead"].deploy_binding.backend_params["device"] == "/dev/video8"
        assert scene.robot_id == "my_so101_bench"

    def test_wizard_rejects_duplicate_sensor_name(self, tmp_path: Path) -> None:
        """Answering the same target sensor name for two cameras re-prompts the
        second camera instead of writing a duplicate ``SensorSpec.name``."""
        from openral_detect.report import V4l2CameraInfo

        out = tmp_path / "robot.yaml"
        cams = [
            V4l2CameraInfo(device_path="/dev/video7", name="cam a"),
            V4l2CameraInfo(device_path="/dev/video8", name="cam b"),
        ]
        with patch("openral_detect.detect.probe_v4l2_cameras", return_value=cams):
            result = runner.invoke(
                app,
                [
                    "detect",
                    "--robot",
                    "so101",
                    "--output",
                    str(out),
                    "--include",
                    "network,cameras_v4l2",
                    "--dds-timeout",
                    "0",
                    "--yes",
                ],
                # name; cam a -> wrist; cam b -> wrist (rejected) -> top; gate n
                input="rig\nwrist\nwrist\ntop\nn\n",
            )
        assert result.exit_code == 0, result.output
        assert "already bound" in result.output
        from openral_core import RobotDescription

        desc = RobotDescription.from_yaml(str(out))
        names = [s.name for s in desc.sensors]
        assert sorted(names) == ["top", "wrist"]  # no duplicate wrist

    def test_detect_wizard_new_robot_sensor(self, tmp_path: Path) -> None:
        """A non-canonical, non-w: answer creates a new robot sensor in robot.yaml
        with the entered parent_frame."""
        from openral_detect.report import V4l2CameraInfo

        out = tmp_path / "robot.yaml"
        cams = [V4l2CameraInfo(device_path="/dev/video7", name="fake chin cam")]
        with patch("openral_detect.detect.probe_v4l2_cameras", return_value=cams):
            result = runner.invoke(
                app,
                [
                    "detect",
                    "--robot",
                    "so101",
                    "--output",
                    str(out),
                    "--include",
                    "network,cameras_v4l2",
                    "--dds-timeout",
                    "0",
                    "--yes",
                ],
                # robot name, then bind the camera to a new sensor "chin", parent_frame
                # "base_link", then decline the joint/safety customization gate.
                input="rig\nchin\nbase_link\nn\n",
            )
        assert result.exit_code == 0, result.output
        from openral_core import RobotDescription

        desc = RobotDescription.from_yaml(str(out))
        chin = next(s for s in desc.sensors if s.name == "chin")
        assert chin.modality == "rgb"
        assert chin.frame_id == "chin_optical_frame"
        assert chin.parent_frame == "base_link"
        assert chin.deploy_binding.backend_params["device"] == "/dev/video7"

    def test_thumbnail_hint_when_device_unreadable(self, tmp_path: Path) -> None:
        """A camera the wizard can't grab a frame from prints '(no thumbnail)'
        instead of silently omitting it."""
        from openral_detect.report import V4l2CameraInfo

        out = tmp_path / "robot.yaml"
        cams = [V4l2CameraInfo(device_path="/dev/video99", name="absent cam")]
        with patch("openral_detect.detect.probe_v4l2_cameras", return_value=cams):
            result = runner.invoke(
                app,
                [
                    "detect",
                    "--robot",
                    "so101",
                    "--output",
                    str(out),
                    "--include",
                    "network,cameras_v4l2",
                    "--dds-timeout",
                    "0",
                    "--yes",
                ],
                # robot name, then skip the camera binding (blank), then decline
                # the joint/safety customization gate.
                input="rig\n\nn\n",
            )
        assert result.exit_code == 0, result.output
        assert "no thumbnail" in result.output.lower()

    def test_detect_wizard_offers_realsense(self, tmp_path: Path) -> None:
        """RealSense devices (serial-keyed, no /dev/video path) are offered
        alongside V4L2 cameras; the binding records the serial, not a device."""
        from openral_detect.report import RealsenseDeviceInfo

        out = tmp_path / "robot.yaml"
        deploy = tmp_path / "workcell.yaml"
        rs = [
            RealsenseDeviceInfo(serial="828612070123", name="Intel RealSense D435", model_id="D435")
        ]
        with patch("openral_detect.detect.probe_realsense_devices", return_value=rs):
            result = runner.invoke(
                app,
                [
                    "detect",
                    "--robot",
                    "so101",
                    "--output",
                    str(out),
                    "--deployment",
                    str(deploy),
                    "--include",
                    "network,cameras_realsense",
                    "--dds-timeout",
                    "0",
                    "--yes",
                ],
                input="rig\nw:front_depth\nn\n",
            )
        assert result.exit_code == 0, result.output
        from openral_core import DeployScene

        scene = DeployScene.from_yaml(str(deploy))
        by_name = {s.name: s for s in scene.sensors}
        assert "front_depth" in by_name
        assert by_name["front_depth"].deploy_binding.backend_params.get("serial") == "828612070123"
        assert "device" not in by_name["front_depth"].deploy_binding.backend_params

    def test_detect_custom_dir_rewrites_file_asset_refs(self, tmp_path: Path) -> None:
        """Writing the custom manifest to a new dir rewrites file: URDF refs so they
        still resolve (via the repo-root fallback)."""
        out = tmp_path / "robots" / "my_bench" / "robot.yaml"
        out.parent.mkdir(parents=True)
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "so101",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
            input="my_bench\nn\n",
        )
        assert result.exit_code == 0, result.output
        from openral_core import RobotDescription
        from openral_core.assets import resolve_asset

        desc = RobotDescription.from_yaml(str(out))
        # The rewritten ref must resolve to the real canonical URDF file.
        resolved = resolve_asset(desc.assets.urdf.ref, "urdf")
        assert resolved is not None and resolved.is_file()
        assert desc.assets.urdf.ref.startswith("file:robots/so101_follower/")

    def test_detect_customizes_first_joint_position_min(self, tmp_path: Path) -> None:
        out = tmp_path / "robot.yaml"
        # name, gate=y, first joint position-min = -1.5, everything else blank.
        inp = "rig\ny\n-1.5\n" + "\n" * 60
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "so101",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
            input=inp,
        )
        assert result.exit_code == 0, result.output
        from openral_core import RobotDescription

        desc = RobotDescription.from_yaml(str(out))
        first = next(j for j in desc.joints if j.position_limits is not None)
        assert first.position_limits[0] == -1.5

    def test_detect_gate_declined_keeps_canonical(self, tmp_path: Path) -> None:
        out = tmp_path / "robot.yaml"
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "so101",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
            input="rig\nn\n",
        )
        assert result.exit_code == 0, result.output
        from openral_core import RobotDescription
        from openral_detect.registry import canonical_robot_path

        desc = RobotDescription.from_yaml(str(out))
        canon = RobotDescription.from_yaml(str(canonical_robot_path("so101")))
        assert [j.velocity_limit for j in desc.joints] == [j.velocity_limit for j in canon.joints]
        assert desc.safety.max_force_n == canon.safety.max_force_n

    def test_robot_name_rejects_invalid_then_accepts(self, tmp_path: Path) -> None:
        out = tmp_path / "robot.yaml"
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "so101",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
            input="bad name/../x\nmy_bench\nn\n",  # invalid, then valid, then gate
        )
        assert result.exit_code == 0, result.output
        assert "invalid" in result.output.lower()
        from openral_core import RobotDescription

        assert RobotDescription.from_yaml(str(out)).name == "my_bench"

    def test_detect_with_report_dump(self, tmp_path: Path) -> None:
        out = tmp_path / "robot.yaml"
        report = tmp_path / "detection.json"
        result = runner.invoke(
            app,
            [
                "detect",
                "--output",
                str(out),
                "--report",
                str(report),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
            input="\nn\n",
        )
        assert result.exit_code == 0, result.output
        assert report.exists()
        # Raw report is JSON.
        import json

        payload = json.loads(report.read_text())
        assert payload["schema_version"] == "0.1"


class TestBhSkillCheck:
    def test_skill_check_missing_robot_yaml_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["rskill", "check", "--robot", str(tmp_path / "missing.yaml")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_skill_check_against_assembled_yaml(self, tmp_path: Path) -> None:
        # Step 1: produce a robot.yaml via `openral detect`.
        out = tmp_path / "robot.yaml"
        runner.invoke(
            app,
            [
                "detect",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
            input="\nn\n",
        )
        # Step 2: run `ral skill check` against an empty registry.
        empty_registry = tmp_path / "empty-registry.json"
        # Point --rskills-dir at a non-existent path so the default ("rskills/")
        # doesn't walk the in-tree rskills/ from the repo cwd.
        missing_rskills = tmp_path / "no-such-rskills"
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", empty_registry):
            result = runner.invoke(
                app,
                [
                    "rskill",
                    "check",
                    "--robot",
                    str(out),
                    "--rskills-dir",
                    str(missing_rskills),
                    "--json",
                ],
            )
        # Empty registry → exit 0 (no incompat rows).
        assert result.exit_code == 0, result.output
        # JSON output parseable.
        import json

        payload = json.loads(result.output)
        assert payload["schema_version"] == "0.1"
        assert "rows" in payload
