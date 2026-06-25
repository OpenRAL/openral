"""Tests for :func:`openral_detect.scaffold_robot_environment`.

Exercises the detect → deploy-config scaffold against the **real** canonical
``robots/so101_follower/robot.yaml`` and real schemas — no mocks (CLAUDE.md
§1.11). The scaffold must pre-fill everything detection knows (robot_id, serial
port, sensors) and leave only ``task`` as a ``TODO`` placeholder.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from openral_core import RobotEnvironment
from openral_core.schemas import (
    RobotDescription,
    SensorModality,
    SensorSpec,
)
from openral_detect import ScaffoldOverrides, scaffold_robot_environment
from openral_detect.report import (
    DetectionReport,
    UsbDeviceRecord,
    UsbMatchRecord,
    UsbProbeResult,
)
from openral_detect.scaffold import TODO_TASK_ID

REPO_ROOT = Path(__file__).resolve().parents[2]
SO101_YAML = REPO_ROOT / "robots" / "so101_follower" / "robot.yaml"


def _so101_description() -> RobotDescription:
    return RobotDescription.from_yaml(str(SO101_YAML))


def _usb_report(port: str) -> DetectionReport:
    dev = UsbDeviceRecord(port=port, vid=0x1A86, pid=0x7523, description="CH340")
    return DetectionReport(
        detected_at="2026-06-25T00:00:00Z",
        host_os="Linux",
        python_version="3.12",
        usb=UsbProbeResult(
            devices=[dev],
            matches=[
                UsbMatchRecord(
                    device=dev,
                    chip="CH340",
                    driver_hint="Feetech serial bus",
                    embodiment_tag="so100_follower",
                    bh_robot_type="so100",
                )
            ],
        ),
    )


class TestScaffoldFromManifest:
    def test_robot_id_matches_description(self) -> None:
        env = scaffold_robot_environment(_so101_description())
        assert env.robot_id == "so101_follower"
        assert env.hal.adapter == "so101_follower"

    def test_port_falls_back_to_manifest_default_without_detection(self) -> None:
        # The SO-101 manifest declares hal.parameters.defaults.port = /dev/ttyUSB0.
        env = scaffold_robot_environment(_so101_description())
        assert env.hal.transport["port"] == "/dev/ttyUSB0"

    def test_manifest_params_pass_through_except_port(self) -> None:
        env = scaffold_robot_environment(_so101_description())
        # calibrate_on_connect is a manifest default; it must survive, but port
        # is promoted into transport and removed from params.
        assert "port" not in env.hal.params
        assert env.hal.params.get("calibrate_on_connect") is False

    def test_safety_is_none_so_robot_limits_apply(self) -> None:
        # Robot limits live in RobotDescription.safety; deploy run uses them
        # when env.safety is None.
        env = scaffold_robot_environment(_so101_description())
        assert env.safety is None

    def test_task_is_todo_placeholder(self) -> None:
        env = scaffold_robot_environment(_so101_description())
        assert env.task.id == TODO_TASK_ID
        assert env.metadata["edit_before_deploy"] == ["task"]

    def test_one_sensor_reader_per_manifest_sensor(self) -> None:
        desc = _so101_description()
        env = scaffold_robot_environment(desc)
        manifest_sensor_names = {s.name for s in desc.sensors}
        scaffolded_ids = {s.sensor_id for s in env.sensors}
        # Every manifest sensor is represented; ids match SensorSpec.name so the
        # runner can bind them.
        assert manifest_sensor_names <= scaffolded_ids


class TestScaffoldUsesDetection:
    def test_detected_port_wins_over_manifest_default(self) -> None:
        env = scaffold_robot_environment(_so101_description(), _usb_report("/dev/ttyACM3"))
        assert env.hal.transport["port"] == "/dev/ttyACM3"

    def test_camera_device_path_populates_backend_params(self) -> None:
        desc = _so101_description()
        # Simulate a detected V4L2 camera (openral detect stores device_path in
        # SensorSpec.metadata).
        cam = SensorSpec(
            name="camera_0",
            modality=SensorModality.RGB,
            frame_id="camera_0_optical_frame",
            rate_hz=30.0,
            encoding="rgb8",
            metadata={"detected_by": "openral detect", "device_path": "/dev/video2"},
        )
        desc = desc.model_copy(update={"sensors": [*desc.sensors, cam]}, deep=True)
        env = scaffold_robot_environment(desc)
        cfg = next(s for s in env.sensors if s.sensor_id == "camera_0")
        assert cfg.backend_params == {"device": "/dev/video2", "fps": 30}


class TestScaffoldRoundTrips:
    def test_yaml_dump_loads_back_as_robot_environment(self, tmp_path: Path) -> None:
        env = scaffold_robot_environment(_so101_description(), _usb_report("/dev/ttyACM0"))
        path = tmp_path / "deploy.yaml"
        path.write_text(
            yaml.safe_dump(env.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        reloaded = RobotEnvironment.from_yaml(str(path))
        assert reloaded.robot_id == "so101_follower"
        assert reloaded.hal.transport["port"] == "/dev/ttyACM0"
        # The task placeholder survives a round-trip and remains schema-valid.
        assert reloaded.task.id == TODO_TASK_ID


class TestScaffoldOverrides:
    def test_overrides_fill_task_safety_and_label(self) -> None:
        ov = ScaffoldOverrides(
            label="bench-arm",
            task_id="pick/desk",
            task_instruction="pick the red block",
            workspace_box_min_xyz=(-0.3, -0.3, 0.0),
            workspace_box_max_xyz=(0.3, 0.3, 0.5),
        )
        env = scaffold_robot_environment(_so101_description(), None, overrides=ov)
        assert env.task.id == "pick/desk"
        assert env.task.instruction == "pick the red block"
        assert env.metadata["label"] == "bench-arm"
        assert env.safety is not None
        assert env.safety.workspace_box_min_xyz == (-0.3, -0.3, 0.0)
        assert env.safety.workspace_box_max_xyz == (0.3, 0.3, 0.5)
        # Nothing left to edit, so the deploy guard is empty.
        assert env.metadata["edit_before_deploy"] == []

    def test_partial_overrides_keep_task_todo(self) -> None:
        # Only the task instruction is supplied; task.id stays TODO, so the
        # guard still flags task (id is the blocking sentinel).
        ov = ScaffoldOverrides(task_instruction="pick the red block")
        env = scaffold_robot_environment(_so101_description(), None, overrides=ov)
        assert env.task.instruction == "pick the red block"
        assert env.task.id == TODO_TASK_ID
        assert env.metadata["edit_before_deploy"] == ["task"]

    def test_lone_workspace_corner_is_ignored(self) -> None:
        # A box needs both corners to be a real constraint; one corner → no safety.
        ov = ScaffoldOverrides(workspace_box_min_xyz=(-0.3, -0.3, 0.0))
        env = scaffold_robot_environment(_so101_description(), None, overrides=ov)
        assert env.safety is None
