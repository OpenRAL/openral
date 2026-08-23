"""End-to-end assembly for a CAN-attached OpenArm cell.

One coherent scenario rather than isolated units: the detection report a
provisioned OpenArm cell actually produces — two PCAN-USB Pro FD channels
named ``openarm_left`` / ``openarm_right``, two Arducam B0495 global-shutter
cameras and a ZED Mini stereo head — assembled into a
:class:`~openral_core.RobotDescription`.

Everything here is real per CLAUDE.md §1.11: the committed
``robots/openarm/robot.yaml`` fixture and the real ``CATALOG`` populated by
every vendor module on import.  The device facts (VID/PIDs, product strings,
CAN bitrates) were read off the hardware, not invented.

The regressions this pins:

- A CAN-only robot used to be invisible.  USB-serial enumeration cannot see
  it (a CAN adapter registers a *network* device), and DDS inference cannot
  see it either (the bringup that would publish those topics is what detect
  runs *before*), so ``openral detect`` scaffolded ``unknown_<hostname>``
  while a fully-wired bimanual arm sat on the bench.
- A stereo head resolved through V4L2 used to be forced into a single
  ``SensorSpec``, dropping its depth and IMU streams.
"""

from __future__ import annotations

from pathlib import Path

from openral_core.schemas import RobotDescription, SensorBundle
from openral_detect import (
    CameraProbeResult,
    CanInterfaceInfo,
    CanMatchRecord,
    CanProbeResult,
    DetectionReport,
    V4l2CameraInfo,
    assemble_robot_description,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENARM_MANIFEST = REPO_ROOT / "robots" / "openarm" / "robot.yaml"


def _can_interface(name: str, state: str = "ERROR-ACTIVE") -> CanInterfaceInfo:
    """One PCAN-USB Pro FD channel as `ip -details -json link show` reports it."""
    return CanInterfaceInfo(
        name=name,
        is_up=True,
        fd_enabled=True,
        bitrate=1_000_000,
        data_bitrate=5_000_000,
        state=state,
        driver="pcan_usb_pro_fd",
        mtu=72,
        vid=0x0C72,
        pid=0x0011,
        adapter="PEAK PCAN-USB Pro FD",
    )


def _openarm_cell_report(*, cameras: bool = True) -> DetectionReport:
    interfaces = [_can_interface("openarm_left"), _can_interface("openarm_right")]
    camera_probe = CameraProbeResult()
    if cameras:
        camera_probe = CameraProbeResult(
            v4l2=[
                V4l2CameraInfo(
                    device_path="/dev/video0",
                    name="Arducam B0495 (USB3 2.3MP)",
                    bus_info="usb-2-3.1.3",
                    vid=0x04B4,
                    pid=0x4950,
                    serial="CELL1_CAM_LEFT",
                ),
                V4l2CameraInfo(
                    device_path="/dev/video2",
                    name="ZED-M: ZED-M",
                    bus_info="usb-2-1",
                    vid=0x2B03,
                    pid=0xF682,
                ),
                V4l2CameraInfo(
                    device_path="/dev/video4",
                    name="Arducam B0495 (USB3 2.3MP)",
                    bus_info="usb-2-3.2",
                    vid=0x04B4,
                    pid=0x4950,
                    serial="CELL1_CAM_RIGHT",
                ),
            ]
        )
    return DetectionReport(
        detected_at="2026-08-22T00:00:00Z",
        host_os="Linux 6.8.12-tegra",
        python_version="3.12.3",
        can=CanProbeResult(
            interfaces=interfaces,
            matches=[
                CanMatchRecord(
                    interfaces=interfaces,
                    chip="PEAK PCAN-USB Pro FD",
                    driver_hint="Damiao CAN FD motor bus — Enactic OpenArm",
                    embodiment_tag="openarm",
                    bh_robot_type="openarm",
                )
            ],
        ),
        cameras=camera_probe,
    )


class TestCanIdentification:
    def test_can_match_loads_the_canonical_openarm_manifest(self) -> None:
        assembled = assemble_robot_description(_openarm_cell_report(cameras=False))
        canonical = RobotDescription.from_yaml(str(OPENARM_MANIFEST))
        assert assembled.name == canonical.name == "openarm_v2"
        # The full 7+1 per side chain survives identification untouched.
        assert len(assembled.joints) == 16
        assert assembled.capabilities.bimanual is True

    def test_a_can_only_robot_is_no_longer_scaffolded_as_unknown(self) -> None:
        # The regression: no USB serial device, no DDS topics, yet a real arm.
        report = _openarm_cell_report(cameras=False)
        assert not report.usb.devices and not report.ros2.topics
        assert not assemble_robot_description(report).name.startswith("unknown_")

    def test_dds_still_outranks_can(self) -> None:
        # A robot that is already up and naming itself over DDS is stronger
        # evidence than an interface name a udev rule pinned at provisioning.
        from openral_detect.report import Ros2TopologyResult

        report = _openarm_cell_report(cameras=False).model_copy(
            update={"ros2": Ros2TopologyResult(inferred_robot_type="so101")}
        )
        assert assemble_robot_description(report).name != "openarm_v2"


class TestCameraEnrichment:
    def test_both_arducams_resolve_to_the_catalog_entry(self) -> None:
        assembled = assemble_robot_description(_openarm_cell_report(), enrich_cameras=True)
        arducams = [s for s in assembled.sensors if s.catalog_id == "arducam/b0495"]
        assert len(arducams) == 2
        assert all(s.model == "B0495" for s in arducams)
        assert all(s.metadata["shutter"] == "global" for s in arducams)

    def test_camera_serials_survive_into_metadata(self) -> None:
        # Two identical cameras on one rig are only addressable by serial.
        assembled = assemble_robot_description(_openarm_cell_report(), enrich_cameras=True)
        serials = {
            s.metadata.get("serial_no")
            for s in assembled.sensors
            if s.catalog_id == "arducam/b0495"
        }
        assert serials == {"CELL1_CAM_LEFT", "CELL1_CAM_RIGHT"}

    def test_lens_dependent_optics_are_flagged_not_fabricated(self) -> None:
        # The B0495 is a bare board with an M12 mount: FOV is the integrator's,
        # so the assembler must not write a made-up pinhole model.
        assembled = assemble_robot_description(_openarm_cell_report(), enrich_cameras=True)
        arducam = next(s for s in assembled.sensors if s.catalog_id == "arducam/b0495")
        assert arducam.intrinsics is None
        assert arducam.metadata["needs_calibration"] is True

    def test_zed_mini_resolves_to_a_multi_stream_bundle(self) -> None:
        # One V4L2 node, four streams — the stereo head must not be flattened.
        assembled = assemble_robot_description(_openarm_cell_report(), enrich_cameras=True)
        zed = next(
            b for b in assembled.sensor_bundles if any(s.model == "ZED Mini" for s in b.sensors)
        )
        assert isinstance(zed, SensorBundle)
        assert [s.modality for s in zed.sensors] == ["rgb", "rgb", "depth", "imu"]
        assert zed.sensors[2].range_max_m == 15.0

    def test_detected_cameras_do_not_collide_with_manifest_sensors(self) -> None:
        assembled = assemble_robot_description(_openarm_cell_report(), enrich_cameras=True)
        names = [s.name for s in assembled.sensors]
        names += [b.bundle_name for b in assembled.sensor_bundles]
        assert len(names) == len(set(names))

    def test_vision_capability_is_promoted(self) -> None:
        # The sim manifest declares has_vision: false; three real cameras flip it.
        canonical = RobotDescription.from_yaml(str(OPENARM_MANIFEST))
        assert canonical.capabilities.has_vision is False
        assembled = assemble_robot_description(_openarm_cell_report(), enrich_cameras=True)
        assert assembled.capabilities.has_vision is True


class TestRoundTrip:
    def test_assembled_cell_round_trips_through_yaml(self, tmp_path: Path) -> None:
        import yaml

        assembled = assemble_robot_description(_openarm_cell_report(), enrich_cameras=True)
        out = tmp_path / "robot.yaml"
        out.write_text(yaml.safe_dump(assembled.model_dump(mode="json")), encoding="utf-8")
        reloaded = RobotDescription.from_yaml(str(out))
        assert reloaded.name == assembled.name
        assert len(reloaded.sensors) == len(assembled.sensors)
        assert len(reloaded.sensor_bundles) == len(assembled.sensor_bundles)

    def test_assembly_does_not_mutate_the_committed_manifest(self) -> None:
        before = OPENARM_MANIFEST.read_text(encoding="utf-8")
        assemble_robot_description(_openarm_cell_report(), enrich_cameras=True)
        assert OPENARM_MANIFEST.read_text(encoding="utf-8") == before
