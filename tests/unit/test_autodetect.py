"""Unit tests for ``openral_cli.autodetect``: USB + DDS discovery helpers.

All tests run without hardware — USB and subprocess calls are patched.
The end-to-end ``openral detect`` flow that consumes these helpers is covered
by ``tests/unit/test_detect_*.py``.
"""

from __future__ import annotations

import platform
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from openral_cli.autodetect import (
    _TOPIC_ROBOT_MAP,
    _VID_PID_TABLE,
    DdsTopic,
    KnownDevice,
    UsbDevice,
    UsbMatch,
    enumerate_usb_devices,
    infer_robot_from_topics,
    match_known_devices,
    scan_dds_topics,
    unitree_candidates_from_topics,
)

# ── VID/PID table ─────────────────────────────────────────────────────────────


class TestVidPidTable:
    def test_table_non_empty(self) -> None:
        assert len(_VID_PID_TABLE) > 0

    def test_ch340_in_table(self) -> None:
        assert (0x1A86, 0x7523) in _VID_PID_TABLE

    def test_cp2102_in_table(self) -> None:
        assert (0x10C4, 0xEA60) in _VID_PID_TABLE

    def test_ftdi_in_table(self) -> None:
        assert (0x0403, 0x6001) in _VID_PID_TABLE

    def test_ch340_maps_to_so101_by_default(self) -> None:
        # SO-100 and SO-101 are indistinguishable over USB; the current SO-101
        # is the default (an SO-100 is selected via `--robot so100`).
        entry = _VID_PID_TABLE[(0x1A86, 0x7523)]
        assert entry.bh_robot_type == "so101"
        assert entry.embodiment_tag == "so101_follower"

    def test_ch343_cdc_acm_maps_to_so101(self) -> None:
        # Many SO-101 control boards ship the WCH CH343 (1a86:55d3), which
        # enumerates as a CDC-ACM /dev/ttyACM* node — it must map to SO-101
        # just like the CH340/CH9102 variants.
        entry = _VID_PID_TABLE[(0x1A86, 0x55D3)]
        assert entry.bh_robot_type == "so101"
        assert entry.embodiment_tag == "so101_follower"

    def test_known_device_is_named_tuple(self) -> None:
        entry = _VID_PID_TABLE[(0x1A86, 0x7523)]
        assert isinstance(entry, KnownDevice)
        assert entry.chip  # non-empty string

    def test_all_entries_have_chip_and_hint(self) -> None:
        for (vid, pid), kd in _VID_PID_TABLE.items():
            assert kd.chip, f"VID={vid:04X} PID={pid:04X} has empty chip name"
            assert kd.driver_hint, f"VID={vid:04X} PID={pid:04X} has empty driver_hint"


# ── match_known_devices ────────────────────────────────────────────────────────


class TestMatchKnownDevices:
    def test_empty_input(self) -> None:
        assert match_known_devices([]) == []

    def test_known_vid_pid_matched(self) -> None:
        dev = UsbDevice(port="/dev/ttyUSB0", vid=0x1A86, pid=0x7523, description="CH340")
        matches = match_known_devices([dev])
        assert len(matches) == 1
        assert matches[0].device is dev
        assert matches[0].known.bh_robot_type == "so101"

    def test_unknown_vid_pid_not_matched(self) -> None:
        dev = UsbDevice(port="/dev/ttyUSB0", vid=0xDEAD, pid=0xBEEF, description="unknown")
        matches = match_known_devices([dev])
        assert matches == []

    def test_zero_vid_not_matched(self) -> None:
        """Glob fallback devices have vid=0; must not spuriously match."""
        dev = UsbDevice(port="/dev/ttyUSB0", vid=0, pid=0, description="")
        matches = match_known_devices([dev])
        assert matches == []

    def test_multiple_devices(self) -> None:
        devs = [
            UsbDevice("/dev/ttyUSB0", 0x1A86, 0x7523, "CH340"),
            UsbDevice("/dev/ttyUSB1", 0xDEAD, 0xBEEF, "unknown"),
            UsbDevice("/dev/ttyUSB2", 0x10C4, 0xEA60, "CP2102"),
        ]
        matches = match_known_devices(devs)
        assert len(matches) == 2
        ports = [m.device.port for m in matches]
        assert "/dev/ttyUSB0" in ports
        assert "/dev/ttyUSB2" in ports

    def test_returns_usb_match_type(self) -> None:
        dev = UsbDevice("/dev/ttyUSB0", 0x1A86, 0x7523, "CH340")
        (match,) = match_known_devices([dev])
        assert isinstance(match, UsbMatch)

    def test_cp2104_matched(self) -> None:
        dev = UsbDevice("/dev/ttyACM0", 0x10C4, 0xEA6A, "CP2104")
        (match,) = match_known_devices([dev])
        assert match.known.chip == "CP2104"

    def test_ftdi_matched_but_ambiguous(self) -> None:
        """FTDI FT232RL is matched but bh_robot_type is empty (ambiguous)."""
        dev = UsbDevice("/dev/ttyUSB0", 0x0403, 0x6001, "FT232RL")
        (match,) = match_known_devices([dev])
        assert match.known.bh_robot_type == ""  # ambiguous — needs DDS to resolve


# ── infer_robot_from_topics ───────────────────────────────────────────────────


class TestInferRobotFromTopics:
    def test_empty_input(self) -> None:
        assert infer_robot_from_topics([]) is None

    def test_unitree_hg_family_is_ambiguous(self) -> None:
        # G1 / H1-2 / H2 all publish unitree_hg/LowState on /lowstate — the
        # topic list cannot pick one, so inference must not guess.
        topics = [DdsTopic("/lowstate", "unitree_hg/msg/LowState")]
        assert infer_robot_from_topics(topics) is None

    def test_unitree_go_family_is_ambiguous(self) -> None:
        # The original H1, Go2 and B2 all use unitree_go/LowCmd.
        topics = [DdsTopic("/lowcmd", "unitree_go/msg/LowCmd")]
        assert infer_robot_from_topics(topics) is None

    def test_unitree_singleton_family_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Extension point: a family with exactly one model resolves to it.
        import openral_cli.autodetect as ad

        monkeypatch.setitem(ad._UNITREE_FAMILY_CANDIDATES, "unitree_hg", ("g1",))
        topics = [DdsTopic("/lowstate", "unitree_hg/msg/LowState")]
        assert infer_robot_from_topics(topics) == "g1"

    def test_ambiguous_unitree_does_not_mask_other_robot(self) -> None:
        topics = [
            DdsTopic("/lowstate", "unitree_hg/msg/LowState"),
            DdsTopic("/so101/joint_state", "sensor_msgs/msg/JointState"),
        ]
        assert infer_robot_from_topics(topics) == "so101"

    def test_aloha_topic(self) -> None:
        topics = [DdsTopic("/follower_arms_position_goal", "std_msgs/msg/Float64MultiArray")]
        assert infer_robot_from_topics(topics) == "aloha"

    def test_so101_topic(self) -> None:
        topics = [DdsTopic("/so101/joint_state", "sensor_msgs/msg/JointState")]
        assert infer_robot_from_topics(topics) == "so101"

    def test_so100_topic_still_resolves_explicitly(self) -> None:
        # An explicit SO-100 lifecycle node is still honoured as so100.
        topics = [DdsTopic("/so100/joint_state", "sensor_msgs/msg/JointState")]
        assert infer_robot_from_topics(topics) == "so100"

    def test_unknown_topic_returns_none(self) -> None:
        topics = [DdsTopic("/rosout", "rcl_interfaces/msg/Log")]
        assert infer_robot_from_topics(topics) is None

    def test_first_match_wins(self) -> None:
        """When multiple robot topics are present, the alphabetically first is returned."""
        topics = sorted(
            [
                DdsTopic("/lowstate", "unitree_go/msg/LowState"),
                DdsTopic("/follower_arms_position_goal", "std_msgs/msg/Float64MultiArray"),
            ],
            key=lambda t: t.name,
        )
        # /follower_arms_position_goal < /lowstate alphabetically → aloha first
        assert infer_robot_from_topics(topics) == "aloha"

    def test_topic_map_non_empty(self) -> None:
        assert len(_TOPIC_ROBOT_MAP) > 0

    def test_case_insensitive_match(self) -> None:
        """Topic names may be uppercase on some DDS implementations."""
        topics = [DdsTopic("/LowState", "unitree_hg/msg/LowState")]
        assert unitree_candidates_from_topics(topics) == ("g1", "h1_2", "h2")


# ── unitree_candidates_from_topics ────────────────────────────────────────────


class TestUnitreeCandidates:
    def test_no_unitree_topic(self) -> None:
        assert unitree_candidates_from_topics([DdsTopic("/tf", "tf2_msgs/msg/TFMessage")]) == ()

    def test_hg_family(self) -> None:
        # unitree_sdk2 example/g1, example/h1 (h1_2_*), example/h2 use idl/hg.
        topics = [DdsTopic("/lowstate", "unitree_hg/msg/LowState")]
        assert unitree_candidates_from_topics(topics) == ("g1", "h1_2", "h2")

    def test_go_family(self) -> None:
        # unitree_sdk2 example/h1 (humanoid.hpp), example/go2, example/b2 use idl/go2.
        topics = [DdsTopic("/lowstate", "unitree_go/msg/LowState")]
        assert unitree_candidates_from_topics(topics) == ("h1", "go2", "b2")

    def test_untyped_topic_keeps_every_model(self) -> None:
        topics = [DdsTopic("/lowstate", "std_msgs/msg/String")]
        assert set(unitree_candidates_from_topics(topics)) == {
            "g1",
            "h1_2",
            "h2",
            "h1",
            "go2",
            "b2",
        }

    def test_sportmodestate_alone_does_not_pick_family(self) -> None:
        # SportModeState lives only in unitree_go, whichever family publishes it.
        topics = [DdsTopic("/sportmodestate", "unitree_go/msg/SportModeState")]
        assert len(unitree_candidates_from_topics(topics)) == 6

    def test_candidates_with_manifest_resolve_to_robot_yaml(self) -> None:
        from openral_detect.registry import canonical_robot_path

        for model in ("g1", "h1"):
            path = canonical_robot_path(model)
            assert path is not None and path.parent.name == model

    def test_dfu_bootloader_not_mapped_to_a_robot(self) -> None:
        # 0483:df11 is the generic STM32 ROM DFU bootloader, not a Unitree id.
        assert (0x0483, 0xDF11) not in _VID_PID_TABLE


# ── scan_dds_topics ───────────────────────────────────────────────────────────


class TestScanDdsTopics:
    def test_ros2_not_on_path_returns_empty(self) -> None:
        with patch("shutil.which", return_value=None):
            topics = scan_dds_topics(timeout_s=0.1)
        assert topics == []

    def test_subprocess_timeout_returns_empty(self) -> None:
        import subprocess

        with (
            patch("shutil.which", return_value="/usr/bin/ros2"),
            patch(
                "subprocess.check_output",
                side_effect=subprocess.TimeoutExpired(cmd="ros2", timeout=0.1),
            ),
        ):
            topics = scan_dds_topics(timeout_s=0.1)
        assert topics == []

    def test_parses_ros2_topic_list_output(self) -> None:
        fake_output = (
            "/lowstate [unitree_go/msg/LowState]\n"
            "/lowcmd [unitree_go/msg/LowCmd]\n"
            "/rosout [rcl_interfaces/msg/Log]\n"
        )
        with (
            patch("shutil.which", return_value="/usr/bin/ros2"),
            patch("subprocess.check_output", return_value=fake_output),
        ):
            topics = scan_dds_topics()

        assert len(topics) == 3
        names = [t.name for t in topics]
        assert "/lowstate" in names
        assert "/rosout" in names

    def test_topic_type_parsed(self) -> None:
        fake_output = "/lowstate [unitree_go/msg/LowState]\n"
        with (
            patch("shutil.which", return_value="/usr/bin/ros2"),
            patch("subprocess.check_output", return_value=fake_output),
        ):
            (topic,) = scan_dds_topics()

        assert topic.name == "/lowstate"
        assert topic.type_name == "unitree_go/msg/LowState"

    def test_topics_sorted_by_name(self) -> None:
        fake_output = "/z_topic [a/b/C]\n/a_topic [x/y/Z]\n"
        with (
            patch("shutil.which", return_value="/usr/bin/ros2"),
            patch("subprocess.check_output", return_value=fake_output),
        ):
            topics = scan_dds_topics()

        assert topics[0].name == "/a_topic"
        assert topics[1].name == "/z_topic"

    def test_empty_output_returns_empty(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/ros2"),
            patch("subprocess.check_output", return_value=""),
        ):
            topics = scan_dds_topics()
        assert topics == []


# ── enumerate_usb_devices ─────────────────────────────────────────────────────


class TestEnumerateUsbDevices:
    def test_returns_list(self) -> None:
        """enumerate_usb_devices() must always return a list (even if empty)."""
        devs = enumerate_usb_devices()
        assert isinstance(devs, list)

    def test_linux_pyudev_happy_path(self) -> None:
        """Simulate pyudev returning one USB serial device on Linux."""
        mock_dev: dict[str, Any] = {
            "DEVNAME": "/dev/ttyUSB0",
            "ID_VENDOR_ID": "1a86",
            "ID_MODEL_ID": "7523",
            "ID_MODEL": "USB_Serial",
        }

        def fake_get(key: str, default: str = "") -> str:
            return str(mock_dev.get(key, default))

        mock_udev_dev = MagicMock()
        mock_udev_dev.get.side_effect = fake_get
        mock_udev_dev.find_parent.return_value = mock_udev_dev  # parent = self for simplicity

        mock_ctx = MagicMock()
        mock_ctx.list_devices.return_value = [mock_udev_dev]

        mock_pyudev = MagicMock()
        mock_pyudev.Context.return_value = mock_ctx

        with (
            patch.dict("sys.modules", {"pyudev": mock_pyudev}),
            patch("platform.system", return_value="Linux"),
        ):
            from openral_cli import autodetect

            devs = autodetect._enumerate_linux_pyudev()

        assert len(devs) == 1
        assert devs[0].port == "/dev/ttyUSB0"
        assert devs[0].vid == 0x1A86
        assert devs[0].pid == 0x7523

    def test_linux_pyudev_failure_fallback_to_empty(self) -> None:
        """pyudev import error → returns empty list (caller falls back to glob)."""
        mock_pyudev = MagicMock()
        mock_pyudev.Context.side_effect = RuntimeError("no udev")

        with patch.dict("sys.modules", {"pyudev": mock_pyudev}):
            from openral_cli import autodetect

            devs = autodetect._enumerate_linux_pyudev()

        assert devs == []

    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux glob test")
    def test_glob_fallback_linux(self) -> None:
        """Glob fallback returns UsbDevice with vid=0."""
        with patch(
            "glob.glob",
            side_effect=lambda p: [p.replace("*", "0")] if "USB" in p or "ACM" in p else [],
        ):
            from openral_cli import autodetect

            devs = autodetect._enumerate_glob_fallback()

        # All glob fallback devices have vid=0
        for d in devs:
            assert d.vid == 0
            assert d.pid == 0


# ── probe_dds: Unitree ambiguity is reported, not guessed ─────────────────────


class TestProbeDdsUnitreeAmbiguity:
    def test_ambiguous_unitree_topology_warns_with_candidates(self) -> None:
        from openral_detect.probes.dds import probe_dds

        fake_output = "/lowcmd [unitree_hg/msg/LowCmd]\n/lowstate [unitree_hg/msg/LowState]\n"
        warnings: list[str] = []
        with (
            patch("shutil.which", return_value="/usr/bin/ros2"),
            patch("subprocess.check_output", return_value=fake_output),
        ):
            result = probe_dds(timeout_s=0.1, warnings=warnings)
        assert result.inferred_robot_type is None
        (msg,) = [w for w in warnings if "Unitree" in w]
        assert "g1, h1_2, h2" in msg
        assert "--robot" in msg
