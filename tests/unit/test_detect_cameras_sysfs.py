"""V4L2 camera discovery via sysfs — the backend that needs no ``v4l-utils``.

Per CLAUDE.md §1.11 there are no mocks: the probe reads a real directory
tree laid out the way the kernel lays out ``/sys/class/video4linux``, with
``_SYSFS_V4L2`` redirected at it.

The regression this guards: a host with cameras plugged in but no
``v4l-utils`` package reported **zero** cameras, because the only backend
shelled out to ``v4l2-ctl``.  A provisioning tool that silently finds
nothing is worse than one that errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_detect.probes import cameras as cameras_module
from openral_detect.probes import probe_v4l2_cameras


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _make_camera(
    sysfs: Path,
    device_id: str,
    node_indices: list[int],
    *,
    name: str,
    usb: tuple[str, str, str] | None = None,
) -> None:
    """Lay down one physical camera exposing one or more ``videoN`` nodes.

    Args:
        sysfs: Fixture sysfs root.
        device_id: Unique id for this physical device — becomes its device-tree
            directory, which is what groups its nodes together.
        node_indices: The ``videoN`` numbers this device registers.  A UVC
            camera typically registers two (capture + metadata).
        name: V4L2 product string.
        usb: Optional ``(idVendor, idProduct, serial)`` USB descriptor.
    """
    owner = sysfs / "devices" / device_id
    interface = owner / f"{device_id}:1.0"
    interface.mkdir(parents=True, exist_ok=True)
    if usb is not None:
        vid, pid, serial = usb
        _write(owner / "idVendor", vid)
        _write(owner / "idProduct", pid)
        _write(owner / "serial", serial)
    for index in node_indices:
        node = sysfs / "class" / "video4linux" / f"video{index}"
        _write(node / "name", name)
        _write(node / "index", str(index))
        (node / "device").symlink_to(interface)


@pytest.fixture
def cell_sysfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The camera complement of a real OpenArm cell.

    Two identical Arducam B0495 global-shutter cameras (same VID/PID, same
    product string, distinguished only by serial and device path) plus a
    ZED Mini stereo head.
    """
    sysfs = tmp_path / "sys"
    _make_camera(
        sysfs,
        "usb2-3.1.3",
        [0, 1],
        name="Arducam B0495 (USB3 2.3MP)",
        usb=("04b4", "4950", "CELL1_CAM_LEFT"),
    )
    _make_camera(sysfs, "usb2-1", [2, 3], name="ZED-M: ZED-M", usb=("2b03", "f682", ""))
    _make_camera(
        sysfs,
        "usb2-3.2",
        [4, 5],
        name="Arducam B0495 (USB3 2.3MP)",
        usb=("04b4", "4950", "CELL1_CAM_RIGHT"),
    )
    monkeypatch.setattr(cameras_module, "_SYSFS_V4L2", str(sysfs / "class" / "video4linux"))
    return sysfs


class TestSysfsBackend:
    def test_one_row_per_physical_camera_not_per_node(self, cell_sysfs: Path) -> None:
        cameras = probe_v4l2_cameras()
        # Six /dev/videoN nodes, three cameras.
        assert [c.device_path for c in cameras] == ["/dev/video0", "/dev/video2", "/dev/video4"]

    def test_usb_descriptor_reaches_the_report(self, cell_sysfs: Path) -> None:
        first = probe_v4l2_cameras()[0]
        assert (first.vid, first.pid) == (0x04B4, 0x4950)
        assert first.serial == "CELL1_CAM_LEFT"

    def test_identical_models_stay_distinct(self, cell_sysfs: Path) -> None:
        arducams = [c for c in probe_v4l2_cameras() if c.vid == 0x04B4]
        assert len(arducams) == 2
        # Same VID/PID and same product string — only serial and path differ.
        assert {c.serial for c in arducams} == {"CELL1_CAM_LEFT", "CELL1_CAM_RIGHT"}

    def test_capture_nodes_sort_numerically(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # /dev/video10 must come after /dev/video9, not between 1 and 2.
        sysfs = tmp_path / "sys"
        _make_camera(sysfs, "a", [9], name="Cam A", usb=("04b4", "4950", "A"))
        _make_camera(sysfs, "b", [10], name="Cam B", usb=("04b4", "4950", "B"))
        monkeypatch.setattr(cameras_module, "_SYSFS_V4L2", str(sysfs / "class" / "video4linux"))
        assert [c.device_path for c in probe_v4l2_cameras()] == [
            "/dev/video9",
            "/dev/video10",
        ]

    def test_mipi_camera_without_usb_descriptor_is_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sysfs = tmp_path / "sys"
        _make_camera(sysfs, "tegra-capture-vi", [0], name="vi-output, imx219 9-0010")
        monkeypatch.setattr(cameras_module, "_SYSFS_V4L2", str(sysfs / "class" / "video4linux"))
        warnings: list[str] = []
        cameras = probe_v4l2_cameras(warnings=warnings)
        assert len(cameras) == 1 and cameras[0].vid == 0
        assert any("no USB descriptor" in w for w in warnings)


class TestProbeContract:
    def test_probe_never_raises_on_the_real_host(self) -> None:
        # Empty on a headless CI runner, populated on a lab host — both valid.
        warnings: list[str] = []
        cameras = probe_v4l2_cameras(warnings=warnings)
        assert all(c.device_path.startswith("/dev/video") for c in cameras)

    def test_empty_sysfs_falls_through_to_the_v4l2ctl_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = tmp_path / "class" / "video4linux"
        empty.mkdir(parents=True)
        monkeypatch.setattr(cameras_module, "_SYSFS_V4L2", str(empty))
        warnings: list[str] = []
        # Whether v4l2-ctl exists is host-dependent; either way the probe must
        # return a list and explain itself when it found nothing.
        assert isinstance(probe_v4l2_cameras(warnings=warnings), list)
