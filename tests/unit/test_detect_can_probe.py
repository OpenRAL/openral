"""SocketCAN discovery — the transport that finds a CAN-attached arm.

Per CLAUDE.md §1.11 there are no mocks here.  The enumeration reads a real
directory tree in the shapes the kernel produces; ``openral_core.can.SYSFS_NET`` is
redirected at that tree with ``monkeypatch.setattr`` so the code under test
is the production code, reading real files.

Fixture shape mirrors a provisioned OpenArm cell: two PCAN-USB Pro FD
channels renamed by udev to ``openarm_left`` / ``openarm_right``, plus the
carrier board's unused onboard ``can0`` and an ordinary Ethernet link that
must not be mistaken for a CAN bus.
"""

from __future__ import annotations

from pathlib import Path

import openral_core.can as core_can
import pytest
from openral_cli.autodetect import (
    CanInterface,
    can_adapter_name,
    enumerate_can_interfaces,
    infer_robot_from_can,
    match_can_interfaces,
)
from openral_detect.probes import diagnose_can_matches, probe_can
from openral_detect.report import CanInterfaceInfo, CanMatchRecord

_ARPHRD_CAN = "280"
_ARPHRD_ETHER = "1"


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _make_link(
    sysfs: Path,
    name: str,
    *,
    arphrd: str = _ARPHRD_CAN,
    operstate: str = "up",
    mtu: int = 72,
    usb: tuple[str, str, str] | None = None,
) -> None:
    """Lay down one ``/sys/class/net/<name>`` entry and its device tree.

    Reproduces the kernel's real layout: the class entry's ``device`` is a
    **symlink** into a separate device tree, and the USB descriptor lives on
    an ancestor of the symlink target — so the ancestor walk is genuinely
    exercised rather than short-circuited by a flat directory.

    Args:
        sysfs: The fixture's sysfs root; ``sysfs/class/net`` holds the class
            entries and ``sysfs/devices`` the device tree.
        name: Interface name.
        arphrd: ``type`` attribute — ``"280"`` marks a CAN link.
        operstate: ``operstate`` attribute.
        mtu: ``mtu`` attribute — 16 classic CAN, 72 CAN FD.
        usb: Optional ``(idVendor, idProduct, product)`` for a USB parent.
            ``None`` models an on-SoC controller with no USB ancestor.
    """
    link = sysfs / "class" / "net" / name
    _write(link / "type", arphrd)
    _write(link / "operstate", operstate)
    _write(link / "mtu", str(mtu))

    # The USB device node (or the SoC platform node) that owns this interface.
    owner = sysfs / "devices" / name
    interface = owner / f"{name}:1.0"
    interface.mkdir(parents=True, exist_ok=True)
    if usb is not None:
        vid, pid, product = usb
        _write(owner / "idVendor", vid)
        _write(owner / "idProduct", pid)
        _write(owner / "product", product)
    (link / "device").symlink_to(interface)


@pytest.fixture
def openarm_sysfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A provisioned OpenArm cell as the kernel would present it."""
    sysfs = tmp_path / "sys"
    for side in ("openarm_left", "openarm_right"):
        _make_link(sysfs, side, usb=("0c72", "0011", "PCAN-USB Pro FD"))
    # Onboard carrier-board CAN, never brought up, no USB ancestor.
    _make_link(sysfs, "can0", operstate="down", mtu=16)
    # An Ethernet link must never be reported as a CAN bus.
    _make_link(sysfs, "eth0", arphrd=_ARPHRD_ETHER, mtu=1500)
    monkeypatch.setattr(core_can, "SYSFS_NET", str(sysfs / "class" / "net"))
    return sysfs


class TestEnumeration:
    def test_only_can_links_are_enumerated(self, openarm_sysfs: Path) -> None:
        names = [i.name for i in enumerate_can_interfaces()]
        assert names == ["can0", "openarm_left", "openarm_right"]
        assert "eth0" not in names

    def test_usb_parent_descriptor_is_read_through_the_ancestor_walk(
        self, openarm_sysfs: Path
    ) -> None:
        left = next(i for i in enumerate_can_interfaces() if i.name == "openarm_left")
        assert (left.vid, left.pid) == (0x0C72, 0x0011)
        assert left.description == "PCAN-USB Pro FD"

    def test_onboard_controller_has_no_usb_ancestor(self, openarm_sysfs: Path) -> None:
        can0 = next(i for i in enumerate_can_interfaces() if i.name == "can0")
        assert (can0.vid, can0.pid) == (0, 0)
        assert can_adapter_name(can0) == "SocketCAN"

    def test_mtu_separates_classic_can_from_can_fd(self, openarm_sysfs: Path) -> None:
        by_name = {i.name: i for i in enumerate_can_interfaces()}
        # `ip` is not consulted for this fixture tree (its links do not exist
        # on the host), so FD mode is inferred from the MTU.
        assert by_name["can0"].mtu == 16
        assert by_name["openarm_left"].mtu == 72

    def test_missing_sysfs_root_yields_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(core_can, "SYSFS_NET", str(tmp_path / "absent"))
        assert enumerate_can_interfaces() == []


class TestMatching:
    def test_both_arms_group_into_one_robot(self, openarm_sysfs: Path) -> None:
        matches = match_can_interfaces(enumerate_can_interfaces())
        assert len(matches) == 1
        assert [i.name for i in matches[0].interfaces] == ["openarm_left", "openarm_right"]
        assert matches[0].known.bh_robot_type == "openarm"
        # The adapter, not the generic transport, is what gets named.
        assert matches[0].known.chip == "PEAK PCAN-USB Pro FD"

    def test_inference_returns_the_robot_slug(self, openarm_sysfs: Path) -> None:
        assert infer_robot_from_can(enumerate_can_interfaces()) == "openarm"

    def test_a_bare_can0_is_not_a_robot(self) -> None:
        # `can0` is a kernel enumeration artefact, not a declaration of intent.
        bare = CanInterface("can0", True, False, 500000, 0, "ERROR-ACTIVE", "mttcan", 16, 0, 0, "")
        assert match_can_interfaces([bare]) == []
        assert infer_robot_from_can([bare]) is None

    def test_a_down_link_does_not_match(self) -> None:
        # The kernel knowing about a controller is not evidence of a robot.
        down = CanInterface(
            "openarm_left",
            False,
            True,
            1000000,
            5000000,
            "STOPPED",
            "pcan_usb_pro_fd",
            72,
            0x0C72,
            0x0011,
            "PCAN-USB Pro FD",
        )
        assert match_can_interfaces([down]) == []


class TestProbeContract:
    def test_probe_never_raises_and_matches_refer_to_enumerated_links(self) -> None:
        # Host-agnostic: empty on a CI runner with no CAN controller, populated
        # on a lab host with an arm wired up — both are valid.
        warnings: list[str] = []
        result = probe_can(warnings=warnings)
        names = {i.name for i in result.interfaces}
        for match in result.matches:
            assert {i.name for i in match.interfaces} <= names

    def test_unmatched_interfaces_explain_the_udev_remedy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sysfs = tmp_path / "sys"
        _make_link(sysfs, "can0", mtu=16)
        monkeypatch.setattr(core_can, "SYSFS_NET", str(sysfs / "class" / "net"))
        warnings: list[str] = []
        result = probe_can(warnings=warnings)
        assert result.interfaces and not result.matches
        assert any("udev" in w for w in warnings)


class TestDiagnosis:
    """`diagnose_can_matches` is the operator-facing reading of `state`."""

    @staticmethod
    def _match(state: str) -> CanMatchRecord:
        return CanMatchRecord(
            interfaces=[
                CanInterfaceInfo(
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
                for name in ("openarm_left", "openarm_right")
            ],
            chip="PEAK PCAN-USB Pro FD",
            driver_hint="Damiao CAN FD motor bus",
            embodiment_tag="openarm",
            bh_robot_type="openarm",
        )

    def test_error_passive_is_named_as_unpowered_motors(self) -> None:
        # Frames leave the adapter, nothing ACKs: the arm is detected but will
        # not move, and the operator must not have to decode `state` by hand.
        lines = diagnose_can_matches([self._match("ERROR-PASSIVE")])
        assert len(lines) == 1
        assert "openarm_left" in lines[0] and "unpowered" in lines[0]

    def test_bus_off_points_at_termination_and_bitrate(self) -> None:
        lines = diagnose_can_matches([self._match("BUS-OFF")])
        assert len(lines) == 1
        assert "termination" in lines[0]

    def test_a_healthy_bus_says_nothing(self) -> None:
        assert diagnose_can_matches([self._match("ERROR-ACTIVE")]) == []
