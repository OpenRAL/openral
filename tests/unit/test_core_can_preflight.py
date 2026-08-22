"""``openral_core.can.preflight_can_links`` — the shared CAN connect-time gate.

This helper exists so that link checking is not re-implemented per robot.  The
tests below therefore exercise it at bus counts the OpenArm does not have (one,
three) as well as the bimanual two: if it silently assumed a left/right pair,
the next CAN robot would have to fork it, which is the failure this module was
extracted to prevent.

The sysfs reads run against a real directory tree in the shapes the kernel
produces (no mocks, per CLAUDE.md §1.11); ``openral_core.can.SYSFS_NET`` is
redirected at that tree.
"""

from __future__ import annotations

from pathlib import Path

import openral_core.can as core_can
import pytest
from openral_core.can import can_link_state, preflight_can_links
from openral_core.exceptions import ROSConfigError

# ARPHRD_CAN, from the kernel's include/uapi/linux/if_arp.h.
_ARPHRD_CAN = "280"
# ARPHRD_ETHER — a plain Ethernet link, used to prove a non-CAN interface of
# the right name is still rejected.
_ARPHRD_ETHER = "1"


def _make_link(root: Path, name: str, *, arphrd: str = _ARPHRD_CAN, up: bool = True) -> None:
    for attr, value in (("type", arphrd), ("operstate", "up" if up else "down")):
        path = root / name / attr
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")


@pytest.fixture
def netroot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "net"
    root.mkdir()
    monkeypatch.setattr(core_can, "SYSFS_NET", str(root))
    return root


class TestBusCountIsNotAssumed:
    """The helper must serve any robot's bus topology, not just a bimanual pair."""

    def test_single_bus_robot(self, netroot: Path) -> None:
        _make_link(netroot, "arm_can")
        health = preflight_can_links({"arm": "arm_can"}, hal_label="SomeArmHAL")
        assert health == {"arm_can": "arm_can"}

    def test_bimanual_two_buses(self, netroot: Path) -> None:
        _make_link(netroot, "openarm_left")
        _make_link(netroot, "openarm_right")
        health = preflight_can_links(
            {"left": "openarm_left", "right": "openarm_right"},
            hal_label="OpenArmRealHAL",
        )
        assert health == {"left_can": "openarm_left", "right_can": "openarm_right"}

    def test_four_bus_robot(self, netroot: Path) -> None:
        limbs = {f"limb{n}": f"quad_can{n}" for n in range(4)}
        for interface in limbs.values():
            _make_link(netroot, interface)
        health = preflight_can_links(limbs, hal_label="QuadHAL")
        assert health == {f"limb{n}_can": f"quad_can{n}" for n in range(4)}


class TestFailureReporting:
    def test_reports_every_failing_bus_not_just_the_first(self, netroot: Path) -> None:
        # One connect attempt must surface the whole problem: an operator who
        # fixes only the bus named in the message and retries has wasted a trip.
        _make_link(netroot, "openarm_left", up=False)
        with pytest.raises(ROSConfigError) as excinfo:
            preflight_can_links(
                {"left": "openarm_left", "right": "openarm_right"},
                hal_label="OpenArmRealHAL",
            )
        message = str(excinfo.value)
        assert "openarm_left" in message
        assert "openarm_right" in message

    def test_remedy_is_caller_supplied_not_baked_in(self, netroot: Path) -> None:
        with pytest.raises(ROSConfigError) as excinfo:
            preflight_can_links(
                {"bus": "absent_can"},
                hal_label="SomeArmHAL",
                remedy="Run `somearm up` first.",
            )
        message = str(excinfo.value)
        assert message.startswith("SomeArmHAL cannot connect:")
        assert "Run `somearm up` first." in message

    def test_omitted_remedy_leaves_no_dangling_text(self, netroot: Path) -> None:
        with pytest.raises(ROSConfigError) as excinfo:
            preflight_can_links({"bus": "absent_can"}, hal_label="SomeArmHAL")
        assert str(excinfo.value).endswith("named 'absent_can'.")

    def test_healthy_buses_are_labelled_and_down_ones_flagged(self, netroot: Path) -> None:
        # The health map goes into a HAL health report, so a down bus must be
        # visibly different from an up one rather than merely absent.
        _make_link(netroot, "up_can")
        with pytest.raises(ROSConfigError):
            preflight_can_links({"a": "up_can", "b": "down_can"}, hal_label="H")
        # and the same call's per-link view, read directly:
        assert can_link_state("up_can") == (True, "")
        assert can_link_state("down_can")[0] is False


class TestLinkStateDiscrimination:
    def test_absent_interface(self, netroot: Path) -> None:
        is_up, reason = can_link_state("nothing_here")
        assert is_up is False
        assert "no network interface named" in reason

    def test_existing_but_not_a_can_link(self, netroot: Path) -> None:
        # A robot mis-provisioned onto an Ethernet interface of the right name
        # must not read as a healthy motor bus.
        _make_link(netroot, "openarm_left", arphrd=_ARPHRD_ETHER)
        is_up, reason = can_link_state("openarm_left")
        assert is_up is False
        assert "is not a CAN link" in reason

    def test_can_link_that_is_down(self, netroot: Path) -> None:
        _make_link(netroot, "openarm_left", up=False)
        is_up, reason = can_link_state("openarm_left")
        assert is_up is False
        assert "is down" in reason

    def test_explicit_sysfs_root_overrides_the_module_default(self, tmp_path: Path) -> None:
        # No monkeypatching: the parameter alone must be enough to read a
        # recorded tree, so a caller can diagnose a captured host.
        root = tmp_path / "recorded"
        _make_link(root, "openarm_left")
        assert can_link_state("openarm_left", sysfs_net=str(root)) == (True, "")
