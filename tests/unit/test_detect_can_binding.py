"""``_enrich_can_buses`` — writing the host's real CAN names into the config.

A CAN interface name is a property of the host, not of the robot: the same
bimanual arm is ``openarm_left`` / ``openarm_right`` where a udev rule pins it
and ``can1`` / ``can0`` where none does. So a canonical manifest can only
declare *which parameter each bus fills*; the value has to be discovered.

These tests deliberately use host names that differ from the OpenArm
manifest's own defaults — otherwise a no-op would pass. They also exercise bus
counts the OpenArm does not have (one, four), because the mechanism is meant to
serve any CAN robot, and check that an unresolvable role warns rather than
guessing: binding one limb to another limb's bus would look like it worked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from openral_core.schemas import RobotDescription
from openral_detect.assemble import _enrich_can_buses
from openral_detect.report import (
    CanInterfaceInfo,
    CanMatchRecord,
    CanProbeResult,
    DetectionReport,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENARM_MANIFEST = REPO_ROOT / "robots" / "openarm" / "robot.yaml"


def _openarm(**parameter_overrides: Any) -> RobotDescription:
    """The real committed OpenArm manifest, optionally re-parameterised."""
    raw = yaml.safe_load(OPENARM_MANIFEST.read_text(encoding="utf-8"))
    description = RobotDescription.model_validate(raw)
    if not parameter_overrides:
        return description
    parameters = description.hal.parameters.model_copy(update=parameter_overrides)
    return description.model_copy(
        update={"hal": description.hal.model_copy(update={"parameters": parameters})}
    )


def _report(*interface_names: str, robot: str = "openarm") -> DetectionReport:
    """A DetectionReport whose CAN probe matched ``interface_names`` to ``robot``."""
    report = DetectionReport(detected_at="2026-05-10T00:00:00Z")
    if interface_names:
        report.can = CanProbeResult(
            interfaces=[CanInterfaceInfo(name=n, is_up=True) for n in interface_names],
            matches=[
                CanMatchRecord(
                    interfaces=[CanInterfaceInfo(name=n, is_up=True) for n in interface_names],
                    chip="SocketCAN",
                    driver_hint="test",
                    embodiment_tag=robot,
                    bh_robot_type=robot,
                )
            ],
        )
    return report


def _can_defaults(description: RobotDescription) -> dict[str, object]:
    return {k: v for k, v in description.hal.parameters.defaults.items() if "can_interface" in k}


class TestItActuallyOverwrites:
    def test_host_names_replace_the_manifest_defaults(self) -> None:
        # The committed manifest says openarm_left/openarm_right. This host
        # names them differently; the generated config must follow the host.
        report = _report("arm_left_bus", "arm_right_bus")
        out = _enrich_can_buses(_openarm(), report)
        assert _can_defaults(out) == {
            "left_can_interface": "arm_left_bus",
            "right_can_interface": "arm_right_bus",
        }
        assert [w for w in report.warnings if w.startswith("can.bind")] == []

    def test_binding_is_by_role_not_by_position(self) -> None:
        # Sorted order here puts the RIGHT bus first. A positional zip would
        # bind the left arm to the right arm's bus and look perfectly fine.
        report = _report("aaa_right", "zzz_left")
        out = _enrich_can_buses(_openarm(), report)
        assert _can_defaults(out) == {
            "left_can_interface": "zzz_left",
            "right_can_interface": "aaa_right",
        }

    def test_the_input_description_is_not_mutated(self) -> None:
        base = _openarm()
        before = dict(base.hal.parameters.defaults)
        _enrich_can_buses(base, _report("arm_left_bus", "arm_right_bus"))
        assert base.hal.parameters.defaults == before


class TestAnyBusCount:
    def test_single_bus_robot(self) -> None:
        description = _openarm(
            defaults={"can_interface": "can0"},
            can_bus_bindings={"can_interface": "arm"},
        )
        out = _enrich_can_buses(description, _report("arm_bus0"))
        assert out.hal.parameters.defaults["can_interface"] == "arm_bus0"

    def test_four_bus_robot(self) -> None:
        description = _openarm(
            defaults={f"limb{n}_can_interface": "can0" for n in range(4)},
            can_bus_bindings={f"limb{n}_can_interface": f"limb{n}" for n in range(4)},
        )
        out = _enrich_can_buses(description, _report(*(f"quad_limb{n}" for n in range(4))))
        assert out.hal.parameters.defaults == {
            f"limb{n}_can_interface": f"quad_limb{n}" for n in range(4)
        }


class TestNeverGuess:
    def test_ambiguous_role_warns_and_keeps_the_default(self) -> None:
        # Two interfaces contain "left". Choosing either could drive the left
        # arm over the wrong bus, so neither is chosen.
        report = _report("openarm_left", "openarm_left_spare", "openarm_right")
        out = _enrich_can_buses(_openarm(), report)
        assert out.hal.parameters.defaults["left_can_interface"] == "openarm_left"
        warning = next(w for w in report.warnings if "'left_can_interface'" in w)
        assert "need exactly one" in warning
        # The unambiguous sibling still resolves — one bad role does not
        # discard the whole robot's discovery.
        assert out.hal.parameters.defaults["right_can_interface"] == "openarm_right"

    def test_unmatched_role_warns_and_keeps_the_default(self) -> None:
        report = _report("busA", "busB")
        out = _enrich_can_buses(_openarm(), report)
        assert _can_defaults(out) == {
            "left_can_interface": "openarm_left",
            "right_can_interface": "openarm_right",
        }
        assert any("matched nothing" in w for w in report.warnings)

    def test_bindings_declared_but_no_can_bus_found(self) -> None:
        report = _report()
        out = _enrich_can_buses(_openarm(), report)
        assert _can_defaults(out) == {
            "left_can_interface": "openarm_left",
            "right_can_interface": "openarm_right",
        }
        assert any("no CAN interface was matched" in w for w in report.warnings)


class TestRobotsThatOptOut:
    def test_robot_without_bindings_is_untouched(self) -> None:
        description = _openarm(can_bus_bindings={})
        report = _report("openarm_left", "openarm_right")
        out = _enrich_can_buses(description, report)
        assert out is description
        assert report.warnings == []

    @pytest.mark.parametrize("manifest", ["so101_follower", "ur5e", "franka_panda"])
    def test_committed_non_can_robots_declare_no_bindings(self, manifest: str) -> None:
        # Guards the opt-in: a USB or Ethernet robot must never acquire CAN
        # bindings by accident, since that would rewrite its transport config.
        path = REPO_ROOT / "robots" / manifest / "robot.yaml"
        if not path.exists():
            pytest.skip(f"no committed manifest for {manifest}")
        description = RobotDescription.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        assert description.hal.parameters.can_bus_bindings == {}
