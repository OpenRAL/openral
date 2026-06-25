"""HIL test: SO-101 + full openral detect → assemble → check pipeline.

Skipped when no Feetech / SO-101 USB controller is detected on the
host.  Exercises the user-facing flow end-to-end without sims.

The SO-101 is electrically identical to the SO-100 over USB (same Feetech
controller / VID/PID), so a bare plug-in resolves to ``so101_follower`` by
default; an SO-100 is selected explicitly with ``openral detect --robot so100``.
"""

from __future__ import annotations

import pytest


def _has_so101_usb() -> bool:
    try:
        from openral_cli.autodetect import (
            enumerate_usb_devices,
            match_known_devices,
        )
    except ImportError:
        return False
    matches = match_known_devices(enumerate_usb_devices())
    return any(m.known.bh_robot_type == "so101" for m in matches)


@pytest.mark.skipif(not _has_so101_usb(), reason="No SO-101 USB controller detected")
def test_so101_detect_assemble_check_full_pipeline() -> None:  # pragma: no cover
    from openral_detect import (
        assemble_robot_description,
        check_installed_rskills,
        detect_hardware,
    )

    detection = detect_hardware(dds_timeout_s=0.0)
    assert any(m.bh_robot_type == "so101" for m in detection.usb.matches)

    description = assemble_robot_description(detection)
    assert description.name == "so101_follower"
    assert "so101_follower" in description.capabilities.embodiment_tags

    # Empty registry → empty rows, exit code 0.  This is just a smoke test;
    # the integration with installed skills is exercised via `ral skill check`.
    report = check_installed_rskills(description)
    assert isinstance(report.rows, list)
