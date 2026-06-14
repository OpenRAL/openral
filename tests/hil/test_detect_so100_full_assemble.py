"""HIL test: SO-100 + full openral detect → assemble → check pipeline.

Skipped when no Feetech / SO-100 USB controller is detected on the
host.  Exercises the user-facing flow end-to-end without sims.
"""

from __future__ import annotations

import pytest


def _has_so100_usb() -> bool:
    try:
        from openral_cli.autodetect import (
            enumerate_usb_devices,
            match_known_devices,
        )
    except ImportError:
        return False
    matches = match_known_devices(enumerate_usb_devices())
    return any(m.known.bh_robot_type == "so100" for m in matches)


@pytest.mark.skipif(not _has_so100_usb(), reason="No SO-100 USB controller detected")
def test_so100_detect_assemble_check_full_pipeline() -> None:  # pragma: no cover
    from openral_detect import (
        assemble_robot_description,
        check_installed_rskills,
        detect_hardware,
    )

    detection = detect_hardware(dds_timeout_s=0.0)
    assert any(m.bh_robot_type == "so100" for m in detection.usb.matches)

    description = assemble_robot_description(detection)
    assert description.name == "so100_follower"
    assert "so100_follower" in description.capabilities.embodiment_tags

    # Empty registry → empty rows, exit code 0.  This is just a smoke test;
    # the integration with installed skills is exercised via `ral skill check`.
    report = check_installed_rskills(description)
    assert isinstance(report.rows, list)
