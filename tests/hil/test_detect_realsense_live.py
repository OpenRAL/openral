"""HIL test: real Intel RealSense connected to the host.

Skipped automatically when:
- ``pyrealsense2`` is not installed.
- No RealSense devices are connected.

Verifies the full identify-then-enrich pipeline: probe_realsense_devices
returns a typed record with a recognizable ``model_id`` that resolves to
the canonical catalog entry, and `CATALOG.build` produces a
fully-populated ``SensorBundle`` with real intrinsics.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "pyrealsense2",
    reason="pyrealsense2 not installed (Linux x86 + librealsense required)",
)


@pytest.fixture
def has_device() -> None:
    import pyrealsense2 as rs

    devices = list(rs.context().devices)
    if not devices:
        pytest.skip("No Intel RealSense device connected to the host.")


def test_realsense_probe_resolves_to_catalog_entry(has_device: None) -> None:  # pragma: no cover
    """The detected model_id must reverse-look-up via SensorSignature."""
    from openral_detect.probes import probe_realsense_devices
    from openral_detect.registry import signature_for_realsense
    from openral_sensors import CATALOG

    devices = probe_realsense_devices(warnings=[])
    assert devices, "expected at least one connected RealSense device"
    dev = devices[0]
    sig = signature_for_realsense(dev.model_id)
    entry = CATALOG.find_by_signature(sig)
    assert entry is not None, (
        f"RealSense model_id {dev.model_id!r} has no catalog entry; add a "
        "SensorSignature for it in python/sensors/src/openral_sensors/realsense.py."
    )
    bundle = CATALOG.build(entry.id, name="realsense_test", parent_frame="base_link")
    rgb = next(s for s in bundle.sensors if s.modality == "rgb")
    assert rgb.intrinsics is not None
    assert rgb.intrinsics.fx > 0
    assert rgb.intrinsics.width > 0
