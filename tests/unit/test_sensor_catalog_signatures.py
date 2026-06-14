"""Tests for the catalog-signature reverse-lookup added in commit 5.

Per the auto-provisioning plan, ``openral detect`` resolves a probed device
(e.g. a RealSense ``model_id="D435I"``) into the canonical catalog entry
``intel/realsense_d435i`` so it can call ``CATALOG.build(...)`` to pull
the *real* intrinsics, FOV, encoding and rate — never invented from
v4l2 introspection.

These tests use the real global :data:`CATALOG` populated by every
vendor module on import; per CLAUDE.md §1.11 there are no mocks.
"""

from __future__ import annotations

import pytest
from openral_core.schemas import SensorBundle, SensorSpec
from openral_sensors import CATALOG, SensorSignature


class TestRealsenseSignatures:
    @pytest.mark.parametrize(
        "model_id, expected_id",
        [
            ("D435", "intel/realsense_d435"),
            ("D435I", "intel/realsense_d435i"),
            ("D415", "intel/realsense_d415"),
        ],
    )
    def test_realsense_model_id_resolves_to_canonical_entry(
        self, model_id: str, expected_id: str
    ) -> None:
        sig = SensorSignature(kind="realsense", value=model_id)
        entry = CATALOG.find_by_signature(sig)
        assert entry is not None
        assert entry.id == expected_id

    def test_realsense_unknown_model_returns_none(self) -> None:
        sig = SensorSignature(kind="realsense", value="D9999")
        assert CATALOG.find_by_signature(sig) is None

    def test_resolved_entry_materializes_real_intrinsics(self) -> None:
        # Detect → catalog → CATALOG.build → SensorSpec/Bundle with real values.
        entry = CATALOG.find_by_signature(SensorSignature(kind="realsense", value="D435I"))
        assert entry is not None
        bundle = CATALOG.build(entry.id, name="head", parent_frame="base_link")
        assert isinstance(bundle, SensorBundle)
        # Real intrinsics, not placeholder.  D435 RGB stream has fx/fy ~ 600 px @ 640x480.
        # SensorSpec uses use_enum_values=True so .modality is the str "rgb".
        rgb = next(s for s in bundle.sensors if s.modality == "rgb")
        assert rgb.intrinsics is not None
        assert rgb.intrinsics.fx > 100.0
        assert rgb.intrinsics.fy > 100.0
        assert rgb.rate_hz > 0.0


class TestUsbUvcSignatures:
    @pytest.mark.parametrize(
        "vid_pid, expected_id",
        [
            ("0x046d:0x082d", "logitech/c920"),  # Logitech C920
            ("0x046d:0x0892", "logitech/c920"),  # C920e variant
        ],
    )
    def test_usb_vid_pid_resolves(self, vid_pid: str, expected_id: str) -> None:
        entry = CATALOG.find_by_signature(SensorSignature(kind="usb_uvc", value=vid_pid))
        assert entry is not None
        assert entry.id == expected_id

    def test_v4l2_name_resolves(self) -> None:
        entry = CATALOG.find_by_signature(SensorSignature(kind="v4l2_name", value="C920"))
        assert entry is not None
        assert entry.id == "logitech/c920"

    def test_resolved_uvc_entry_materializes_real_spec(self) -> None:
        entry = CATALOG.find_by_signature(SensorSignature(kind="v4l2_name", value="C920"))
        assert entry is not None
        spec = CATALOG.build(entry.id, name="scene", parent_frame="base_link")
        assert isinstance(spec, SensorSpec)
        assert spec.vendor == "Logitech"
        assert spec.intrinsics is not None
        assert spec.intrinsics.width > 0
        assert spec.intrinsics.height > 0


class TestSignatureUniqueness:
    """Each signature must point to at most one catalog entry."""

    def test_no_two_entries_share_a_signature(self) -> None:
        seen: dict[SensorSignature, str] = {}
        for entry in CATALOG.entries():
            for sig in entry.signatures:
                assert sig not in seen, (
                    f"Signature {sig!r} registered by both {seen[sig]!r} and {entry.id!r}"
                )
                seen[sig] = entry.id


class TestEmptySignaturesDefault:
    """Vendor entries that have not been signature-annotated keep an empty
    tuple — the field is opt-in, not a breaking change."""

    def test_unannotated_entries_have_empty_signatures(self) -> None:
        # Robotiq FT 300-S is registered without any signature — live probes
        # for serial F/T sensors are deferred.
        entry = CATALOG.get("robotiq/ft_300s")
        assert entry.signatures == ()
