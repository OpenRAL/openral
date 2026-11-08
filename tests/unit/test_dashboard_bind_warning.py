"""Tests for the dashboard non-loopback bind security warning.

The local listener has no login. Bound to a routable address it gives every host
that reaches it the read-only view plus E-STOP (ADR-0129), so the bind must be
surfaced loudly (security audit 2026-06, finding M5).
"""

from __future__ import annotations

import pytest
from openral_observability.dashboard.server import _exposure_warning


class TestExposureWarning:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "", "LOCALHOST"])
    def test_loopback_hosts_do_not_warn(self, host: str) -> None:
        assert _exposure_warning(host) is None

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.5", "192.168.1.20"])
    def test_non_loopback_hosts_warn(self, host: str) -> None:
        msg = _exposure_warning(host)
        assert msg is not None
        assert "no login" in msg
        assert "E-STOP" in msg
        assert host in msg
