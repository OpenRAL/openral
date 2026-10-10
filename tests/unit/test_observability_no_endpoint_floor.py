"""Without an OTLP endpoint, ``configure_observability`` re-applies the log floor every call.

The no-endpoint early return used to skip the floor on a repeated call for the
same service, so a changed ``OPENRAL_LOG_LEVEL`` left the previous filtering
wrapper installed.
"""

from __future__ import annotations

import pytest
import structlog
from openral_observability import _sdk


def _wrapper_name() -> str:
    return str(structlog.get_config()["wrapper_class"].__name__)


def test_a_changed_log_level_takes_effect_on_the_repeated_no_endpoint_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_sdk._ENV_ENDPOINT, raising=False)
    monkeypatch.setattr(_sdk, "_service_name", None)
    monkeypatch.setattr(_sdk, "_endpoint", None)

    monkeypatch.setenv("OPENRAL_LOG_LEVEL", "WARNING")
    assert _sdk.configure_observability(service_name="openral.test-floor") is False
    at_warning = _wrapper_name()

    monkeypatch.setenv("OPENRAL_LOG_LEVEL", "DEBUG")
    assert _sdk.configure_observability(service_name="openral.test-floor") is False
    at_debug = _wrapper_name()

    assert at_warning != at_debug, "the second no-endpoint call kept the old floor"
    assert "Warning" in at_warning and "Debug" in at_debug
