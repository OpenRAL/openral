# SPDX-License-Identifier: Apache-2.0
"""Which UI the dashboard serves at ``/``, and the one log line that says so (ADR-0131)."""

from __future__ import annotations

import logging
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

import pytest
import structlog
from openral_observability.dashboard import ui_assets
from openral_observability.dashboard.ui_assets import PinnedAsset, ensure_ui_assets
from structlog.testing import capture_logs

_SHA = "a" * 64
_URL = "https://github.com/OpenRAL/openral/releases/download/robot-ui-1/x.tar.gz"
_Logs = list[MutableMapping[str, Any]]


@pytest.fixture
def logs() -> Iterator[_Logs]:
    """structlog's ``capture_logs`` at an INFO floor (the root conftest keeps WARNING+)."""
    saved = structlog.get_config()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
    try:
        with capture_logs() as captured:
            yield captured
    finally:
        structlog.configure(**saved)


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv(ui_assets.UI_DIR_ENV, raising=False)
    return tmp_path / "cache" / "dashboard_assets"


def _ui_source(logs: _Logs) -> _Logs:
    return [e for e in logs if e["event"] == "dashboard.ui_source"]


def test_an_empty_ui_dir_serves_the_fallback_and_never_fetches(
    cache: Path, logs: _Logs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The documented way to run without the pinned console bundle.
    monkeypatch.setattr(ui_assets, "ROBOT_UI", PinnedAsset(_URL, _SHA, 1))
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv(ui_assets.UI_DIR_ENV, str(tmp_path / "empty"))
    assert ensure_ui_assets() is None
    assert not cache.exists()  # ensure_bundle creates it before any download
    assert _ui_source(logs) == [
        {"event": "dashboard.ui_source", "source": "fallback", "log_level": "info"}
    ]


def test_a_local_ui_dir_is_logged(
    cache: Path, logs: _Logs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "index.html").write_text("<!doctype html>")
    monkeypatch.setenv(ui_assets.UI_DIR_ENV, str(tmp_path / "mine"))
    assert ensure_ui_assets() == tmp_path / "mine"
    assert not cache.exists()
    assert _ui_source(logs) == [
        {
            "event": "dashboard.ui_source",
            "source": "local",
            "path": str(tmp_path / "mine"),
            "log_level": "info",
        }
    ]


def test_the_pinned_bundle_is_logged_with_its_url(
    cache: Path, logs: _Logs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ui_assets, "ROBOT_UI", PinnedAsset(_URL, _SHA, 1))
    (cache / "robot-ui" / _SHA).mkdir(parents=True)  # already fetched: no network
    (cache / "robot-ui" / _SHA / "index.html").write_text("<!doctype html>")
    assert ensure_ui_assets() == cache / "robot-ui" / _SHA
    assert _ui_source(logs) == [
        {
            "event": "dashboard.ui_source",
            "source": "pinned",
            "url": _URL,
            "log_level": "info",
        }
    ]
