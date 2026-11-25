"""The clock refusal on a real robot with peers (openral#765, HZ-0137-11)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

import pytest
from openral_cli._clock_sync import chrony_tracking, require_synced_clock
from openral_core.exceptions import ROSConfigError

_CSV = (
    "A9FEA97B,169.254.169.123,4,1764000000.0,-0.000012345,0.0,0.0,0.0,0.0,0.0,0.0,0.0,64.0,Normal\n"
)


def _run(
    stdout: str = "", code: int = 0, raises: BaseException | None = None
) -> Callable[..., subprocess.CompletedProcess[str]]:
    # Process-boundary double for subprocess.run (CLAUDE.md 1.11).
    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(cmd, code, stdout, "")

    return run


def test_tracking_parses_offset_and_leap() -> None:
    assert chrony_tracking(_run(_CSV)) == (-1.2345e-05, "Normal")


@pytest.mark.parametrize(
    "run",
    [
        _run(raises=FileNotFoundError()),
        _run(code=1),
        _run(raises=subprocess.TimeoutExpired("chronyc", 5)),
    ],
)
def test_tracking_none_when_unavailable(
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    assert chrony_tracking(run) is None


def test_synced_passes() -> None:
    require_synced_clock((0.0, "Normal"))


@pytest.mark.parametrize(
    ("tracking", "match"),
    [((0.011, "Normal"), "11"), ((0.0, "Not synchronised"), "ynchronised"), (None, "chrony")],
)
def test_unsynced_refuses(tracking: tuple[float, str] | None, match: str) -> None:
    with pytest.raises(ROSConfigError, match=match):
        require_synced_clock(tracking)
