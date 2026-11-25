"""Host-clock check for a real robot with peers (openral#765, HZ-0137-11).

Peer data from other hosts is checked against a 300 ms freshness budget, so the host clock
must be disciplined by chrony.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from openral_core.exceptions import ROSConfigError

CLOCK_LIMIT_S = 0.010
_TIMEOUT_S = 5


def chrony_tracking(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[float, str] | None:
    """``(system time offset in s, leap status)`` from ``chronyc -c tracking``.

    ``None`` if chronyc is missing, hangs, fails or prints something unparsable.
    """
    try:
        done = run(
            ["chronyc", "-c", "tracking"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT_S,
        )
        if done.returncode != 0:
            return None
        fields = done.stdout.strip().split(",")
        return float(fields[4]), fields[-1]
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def require_synced_clock(
    tracking: tuple[float, str] | None, limit_s: float = CLOCK_LIMIT_S
) -> None:
    """Raise ``ROSConfigError`` unless chrony reports a normal, in-limit clock."""
    if tracking is None:
        raise ROSConfigError(
            "this robot has peers on other hosts, so its clock must be synchronised: "
            "install and start chrony (`chronyc tracking` failed)."
        )
    offset, leap = tracking
    if leap != "Normal":
        raise ROSConfigError(f"the host clock is not synchronised by chrony (leap status: {leap}).")
    if abs(offset) > limit_s:
        raise ROSConfigError(
            f"the host clock is {offset * 1000:.1f} ms off chrony's reference; "
            f"peers need within {limit_s * 1000:.0f} ms."
        )
