"""Host-clock check for a real robot with peers (openral#765, HZ-0137-11).

Peer data from other hosts is checked against a 300 ms freshness budget, so the host clock
must be disciplined by chrony. ``openral deploy run`` and ``deploy_e2e.launch.py`` under
``hal_mode:=real`` both run it, so a direct ``ros2 launch`` is checked too; it lives in
``openral_core`` because the launch file cannot import ``openral_cli``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import NamedTuple

from openral_core.exceptions import ROSConfigError

__all__ = ["CLOCK_LIMIT_S", "ChronyTracking", "chrony_tracking", "require_synced_clock"]

#: The largest worst-case clock error a real robot with peers accepts, in seconds.
#: Provisional until the Safety-WG sign-off (openral-pro#106 step 5).
CLOCK_LIMIT_S = 0.010
_TIMEOUT_S = 5


class ChronyTracking(NamedTuple):
    """The ``chronyc -c tracking`` fields the check reads (all times in seconds)."""

    offset_s: float
    root_delay_s: float
    root_dispersion_s: float
    leap: str

    @property
    def worst_case_s(self) -> float:
        """Chrony's bound on the error against true time: ``|offset| + dispersion + delay/2``.

        Example:
            >>> round(ChronyTracking(-0.001, 0.004, 0.002, "Normal").worst_case_s, 6)
            0.005
        """
        return abs(self.offset_s) + self.root_dispersion_s + self.root_delay_s / 2


def chrony_tracking(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ChronyTracking | None:
    """Offset, root delay, root dispersion and leap status from ``chronyc -c tracking``.

    The CSV's fields 4, 10, 11 and the last. ``None`` if chronyc is missing, hangs, fails or
    prints something unparsable.
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
        return ChronyTracking(float(fields[4]), float(fields[10]), float(fields[11]), fields[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def require_synced_clock(tracking: ChronyTracking | None, limit_s: float = CLOCK_LIMIT_S) -> None:
    """Raise ``ROSConfigError`` unless chrony reports a normal clock within ``limit_s``.

    Within means its worst-case error (``ChronyTracking.worst_case_s``) is.

    Example:
        >>> require_synced_clock(ChronyTracking(0.001, 0.002, 0.001, "Normal"))
        >>> require_synced_clock(ChronyTracking(0.001, 0.02, 0.001, "Normal"))
        Traceback (most recent call last):
        openral_core.exceptions.ROSConfigError: the host clock may be 12.0 ms off true time (offset 1.0 ms + root dispersion 1.0 ms + half the root delay 10.0 ms); peers need within 10 ms.
    """  # noqa: E501
    if tracking is None:
        raise ROSConfigError(
            "this robot has peers on other hosts, so its clock must be synchronised: "
            "install and start chrony (`chronyc tracking` failed)."
        )
    if tracking.leap != "Normal":
        raise ROSConfigError(
            f"the host clock is not synchronised by chrony (leap status: {tracking.leap})."
        )
    worst = tracking.worst_case_s
    if not worst <= limit_s:  # fails closed on nan; inf is > limit
        raise ROSConfigError(
            f"the host clock may be {worst * 1000:.1f} ms off true time "
            f"(offset {abs(tracking.offset_s) * 1000:.1f} ms + root dispersion "
            f"{tracking.root_dispersion_s * 1000:.1f} ms + half the root delay "
            f"{tracking.root_delay_s * 500:.1f} ms); peers need within {limit_s * 1000:.0f} ms."
        )
