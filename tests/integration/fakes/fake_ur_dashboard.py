"""Process-boundary fake of `ur_robot_driver`'s dashboard client (`std_srvs/Trigger` services).

The real `/dashboard_client/stop` lives in the vendor's driver process and talks to the teach
pendant over the dashboard TCP protocol — a network boundary CLAUDE.md §1.11 allows a double
at. This serves the same service name with the same type so the production
`RosControlTransport.call_trigger` path is exercised end to end; it records every call and
flips a `program_running` flag exactly as the real dashboard would after `stop`.
"""

from __future__ import annotations

from typing import Any

__all__ = ["FakeURDashboard"]


class FakeURDashboard:
    """Serve `/dashboard_client/stop` (`std_srvs/Trigger`) on a live rclpy node."""

    def __init__(self, node: Any, *, refuse: bool = False) -> None:
        from std_srvs.srv import Trigger

        self.stop_calls = 0
        self.program_running = True
        self._refuse = refuse
        self._service = node.create_service(Trigger, "/dashboard_client/stop", self._on_stop)

    def _on_stop(self, _request: Any, response: Any) -> Any:
        self.stop_calls += 1
        if self._refuse:
            response.success = False
            response.message = "Dashboard server not connected"
            return response
        self.program_running = False
        response.success = True
        response.message = "Stopped program"
        return response
