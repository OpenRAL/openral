"""Process-boundary fake of `ur_robot_driver`'s dashboard client (`std_srvs/Trigger` services).

The real `/dashboard_client/stop` lives in the vendor's driver process and talks to the teach
pendant over the dashboard TCP protocol — a network boundary CLAUDE.md §1.11 allows a double
at. This serves the same service name with the same type so the production
`RosControlTransport.call_trigger` path is exercised end to end; it records every call and
flips a `program_running` flag exactly as the real dashboard would after `stop`.

It spins on its **own** node and executor thread, as a separate process would. The e-stop
callback under test blocks the lifecycle node's single-threaded executor while it waits for
this Trigger, so a service hosted on that executor could never answer — the first run of the
live test found exactly that deadlock.
"""

from __future__ import annotations

import threading
from typing import Any

__all__ = ["FakeURDashboard"]


class FakeURDashboard:
    """Serve `/dashboard_client/stop` (`std_srvs/Trigger`) on a private, self-spinning node."""

    def __init__(self, *, refuse: bool = False, name: str = "fake_ur_dashboard") -> None:
        """Create the private node, the ``stop`` service and the spinning thread."""
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from std_srvs.srv import Trigger

        self.stop_calls = 0
        self.program_running = True
        self._refuse = refuse
        self._node = rclpy.create_node(name)
        self._service = self._node.create_service(Trigger, "/dashboard_client/stop", self._on_stop)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        """Executor loop for the private node until ``close``."""
        while not self._stop.is_set():
            self._executor.spin_once(timeout_sec=0.05)

    def _on_stop(self, _request: Any, response: Any) -> Any:
        """Answer ``stop``: refuse when configured to, else stop the (simulated) program."""
        self.stop_calls += 1
        if self._refuse:
            response.success = False
            response.message = "Dashboard server not connected"
            return response
        self.program_running = False
        response.success = True
        response.message = "Stopped program"
        return response

    def close(self) -> None:
        """Stop spinning and destroy the node."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._executor.remove_node(self._node)
        self._node.destroy_node()
