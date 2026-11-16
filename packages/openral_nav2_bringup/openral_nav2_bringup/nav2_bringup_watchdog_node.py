#!/usr/bin/env python3
"""Restart Nav2's bring-up when ``lifecycle_manager_navigation`` aborts it.

The manager brings its servers up once. When one server fails to activate it logs
``Failed to bring up all requested nodes. Aborting bringup.`` and stops: the servers it
already activated stay active, the rest stay inactive, ``bt_navigator`` rejects every goal
as "action server inactive", and nothing retries. On Spark (Isaac + pi0.5, 2026-11-16) this
happened when ``bt_navigator``'s BT never discovered the already-active ``behavior_server``
within ``wait_for_service_timeout`` (2 runs of 12; a healthy run finds it in 80 ms).

This node reads the manager's ``/rosout`` lines. On that abort it returns every managed
server to ``unconfigured`` (deactivate the active ones, clean up the inactive ones, in reverse
bring-up order) and asks the manager for ``STARTUP`` again, ``retry_delay_s`` apart, at most
``max_attempts`` times.
The manager's own ``RESET`` cannot do this: it stops at the first server whose transition
is invalid, which an aborted bring-up always leaves behind.

A dead server's bond (``CRITICAL FAILURE: SERVER ... IS DOWN``) is not handled here: the
manager resets the stack itself and, with ``attempt_respawn_reconnection``, restarts it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

ABORT_MARKER = "Failed to bring up all requested nodes"

# lifecycle_msgs/msg/Transition ids.
TRANSITION_CLEANUP = 2
TRANSITION_DEACTIVATE = 4


def recovery_plan(states: Mapping[str, str], order: Sequence[str]) -> list[tuple[str, int]]:
    """The transitions that return every managed server to ``unconfigured``.

    Reverse bring-up order, as the manager itself tears down: each ``active`` server is
    deactivated, then each server that is (or will then be) ``inactive`` is cleaned up.

    Example:
        >>> recovery_plan(
        ...     {
        ...         "planner_server": "active",
        ...         "bt_navigator": "inactive",
        ...         "docking_server": "unconfigured",
        ...     },
        ...     ["planner_server", "bt_navigator", "docking"],
        ... )
        [('planner_server', 4), ('bt_navigator', 2), ('planner_server', 2)]
    """
    reverse = list(reversed(order))
    plan = [(n, TRANSITION_DEACTIVATE) for n in reverse if states.get(n) == "active"]
    plan += [(n, TRANSITION_CLEANUP) for n in reverse if states.get(n) in ("active", "inactive")]
    return plan


def main() -> None:
    """Spin the watchdog until shutdown."""
    import rclpy
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState, GetState
    from nav2_msgs.srv import ManageLifecycleNodes
    from rcl_interfaces.msg import Log
    from rcl_interfaces.srv import GetParameters
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    class Nav2BringupWatchdog(Node):  # type: ignore[misc]  # reason: rclpy.node.Node is untyped
        def __init__(self) -> None:
            super().__init__("openral_nav2_bringup_watchdog")
            self.declare_parameter("manager", "lifecycle_manager_navigation")
            self.declare_parameter("max_attempts", 3)
            # Between restarts: a server whose discovery stalled gets time to show up.
            self.declare_parameter("retry_delay_s", 5.0)
            self.declare_parameter("service_timeout_s", 30.0)
            # STARTUP returns when the manager's whole bring-up does, BT server waits included.
            self.declare_parameter("startup_timeout_s", 180.0)
            self._manager = str(self.get_parameter("manager").value)
            self._max_attempts = int(self.get_parameter("max_attempts").value)
            self._retry_delay_s = float(self.get_parameter("retry_delay_s").value)
            self._timeout_s = float(self.get_parameter("service_timeout_s").value)
            self._startup_timeout_s = float(self.get_parameter("startup_timeout_s").value)
            self._started = threading.Event()
            # /rosout's own QoS: a line logged before this node subscribed is still delivered.
            qos = QoSProfile(
                depth=1000,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(Log, "/rosout", self._on_log, qos)

        def _on_log(self, msg: Log) -> None:
            # Only the first abort starts the recovery; the aborts its own STARTUPs cause
            # are handled by its retry loop.
            if msg.name != self._manager or ABORT_MARKER not in msg.msg or self._started.is_set():
                return
            self._started.set()
            threading.Thread(target=self._recover, daemon=True).start()

        def _call(
            self, srv_type: type, name: str, request: object, timeout_s: float | None = None
        ) -> Any:  # reason: the rosidl response type of srv_type, or None on timeout
            timeout_s = self._timeout_s if timeout_s is None else timeout_s
            client = self.create_client(srv_type, name)
            try:
                if not client.wait_for_service(timeout_sec=self._timeout_s):
                    return None
                future = client.call_async(request)
                deadline = time.monotonic() + timeout_s
                while not future.done() and time.monotonic() < deadline:
                    time.sleep(0.05)
                return future.result() if future.done() else None
            finally:
                self.destroy_client(client)

        def _recover(self) -> None:
            log = self.get_logger()
            for attempt in range(1, self._max_attempts + 1):
                log.warning(
                    f"nav2_bringup_watchdog: {self._manager} aborted Nav2 bring-up; "
                    f"restart {attempt}/{self._max_attempts} in {self._retry_delay_s:g} s"
                )
                time.sleep(self._retry_delay_s)
                if self._restart():
                    log.info(f"nav2_bringup_watchdog: Nav2 active after restart {attempt}")
                    return
            log.error(
                f"nav2_bringup_watchdog: Nav2 bring-up still failing after "
                f"{self._max_attempts} restart(s); giving up. Nav2 stays inactive and "
                "rejects every goal."
            )

        def _restart(self) -> bool:
            """Return every managed server to unconfigured, then ask for STARTUP."""
            log = self.get_logger()
            req = GetParameters.Request()
            req.names = ["node_names"]
            resp = self._call(GetParameters, f"/{self._manager}/get_parameters", req)
            names = list(resp.values[0].string_array_value) if resp and resp.values else []
            if not names:
                log.error("nav2_bringup_watchdog: could not read the manager's node_names")
                return False
            states: dict[str, str] = {}
            for name in names:
                state = self._call(GetState, f"/{name}/get_state", GetState.Request())
                states[name] = state.current_state.label if state is not None else "unknown"
            log.info(f"nav2_bringup_watchdog: server states {states}")
            for name, transition_id in recovery_plan(states, names):
                req = ChangeState.Request()
                req.transition = Transition(id=transition_id)
                done = self._call(ChangeState, f"/{name}/change_state", req)
                if done is None or not done.success:
                    log.error(f"nav2_bringup_watchdog: transition {transition_id} of {name} failed")
                    return False
            req = ManageLifecycleNodes.Request()
            req.command = ManageLifecycleNodes.Request.STARTUP
            done = self._call(
                ManageLifecycleNodes,
                f"/{self._manager}/manage_nodes",
                req,
                timeout_s=self._startup_timeout_s,
            )
            return bool(done is not None and done.success)

    rclpy.init()
    node = Nav2BringupWatchdog()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
