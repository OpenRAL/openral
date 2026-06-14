"""ROS 2 ``diagnostic_msgs/DiagnosticArray`` heartbeat helper.

ADR-0018 §F8 mandates a uniform 1 Hz ``DiagnosticArray`` publication from
every lifecycle node in the OpenRAL graph. Centralising the publisher
here keeps the cadence, ``hardware_id`` shape, and level-mapping
identical across `openral_world_state`, `openral_hal_*`,
`openral_safety`, `openral_rskill_ros`, and any future node.

The helper imports ``rclpy`` and ``diagnostic_msgs`` lazily so this
module stays import-safe on pure-Python hosts (CI, tests that do not
build the colcon workspace). Consumers that do not call
:meth:`DiagnosticsHeartbeat.start` pay zero ROS cost.

Per ADR-0018 §F8 the diagnostics topic answers *"what is the system
state right now"*; the ``/openral/failure/*`` bus (F3, next PR) answers
*"what just happened"*. Sustained ``ERROR`` keeps the level latched on
this topic; the matching ``FailureTrigger`` fires once on the
transition. No duplication.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rclpy.lifecycle import LifecycleNode

__all__ = ["DiagnosticsHeartbeat", "Level"]


class Level:
    """Mirror of ``diagnostic_msgs/DiagnosticStatus`` level constants.

    Re-exported here so ``status_fn`` callbacks can return the right
    integer without importing ``diagnostic_msgs`` (which fails on
    pure-Python hosts).
    """

    OK = 0
    WARN = 1
    ERROR = 2
    STALE = 3


StatusFn = Callable[[], "tuple[int, str, dict[str, str]]"]


class DiagnosticsHeartbeat:
    """1 Hz ``DiagnosticArray`` publisher driven by a per-node status callback.

    Lifecycle:

    * Construct in ``__init__`` — no ROS work yet.
    * Call :meth:`create_publisher` in ``on_configure`` — opens the publisher.
    * Call :meth:`start` in ``on_activate`` — starts the timer.
    * Call :meth:`stop` in ``on_deactivate`` — cancels the timer.
    * Call :meth:`destroy` in ``on_cleanup`` — destroys the publisher.

    Args:
        node: The owning ``rclpy.lifecycle.LifecycleNode``. The helper
            uses ``node.create_publisher`` / ``node.create_timer`` /
            ``node.get_clock``; nothing else.
        hardware_id: ADR-0018 §F8 disambiguator — the ``hardware_id``
            field on every ``DiagnosticStatus``. Convention:
            ``"<component>:<instance>"`` e.g.
            ``"openral_skill_runner:so100"``.
        component_name: Short component label (``"skill_runner"``,
            ``"safety"``). Recorded as ``DiagnosticStatus.name``.
        status_fn: Zero-arg callable returning
            ``(level, message, key_values)`` where ``level`` is one of
            :class:`Level`, ``message`` is a short human-readable
            summary, and ``key_values`` is a flat ``dict[str, str]`` of
            extra metadata.
        rate_hz: Publish rate. Defaults to 1.0 Hz per ADR-0018 §F8;
            override only with safety-working-group sign-off.

    Example:
        >>> # Real usage exercised in
        >>> # tests/unit/test_diagnostics_heartbeat.py against a real
        >>> # LifecycleNode and real DiagnosticArray subscriber.
        >>> pass
    """

    def __init__(
        self,
        node: LifecycleNode,
        *,
        hardware_id: str,
        component_name: str,
        status_fn: StatusFn,
        rate_hz: float = 1.0,
    ) -> None:
        """Store references; opens no ROS resources."""
        if not hardware_id:
            raise ValueError("hardware_id must be a non-empty string")
        if not component_name:
            raise ValueError("component_name must be a non-empty string")
        if rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be positive (got {rate_hz!r})")
        self._node = node
        self._hardware_id = hardware_id
        self._component_name = component_name
        self._status_fn = status_fn
        self._rate_hz = rate_hz
        # ``Any`` here because ``rclpy.Publisher`` / ``rclpy.Timer`` carry no
        # py.typed marker — mypy --strict otherwise rejects the duck-typed
        # ``.publish()`` / ``.cancel()`` calls below.
        self._publisher: Any = None
        self._timer: Any = None

    @property
    def hardware_id(self) -> str:
        """Return the configured ``hardware_id`` field."""
        return self._hardware_id

    @property
    def component_name(self) -> str:
        """Return the configured ``DiagnosticStatus.name`` value."""
        return self._component_name

    def create_publisher(self) -> None:
        """Open the ``/diagnostics`` publisher. Call from ``on_configure``.

        Idempotent: re-calling without a prior :meth:`destroy` is a
        no-op so the lifecycle ``cleanup → configure`` round trip stays
        safe.
        """
        if self._publisher is not None:
            return
        from diagnostic_msgs.msg import DiagnosticArray

        self._publisher = self._node.create_publisher(
            DiagnosticArray,
            "/diagnostics",
            10,
        )

    def start(self) -> None:
        """Start the 1 Hz publish timer. Call from ``on_activate``.

        Requires :meth:`create_publisher` to have been called first;
        otherwise raises :class:`RuntimeError` (a missed configure step
        is a programmer error, not a runtime degraded mode).
        """
        if self._publisher is None:
            raise RuntimeError(
                "DiagnosticsHeartbeat.start() called before create_publisher(); "
                "did on_configure run?"
            )
        if self._timer is not None:
            return
        period_s = 1.0 / self._rate_hz
        self._timer = self._node.create_timer(period_s, self._publish)

    def stop(self) -> None:
        """Cancel the publish timer. Call from ``on_deactivate``.

        Idempotent: cancels-then-clears, safe to call when never
        started.
        """
        if self._timer is None:
            return
        self._timer.cancel()
        self._node.destroy_timer(self._timer)
        self._timer = None

    def destroy(self) -> None:
        """Destroy the publisher. Call from ``on_cleanup`` / ``on_shutdown``.

        Idempotent: also calls :meth:`stop` so a single ``destroy()``
        from ``on_shutdown`` handles the unsorted-state case.
        """
        self.stop()
        if self._publisher is None:
            return
        self._node.destroy_publisher(self._publisher)
        self._publisher = None

    def publish_once(self) -> None:
        """Publish one ``DiagnosticArray`` immediately.

        Useful for tests that need a deterministic publication without
        waiting for the timer. The timer also calls this internally.
        """
        self._publish()

    # ── internals ──────────────────────────────────────────────────────────

    def _publish(self) -> None:
        """Timer callback: build one DiagnosticArray from ``status_fn``."""
        if self._publisher is None:
            return
        # No type-ignore needed here: the first import in ``create_publisher``
        # already taught mypy that ``diagnostic_msgs.msg`` is unresolvable,
        # so subsequent imports in the same module inherit that state.
        from diagnostic_msgs.msg import (
            DiagnosticArray,
            DiagnosticStatus,
            KeyValue,
        )

        try:
            level, message, key_values = self._status_fn()
        except Exception as exc:  # reason: never crash the timer on a status bug
            level = Level.ERROR
            message = f"status_fn raised {type(exc).__name__}: {exc}"
            key_values = {"exception_type": type(exc).__name__}

        status = DiagnosticStatus()
        status.level = bytes([int(level) & 0xFF])
        status.name = self._component_name
        status.message = message
        status.hardware_id = self._hardware_id
        status.values = [KeyValue(key=str(k), value=str(v)) for k, v in key_values.items()]

        array = DiagnosticArray()
        array.header.stamp = self._node.get_clock().now().to_msg()
        array.status = [status]
        self._publisher.publish(array)
