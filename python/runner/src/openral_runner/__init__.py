"""openral inference runner — hardware-side counterpart to ``openral_sim``.

This package hosts the :class:`InferenceRunner` Protocol +
:class:`InferenceRunnerBase` shared between sim and hardware paths, plus
the :class:`SensorReader` Protocol and the per-backend sensor readers
(``openral_runner.backends``).

Public surface today:

- ``InferenceRunner``: structural Protocol every runner satisfies
  (``activate / tick / run / deactivate``; ``rate_hz``).
- ``InferenceRunnerBase``: shared abstract base with the rate-limited
  ``run()`` loop, ``rskill.tick`` OTel parent span, ``RunResult``
  aggregation, and deadline-overrun policy. Subclasses implement
  ``_tick_impl``.
- ``SensorReader``: structural Protocol every sensor backend satisfies
  (``open / close / read_latest``). Concrete backends live under
  ``openral_runner.backends``; ``OpenCVThreadSensorReader`` is the
  default.
- ``SafetyClient`` / ``NullSafetyClient``: pre-action safety seam called
  by the runner before HAL dispatch. ``NullSafetyClient`` is a no-op
  stub awaiting the real C++ safety kernel (CLAUDE.md §6 Layer 6).
- ``DeployRunner``: concrete :class:`InferenceRunnerBase` subclass
  that composes a real :class:`HAL`, :class:`Skill`,
  :class:`WorldStateAggregator`, a list of :class:`SensorReader`s, and a
  :class:`SafetyClient`. First end-to-end loop on real hardware
  (or a digital twin like :class:`SO100DigitalTwin`).
- ``precise_sleep`` / ``sleep_until``: cadence helpers (mirrors lerobot's
  ``precise_sleep`` shape).

Imports are PEP 562 lazy: ``import openral_runner`` no longer
eagerly drags in torch (via ``base``). Symbols are still available
through attribute access — ``openral_runner.InferenceRunnerBase``
works exactly as before — but they are only resolved on first use.
This is load-bearing: importing the gstreamer subpackage (or any
subpackage that does not need torch) used to triple-load 582 torch
modules at import, and the resulting ``glib`` state conflicts with
``rclpy.init()`` / ``Node()`` inside the x86-ros Docker image
(observed segfaults in the ROS-tee smoke test, PR I/8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Light imports — no torch, no gi, no rclpy in any of these.
from openral_runner.clock import precise_sleep, sleep_until
from openral_runner.protocol import InferenceRunner
from openral_runner.sensor_reader import SensorReader

if TYPE_CHECKING:
    # Type checkers see the symbols at their true module locations.
    from openral_runner.base import InferenceRunnerBase
    from openral_runner.deploy_runner import DeployRunner
    from openral_runner.factory import (
        SENSOR_BACKEND_REGISTRY,
        SKILL_REGISTRY,
    )
    from openral_runner.ros_publishing_hal import ROSPublishingHAL
    from openral_runner.safety import NullSafetyClient, SafetyClient

__all__ = [
    "SENSOR_BACKEND_REGISTRY",
    "SKILL_REGISTRY",
    "DeployRunner",
    "InferenceRunner",
    "InferenceRunnerBase",
    "NullSafetyClient",
    "ROSPublishingHAL",
    "SafetyClient",
    "SensorReader",
    "precise_sleep",
    "sleep_until",
]
__version__ = "0.1.0"


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "InferenceRunnerBase": ("openral_runner.base", "InferenceRunnerBase"),
    "SENSOR_BACKEND_REGISTRY": ("openral_runner.factory", "SENSOR_BACKEND_REGISTRY"),
    "SKILL_REGISTRY": ("openral_runner.factory", "SKILL_REGISTRY"),
    "DeployRunner": ("openral_runner.deploy_runner", "DeployRunner"),
    "NullSafetyClient": ("openral_runner.safety", "NullSafetyClient"),
    "ROSPublishingHAL": (
        "openral_runner.ros_publishing_hal",
        "ROSPublishingHAL",
    ),
    "SafetyClient": ("openral_runner.safety", "SafetyClient"),
}


def __getattr__(name: str) -> Any:  # noqa: ANN401  # reason: PEP 562 attribute hook
    """Resolve heavy symbols on first access (torch / glib-sensitive deferral)."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    import importlib  # noqa: PLC0415

    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
