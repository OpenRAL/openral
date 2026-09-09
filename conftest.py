"""Repo-root pytest conftest — runs for every pytest invocation under this repo.

Discovered by pytest because ``pyproject.toml`` lives in this directory, so
this conftest is loaded for both the standard ``pytest tests/...`` runs and
for the ``pytest --doctest-modules python/...`` subprocess that
:mod:`tests.unit.test_doctest_runner` spawns.

Responsibilities
----------------
- **Silence structlog logging during tests.**  Several public modules emit
  ``log.info(...)`` from constructors / ``connect()`` / ``disconnect()``
  (e.g. :class:`openral_world_state.WorldStateAggregator`,
  :class:`openral_hal.RosControlHAL`,
  :class:`openral_hal.SO100FollowerHAL`).  structlog's default
  ``PrintLoggerFactory`` writes those records to *stdout*, which breaks
  doctest matching for any docstring example that constructs one of these
  objects.  Production deployments configure structlog with an OTel handler
  per CLAUDE.md §5.1; tests deserve the same explicit configuration so that
  test stdout reflects only the test's own assertions.

This conftest is intentionally narrow — Rich console neutering still lives
in ``tests/conftest.py`` because it only matters when running tests under
``tests/``.
"""

from __future__ import annotations

import logging
import pathlib
import sys
from collections.abc import Iterator

import pytest
import structlog

# Make ROS-package-bundled pure-Python modules importable when running
# pytest from the repo root (i.e. outside of a colcon/ament environment).
# Each entry mirrors ``packages/<name>/<name>/`` — the ament_python_install
# tree the ROS build copies to ``install/<name>/lib/python.../<name>``.
# We only add packages that ship pure-Python code we want to unit-test
# without a sourced ROS install; rclpy-gated tests still skip when ROS
# isn't available (CLAUDE.md §1.11).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent
for _pkg in (
    "openral_safety",
    "openral_safety_watchdog",
    "openral_human_estop",
    # pure-Python image_convert (no rclpy) — tests/unit/test_image_convert.py
    "openral_perception_ros",
    # pure-Python sensor_leg (rclpy deferred) — tests/unit/test_sensor_leg.py
    "openral_rskill_ros",
):
    _pkg_dir = _REPO_ROOT / "packages" / _pkg
    if _pkg_dir.is_dir() and str(_pkg_dir) not in sys.path:
        sys.path.insert(0, str(_pkg_dir))

# Filter out anything below WARNING.  CLAUDE.md §5.4 forbids logging at INFO
# at module import; this also keeps construction / connect / disconnect
# records out of test stdout, where they would race doctest expected output.
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    cache_logger_on_first_use=True,
)


# ── ROS 2 `launch` swaps the global logger class ────────────────────────────
# Importing `launch` runs `launch.logging.reset()` at module scope, which calls
# `logging.setLoggerClass(LaunchLogger)` — process-wide and permanent. Every
# logger created after that point, in any package, is a `LaunchLogger` whose
# `__init__` sets `self.propagate = False`.
#
# In production that is harmless: `launch` lives in the launcher process and
# the nodes it spawns are separate processes that never import it. In a test
# session it is not, because one `importorskip("launch.actions")` silently
# changes how every later test's loggers behave. It broke
# `test_sim_attached_hal.py::test_task_success_logger_reaches_the_openral_otel_bridge_namespace`,
# which asserts that `openral.sim.task_success` records reach the `openral`
# logger the OTel bridge is attached to — true in isolation, false after any
# test in the same worker had imported `launch`, and invisible in CI's ordering
# for a long while.
#
# Restoring the class is not enough on its own: loggers *already* built inside
# the polluted window keep `propagate = False` forever, so the `openral`
# namespace is repaired too. `openral.otel_bridge` is deliberately left alone —
# `install_structlog_bridge` sets `propagate = False` there on purpose, to stop
# double-emission through the root logger.
_BRIDGE_LOGGER = "openral.otel_bridge"


def _restore_stdlib_logging() -> None:
    """Put the stdlib logger class back and re-arm the `openral` namespace."""
    if logging.getLoggerClass() is not logging.Logger:
        logging.setLoggerClass(logging.Logger)
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if not name.startswith("openral") or name == _BRIDGE_LOGGER:
            continue
        # `type(...) is not` rather than `isinstance`: LaunchLogger subclasses
        # Logger, so an isinstance check matches it and repairs nothing.
        if isinstance(logger, logging.Logger) and type(logger) is not logging.Logger:
            logger.propagate = True


@pytest.fixture(autouse=True)
def _stdlib_logger_class() -> Iterator[None]:
    """Undo ROS 2 `launch`'s global `setLoggerClass` around every test.

    Run on the way in as well as out: `launch` is imported at module scope by
    several test modules, so the swap can happen during collection — before any
    fixture of the polluting test exists to tear down.
    """
    _restore_stdlib_logging()
    yield
    _restore_stdlib_logging()
