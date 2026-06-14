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
