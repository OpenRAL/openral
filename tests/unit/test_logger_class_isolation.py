# SPDX-License-Identifier: Apache-2.0
"""Importing ROS 2 `launch` must not sever the `openral` logging namespace.

`launch/logging/__init__.py` runs `reset()` at module scope, and `reset()` ends
with `logging.setLoggerClass(LaunchLogger)` — process-wide and permanent.
`LaunchLogger.__init__` sets `self.propagate = False`, so from that import
onward *every* logger created anywhere in the process stops propagating.

Production is unaffected: `launch` lives in the launcher process and the nodes
it spawns never import it. A pytest worker is a single process, so one
`importorskip("launch.actions")` changed how every later test's loggers behaved
— which is what made
`test_sim_attached_hal.py::test_task_success_logger_reaches_the_openral_otel_bridge_namespace`
pass alone and fail in the full tier, for a long time, invisibly.

The guard lives in the root `conftest.py` as an autouse fixture. This pins its
behaviour directly rather than through test ordering, because `pytest-randomly`
means an ordering-based regression test would itself be a coin flip.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

import conftest  # the repo-root conftest; pytest has already imported it


class _PropagateOffLogger(logging.Logger):
    """Stands in for `launch.logging.LaunchLogger`, which does exactly this."""

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        super().__init__(name, level)
        self.propagate = False


@pytest.fixture
def foreign_logger_class() -> Iterator[None]:
    """Install the foreign class for the duration, whatever the test does."""
    original = logging.getLoggerClass()
    logging.setLoggerClass(_PropagateOffLogger)
    try:
        yield
    finally:
        logging.setLoggerClass(original)


def test_the_swap_really_does_break_propagation(foreign_logger_class: None) -> None:
    """Guard the premise: if this stops holding, the fixture below proves nothing."""
    polluted = logging.getLogger("openral.test_pollution_premise")
    assert polluted.propagate is False


def test_restoring_puts_the_class_back_and_re_arms_existing_loggers(
    foreign_logger_class: None,
) -> None:
    """Both halves matter.

    Restoring the class alone leaves every logger built inside the polluted
    window propagating nowhere for the rest of the session — and those are
    exactly the ones a later test asserts against.
    """
    polluted = logging.getLogger("openral.test_pollution_repair")
    assert polluted.propagate is False

    conftest._restore_stdlib_logging()

    assert logging.getLoggerClass() is logging.Logger
    assert polluted.propagate is True


def test_the_otel_bridge_logger_keeps_its_deliberate_propagate_false(
    foreign_logger_class: None,
) -> None:
    """`install_structlog_bridge` turns it off on purpose.

    It attaches the OTel handler to `openral.otel_bridge` and stops propagation
    so records are not emitted a second time via the root logger. A blanket
    re-arm of the namespace would undo that silently.
    """
    bridge = logging.getLogger("openral.otel_bridge")
    bridge.propagate = False

    conftest._restore_stdlib_logging()

    assert bridge.propagate is False


def test_a_non_openral_namespace_is_left_alone(foreign_logger_class: None) -> None:
    """The repair is scoped to the namespace the OTel bridge cares about.

    Third-party loggers that legitimately set `propagate = False` are none of
    our business, and re-arming them would duplicate their output into pytest's
    capture.
    """
    other = logging.getLogger("some_vendor.subsystem")
    other.propagate = False

    conftest._restore_stdlib_logging()

    assert other.propagate is False
