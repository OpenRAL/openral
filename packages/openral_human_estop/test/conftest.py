"""ROS_DOMAIN_ID isolation for parallel `colcon test` runs.

The human-estop forwarder publishes ``Empty`` to ``/openral/estop``
and ``FailureTrigger(KIND_HUMAN)`` to ``/openral/failure/safety``.
Both are shared safety buses in production; when ``colcon test`` runs
the sibling ``openral_safety_watchdog`` package in parallel on the
same DDS domain, the watchdog's KIND_TIMEOUT lands on this package's
``failures_received`` subscription and the test's helper publisher
also feeds estops into the watchdog's node (latching it). Kind
filtering only fixes the receive side; for true isolation each
pytest subprocess needs its own DDS domain.

Mirror of ``packages/openral_safety_watchdog/test/conftest.py`` — the
two packages mutually contaminate each other and need the same
treatment.
"""

from __future__ import annotations

import os

if "ROS_DOMAIN_ID" not in os.environ:
    os.environ["ROS_DOMAIN_ID"] = str((os.getpid() % 100) + 100)
