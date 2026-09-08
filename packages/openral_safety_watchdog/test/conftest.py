"""ROS_DOMAIN_ID isolation for parallel `colcon test` runs.

The deadman watchdog subscribes to ``/openral/estop`` itself as a
defense-in-depth latch: an externally-arriving estop suppresses its own
publish. That's a flake risk when ``colcon test`` runs the sibling
``openral_human_estop`` package in parallel on the same DDS domain — its
forwarder publishes ``Empty`` to ``/openral/estop``, this package's watchdog
latches, and ``test_deadman_fires_when_safe_action_stops`` sees zero
KIND_TIMEOUT FailureTriggers. Kind-filtering (the fix for
``/openral/failure/safety`` cross-talk) can't rescue this, since the node's
behavior — not just the test's assertion — is contaminated; each pytest
subprocess needs its own DDS domain.

``ament_cmake_pytest`` invokes one pytest process per test file, so a
PID-derived domain gives stable per-test-file isolation: ``(pid % 100) + 100``
lands inside [100, 199], inside DDS's safe domain range. Only mutates
``ROS_DOMAIN_ID`` if unset, so manual ``ROS_DOMAIN_ID=N just ros2-test`` keeps
working.
"""

from __future__ import annotations

import os

if "ROS_DOMAIN_ID" not in os.environ:
    os.environ["ROS_DOMAIN_ID"] = str((os.getpid() % 100) + 100)
