"""ROS_DOMAIN_ID isolation for parallel `colcon test` runs.

The deadman watchdog subscribes to ``/openral/estop`` itself as a
defense-in-depth latch (see ``packages/openral_safety_watchdog/
package.xml``): if an estop arrives externally before the watchdog's
own deadline expires, it suppresses its own publish. That production
behavior turns into a flake when ``colcon test`` runs the sibling
``openral_human_estop`` package in parallel on the same DDS domain
— the human-estop forwarder publishes ``Empty`` to ``/openral/estop``,
this package's watchdog latches, and
``test_deadman_fires_when_safe_action_stops`` sees zero KIND_TIMEOUT
FailureTriggers within its 3 s window because the watchdog never
fired.

Kind-filtering at the test layer (the fix used for the
``/openral/failure/safety`` cross-talk) cannot rescue this case
because the node's behavior, not just the test's assertion, is
contaminated. The clean fix is to put each pytest subprocess on its
own DDS domain so the helper publishers, the watchdog node, and the
sibling tests cannot see each other.

ament_cmake_pytest invokes one pytest process per test file, so a
PID-derived domain gives stable per-test-file isolation. Pinning to
``(pid % 100) + 100`` lands inside [100, 199], inside DDS's safe
domain range (0..101 is fine; >101 would need DDS port-base tuning).
We only mutate ``ROS_DOMAIN_ID`` if the caller hasn't already set
one — manual ``ROS_DOMAIN_ID=N just ros2-test`` keeps working.
"""

from __future__ import annotations

import os

if "ROS_DOMAIN_ID" not in os.environ:
    os.environ["ROS_DOMAIN_ID"] = str((os.getpid() % 100) + 100)
