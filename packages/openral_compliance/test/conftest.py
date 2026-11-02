"""ROS_DOMAIN_ID isolation for parallel `colcon test` runs.

The compliance recorder subscribes to the shared safety buses
(``/openral/failure/*``, ``/openral/safety_status``); a sibling package's
test publishing on the same DDS domain would land in this package's
assertions. Each pytest subprocess gets its own domain, as in
``packages/openral_human_estop/test/conftest.py``.
"""

from __future__ import annotations

import os

if "ROS_DOMAIN_ID" not in os.environ:
    os.environ["ROS_DOMAIN_ID"] = str((os.getpid() % 100) + 100)
