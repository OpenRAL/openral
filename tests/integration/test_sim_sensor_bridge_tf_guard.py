"""Live ROS integration test for the ``SimSensorBridge`` mobile-base TF guard.

``SimSensorBridge`` publishes a static ``world -> base_frame`` root for a
fixed-base sim arm, and must skip it for a mobile robot, whose
``MobileBaseBridge`` already publishes a live ``odom -> base_link``. Two parents
for one frame split ``/tf`` into unconnected trees: ``map -> base_link`` stops
resolving and Nav2's global costmap times out.

The predicate behind that guard (``describes_mobile_base``) is pinned without
ROS in ``tests/unit/test_mobile_base_tf_guard.py``. This is the half that needs
a real ``rclpy`` node and the real in-process ``PandaMobileHAL`` — so it lives
here, gated on ``OPENRAL_TEST_ROS_LIVE=1`` like its neighbours and listed in
``scripts/ros_live_tests.sh``. It ran on no CI lane while it sat under
``tests/unit/``: test-selective has no ``rclpy``, and the docker ``ros-live``
lane runs only the files that script enumerates.

CI runs it inside ``openral:x86`` (the ``docker-build`` workflow). Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live            # whole suite; `-k <expr>` narrows it
"""

from __future__ import annotations

import os

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy node construction — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "(no torch import) and source install/setup.bash first."
)


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_sim_sensor_bridge_skips_the_world_root_for_a_mobile_robot() -> None:
    """The guard itself (not just the predicate) short-circuits for panda_mobile.

    Exercised on the real bridge with a real rclpy node and the real in-process
    ``PandaMobileHAL``. No MuJoCo handle is passed: the mobile decision must be
    reached from the manifest alone, before the bridge reads the model — which
    is also why no static broadcaster is ever constructed.
    """
    rclpy = pytest.importorskip("rclpy")

    from openral_hal.panda_mobile import PANDA_MOBILE_DESCRIPTION, PandaMobileHAL
    from openral_hal.sim_sensor_bridge import SimSensorBridge
    from rclpy.node import Node

    rclpy.init()
    try:
        node = Node("test_mobile_base_tf_guard")
        try:
            bridge = SimSensorBridge(
                node,
                PandaMobileHAL(),
                PANDA_MOBILE_DESCRIPTION,
                viewer_enabled=False,
            )
            bridge._publish_world_base_tf(None, None)

            # The settled flag is the assertion that discriminates: reaching it
            # means the guard fired. Falling through to the "no base body
            # resolved yet" return also publishes nothing *right now*, but
            # leaves the decision pending — so the static world->base_link goes
            # out the moment the MuJoCo base body resolves, which is exactly
            # what the dead guard did on RoboCasa.
            assert bridge._world_base_published, (
                "a mobile robot must get no static world->base_link — MobileBaseBridge "
                "already publishes odom->base_link, and a second parent splits /tf"
            )
            assert bridge._static_tf_broadcaster is None
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
