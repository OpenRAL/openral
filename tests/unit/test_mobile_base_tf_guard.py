"""Pin the mobile-base predicate that keeps ``/tf`` a single connected tree.

``SimSensorBridge`` publishes a static ``world -> base_frame`` root for a
fixed-base sim arm (a robosuite-attached LIBERO franka roots its tree at
``panda_link0`` with no parent, yet sits at a non-origin mount pose), and must
skip it for a mobile robot, whose ``MobileBaseBridge`` already publishes a live
``odom -> base_link``. Two parents for one frame split ``/tf`` into unconnected
trees: ``map -> base_link`` stops resolving and Nav2's global costmap times
out.

The guard and the ``odom`` publisher therefore have to agree on what "mobile"
means, so both call :func:`describes_mobile_base`. Validated against the real
``robots/*/robot.yaml`` manifests — no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.schemas import RobotCapabilities
from openral_hal.mobile_base_bridge import describes_mobile_base

_REPO = Path(__file__).resolve().parents[2]


def _description(robot: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml"))


def test_panda_mobile_reads_as_mobile() -> None:
    """The guard must fire for panda_mobile, so no second parent for ``base_link``."""
    from openral_hal.panda_mobile import PANDA_MOBILE_DESCRIPTION

    assert describes_mobile_base(PANDA_MOBILE_DESCRIPTION)
    assert describes_mobile_base(_description("panda_mobile_vslam"))


def test_the_predicate_matches_the_field_mobile_base_bridge_is_attached_on() -> None:
    """``base_joints`` is the manifest field the lifecycle node attaches ``/odom`` on.

    Reading anything else lets the two drift: the original guard asked
    ``capabilities.footprint_radius``, which ``RobotCapabilities`` does not
    define, so ``getattr`` returned ``None`` for every robot and the guard was
    dead code.
    """
    assert "base_joints" in RobotDescription.model_fields
    assert "footprint_radius" not in RobotCapabilities.model_fields

    panda_mobile = _description("panda_mobile")
    assert panda_mobile.base_joints
    assert getattr(panda_mobile.capabilities, "footprint_radius", None) is None


def test_fixed_base_arm_is_not_mobile() -> None:
    """A fixed-base arm keeps its static ``world -> base_frame`` root."""
    franka = _description("franka_panda")

    assert franka.base_joints is None
    assert not describes_mobile_base(franka)


def test_sim_sensor_bridge_skips_the_world_root_for_a_mobile_robot() -> None:
    """The guard itself (not just the predicate) short-circuits for panda_mobile.

    Exercised on the real bridge with a real rclpy node and the real in-process
    ``PandaMobileHAL``. No MuJoCo handle is passed: the mobile decision must be
    reached from the manifest alone, before the bridge reads the model — which
    is also why no static broadcaster is ever constructed.
    """
    rclpy = pytest.importorskip("rclpy", reason="the bridge half of this test needs rclpy")

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


def test_declared_footprint_radius_alone_does_not_make_a_robot_mobile() -> None:
    """``footprint_radius`` is a Nav2 tuning knob, not the mobile-base signal.

    A mobile robot may omit it (nothing in the schema requires it), so a guard
    keyed off it would resume publishing the second parent the day someone
    lands a mobile manifest without Nav2 tuning.
    """
    panda_mobile = _description("panda_mobile")
    without_nav2_tuning = panda_mobile.model_copy(update={"footprint_radius": None})

    assert describes_mobile_base(without_nav2_tuning)
