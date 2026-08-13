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

These are the pure-predicate half, which needs no ROS. The bridge half — the
guard running on a real ``SimSensorBridge`` with a real ``rclpy`` node — is
``tests/integration/test_sim_sensor_bridge_tf_guard.py``, gated on
``OPENRAL_TEST_ROS_LIVE=1`` and listed in ``scripts/ros_live_tests.sh``. It
lived here behind an ``importorskip`` and so ran on no CI lane at all.
"""

from __future__ import annotations

from pathlib import Path

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


def test_footprint_radius_decides_nothing_in_either_direction() -> None:
    """``footprint_radius`` is a Nav2 tuning knob, not the mobile-base signal.

    Both directions matter, because a guard keyed off it fails both ways:

    * A mobile robot may omit it (nothing in the schema requires it), so such a
      guard would resume publishing the second parent the day someone lands a
      mobile manifest without Nav2 tuning.
    * A fixed-base arm may carry one, and such a guard would then drop the
      static ``world -> base_frame`` root the arm needs to be placed at all.
    """
    panda_mobile = _description("panda_mobile")
    without_nav2_tuning = panda_mobile.model_copy(update={"footprint_radius": None})
    assert describes_mobile_base(without_nav2_tuning)

    franka = _description("franka_panda")
    with_nav2_tuning = franka.model_copy(update={"footprint_radius": 0.35})
    assert not describes_mobile_base(with_nav2_tuning)
