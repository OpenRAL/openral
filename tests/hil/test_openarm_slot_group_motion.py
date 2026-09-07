# SPDX-License-Identifier: Apache-2.0
"""HIL: one slot-dispatched tick actually MOVES the joint it addresses.

The last unclosed gate on ADR-0102. `tests/hil/test_openarm_restock_deploy_preflight.py`
proves the composed 16-DoF vector reaches the four controllers correctly
named and correctly sliced — but it never publishes, so it cannot prove the
values *arrive somewhere physical*. A sign flip, a scale error, or a joint the
controller silently ignores is invisible to it. This is the test that closes
that, and the only one in the tree that commands a real OpenArm to move.

**It requires a person at the E-stop.** Two independent gates, both explicit:

1. ``OPENRAL_OPENARM_ALLOW_MOTION=1`` — never set in CI or by ``just test``.
2. ``OPENRAL_OPENARM_ATTENDED=1`` — the human attestation. Separate on purpose:
   the first says "this bench can move", the second says "someone is watching
   it right now". A rig that leaves gate 1 exported must not thereby become a
   rig that moves unattended.

Why an unattended "tiny step" is not a thing
--------------------------------------------
The wire format is ``trajectory_msgs/JointTrajectory``: an **absolute**
position with a deadline, not a delta. Motion is
``(target - measured) / time_from_start``, so the distance travelled is set by
how wrong the command is — exactly the quantity under test. Picking a small
number does not bound it. Three things bound it here instead:

* every target is **measured pose + delta on one joint**, the other fifteen
  held at their measured values, so a correct command is a near-no-op;
* the run **refuses to start** unless all 16 joints have been seen on
  ``/joint_states``. ``OpenArmRealHAL`` zero-fills unreported joints into a
  full 16-DoF vector, so a partial state reads as a plausible pose containing
  zeros — command "measured + delta" on top of that and the cell slews to
  approximately home;
* ``time_from_start`` is stretched to 0.8 s (production uses 0.1 s), bounding
  the rate by construction. This test therefore does **not** validate the
  production 100 ms deadline — only where the values land.

Each case restores the joint to its measured pose before returning, so the
test is idempotent (CLAUDE.md §2), and the teardown latches the HAL.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from tests.hil.conftest import _can_links_up

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT = REPO_ROOT / "robots" / "openarm" / "robot.yaml"
RSKILL = REPO_ROOT / "rskills" / "rskill-pi05-openarm-restock_shelf-bf16" / "rskill.yaml"

_CAN_LINKS = ("openarm_left", "openarm_right")

#: Radians. ~1.15°, small enough to be unremarkable on a correct command and
#: large enough to be unambiguous against encoder noise.
_DELTA_RAD = 0.02

#: A joint we did not address must not wander further than this. Generous
#: enough for gravity sag and controller jitter, tight enough that a misrouted
#: `_DELTA_RAD` (which would land a full delta on the wrong joint) trips it.
_QUIET_TOL_RAD = 0.01

#: How close the addressed joint must get to its target. A trajectory
#: controller has steady-state error; this asserts arrival, not stiffness.
_ARRIVAL_TOL_RAD = 0.015

#: Seconds to let the 0.8 s trajectory finish and the state settle.
_SETTLE_S = 2.5


requires_can = pytest.mark.skipif(
    not _can_links_up(_CAN_LINKS), reason="OpenArm CAN links are not both up — not on the cell"
)
requires_rclpy = pytest.mark.skipif(
    importlib.util.find_spec("rclpy") is None, reason="rclpy not available"
)
requires_motion_optin = pytest.mark.skipif(
    os.environ.get("OPENRAL_OPENARM_ALLOW_MOTION") != "1",
    reason="motion gate closed — export OPENRAL_OPENARM_ALLOW_MOTION=1 to enable",
)
requires_attended = pytest.mark.skipif(
    os.environ.get("OPENRAL_OPENARM_ATTENDED") != "1",
    reason=(
        "unattended — export OPENRAL_OPENARM_ATTENDED=1 only while someone is "
        "physically at the E-stop"
    ),
)


@requires_can
@requires_rclpy
@requires_motion_optin
@requires_attended
@pytest.mark.parametrize(
    "joint_index,joint_label",
    [
        # The gripper first and by default: 1-DoF, lowest inertia, no reach —
        # and the exact actuator ADR-0102 exists for, since the pre-0102 code
        # dropped its command silently.
        (15, "right_gripper"),
        # Then the most distal arm joint, which proves an ARM slot lands on the
        # right arm joint rather than only that the gripper slot works.
        (14, "right_joint7"),
    ],
)
def test_one_slot_dispatched_tick_moves_only_the_joint_it_addresses(
    joint_index: int, joint_label: str
) -> None:  # pragma: no cover
    """Command measured-pose + delta on one joint; prove only that joint moved.

    Drives the production path end to end — `_dispatch_slots` over the
    committed manifest, into `OpenArmRealHAL`, out through four real
    `JointTrajectory` publishers to `openarm_bringup`'s controllers — and then
    reads the physical result back off `/joint_states`.
    """
    import numpy as np
    import rclpy
    from openral_core.exceptions import ROSEStopRequested
    from openral_core.schemas import RobotDescription, RSkillManifest
    from openral_hal.openarm_real import OpenArmRealHAL
    from rclpy.node import Node

    from tests.hil._openarm_ros_transport import OpenArmHILTransport

    pytest.importorskip("openral_rskill_ros", reason="needs the built ROS overlay")
    from openral_rskill_ros.rskill_runner_node import _dispatch_slots

    robot = RobotDescription.from_yaml(str(ROBOT))
    manifest = RSkillManifest.from_yaml(str(RSKILL))

    context = rclpy.Context()
    context.init()
    node = Node("openarm_slot_motion_hil", context=context)
    hal = None
    try:
        probe = OpenArmRealHAL(robot, require_can_links=False)
        transport = OpenArmHILTransport(
            node,
            probe.ros2_control_joint_names(),
            command_topics=list(probe.command_topics()),
            time_from_start_s=0.8,
        )
        hal = OpenArmRealHAL(robot, publish_fn=transport.publish, state_fn=transport.state)
        hal.connect()

        # Gate on a COMPLETE joint state. A partial one zero-fills into a
        # plausible-looking pose, and "measured + delta" on top of that is a
        # command to slew the whole cell home.
        assert transport.wait_for_every_joint(deadline_s=5.0), (
            "/joint_states never reported these joints: "
            f"{transport.missing_joints()}. Refusing to move: the HAL zero-fills "
            "unreported joints, so the 'measured' pose would be partly fiction. "
            "Is openarm_bringup's ros2_control stack up?"
        )

        measured = list(hal.read_state().position)
        assert len(measured) == 16

        target = list(measured)
        target[joint_index] = measured[joint_index] + _DELTA_RAD

        actions = _dispatch_slots(
            manifest.action_contract.slots,
            np.asarray(target, dtype=np.float32),
            description=robot,
        )
        for action in actions:
            action.tick_index = 1
        for action in actions:
            hal.send_action(action)

        transport.spin_for(_SETTLE_S)
        after = list(hal.read_state().position)

        moved = after[joint_index] - measured[joint_index]
        assert abs(after[joint_index] - target[joint_index]) < _ARRIVAL_TOL_RAD, (
            f"{joint_label} did not reach its target: measured "
            f"{measured[joint_index]:.4f} → commanded {target[joint_index]:.4f}, "
            f"got {after[joint_index]:.4f} (moved {moved:+.4f} rad)"
        )

        for i, name in enumerate(probe.ros2_control_joint_names()):
            if i == joint_index:
                continue
            drift = abs(after[i] - measured[i])
            assert drift < _QUIET_TOL_RAD, (
                f"commanded only {joint_label}, but {name} moved {drift:.4f} rad "
                f"({measured[i]:.4f} → {after[i]:.4f}). A misrouted slot puts the "
                "delta on the wrong joint, which is exactly this assertion."
            )

        # Idempotence: put it back where we found it before leaving.
        restore = _dispatch_slots(
            manifest.action_contract.slots,
            np.asarray(measured, dtype=np.float32),
            description=robot,
        )
        for action in restore:
            action.tick_index = 2
        for action in restore:
            hal.send_action(action)
        transport.spin_for(_SETTLE_S)
        back = list(hal.read_state().position)
        assert abs(back[joint_index] - measured[joint_index]) < _ARRIVAL_TOL_RAD, (
            f"{joint_label} did not return to {measured[joint_index]:.4f} "
            f"(left at {back[joint_index]:.4f}) — the cell is not where the test "
            "found it."
        )
    finally:
        # CLAUDE.md §2 requires an e-stop hook on HIL teardown. For this adapter
        # `estop()` latches the HAL and raises rather than commanding hardware —
        # the physical stop is the button, which is why this test is gated on
        # someone being next to it.
        if hal is not None:
            try:
                hal.estop()
            except ROSEStopRequested:
                pass
            hal.disconnect()
        node.destroy_node()
        context.try_shutdown()
