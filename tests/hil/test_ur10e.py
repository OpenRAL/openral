"""HIL tests for the Universal Robots UR10e via ``ur_robot_driver``.

Sister of ``tests/hil/test_ur5e.py`` — same driver, different URDF /
payload envelope.  Gated by the ``[self-hosted, lab-ur10e]`` runner label.

Environment:
    UR10E_HOST: Static IP of the UR10e controller.
    UR10E_COMMAND_TOPIC: Override for the trajectory command topic.
    UR10E_STATE_TOPIC: Override for the joint-state topic.

Safety rules: see ``tests/hil/test_ur5e.py``; the UR10e is bigger (12.5 kg
payload) and the same hold-in-place / no-motion conventions apply.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import time
from typing import TYPE_CHECKING

import pytest
from openral_core import Action, ControlMode
from openral_core.exceptions import ROSEStopRequested, ROSPerceptionStale, ROSRuntimeError
from openral_hal.ur_real import UR10e_REAL_DESCRIPTION, UR10eRealHAL

if TYPE_CHECKING:
    from collections.abc import Iterator

UR10E_HOST = os.environ.get("UR10E_HOST", "")
UR10E_COMMAND_TOPIC = os.environ.get(
    "UR10E_COMMAND_TOPIC", "/scaled_joint_trajectory_controller/joint_trajectory"
)
UR10E_STATE_TOPIC = os.environ.get("UR10E_STATE_TOPIC", "/joint_states")

# ── Skip guards ──────────────────────────────────────────────────────────────

pytestmark = [
    pytest.mark.skipif(
        not UR10E_HOST,
        reason="UR10E_HOST not set — no live UR10e controller available.",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("rclpy") is None,
        reason="rclpy not installed — UR HIL needs a live ROS 2 stack.",
    ),
]


@pytest.fixture()
def ur10e_hal() -> Iterator[UR10eRealHAL]:
    """Connect a UR10eRealHAL bridged to the live ros2_control driver."""
    from tests.hil._ros_control_transport import make_hil_transport  # reason: HIL-only import

    joint_names = [j.name for j in UR10e_REAL_DESCRIPTION.joints]
    _node, transport, cleanup = make_hil_transport(
        node_name="openral_hil_ur10e",
        joint_names=joint_names,
        command_topic=UR10E_COMMAND_TOPIC,
        joint_state_topic=UR10E_STATE_TOPIC,
    )
    hal = UR10eRealHAL(
        robot_ip=UR10E_HOST,
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    try:
        hal.connect()
        assert transport.wait_for_first_state(deadline_s=2.0), (
            "driver published no joint state within 2 s — is ur_robot_driver running?"
        )
        yield hal
    finally:
        with contextlib.suppress(Exception):
            hal.disconnect()
        with contextlib.suppress(Exception):
            cleanup()


# ── Tests ────────────────────────────────────────────────────────────────────


class TestUR10eHIL:
    def test_hal_advertises_real_hal(self, ur10e_hal: UR10eRealHAL) -> None:
        assert ur10e_hal.description.sdk_kind == "closed"
        assert ur10e_hal.description.hal.real == "openral_hal.ur_real:UR10eRealHAL"
        assert ur10e_hal.description.name == "ur10e"
        assert len(ur10e_hal.description.joints) == 6

    def test_read_state_under_200ms(self, ur10e_hal: UR10eRealHAL) -> None:
        t0 = time.monotonic()
        state = ur10e_hal.read_state()
        elapsed_ms = (time.monotonic() - t0) * 1e3
        assert state.name == [j.name for j in ur10e_hal.description.joints]
        assert len(state.position) == 6
        assert elapsed_ms < 200, f"read_state took {elapsed_ms:.1f} ms (limit 200 ms)"

    def test_hold_in_place_no_motion(self, ur10e_hal: UR10eRealHAL) -> None:
        before = ur10e_hal.read_state()
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[list(before.position)],
            stamp_ns=time.time_ns(),
        )
        ur10e_hal.send_action(action)
        time.sleep(0.2)
        after = ur10e_hal.read_state()
        for b, a in zip(before.position, after.position, strict=True):
            assert abs(a - b) < 0.0175, (  # 1° in radians
                f"joint moved {abs(a - b):.4f} rad during hold (before={b:.4f}, after={a:.4f})"
            )

    def test_disconnect_is_idempotent(self, ur10e_hal: UR10eRealHAL) -> None:
        ur10e_hal.disconnect()
        ur10e_hal.disconnect()
        with pytest.raises(ROSRuntimeError):
            ur10e_hal.read_state()

    def test_deadman_topic_advertised(self, ur10e_hal: UR10eRealHAL) -> None:
        assert ur10e_hal.deadman_topic == "/io_and_status_controller/safety_mode"

    def test_estop_raises_estoprequested(self, ur10e_hal: UR10eRealHAL) -> None:
        with pytest.raises(ROSEStopRequested):
            ur10e_hal.estop()

    def test_perception_stale_after_long_silence(self, ur10e_hal: UR10eRealHAL) -> None:
        ur10e_hal._last_state_time -= ur10e_hal._staleness_limit_s + 0.1
        with pytest.raises(ROSPerceptionStale):
            ur10e_hal.read_state()
