# SPDX-License-Identifier: Apache-2.0
"""The sim bridge never certifies "nothing carried" for a twin whose grasp it cannot measure.

Issue #752: when ``SimAttachmentEvidenceTracker`` refused a robot, ``SimSensorBridge`` kept
heartbeating an empty, fresh ``/openral/attachment_state``, so the safety kernel read
"nothing carried" while the twin could close its gripper on a payload. Now:

* a twin that drives a gripper (``twin_drives_gripper``) and whose tracker refuses has the
  stream withdrawn (its publisher destroyed), so the kernel's attached check fails closed;
* a twin with no gripper keeps the empty heartbeat, which is the truth there;
* the SO-101, one of the robots the issue listed, arms;
* a scene model whose bodies the kernel does not name (LIBERO's robosuite Panda, beside a
  kernel lowered from the menagerie twin) arms under the twin's names.

Real components (CLAUDE.md §1.11): a real ``rclpy`` ``LifecycleNode``, the real
``SimSensorBridge`` and ``SimAttachedHAL`` over the native-MuJoCo ``so101_box`` scene, the
real ``robots/so101_follower/robot.yaml``. The two refusal cases edit that manifest in
memory (no fixed jaw; no gripper at all), the way a robot the tracker cannot model looks.
Skips without ROS 2 + colcon-built ``openral_msgs``, ``openral_sim`` or ``mujoco``; the LIBERO
case also needs the ``libero`` group and ``openral_safety`` (its kernel link names).

In-process rather than ``launch_testing`` (CLAUDE.md §2 names launch_testing for this tier),
like ``test_sim_idle_stepper_wiring.py``: what is asserted is the bridge's own setup outcome
(publisher, heartbeat timer, tracker), which a black-box launch cannot observe without a
kernel and a wall-clock wait.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — this test constructs a real rclpy LifecycleNode.",
)


@pytest.fixture
def setup_bridge() -> Iterator[Callable[..., Any]]:
    """Build the bridge for a (possibly edited) SO-101 description and arm its attachment state."""
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")

    import rclpy
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml
    from openral_hal.sim_sensor_bridge import SimSensorBridge
    from rclpy.lifecycle import LifecycleNode

    rclpy.init()
    made: list[tuple[Any, Any, Any]] = []

    def build(
        description: Any,
        *,
        scene: str = "scenes/sim/so101_tube_insertion.yaml",
        kernel_link_names: list[str] | None = None,
        no_mujoco: bool = False,
    ) -> Any:
        if no_mujoco:
            # A rollout with no MuJoCo handles (an Isaac / SAPIEN backend's shape), through
            # the SimRollout boundary double.
            from tests.unit.fakes.fake_sim_env import FakeSimEnv

            env: Any = FakeSimEnv(action_dim=6)
            seed: int | None = None
        else:
            env, seed = build_sim_env_from_yaml(scene, robot_id_fallback=description.name)
        hal = SimAttachedHAL(
            env,
            RobotDescription.from_yaml(f"robots/{description.name}/robot.yaml"),
            env_reset_seed=seed,
            env_action_dim=6 if description.name == "so101_follower" else None,
        )
        hal.connect()
        node = LifecycleNode(f"test_sim_attachment_fail_closed_{len(made)}")
        bridge = SimSensorBridge(
            node,
            hal,
            description,
            viewer_enabled=False,
            attachment_heartbeat=True,
            attachment_kernel_checked=True,
            attachment_kernel_link_names=kernel_link_names,
        )
        made.append((bridge, node, env))
        bridge._setup_attachment_state()
        return bridge

    try:
        yield build
    finally:
        for bridge, node, env in made:
            bridge.teardown()
            node.destroy_node()
            env.close()  # SimAttachedHAL.disconnect does not close the env it was given
        rclpy.shutdown()


def _so101() -> Any:
    from openral_core import RobotDescription

    return RobotDescription.from_yaml("robots/so101_follower/robot.yaml")


def test_a_twin_the_tracker_models_arms(setup_bridge: Callable[[Any], Any]) -> None:
    bridge = setup_bridge(_so101())
    assert bridge._attachment_tracker is not None
    assert bridge._attachment_pub is not None and bridge._attachment_timer is not None


def test_a_gripper_twin_the_tracker_refuses_withdraws_the_stream(
    setup_bridge: Callable[[Any], Any],
) -> None:
    desc = _so101()
    joints = [j.model_copy(update={"fixed_jaw_link": None}) for j in desc.joints]
    bridge = setup_bridge(desc.model_copy(update={"joints": joints}))
    assert bridge._attachment_tracker is None
    assert bridge._attachment_pub is None and bridge._attachment_timer is None
    # Destroyed, not just dropped: no silent publisher stays advertised on the topic.
    assert bridge._node.count_publishers("/openral/attachment_state") == 0


def test_a_twin_without_a_gripper_keeps_the_empty_heartbeat(
    setup_bridge: Callable[[Any], Any],
) -> None:
    desc = _so101()
    joints = [
        j.model_copy(update={"role": "unknown", "fixed_jaw_link": None})
        if j.role == "gripper"
        else j
        for j in desc.joints
    ]
    assert desc.sim is not None
    sim = desc.sim.model_copy(update={"grippers": []})
    bridge = setup_bridge(desc.model_copy(update={"joints": joints, "sim": sim}))
    assert bridge._attachment_tracker is None
    assert bridge._attachment_pub is not None and bridge._attachment_timer is not None


def test_a_scene_model_takes_the_kernel_twins_names(setup_bridge: Callable[..., Any]) -> None:
    """LIBERO's Panda arms under the menagerie twin's names, the ones the kernel was lowered to."""
    pytest.importorskip("libero")
    lowering = pytest.importorskip("openral_safety.mjcf_lowering")
    import mujoco
    from openral_core import RobotDescription
    from openral_core.assets import resolve_asset

    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    assert desc.assets.mjcf is not None
    twin = mujoco.MjModel.from_xml_path(str(resolve_asset(desc.assets.mjcf, "mjcf")))
    names = lowering.lower_collision_params(twin, [j.name for j in desc.joints], margin_m=0.0)[
        "collision_link_names"
    ]
    bridge = setup_bridge(desc, scene="scenes/deploy/libero_pnp.yaml", kernel_link_names=names)
    assert bridge._attachment_pub is not None and bridge._attachment_timer is not None
    (hand,) = bridge._attachment_tracker._hands
    assert (hand.attach_link, sorted(hand.touch_links)) == ("hand", ["left_finger", "right_finger"])
    # Contacts stay measured on the scene's own bodies.
    model, _ = bridge._hal.mujoco_handles()
    assert model.body(hand.attach_body_id).name == "robot0_right_hand"


def test_a_kernel_checked_hal_without_mujoco_contacts_withdraws_the_stream(
    setup_bridge: Callable[..., Any],
) -> None:
    """The launch said this scene measures grasps (a ``mujoco`` backend label), the HAL has
    no MuJoCo handles: nothing would measure, so the stream is withdrawn (#754 review)."""
    bridge = setup_bridge(_so101(), no_mujoco=True)
    assert bridge._attachment_pub is None and bridge._attachment_timer is None
    assert bridge._node.count_publishers("/openral/attachment_state") == 0
