"""ADR-0019 — DatasetRecorderBridge records the deploy bus to a rosbag2 mcap.

Drives :class:`openral_runner.dataset_recorder_bridge.DatasetRecorderBridge`
directly (no full ``ros2 launch``) with a minimal stand-in node, a REAL
:class:`~openral_world_state.WorldStateAggregator` fed real
:class:`~openral_core.JointState` + :class:`~openral_core.SensorFrame`, a
REAL :class:`openral_dataset.Rosbag2Sink`, and synthetic
``openral_msgs/Episode`` + ``ActionChunk`` payloads.

Per CLAUDE.md §1.11 — no mocked aggregator / sink / bag reader. The only
stand-in is the rclpy node (its sole role here is to vend / release
subscription handles; the bridge's callbacks are invoked directly, exactly
as the executor would). Parametrised over TWO embodiments (SO-100 6-DoF and
Franka 8-DoF) to prove the bridge is embodiment-agnostic — nothing in it is
robot-specific.

Requires ``openral_msgs`` (built ROS workspace) for the ``ActionChunk`` /
``Episode`` types the bridge imports; skips cleanly otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("openral_msgs", reason="openral_msgs not built; source the ROS workspace")
pytest.importorskip("rclpy", reason="rclpy unavailable; source the ROS workspace")

from openral_core import JointState, RobotDescription, SensorFrame
from openral_dataset import Rosbag2Sink
from openral_dataset.recorder import RolloutRecorder
from openral_runner.dataset_recorder_bridge import DatasetRecorderBridge

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _StandinNode:
    """Minimal rclpy.node.Node stand-in: vends + releases subscription handles."""

    def __init__(self) -> None:
        self.subs: list[Any] = []

    def create_subscription(self, _msg_type: Any, _topic: str, _cb: Any, _qos: Any) -> object:
        sub = object()
        self.subs.append(sub)
        return sub

    def destroy_subscription(self, sub: Any) -> None:
        self.subs.remove(sub)


def _rgb_sensor_names(robot: RobotDescription) -> list[str]:
    return [s.name for s in robot.sensors if getattr(s, "modality", None) == "rgb"]


def _feed_snapshot(
    aggregator: Any, robot: RobotDescription, *, state_dim: int, fill: int, step: int
) -> None:
    """Push one joint state + one frame per RGB sensor into the aggregator."""
    joint_names = [j.name for j in robot.joints][:state_dim]
    while len(joint_names) < state_dim:
        joint_names.append(f"extra_{len(joint_names)}")
    aggregator.update_joint_state(
        JointState(
            name=joint_names,
            position=[float(step)] * state_dim,
            velocity=[0.0] * state_dim,
            effort=[0.0] * state_dim,
            stamp_ns=step + 1,
        )
    )
    for name in _rgb_sensor_names(robot):
        h, w, c = 16, 16, 3
        aggregator.update_image_frame(
            name,
            SensorFrame(
                sensor_id=name,
                stamp_monotonic_ns=step + 1,
                stamp_wall_ns=step + 1,
                encoding="rgb8",
                width=w,
                height=h,
                channels=c,
                data=bytes([fill]) * (h * w * c),
            ),
        )


@pytest.mark.parametrize(
    ("robot_yaml", "state_dim", "action_dim"),
    [
        ("robots/so100_follower/robot.yaml", 6, 6),
        ("robots/franka_panda/robot.yaml", 8, 7),
    ],
)
def test_bridge_records_real_bus_data(
    robot_yaml: str, state_dim: int, action_dim: int, tmp_path: Path
) -> None:
    """The bridge writes a bag whose ticks carry real proprio + action + images.

    Embodiment-agnostic: SO-100 (6-DoF, observation_spec present) and Franka
    (8-DoF, no observation_spec) both record without any robot-specific code.
    """
    robot = RobotDescription.from_yaml(str(_REPO_ROOT / robot_yaml))
    from openral_world_state import WorldStateAggregator

    aggregator = WorldStateAggregator(robot)
    bag_path = tmp_path / "deploy.mcap"
    recorder = RolloutRecorder(
        robot=robot,
        task_string="",
        fps=30.0,
        sinks=[Rosbag2Sink(bag_path=bag_path)],
        repo_id=f"openral/dataset-{robot.name}",
    )
    node = _StandinNode()
    bridge = DatasetRecorderBridge(node, robot=robot, aggregator=aggregator, recorder=recorder)
    assert len(node.subs) == 2  # action + episode

    n_rgb = len(_rgb_sensor_names(robot))
    n_ticks = 3

    # Episode opens → ticks (each joins latest snapshot + this chunk) → closes.
    bridge._on_episode(SimpleNamespace(phase=0, task_string="pick the cube", success=False))
    for step in range(n_ticks):
        _feed_snapshot(aggregator, robot, state_dim=state_dim, fill=100 + step, step=step)
        chunk = SimpleNamespace(
            flat=[float(step) + 0.5] * action_dim,
            n_dof=action_dim,
        )
        bridge._on_action(chunk)
    bridge._on_episode(SimpleNamespace(phase=1, task_string="pick the cube", success=True))

    # An action arriving outside an open episode is ignored.
    bridge._on_action(SimpleNamespace(flat=[9.0] * action_dim, n_dof=action_dim))

    bridge.destroy()  # finalizes the bag

    # Read the bag back and assert real per-tick content.
    from mcap.reader import make_reader

    ticks: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    with bag_path.open("rb") as f:
        for _schema, channel, message in make_reader(f).iter_messages():
            payload = json.loads(message.data.decode("utf-8"))
            if channel.topic == "/openral/tick":
                ticks.append(payload)
            elif channel.topic == "/openral/episode":
                episodes.append(payload)
            elif channel.topic == "/openral/dataset/image":
                images.append(payload)

    assert len(ticks) == n_ticks, "one tick per in-episode action chunk"
    assert len(images) == n_ticks * n_rgb, "one image message per camera per tick"
    assert [e["phase"] for e in episodes] == [0, 1]
    assert episodes[1]["success"] is True

    # Proprio + action are REAL (not zeros) and embodiment-shaped.
    for step, tick in enumerate(sorted(ticks, key=lambda t: t["step_idx"])):
        assert tick["observation_state"] == pytest.approx([float(step)] * state_dim)
        assert tick["action"] == pytest.approx([float(step) + 0.5] * action_dim)

    # Camera pixels round-trip (constant fill per step).
    assert all(img["encoding"] == "raw_u8" for img in images)
    assert {img["camera"] for img in images}  # at least one slot recorded


def test_bridge_reassembles_slot_dispatched_action(tmp_path: Path) -> None:
    """Multi-slot (ADR-0028b) ticks reassemble into one full action vector.

    Regression guard for the deploy-graph action-fidelity bug: a
    slot-dispatched skill (e.g. LIBERO = a 6-D cartesian_delta ActionChunk +
    a separate 1-D gripper ActionChunk per tick) must record ONE 7-D action
    per tick — the concatenation of its slots — not just the last-delivered
    chunk. The bridge detects the tick boundary by the slot cycle.
    """
    # franka_panda is the real slot-dispatch / LIBERO deploy robot
    # (observation_spec / action_spec = None — its layout lives on the rSkill
    # contract), so the recorder does not constrain the 7-D reassembled action.
    robot = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"))
    from openral_world_state import WorldStateAggregator

    aggregator = WorldStateAggregator(robot)
    bag_path = tmp_path / "slots.mcap"
    recorder = RolloutRecorder(
        robot=robot, task_string="", fps=20.0, sinks=[Rosbag2Sink(bag_path=bag_path)]
    )
    bridge = DatasetRecorderBridge(
        _StandinNode(), robot=robot, aggregator=aggregator, recorder=recorder
    )

    # Distinct control_mode ints for the two slots (cartesian_delta vs gripper).
    cm_cartesian, cm_gripper = 3, 4
    n_ticks = 3
    bridge._on_episode(SimpleNamespace(phase=0, task_string="pick", success=False))
    for step in range(n_ticks):
        _feed_snapshot(aggregator, robot, state_dim=8, fill=100 + step, step=step)
        # Slot 1: 6-D cartesian delta; slot 2: 1-D gripper — same fixed order
        # each tick, both stamped with the same 1-based tick_index (ADR-0019).
        tick = step + 1
        bridge._on_action(
            SimpleNamespace(
                control_mode=cm_cartesian,
                ee_name="",
                n_dof=6,
                flat=[float(step)] * 6,
                tick_index=tick,
            )
        )
        bridge._on_action(
            SimpleNamespace(
                control_mode=cm_gripper,
                ee_name="",
                n_dof=1,
                flat=[float(step) + 0.9],
                tick_index=tick,
            )
        )
    bridge._on_episode(SimpleNamespace(phase=1, task_string="pick", success=True))
    bridge.destroy()

    from mcap.reader import make_reader

    ticks = []
    with bag_path.open("rb") as f:
        for _s, ch, m in make_reader(f).iter_messages():
            if ch.topic == "/openral/tick":
                ticks.append(json.loads(m.data))
    assert len(ticks) == n_ticks, "one reassembled frame per tick, not per slot chunk"
    for step, tick in enumerate(sorted(ticks, key=lambda t: t["step_idx"])):
        # Full 7-D action = 6 cartesian + 1 gripper, concatenated in slot order.
        assert tick["action"] == pytest.approx([float(step)] * 6 + [float(step) + 0.9])


def test_bridge_tick_index_groups_same_key_slots(tmp_path: Path) -> None:
    """tick_index reassembles even when two slots share (control_mode, ee_name).

    This is the case the slot-cycle fallback CANNOT handle (a repeated key would
    flush prematurely): e.g. a bimanual robot emitting two same-mode joint
    chunks with empty ee_name in one tick. ADR-0019's explicit ActionChunk
    tick_index groups them robustly — both chunks of a tick share the index, so
    they reassemble into one frame regardless of key collisions.
    """
    robot = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"))
    from openral_world_state import WorldStateAggregator

    aggregator = WorldStateAggregator(robot)
    bag_path = tmp_path / "samekey.mcap"
    recorder = RolloutRecorder(
        robot=robot, task_string="", fps=20.0, sinks=[Rosbag2Sink(bag_path=bag_path)]
    )
    bridge = DatasetRecorderBridge(
        _StandinNode(), robot=robot, aggregator=aggregator, recorder=recorder
    )
    bridge._on_episode(SimpleNamespace(phase=0, task_string="t", success=False))
    n_ticks = 3
    for step in range(n_ticks):
        _feed_snapshot(aggregator, robot, state_dim=8, fill=100 + step, step=step)
        tick = step + 1
        # Two chunks, IDENTICAL (control_mode, ee_name) — only tick_index keeps
        # them in the same tick (slot-cycle alone would split them).
        bridge._on_action(
            SimpleNamespace(
                control_mode=1, ee_name="", n_dof=4, flat=[float(step)] * 4, tick_index=tick
            )
        )
        bridge._on_action(
            SimpleNamespace(
                control_mode=1, ee_name="", n_dof=3, flat=[float(step) + 0.5] * 3, tick_index=tick
            )
        )
    bridge._on_episode(SimpleNamespace(phase=1, task_string="t", success=True))
    bridge.destroy()

    from mcap.reader import make_reader

    ticks = []
    with bag_path.open("rb") as f:
        for _s, ch, m in make_reader(f).iter_messages():
            if ch.topic == "/openral/tick":
                ticks.append(json.loads(m.data))
    assert len(ticks) == n_ticks, "same-key slots must stay in ONE tick via tick_index"
    for step, tick in enumerate(sorted(ticks, key=lambda t: t["step_idx"])):
        assert tick["action"] == pytest.approx([float(step)] * 4 + [float(step) + 0.5] * 3)
