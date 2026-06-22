"""ADR-0019 PR3 integration tests — DeployRunner episode API + Rosbag2Sink.

Real components per CLAUDE.md §1.11:
  * Real SO100FollowerHAL backed by SO100DigitalTwin (no serial port).
  * Real WorldStateAggregator over the SO-100 description.
  * Real ``RolloutRecorder`` + real :class:`Rosbag2Sink` writing a real
    mcap bag to ``tmp_path``.
  * Real ``rSkillBase`` subclass driven through its full lifecycle.

The covered surface:
  * :meth:`DeployRunner.episode_start` / :meth:`episode_end` driving
    the recorder's episode lifecycle (and propagating through to the
    sink as PHASE_START / PHASE_END markers).
  * In-tick fan-out of state / action into the bag via the recorder's
    ``record_frame`` path.
  * Idempotent deactivation: a still-open recorder episode is closed as
    a failure on teardown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from openral_core import Action, ControlMode, RobotDescription
from openral_core.schemas import WorldState
from openral_dataset import RolloutRecorder, Rosbag2Sink
from openral_dataset.bag import PHASE_END, PHASE_START, TOPIC_EPISODE, TOPIC_TICK
from openral_hal.so100_follower import SO100FollowerHAL
from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
from openral_rskill.base import rSkillBase
from openral_runner import DeployRunner
from openral_world_state.aggregator import WorldStateAggregator

if TYPE_CHECKING:
    from collections.abc import Generator


class _NoOpSkill(rSkillBase):
    """Minimal inline rSkillBase that returns a zero action chunk every tick."""

    def __init__(self, n_joints: int = 6) -> None:
        super().__init__(name="noop_pr3", embodiment_tags=["so100_follower"])
        self._n_joints = n_joints

    def _configure_impl(self) -> None:
        return None

    def _activate_impl(self) -> None:
        return None

    def _deactivate_impl(self) -> None:
        return None

    def _shutdown_impl(self) -> None:
        return None

    def _step_impl(self, world_state: WorldState) -> Action:
        del world_state
        return Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * self._n_joints],
            confidence=1.0,
        )


def _read_bag(bag_path: Path) -> list[tuple[str, dict[str, object]]]:
    from mcap.reader import make_reader

    out: list[tuple[str, dict[str, object]]] = []
    with bag_path.open("rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            out.append((channel.topic, json.loads(message.data.decode("utf-8"))))
    return out


@pytest.fixture
def so100_robot_description() -> RobotDescription:
    """Real SO-100 follower robot description from the digital twin's HAL."""
    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    hal = SO100FollowerHAL(robot=twin)
    return hal.description


@pytest.fixture
def real_runner_stack(
    so100_robot_description: RobotDescription, tmp_path: Path
) -> Generator[tuple[DeployRunner, RolloutRecorder, Rosbag2Sink, Path], None, None]:
    """Wire a real DeployRunner + RolloutRecorder + Rosbag2Sink end-to-end.

    Yields ``(runner, recorder, sink, bag_path)``. The fixture also
    handles teardown: skill shutdown, recorder finalize (idempotent),
    runner deactivation.
    """
    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    hal = SO100FollowerHAL(robot=twin)
    aggregator = WorldStateAggregator(so100_robot_description)
    skill = _NoOpSkill()
    skill.configure()
    skill.activate()

    bag_path = tmp_path / "hardware.mcap"
    sink = Rosbag2Sink(bag_path=bag_path)
    recorder = RolloutRecorder(
        robot=so100_robot_description,
        task_string="pick the cube",
        fps=30.0,
        sinks=[sink],
    )
    runner = DeployRunner(
        hal=hal,
        skill=skill,
        aggregator=aggregator,
        recorder=recorder,
    )
    runner.activate()
    try:
        yield runner, recorder, sink, bag_path
    finally:
        runner.deactivate()
        if skill.info.state.value == "active":
            skill.deactivate()
        if skill.info.state.value != "finalized":
            skill.shutdown()


# ── Tests ────────────────────────────────────────────────────────────────────


def test_runner_episode_start_without_recorder_returns_minus_one(
    so100_robot_description: RobotDescription,
) -> None:
    """When no recorder is attached, episode_start returns -1 (no-op)."""
    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    hal = SO100FollowerHAL(robot=twin)
    aggregator = WorldStateAggregator(so100_robot_description)
    skill = _NoOpSkill()
    skill.configure()
    skill.activate()
    runner = DeployRunner(hal=hal, skill=skill, aggregator=aggregator)
    runner.activate()
    try:
        assert runner.episode_start("task") == -1
        # episode_end with no recorder is also a no-op (must not raise).
        runner.episode_end(success=True)
    finally:
        runner.deactivate()
        skill.deactivate()
        skill.shutdown()


def test_runner_episode_lifecycle_writes_bag_markers(
    real_runner_stack: tuple[DeployRunner, RolloutRecorder, Rosbag2Sink, Path],
) -> None:
    """episode_start + 2 ticks + episode_end produces PHASE_START + 2 Ticks + PHASE_END."""
    runner, _recorder, sink, bag_path = real_runner_stack

    idx = runner.episode_start("pick the cube")
    assert idx == 0
    runner.run(max_ticks=2)
    runner.episode_end(success=True)
    # Force finalization through deactivate (in the fixture). Read the
    # bag back AFTER teardown to ensure the writer thread drained.
    runner.deactivate()

    assert sink.n_ticks_written == 2
    # PHASE_START + PHASE_END markers.
    assert sink.n_episode_markers_written == 2

    messages = _read_bag(bag_path)
    topics = [topic for topic, _ in messages]
    assert topics[0] == TOPIC_EPISODE
    assert topics[-1] == TOPIC_EPISODE
    assert topics.count(TOPIC_TICK) == 2
    assert topics.count(TOPIC_EPISODE) == 2

    start_msg = messages[0][1]
    end_msg = messages[-1][1]
    assert start_msg["phase"] == PHASE_START
    assert start_msg["task_string"] == "pick the cube"
    assert end_msg["phase"] == PHASE_END
    assert end_msg["success"] is True
    assert end_msg["episode_idx"] == 0


def test_runner_episode_start_twice_raises(
    real_runner_stack: tuple[DeployRunner, RolloutRecorder, Rosbag2Sink, Path],
) -> None:
    """Calling episode_start twice without episode_end raises RuntimeError."""
    runner, _recorder, _sink, _bag = real_runner_stack
    runner.episode_start("t1")
    with pytest.raises(RuntimeError, match=r"still open"):
        runner.episode_start("t2")


def test_runner_episode_end_without_start_raises(
    real_runner_stack: tuple[DeployRunner, RolloutRecorder, Rosbag2Sink, Path],
) -> None:
    """Calling episode_end without a matching episode_start raises RuntimeError."""
    runner, _recorder, _sink, _bag = real_runner_stack
    with pytest.raises(RuntimeError, match=r"no recorder episode open"):
        runner.episode_end(success=True)


def test_runner_deactivate_closes_open_episode_as_failure(
    so100_robot_description: RobotDescription, tmp_path: Path
) -> None:
    """If deactivate fires with an open episode, it gets closed as success=False.

    Mirrors SimRunner's __exit__ contract — half-open episodes on
    teardown are a recoverable wiring bug; the sink must still see a
    clean PHASE_END marker so downstream consumers can reason about it.
    """
    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    hal = SO100FollowerHAL(robot=twin)
    aggregator = WorldStateAggregator(so100_robot_description)
    skill = _NoOpSkill()
    skill.configure()
    skill.activate()
    bag_path = tmp_path / "half_open.mcap"
    sink = Rosbag2Sink(bag_path=bag_path)
    recorder = RolloutRecorder(
        robot=so100_robot_description,
        task_string="abandoned",
        fps=30.0,
        sinks=[sink],
    )
    runner = DeployRunner(hal=hal, skill=skill, aggregator=aggregator, recorder=recorder)
    runner.activate()
    runner.episode_start("abandoned")
    runner.run(max_ticks=1)
    # NOTE: no episode_end before deactivate.
    runner.deactivate()
    skill.deactivate()
    skill.shutdown()

    messages = _read_bag(bag_path)
    end_markers = [m for t, m in messages if t == TOPIC_EPISODE and m["phase"] == PHASE_END]
    assert len(end_markers) == 1
    assert end_markers[0]["success"] is False
