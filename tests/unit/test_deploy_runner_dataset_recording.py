"""Dataset-bridge integration tests — DeployRunner episode API + Rosbag2Sink.

Real components per CLAUDE.md §1.11:
  * Real SO100FollowerHAL backed by SO100DigitalTwin (no serial port).
  * Real WorldStateAggregator over the SO-100 description.
  * Real ``RolloutRecorder`` + real ``Rosbag2Sink`` writing a real
    mcap bag to ``tmp_path``.
  * Real ``rSkillBase`` subclass driven through its full lifecycle.

The covered surface:
  * ``DeployRunner.episode_start`` / ``episode_end`` driving
    the recorder's episode lifecycle (and propagating through to the
    sink as PHASE_START / PHASE_END markers).
  * In-tick fan-out of state / action into the bag via the recorder's
    ``record_frame`` path, with the camera frames the skill stepped on at
    capture resolution (this path used to write 1x1 zero placeholders).
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
from openral_dataset.bag import PHASE_END, PHASE_START, TOPIC_EPISODE, TOPIC_IMAGE, TOPIC_TICK
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


class _FrameWatchingSkill(_NoOpSkill):
    """``_NoOpSkill`` that keeps the camera frame each step was given."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[object] = []

    def _step_impl(self, world_state: WorldState) -> Action:
        self.seen.append((world_state.image_frames or {}).get("wrist"))
        return super()._step_impl(world_state)


def test_runner_records_the_native_camera_frame_the_skill_saw(
    so100_robot_description: RobotDescription, tmp_path: Path
) -> None:
    """A real BGR8 reader's frame reaches the skill's snapshot and the bag, full-size and RGB.

    The reader is a real ``opencv_thread`` on an MJPG clip, so the tick's sensor
    phase — not the test — is what puts the frame into ``image_frames``; the
    runner used to forward only topic refs, leaving the skill and the recorder
    blind to every inline camera. The bag image must be byte-identical to the
    array ``decode_policy_image`` builds from the frame the skill stepped on.
    """
    import base64
    import time

    import numpy as np
    from openral_core.exceptions import ROSPerceptionStale
    from openral_runner.backends.opencv_thread import OpenCVThreadSensorReader
    from openral_runner.dataset_recorder_bridge import decode_policy_image

    cv2 = pytest.importorskip("cv2")
    width, height = 96, 60  # an Arducam's 960x600 aspect, small enough for a unit test
    clip = tmp_path / "wrist.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter MJPG codec unavailable on this host")
    try:
        for _ in range(30):
            writer.write(np.full((height, width, 3), (200, 0, 0), dtype=np.uint8))  # blue, BGR
    finally:
        writer.release()

    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    hal = SO100FollowerHAL(robot=twin)
    aggregator = WorldStateAggregator(so100_robot_description)
    skill = _FrameWatchingSkill()
    skill.configure()
    skill.activate()
    bag_path = tmp_path / "hardware.mcap"
    recorder = RolloutRecorder(
        robot=so100_robot_description,
        task_string="pick the cube",
        fps=30.0,
        sinks=[Rosbag2Sink(bag_path=bag_path)],
    )
    # The clip's last frame stays the reader's latest after EOF; a wide age
    # budget keeps it fresh for the tick however slow the host.
    reader = OpenCVThreadSensorReader(
        sensor_id="wrist", device=str(clip), fps=30, default_max_age_ms=10_000
    )
    runner = DeployRunner(
        hal=hal, skill=skill, aggregator=aggregator, sensor_readers=[reader], recorder=recorder
    )
    runner.activate()
    try:
        deadline = time.monotonic() + 5.0
        while True:  # first capture
            try:
                reader.read_latest()
                break
            except ROSPerceptionStale:
                assert time.monotonic() < deadline, "opencv_thread never produced a frame"
                time.sleep(0.02)
        runner.episode_start("pick the cube")
        runner.run(max_ticks=1)
        runner.episode_end(success=True)
    finally:
        runner.deactivate()
        skill.deactivate()
        skill.shutdown()

    seen = skill.seen[0]
    assert seen is not None, "the skill stepped on a snapshot without the camera frame"
    policy_rgb = decode_policy_image(seen)
    assert policy_rgb is not None and policy_rgb.shape == (height, width, 3)

    images = [m for t, m in _read_bag(bag_path) if t == TOPIC_IMAGE]
    assert len(images) == 1
    img = images[0]
    assert (img["camera"], img["width"], img["height"]) == ("wrist", width, height)
    bagged = np.frombuffer(base64.b64decode(str(img["data_b64"])), dtype=np.uint8)
    assert bagged.tobytes() == policy_rgb.tobytes()
    # Blue in the BGR capture is blue in the RGB bag (MJPG is lossy; the hue is not).
    assert policy_rgb[..., 2].mean() > 150 and policy_rgb[..., 0].mean() < 60


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
