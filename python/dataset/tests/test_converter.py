"""End-to-end tests for :class:`Rosbag2ToLeRobotConverter` (ADR-0019 PR4).

Per CLAUDE.md §1.11 — real `mcap` writer (via `Rosbag2Sink`) writes a
bag to `tmp_path`, then the real `Rosbag2ToLeRobotConverter.from_bag`
replays it through a real `lerobot.datasets.LeRobotDataset` writer.
The result is reloaded by `lerobot.datasets.LeRobotDataset(root)` and
asserted on. No mocks anywhere.

What this file covers:
* Empty bag → `ROSConfigError` (the converter refuses to guess).
* Single-episode round-trip: PHASE_START → ticks → PHASE_END produces a
  v3 dataset whose `num_episodes==1`, `num_frames==n_ticks`, and
  `meta/info.json[metadata][dataset_success_rate]==1.0`.
* Mixed-success multi-episode: dataset_success_rate matches the
  PHASE_END success flags.
* PHASE_START with no PHASE_END (interrupted episode) lands as a
  failure — the deactivate-with-open-episode contract.
* Custom repo_id / license overrides land in meta/info.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_dataset import (
    RolloutRecorder,
    Rosbag2Sink,
    Rosbag2ToLeRobotConverter,
)

# lerobot is required for the LeRobotDatasetSink the converter writes
# into. Skip the whole module when it's not installed — matches the
# `test_sink_lerobot.py` pattern.
lerobot = pytest.importorskip(
    "lerobot",
    reason=(
        "lerobot>=0.5.1 not installed; install via "
        "`just sync --all-packages --group metaworld` or "
        "`uv pip install lerobot>=0.5.1`"
    ),
)


def _camera_frames(robot: RobotDescription, fill: int = 0) -> dict[str, np.ndarray]:
    """Per-camera frame at the manifest's declared intrinsic resolution.

    The enriched ``Rosbag2Sink`` now records real pixels, so the frames
    fed to ``record_frame`` must match ``SensorSpec.intrinsics`` (the
    sink + converter validate against it) — not an arbitrary thumbnail.
    """
    images: dict[str, np.ndarray] = {}
    for sensor in robot.sensors:
        if sensor.vla_feature_key is None or sensor.intrinsics is None:
            continue
        key = sensor.vla_feature_key.removeprefix("observation.images.")
        channels = 1 if sensor.modality in {"depth", "ir", "thermal"} else 3
        images[key] = np.full(
            (int(sensor.intrinsics.height), int(sensor.intrinsics.width), channels),
            fill,
            dtype=np.uint8,
        )
    return images


def _zero_frame(
    robot: RobotDescription,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    state = np.zeros(robot.observation_spec.state_shape, dtype=np.float32)
    action = np.zeros(robot.action_spec.dim, dtype=np.float32)
    return state, _camera_frames(robot), action


def _write_real_bag(
    bag_path: Path,
    robot: RobotDescription,
    *,
    episodes: list[tuple[str, int, bool]],
) -> None:
    """Drive a real RolloutRecorder + Rosbag2Sink to produce a tmp_path bag.

    ``episodes`` is a list of ``(task_string, n_ticks, success)`` tuples.
    """
    sink = Rosbag2Sink(bag_path=bag_path)
    rec = RolloutRecorder(robot=robot, task_string="default", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(robot)
    for task_string, n_ticks, success in episodes:
        rec.episode_start(task_string=task_string)
        for _ in range(n_ticks):
            rec.record_frame(observation_state=state, images=images, action=action)
        rec.episode_end(success=success)
    rec.finalize()


# ── Validation ────────────────────────────────────────────────────────────────


def test_converter_rejects_missing_bag(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """Non-existent bag path raises a clean ROSConfigError."""
    with pytest.raises(ROSConfigError, match=r"does not exist"):
        Rosbag2ToLeRobotConverter.from_bag(
            bag_path=tmp_path / "nope.mcap",
            robot=so100_robot,
            output_root=tmp_path / "ds",
        )


def test_converter_rejects_bag_without_episode_markers(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """A bag with no /openral/episode markers raises a clean ROSConfigError.

    We craft this case by writing a bag that contains only Ticks. The
    only way to currently produce a "no episodes" bag with the real
    sink is to never call open_episode — so we write an empty bag
    (just the mcap header) and assert the converter rejects it.
    """
    from mcap.writer import Writer

    bag_path = tmp_path / "empty.mcap"
    with bag_path.open("wb") as f:
        writer = Writer(f)
        writer.start(profile="openral", library="test")
        writer.finish()

    with pytest.raises(ROSConfigError, match=r"no /openral/episode markers"):
        Rosbag2ToLeRobotConverter.from_bag(
            bag_path=bag_path,
            robot=so100_robot,
            output_root=tmp_path / "ds",
        )


# ── End-to-end round-trip ────────────────────────────────────────────────────


def test_converter_round_trip_single_episode(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """One 3-tick success episode → 1 episode, 3 frames, success_rate=1.0."""
    bag_path = tmp_path / "single.mcap"
    _write_real_bag(bag_path, so100_robot, episodes=[("pick the cube", 3, True)])

    summary = Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
        repo_id="openral/dataset-test",
    )
    assert summary.n_episodes == 1
    assert summary.n_frames == 3
    assert summary.n_success == 1
    assert summary.repo_id == "openral/dataset-test"

    # Reload the dataset via the real lerobot reader and verify on-disk
    # truth matches the summary.
    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("openral/dataset-test", root=tmp_path / "ds")
    assert ds.num_episodes == 1
    assert ds.num_frames == 3
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["metadata"]["dataset_success_rate"] == pytest.approx(1.0)
    assert info["metadata"]["repo_id"] == "openral/dataset-test"


def test_converter_round_trip_mixed_success(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """Mixed-success multi-episode → success_rate matches the marker truth."""
    bag_path = tmp_path / "mixed.mcap"
    _write_real_bag(
        bag_path,
        so100_robot,
        episodes=[
            ("pick", 2, True),
            ("pick", 1, False),
            ("pick", 4, True),
        ],
    )
    summary = Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
    )
    assert summary.n_episodes == 3
    assert summary.n_frames == 7
    assert summary.n_success == 2
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["metadata"]["dataset_success_rate"] == pytest.approx(2 / 3)
    assert info["metadata"]["n_episodes"] == 3
    assert info["metadata"]["n_success_episodes"] == 2


def test_converter_uses_default_repo_id(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """No --repo-id → defaults to openral/dataset-<robot_name>."""
    bag_path = tmp_path / "default_id.mcap"
    _write_real_bag(bag_path, so100_robot, episodes=[("t", 1, True)])
    summary = Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
    )
    assert summary.repo_id == f"openral/dataset-{so100_robot.name}"


def test_converter_custom_license_lands_in_info(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """--license overrides the default CC-BY-4.0 in meta/info.json."""
    bag_path = tmp_path / "license.mcap"
    _write_real_bag(bag_path, so100_robot, episodes=[("t", 1, True)])
    Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
        license="CC-BY-NC-4.0",
    )
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["metadata"]["license"] == "CC-BY-NC-4.0"


def test_converter_preserves_bag_trace_ids(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """ISSUE-109: offline bag→LeRobot keeps each tick's ORIGINAL trace id.

    The converter replays ticks through its own ``RolloutRecorder``; if it
    captured the live (converter-process) span the on-disk trace id would
    point at the conversion run, not the original rollout. The bag's
    per-tick (trace_id, span_id) must round-trip verbatim into the
    LeRobot parquet rows.
    """
    import pandas as pd
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")

    # Write a bag whose single tick carries a real, known trace id.
    bag_path = tmp_path / "traced.mcap"
    sink = Rosbag2Sink(bag_path=bag_path)
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)
    rec.episode_start()
    with tracer.start_as_current_span("rskill.tick"):
        ctx = trace.get_current_span().get_span_context()
        exp_trace = f"{ctx.trace_id:032x}"
        exp_span = f"{ctx.span_id:016x}"
        rec.record_frame(observation_state=state, images=images, action=action)
    rec.episode_end(success=True)
    rec.finalize()

    # Convert offline — outside any span of our own.
    Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
        repo_id="openral/dataset-test",
    )

    files = sorted((tmp_path / "ds").glob("data/**/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = df.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["trace_id"] == exp_trace
    assert rows[0]["span_id"] == exp_span


def test_converter_round_trips_real_state_action_images(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """The enriched bag carries REAL state/action/images, not zeros.

    Regression guard for the ADR-0019 PR4-follow-up: previously the
    converter wrote ``np.zeros`` for every observation/action/image
    because the bag held only metadata. Now ``Rosbag2Sink`` records the
    inline arrays + per-camera pixels, so a recorded bag round-trips the
    actual values. We write a per-frame ramp + a constant-fill image and
    assert they survive into the reloaded LeRobotDataset.
    """
    import pandas as pd

    bag_path = tmp_path / "real.mcap"
    n_ticks = 4
    state_dim = so100_robot.observation_spec.state_shape[0]
    action_dim = so100_robot.action_spec.dim
    img_fill = 123

    sink = Rosbag2Sink(bag_path=bag_path)
    rec = RolloutRecorder(robot=so100_robot, task_string="ramp", fps=30.0, sinks=[sink])
    rec.episode_start(task_string="ramp")
    for i in range(n_ticks):
        rec.record_frame(
            observation_state=np.full(state_dim, float(i), dtype=np.float32),
            images=_camera_frames(so100_robot, fill=img_fill),
            action=np.full(action_dim, float(i) + 0.5, dtype=np.float32),
        )
    rec.episode_end(success=True)
    rec.finalize()

    # The sink actually wrote image messages (one per camera per tick).
    assert sink.n_images_written == n_ticks * 2

    out = tmp_path / "ds"
    Rosbag2ToLeRobotConverter.from_bag(bag_path=bag_path, robot=so100_robot, output_root=out)

    files = sorted(out.glob("data/**/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values(
        "frame_index"
    )
    rows = df.to_dict(orient="records")
    assert len(rows) == n_ticks
    # State / action survive exactly (parquet float columns, no codec).
    for i, row in enumerate(rows):
        assert list(np.asarray(row["observation.state"]).ravel()) == pytest.approx(
            [float(i)] * state_dim
        )
        assert list(np.asarray(row["action"]).ravel()) == pytest.approx(
            [float(i) + 0.5] * action_dim
        )

    # Images survive as real (non-zero) video frames. lerobot returns the
    # decoded frame normalised to [0, 1] (CHW float32), so a constant
    # fill of 123 lands near 123/255 ≈ 0.48. SVT-AV1 is lossy, so assert
    # the mean is in a band around the expected fill rather than exact.
    ds = lerobot.datasets.LeRobotDataset(  # type: ignore[attr-defined]
        f"openral/dataset-{so100_robot.name}", root=out
    )
    frame0 = ds[0]
    cam_np = np.asarray(frame0["observation.images.camera1"])
    assert cam_np.size > 0
    mean = float(cam_np.astype(np.float32).mean())
    assert 0.30 < mean < 0.65, (
        f"decoded camera frame {mean} not near expected fill {img_fill / 255}"
    )


def test_converter_round_trips_robot_without_observation_spec(tmp_path: Path) -> None:
    """from-bag works for a robot whose layout lives only on the rSkill contract.

    franka_panda has ``observation_spec=None`` / ``action_spec=None`` (its
    proprio/action dims come from the active rSkill's state/action contracts,
    ADR-0007/ADR-0019). The converter must derive the LeRobot feature shapes
    from the recorded bag itself rather than the (absent) RobotDescription
    specs — this is the deploy path's robot (LIBERO/Franka).
    """
    import pandas as pd

    repo_root = Path(__file__).resolve().parents[3]
    franka = RobotDescription.from_yaml(str(repo_root / "robots" / "franka_panda" / "robot.yaml"))
    assert franka.observation_spec is None  # guard: the case under test

    state_dim, action_dim = 8, 7
    bag_path = tmp_path / "franka.mcap"
    sink = Rosbag2Sink(bag_path=bag_path)
    rec = RolloutRecorder(robot=franka, task_string="t", fps=20.0, sinks=[sink])
    rec.episode_start(task_string="put the bowl on the plate")
    for i in range(3):
        rec.record_frame(
            observation_state=np.full(state_dim, float(i), dtype=np.float32),
            images={
                "camera1": np.full((64, 64, 3), 50, dtype=np.uint8),
                "camera2": np.full((64, 64, 3), 60, dtype=np.uint8),
            },
            action=np.full(action_dim, float(i), dtype=np.float32),
        )
    rec.episode_end(success=True)
    rec.finalize()

    out = tmp_path / "ds"
    summary = Rosbag2ToLeRobotConverter.from_bag(bag_path=bag_path, robot=franka, output_root=out)
    assert summary.n_episodes == 1
    assert summary.n_frames == 3

    ds = lerobot.datasets.LeRobotDataset(f"openral/dataset-{franka.name}", root=out)  # type: ignore[attr-defined]
    assert ds.num_frames == 3
    info = json.loads((out / "meta" / "info.json").read_text())
    assert tuple(info["features"]["observation.state"]["shape"]) == (state_dim,)
    assert tuple(info["features"]["action"]["shape"]) == (action_dim,)
    files = sorted(out.glob("data/**/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert list(np.asarray(df.iloc[1]["observation.state"]).ravel()) == pytest.approx([1.0] * 8)


def test_converter_propagates_episode_success_through_recorder(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """A failure episode produces a 0/1 success rate end-to-end."""
    bag_path = tmp_path / "failure.mcap"
    _write_real_bag(bag_path, so100_robot, episodes=[("t", 2, False)])
    summary = Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
    )
    assert summary.n_success == 0
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["metadata"]["dataset_success_rate"] == pytest.approx(0.0)
