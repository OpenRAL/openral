"""End-to-end tests for ``openral_dataset.LeRobotDatasetSink``.

Per CLAUDE.md §1.11 — uses the real
:class:`lerobot.datasets.LeRobotDataset` writer (v3.0,
codebase_version="3.0") and real SO-100 RobotDescription. Tests
``pytest.skip`` with a typed reason on hosts without lerobot
installed.

What this test asserts end-to-end:
* The sink writes a valid v3.0 on-disk format that can be re-loaded
  by ``LeRobotDataset(root)``.
* Per-frame ``observation.state`` / ``action`` / camera arrays round-trip.
* ``dataset_success_rate`` lands in ``meta/info.json["metadata"]``.
* Failed-episode rows tag ``next.success`` independently of success
  episodes (the persist-all-with-flag decision in ADR-0019 §3).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_dataset import LeRobotDatasetSink, RolloutRecorder

lerobot = pytest.importorskip(
    "lerobot",
    reason=(
        "lerobot>=0.5.1 not installed; install with "
        "`just sync --all-packages --group metaworld` or "
        "`uv pip install lerobot>=0.5.1`"
    ),
)


def _zero_frame(robot: RobotDescription) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    state = np.zeros(robot.observation_spec.state_shape, dtype=np.float32)
    action = np.zeros(robot.action_spec.dim, dtype=np.float32)
    # ADR-0019: frame shape MUST match SensorSpec.intrinsics — SO-100
    # declares 256x256 for both cameras.
    images = {
        "camera1": np.zeros((256, 256, 3), dtype=np.uint8),
        "camera2": np.zeros((256, 256, 3), dtype=np.uint8),
    }
    return state, images, action


def test_sink_round_trip_two_episodes(
    so100_robot: RobotDescription,
    tmp_path: Path,
    require_video_decode: Callable[[], None],
) -> None:
    """Write two episodes (one success, one failure) and reload."""
    root = tmp_path / "ds"
    sink = LeRobotDatasetSink(
        root=root, robot=so100_robot, fps=30.0, repo_id="openral/dataset-test"
    )
    rec = RolloutRecorder(
        robot=so100_robot,
        task_string="pick the cube",
        fps=30.0,
        sinks=[sink],
        repo_id="openral/dataset-test",
    )
    state, images, action = _zero_frame(so100_robot)

    # Episode 0 — success, 3 frames
    rec.episode_start()
    for i in range(3):
        rec.record_frame(
            observation_state=state + i,
            images=images,
            action=action + i,
            reward=float(i),
        )
    rec.episode_end(success=True)

    # Episode 1 — failure, 2 frames
    rec.episode_start()
    for i in range(2):
        rec.record_frame(
            observation_state=state + i + 10,
            images=images,
            action=action,
            reward=0.0,
            truncated=(i == 1),
        )
    rec.episode_end(success=False)

    rec.finalize()

    # Reload via the real lerobot v3 reader and verify the on-disk format.
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset("openral/dataset-test", root=root)
    assert dataset.num_episodes == 2, f"expected 2 episodes, got {dataset.num_episodes}"
    assert dataset.num_frames == 5, f"expected 5 frames total, got {dataset.num_frames}"

    # Indexing decodes the recorded MP4 via torchcodec; skip when the host's
    # video backend can't load (CLAUDE.md §1.11). Counts above need no decode.
    require_video_decode()
    # First frame of episode 0 should round-trip the state vector.
    row0 = dataset[0]
    assert row0["observation.state"].numpy().tolist() == [0.0] * 6
    # Third frame of episode 0 has state +2 across all 6 dims.
    row2 = dataset[2]
    assert row2["observation.state"].numpy().tolist() == [2.0] * 6
    # Reward column on frame 2 must be 2.0. v3 reads scalars back as
    # 0-d tensors regardless of the (1,) shape we declared at create-time,
    # so use `.item()` rather than `[0]`.
    assert float(row2["next.reward"].item()) == pytest.approx(2.0)


def test_sink_writes_dataset_success_rate(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """meta/info.json carries the dataset-level success rate aggregate."""
    import json

    root = tmp_path / "ds"
    sink = LeRobotDatasetSink(root=root, robot=so100_robot, fps=30.0)
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)

    # 3 episodes, 2 successful → 2/3 = 0.6666...
    for success in (True, False, True):
        rec.episode_start()
        rec.record_frame(observation_state=state, images=images, action=action)
        rec.episode_end(success=success)
    rec.finalize()

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    meta = info.get("metadata", {})
    assert meta["n_episodes"] == 3
    assert meta["n_success_episodes"] == 2
    assert meta["dataset_success_rate"] == pytest.approx(2 / 3)
    assert meta["license"] == "CC-BY-4.0"
    assert meta["repo_id"] == f"openral/dataset-{so100_robot.name}"
    assert meta["robot_name"] == so100_robot.name


def test_sink_camera_shape_comes_from_intrinsics(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """ADR-0019: camera shapes are taken from SensorSpec.intrinsics, not the first frame.

    The sink declares features at construction time using the
    intrinsics on every camera-bearing sensor. The SO-100 manifest
    declares 256x256 for both cameras.
    """
    import json

    root = tmp_path / "ds"
    sink = LeRobotDatasetSink(root=root, robot=so100_robot, fps=30.0)
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)
    rec.episode_start()
    rec.record_frame(observation_state=state, images=images, action=action)
    rec.episode_end(success=True)
    rec.finalize()

    info = json.loads((root / "meta" / "info.json").read_text())
    cam_feature = info["features"]["observation.images.camera1"]
    # info.json serialises tuples as lists.
    assert tuple(cam_feature["shape"]) == (256, 256, 3), (
        f"camera shape must come from SensorSpec.intrinsics (256x256); got {cam_feature['shape']!r}"
    )


def test_sink_rejects_camera_frame_shape_mismatch(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """ADR-0019: per-frame shape validation rejects frames that don't match intrinsics.

    A camera frame that arrives at the wrong resolution is a wiring
    bug; the sink raises ValueError immediately rather than producing
    a corrupted dataset.
    """
    root = tmp_path / "ds"
    sink = LeRobotDatasetSink(root=root, robot=so100_robot, fps=30.0)
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state = np.zeros(so100_robot.observation_spec.state_shape, dtype=np.float32)
    action = np.zeros(so100_robot.action_spec.dim, dtype=np.float32)
    wrong_shape_images = {
        "camera1": np.zeros((48, 64, 3), dtype=np.uint8),
        "camera2": np.zeros((48, 64, 3), dtype=np.uint8),
    }
    rec.episode_start()
    with pytest.raises(ValueError, match=r"does not match declared feature shape"):
        rec.record_frame(observation_state=state, images=wrong_shape_images, action=action)


def test_sink_construction_rejects_robot_without_intrinsics(
    tmp_path: Path, repo_root: Path
) -> None:
    """ADR-0019: a robot whose camera sensors lack intrinsics is rejected at sink __init__.

    Loud failure at construction beats a confusing error inside lerobot
    on the first ``add_frame``.
    """
    from openral_core import RobotDescription
    from openral_core.exceptions import ROSConfigError

    # franka_panda has intrinsics; aloha_bimanual has intrinsics. We
    # need a robot without them — sawyer/ur5e/etc. lack observation_spec
    # entirely, which fails earlier. So we construct an in-test robot
    # by loading SO-100 and pydantic-stripping the intrinsics off both
    # sensors. This is the cleanest way to exercise the failure path
    # without committing a broken manifest to robots/.
    robot = RobotDescription.from_yaml(str(repo_root / "robots" / "so100_follower" / "robot.yaml"))
    robot_no_intrinsics = robot.model_copy(
        update={"sensors": [s.model_copy(update={"intrinsics": None}) for s in robot.sensors]},
    )
    with pytest.raises(ROSConfigError, match=r"intrinsics"):
        LeRobotDatasetSink(root=tmp_path / "ds", robot=robot_no_intrinsics, fps=30.0)


def _read_parquet_rows(root: Path) -> list[dict[str, object]]:
    """Read every v3 data parquet under ``root`` as plain row dicts.

    Reads the columns directly (pandas / pyarrow) so the assertion never
    touches the video backend — the trace_id / span_id round-trip must be
    verifiable on hosts where torchcodec/ffmpeg cannot decode the MP4s.
    """
    import pandas as pd

    files = sorted(root.glob("data/**/*.parquet"))
    assert files, f"no data parquet files under {root}/data"
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df.to_dict(orient="records")


def test_sink_writes_per_frame_trace_and_span_ids(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """ISSUE-109: a written LeRobotDataset frame carries the producing tick's ids.

    Drives the recorder inside a real ``rskill.tick`` span, then reads the
    on-disk parquet back and asserts every row's ``trace_id`` (32 hex) /
    ``span_id`` (16 hex) is the non-empty id of the span that produced it.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")

    root = tmp_path / "ds"
    sink = LeRobotDatasetSink(
        root=root, robot=so100_robot, fps=30.0, repo_id="openral/dataset-test"
    )
    rec = RolloutRecorder(
        robot=so100_robot, task_string="t", fps=30.0, sinks=[sink], repo_id="openral/dataset-test"
    )
    state, images, action = _zero_frame(so100_robot)

    captured: list[tuple[str, str]] = []
    rec.episode_start()
    for _ in range(2):
        with tracer.start_as_current_span("rskill.tick"):
            ctx = trace.get_current_span().get_span_context()
            captured.append((f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"))
            rec.record_frame(observation_state=state, images=images, action=action)
    rec.episode_end(success=True)
    rec.finalize()

    rows = _read_parquet_rows(root)
    assert len(rows) == 2
    for row, (exp_trace, exp_span) in zip(rows, captured, strict=True):
        assert row["trace_id"] == exp_trace
        assert row["span_id"] == exp_span
        assert len(row["trace_id"]) == 32
        assert len(row["span_id"]) == 16


def test_sink_writes_dataset_and_episode_level_trace_pointers(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """ISSUE-109 follow-up: meta carries dataset- and episode-level trace pointers.

    Each episode is recorded under its own root trace (the multi-run /
    resume-append shape), so the dataset-level ``trace_ids`` is the
    distinct set and ``meta/openral_traces.json`` maps every
    ``episode_index`` to the trace that produced it.
    """
    import json

    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")
    root = tmp_path / "ds"
    sink = LeRobotDatasetSink(
        root=root, robot=so100_robot, fps=30.0, repo_id="openral/dataset-test"
    )
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)

    ep_traces: list[str] = []
    for _ in range(2):
        # One root span per episode → a distinct trace_id per episode.
        with tracer.start_as_current_span("cli.command") as run_span:
            ep_traces.append(f"{run_span.get_span_context().trace_id:032x}")
            rec.episode_start()
            for _ in range(2):
                with tracer.start_as_current_span("rskill.tick"):
                    rec.record_frame(observation_state=state, images=images, action=action)
            rec.episode_end(success=True)
    rec.finalize()

    # Dataset-level — distinct trace_ids land in meta/info.json.
    info = json.loads((root / "meta" / "info.json").read_text())
    meta = info["metadata"]
    assert meta["trace_ids"] == sorted(set(ep_traces))
    assert meta["n_traces"] == 2

    # Episode-level — every episode maps to its producing trace.
    sidecar = json.loads((root / "meta" / "openral_traces.json").read_text())
    by_ep = {e["episode_index"]: e["trace_id"] for e in sidecar["episodes"]}
    assert by_ep == {0: ep_traces[0], 1: ep_traces[1]}


def test_sink_construction_without_lerobot_raises(
    so100_robot: RobotDescription, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ROSConfigError is raised at construction when lerobot is unimportable."""
    import sys

    from openral_core.exceptions import ROSConfigError

    # Pretend lerobot isn't installed by stubbing the import. We don't
    # actually uninstall it — the sink does `import lerobot` as a presence
    # probe; replacing sys.modules['lerobot'] with None makes the import
    # raise ImportError.
    monkeypatch.setitem(sys.modules, "lerobot", None)
    with pytest.raises(ROSConfigError, match=r"lerobot>=0\.5\.1"):
        LeRobotDatasetSink(root=tmp_path / "ds", robot=so100_robot, fps=30.0)
