# openral-dataset

**ADR-0019** — rosbag2 ↔ LeRobotDataset v3 bridge. Every skill execution (sim
or hardware) becomes a row in a LeRobotDataset v3.0 (`codebase_version="3.0"`,
`lerobot>=0.5.1`). Successful and failed episodes are both persisted; the
per-row `next.success` flag and the per-dataset `meta/info.json["metadata"]
["dataset_success_rate"]` let downstream consumers filter.

See [`docs/adr/0019-rosbag2-lerobot-dataset-bridge.md`](../../docs/adr/0019-rosbag2-lerobot-dataset-bridge.md)
for the architectural rationale (top-level package vs nested submodule, v3.0
vs v2.1, persist-all vs discard, license posture, PR sequencing).

## Public API

```python
from openral_core import RobotDescription
from openral_dataset import LeRobotDatasetSink, RolloutRecorder

robot = RobotDescription.from_yaml("robots/so100_follower/robot.yaml")
sink = LeRobotDatasetSink(
    root="/tmp/ds",
    robot=robot,
    fps=30.0,
    repo_id="openral/dataset-pick-cube",
    license="CC-BY-4.0",
)
with RolloutRecorder(
    robot=robot,
    task_string="pick the cube",
    fps=30.0,
    sinks=[sink],
    repo_id="openral/dataset-pick-cube",
) as rec:
    rec.episode_start()
    # for each tick:
    rec.record_frame(
        observation_state=state_vec,    # (state_dim,) float32
        images={"camera1": rgb_frame},  # camera_key → (H, W, 3) uint8
        action=action_vec,              # (action_dim,) float32
        reward=0.0,
        terminated=False,
        truncated=False,
    )
    rec.episode_end(success=True)
```

## Components

- **`RolloutRecorder`** — in-memory per-rollout accumulator with multi-sink
  fan-out. Writes the `openral.dataset.repo_id` / `episode_idx` / `frame_idx`
  OTel attributes on the active `rskill.tick` span so the Jaeger trace can be
  joined to the on-disk frame.
- **`DatasetSink` Protocol** — every sink (online / offline) implements
  `open_episode` / `write_frame` / `close_episode` / `finalize`.
- **`LeRobotDatasetSink`** — wraps the real
  `lerobot.datasets.LeRobotDataset.create / add_frame / save_episode /
  finalize`. Lazy-imports lerobot; raises `ROSConfigError` with an install
  hint when lerobot is absent.
- **`features_from_robot`** — pure `RobotDescription` → LeRobot v3 features
  dict mapping. No I/O, no lerobot import.

The PR series adds these in later sessions (still planned):

- **PR2** — `SensorRosPublisher` in `python/sensors/` + new
  `packages/openral_sensors_ros/` lifecycle node. Closes the camera-topic
  publishing gap on the hardware path.
- **PR3** — `Rosbag2Sink` (mcap, daemon writer thread) + `openral_msgs/Tick`
  / `openral_msgs/Episode` IDLs + explicit `episode_start` / `episode_end`
  API on `HardwareRunner`.
- **PR4** — `Rosbag2ToLeRobotConverter.from_bag` + `openral dataset from-bag`
  CLI subcommand.
- **PR5** — `openral dataset push` with consent prompt + `_hf_publish` shared
  helper de-duped from `tools/rskill_publisher.py`.

## CLI surface

`openral sim run` accepts three new flags from PR1:

- `--dataset-out PATH` — write a LeRobotDataset v3 to `PATH` as the sim runs.
  Path must not pre-exist; lerobot v3 refuses to write into a populated root.
- `--dataset-repo-id STR` — repo id stored in `meta/info.json`. Defaults to
  `openral/dataset-<robot_id>`. Not pushed by `openral sim run`; PR5's
  `openral dataset push` owns publishing.
- `--dataset-license SPDX` — defaults to `CC-BY-4.0` (the LeRobot
  convention). PII-bearing datasets must set a more restrictive license; the
  PR5 consent prompt enforces this.

## Verification

```bash
uv run pytest python/dataset/tests/ -v
uv run mypy --strict -p openral_dataset
uv run ruff check python/dataset/
```

End-to-end against a real `lerobot.datasets.LeRobotDataset` v3.0 round-trip:

```bash
uv run pytest python/dataset/tests/test_sink_lerobot.py -v
```

Per CLAUDE.md §1.11 (no mocks): every test loads the real SO-100
`RobotDescription` from `robots/so100_follower/robot.yaml` and exercises the
real `lerobot.datasets.LeRobotDataset` writer. Tests `pytest.skip` with a
typed reason on hosts without `lerobot>=0.5.1` installed (it lives behind
the `libero` / `metaworld` dependency groups today).
