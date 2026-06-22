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

The rest of the ADR-0019 PR series has since landed:

- **PR2** *(partial)* — `SensorRosPublisher` in `python/sensors/`
  (`ros_publisher.py`) ships and is tested. The wrapping
  `packages/openral_sensors_ros/` lifecycle node specified by ADR-0019 is
  **not yet built**, so feeding live camera topics into a recorder on the
  hardware path is the one remaining gap.
- **PR3** ✅ — `Rosbag2Sink` (mcap, daemon writer thread) + `openral_msgs/Tick`
  / `openral_msgs/Episode` IDLs + explicit `episode_start` / `episode_end`
  API on `DeployRunner` (`bag.py`).
- **PR4** ✅ — `Rosbag2ToLeRobotConverter.from_bag` + `openral dataset from-bag`
  CLI subcommand (`converter.py`).
- **PR5** ✅ — `openral dataset push` with consent prompt + `_hf_publish` shared
  helper de-duped from `tools/rskill_publisher.py`.

## CLI surface

**Sim path** — `openral sim run` accepts three flags (PR1):

- `--dataset-out PATH` — write a LeRobotDataset v3 to `PATH` as the sim runs.
  Path must not pre-exist; lerobot v3 refuses to write into a populated root.
- `--dataset-repo-id STR` — repo id stored in `meta/info.json`. Defaults to
  `openral/dataset-<robot_id>`. Not pushed by `openral sim run`;
  `openral dataset push` owns publishing.
- `--dataset-license SPDX` — defaults to `CC-BY-4.0` (the LeRobot
  convention). PII-bearing datasets must set a more restrictive license; the
  `openral dataset push` consent prompt enforces this.

**Hardware / rosbag path** — record on a live ROS 2 graph, then convert
offline:

- `openral record --profile slim|full --out DIR` — wrap `ros2 bag record`
  with curated topic profiles (mcap storage). `--dry-run` prints the composed
  argv without a sourced ROS 2 install.
- `openral dataset from-bag BAG --robot robot.yaml --output DS` — replay a
  `Rosbag2Sink` mcap into a LeRobotDataset v3 (`--repo-id` / `--license` /
  `--fps` optional). PR4.
- `openral dataset push DS [--repo-id ID] [--yes] [--dry-run]` — consent-gated
  upload of a local dataset root to the HF Hub. PR5.

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

### Coverage scope

- **Sim path** is covered end-to-end: `tests/sim/test_dataset_emission.py`
  drives `openral sim run --dataset-out` against `aloha_transfer_cube.yaml`
  with real ACT weights + gym-aloha physics + SVT-AV1 video encoding, then
  re-opens the produced v3 dataset (CUDA + `lerobot` + `gym_aloha` gated; skips
  on CPU/CI).
- **Hardware / rosbag path** is covered only in unit isolation:
  `tests/unit/test_deploy_runner_dataset_recording.py` drives
  `DeployRunner` + `Rosbag2Sink` directly, plus the converter / CLI unit
  tests. It has **not** been exercised through `openral deploy sim` or a live
  ROS 2 graph — the deploy-sim node (`rskill_runner_node.py`) does not yet
  construct or attach a `RolloutRecorder`.
