# `tests/` — how the suite is laid out and run

What lives where, how each tier runs locally and in CI, and the rules for adding
a test. Current per-tier counts, dead/duplicate tests and no-assertion candidates
are generated, not hand-kept: [`test-audit.md`](../docs/contributing/test-audit.md)
(`just test-audit`). Normative policy: CLAUDE.md §1.7 (tests are part of the
change), §1.11 (real components, no mocks) and §2 ("Tests").

---

## 1. How the suite runs

### Local

| Command | Runs | Notes |
| --- | --- | --- |
| `just test` | `tests/unit/` | No ROS 2, no GPU, no hardware. The three GStreamer files run only when PyGObject is importable. |
| `just test-doctest` | `pytest --doctest-modules` over a curated path list | The list lives in the recipe and in `tests/unit/test_doctest_runner.py::DOCTEST_TARGETS`; add a path to both. |
| `just test-changed` / `just test-changed-run` | the targets a diff can affect | See [Selective testing](../docs/contributing/selective-testing.md). |
| `just test-integration` | `tests/integration/` | Needs `source /opt/ros/jazzy/setup.bash`. |
| `just test-ros-live` | the live-ROS targets in `scripts/ros_live_tests.sh` | Needs `just ros2-build && source install/setup.bash`; see [development.md](../docs/contributing/development.md). |
| `just test-sim` | `tests/sim/ -m sim` | Only files carrying the `sim` marker; real weights / simulators, opt-in. |
| `just hal-twin-sweep` | `tests/sim/` HAL twins + `test_all_hals_via_runner.py` | Every HAL against its MuJoCo twin. |
| `just test-robocasa-sim` | the RoboCasa targets in `scripts/robocasa_sim_tests.sh` | Manual only; no CI lane (see [development.md](../docs/contributing/development.md)). |
| `just hil <name>` | `tests/hil/test_<name>.py` | Nothing collected (exit 4/5) prints `SKIPPED` and exits 0. Also `just hil-so101`, `just hil-openarm-deploy`. |
| `just test-k <kw>` | `pytest -k <kw>` over the whole tree | Ad-hoc filter. |

### CI workflows (`.github/workflows/`)

| Workflow | Triggers | What it runs | Check |
| --- | --- | --- | --- |
| `test-selective.yml` | PR to `master`, `workflow_dispatch` | The pytest targets `tools/select_tests.py` selects; a blast-radius diff runs all of `tests/unit/` as 4 shards + an isolated shard. | `select-and-test` (required) |
| `heavy-lanes.yml` | `workflow_dispatch` only, sent by `heavy-lanes-trigger.yml` | The opt-in dependency lanes (`sim`, `libero`, `robocasa`, …), one matrix job per selected lane, plus the ledger attest. | `heavy-lanes` commit status (required) |
| `heavy-lanes-trigger.yml` | `workflow_run` of `test-selective` / `quality` / `DCO` / `test-ros2`, every 10 min, `workflow_dispatch` | `tools/heavy_lanes_trigger.py`: dispatches the lanes once a PR is up to date, otherwise green, threads resolved and not a draft. | none |
| `quality.yml` | PR to `master`, `workflow_dispatch` | `uv lock --check`, ruff, `tools/run_mypy.sh`, schema / ROS-topic-graph / rSkill `SKILL.md` drift, `openral check`, `mkdocs build --strict`. No tests. | `quality` (required) |
| `dco.yml` | PR (`opened` / `synchronize` / `reopened`) | `Signed-off-by:` on every commit. | `Verify Signed-off-by` (required) |
| `test-ros2.yml` | PR to `master`, `workflow_dispatch` | `colcon build` + `colcon test` in `ros:jazzy-ros-base`; `packages` defaults to `openral_safety_kernel openral_octomap_bridge`. | `colcon build + ctest — Jazzy` |
| `docker-build.yml` | push to `master` (path-filtered), `workflow_dispatch` | Builds `openral:x86`, its smoke tests and the live-ROS suite (the only CI surface with ROS 2 and the colcon overlay for pytest). | none |

`pages.yml`, `release-please.yml` and `release-pypi.yml` run no tests.

### Markers and skips

`pyproject.toml` declares two markers under `--strict-markers`: `sim` (real
weights and/or a simulated robot) and `slow` (>5 s on the reference host).
`asyncio_mode = "strict"`.

Optional dependencies are gated with `pytest.importorskip(...)` or `skipif`,
never faked. In a heavy lane a skip must match a declared capability gap
(`[capability_gaps]` in `tools/test_selection.toml`) or the lane fails; see
[Lane policy](../docs/contributing/selective-testing.md#lane-policy--what-a-skip-is-allowed-to-mean).

### Top-level fixtures

- `tests/conftest.py` pins `NO_COLOR=1` / `TERM=dumb` and patches
  `rich.console.Console.__init__` so ANSI escapes never reach `CliRunner.output`.
- `tests/sim/conftest.py` imports `openral_rskill._lerobot_compat` before any
  sim test imports `lerobot.policies`.
- Process/network-boundary doubles (the only ones allowed, CLAUDE.md §1.11)
  live in `tests/unit/fakes/` and `tests/integration/fakes/`.

---

## 2. Tiers

| Directory | Scope | Budget (CLAUDE.md §2) |
| --- | --- | --- |
| `tests/unit/` | Schemas, CLI, HAL contracts against digital twins / `SimTransport`, runners, tools. | <30 s |
| `tests/integration/` | Real `rclpy` lifecycle nodes, real IDL, DDS graphs; the live-ROS subset is gated by `OPENRAL_TEST_ROS_LIVE=1`. | <5 min |
| `tests/sim/` (+ `tests/sim/safety/`) | Headless rollouts (MuJoCo, LIBERO, RoboCasa, ManiSkill3, Isaac, …) and kernel-in-the-loop safety checks. | <10 min |
| `tests/hil/` | Real hardware; every file skips off-rig. | <10 min |
| `python/<pkg>/tests/`, `packages/<pkg>/test/` | Package-local tests (`colcon test` for ROS packages). | per tier |

**Sim naming.** `test_<robot>_<vla>_<sim>.py` for end-to-end rollouts
(`test_franka_panda_smolvla_libero.py`); HAL-only sim tests use `<vla>=hal`
(`test_ur5e_hal_mujoco.py`).

### `tests/hil/`

No self-hosted runner is registered, so every HIL gate runs by hand on the cell
([Lab HIL runners](../docs/contributing/development.md#lab-hil-runners-self-hosted-lab-robot)).

| File | Hardware |
| --- | --- |
| `test_ur5e.py`, `test_ur10e.py`, `test_franka_panda.py`, `test_sawyer.py` | Vendor ROS 2 drivers through the `_ros_control_transport.py` bridge. None has met a physical arm. |
| `test_aloha.py` | Trossen ALOHA through `_aloha_ros_transport.py`. The bridge mirrors `AlohaHAL`'s topics, which a real ALOHA (`xs_sdk`) does not expose. |
| `test_so101_serial_live.py` | SO-101 on USB serial, non-motion (`just hil-so101`). |
| `test_openarm_can_live.py`, `test_openarm_deploy.py` | OpenArm v2 over CAN; `test_openarm_deploy.py` attaches to a running graph (`just hil-openarm-deploy`). |
| `test_openarm_bringup_agreement.py`, `test_openarm_ros_transport.py` | OpenArm HAL/bringup config agreement and the HIL transport, checked without the cell. |
| `test_openarm_slot_group_motion.py` | Commands the OpenArm; behind a separate attended-bench opt-in. |
| `test_galaxea_a1.py`, `test_galaxea_a1_deploy.py` | Galaxea A1 through its ROS 1 sidecar. |
| `test_detect_jetson_live.py` | Jetson auto-provisioning probe. |

There is no SO-100, G1 or RealSense HIL file.

---

## 3. Rules for adding a test

1. Tests ship in the same PR as the change (CLAUDE.md §1.7); untested
   actuation-path code is rejected.
2. Real manifests from `robots/`, `rskills/`, `scenes/`; no mocks or stubs
   (CLAUDE.md §1.11). An unavailable dependency is a `pytest.skip(reason=...)`.
3. A module flipped to `green` in the
   [repo state map](../docs/architecture/repo-state-map.html) needs source **and**
   tests on disk (CLAUDE.md §4.3).
4. A live-ROS test goes in `tests/integration/`, in `scripts/ros_live_tests.sh`
   `TARGETS` and in `docker-build.yml`'s `paths:`; a RoboCasa-gated sim test goes
   in `scripts/robocasa_sim_tests.sh`. `test_ros_live_targets.py` and
   `test_robocasa_sim_targets.py` fail the unit suite otherwise.
5. A test needing a different optional stack from its file's neighbours gets its
   own file (lanes are assigned per file).
