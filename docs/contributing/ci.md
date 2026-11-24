# CI overview

What runs on a pull request, what blocks the merge, and how the slow suites (heavy lanes, live-ROS, RoboCasa, HIL) are scheduled. Workflow definitions live in [`.github/workflows/`](https://github.com/OpenRAL/openral/tree/master/.github/workflows); which tests a diff selects is in [Selective testing](selective-testing.md).

## Workflows

| Workflow | Runs on | What it does |
| --- | --- | --- |
| `quality` | every PR | `uv lock --check`, ruff, mypy `--strict`, schema drift, `mkdocs build --strict`. Under 3 min. |
| `test-selective` | every push | Selects the pytest targets your diff touches and runs them (`select-and-test`). |
| `DCO` | every PR | `Verify Signed-off-by`: every commit must carry a `Signed-off-by` line. |
| `test-ros2` | every PR | `colcon build` + ctest for the C++ / ROS 2 subset (the safety kernel and its dependencies) that the Python selector skips. |
| `heavy-lanes` | dispatched by `heavy-lanes-trigger` | One job per opt-in dependency group (LIBERO, RoboCasa, GR00T, ...). |
| `docker-build` | push to `master` (path-filtered), manual | Builds `openral:x86`, runs the live-ROS suite in it, pushes to GHCR. |
| `pages` | push to `master` | Deploys MkDocs to docs.openral.com. |
| `cache-warm` | `master` when `uv.lock`, `robots/**` or the test setup changes | Refills the uv and robot-asset caches that PR runs restore. |
| `reasoner-models` | `master` when `REASONER_MODELS` changes | Exports the curated model list for the website. |
| `release-please`, `release-pypi` | `master` / tags | See [Releasing](releasing.md). |

Merging needs `quality`, `select-and-test`, `heavy-lanes` and `Verify Signed-off-by` green and every review thread resolved. Stale runs on the same branch are cancelled when you push again.

## Review policy

Every change reaches `master` through a PR that the maintainer reviews and merges. The branch ruleset requires a PR, the four green checks and resolved threads, but **0 approving reviews**: GitHub does not let an author approve their own PR, so a required approval would force a ruleset bypass on every merge. Only `OrganizationAdmin` can bypass it.

When a second account gets write access, set required approvals back to 1 (with code-owner review) the same day, or that account could merge its own PRs unreviewed.

Fork PRs run CI like any other PR, except a brand-new GitHub account's first PR, which waits for a maintainer to click "Approve and run".

## Job graph

```
test-selective.yml (every push):
select ──┬── core_full     (4 file-shards + isolated)   ──┐
         └── core_selected (6 target-shards + isolated) ──┴── select-and-test (required)

heavy-lanes-trigger.yml (after PR workflows finish, and every 10 min):
tools/heavy_lanes_trigger.py ── workflow_dispatch, once the PR is ready ──┐
                                                                          ▼
heavy-lanes.yml: lanes-select ── lane (one job per dependency group) ── heavy-lanes (required)
```

Design points:

- `select` runs `select_tests.py` in an ephemeral `uv run --isolated --with pydantic` env, so selection costs seconds.
- `core_full`, `core_selected` and `lane` are skipped when nothing is selected; a docs-only PR does no test work.
- A shard with nothing to run exits before setup. Fork-isolated targets get their own shard.
- Lanes run in parallel, so wall time is the slowest lane, not the sum. The pure-CPU `lowering` lane runs round-robin over 4 processes.
- Only `master` saves the uv and asset caches (`cache-warm.yml`); PR runs restore them.

### When heavy lanes start

`heavy-lanes` starts by itself, with no label or approval, once the PR head is:

- 0 commits behind `master` (rebase or merge `master` in);
- green on every other check (`select-and-test`, `quality` and `Verify Signed-off-by` have reported);
- free of unresolved review threads;
- not a draft.

[`tools/heavy_lanes_trigger.py`](https://github.com/OpenRAL/openral/blob/master/tools/heavy_lanes_trigger.py) re-checks each open PR whenever a PR workflow finishes and every 10 minutes (GitHub has no event for "thread resolved" or "`master` moved", so those can take up to 10 minutes to be noticed). It dispatches once per head commit; a new push waits for the conditions again.

Progress shows as the `heavy-lanes` commit status: pending with what it waits for ("Waiting: 2 commit(s) behind master"), pending "Running", then passed or failed with a link to the run. It is a commit status because a `workflow_dispatch` run's check runs are not linked to a PR. Re-run a failed lane from its Actions run page.

**Fork PRs** get the lanes too. `workflow_dispatch` cannot target a fork branch and a fork's copy of the workflow is untrusted, so the trigger points a `heavy-lanes/fork-<pr>` branch at a copy of `master`'s tree and dispatches there with the fork head as the commit to test. The workflow file is always `master`'s, the lane jobs run with a read-only token and no secrets, the branch has its own cache scope, and the job that writes statuses never checks out fork code. The branch is deleted when the PR closes.

## `docker-build` and the live-ROS suite

`docker-build` builds the open GStreamer-free deploy image `openral:x86` from `docker/inference/Dockerfile.x86` (about 15-20 min). It is **not** a PR check. On push to `master` it runs only when a build input changes: `docker/inference/**`, `pyproject.toml`, `uv.lock`, `packages/**`, `cpp/**`, `python/**`, or one of the live-ROS test files listed in its `paths:` filter. Run it against your branch with `gh workflow run docker-build.yml --ref <branch>`; a manual run never pushes the image.

The job then commits a Fast-DDS-enabled copy as `openral:x86-live` and runs the live-ROS suite in it. These tests (reasoner dispatch, Tier-C critic producer, HAL sim-sensor `/tf` guard, kernel bands, Nav2 scan filter, ...) need a real DDS graph and the colcon overlay, so they are gated behind `OPENRAL_TEST_ROS_LIVE=1`. `docker-build` is the only CI surface with a real ROS 2 install: the `test-selective` runner has none, so anything not run here would `importorskip` silently.

| Caller | Environment |
| --- | --- |
| `just test-ros-live` | Your host, after `source /opt/ros/jazzy/setup.bash && just ros2-build && source install/setup.bash`. No GPU needed. |
| `docker-build`, "Live ROS tests" step | `openral:x86-live`, one container per shard, each on its own Docker network (DDS discovery would otherwise cross containers). |

Both run [`scripts/ros_live_tests.sh`](https://github.com/OpenRAL/openral/blob/master/scripts/ros_live_tests.sh), which owns the target list. Narrow a local run with `just test-ros-live -k <expr>`; `OPENRAL_LIVE_SHARD=2/4 just test-ros-live` reproduces CI shard 2 of 4. Shards are balanced from wall times in `scripts/ros_live_durations.txt`. A test that asserts a wall-clock bound carries `@pytest.mark.wall_clock`; shards deselect it and `OPENRAL_LIVE_SHARD=quiet` runs it alone afterwards.

To add a live-ROS test:

1. Put it under `tests/integration/` (a live test under `tests/unit/` would run on no lane).
2. Add the file to `TARGETS` in `scripts/ros_live_tests.sh`.
3. Add it to the `paths:` filter in `.github/workflows/docker-build.yml`, so a diff touching only that test still triggers the run.

`tests/unit/test_ros_live_targets.py` fails if a gated file under `tests/integration/` is missing from `TARGETS`.

If a run prints `DDS threads still alive at session end`, a test leaked an rclpy node or context. Fix that test's teardown; the repo-root `conftest.py` already shuts rclpy down at session end.

Tests that assert a VLA reaches the reasoner's palette use the `require_policy_runtime` fixture (`tests/integration/conftest.py`). Locally they skip, naming the missing `just sync --group`; CI sets `OPENRAL_TEST_REQUIRE_POLICY_RUNTIMES=1`, which turns that skip into a failure.

## The RoboCasa sim suite (manual)

The RoboCasa-gated tests under `tests/sim/` need the colcon overlay (they spawn the real `safety_kernel_node`), MuJoCo and a provisioned RoboCasa backend with about 23 GB of assets. No CI runner has that disk and a self-hosted runner is not safe on this public repo (below), so the suite is manual. `scripts/robocasa_sim_tests.sh` holds the target list; `tests/unit/test_robocasa_sim_targets.py` fails if a RoboCasa-gated file is missing from it.

```bash
source /opt/ros/jazzy/setup.bash && just ros2-build \
  && source install/setup.bash && just test-robocasa-sim
```

If the venv lacks `robocasa` after `just sync`, provision it:

```bash
OPENRAL_AUTO_INSTALL_DEPS=1 uv run --no-sync python -c \
  "from openral_sim._deps import ensure_backend_deps; ensure_backend_deps('robocasa_kitchen')"
```

Run it before merging anything that touches the kernel, the `panda_mobile` manifest, the HAL sim bridge or a `scenes/deploy/robocasa_*.yaml` pin. On a small machine narrow it with `-k` (a kitchen compose is memory-hungry; extra args are appended, so a path adds to the target list rather than selecting it).

## Lab HIL runners: none registered

No self-hosted runner is registered for this repository, and none should be until the blocker below clears. The `[self-hosted, lab-<robot>]` labels in `tests/hil/` docstrings say where a gate would run, not a lane that exists. HIL runs by hand on the cell; every HIL file gates on an env var plus a live hardware probe and skips (never fails) off-rig.

**Why not.** A runner label is a routing request made by a workflow, not an access control. For `pull_request`, GitHub runs the workflow definition from the fork's ref, and this public repo has fork-reachable `pull_request` workflows (`dco`, `quality`, `test-ros2`, `test-selective`). A fork PR could add `runs-on: [self-hosted, lab-openarm]` with arbitrary steps and get code execution on the lab host: its SSH keys, `gh` credentials, LAN access to the robots and, for `lab-openarm`, the CAN buses of a powered arm. Safe triggers on the protected workflow do not help; the exposed asset is the runner.

Registering one needs **either** an organisation runner group restricted to selected repositories *and* selected workflows (an Enterprise feature; this org is on GitHub Free), **or** HIL moved to a private mirror repo with no fork-reachable triggers, dispatched only by `workflow_dispatch` / `repository_dispatch`. Neither exists.

If one is ever registered, the lane must be dispatch-only with a `timeout-minutes` at or under the 10-minute HIL budget, and the runner service must not run as root (a job that can write CAN as root can also edit the udev rules that keep left and right buses from swapping).

!!! warning "A `lab-openarm` lane must never start the OpenArm graph unattended"
    `just hil-openarm-deploy` is non-motion and attaches to a graph someone else started. Starting `openral deploy run` on real OpenArm hardware is itself a motion event: `OpenArmHW::on_activate` enables all motors and sends an unramped position command to zero on every joint. A CI lane must never launch it, and `tests/hil/test_openarm_slot_group_motion.py` must keep its `OPENRAL_OPENARM_ATTENDED=1` gate unexported on any runner host.
