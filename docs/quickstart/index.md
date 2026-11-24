# Quickstart

Goal: install OpenRAL from a clone and confirm the host with `openral doctor`.

## Prerequisites

- Linux (Ubuntu 24.04 recommended) with `sudo`; [`just`](https://github.com/casey/just) installed. macOS works for development only.
- An NVIDIA GPU is optional. The first sim rollout runs on CPU.

## 1. Install

```bash
git clone https://github.com/OpenRAL/openral && cd openral
just bootstrap          # uv, ROS 2 Jazzy, system deps (the slow step)
just sync               # Python workspace; always `just sync`, never bare `uv sync`
```

Simulators are opt-in: `just sync --group sim` (or `--group libero`, `--group robocasa`, `--group metaworld`, `--group maniskill3`). LIBERO and RoboCasa groups exclude each other. Rules: [toolchain](../contributing/toolchain.md#managing-the-python-environment-dependency-groups).

You should see: `just sync` ends without errors and `.venv/` exists.

## 2. Run `openral doctor`

```bash
uv run openral doctor
```

The CLI lives in `.venv/bin/openral`. Call it with `uv run openral ...`, or `source .venv/bin/activate` and drop the prefix. Global install: `uv tool install --editable python/cli`.

You should see a table like this (excerpt; a real run adds `ComputeSpec` rows and a `Reasoner LLM` row, and every value is that machine's):

```
┃ check              ┃ status  ┃ details                              ┃
│ Python             │ ok      │ 3.12.3                               │
│ openral-core       │ ok      │ 0.4.0                                │
│ ROS 2 distro       │ ok      │ jazzy                                │
│ RMW                │ info    │ rmw_fastrtps_cpp (default)           │
│ GPU 0              │ ok      │ NVIDIA RTX 4090 (24576 MiB, sm_89)   │
```

The `RMW` row shows the ROS default when `RMW_IMPLEMENTATION` is unset; the dev and inference images set Cyclone DDS. On Jetson or Apple Silicon, `doctor` prints `Jetson` or `GPU info` rows instead of an NVIDIA GPU row.

| Status | Meaning |
|---|---|
| `ok` | present and working |
| `info` | informational, not a problem |
| `absent` | optional tool not installed |
| `warn` | present but unexpected |
| `missing` | required tool not found (fatal) |
| `fail` | found but broken (fatal) |

`doctor` exits 0 unless a check is `fail` or `missing`. `uv run openral doctor --json` prints every row as a JSON array (`check`, `status`, `details`) for scripts and CI.

## Other install paths

| You want | Do this |
|---|---|
| The CLI only, no clone, no sudo (CPU torch, ~1.6 GB) | `curl -fsSL https://raw.githubusercontent.com/OpenRAL/openral/master/scripts/install.sh \| bash`, then `openral doctor`. Opt-in extras: `openral install sim`, `openral install ros` (needs sudo), `openral install list`. On an NVIDIA host prefix the install line with `OPENRAL_TORCH_BACKEND=auto`. |
| A real-hardware deploy image (x86_64, NVIDIA driver >= 580.65) | `just docker-build-x86`, then `docker run --rm --gpus all openral:x86 doctor`. Mount your Hugging Face cache (`-v ~/.cache/huggingface:/opt/openral/hf-cache`) or every `--rm` run re-downloads the rSkill weights. The image holds the deploy graph, not the sim suites. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ROS 2 binary` is `absent` | ROS is opt-in with the one-liner install. Run `openral install ros` (sudo + apt, Linux only). |
| `ROS 2 distro` is `info` | ROS is installed but not sourced. `source /opt/ros/jazzy/setup.bash` (24.04) or `/opt/ros/humble/setup.bash` (22.04), and add it to `~/.bashrc`. |
| `ROS 2 distro` is `missing` | No `/opt/ros/*` exists. Run `openral install ros`. |
| `openral-core` is `fail` | Run `just sync` from the repo root. |
| `USB devices` is `none found` with a robot attached | Add yourself to `dialout`: `sudo usermod -aG dialout $USER`, log out and in; then `ls /dev/ttyUSB* /dev/ttyACM*`. |
| `GPU 0` is `absent` on NVIDIA hardware | Install the NVIDIA driver and check `nvidia-smi`. |
| `Reasoner LLM` is `absent` | Expected until you pick a planner; the reasoner has no hidden default. It matters only for `openral deploy`. The row is informational. |

To pick a planner, set `OPENRAL_REASONER_MODEL` (for example `claude-opus-4-8`, `gpt-5.5` or `cosmos3-edge`) and, where the endpoint needs one, `OPENRAL_REASONER_API_KEY`. The model and endpoint matrix is in the [reasoner reference](../reference/reasoner.md).

## Next

- [Your first sim rollout](../tutorials/sim/first-rollout.md): a real policy driving a simulated robot, no GPU.
- [Dashboard](dashboard.md): the robot's live page.
- [Deploy on a robot](../tutorials/deploy/deploy-run-and-dashboard.md): the real-hardware graph.
- [Write & publish an rSkill](../tutorials/rskill/write-and-publish-an-rskill.md).
- [Architecture overview](../architecture/overview.md) and [development setup](../contributing/development.md).
