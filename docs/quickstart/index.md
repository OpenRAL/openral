# Quickstart

From zero to a working `openral doctor` check in under five minutes.

---

## 1. Install

```bash
git clone https://github.com/OpenRAL/openral && cd openral
just bootstrap          # installs uv, ROS 2, system deps
just sync               # install Python workspace (wraps uv sync + repair)
```

---

## 2. Run `openral doctor`

`openral doctor` reports the status of every host dependency OpenRAL needs.

```bash
uv run openral doctor
```

The CLI lives in the workspace venv at `.venv/bin/openral`: call it with
`uv run openral …`, or `source .venv/bin/activate` and drop the prefix. Global
install: `uv tool install --editable python/cli`.

Example output on a well-configured Ubuntu 24.04 machine with an NVIDIA GPU —
**abridged**: a real run also prints the derived `ComputeSpec (local) / …` rows
and the `Reasoner LLM` row (see below).

```
           openral doctor
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ check              ┃ status  ┃ details                        ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Python             │ ok      │ 3.12.3                         │
│ Platform           │ info    │ Linux 6.8.0-47-generic         │
│ openral-core       │ ok      │ 0.3.1                          │
│ ROS 2 binary       │ ok      │ /opt/ros/jazzy/bin/ros2        │
│ ROS 2 distro       │ ok      │ jazzy                          │
│ RMW                │ info    │ rmw_fastrtps_cpp (default)     │
│ colcon             │ ok      │ /usr/bin/colcon                │
│ GPU 0              │ ok      │ NVIDIA RTX 4090 (24576 MiB)    │
│ USB devices        │ info    │ none found                     │
│ just               │ ok      │ /usr/local/bin/just            │
└────────────────────┴─────────┴────────────────────────────────┘
```

**Status colours:**

| Status | Colour | Meaning |
|--------|--------|---------|
| `ok` | green | Fully present and working |
| `info` | yellow | Informational — not a problem |
| `absent` | yellow | Optional tool not installed |
| `warn` | yellow | Present but something unexpected |
| `missing` | red | Required tool not found |
| `fail` | red | Found but broken |

`openral doctor` exits **0** when no check is `fail` or `missing`; exits **1** otherwise.

---

## 3. Machine-readable output

`--json` prints every table row as a JSON array, for scripting or CI assertions
(sample abridged as above):

```bash
uv run openral doctor --json
```

```json
[
  {"check": "Python",           "status": "ok",   "details": "3.12.3"},
  {"check": "Platform",         "status": "info", "details": "Linux 6.8.0"},
  {"check": "openral-core",     "status": "ok",   "details": "0.3.1"},
  {"check": "ROS 2 binary",     "status": "ok",   "details": "/opt/ros/jazzy/bin/ros2"},
  {"check": "ROS 2 distro",     "status": "ok",   "details": "jazzy"},
  {"check": "RMW",              "status": "info", "details": "rmw_fastrtps_cpp (default)"},
  {"check": "colcon",           "status": "ok",   "details": "/usr/bin/colcon"},
  {"check": "GPU 0",            "status": "ok",   "details": "NVIDIA RTX 4090 (24576 MiB)"},
  {"check": "USB devices",      "status": "info", "details": "none found"},
  {"check": "just",             "status": "ok",   "details": "/usr/local/bin/just"}
]
```

---

## 4. Common issues and fixes

### ROS 2 binary shows `absent`

ROS 2 is opt-in: the one-liner install ships without it, so `absent` is not a
failure and does not change the exit code. To install it:

```bash
openral install ros          # needs sudo + apt; Linux only
```

### ROS 2 distro shows `info`

ROS 2 is installed but not sourced; the row names the distros it found. Fix:

```bash
source /opt/ros/jazzy/setup.bash    # Ubuntu 24.04
# or
source /opt/ros/humble/setup.bash   # Ubuntu 22.04
```

Add to `~/.bashrc` for permanent effect. Then re-run `openral doctor`.

`missing` means no `/opt/ros/*` exists at all: run `openral install ros`
instead of sourcing. It is fatal, so `doctor` exits 1.

### `openral-core` shows `fail`

Run `just sync` from the repo root (`just sync` wraps `uv sync --all-packages` and runs the hf-libero repair script).

### USB devices shows `none found` with a robot connected

Check that your user is in the `dialout` group:

```bash
sudo usermod -aG dialout $USER   # then log out and back in
ls /dev/ttyUSB* /dev/ttyACM*     # should list your device
```

### GPU shows `absent` but you have NVIDIA hardware

Install the NVIDIA drivers and CUDA toolkit. Verify with:

```bash
nvidia-smi
```

### `Reasoner LLM` shows `absent`

Expected until you pick a planner model: the reasoner (S2) has no hidden
default, by design. It matters only for the deploy graph (`openral deploy sim`
/ `openral deploy run`); the row is informational, so `doctor` still exits 0:

```bash
export OPENRAL_REASONER_MODEL=claude-opus-4-8   # or gpt-5.5 / gpt-5.6 / cosmos3-edge
export OPENRAL_REASONER_API_KEY=sk-ant-...      # only where the endpoint needs it
```

Full model / endpoint matrix: [reasoner reference](../reference/reasoner.md#reasoner-model-selection).

---

## 5. Next steps

- [Run your first sim rollout](../tutorials/sim/first-rollout.md) — a real policy driving a simulated robot in one command, no GPU needed.
- [OpenRAL dashboard](dashboard.md) — live debugging UI over the OTel stream.
- [Write & publish an rSkill](../tutorials/rskill/write-and-publish-an-rskill.md) — package a policy as an installable skill.
- [Deploy on a robot](../tutorials/deploy/deploy-run-and-dashboard.md) — the real-hardware graph.
- [Architecture overview](../architecture/overview.md) — the seven layers.
- [Development setup](../contributing/development.md) — a full dev environment.
