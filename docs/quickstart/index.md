# Quickstart

This page gets you from zero to a working `openral doctor` check in under five minutes.

---

## 1. Install

```bash
git clone https://github.com/OpenRAL/openral && cd OpenRAL
just bootstrap          # installs uv, ROS 2, system deps
uv sync --all-packages  # install Python workspace
```

---

## 2. Run `openral doctor`

`openral doctor` checks your host environment and reports the status of every dependency OpenRAL needs.

```bash
openral doctor
```

Example output on a well-configured Ubuntu 24.04 machine with an NVIDIA GPU:

```
           openral doctor
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ check              ┃ status  ┃ details                        ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Python             │ ok      │ 3.12.3                         │
│ Platform           │ info    │ Linux 6.8.0-47-generic         │
│ openral-core   │ ok      │ 0.1.0                          │
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

Pass `--json` to get a JSON array — useful for scripting or CI assertions:

```bash
openral doctor --json
```

```json
[
  {"check": "Python",           "status": "ok",   "details": "3.12.3"},
  {"check": "Platform",         "status": "info", "details": "Linux 6.8.0"},
  {"check": "openral-core", "status": "ok",   "details": "0.1.0"},
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

### ROS 2 distro shows `missing` or `info`

You have not sourced the ROS 2 environment. Fix:

```bash
source /opt/ros/jazzy/setup.bash    # Ubuntu 24.04
# or
source /opt/ros/humble/setup.bash   # Ubuntu 22.04
```

Add to `~/.bashrc` for permanent effect. Then re-run `openral doctor`.

### `openral-core` shows `fail`

Run `uv sync --all-packages` from the repo root. Plain `uv sync` does not install workspace member packages.

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

---

## 5. Next steps

- [OpenRAL dashboard](dashboard.md) — live debugging UI over the OTel stream.
- [Architecture overview](../architecture/overview.md) — understand the eight-layer architecture.
- [Development setup](../contributing/development.md) — set up a full dev environment.
