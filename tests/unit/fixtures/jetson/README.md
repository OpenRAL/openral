# Jetson detection fixtures

Recorded real-device output for `_probe_jetson()` unit tests
(ADR-0013 PR 2/3). No mocks per CLAUDE.md §1.11 — each `model` file
is the contents of `/proc/device-tree/model` on a real board, and
each `nv_tegra_release` file is the first line of `/etc/nv_tegra_release`
from a real JetPack install.

| Directory       | Board                         | Real model string                                | JetPack | CC    |
|-----------------|-------------------------------|--------------------------------------------------|---------|-------|
| `orin_agx/`     | Jetson AGX Orin Developer Kit | `NVIDIA Jetson AGX Orin Developer Kit`           | r36.4   | 8.7   |
| `orin_nx/`      | Jetson Orin NX Dev Kit        | `NVIDIA Jetson Orin NX Engineering Reference …`  | r36.4   | 8.7   |
| `orin_nano/`    | Jetson Orin Nano Dev Kit      | `NVIDIA Orin Nano Developer Kit`                 | r36.4   | 8.7   |
| `xavier_agx/`   | Jetson AGX Xavier Dev Kit     | `Nvidia Jetson AGX Xavier Developer Kit`         | r35.4   | 7.2   |
| `xavier_nx/`    | Jetson Xavier NX Dev Kit      | `NVIDIA Jetson Xavier NX Developer Kit`          | r35.4   | 7.2   |
| `maxwell_nano/` | Jetson Nano Dev Kit (Maxwell) | `NVIDIA Jetson Nano Developer Kit`               | r32.7.4 | 5.3   |

The trailing `\0` byte that the real device-tree shim appends is
re-added by `_probe_jetson` via `.strip("\x00 \n")`; the fixtures
omit it for portability across editors.

When a new Jetson generation lands (Thor, Spark, …) add a directory
under this tree with a fresh real-device capture before extending
`_JETSON_CC_BY_BOARD_KEYWORD` in `python/detect/src/openral_detect/probes/gpu.py`.
