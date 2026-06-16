"""Boot the Qwen3.5-4B scene-VLM inference server in an isolated sidecar venv.

The ``query_scene`` reasoner tool (ADR-0047) asks a vision-language model
open-ended questions about the current camera view ("has the robot grasped the
mug?", "is the task complete?"). The model is ``Qwen/Qwen3.5-4B``, loaded NF4
via bitsandbytes, and is run **out-of-process** for three reasons:

* **Dependency isolation.** The openral runtime venv hard-pins
  ``transformers==5.3.0`` for lerobot (every VLA adapter aliases that pin —
  see ``pyproject.toml``). The VLM needs ``bitsandbytes`` (NF4), ``qwen-vl-utils``
  (vision preprocessing), and the optional Gated DeltaNet kernels
  (``fla`` / ``causal-conv1d``). Resolving those into the lerobot-pinned env
  risks perturbing the VLA stack; a separate venv keeps the resolver clean.
* **VRAM / process isolation.** A 4B model + CUDA context should not live in
  the ``rclpy`` reasoner process. On an 8 GB GPU a model OOM must not take
  down the reasoner; the sidecar owns its own VRAM lifecycle and can be torn
  down independently.
* **Same pattern as the rest of the tree.** ``tools/locateanything_sidecar.py``
  (LocateAnything detector) and ``tools/gr00t_sidecar.py`` (GR00T) already run
  models out-of-process over ZMQ REQ/REP + msgpack. This is that pattern.

The openral side is
:class:`openral_runner.backends.gstreamer.qwen_scene_vlm.QwenSceneVlm`, which
auto-spawns this sidecar on first use and talks to ``_qwen_vlm_server.py`` over
ZMQ.

Usage::

    python tools/qwen_vlm_sidecar.py --port 5759

The script blocks and forwards signals; SIGINT cleanly stops the server.

CLAUDE.md compliance:
* Real subprocess running real upstream model code — no mocks (§1.11). The
  openral-side wire protocol is a real ZMQ client.
* ``Qwen/Qwen3.5-4B`` is Apache-2.0 (commercial OK) — no license guard needed
  here (the ``RSkillManifest`` loader handles posture, ADR-0047 / ADR-0012).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openral_sim._sidecar_common import ensure_pip_venv, run_cmd

_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "qwen-vlm-sidecar"
_VENV_ENV = "OPENRAL_QWEN_VLM_SIDECAR_VENV"
_HOME_ENV = "OPENRAL_QWEN_VLM_SIDECAR_HOME"

# Fully-pinned, hash-locked deps. transformers==5.3.0 matches the runtime pin
# (Qwen3.5 support landed in the 5.x line); torch/torchvision resolve to +cu128.
# ``fla`` / ``causal-conv1d`` (optional Gated-DeltaNet kernels) are intentionally
# NOT in the lock — without them transformers falls back to slower PyTorch ops
# but the model still loads, and their wheels do not resolve on every platform.
# Regenerate after editing the .in source with:
#   uv pip compile tools/sidecar_requirements/qwen_vlm.in \
#     --universal --torch-backend=cu128 --generate-hashes --python-version 3.12 \
#     -o tools/sidecar_requirements/qwen_vlm.lock
_LOCK = Path(__file__).resolve().parent / "sidecar_requirements" / "qwen_vlm.lock"


def ensure_venv(home: Path, *, override: str | None = None) -> Path:
    """Return the sidecar venv python, creating + populating it if needed.

    ``override`` (or ``$OPENRAL_QWEN_VLM_SIDECAR_VENV``) points at an existing
    venv to reuse instead of provisioning one under ``home`` — handy for
    development against an already-built env. Otherwise a Python 3.12 venv is
    provisioned from the hash-locked ``qwen_vlm.lock`` for reproducibility
    (CLAUDE.md §1.8).
    """

    def _install(uv: str, py: Path) -> None:
        # ``-r <lock>`` installs the pinned, hash-bearing lock (uv verifies the
        # recorded hashes); we deliberately do NOT pass ``--require-hashes`` —
        # the cu128 torch wheels surface a marker-only transitive (torchcodec)
        # that uv's compile drops, which --require-hashes rejects even though it
        # is never installed on this platform. Pinned versions still give the
        # reproducibility we want (CLAUDE.md §1.8).
        run_cmd(
            "qwen-sidecar",
            [uv, "pip", "install", "--python", str(py), "--torch-backend=cu128", "-r", str(_LOCK)],
        )

    return ensure_pip_venv(
        label="qwen-sidecar",
        home=home,
        python="3.12",
        install=_install,
        override=override,
        override_env=_VENV_ENV,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5759)
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument(
        "--home",
        type=Path,
        default=Path(os.environ.get(_HOME_ENV, _DEFAULT_HOME)),
        help=f"Sidecar work directory (default {_DEFAULT_HOME}).",
    )
    p.add_argument("--venv", default=None, help=f"Reuse this venv (or set {_VENV_ENV}).")
    args = p.parse_args()

    py = ensure_venv(args.home, override=args.venv)
    server = Path(__file__).resolve().parent / "_qwen_vlm_server.py"

    env = os.environ.copy()
    # Drop PYTHONPATH/PYTHONHOME so the sidecar interpreter boots from its own
    # site-packages — ROS 2 / colcon populate PYTHONPATH with the workspace
    # wheels, which would shadow the sidecar's pinned deps (same failure mode
    # tools/locateanything_sidecar.py documents).
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)

    cmd = [
        str(py),
        str(server),
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-side",
        str(args.max_side),
    ]
    print(f"[qwen-sidecar] launching server: model={args.model} port={args.port}", flush=True)
    os.execvpe(str(py), cmd, env)


if __name__ == "__main__":
    sys.exit(main() or 0)
