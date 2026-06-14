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
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "qwen-vlm-sidecar"
_VENV_ENV = "OPENRAL_QWEN_VLM_SIDECAR_VENV"
_HOME_ENV = "OPENRAL_QWEN_VLM_SIDECAR_HOME"

# transformers pin matches the runtime's 5.3.0 (Qwen3.5 support landed in the
# 5.x line); the isolation value here is the bitsandbytes + qwen-vl-utils +
# Gated-DeltaNet-kernel dep surface, plus process/VRAM separation (see module
# docstring). ``fla`` / ``causal-conv1d`` are the optional fast linear-attention
# kernels Qwen3.5 uses; without them transformers falls back to slower PyTorch
# ops but the model still loads, so they are best-effort (installed if the
# wheels resolve on this platform).
_DEPS = (
    "transformers==5.3.0",
    "torch==2.8.0",
    "torchvision==0.23.0",
    "accelerate",
    "bitsandbytes",
    "qwen-vl-utils",
    "pillow",
    "einops",
    "safetensors",
    "pyzmq",
    "msgpack",
)


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run ``cmd`` and raise on non-zero exit."""
    print(f"[qwen-sidecar] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)


def _ensure_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(
            "uv not found on PATH. Install it: "
            "https://docs.astral.sh/uv/getting-started/installation/"
        )
    return uv


def ensure_venv(home: Path, *, override: str | None = None) -> Path:
    """Return the sidecar venv python, creating + populating it if needed.

    ``override`` (or ``$OPENRAL_QWEN_VLM_SIDECAR_VENV``) points at an existing
    venv to reuse instead of provisioning one under ``home`` — handy for
    development against an already-built env.
    """
    override = override or os.environ.get(_VENV_ENV)
    if override:
        py = Path(override) / "bin" / "python"
        if not py.exists():
            raise SystemExit(f"{_VENV_ENV} points at {override} but {py} does not exist")
        return py

    uv = _ensure_uv()
    venv = home / ".venv"
    py = venv / "bin" / "python"
    sentinel = venv / ".deps-installed"
    if py.exists() and sentinel.exists():
        print(f"[qwen-sidecar] reusing venv at {venv}", flush=True)
        return py

    home.mkdir(parents=True, exist_ok=True)
    _run([uv, "venv", str(venv), "--python", "3.12"])
    _run([uv, "pip", "install", "--python", str(py), "--torch-backend=cu128", *_DEPS])
    sentinel.write_text("ok\n")
    return py


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
