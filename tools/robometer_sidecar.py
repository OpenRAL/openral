"""Provision + launch the Robometer reward-monitor sidecar (ADR-0057).

The Robometer reward model (``robometer/Robometer-4B``, an NF4-quantized
Qwen3-VL-4B reward foundation model) runs out-of-process in its own venv because
it cannot be loaded by vanilla ``transformers.AutoModel`` (its HF ``config.json``
advertises ``architectures: ["RFM"]`` with no ``auto_map``) — it requires the
upstream ``robometer`` package, pinned, with ``transformers==4.57.1`` (5.x breaks
its processor kwargs). This wrapper builds/reuses that venv and execs
``_robometer_server.py`` inside it.

The node-side client is :class:`openral_runner.backends.reward.robometer_reward.RobometerReward`,
which auto-spawns this script. Provision the venv ahead of time and point at it
with ``$OPENRAL_ROBOMETER_SIDECAR_VENV`` to skip the first-launch install.

Trust note (CLAUDE.md §3): ``robometer`` is not an OpenRAL-trusted org. It is
pinned by commit and runs only in this isolated venv, never the main repo env.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "robometer-sidecar"
_VENV_ENV = "OPENRAL_ROBOMETER_SIDECAR_VENV"
_HOME_ENV = "OPENRAL_ROBOMETER_SIDECAR_HOME"

# Pinned upstream robometer commit (ADR-0057 Phase 0). The package pulls torch
# 2.8 / decord / qwen-vl-utils / bitsandbytes; we then FORCE transformers back to
# 4.57.1 (the resolver pulls 5.x, which drops `input_ids` from the processor).
_ROBOMETER_PIN = "a669dffc241d7d76bec12f36efd4084d914d017c"
_ROBOMETER_SPEC = (
    f"robometer[robometer,quantization] @ "
    f"git+https://github.com/robometer/robometer@{_ROBOMETER_PIN}"
)
_TRANSFORMERS_PIN = "transformers==4.57.1"
_EXTRA_DEPS = ("pyzmq", "msgpack")


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"[robometer-sidecar] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
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
    """Return the sidecar venv python, creating + populating it if needed."""
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
        print(f"[robometer-sidecar] reusing venv at {venv}", flush=True)
        return py

    home.mkdir(parents=True, exist_ok=True)
    _run([uv, "venv", str(venv), "--python", "3.12"])
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(py),
            "--torch-backend=cu128",
            _ROBOMETER_SPEC,
            *_EXTRA_DEPS,
        ]
    )
    # Force the transformers pin AFTER robometer (its resolver pulls 5.x).
    _run([uv, "pip", "install", "--python", str(py), _TRANSFORMERS_PIN])
    sentinel.write_text("ok\n")
    return py


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--weights", default="robometer/Robometer-4B")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5769)
    p.add_argument(
        "--home",
        type=Path,
        default=Path(os.environ.get(_HOME_ENV, _DEFAULT_HOME)),
        help=f"Sidecar work directory (default {_DEFAULT_HOME}).",
    )
    p.add_argument("--venv", default=None, help=f"Reuse this venv (or set {_VENV_ENV}).")
    args = p.parse_args()

    py = ensure_venv(args.home, override=args.venv)
    server = Path(__file__).resolve().parent / "_robometer_server.py"

    env = os.environ.copy()
    # Drop PYTHONPATH/PYTHONHOME so the sidecar boots from its own site-packages
    # (ROS 2 / colcon populate PYTHONPATH and would shadow the pinned deps).
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # Empirically required to fit NF4 + the forward in 8 GB (ADR-0058 Phase 2).
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        str(py),
        str(server),
        "--weights",
        args.weights,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print(
        f"[robometer-sidecar] launching server: weights={args.weights} port={args.port}",
        flush=True,
    )
    os.execvpe(str(py), cmd, env)


if __name__ == "__main__":
    raise SystemExit(main())
