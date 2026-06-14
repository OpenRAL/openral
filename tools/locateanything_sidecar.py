"""Boot the LocateAnything-3B inference server in an isolated sidecar venv.

``nvidia/LocateAnything-3B`` ships ``trust_remote_code`` modeling files written
against ``transformers==4.57.1`` (see the model card / the nvidia/LocateAnything
Space ``requirements.txt``). The openral runtime is ``transformers>=5`` and
Python 3.12-only (CLAUDE.md §3); transformers 5.x removed/renamed the APIs the
model's custom code calls (``config.rope_theta``, the GenerationMixin
inheritance, ``_check_and_adjust_attn_implementation`` signature), so the model
cannot load in the runtime venv. We therefore run it out-of-process in its own
venv and talk to it from the
:class:`openral_runner.backends.gstreamer.locateanything_detector.LocateAnythingDetector`
backend over ZMQ REQ/REP + msgpack — the same pattern as
:mod:`tools.rldx_sidecar`.

Unlike RLDX-1, there is no upstream repo to clone: the model is custom-code on
the Hub, so the sidecar only needs a venv with the pinned dependencies plus the
thin server in :mod:`tools._locateanything_server`.

Usage::

    python tools/locateanything_sidecar.py --port 5757

The script blocks and forwards signals; SIGINT cleanly stops the server.

CLAUDE.md compliance:
* The sidecar is a real subprocess running real upstream model code — no mocks
  (§1.11). The openral-side wire protocol is a real ZMQ client.
* Version isolation is the only safe bridge between transformers 4.57.1
  (LocateAnything) and transformers 5.x (openral runtime); see §3.
* ``nvidia/LocateAnything-3B`` weights are NVIDIA non-commercial; the license
  guard is enforced upstream in the ``RSkillManifest`` loader, not here.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "locateanything-sidecar"
_VENV_ENV = "OPENRAL_LOCATEANYTHING_SIDECAR_VENV"
_HOME_ENV = "OPENRAL_LOCATEANYTHING_SIDECAR_HOME"

# Pins mirror the nvidia/LocateAnything Space requirements.txt (transformers
# 4.57.1 is the load-bearing pin; the rest are its known-good companions) plus
# the ZMQ/msgpack wire deps the server needs.
_DEPS = (
    "transformers==4.57.1",
    "torch==2.8.0",
    "torchvision==0.23.0",
    "accelerate",
    "bitsandbytes",
    "peft",
    "decord==0.6.0",
    "lmdb",
    "opencv-python-headless==4.11.0.86",
    "einops",
    "safetensors",
    "pillow",
    "pyzmq",
    "msgpack",
)


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run ``cmd`` and raise on non-zero exit."""
    print(f"[la-sidecar] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
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

    ``override`` (or ``$OPENRAL_LOCATEANYTHING_SIDECAR_VENV``) points at an
    existing venv to reuse instead of provisioning one under ``home`` — handy
    for development against an already-built ``transformers==4.57.1`` env.
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
        print(f"[la-sidecar] reusing venv at {venv}", flush=True)
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
    p.add_argument("--model", default="nvidia/LocateAnything-3B")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5757)
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
    server = Path(__file__).resolve().parent / "_locateanything_server.py"

    env = os.environ.copy()
    # Drop PYTHONPATH/PYTHONHOME so the sidecar interpreter boots from its own
    # site-packages — ROS 2 / colcon populate PYTHONPATH with the workspace's
    # transformers 5.x wheels, which would shadow the sidecar's 4.57.1 and
    # crash the import (same failure mode tools/rldx_sidecar.py documents).
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # The custom modeling files execute on load; the operator opted in by
    # launching this sidecar for this specific model.
    env.setdefault("OPENRAL_ALLOW_REMOTE_CODE", "1")

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
    print(f"[la-sidecar] launching server: model={args.model} port={args.port}", flush=True)
    os.execvpe(str(py), cmd, env)


if __name__ == "__main__":
    sys.exit(main() or 0)
