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
import sys
from pathlib import Path

from openral_sim._sidecar_common import ensure_pip_venv, run_cmd

_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "locateanything-sidecar"
_VENV_ENV = "OPENRAL_LOCATEANYTHING_SIDECAR_VENV"
_HOME_ENV = "OPENRAL_LOCATEANYTHING_SIDECAR_HOME"

# Fully-pinned, hash-locked deps (transformers==4.57.1 is the load-bearing pin;
# torch/torchvision resolve to the +cu128 wheels). Regenerate after editing the
# .in source with:
#   uv pip compile tools/sidecar_requirements/locateanything.in \
#     --universal --torch-backend=cu128 --generate-hashes --python-version 3.12 \
#     -o tools/sidecar_requirements/locateanything.lock
_LOCK = Path(__file__).resolve().parent / "sidecar_requirements" / "locateanything.lock"


def ensure_venv(home: Path, *, override: str | None = None) -> Path:
    """Return the sidecar venv python, creating + populating it if needed.

    ``override`` (or ``$OPENRAL_LOCATEANYTHING_SIDECAR_VENV``) points at an
    existing venv to reuse instead of provisioning one under ``home`` — handy
    for development against an already-built ``transformers==4.57.1`` env.
    Otherwise a Python 3.12 venv is provisioned from the hash-locked
    ``locateanything.lock`` so the env is reproducible (CLAUDE.md §1.8).
    """

    def _install(uv: str, py: Path) -> None:
        # ``-r <lock>`` installs the pinned, hash-bearing lock (uv verifies the
        # recorded hashes); we deliberately do NOT pass ``--require-hashes`` —
        # the cu128 torch wheels surface a marker-only transitive (torchcodec)
        # that uv's compile drops, which --require-hashes rejects even though it
        # is never installed on this platform. Pinned versions still give the
        # reproducibility we want (CLAUDE.md §1.8).
        run_cmd(
            "la-sidecar",
            [uv, "pip", "install", "--python", str(py), "--torch-backend=cu128", "-r", str(_LOCK)],
        )

    return ensure_pip_venv(
        label="la-sidecar",
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
