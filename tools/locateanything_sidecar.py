"""Boot the LocateAnything-3B inference server in an isolated sidecar venv.

``nvidia/LocateAnything-3B`` ships ``trust_remote_code`` modeling files pinned
to ``transformers==4.57.1``; the openral runtime is ``transformers>=5``
(Python 3.12-only, CLAUDE.md §3), which removed/renamed APIs the model's
custom code calls (``config.rope_theta``, GenerationMixin inheritance,
``_check_and_adjust_attn_implementation``). It therefore runs out-of-process,
talking to
:class:`openral_runner.backends.gstreamer.locateanything_detector.LocateAnythingDetector`
over ZMQ REQ/REP + msgpack (same pattern as :mod:`tools.rldx_sidecar`). No
upstream repo to clone — the model is custom-code on the Hub — so the sidecar
is just the pinned venv plus the thin server in
:mod:`tools._locateanything_server`.

Usage::

    python tools/locateanything_sidecar.py --port 5757

Real subprocess, no mocks (§1.11); version isolation bridges transformers
4.57.1 and 5.x (§3). Weights are NVIDIA non-commercial; the license guard is
enforced upstream in the ``RSkillManifest`` loader, not here.
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
#     --overrides tools/sidecar_requirements/aarch64-nvrtc-override.txt \
#     -o tools/sidecar_requirements/locateanything.lock
_LOCK = Path(__file__).resolve().parent / "sidecar_requirements" / "locateanything.lock"
# Raises nvrtc past the sm_121 cliff on aarch64 (see the file's header). Passed
# at *install* time as well as compile time: torch's own metadata pins
# ``nvidia-cuda-nvrtc-cu12==12.8.93``, so without the override uv rejects the
# lock's aarch64 line as a conflict instead of honouring it.
_NVRTC_OVERRIDE = (
    Path(__file__).resolve().parent / "sidecar_requirements" / "aarch64-nvrtc-override.txt"
)


def ensure_venv(home: Path, *, override: str | None = None) -> Path:
    """Return the sidecar venv python, creating + populating it if needed.

    ``override`` (or ``$OPENRAL_LOCATEANYTHING_SIDECAR_VENV``) reuses an
    existing venv instead of provisioning one under ``home``. Otherwise
    provisions Python 3.12 from the hash-locked ``locateanything.lock``
    (CLAUDE.md §1.8).
    """

    def _install(uv: str, py: Path) -> None:
        # No --require-hashes: cu128 torch wheels surface a marker-only
        # transitive (torchcodec) the lock drops, which --require-hashes
        # would reject even though it's never installed here.
        run_cmd(
            "la-sidecar",
            [
                uv,
                "pip",
                "install",
                "--python",
                str(py),
                "--torch-backend=cu128",
                "--overrides",
                str(_NVRTC_OVERRIDE),
                "-r",
                str(_LOCK),
            ],
        )

    return ensure_pip_venv(
        label="la-sidecar",
        home=home,
        python="3.12",
        install=_install,
        override=override,
        override_env=_VENV_ENV,
        # Keyed on the lock text + the nvrtc override so recompiling either
        # repairs an existing venv.
        spec=(_LOCK.read_text(encoding="utf-8"), _NVRTC_OVERRIDE.read_text(encoding="utf-8")),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        required=True,
        help="Immutable Hugging Face reference in repo@revision form.",
    )
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
