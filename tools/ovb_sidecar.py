"""Boot the Open Vision Banana inference server in an isolated sidecar venv.

Open Vision Banana (OVB, arXiv 2611.15844) is a fine-tuned FLUX.2 [klein]
transformer loaded into ``diffusers.Flux2KleinPipeline``. Upstream validates it
on ``diffusers==0.39.0`` + ``transformers==4.57.6``; the openral runtime is
``transformers>=5`` (CLAUDE.md §3), so it runs out-of-process, talking to
``openral_runner.backends.gstreamer.ovb_detector.OpenVisionBananaDetector`` over
ZMQ REQ/REP + msgpack (same pattern as ``tools.locateanything_sidecar``).

Usage::

    python tools/ovb_sidecar.py --port 5759 \
        --model FunAILab/Open-Vision-Banana@<sha> \
        --subfolder final_models/4B/zeroshot/checkpoint-87500-95000-w25/transformer \
        --base black-forest-labs/FLUX.2-klein-base-4B@<sha>

Weights load in bf16 (open path, ADR-0123). OVB weights are FLUX Non-Commercial;
the license guard is enforced upstream in the ``RSkillManifest`` loader, not here.
No checkpoint code executes: diffusers builds the model from its own classes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openral_sim._sidecar_common import ensure_pip_venv, run_cmd, scrub_environ, torch_backend

_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "ovb-sidecar"
_VENV_ENV = "OPENRAL_OVB_SIDECAR_VENV"
_HOME_ENV = "OPENRAL_OVB_SIDECAR_HOME"
_REQS = Path(__file__).resolve().parent / "sidecar_requirements"
_LOCK = _REQS / ("ovb-aarch64.lock" if torch_backend() == "cu130" else "ovb.lock")
_NVRTC_OVERRIDE = _REQS / "aarch64-nvrtc-override.txt"


def ensure_venv(home: Path, *, override: str | None = None) -> Path:
    """Return the sidecar venv python, creating + populating it from the lock if needed."""

    def _install(uv: str, py: Path) -> None:
        run_cmd(
            "ovb-sidecar",
            [
                uv,
                "pip",
                "install",
                "--no-config",
                "--python",
                str(py),
                f"--torch-backend={torch_backend()}",
                "--overrides",
                str(_NVRTC_OVERRIDE),
                "-r",
                str(_LOCK),
            ],
        )

    return ensure_pip_venv(
        label="ovb-sidecar",
        home=home,
        python="3.12",
        install=_install,
        override=override,
        override_env=_VENV_ENV,
        spec=(_LOCK.read_text(encoding="utf-8"), _NVRTC_OVERRIDE.read_text(encoding="utf-8")),
    )


def main() -> int:
    # Allowlist the environment before any install or the exec; HF_TOKEN stays.
    scrub_environ()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="OVB repo as repo@<40-hex sha>.")
    p.add_argument("--subfolder", required=True, help="transformer/ folder inside --model.")
    p.add_argument("--base", required=True, help="Base FLUX.2 [klein] pipeline, repo@<sha>.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5759)
    p.add_argument(
        "--offload",
        default="model",
        choices=["none", "model", "sequential"],
        help="diffusers CPU offload mode (default 'model'; 'none' needs ~17 GB VRAM for 4B).",
    )
    p.add_argument(
        "--home",
        type=Path,
        default=Path(os.environ.get(_HOME_ENV, _DEFAULT_HOME)),
        help=f"Sidecar work directory (default {_DEFAULT_HOME}).",
    )
    p.add_argument("--venv", default=None, help=f"Reuse this venv (or set {_VENV_ENV}).")
    args = p.parse_args()

    py = ensure_venv(args.home, override=args.venv)
    server = Path(__file__).resolve().parent / "_ovb_server.py"
    env = os.environ.copy()
    # The workspace PYTHONPATH carries transformers 5.x, which would shadow the
    # venv's 4.57.6 (same failure tools/locateanything_sidecar.py documents).
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    cmd = [
        str(py),
        str(server),
        "--model",
        args.model,
        "--subfolder",
        args.subfolder,
        "--base",
        args.base,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--offload",
        args.offload,
    ]
    print(f"[ovb-sidecar] launching server: model={args.model} port={args.port}", flush=True)
    os.execvpe(str(py), cmd, env)


if __name__ == "__main__":
    sys.exit(main() or 0)
