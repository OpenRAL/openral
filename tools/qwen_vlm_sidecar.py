"""Boot the Qwen3.5-4B scene-VLM inference server in an isolated sidecar venv.

The ``query_scene`` reasoner tool asks ``Qwen/Qwen3.5-4B`` (NF4 via
bitsandbytes) open-ended questions about the camera view. Runs
out-of-process: the runtime venv hard-pins ``transformers==5.3.0`` for
lerobot, and resolving ``bitsandbytes``/``qwen-vl-utils``/optional
Gated-DeltaNet kernels (``fla``/``causal-conv1d``) into that env risks
perturbing the VLA stack; a 4B model + CUDA context also should not live in
the ``rclpy`` reasoner process (its OOM must not take the reasoner down).
Same ZMQ REQ/REP + msgpack pattern as ``tools/locateanything_sidecar.py`` and
``tools/rldx_sidecar.py``.

The openral side is
:class:`openral_runner.backends.gstreamer.qwen_scene_vlm.QwenSceneVlm`, which
auto-spawns this sidecar and talks to ``_qwen_vlm_server.py`` over ZMQ.

Usage::

    python tools/qwen_vlm_sidecar.py --port 5759

Real subprocess, no mocks (§1.11). ``Qwen/Qwen3.5-4B`` is Apache-2.0
(commercial OK) — no license guard needed here.
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

# Hash-locked deps: transformers==5.3.0 (matches the runtime pin), torch/torchvision +cu128.
# `fla`/`causal-conv1d` (optional Gated-DeltaNet kernels) deliberately excluded — without them
# transformers falls back to slower PyTorch ops but still loads; their wheels don't resolve
# everywhere. Regenerate:
#   uv pip compile tools/sidecar_requirements/qwen_vlm.in \
#     --universal --torch-backend=cu128 --generate-hashes --python-version 3.12 \
#     --overrides tools/sidecar_requirements/aarch64-nvrtc-override.txt \
#     -o tools/sidecar_requirements/qwen_vlm.lock
_LOCK = Path(__file__).resolve().parent / "sidecar_requirements" / "qwen_vlm.lock"
# Raises nvrtc past the sm_121 cliff on aarch64. Load-bearing here: Qwen3.5's
# `image_grid_thw.prod(-1)` is a jiterator op — every GB10 query died in nvrtc without this.
# Passed at install time too: torch pins `nvidia-cuda-nvrtc-cu12==12.8.93`, so without the
# override uv rejects the lock's aarch64 line as a conflict.
_NVRTC_OVERRIDE = (
    Path(__file__).resolve().parent / "sidecar_requirements" / "aarch64-nvrtc-override.txt"
)


def ensure_venv(home: Path, *, override: str | None = None) -> Path:
    """Return the sidecar venv python, creating + populating it if needed.

    ``override`` (or ``$OPENRAL_QWEN_VLM_SIDECAR_VENV``) reuses an existing
    venv instead of provisioning one under ``home``. Otherwise provisions
    Python 3.12 from the hash-locked ``qwen_vlm.lock`` (CLAUDE.md §1.8).
    """

    def _install(uv: str, py: Path) -> None:
        # No --require-hashes: cu128 torch wheels surface a marker-only
        # transitive (torchcodec) the lock drops, which --require-hashes
        # would reject even though it's never installed here.
        run_cmd(
            "qwen-sidecar",
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
        label="qwen-sidecar",
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
