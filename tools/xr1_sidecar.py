#!/usr/bin/env python3
"""Provision, export, or serve Xiaomi Robotics XR-1 in an isolated venv.

``--quantization nf4`` defines the same runtime rule used by the XR-1 policy
adapter. ``--export-dir`` persists that loaded bitsandbytes model as a local
Transformers checkpoint so later rSkill loads use packed NF4 tensors directly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from openral_sim._sidecar_common import (
    ensure_pip_venv,
    make_isolated_env,
    run_cmd,
    write_sidecar_identity,
)

_LABEL = "xr1-sidecar"
_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "xr1-sidecar"
_SERVER = Path(__file__).resolve().parent / "_xr1_server.py"

# Pinned deps, installed in three passes (torch index redirect / plain PyPI /
# no-build-isolation for flash-attn). Named constants rather than inline lists
# so the same pins key the provisioning sentinel — editing one repairs an
# already-provisioned venv instead of being ignored (see ``ensure_pip_venv``).
#
# torch is 2.9.1, not the 2.8.0 upstream XR-1 validated against: the ``cu128``
# 2.8.0 build publishes no ``linux_aarch64`` wheel (only manylinux x86_64 +
# win_amd64) and its required ``triton==3.4.0`` has no aarch64 wheel either, so
# the sidecar could not provision at all on an aarch64 CUDA host (GB10 / DGX
# Spark, Jetson Thor). 2.9.1+cu128 publishes ``manylinux_2_28_aarch64`` and
# pulls triton 3.5.1, which does too. See ``docs/reference/aarch64-support.md``.
_XR1_TORCH_DEPS = ("torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1")
_XR1_RUNTIME_DEPS = (
    "transformers==4.57.1",
    "accelerate==1.11.0",
    "safetensors==0.6.2",
    "liger-kernel==0.6.5",
    "qwen-vl-utils>=0.0.14",
    "einops",
    "pillow",
    "pyzmq",
    "msgpack",
    "bitsandbytes==0.49.2",
)
_XR1_ATTN_DEPS = ("flash-attn==2.8.3",)


def _ensure_venv(home: Path) -> Path:
    override = os.environ.get("OPENRAL_XR1_SIDECAR_PYTHON")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise SystemExit(f"OPENRAL_XR1_SIDECAR_PYTHON={override!r} is not a file")
        return path

    def _install(uv: str, py: Path) -> None:
        run_cmd(
            _LABEL,
            [uv, "pip", "install", "--python", str(py), "--torch-backend=cu128", *_XR1_TORCH_DEPS],
        )
        run_cmd(_LABEL, [uv, "pip", "install", "--python", str(py), *_XR1_RUNTIME_DEPS])
        run_cmd(
            _LABEL,
            [uv, "pip", "install", "--python", str(py), "--no-build-isolation", *_XR1_ATTN_DEPS],
        )

    return ensure_pip_venv(
        label=_LABEL,
        home=home,
        python="3.12",
        install=_install,
        override_env="OPENRAL_XR1_SIDECAR_VENV",
        sentinel_name=".xr1-nf4-deps-installed",
        spec=(*_XR1_TORCH_DEPS, *_XR1_RUNTIME_DEPS, *_XR1_ATTN_DEPS),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--profile",
        required=True,
        choices=("robocasa_mg", "robocasa365", "vlabench_choice"),
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "nf4", "prequantized_nf4"),
        default="nf4",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Save the loaded NF4 model + processor here, then exit without serving.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--home",
        type=Path,
        default=Path(os.environ.get("OPENRAL_XR1_SIDECAR_HOME", _DEFAULT_HOME)),
    )
    args = parser.parse_args()
    if os.environ.get("OPENRAL_ALLOW_REMOTE_CODE") != "1":
        raise SystemExit(
            "XR-1 executes checkpoint-shipped Transformers code. Review the pinned "
            "revision, then set OPENRAL_ALLOW_REMOTE_CODE=1."
        )

    venv_py = _ensure_venv(args.home)
    if args.export_dir is None:
        write_sidecar_identity(
            port=args.port,
            family="xr1",
            model=args.model,
            embodiment_tag=args.profile,
            quantization=args.quantization,
        )
    env = make_isolated_env(venv_py.parent.parent)
    server_argv = [
        str(venv_py),
        str(_SERVER),
        "--model",
        args.model,
        "--profile",
        args.profile,
        "--quantization",
        args.quantization,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.export_dir is not None:
        server_argv.extend(["--export-dir", str(args.export_dir)])
    os.execvpe(
        str(venv_py),
        server_argv,
        env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
