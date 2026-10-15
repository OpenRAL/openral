#!/usr/bin/env python3
"""Provision, export, or serve Xiaomi Robotics XR-1 in an isolated venv.

``--quantization nf4`` defines the same runtime rule used by the XR-1 policy
adapter. ``--export-dir`` persists that loaded bitsandbytes model as a local
Transformers checkpoint so later rSkill loads use packed NF4 tensors directly.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

from openral_sim._sidecar_common import (
    ensure_pip_venv,
    make_isolated_env,
    run_cmd,
    torch_backend,
    write_sidecar_identity,
)

_LABEL = "xr1-sidecar"
_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "xr1-sidecar"
_SERVER = Path(__file__).resolve().parent / "_xr1_server.py"

# Pinned deps, installed in three passes (torch index redirect / plain PyPI /
# no-build-isolation for flash-attn); named constants so editing one repairs
# an already-provisioned venv (keys the sentinel, see ``ensure_pip_venv``).
# torch 2.9.1, not upstream's validated 2.8.0: the cu128 2.8.0 build has no
# linux_aarch64 wheel (x86_64/win_amd64 only) and needs triton==3.4.0, which
# also has none — blocking provisioning on aarch64 (GB10/DGX Spark, Jetson
# Thor). 2.9.1+cu128 publishes manylinux_2_28_aarch64 and pulls triton 3.5.1,
# which does too. See ``docs/reference/aarch64-support.md``.
_XR1_TORCH_DEPS = ("torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1")
# aarch64 takes torch 2.10 on ``torch_backend()``'s cu130 index instead, the only
# torch that has both ``sm_110`` kernels (Jetson Thor) and a flash-attn 2.8.3
# aarch64 prebuilt. x86_64 keeps the 2.9.1 / cu128 stack above unchanged.
_XR1_AARCH64_TORCH_DEPS = ("torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0")
# Raises nvrtc past the sm_121 ceiling on aarch64 (GB10 / Jetson Thor). torch's
# own metadata pins ``nvidia-cuda-nvrtc-cu12==12.8.93``, whose newest supported
# arch is sm_120, so any kernel torch compiles at runtime through the nvrtc
# jiterator dies with "invalid value for --gpu-architecture". Precompiled
# kernels are unaffected, which is why this hides until a jiterator op is hit.
# Shared with the other sidecars — see the file's header.
_NVRTC_OVERRIDE = (
    Path(__file__).resolve().parent / "sidecar_requirements" / "aarch64-nvrtc-override.txt"
)
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
# The PyPI sdist's setup.py fetches its prebuilt wheel by a name it computes from
# torch, but it maps every non-11 CUDA major to "12" — under torch 2.10+cu130 it asks
# for a ``cu12torch2.10`` asset that does not exist and falls back to an hour-plus
# nvcc build. So aarch64 names the real ``cu13torch2.10`` release asset directly.
_XR1_AARCH64_ATTN_DEPS = (
    "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_aarch64.whl",
)


def _deps(machine: str | None = None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(torch_deps, attn_deps)`` for ``machine`` (``None`` reads this host)."""
    if (machine or platform.machine()) == "aarch64":
        return _XR1_AARCH64_TORCH_DEPS, _XR1_AARCH64_ATTN_DEPS
    return _XR1_TORCH_DEPS, _XR1_ATTN_DEPS


def _ensure_venv(home: Path) -> Path:
    override = os.environ.get("OPENRAL_XR1_SIDECAR_PYTHON")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise SystemExit(f"OPENRAL_XR1_SIDECAR_PYTHON={override!r} is not a file")
        return path
    torch_deps, attn_deps = _deps()

    def _install(uv: str, py: Path) -> None:
        # Override rides on ALL THREE passes: uv re-resolves the whole env on
        # every `pip install`, so a pass without it sees torch's exact
        # `nvidia-cuda-nvrtc-cu12==12.8.93` pin again and silently downgrades
        # an earlier pass's shim (same failure the LingBot sidecar hit).
        # Invariant: every uv pass that can re-resolve carries the override.
        # --no-config: launched from the repo root, uv would otherwise apply the root
        # pyproject's [tool.uv] overrides, which on aarch64 swap torch for 2.13 (seen
        # on Thor: torch 2.13.0 next to torchaudio 2.10.0, flash-attn's .so unloadable).
        pip = [uv, "pip", "install", "--no-config", "--python", str(py)]
        pip += ["--overrides", str(_NVRTC_OVERRIDE)]
        # cu130 on aarch64 (Jetson Thor sm_110), cu128 on x86_64. See _deps() and
        # docs/reference/aarch64-support.md.
        run_cmd(_LABEL, [*pip, f"--torch-backend={torch_backend()}", *torch_deps])
        run_cmd(_LABEL, [*pip, *_XR1_RUNTIME_DEPS])
        run_cmd(_LABEL, [*pip, "--no-build-isolation", *attn_deps])

    return ensure_pip_venv(
        label=_LABEL,
        home=home,
        python="3.12",
        install=_install,
        override_env="OPENRAL_XR1_SIDECAR_VENV",
        sentinel_name=".xr1-nf4-deps-installed",
        spec=(
            *torch_deps,
            *_XR1_RUNTIME_DEPS,
            *attn_deps,
            _NVRTC_OVERRIDE.read_text(encoding="utf-8"),
        ),
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
