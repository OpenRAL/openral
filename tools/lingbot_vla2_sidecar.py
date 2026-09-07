"""Boot the LingBot-VLA 2.0 inference server in an isolated py3.12 sidecar venv.

Robbyant's ``lingbotvla`` (https://github.com/robbyant/lingbot-vla-v2,
Apache-2.0 code+weights) pins ``torch==2.8.0``/``transformers==4.57.3``/
``triton==3.4.0`` and carries custom Triton MoE kernels + a ``sys.path``-based
layout — incompatible with the openral py3.12/transformers>=5 workspace
(CLAUDE.md §3). Runs the upstream ``LingbotVLAv2Server`` out-of-process,
driven from :mod:`openral_sim.policies.lingbot_vla2` over ZMQ REQ/REP +
msgpack (same transport as rldx / rlbench-3dda).

Auto-provisions on first use, under the openral interpreter:
1. **Clone** ``lingbot-vla-v2`` at :data:`_PINNED_SHA` into ``<home>/source``
   (``$OPENRAL_LINGBOT_VLA2_REPO`` overrides with an existing checkout).
2. **Venv** — Python 3.12 from upstream ``requirements.txt`` under
   :data:`_V2_OVERRIDES` (torch 2.9.1 / triton 3.5.1, not upstream's
   aarch64-less 2.8.0/3.4.0) plus ``pyzmq`` + ``bitsandbytes``. flash-attn is
   deliberately not installed — the server coerces to sdpa/eager.
   (``$OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON`` overrides with an existing
   interpreter.)
3. **Exec** ``tools/_lingbot_vla2_server.py`` with the resolved
   ``--model/--robo-name/--quantization/--attn/--host/--port``.

Usage (normally auto-spawned; run by hand to pre-provision)::

    python tools/lingbot_vla2_sidecar.py --model robbyant/lingbot-vla-v2-6b --port 5555

Real subprocess, no mocks (§1.11); py-version/dep isolation is the only safe
bridge (§3). Apache-2.0 code + weights — no license guard needed.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from collections.abc import Callable
from pathlib import Path

from openral_sim._sidecar_common import (
    ensure_pip_venv,
    ensure_uv,
    make_isolated_env,
    run_cmd,
    write_sidecar_identity,
)

_LABEL = "lingbot-sidecar"
_REPO_URL = "https://github.com/robbyant/lingbot-vla-v2.git"
# Pinned upstream commit (HEAD of the reference checkout used to validate the
# integration). Bump deliberately when re-validating against a newer revision.
_PINNED_SHA = "69729b4ef24c63ec25e750915491635f4753be1d"
_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "lingbot-vla2-sidecar"
_HOME_ENV = "OPENRAL_LINGBOT_VLA2_SIDECAR_HOME"
_VENV_ENV = "OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON"
_REPO_ENV = "OPENRAL_LINGBOT_VLA2_REPO"

# LingBot-VLA 1.0 (4B) variant: the V1 codebase is a different repo with a
# different stack (transformers==4.51.3 + lerobot==0.4.2 flat layout) than the v2
# 6B model, so it gets its own clone + venv + home. See _lingbot_vla2_server.py's
# _LingBotV1Policy for the loading path.
_REPO_URL_V1 = "https://github.com/robbyant/lingbot-vla.git"
_PINNED_SHA_V1 = "4eb34b7693a0565c67433f8fac9c59a2e67eb60b"
_DEFAULT_HOME_V1 = Path.home() / ".cache" / "openral" / "lingbot-vla-v1-sidecar"
_VENV_ENV_V1 = "OPENRAL_LINGBOT_VLA_SIDECAR_PYTHON"
_REPO_ENV_V1 = "OPENRAL_LINGBOT_VLA_REPO"

_SERVER = Path(__file__).resolve().parent / "_lingbot_vla2_server.py"
# Raises nvrtc past the sm_121 ceiling on aarch64 (GB10 / Jetson Thor); shared
# with the other sidecars, see that file's header. Passed as a SECOND
# ``--overrides`` alongside the generated torch-stack overrides below.
_NVRTC_OVERRIDE = (
    Path(__file__).resolve().parent / "sidecar_requirements" / "aarch64-nvrtc-override.txt"
)

# V2 torch stack, newer than upstream's torch==2.8.0/torchvision==0.23.0/
# torchaudio==2.8.0/triton==3.4.0: cu128 torch 2.8.0 and triton 3.4.0 ship no
# aarch64 wheel (GB10/DGX Spark/Jetson Thor), so upstream's pins are
# uninstallable there. 2.9.1+cu128 (-> triton 3.5.1) does ship aarch64. Fed to
# `uv pip install --overrides`; lingbotvla runs off sys.path (not installed),
# so it has no metadata cap to fight. V1 is NOT covered — see
# :func:`_install_v1`. See docs/reference/aarch64-support.md.
_TORCH_PIN = "torch==2.9.1"
_TORCHVISION_PIN = "torchvision==0.24.1"
_TORCHAUDIO_PIN = "torchaudio==2.9.1"
# torchcodec ships per-torch-minor builds (0.6.x<->2.8, 0.9.x<->2.9); pairing
# upstream's 0.6.0 with torch 2.9 breaks the c10 ABI. Dropped on aarch64
# entirely (no wheel below 0.11.0, a torch-2.10 build) — it's a training-only
# video decoder the inference server never imports.
_TORCHCODEC_PIN = 'torchcodec==0.9.1 ; platform_machine != "aarch64"'
_TRITON_PIN = "triton==3.5.1"
_V2_OVERRIDES = (
    _TORCH_PIN,
    _TORCHVISION_PIN,
    _TORCHAUDIO_PIN,
    _TORCHCODEC_PIN,
    _TRITON_PIN,
)


def _ensure_source(
    home: Path, *, url: str = _REPO_URL, sha: str = _PINNED_SHA, repo_env: str = _REPO_ENV
) -> Path:
    """Clone the upstream lingbotvla repo at the pinned commit into ``<home>/source``.

    ``$<repo_env>`` overrides with an existing checkout (dev escape hatch). Unlike
    the shared ``ensure_source`` (which shallow-clones the default branch HEAD), we
    fetch exactly the pinned commit so the sidecar is reproducible (CLAUDE.md §1.8)
    regardless of where upstream ``main`` has moved.
    """
    override = os.environ.get(repo_env)
    if override:
        src = Path(override).expanduser()
        if not (src / "lingbotvla").is_dir():
            raise SystemExit(f"{repo_env}={override!r} has no lingbotvla/ package dir")
        print(f"[{_LABEL}] reusing operator-provided checkout at {src}", flush=True)
        return src
    source = home / "source"
    if (source / ".git").is_dir():
        print(f"[{_LABEL}] reusing existing checkout at {source}", flush=True)
        return source
    home.mkdir(parents=True, exist_ok=True)
    # Shallow-fetch exactly the pinned commit (no full history) and check it out.
    run_cmd(_LABEL, ["git", "init", "-q", str(source)])
    run_cmd(_LABEL, ["git", "-C", str(source), "remote", "add", "origin", url])
    run_cmd(_LABEL, ["git", "-C", str(source), "fetch", "--depth", "1", "origin", sha])
    run_cmd(_LABEL, ["git", "-C", str(source), "checkout", "-q", "FETCH_HEAD"])
    return source


def _install_v1(uv: str, py: Path) -> None:
    """Provision the V1 (4B) sidecar venv.

    V1's ``requirements.txt`` is training-oriented and omits lerobot/
    bitsandbytes/pyzmq, so they're installed explicitly (torch cu128,
    transformers==4.51.3; flash-attn deliberately skipped — server coerces to
    eager).

    x86_64 only: ``lerobot==0.4.2`` (a hard V1 dependency) caps
    ``torch<2.8.0``, and no torch under that cap ships an aarch64 cu128 wheel
    (2.9.x is over the cap; 2.7.x is under it but needs x86_64-only
    triton==3.3.1); ``torchcodec`` is likewise x86_64/darwin-arm64-only below
    0.11.0. Lifting this needs moving V1 off lerobot 0.4.2. See
    docs/reference/aarch64-support.md.
    """
    run_cmd(
        _LABEL,
        [
            uv,
            "pip",
            "install",
            "--python",
            str(py),
            "--torch-backend=cu128",
            "torch==2.8.0",
            "torchvision==0.23.0",
        ],
    )
    run_cmd(_LABEL, [uv, "pip", "install", "--python", str(py), "lerobot==0.4.2"])
    run_cmd(
        _LABEL,
        [
            uv,
            "pip",
            "install",
            "--python",
            str(py),
            "transformers==4.51.3",
            "numpy==1.26.4",
            "torchcodec==0.6.0",
            "datasets==3.6.0",
        ],
    )
    run_cmd(
        _LABEL,
        [
            uv,
            "pip",
            "install",
            "--python",
            str(py),
            "pyzmq",
            "msgpack",
            "bitsandbytes",
            "accelerate",
            "einops",
            "matplotlib",
            "pillow",
            "safetensors",
            "pyyaml",
            "opencv-python-headless",
            "h5py",
            "ipdb",
            "websockets",
            "torchdata",
            "blobfile",
            "tensorboard",
            "numpydantic",
        ],
    )


def _ensure_venv(
    home: Path,
    source: Path,
    *,
    install: Callable[[str, Path], None] | None = None,
    venv_env: str = _VENV_ENV,
) -> Path:
    """Return the sidecar venv python, provisioning it from the upstream pins.

    ``$OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON`` reuses an existing interpreter.
    Otherwise builds Python 3.12 from upstream ``requirements.txt`` (cu128)
    under :data:`_V2_OVERRIDES`, plus ``pyzmq`` + ``bitsandbytes``.
    """
    override = os.environ.get(venv_env)
    if override:
        py = Path(override).expanduser()
        if not py.is_file():
            raise SystemExit(f"{venv_env}={override!r} is not a file")
        print(f"[{_LABEL}] reusing operator-provided interpreter at {py}", flush=True)
        return py

    reqs = source / "requirements.txt"

    def _install(uv: str, py: Path) -> None:
        # ``-r requirements.txt`` is fully version-pinned upstream (torch==2.8.0,
        # transformers==4.57.3, triton==3.4.0, numpy==1.26.4, …). No
        # --require-hashes: the cu128 torch wheels surface marker-only transitives
        # uv's resolver drops, which --require-hashes would reject even though they
        # never install on this platform (same rationale as locateanything.lock).
        #
        # --overrides replaces the five upstream torch-stack pins (see
        # _V2_OVERRIDES), two of which cannot be installed on aarch64 at all.
        # Written into the sidecar home rather than the checkout so the
        # pinned-SHA clone stays pristine.
        overrides = home / "torch-overrides.txt"
        home.mkdir(parents=True, exist_ok=True)
        overrides.write_text("\n".join(_V2_OVERRIDES) + "\n", encoding="utf-8")
        run_cmd(
            _LABEL,
            [
                uv,
                "pip",
                "install",
                "--python",
                str(py),
                "--torch-backend=cu128",
                "--overrides",
                str(overrides),
                "--overrides",
                str(_NVRTC_OVERRIDE),
                "-r",
                str(reqs),
            ],
        )
        # pyzmq (wire) + bitsandbytes (NF4) — neither is in requirements.txt.
        # aarch64 also gets a CUDA 12.9 ptxas: this sidecar runs its own Triton
        # kernels (lingbotvla's MoE), and triton 3.5.1's bundled CUDA 12.8
        # ptxas can't target sm_121 ("Value 'sm_121a' is not defined for
        # option 'gpu-name'"); `make_isolated_env` sets TRITON_PTXAS_PATH.
        # The override goes on THIS pass too — uv re-resolves on every `pip
        # install`, so a pass without it re-downgrades
        # `nvidia-cuda-nvrtc-cu12==12.8.93`. Invariant: every re-resolving
        # pass carries it.
        extras = ["pyzmq", "bitsandbytes"]
        if platform.machine() == "aarch64":
            extras.append("nvidia-cuda-nvcc-cu12==12.9.86")
        run_cmd(
            _LABEL,
            [
                uv,
                "pip",
                "install",
                "--python",
                str(py),
                "--overrides",
                str(_NVRTC_OVERRIDE),
                *extras,
            ],
        )

    ensure_uv()
    return ensure_pip_venv(
        label=_LABEL,
        home=home,
        python="3.12",
        install=install or _install,
        # Keyed on the upstream pins (which move with _PINNED_SHA) + our torch
        # overrides + our extras, so a repinned checkout *or* a corrected torch
        # override repairs an existing venv. An injected custom ``install`` has
        # no spec we can key on, so it keeps the opaque marker.
        spec=(
            None
            if install
            else (
                reqs.read_text(encoding="utf-8"),
                *_V2_OVERRIDES,
                _NVRTC_OVERRIDE.read_text(encoding="utf-8"),
                "pyzmq",
                "bitsandbytes",
            )
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    p.add_argument("--robo-name", default="robotwin", help="upstream robot config stem")
    p.add_argument("--quantization", choices=("none", "nf4"), default="nf4")
    p.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Inference device (cpu forces bf16; frees the GPU for a co-resident sim).",
    )
    p.add_argument(
        "--attn",
        choices=("sdpa", "eager"),
        default="sdpa",
        help="flash-free attention backend (flash-attn is not installed).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument(
        "--variant",
        choices=("v2", "v1"),
        default="v2",
        help="Model family: v2 = 6B Qwen3-VL MoE; v1 = 4B Qwen2.5-VL dense expert "
        "(LingBot-VLA 1.0 / posttrain-robotwin). Selects the repo + venv + server variant.",
    )
    p.add_argument(
        "--home",
        type=Path,
        default=None,
        help="Sidecar work directory (default depends on --variant).",
    )
    args = p.parse_args()

    # Per-variant provisioning: repo URL/SHA, venv installer, home, env names.
    if args.variant == "v1":
        default_home = Path(os.environ.get(_HOME_ENV, _DEFAULT_HOME_V1))
        url, sha, repo_env, venv_env, install = (
            _REPO_URL_V1,
            _PINNED_SHA_V1,
            _REPO_ENV_V1,
            _VENV_ENV_V1,
            _install_v1,
        )
        qwen_envs = ("OPENRAL_QWEN25VL_PATH", "QWEN25_PATH")
    else:
        default_home = Path(os.environ.get(_HOME_ENV, _DEFAULT_HOME))
        url, sha, repo_env, venv_env, install = (
            _REPO_URL,
            _PINNED_SHA,
            _REPO_ENV,
            _VENV_ENV,
            None,
        )
        qwen_envs = ("OPENRAL_QWEN3VL_PATH", "QWEN3VL_PATH")
    home = args.home or default_home

    source = _ensure_source(home, url=url, sha=sha, repo_env=repo_env)
    venv_py = _ensure_venv(home, source, install=install, venv_env=venv_env)
    venv = venv_py.parent.parent

    # Record which checkpoint this sidecar serves so the adapter's SidecarClient
    # can reap a stale sidecar bound to this port (identity is keyed by port).
    write_sidecar_identity(
        port=int(args.port),
        family=f"lingbot_vla_{args.variant}",
        model=str(args.model),
        embodiment_tag=str(args.robo_name),
        quantization=str(args.quantization),
    )

    env = make_isolated_env(venv)
    env[repo_env] = str(source)
    for qenv in qwen_envs:
        val = os.environ.get(qenv)
        if val:
            env[qenv] = val

    cmd = [
        str(venv_py),
        str(_SERVER),
        "--model",
        str(args.model),
        "--robo-name",
        str(args.robo_name),
        "--quantization",
        str(args.quantization),
        "--device",
        str(args.device),
        "--attn",
        str(args.attn),
        "--variant",
        str(args.variant),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
    ]
    print(
        f"[{_LABEL}] launching server (variant={args.variant}): model={args.model} "
        f"port={args.port} quant={args.quantization} device={args.device} repo={source}",
        flush=True,
    )
    os.execvpe(str(venv_py), cmd, env)
    return 0  # unreachable — execvpe replaced the process


if __name__ == "__main__":
    sys.exit(main() or 0)
