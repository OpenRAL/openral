"""Boot the LingBot-VLA 2.0 inference server in an isolated py3.12 sidecar venv.

Robbyant's ``lingbotvla`` package (https://github.com/robbyant/lingbot-vla-v2,
Apache-2.0 code + weights) pins ``torch==2.8.0`` / ``transformers==4.57.3`` /
``triton==3.4.0`` and carries custom Triton MoE kernels + a ``sys.path``-based
layout, so it cannot coexist in the openral py3.12 (torch>=2.9 / transformers>=5)
workspace (CLAUDE.md §3). We run the upstream ``LingbotVLAv2Server`` out-of-process
and drive it from :mod:`openral_sim.policies.lingbot_vla2` over ZMQ REQ/REP framed
by msgpack — the same transport the rldx / rlbench-3dda sidecars use.

This file is the **boot helper**: it runs under the openral interpreter (which has
``openral_sim`` installed, so it can import :mod:`openral_sim._sidecar_common`),
auto-provisions the sidecar on first use, then ``os.execvpe``-s into the sidecar
venv running :mod:`tools._lingbot_vla2_server` (the server side, no openral import).
Provisioning is the same shape as the locateanything / qwen-vlm pip sidecars, with
a pinned-SHA git clone of the upstream repo added (the ``lingbotvla`` package +
``configs/`` + ``assets/norm_stats`` live in the checkout, put on the server's
``sys.path``):

1. **Clone** ``lingbot-vla-v2`` at :data:`_PINNED_SHA` into ``<home>/source``
   (skipped when ``$OPENRAL_LINGBOT_VLA2_REPO`` points at an existing checkout).
2. **Venv** — a Python 3.12 ``<home>/.venv`` populated from the upstream
   ``requirements.txt`` (fully version-pinned: torch==2.8.0, transformers==4.57.3,
   triton==3.4.0, numpy==1.26.4, …) plus the openral-side wire/quant extras
   (``pyzmq`` + ``bitsandbytes``). flash-attn is deliberately NOT installed — the
   server coerces the upstream ``flash_attention_2`` hardcode to sdpa/eager.
   Skipped when ``$OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON`` points at an existing
   interpreter.
3. **Exec** ``tools/_lingbot_vla2_server.py`` in that venv with the resolved
   ``--model/--robo-name/--quantization/--attn/--host/--port``.

Usage (normally auto-spawned by the adapter; run by hand to pre-provision)::

    python tools/lingbot_vla2_sidecar.py --model robbyant/lingbot-vla-v2-6b --port 5555

CLAUDE.md compliance: real upstream code in a real subprocess (no mocks, §1.11);
py-version/dep isolation is the only safe bridge (§3); Apache-2.0 code + weights
carry no license guard.
"""

from __future__ import annotations

import argparse
import os
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

    The V1 repo's ``requirements.txt`` is training-oriented and omits the
    inference stack we need (lerobot / bitsandbytes / pyzmq), so we install the
    pinned set explicitly. torch is cu128 to match the host driver; ``lerobot``
    only carries transformers as an optional extra, so base install coexists with
    the pinned ``transformers==4.51.3``. flash-attn is deliberately NOT installed —
    the server coerces the upstream flash_attention_2 hardcode to eager.
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

    ``$OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON`` reuses an existing interpreter
    verbatim (dev escape hatch). Otherwise a Python 3.12 venv is built and the
    fully-pinned upstream ``requirements.txt`` is installed (cu128 torch wheels),
    plus the openral-side wire/quant deps (``pyzmq`` + ``bitsandbytes``).
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
        run_cmd(
            _LABEL,
            [uv, "pip", "install", "--python", str(py), "--torch-backend=cu128", "-r", str(reqs)],
        )
        # openral-side wire (pyzmq/msgpack — msgpack is already in the upstream
        # reqs) + NF4 quantization (bitsandbytes); neither is in requirements.txt.
        run_cmd(
            _LABEL,
            [uv, "pip", "install", "--python", str(py), "pyzmq", "bitsandbytes"],
        )

    ensure_uv()
    return ensure_pip_venv(label=_LABEL, home=home, python="3.12", install=install or _install)


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
