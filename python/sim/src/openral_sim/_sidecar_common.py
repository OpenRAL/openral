"""Shared scaffolding for the out-of-process VLA sidecars (ADR-0010 / ADR-0046).

Two policy families run their ~3B checkpoints out-of-process in an isolated
Python 3.10 venv, because their upstream packages pin Python 3.10 + flash-attn
+ CUDA — incompatible with the 3.12-only workspace (CLAUDE.md §3):

* ``rldx``  — RLWRLD RLDX-1 (a GR00T-N1.5 finetune; the ``rldx`` package),
* ``gr00t`` — NVIDIA Isaac GR00T (the ``gr00t`` package).

They are **siblings**, not a class hierarchy: neither package depends on the
other. They share (a) the ZMQ + msgpack wire on the openral side — RLDX-1 keeps
GR00T's ``PolicyServer`` contract, so one ``_Gr00tFamilySidecarAdapter`` drives
both — and (b) ~70% of the boot scaffolding: clone, venv, install, env
isolation, exec. That scaffolding lives here; each ``tools/<family>_sidecar.py``
supplies only the per-family pieces (repo URL, dependency install, server
wrapper).

The ``tools/<family>_sidecar.py`` boot helpers import this module — they run
under the openral interpreter (which has ``openral_sim`` installed) to bootstrap
the sidecar, *then* ``exec`` into the isolated 3.10 venv. The isolation
(:func:`make_isolated_env`) only applies to that exec'd child.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

# Where each booted sidecar records *what* it is serving, so the openral-side
# adapter can refuse to silently reuse a sidecar that belongs to a different
# checkpoint / family (the "always RLDX" cross-process-sharing bug — two
# sidecars defaulting to the same port made any second eval bind to whatever
# model was already loaded). Keyed by port because a sidecar is local and the
# port is its unique bind; the adapter connects to 127.0.0.1 while the wrapper
# binds 0.0.0.0, so host is deliberately not part of the key.
_SIDECAR_REGISTRY_DIR = Path.home() / ".cache" / "openral" / "sidecars"


def sidecar_identity_path(port: int) -> Path:
    """Return the on-disk identity record path for a sidecar bound to ``port``."""
    return _SIDECAR_REGISTRY_DIR / f"port-{port}.json"


def write_sidecar_identity(
    *, port: int, family: str, model: str, embodiment_tag: str, quantization: str
) -> None:
    """Record which checkpoint a sidecar is about to serve on ``port``.

    Written by :func:`run_sidecar` just before it execs the server (whether
    the boot was auto-spawned by the adapter or launched by hand by an
    operator), so *every* sidecar this repo starts is identifiable. The
    adapter reads it back via :func:`read_sidecar_identity` before reusing a
    pre-existing sidecar.
    """
    _SIDECAR_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "port": port,
        "family": family,
        "model": model,
        "embodiment_tag": embodiment_tag,
        "quantization": quantization,
        "pid": os.getpid(),
    }
    sidecar_identity_path(port).write_text(json.dumps(payload), encoding="utf-8")


def read_sidecar_identity(port: int) -> dict[str, str] | None:
    """Return the recorded identity for the sidecar on ``port``, or ``None``.

    ``None`` means no record exists — e.g. a sidecar booted before this
    control landed, or by some path other than :func:`run_sidecar`. Callers
    treat that as "unverifiable" rather than "mismatched".
    """
    path = sidecar_identity_path(port)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    return {str(k): str(v) for k, v in loaded.items()}


def run_cmd(
    label: str, cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    """Run ``cmd`` (echoed with a ``[label]`` prefix) and raise on non-zero exit."""
    print(f"[{label}] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_uv() -> str:
    """Return the path to ``uv`` or exit with an install hint."""
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(
            "uv not found on PATH. Install it: "
            "https://docs.astral.sh/uv/getting-started/installation/"
        )
    return uv


def ensure_source(label: str, work: Path, repo_url: str) -> Path:
    """Shallow-clone ``repo_url`` into ``<work>/source`` if absent; return it."""
    source = work / "source"
    if (source / ".git").is_dir():
        print(f"[{label}] reusing existing checkout at {source}", flush=True)
        return source
    work.mkdir(parents=True, exist_ok=True)
    run_cmd(label, ["git", "clone", "--depth", "1", repo_url, str(source)])
    return source


def make_isolated_env(venv: Path) -> dict[str, str]:
    """Build the environment for the sidecar interpreter.

    Drops the workspace ``PYTHONPATH`` / ``PYTHONHOME`` (their 3.12 wheels would
    shadow the 3.10 venv and crash ``import transformers`` on a regex ABI
    mismatch), points at the venv, and disables torch.compile / inductor (which
    stalls post-load on small GPUs). The caller follows with :func:`exec_server`.

    Also defaults the CUDA caching-allocator to ``expandable_segments:True``.
    On an 8 GB-class GPU the sidecar's 3B checkpoint load (NF4 quant peak via
    bitsandbytes) co-exists with the main process's sim render context (LIBERO
    robosuite EGL / Isaac RTX); without expandable segments the allocator
    reserves over-large contiguous blocks and the load is OOM-killed mid-shard
    (observed as ``rldx_sidecar_died_during_boot returncode=-9`` in
    ``openral benchmark run`` — the suite path doesn't export the var the way
    the ``benchmark scene`` smoke wrapper did). Set here so every sidecar boot
    gets it regardless of how ``openral`` was launched; ``setdefault`` lets an
    explicit caller value win.
    """
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    env.setdefault("TORCHINDUCTOR_DISABLE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def exec_server(venv: Path, wrapper: Path, env: dict[str, str]) -> None:
    """Replace this process with the sidecar venv's python running ``wrapper``.

    Uses ``os.execvpe`` so signals (Ctrl-C) reach the server directly. Does not
    return.
    """
    py = str(venv / "bin" / "python")
    os.execvpe(py, [py, str(wrapper)], env)


def build_parser(
    *,
    description: str,
    default_home: Path,
    default_embodiment_tag: str,
    model_help: str,
    quant_help: str,
) -> argparse.ArgumentParser:
    """Construct the shared sidecar CLI (``--model/--port/--quantization/...``)."""
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help=model_help)
    p.add_argument("--port", type=int, default=5555, help="ZMQ port to bind (default 5555).")
    p.add_argument(
        "--quantization", choices=("none", "nf4", "int8"), default="nf4", help=quant_help
    )
    p.add_argument(
        "--embodiment-tag",
        default=default_embodiment_tag,
        help="EmbodimentTag passed to the server (drives the modality config — "
        "video / state / action / language keys + horizons).",
    )
    p.add_argument(
        "--home",
        type=Path,
        default=default_home,
        help=f"Sidecar work directory (default {default_home}).",
    )
    return p


def run_sidecar(
    *,
    label: str,
    family: str,
    repo_url: str,
    args: argparse.Namespace,
    install_deps: Callable[..., Path],
    make_wrapper: Callable[..., Path],
) -> int:
    """Orchestrate a sidecar boot: ensure uv → clone → install → wrapper → exec.

    ``family`` (``"rldx"`` / ``"gr00t"``) is recorded — together with the
    checkpoint, embodiment tag, and quantization — in the sidecar identity
    file so the adapter can verify a pre-existing sidecar before reusing it.
    ``install_deps(source, uv, quantization) -> venv`` and
    ``make_wrapper(work, source, args) -> wrapper_path`` are the per-family
    callbacks. Does not return (``exec_server`` replaces the process); the
    ``return 0`` is for type-checkers only.
    """
    uv = ensure_uv()
    work: Path = args.home
    source = ensure_source(label, work, repo_url)
    venv = install_deps(source=source, uv=uv, quantization=args.quantization)
    wrapper = make_wrapper(work=work, source=source, args=args)
    # Stamp the identity record before exec: the server binds the port within
    # the same process (execvpe keeps the PID), so by the time the adapter's
    # ping succeeds this file is already in place.
    write_sidecar_identity(
        port=int(args.port),
        family=family,
        model=str(args.model),
        embodiment_tag=str(args.embodiment_tag),
        quantization=str(args.quantization),
    )
    print(
        f"[{label}] launching server: model={args.model} port={args.port} "
        f"quant={args.quantization} embodiment={args.embodiment_tag}",
        flush=True,
    )
    env = make_isolated_env(venv)
    exec_server(venv, wrapper, env)
    return 0  # unreachable — execvpe replaced the process
