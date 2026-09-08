"""Boot the NVIDIA Cosmos 3 reasoner behind vLLM's OpenAI-compatible API.

Serves the ``cosmos3-edge`` reasoner model (``OPENRAL_REASONER_MODEL=cosmos3-edge``,
``openral_reasoner.cosmos3.Cosmos3ToolUseClient``): the reasoner tower of
NVIDIA Cosmos 3, default 4B on-device Edge tier (``nvidia/Cosmos3-Edge``,
OpenMDW-1.1, commercial OK). vLLM serves only the reasoner tower via
chat-completions + tool calling (CLAUDE.md §3: typed tool-use, no free-form
JSON). Out-of-process for dependency isolation (vLLM pins its own torch/CUDA)
and VRAM isolation — same ``uv`` venv + ``os.execvpe`` pattern as
``tools/qwen_vlm_sidecar.py``. ``--tool-call-parser`` defaults to ``hermes``
(Qwen3-VL-compatible; Nano/Super are Qwen3-VL-initialised, Edge is a
Nemotron backbone that keeps the message format).

Edge ships as a diffusers ``Cosmos3OmniPipeline`` (weights under
``transformer/``, ``vision_encoder/``). Transformers-fallback vLLM only
resolves bare top-level filenames, so ``materialize_reasoner_view``
builds a flat symlinked view (RTX 4070 8 GB: ~6.4 GB resident BF16,
8192-token KV). Native vLLM (vllm-project/vllm#48291) reads the diffusers
layout by path and the flattened view **breaks** it (Jetson AGX Thor, vLLM
0.28.0: ``RuntimeError: Cannot find any model weights``) — serve the
snapshot as-is there (3 shards, 4.66 GiB, 5.71 s).
``vllm_has_native_edge_model`` asks the venv's ``ModelRegistry`` rather
than comparing version strings.

Lock resolves differently per platform: x86_64 pins vLLM 0.24.0, whose
Transformers fallback loads Edge but crashes in
``cosmos3_edge.get_rope_index`` (1-D vs 2-D ``input_ids``; model itself is
sound at ~46 tok/s under plain ``transformers.generate``). aarch64 resolves
vLLM 0.28.0, which already carries #48291 and loads natively (validated live
tool call on RTX 4070, 1.5-2 s/tick warm, ``--kv-cache-dtype fp8`` for the
8192 window). Retire ``materialize_reasoner_view`` once every platform
resolves a vLLM with #48291. Full findings:
``docs/reference/cosmos3-edge-reasoner.md``.

Usage::

    python tools/cosmos3_reasoner_sidecar.py --port 8901
    python tools/cosmos3_reasoner_sidecar.py --model nvidia/Cosmos3-Nano

Blocks and forwards signals; SIGINT stops cleanly. First boot downloads ~9 GB
BF16 weights (``uvx hf auth login`` first if needed).

CLAUDE.md compliance: real subprocess, no mocks (§1.11); weights OpenMDW-1.1
(commercial + noncommercial OK), no license guard needed (§1.9).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from openral_sim._sidecar_common import (
    alloc_conf_var,
    ensure_pip_venv,
    run_cmd,
    venv_torch_version,
)

_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "cosmos3-reasoner-sidecar"
_VENV_ENV = "OPENRAL_COSMOS3_SIDECAR_VENV"
_HOME_ENV = "OPENRAL_COSMOS3_SIDECAR_HOME"

# vllm>=0.23.0 is the first release with Cosmos 3 reasoner serving (Nano/Super
# reuse upstream Qwen3-VL support; Edge has a dedicated integration). Pure-PyPI
# resolution, no --torch-backend: vLLM's torch wheels bundle the CUDA runtime.
# Regenerate: uv pip compile tools/sidecar_requirements/cosmos3_reasoner.in
#   --universal --generate-hashes --python-version 3.12
#   -o tools/sidecar_requirements/cosmos3_reasoner.lock
_LOCK = Path(__file__).resolve().parent / "sidecar_requirements" / "cosmos3_reasoner.lock"

# TEMPORARY: `cosmos3_edge` (Edge's Nemotron-backbone reasoner, released
# 2026-07-20) needs transformers `main` (huggingface/transformers#47181 +
# #47399); no release has it as of 2026-07-20 (5.14.1, what the lock
# resolves, rejects the checkpoint: "does not recognize this architecture").
# Verified on RTX 4070 at this SHA. REMOVE once a transformers release
# >5.14.1 ships cosmos3_edge: drop this + the overlay install below, pin
# `transformers>=<that version>` in cosmos3_reasoner.in, recompile the lock.
_TRANSFORMERS_EDGE_SHA = "cbf4d720ec734edb77d25452b2790c7e4be2f8d7"


def ensure_venv(home: Path, *, override: str | None = None) -> Path:
    """Return the sidecar venv python, creating + populating it if needed.

    ``override`` (or ``$OPENRAL_COSMOS3_SIDECAR_VENV``) points at an existing
    venv to reuse instead of provisioning one under ``home``. Otherwise a
    Python 3.12 venv is provisioned from the pinned
    ``cosmos3_reasoner.lock`` for reproducibility (CLAUDE.md §1.8), plus the
    SHA-pinned transformers overlay (see ``_TRANSFORMERS_EDGE_SHA``).
    """

    def _install(uv: str, py: Path) -> None:
        # Pure PyPI, no --torch-backend (see _LOCK comment). No --require-hashes
        # — uv still verifies the recorded hashes for everything it installs.
        run_cmd(
            "cosmos3-sidecar",
            [uv, "pip", "install", "--python", str(py), "-r", str(_LOCK)],
        )
        # SHA-pinned git overlay (see _TRANSFORMERS_EDGE_SHA); runs inside the
        # same ensure_pip_venv sentinel, so it happens exactly once per venv.
        run_cmd(
            "cosmos3-sidecar",
            [
                uv,
                "pip",
                "install",
                "--python",
                str(py),
                "transformers @ git+https://github.com/huggingface/transformers.git"
                f"@{_TRANSFORMERS_EDGE_SHA}",
            ],
        )

    return ensure_pip_venv(
        label="cosmos3-sidecar",
        home=home,
        python="3.12",
        install=_install,
        override=override,
        override_env=_VENV_ENV,
        # Keyed on the lock text + the overlay SHA so bumping either repairs an
        # existing venv instead of being ignored.
        spec=(_LOCK.read_text(encoding="utf-8"), _TRANSFORMERS_EDGE_SHA),
    )


def is_diffusers_reasoner_layout(model_dir: Path) -> bool:
    """True when ``model_dir`` is an Edge-style diffusers pipeline needing a view.

    The tell is a top-level ``model.safetensors.index.json`` whose weight_map
    points into subfolders (``transformer/…``) — the layout vLLM cannot load
    directly. Nano/Super (standard top-level ``model-*.safetensors``) return
    False and are served as-is.
    """
    idx = model_dir / "model.safetensors.index.json"
    if not idx.is_file():
        return False
    try:
        weight_map: dict[str, str] = json.loads(idx.read_text())["weight_map"]
    except (json.JSONDecodeError, KeyError, OSError):
        return False
    return any("/" in dest for dest in weight_map.values())


def materialize_reasoner_view(model_dir: Path, dest: Path) -> Path:
    """Build a vLLM-loadable flat view of an Edge diffusers checkpoint.

    Symlinks each unique weight shard to a bare name at ``dest``, writes a copy
    of ``model.safetensors.index.json`` with the weight_map rewritten to those
    bare names, and symlinks the config/tokenizer files vLLM + the processor
    need. Idempotent: existing links are refreshed. Returns ``dest``.

    This is the fix for vLLM's "Cannot find any model weights" on the Edge
    layout (see the module docstring). Verified live on an RTX 4070.
    """
    dest.mkdir(parents=True, exist_ok=True)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    # Rewrite weight_map subfolder paths to bare basenames, and remember the
    # subfolder source for each unique shard so we can symlink it in.
    shard_src: dict[str, str] = {}
    for tensor, rel in index["weight_map"].items():
        base = os.path.basename(rel)
        index["weight_map"][tensor] = base
        shard_src[base] = rel
    (dest / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    for base, rel in shard_src.items():
        _relink(dest / base, (model_dir / rel).resolve())
    # Non-weight files vLLM + the HF processor read (top-level only).
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    ):
        src = model_dir / name
        if src.exists():
            _relink(dest / name, src.resolve())
    return dest


def _relink(link: Path, target: Path) -> None:
    """Point ``link`` at ``target`` (replacing any existing link/file)."""
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def vllm_has_native_edge_model(py: Path) -> bool:
    """Whether the serving venv's vLLM ships the native Edge model.

    ``Cosmos3EdgeForConditionalGeneration`` arrived upstream in
    `vllm#48291 <https://github.com/vllm-project/vllm/pull/48291>`_, after the
    0.24.0 release the x86 branch pins but before the 0.28.0 the aarch64
    branch resolves (see module docstring); decides how weights are served
    in ``resolve_served_model``. Asks vLLM's registry rather than
    comparing version strings.

    Args:
        py: The serving venv's interpreter.

    Returns:
        ``True`` only on a clean, affirmative answer. Any failure (import
        error, timeout, too old a vLLM) returns ``False``, selecting the
        flattened-view path.
    """
    probe = (
        "from vllm.model_executor.models.registry import ModelRegistry;"
        "print('Cosmos3EdgeForConditionalGeneration' in ModelRegistry.get_supported_archs())"
    )
    try:
        done = subprocess.run(  # reason: argv list, interpreter path is ours
            [str(py), "-c", probe],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.stdout.strip().splitlines()[-1:] == ["True"]


def resolve_served_model(
    model: str, home: Path, *, native_edge: bool = False
) -> tuple[str, str | None]:
    """Resolve ``model`` to a vLLM ``serve`` target + optional served-model-name.

    A local directory is served verbatim (``served_name=None``). A HF repo id
    is snapshot-downloaded; an Edge diffusers layout is served as-is when
    ``native_edge`` (see module docstring for why the two vLLM generations
    read it differently), else via ``materialize_reasoner_view``.
    ``served_name = model`` keeps the client's ``model_id`` matching either
    way. Nano/Super (standard layout) are served by repo id directly.

    Args:
        model: Repo id or local directory.
        home: Sidecar work directory the flattened view is built under.
        native_edge: Whether the serving vLLM has the native Edge model —
            ``vllm_has_native_edge_model`` answers this.

    Returns:
        ``(serve_target, served_model_name_or_None)``.
    """
    if Path(model).is_dir():
        return model, None
    # Lazy import: only the boot path needs it; provided by the openral env.
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(model))
    if not is_diffusers_reasoner_layout(snapshot):
        return model, None
    if native_edge:
        return str(snapshot), model
    view = materialize_reasoner_view(snapshot, home / f"{model.rsplit('/', 1)[-1]}-reasoner")
    return str(view), model


def build_serve_argv(
    *,
    vllm_bin: Path,
    model: str,
    host: str,
    port: int,
    tool_call_parser: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    served_model_name: str | None = None,
    kv_cache_dtype: str = "auto",
) -> list[str]:
    """Build the ``vllm serve`` argv (split out for unit-testability).

    ``--enable-auto-tool-choice`` + ``--tool-call-parser`` turn on OpenAI tool
    calling (the reasoner sends ``tool_choice="required"``). ``--max-model-len``
    caps the KV cache: Cosmos 3 supports up to 256K tokens, far beyond an
    8-32 GB edge GPU; the default 8192 covers the system prompt + tool
    schemas (~4-5K tokens). ``served_model_name`` keeps the public id stable
    when serving a local view dir (see ``resolve_served_model``).
    """
    argv = [
        str(vllm_bin),
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--async-scheduling",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        tool_call_parser,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if enforce_eager:
        argv.append("--enforce-eager")
    if served_model_name is not None:
        argv += ["--served-model-name", served_model_name]
    if kv_cache_dtype != "auto":
        argv += ["--kv-cache-dtype", kv_cache_dtype]
    return argv


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="nvidia/Cosmos3-Edge")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8901)
    p.add_argument("--tool-call-parser", default="hermes")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("OPENRAL_COSMOS3_GPU_MEM_UTIL", "0.90")),
        help="vLLM GPU memory fraction (default 0.90, or $OPENRAL_COSMOS3_GPU_MEM_UTIL). "
        "vLLM's own 0.92 default fails when the GPU already hosts a display / another "
        "process; drop it further if engine init reports 'free memory on startup'.",
    )
    p.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Enable CUDA-graph capture (off by default: capture overflows an 8 GB "
        "card's VRAM and adds minutes to startup; the 0.2 Hz reasoner doesn't need it).",
    )
    p.set_defaults(enforce_eager=True)
    p.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="vLLM KV-cache dtype (default auto = model dtype). On an 8 GB card pass "
        "'fp8' — halves the KV footprint; with the native Edge impl (vLLM main) the "
        "8192-token window only fits 8 GB with fp8 KV (verified live on RTX 4070).",
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

    env = os.environ.copy()
    # Drop PYTHONPATH/PYTHONHOME so the sidecar interpreter boots from its own
    # site-packages — ROS 2 / colcon populate PYTHONPATH with the workspace
    # wheels, which would shadow the sidecar's pinned deps.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # Fragmentation fix (RTX 4070 8 GB: without it Edge OOMs with ~670 MiB
    # "reserved but unallocated"). Must precede the first CUDA allocation.
    # torch renamed the var in 2.9 (PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF,
    # warns on the old spelling), so resolve it from the venv's actual torch.
    env.setdefault(alloc_conf_var(venv_torch_version(py.parent.parent)), "expandable_segments:True")

    # FlashInfer's sampler JIT-compiles and needs CUDA toolkit headers a
    # JetPack 7 Jetson lacks: engine init died in `determine_available_memory`
    # with `flashinfer/sampling.cuh:20:10: fatal error: curand.h: No such
    # file or directory`. Prefer vLLM's PyTorch sampler; an operator with the
    # headers can export the var to re-enable it.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    # snapshot dir as-is for the native model, flattened view for the
    # Transformers fallback (see module docstring); served_name keeps the
    # public model id stable.
    native_edge = vllm_has_native_edge_model(py)
    served_path, served_name = resolve_served_model(args.model, args.home, native_edge=native_edge)

    cmd = build_serve_argv(
        vllm_bin=py.parent / "vllm",
        model=served_path,
        host=args.host,
        port=args.port,
        tool_call_parser=args.tool_call_parser,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        served_model_name=served_name,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    if "edge" in args.model.lower() and not native_edge:
        # Only true without the native Edge model (see module docstring):
        # avoid a false warning on a vLLM carrying vllm#48291.
        print(
            "[cosmos3-sidecar] WARNING: this vLLM has no native "
            "Cosmos3EdgeForConditionalGeneration, so the Edge server boots via the "
            "Transformers fallback and is expected to fail on inference (upstream "
            "cosmos3_edge get_rope_index bug). See "
            "docs/reference/cosmos3-edge-reasoner.md; use a cloud/local reasoner "
            "baseline meanwhile, or MODEL=nvidia/Cosmos3-Nano on a >=24 GB GPU.",
            flush=True,
        )
    print(
        f"[cosmos3-sidecar] launching vllm serve: model={args.model} "
        f"(served from {served_path}) port={args.port}",
        flush=True,
    )
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main() or 0)
