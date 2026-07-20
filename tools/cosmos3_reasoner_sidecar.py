"""Boot the NVIDIA Cosmos 3 reasoner behind vLLM's OpenAI-compatible API.

The ``cosmos`` reasoner provider (``OPENRAL_REASONER_LLM_PROVIDER=cosmos``,
:class:`openral_reasoner.cosmos3.Cosmos3ToolUseClient`) plans with the
**reasoner tower** of an NVIDIA Cosmos 3 omnimodal world model — by default the
4B on-device **Edge** tier (``nvidia/Cosmos3-Edge``, OpenMDW-1.1, commercial
OK). vLLM loads only the autoregressive reasoner tower (not the diffusion
generator) and serves the standard chat-completions API with tool calling, so
the reasoner's typed tool-use contract (CLAUDE.md §3 — provider tool-use API,
no free-form JSON) is preserved end to end.

The server runs **out-of-process** for the same three reasons as the Qwen
scene-VLM sidecar (`tools/qwen_vlm_sidecar.py`):

* **Dependency isolation.** vLLM pins its own torch/CUDA stack; resolving it
  into the lerobot-pinned openral runtime venv would perturb the VLA stack.
* **VRAM / process isolation.** A served 4B model + CUDA context should not
  live in the ``rclpy`` reasoner process; an OOM in the server must not take
  down the reasoner node.
* **Same pattern as the rest of the tree** — provision an isolated venv with
  ``uv``, then ``os.execvpe`` into the server. The transport here is vLLM's
  own HTTP API rather than ZMQ because the OpenAI-compatible surface *is* the
  provider contract the reasoner already speaks.

Tool-call parsing: the Cosmos 3 reasoner follows Qwen3-VL-compatible message
conventions (Nano/Super are Qwen3-VL-initialised; Edge is trained from scratch
on a Nemotron backbone but keeps the message format), so the default
``--tool-call-parser hermes`` matches the Qwen-style ``<tool_call>`` emission.
Override with ``--tool-call-parser`` if a future vLLM ships a dedicated cosmos3
parser.

Edge weight layout — the flattened reasoner view. ``nvidia/Cosmos3-Edge`` is
published as a *diffusers* ``Cosmos3OmniPipeline``: the reasoner weights live in
the ``transformer/`` (and ``vision_encoder/``) subfolders, and the top-level
``model.safetensors.index.json`` maps every tensor to those subfolder paths.
vLLM's weight loader only resolves *bare top-level* filenames, so serving the
repo id directly fails with "Cannot find any model weights". :func:`materialize_reasoner_view`
builds a small directory of symlinks (bare-named shards + a rewritten index +
tokenizer/config) that vLLM can read; the sidecar serves that view with
``--served-model-name nvidia/Cosmos3-Edge`` so the client's ``model_id`` still
matches. Verified live on an RTX 4070 (8 GB): with the view + the transformers
overlay below, vLLM loads the reasoner (~6.4 GB resident at BF16, 8192-token KV)
and reports "Application startup complete".

KNOWN UPSTREAM BLOCKER (2026-07-20). Once loaded, the *first forward pass*
crashes in ``transformers.models.cosmos3_edge.modeling_cosmos3_edge.get_rope_index``
(``IndexError: too many indices for tensor of dimension 1``): vLLM's
Transformers-multimodal backend calls the model's mRoPE ``get_rope_index`` with a
1-D ``input_ids`` where the 5-day-old modeling code expects 2-D ``(batch, seq)``.
This affects **every** request (text-only and vision+text), so end-to-end serving
is blocked until the vLLM×transformers cosmos3_edge integration is fixed
upstream. The model itself is sound: under plain ``transformers.generate`` on the
same 4070 it produced coherent physical-reasoning text at ~46 tok/s. Full
findings + reproduction in ``docs/reference/cosmos3-edge-reasoner.md``.

Usage::

    python tools/cosmos3_reasoner_sidecar.py --port 8901
    python tools/cosmos3_reasoner_sidecar.py --model nvidia/Cosmos3-Nano

The script blocks and forwards signals; SIGINT cleanly stops the server. The
first boot downloads ~9 GB of BF16 weights from HF Hub (authenticate with
``uvx hf auth login`` first if needed).

CLAUDE.md compliance:
* Real subprocess running the real upstream serving stack — no mocks (§1.11).
* Cosmos 3 weights are OpenMDW-1.1 (commercial + noncommercial OK) — no
  license guard needed here (§1.9); posture recorded in
  ``docs/reference/cosmos3-edge-reasoner.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openral_sim._sidecar_common import ensure_pip_venv, run_cmd

_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "cosmos3-reasoner-sidecar"
_VENV_ENV = "OPENRAL_COSMOS3_SIDECAR_VENV"
_HOME_ENV = "OPENRAL_COSMOS3_SIDECAR_HOME"

# Pinned deps for the serving venv. vllm>=0.23.0 is the first release line
# with Cosmos 3 reasoner serving (Nano/Super reuse upstream Qwen3-VL support;
# the Edge tier's Nemotron-backbone reasoner has a dedicated integration).
# Pure-PyPI resolution (no --torch-backend): vLLM's torch dependency ships the
# CUDA runtime in its PyPI wheels, so no PyTorch index redirect is needed —
# unlike the transformers-based sidecars. Regenerate after editing the .in with:
#   uv pip compile tools/sidecar_requirements/cosmos3_reasoner.in \
#     --universal --generate-hashes --python-version 3.12 \
#     -o tools/sidecar_requirements/cosmos3_reasoner.lock
_LOCK = Path(__file__).resolve().parent / "sidecar_requirements" / "cosmos3_reasoner.lock"

# TEMPORARY overlay, pinned for reproducibility: `cosmos3_edge` (the Edge
# tier's Nemotron-backbone reasoner, released 2026-07-20) is on transformers
# `main` (huggingface/transformers#47181 + the #47399 packing-order fix) but
# in NO released version as of 2026-07-20 — the newest release (5.14.1, which
# the lock resolves) rejects the checkpoint with "does not recognize this
# architecture". Verified live on RTX 4070: with this exact SHA the serve
# proceeds via vLLM's Transformers multimodal fallback backend.
# REMOVE when a transformers release >5.14.1 ships cosmos3_edge: delete this
# constant + the overlay install below and add `transformers>=<that version>`
# to cosmos3_reasoner.in, then recompile the lock.
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
        # Same install shape as tools/qwen_vlm_sidecar.py minus the PyTorch
        # index redirect: the lock is pure PyPI (vLLM's torch wheels bundle
        # the CUDA runtime). No --require-hashes — uv still verifies the
        # recorded hashes for everything it installs.
        run_cmd(
            "cosmos3-sidecar",
            [uv, "pip", "install", "--python", str(py), "-r", str(_LOCK)],
        )
        # SHA-pinned git overlay for cosmos3_edge support (temporary — see
        # the _TRANSFORMERS_EDGE_SHA comment). Runs inside the same
        # ensure_pip_venv sentinel, so it happens exactly once per venv.
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


def resolve_served_model(model: str, home: Path) -> tuple[str, str | None]:
    """Resolve ``model`` to a vLLM ``serve`` target + optional served-model-name.

    * A local directory is served verbatim (``served_name`` = ``None``).
    * A HF repo id is snapshot-downloaded; an Edge diffusers layout is turned
      into a flat reasoner view (:func:`materialize_reasoner_view`) served with
      ``served_name = model`` so the client's ``model_id`` still matches, while
      Nano/Super (standard layout) are served by their repo id directly.
    """
    if Path(model).is_dir():
        return model, None
    # Lazy import: only the boot path needs it; provided by the openral env.
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(model))
    if not is_diffusers_reasoner_layout(snapshot):
        return model, None
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
) -> list[str]:
    """Build the ``vllm serve`` argv (split out for unit-testability).

    ``--enable-auto-tool-choice`` + ``--tool-call-parser`` turn on OpenAI
    ``tools`` / ``tool_calls`` support (the reasoner sends
    ``tool_choice="required"``). ``--max-model-len`` caps the KV cache: the
    Cosmos 3 reasoner supports up to 256K tokens, far beyond what an 8–32 GB
    edge GPU can cache; the default 8192 holds the reasoner's system prompt +
    tool schemas (~4–5K tokens) with headroom and fits 8 GB. ``--enforce-eager``
    skips CUDA-graph capture — on an 8 GB card that capture both overflows VRAM
    and adds minutes to startup; the reasoner's 0.2 Hz cadence does not need it.
    ``--gpu-memory-utilization`` defaults to 0.90; verified live on an empty
    8 GB RTX 4070 the reasoner loads at ~6.4 GB with an 8192-token KV cache.
    ``served_model_name`` keeps the public id stable when serving a local view
    dir (see :func:`resolve_served_model`).
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

    # Download + (for the Edge diffusers layout) build the flat reasoner view
    # vLLM can load; served_name keeps the public model id stable.
    served_path, served_name = resolve_served_model(args.model, args.home)

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
    )
    print(
        f"[cosmos3-sidecar] launching vllm serve: model={args.model} "
        f"(served from {served_path}) port={args.port}",
        flush=True,
    )
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main() or 0)
