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

Usage::

    python tools/cosmos3_reasoner_sidecar.py --port 8901
    python tools/cosmos3_reasoner_sidecar.py --model nvidia/Cosmos3-Nano

The script blocks and forwards signals; SIGINT cleanly stops the server. The
first boot downloads ~8 GB of BF16 weights from HF Hub (authenticate with
``uvx hf auth login`` first if needed).

CLAUDE.md compliance:
* Real subprocess running the real upstream serving stack — no mocks (§1.11).
* Cosmos 3 weights are OpenMDW-1.1 (commercial + noncommercial OK) — no
  license guard needed here (§1.9); posture recorded in
  ``docs/reference/cosmos3-edge-reasoner.md``.
"""

from __future__ import annotations

import argparse
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


def ensure_venv(home: Path, *, override: str | None = None) -> Path:
    """Return the sidecar venv python, creating + populating it if needed.

    ``override`` (or ``$OPENRAL_COSMOS3_SIDECAR_VENV``) points at an existing
    venv to reuse instead of provisioning one under ``home``. Otherwise a
    Python 3.12 venv is provisioned from the pinned
    ``cosmos3_reasoner.lock`` for reproducibility (CLAUDE.md §1.8).
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

    return ensure_pip_venv(
        label="cosmos3-sidecar",
        home=home,
        python="3.12",
        install=_install,
        override=override,
        override_env=_VENV_ENV,
    )


def build_serve_argv(
    *,
    vllm_bin: Path,
    model: str,
    host: str,
    port: int,
    tool_call_parser: str,
    max_model_len: int,
) -> list[str]:
    """Build the ``vllm serve`` argv (split out for unit-testability).

    ``--enable-auto-tool-choice`` + ``--tool-call-parser`` turn on OpenAI
    ``tools`` / ``tool_calls`` support (the reasoner sends
    ``tool_choice="required"``). ``--max-model-len`` caps the KV cache: the
    Cosmos 3 reasoner supports up to 256K tokens, far beyond what a 8–32 GB
    edge GPU can cache; a reasoner tick's context is a few thousand tokens.
    """
    return [
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
    ]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="nvidia/Cosmos3-Edge")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8901)
    p.add_argument("--tool-call-parser", default="hermes")
    p.add_argument("--max-model-len", type=int, default=32768)
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

    cmd = build_serve_argv(
        vllm_bin=py.parent / "vllm",
        model=args.model,
        host=args.host,
        port=args.port,
        tool_call_parser=args.tool_call_parser,
        max_model_len=args.max_model_len,
    )
    print(
        f"[cosmos3-sidecar] launching vllm serve: model={args.model} port={args.port}",
        flush=True,
    )
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main() or 0)
