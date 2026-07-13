"""Boot the InternVLA-N1 navigation server in an isolated sidecar venv.

InternRobotics' InternNav pins ``transformers==4.51.0`` (plus a matching
diffusers/accelerate set) for InternVLA-N1 / DualVLN inference — incompatible
with the openral py3.12 workspace's transformers 5.x. Like the ``rldx``
sidecar, we run inference out-of-process and talk to it over ZMQ + msgpack
from the ``internvla_n1`` policy adapter
(python/sim/src/openral_sim/policies/internvla_n1.py). The served process is
``tools/_internvla_n1_server.py``.

Unlike rldx (whose upstream lockfile we ``uv sync``), InternNav's setup.py
drags the full eval stack (habitat / isaac extras). We instead provision a
py3.11 venv with the explicit inference-only pin set from
``requirements/internvla_n1.txt`` upstream, install ``internnav`` and
``diffusion_policy`` with ``--no-deps`` (System-1's NavDP needs only
``SinusoidalPosEmb`` from the latter), and skip flash-attn — the server
loads the model with ``attn_implementation="sdpa"``.

Usage::

    python tools/internvla_n1_sidecar.py \\
        --model InternRobotics/InternVLA-N1-DualVLN \\
        --port 5781 \\
        --quantization nf4

CLAUDE.md compliance:
* Real upstream code in a real subprocess — no mocks (§1.11).
* transformers-pin isolation is the sanctioned sidecar pattern (§3).
* The CC-BY-NC-SA weight-license guard is enforced by the RSkillManifest
  loader (``OPENRAL_ALLOW_NONCOMMERCIAL=1``), not here (§1.9).
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from openral_sim._sidecar_common import build_parser, run_cmd, run_sidecar

_LABEL = "internvla-n1-sidecar"
_REPO_URL = "https://github.com/InternRobotics/InternNav.git"
_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "internvla-n1-sidecar"
_PYTHON = "3.11"

# Inference-only pin set, mirroring upstream requirements/internvla_n1.txt
# (transformers/diffusers/accelerate) minus flash-attn (sdpa instead) and
# minus depth-camera-filtering (only used by their Go2 ROS scripts). torch
# is pinned to the newest release transformers 4.51.0 supports.
_PINNED_DEPS = [
    "torch==2.6.0",
    "torchvision==0.21.0",
    "transformers==4.51.0",
    # 0.32.2 (NOT the 0.33.1 in requirements/internvla_n1.txt): the DualVLN
    # checkpoint's NextDiT was trained against 0.32.2's LuminaFeedForward, which
    # applies the SwiGLU `2/3` inner-dim reduction (1536→1024). 0.33.1 dropped
    # that, so its FFN builds 1536 and the checkpoint fails to load (size
    # mismatch on traj_dit.*.feed_forward.linear_{1,3}).
    "diffusers==0.32.2",
    "accelerate==1.4.0",
    "numpy<2",
    "pillow",
    "einops",
    "ftfy==6.3.1",
    "pydantic>=2",  # internnav.configs
    "opencv-python-headless",  # DepthAnythingV2 dpt encoder
    "timm",  # DepthAnythingV2 DINOv2 backbone
    "regex",  # LongCLIP tokenizer
    "setuptools<80",  # LongCLIP needs `pkg_resources.packaging` (dropped in setuptools 81+)
    "gym==0.26.2",  # encoder/__init__ eagerly imports resnet_encoders (uses gym.spaces)
    "pyzmq",
    "msgpack",
]
# System-1's NavDP imports exactly one symbol from diffusion_policy
# (SinusoidalPosEmb, torch-only) — install without its heavy dep tree.
_DIFFUSION_POLICY_PIN = (
    "diffusion_policy @ git+https://github.com/real-stanford/diffusion_policy.git"
    "@5ba07ac6661db573af695b419a7947ecb704690f"
)


def _install_deps(*, source: Path, uv: str, quantization: str) -> Path:
    """Provision ``<source>/.venv`` (py3.11) with the inference pin set."""
    # The shallow clone skips submodules; the encoder package eagerly imports
    # LongCLIP (a submodule), so fetch it. DualVLN's System-1 uses the
    # DepthAnythingV2 encoder, not LongCLIP, but the package __init__ imports
    # both — so the module must be present on disk to load the model at all.
    run_cmd(
        _LABEL,
        ["git", "submodule", "update", "--init", "internnav/model/basemodel/LongCLIP"],
        cwd=source,
    )
    venv = source / ".venv"
    py = venv / "bin" / "python"
    if not py.exists():
        run_cmd(_LABEL, [uv, "venv", str(venv), "--python", _PYTHON], cwd=source)
    pip = [uv, "pip", "install", "--python", str(py)]
    run_cmd(_LABEL, [*pip, *_PINNED_DEPS], cwd=source)
    run_cmd(_LABEL, [*pip, "--no-deps", _DIFFUSION_POLICY_PIN], cwd=source)
    run_cmd(_LABEL, [*pip, "--no-deps", "-e", str(source)], cwd=source)
    if quantization in {"nf4", "int8"}:
        run_cmd(_LABEL, [*pip, "bitsandbytes>=0.45.0"], cwd=source)
    return venv


def _make_wrapper(*, work: Path, source: Path, args: argparse.Namespace) -> Path:
    """Write a tiny argv shim that runs the real server file.

    The server logic lives in ``tools/_internvla_n1_server.py`` (a checked-in
    file, unlike rldx whose wrapper embeds monkey-patches); the shim only
    pins ``sys.argv`` so :func:`openral_sim._sidecar_common.exec_server`'s
    no-args exec contract holds. (``source`` unused — the server resolves
    ``internnav`` from the venv's editable install.)
    """
    server = Path(__file__).resolve().parent / "_internvla_n1_server.py"
    wrapper = work / "boot_server.py"
    wrapper.write_text(
        textwrap.dedent(
            f'''
            """Boot wrapper produced by tools/internvla_n1_sidecar.py."""
            import runpy
            import sys

            sys.argv = [
                "_internvla_n1_server",
                "--model", {args.model!r},
                "--host", "0.0.0.0",
                "--port", str({args.port}),
                "--quantization", {args.quantization!r},
            ]
            runpy.run_path({str(server)!r}, run_name="__main__")
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return wrapper


def main() -> int:
    parser = build_parser(
        description=__doc__ or "",
        default_home=_DEFAULT_HOME,
        default_embodiment_tag="mobile_base",
        model_help="Checkpoint HF id or local path, e.g. InternRobotics/InternVLA-N1-DualVLN.",
        quant_help="Backbone quantization. NF4 is required to fit the 8.3B "
        "checkpoint on ≤ 8 GiB GPUs; the System-1 NavDP head stays bf16 "
        "(default nf4).",
    )
    args = parser.parse_args()
    return run_sidecar(
        label=_LABEL,
        family="internvla_n1",
        repo_url=_REPO_URL,
        args=args,
        install_deps=_install_deps,
        make_wrapper=_make_wrapper,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
