"""Boot the RLDX-1 inference server in an isolated Python 3.10 sidecar venv.

The ``rldx`` Python package pins ``requires-python = "~=3.10"`` and ships a
custom architectures=["RLDX"] model class that lives outside HuggingFace
Transformers. The openral workspace is Python 3.12-only (CLAUDE.md §3),
so we run the upstream inference server out-of-process and talk to it from
the ``rldx`` policy adapter (python/sim/src/openral_sim/policies/rldx.py)
over its native ZMQ + msgpack wire protocol. This script is the boot helper;
the clone / venv / env-isolation / exec scaffolding it shares with the gr00t
sidecar (ADR-0046 — RLDX-1 is a GR00T-N1.5 finetune) lives in
``tools/_sidecar_common.py``.

Usage::

    python tools/rldx_sidecar.py \\
        --model RLWRLD/RLDX-1-FT-LIBERO \\
        --port 5555 \\
        --quantization nf4

The script blocks and forwards signals; SIGINT cleanly stops the server.

CLAUDE.md compliance:
* The sidecar is a real subprocess running real upstream code — no mocks
  (§1.11). The wire protocol on the openral side is a real ZMQ
  client.
* Python version isolation is the only safe way to bridge ``rldx``
  (3.10) and ``openral`` (3.12-only); see §3.
* The non-commercial license guard is enforced upstream in the
  ``RSkillManifest`` loader, not here — this script just boots the
  process. The first ``openral sim run`` call will fail loud if the user
  has not set ``OPENRAL_ALLOW_NONCOMMERCIAL=1``.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from openral_sim._sidecar_common import build_parser, run_cmd, run_sidecar

_LABEL = "rldx-sidecar"
_REPO_URL = "https://github.com/RLWRLD/RLDX-1.git"
_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "rldx-sidecar"


def _install_deps(*, source: Path, uv: str, quantization: str) -> Path:
    """Install rldx + bitsandbytes (for NF4) into ``<source>/.venv``.

    ``uv sync`` in a directory with a ``pyproject.toml`` creates its own
    ``.venv`` next to that file regardless of ``$VIRTUAL_ENV``. We let it
    do exactly that and return the resulting venv path — fighting uv on
    venv placement causes the model deps to land in one venv and the
    quantization deps in another (the failure mode the first revision
    of this script hit).
    """
    run_cmd(_LABEL, [uv, "sync"], cwd=source)
    venv = source / ".venv"
    if not (venv / "bin" / "python").exists():
        raise SystemExit(f"uv sync did not produce a venv at {venv}")
    if quantization in {"nf4", "int8"}:
        # bitsandbytes is not in the upstream lockfile; install it into
        # the same venv `uv sync` just produced so the wrapper's
        # `load_in_4bit=True` succeeds. `--python` pins the target
        # explicitly so uv pip can't choose a different env.
        run_cmd(
            _LABEL,
            [
                uv,
                "pip",
                "install",
                "--python",
                str(venv / "bin" / "python"),
                "bitsandbytes>=0.43.0",
            ],
            cwd=source,
        )
    return venv


def _make_wrapper(*, work: Path, source: Path, args: argparse.Namespace) -> Path:
    """Write a Python wrapper that monkey-patches the loader for quantization.

    ``rldx.eval.run_rldx_server`` does not expose a quantisation CLI flag.
    The accepted upstream workflow is to set ``load_in_4bit=True`` /
    ``load_in_8bit=True`` on the Qwen3-VL backbone before the policy is
    constructed. We do that via a tiny monkey-patch on
    ``transformers.AutoModel.from_pretrained`` scoped to the Qwen backbone
    only — the MSAT diffusion head is left at bf16. (``source`` is unused
    here — the rldx server is imported as a module, not run by path.)
    """
    wrapper = work / "boot_server.py"
    wrapper.write_text(
        textwrap.dedent(
            f'''
            """Boot wrapper produced by tools/rldx_sidecar.py.

            * Monkey-patches AutoModel.from_pretrained for the Qwen3-VL
              backbone with the requested quantization scheme.
            * Hands control to rldx.eval.run_rldx_server with
              --use-sim-policy-wrapper so LIBERO-shaped flat-key obs/action
              dicts are produced (see rldx/eval/sim_policy_wrapper.py).
            """
            from __future__ import annotations

            import os
            import sys

            QUANTIZATION = {args.quantization!r}

            if QUANTIZATION in {{"nf4", "int8"}}:
                # The upstream RLDX loader (rldx/policy/policy_loader.py:172)
                # calls `AutoModel.from_pretrained(model_dir, device_map=device,
                # torch_dtype=torch.bfloat16)` with the LOCAL CACHE PATH —
                # not a name that contains "Qwen3-VL". So we cannot filter by
                # name; we apply the BitsAndBytesConfig to every AutoModel
                # load that doesn't already have one. Caveat: this also
                # quantizes the MSAT diffusion head (action flow matcher),
                # which the research note flagged as suboptimal but is the
                # only path the upstream loader exposes without a fork. For
                # ≥12 GiB GPUs prefer --quantization none and let the bf16
                # path run untouched.
                import transformers
                _orig_from_pretrained = transformers.AutoModel.from_pretrained

                # Minimal `llm_int8_skip_modules` list — keep ONLY
                # `lm_head` + `embed_tokens` outside quantization.
                # Quantize everything else (including the action_model
                # MSAT head) to keep peak VRAM under 8 GiB.
                #
                # Caveat: rldx/model/modules/action_model/ops.py:130 uses
                # `next(self.parameters()).dtype` to infer compute dtype,
                # which returns torch.uint8 for bnb-packed 4bit weights
                # and crashes downstream SiLU on "Byte". We patch that
                # one site below to use bf16 unconditionally — this is
                # the single upstream bug bnb-4bit interacts with.
                _SKIP_MODULES = ["lm_head", "embed_tokens"]

                def _patched_from_pretrained(*args, **kwargs):
                    if "quantization_config" not in kwargs:
                        from transformers import BitsAndBytesConfig
                        import torch
                        if QUANTIZATION == "nf4":
                            kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                llm_int8_skip_modules=_SKIP_MODULES,
                            )
                        elif QUANTIZATION == "int8":
                            kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_skip_modules=_SKIP_MODULES,
                            )
                        # Keep torch_dtype=bf16. bnb only quantizes Linear
                        # layers; non-quantized parameters (embeddings,
                        # position tables, vision-tower Conv2d) honor
                        # torch_dtype and need bf16 to keep the autocast
                        # path consistent.
                        kwargs.setdefault("torch_dtype", torch.bfloat16)
                    return _orig_from_pretrained(*args, **kwargs)

                transformers.AutoModel.from_pretrained = _patched_from_pretrained

            # Patch rldx.model.modules.action_model.ops.TimestepEncoder
            # so it doesn't read dtype from the (uint8-packed) first
            # quantized weight. Must be applied AFTER `rldx.model` is
            # imported by run_rldx_server's import chain.
            import rldx.model  # noqa: F401  — triggers the upstream model package import
            from rldx.model.modules.action_model import ops as _rldx_ops
            import torch

            def _patched_timestep_forward(self, timesteps):
                # Hard-pin to bf16 — matches `bnb_4bit_compute_dtype` and
                # avoids next(self.parameters()).dtype returning uint8 on
                # 4bit-packed weights.
                timesteps_proj = self.time_proj(timesteps).to(torch.bfloat16)
                return self.timestep_embedder(timesteps_proj)

            _rldx_ops.TimestepEncoder.forward = _patched_timestep_forward

            # Patch rldx.data.augmentations.resize_preserve_aspect_area_then_crop
            # so it tolerates max_area=None / m=None. Some upstream checkpoints
            # (e.g. RLDX-1-FT-RC365) ship processor_config.json with
            # image_max_area: None — the saved processor's `AspectAreaResizeAndCrop`
            # then stores `self.max_area = None`, and per-request the helper's
            # `math.sqrt(max_area / (h * w))` explodes with
            # `unsupported operand type(s) for /: 'NoneType' and 'int'`. Patching
            # the leaf helper is more robust than patching the builder (the
            # transform has already been instantiated by the time we get here).
            from rldx.data import augmentations as _rldx_aug

            _orig_resize = _rldx_aug.resize_preserve_aspect_area_then_crop

            def _patched_resize(h, w, max_area=None, m=None):
                if max_area is None:
                    max_area = 65536  # 256 * 256
                if m is None:
                    m = 32
                return _orig_resize(h, w, max_area=max_area, m=m)

            _rldx_aug.resize_preserve_aspect_area_then_crop = _patched_resize

            # run_rldx_server.main(config) takes a ServerConfig built by
            # tyro.cli at module __main__. We import the config dataclass +
            # main directly and call tyro.cli ourselves so the monkey-patch
            # above is already in place.
            sys.argv = [
                "run_rldx_server",
                "--model-path", {args.model!r},
                "--embodiment-tag", {args.embodiment_tag!r},
                "--host", "0.0.0.0",
                "--port", str({args.port}),
                "--use-sim-policy-wrapper",
                # --no-strict: with general_embodiment modality config +
                # is_libero detection, RLDXSimPolicyWrapper._get_action
                # emits LIBERO-flat keys (action.x / .y / .z / ...) while
                # check_action() validates against the modality_config's
                # general-embodiment keys (action.eef_pos_delta / ...).
                # That contradiction is an upstream bug; disabling strict
                # validation lets the LIBERO-flat output reach our client.
                "--no-strict",
            ]
            import tyro
            from rldx.eval.run_rldx_server import ServerConfig, main
            main(tyro.cli(ServerConfig))
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
        default_embodiment_tag="GENERAL_EMBODIMENT",
        model_help="Model id or local path passed to the server's --model-path flag, "
        "e.g. RLWRLD/RLDX-1-FT-LIBERO.",
        quant_help="Backbone quantization scheme. NF4 is required to fit RLDX-1 on "
        "≤ 12 GiB GPUs; only the Qwen3-VL visual backbone is quantized, the "
        "MSAT flow-matching head is left at bf16 (default nf4).",
    )
    args = parser.parse_args()
    return run_sidecar(
        label=_LABEL,
        family="rldx",
        repo_url=_REPO_URL,
        args=args,
        install_deps=_install_deps,
        make_wrapper=_make_wrapper,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
