"""Boot the NVIDIA Isaac GR00T inference server in an isolated Py3.10 sidecar.

Isaac-GR00T pins ``requires-python = ">=3.10,<3.11"`` (on x86 dGPU) plus
``flash-attn==2.7.4.post1`` + a CUDA toolchain. The openral workspace is
Python 3.12-only (CLAUDE.md §3), so we run the upstream GR00T ``PolicyServer``
out-of-process and talk to it from the ``gr00t`` policy adapter
(python/sim/src/openral_sim/policies/gr00t.py) over the same ZMQ + msgpack wire
the ``rldx`` adapter uses — RLDX-1 is a GR00T-N1.5 finetune, so the LIBERO-flat
``state.x`` / ``action.x`` contract is shared. This script is the boot helper;
the clone / venv / env-isolation / exec scaffolding it shares with the rldx
sidecar lives in ``tools/_sidecar_common.py``.

Usage::

    python tools/gr00t_sidecar.py \\
        --model nvidia/GR00T-N1.7-LIBERO/libero_spatial \\
        --port 5555 \\
        --quantization nf4 \\
        --embodiment-tag new_embodiment

VERIFICATION STATUS (ADR-0046 PR2): this boot helper is grounded in the
Isaac-GR00T server API (``gr00t/eval/run_gr00t_server.py`` → ``Gr00tPolicy`` +
``PolicyServer``, ``--use-sim-policy-wrapper``). It is operator-run on a
Python-3.10 GPU host — GR00T N1.7-3B (bf16 ≈ 6 GB + Cosmos-Reason VLM) does not
fit the 8 GB reference laptop without NF4, so the live boot is NOT exercised in
CI. The openral-side wire (``policies/gr00t.py`` reusing the family adapter) is
unit-tested with a real socket; the live round trip is validated on the lab host
via ``tests/sim/test_franka_gr00t_libero.py`` (PR2 eval).

CLAUDE.md compliance:
* The sidecar is a real subprocess running real upstream code — no mocks
  (§1.11). The wire on the openral side is a real ZMQ client.
* Python-version isolation is the only safe way to bridge GR00T (3.10) and
  openral (3.12-only); see §3.
* The license guard is enforced upstream in the ``RSkillManifest`` loader, not
  here — GR00T N1.7 is Open Model License (commercial OK); N1/N1.5/N1.6 require
  ``OPENRAL_ALLOW_NONCOMMERCIAL=1``.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from openral_sim._sidecar_common import build_parser, run_cmd, run_sidecar

_LABEL = "gr00t-sidecar"
_REPO_URL = "https://github.com/NVIDIA/Isaac-GR00T.git"
_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "gr00t-sidecar"
_SERVER_REL = "gr00t/eval/run_gr00t_server.py"


def _install_deps(*, source: Path, uv: str, quantization: str) -> Path:
    """Create a Python 3.10 venv and install Isaac-GR00T (+ flash-attn, bnb).

    Isaac-GR00T installs editable with its own deps; flash-attn must be built
    with ``--no-build-isolation`` against the resolved torch. bitsandbytes is
    added for NF4 / INT8 so the 3B model fits a ≤ 8 GiB GPU.
    """
    venv = source / ".venv"
    run_cmd(_LABEL, [uv, "venv", "--python", "3.10", str(venv)])
    py = venv / "bin" / "python"
    run_cmd(_LABEL, [uv, "pip", "install", "--python", str(py), "-e", "."], cwd=source)
    # flash-attn is the pinned wheel GR00T expects; build without isolation so
    # it links against the just-installed torch.
    run_cmd(
        _LABEL,
        [
            uv,
            "pip",
            "install",
            "--python",
            str(py),
            "--no-build-isolation",
            "flash-attn==2.7.4.post1",
        ],
        cwd=source,
    )
    if quantization in {"nf4", "int8"}:
        run_cmd(
            _LABEL, [uv, "pip", "install", "--python", str(py), "bitsandbytes>=0.43.0"], cwd=source
        )
    if not py.exists():
        raise SystemExit(f"venv python missing at {py}")
    return venv


def _make_wrapper(*, work: Path, source: Path, args: argparse.Namespace) -> Path:
    """Write a wrapper that applies NF4 then runs run_gr00t_server.py.

    GR00T's ``run_gr00t_server`` has no quantization flag; the accepted path is
    to set a ``BitsAndBytesConfig`` on the Cosmos-Reason backbone before the
    policy is constructed. We monkey-patch ``AutoModel.from_pretrained`` (scoped
    to loads without an explicit config), then run the server script as
    ``__main__`` via :func:`runpy.run_path` so its own arg parser handles the
    flags after the patch is in place.
    """
    server_script = source / _SERVER_REL
    if not server_script.is_file():
        raise SystemExit(
            f"Isaac-GR00T server script not found at {server_script}. The upstream "
            "layout may have changed; check gr00t/eval/ for the server entrypoint."
        )
    wrapper = work / "boot_server.py"
    wrapper.write_text(
        textwrap.dedent(
            f'''
            """Boot wrapper produced by tools/gr00t_sidecar.py."""
            from __future__ import annotations

            import runpy
            import sys

            QUANTIZATION = {args.quantization!r}

            if QUANTIZATION in {{"nf4", "int8"}}:
                import torch  # module-level: _safe_to + the TimestepEncoder patch need it
                import transformers

                _orig = transformers.AutoModel.from_pretrained

                def _patched(*args, **kwargs):
                    if "quantization_config" not in kwargs:
                        from transformers import BitsAndBytesConfig

                        if QUANTIZATION == "nf4":
                            kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                llm_int8_skip_modules=["lm_head", "embed_tokens"],
                            )
                        else:
                            kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_skip_modules=["lm_head", "embed_tokens"],
                            )
                        kwargs.setdefault("torch_dtype", torch.bfloat16)
                    return _orig(*args, **kwargs)

                transformers.AutoModel.from_pretrained = _patched

                # gr00t_policy.py:102 does `model.to(device, dtype=bf16)` after
                # load; transformers' PreTrainedModel.to FORBIDS a dtype cast on
                # a bnb-4bit model (ValueError "cannot cast a bitsandbytes model
                # in a new dtype"). Strip the float dtype when quantized — keep
                # the device move. Verified on nvidia/GR00T-N1.7-3B: NF4 load =
                # 468 Linear4bit modules, 3.3 GB peak VRAM on an 8 GB GPU.
                import transformers.modeling_utils as _mu

                _orig_to = _mu.PreTrainedModel.to

                def _safe_to(self, *a, **k):
                    if getattr(self, "is_quantized", False) or getattr(
                        self, "is_loaded_in_4bit", False
                    ):
                        k.pop("dtype", None)
                        a = tuple(
                            x for x in a if not (isinstance(x, torch.dtype) and x.is_floating_point)
                        )
                    return _orig_to(self, *a, **k)

                _mu.PreTrainedModel.to = _safe_to

                # gr00t/model/modules/dit.py TimestepEncoder.forward infers its
                # compute dtype via `next(self.parameters()).dtype`, which is
                # uint8 for bnb-4bit-packed weights — the subsequent SiLU then
                # crashes ("silu_cuda not implemented for 'Byte'"). Hard-pin the
                # timestep projection to bf16 (the bnb_4bit_compute_dtype). This
                # is the same bug + fix the rldx sidecar carries (RLDX-1 is a
                # GR00T finetune). Verified: full NF4 get_action forward = 683 ms,
                # 2.75 GB peak VRAM on an 8 GB GPU.
                from gr00t.model.modules import dit as _dit

                def _ts_forward(self, timesteps):
                    return self.timestep_embedder(self.time_proj(timesteps).to(torch.bfloat16))

                _dit.TimestepEncoder.forward = _ts_forward

            # Wire-codec bridge (ADR-0046). GR00T's PolicyServer serializes
            # ndarrays with msgpack_numpy; the openral gr00t adapter
            # (policies/gr00t.py, shared with rldx) uses an np.save-bytes msgpack
            # codec. Repoint MsgSerializer.to_bytes/from_bytes at the np.save
            # codec so the server speaks the adapter's wire. The obs/action keys
            # (video.image / state.x / annotation.human.action.task_description /
            # action.x …), the {{observation, options}} envelope, and the
            # ping/get_action/reset endpoints already match — only the ndarray
            # encoding differed.
            import io as _io
            import msgpack as _mp
            import numpy as _np
            from gr00t.policy.server_client import MsgSerializer as _MS

            def _enc(o):
                if isinstance(o, _np.ndarray):
                    b = _io.BytesIO()
                    _np.save(b, o, allow_pickle=False)
                    return {{"__ndarray_class__": True, "as_npy": b.getvalue()}}
                return o

            def _dec(o):
                if "__ndarray_class__" in o:
                    return _np.load(_io.BytesIO(o["as_npy"]), allow_pickle=False)
                return o

            _MS.to_bytes = staticmethod(lambda d: _mp.packb(d, default=_enc, use_bin_type=True))
            _MS.from_bytes = staticmethod(lambda d: _mp.unpackb(d, object_hook=_dec, raw=False))

            sys.argv = [
                "run_gr00t_server",
                "--model-path", {args.model!r},
                "--embodiment-tag", {args.embodiment_tag!r},
                "--host", "0.0.0.0",
                "--port", str({args.port}),
                "--use-sim-policy-wrapper",
            ]
            runpy.run_path({str(server_script)!r}, run_name="__main__")
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
        default_embodiment_tag="new_embodiment",
        model_help="Model id or local path passed to the server's --model-path flag, e.g. "
        "nvidia/GR00T-N1.7-LIBERO/libero_spatial (use a local snapshot path for a "
        "per-suite subfolder).",
        quant_help="Backbone quantization. NF4 fits GR00T-3B on ≤ 8 GiB GPUs (default nf4); "
        "use `none` for ≥ 16 GiB.",
    )
    args = parser.parse_args()
    return run_sidecar(
        label=_LABEL,
        family="gr00t",
        repo_url=_REPO_URL,
        args=args,
        install_deps=_install_deps,
        make_wrapper=_make_wrapper,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
