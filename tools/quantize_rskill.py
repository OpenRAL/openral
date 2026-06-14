r"""Quantize a lerobot policy and upload the result to the HuggingFace Hub.

A generic, policy-agnostic counterpart to the in-process fast-path in
:mod:`openral_sim._quantization`. Loads ANY lerobot policy class
through its standard ``from_pretrained``, applies an in-place
quantization rewrite, saves the resulting state dict to a local
``model.safetensors`` plus a sibling ``quantization_metadata.json``,
and uploads the bundle to a target rSkill repo on the Hub.

Why this exists
---------------
Loading a 3.6 B-param π0.5 RoboCasa checkpoint takes ~90 s on a
4070-mobile: ~20 s reading bf16 safetensors from cache, ~10 s
restoring transformers/lerobot metadata, and the rest is the on-line
``bitsandbytes`` nf4 conversion that
:func:`openral_sim._quantization.quantize_nf4_in_place` runs every
process launch. Packaging the *post*-quantization state dict cuts the
quantization step out for every subsequent launch -- the policy
adapter's :func:`load_prequantized_state_for_rskill` detects the
``quantization_metadata.json`` sentinel, downloads ``model.safetensors``,
and runs :func:`install_prequantized_linears` to drop the packed nf4
weights directly into pre-allocated ``Linear4bit`` modules.

This script is **not** part of CI. It is a one-shot tooling helper
that mutates a HuggingFace Hub repo, so it is intentionally gated on
the operator setting ``HF_TOKEN=<your-write-token>`` in the
environment. Read-only contributors can ignore it.

Usage
-----

The defaults mirror the validated π0.5 RoboCasa path::

    HF_TOKEN=<your-token> uv run python tools/quantize_rskill.py \\
        --source outputs/run_artifacts/r365_pi05_ckpt_lerobot \\
        --target OpenRAL/rskill-pi05-robocasa365-human300-nf4

To package a different lerobot policy with the same nf4 rule, point
``--policy-class`` at its modeling module's policy class::

    HF_TOKEN=<your-token> uv run python tools/quantize_rskill.py \\
        --source <hf-org>/<smolvla-finetune> \\
        --target <hf-org>/<smolvla-finetune>-nf4 \\
        --policy-class lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy

The script:

1. Imports ``--policy-class`` via :func:`importlib.import_module` so
   no per-policy code changes are needed when a new lerobot family
   lands.
2. Loads the source repo through that policy's ``from_pretrained``
   into bf16 on CPU.
3. Calls :func:`quantize_nf4_in_place` from the shared
   :mod:`openral_sim._quantization` module (so any future change
   to the quantization rule carries over automatically to every
   policy).
4. Moves the policy to ``--device`` so bitsandbytes packs the 4-bit
   weights.
5. Saves ``policy.state_dict()`` to ``<temp>/model.safetensors`` plus
   a sibling ``quantization_metadata.json`` recording the rule
   (min-param threshold, compute_dtype, source repo, source revision,
   policy class).
6. Copies the upstream ``*.json`` / ``*.md`` /
   ``policy_*_processor.safetensors`` sidecars so the rSkill loader
   has the same architectural metadata the source carried.
7. Stamps a ``quantization_config`` block into the copied
   ``config.json`` (when present) so the HuggingFace Hub auto-tags the
   mirror (``bitsandbytes`` / ``nf4`` / ``4-bit`` / ``8-bit``). Without
   this the bf16 source ``config.json`` is uploaded verbatim and the
   quantized mirror is mistagged.
8. Creates the target repo (idempotent) and uploads the bundle.

The upload is bandwidth-bound; expect 15-30 min on home WiFi for the
~2 GiB nf4 bundle.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, get_token, snapshot_download
from safetensors.torch import save_file


def _resolve_policy_class(dotted: str) -> type:
    """Import ``module.attr`` and return the resolved class.

    Args:
        dotted: Fully qualified dotted path, e.g.
            ``lerobot.policies.pi05.modeling_pi05.PI05Policy``.
    """
    if "." not in dotted:
        raise ValueError(
            f"--policy-class must be a dotted path 'package.module.ClassName', got {dotted!r}"
        )
    mod_name, _, attr = dotted.rpartition(".")
    module = importlib.import_module(mod_name)
    return getattr(module, attr)  # type: ignore[no-any-return]


def _load_lerobot_policy(source_repo: str, policy_class: type) -> Any:
    """Load a lerobot ``PreTrainedPolicy`` checkpoint onto CPU in bf16.

    The default path: every lerobot policy exposes ``from_pretrained`` and a
    matching ``PreTrainedConfig``. Construction is forced onto CPU so the
    quantize / cast steps run before any device transfer.
    """
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(source_repo)
    cfg.device = "cpu"
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        # `from_pretrained` is a contract every lerobot PreTrainedPolicy
        # exposes, but the base class isn't in this module's stubs so
        # mypy sees `type` here.
        return policy_class.from_pretrained(source_repo, config=cfg)  # type: ignore[attr-defined]
    finally:
        torch.set_default_dtype(prev_dtype)


# Loading a ``trust_remote_code`` model executes ``modeling_*.py`` shipped in
# the source repo — a remote-code-execution sink (security audit 2026-06, C3).
# Only the project's own HF org is trusted by default; extend via
# OPENRAL_TRUSTED_REMOTE_CODE_ORGS (comma-separated), or acknowledge an untrusted
# repo for one run with OPENRAL_ALLOW_REMOTE_CODE=1.
_DEFAULT_TRUSTED_REMOTE_CODE_ORGS = frozenset({"openral"})
_TRUSTED_ORGS_ENV = "OPENRAL_TRUSTED_REMOTE_CODE_ORGS"
_ALLOW_REMOTE_CODE_ENV = "OPENRAL_ALLOW_REMOTE_CODE"


def _require_trusted_remote_code(source_repo: str) -> None:
    """Refuse ``--trust-remote-code`` for repos outside the trusted-org allowlist.

    Executing a repo's custom code is RCE; gate it like the runtime molmoact2
    loader. A local source path (already on the operator's disk) is exempt — the
    supply-chain threat is *remote* unverified repos. Hard-blocks an untrusted HF
    org unless ``OPENRAL_ALLOW_REMOTE_CODE=1`` acknowledges the one-off risk.

    Args:
        source_repo: The ``--source`` value (HF repo id or local path).

    Raises:
        ValueError: If the source org is not trusted and the override env is unset.
    """
    if os.path.isabs(source_repo) or os.path.exists(source_repo):
        return  # local checkpoint — operator-controlled, not a remote-supply-chain risk
    repo = source_repo.removeprefix("hf://")
    org = repo.split("/", 1)[0].split("@", 1)[0].lower()
    trusted = set(_DEFAULT_TRUSTED_REMOTE_CODE_ORGS)
    trusted.update(
        o.strip().lower() for o in os.environ.get(_TRUSTED_ORGS_ENV, "").split(",") if o.strip()
    )
    if org in trusted:
        return
    if os.environ.get(_ALLOW_REMOTE_CODE_ENV) == "1":
        print(
            f"[quantize] WARNING: --trust-remote-code executing custom code from untrusted "
            f"org {org!r} ({source_repo}); allowed by {_ALLOW_REMOTE_CODE_ENV}=1.",
            flush=True,
        )
        return
    raise ValueError(
        f"--trust-remote-code would execute custom code from '{source_repo}', whose org "
        f"{org!r} is not in the trusted set ({sorted(trusted)}). This is a remote-code-execution "
        f"risk. Add the org to {_TRUSTED_ORGS_ENV} (comma-separated) if you trust it, or set "
        f"{_ALLOW_REMOTE_CODE_ENV}=1 to acknowledge the risk for this run."
    )


def _load_transformers_model(
    source_repo: str,
    *,
    trust_remote_code: bool,
    auto_class: str,
) -> Any:
    """Load a transformers custom-code checkpoint on CPU.

    The escape hatch for action-reasoning models that ship as transformers
    custom-code models rather than first-class lerobot policies — e.g.
    ``allenai/MolmoAct2-LIBERO`` (arch ``molmoact2``, ~5.5 B params,
    ``custom_code``). ``quantize_nf4_in_place`` operates on any
    ``torch.nn.Module`` tree, so the rest of the pipeline (rewrite, save,
    sidecar copy, upload) is unchanged.
    """
    if trust_remote_code:
        _require_trusted_remote_code(source_repo)

    if auto_class == "AutoModel":
        from transformers import AutoModel

        model_cls = AutoModel
    elif auto_class == "AutoModelForImageTextToText":
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    else:  # pragma: no cover — argparse choices guard this path
        raise ValueError(f"Unsupported transformers auto class: {auto_class!r}")

    return model_cls.from_pretrained(
        source_repo,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )


def _build_policy_and_quantize(
    source_repo: str,
    *,
    device: str,
    policy_class: type | None,
    loader: str,
    trust_remote_code: bool,
    transformers_auto_class: str,
    scheme: str,
    min_params: int,
) -> tuple[Any, str]:
    """Load the source checkpoint and apply the in-place quantization.

    Returns the policy and the resolved HF Hub revision so the upload
    metadata can pin it.

    Args:
        loader: ``"lerobot"`` (default) loads via ``policy_class.from_pretrained``;
            ``"transformers"`` loads via ``AutoModelForImageTextToText`` for
            custom-code action-reasoning models (e.g. MolmoAct2). The quantize /
            save / upload steps are loader-agnostic.
        trust_remote_code: Forwarded to the transformers loader; required for
            ``custom_code`` models.
    """
    # Defer heavy imports so ``-h`` is cheap.
    from openral_sim._quantization import quantize_int8_in_place, quantize_nf4_in_place

    print(f"[quantize] loading {source_repo} (bf16, CPU, loader={loader})...", flush=True)
    t0 = time.perf_counter()
    if loader == "transformers":
        policy = _load_transformers_model(
            source_repo,
            trust_remote_code=trust_remote_code,
            auto_class=transformers_auto_class,
        )
    else:
        if policy_class is None:  # pragma: no cover — guarded in main()
            raise ValueError("loader='lerobot' requires a resolved --policy-class")
        policy = _load_lerobot_policy(source_repo, policy_class)
    print(f"[quantize] load: {time.perf_counter() - t0:.1f} s", flush=True)

    # Cast every fp32 leaf to bf16 so non-Linear bits don't mix dtypes
    # with bnb's bf16 compute path.
    for p in policy.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
    for b in policy.buffers():
        if b.dtype == torch.float32:
            b.data = b.data.to(torch.bfloat16)

    print(f"[quantize] rewriting Linear -> {scheme} (min_params={min_params:_}) ...", flush=True)
    t0 = time.perf_counter()
    if scheme == "nf4":
        quantize_nf4_in_place(
            policy,
            torch=torch,
            compute_dtype=torch.bfloat16,
            min_params=min_params,
        )
    elif scheme == "int8":
        # LLM.int8 path (bnb.nn.Linear8bitLt). The runtime adapter
        # cannot consume the resulting pack today — this branch is
        # upload-only and exists so an operator can mirror an int8
        # variant of a checkpoint on the Hub. Add a sibling
        # ``install_prequantized_int8_linears`` in
        # ``openral_sim._quantization`` and wire it through the pi05
        # adapter before pointing a manifest's ``weights_uri`` at an
        # int8 mirror.
        # Mirror the nf4 path's compute_dtype=bf16 default; the int8
        # helper's signature became compute_dtype-required when the
        # remote landed the threshold / new_modules_on_meta knobs.
        # ``threshold`` and ``new_modules_on_meta`` keep their helper
        # defaults (LLM.int8 paper value 6.0; no accelerate wrap).
        quantize_int8_in_place(
            policy,
            torch=torch,
            compute_dtype=torch.bfloat16,
            min_params=min_params,
        )
    else:
        raise NotImplementedError(
            f"Quantization scheme {scheme!r} is not implemented yet. "
            "Wired schemes today: 'nf4', 'int8'. Add a sibling helper in "
            "openral_sim._quantization for new schemes."
        )

    # Move to the requested device so bitsandbytes packs the 4-bit
    # weights (it defers until the first `.to(device)`).
    policy = policy.to(device=device)
    policy.eval()
    print(f"[quantize] convert + move: {time.perf_counter() - t0:.1f} s", flush=True)

    # Resolve the HEAD revision so the manifest can pin it.
    info = HfApi().repo_info(source_repo, repo_type="model")
    revision = info.sha or "main"
    return policy, revision


def _save_state(
    policy: Any,  # reason: lerobot policy classes share no common base
    out_dir: Path,
    *,
    source_repo: str,
    revision: str,
    policy_class_dotted: str,
    scheme: str,
    min_params: int,
) -> None:
    """Dump the quantized state dict + metadata into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    state = policy.state_dict()
    # safetensors needs every value to be a plain torch.Tensor. bnb's
    # Params4bit serialises into multiple state-dict entries (.weight is
    # the packed uint8 tensor; absmax / quant_map / nested params are
    # added under sibling keys). Any non-tensor entry (e.g. a
    # quant_state object dropped in by older bnb) gets dropped with a
    # clear log.
    cleaned: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            cleaned[k] = v.contiguous().cpu()
        else:
            dropped.append(k)
    if dropped:
        print(f"[quantize] dropped {len(dropped)} non-tensor entries: {dropped[:5]}...", flush=True)

    out_path = out_dir / "model.safetensors"
    print(f"[quantize] writing {out_path} ({len(cleaned)} tensors)...", flush=True)
    t0 = time.perf_counter()
    save_file(cleaned, str(out_path))
    size_gb = out_path.stat().st_size / 1e9
    elapsed = time.perf_counter() - t0
    print(f"[quantize] write: {elapsed:.1f} s ({size_gb:.2f} GiB)", flush=True)

    if scheme == "nf4":
        rule = (
            f"Linear modules with >={min_params:_} weight elements rewritten to "
            "bnb.nn.Linear4bit; smaller heads kept in compute_dtype (bfloat16)."
        )
        runtime_status = "loader-backed (install_prequantized_linears)"
        compute_dtype = "bfloat16"
    elif scheme == "int8":
        rule = (
            f"Linear modules with >={min_params:_} weight elements rewritten to "
            "bnb.nn.Linear8bitLt (has_fp16_weights=False, threshold=6.0); "
            "smaller heads kept in their original dtype. "
            "State dict carries weight (int8), SCB (fp32 per-row scale), "
            "weight_format (uint8 sentinel)."
        )
        runtime_status = (
            "upload-only — the pi05 adapter does NOT yet wire a fast-path "
            "for int8 packs. Do not point a manifest's weights_uri here "
            "without first adding install_prequantized_int8_linears in "
            "openral_sim._quantization and a sibling detect call in "
            "load_prequantized_state_for_rskill."
        )
        compute_dtype = "int8"
    else:  # pragma: no cover — guarded earlier
        raise NotImplementedError(scheme)

    meta = {
        "source_repo": source_repo,
        "source_revision": revision,
        "policy_class": policy_class_dotted,
        "quantization": {
            "scheme": scheme,
            "backend": "bitsandbytes",
            "compute_dtype": compute_dtype,
            "min_params_to_quantize": min_params,
            "rule": rule,
            "runtime_status": runtime_status,
        },
        "dropped_state_entries": dropped,
    }
    (out_dir / "quantization_metadata.json").write_text(json.dumps(meta, indent=2))


def _bnb_quantization_config(scheme: str) -> dict[str, Any]:
    """Return the ``transformers`` BitsAndBytesConfig block matching ``scheme``.

    The values mirror what :mod:`openral_sim._quantization` builds at runtime —
    nf4 ``Linear4bit`` with bf16 compute and nested (double) quantization, or
    LLM.int8 ``Linear8bitLt`` with the paper's 6.0 outlier threshold and
    fp16 weights disabled. Stamping this into the uploaded ``config.json`` is
    what lets the HuggingFace Hub auto-tag the repo (``bitsandbytes`` / ``nf4`` /
    ``4-bit`` / ``8-bit``); without it the tool copies the *source* (bf16)
    ``config.json`` verbatim and the quantized mirror is mistagged (the SO-101
    nf4 repo showed only a stray ``8-bit`` tag and no ``nf4`` / ``4-bit``).

    Args:
        scheme: ``"nf4"`` or ``"int8"``.

    Returns:
        A JSON-serialisable dict suitable as the value of ``config.json``'s
        ``quantization_config`` key.
    """
    if scheme == "nf4":
        return {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        }
    if scheme == "int8":
        return {
            "quant_method": "bitsandbytes",
            "load_in_8bit": True,
            "load_in_4bit": False,
            "llm_int8_threshold": 6.0,
            "llm_int8_has_fp16_weight": False,
        }
    raise NotImplementedError(f"no BitsAndBytesConfig mapping for scheme {scheme!r}")


def _stamp_quantization_config(out_dir: Path, *, scheme: str, loader: str) -> None:
    """Inject a ``quantization_config`` into the staged ``config.json``.

    The model bytes are already packed (``_save_state``); this only writes the
    metadata the Hub reads to auto-tag the repo and that a direct
    ``from_pretrained`` would read to rebuild the bnb config. A no-op (with a
    clear log) when the staging dir carries no ``config.json`` — lerobot
    checkpoints keep their config elsewhere and rely on
    ``quantization_metadata.json`` + the runtime overlay instead.

    Gated to ``loader == "transformers"``: that path's ``from_pretrained``
    reads ``quantization_config`` to rebuild the bnb config. A lerobot policy
    config class (e.g. ``PI05Config``) raises ``DecodingError: fields
    quantization_config are not valid`` on the unknown field, and the lerobot
    runtime fast-path keys off ``quantization_metadata.json`` not
    ``config.json`` — so stamping a lerobot ``config.json`` (which
    ``_copy_source_config`` does bring into the staging dir) breaks the load.
    """
    if loader != "transformers":
        print(
            f"[quantize] loader={loader!r}: skipping quantization_config stamp "
            "(lerobot configs reject it; runtime reads quantization_metadata.json).",
            flush=True,
        )
        return
    config_path = out_dir / "config.json"
    if not config_path.exists():
        print(
            f"[quantize] no config.json in staging dir; skipping quantization_config stamp "
            f"(scheme={scheme}). The runtime overlay reads quantization_metadata.json instead.",
            flush=True,
        )
        return
    config = json.loads(config_path.read_text())
    if "quantization_config" in config:
        print(
            "[quantize] WARNING: source config.json already carries a quantization_config; "
            "overwriting with the freshly-applied scheme.",
            flush=True,
        )
    config["quantization_config"] = _bnb_quantization_config(scheme)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"[quantize] stamped quantization_config (scheme={scheme}) into config.json", flush=True)


def _copy_source_config(source_repo: str, out_dir: Path, *, loader: str) -> None:
    """Pull the source's config / metadata / processor / custom-code sidecars.

    Copies every ``*.json``, ``*.md`` and ``policy_*_processor.safetensors``
    from the source repo so the lerobot processor pipeline can
    reconstruct without re-fetching from the source. The processor
    sidecars are small (~10 KB each) but lerobot's PreTrainedConfig
    refuses to instantiate the preprocessor / postprocessor without
    them, so missing them turns the new rSkill into a hard error.

    For ``loader == "transformers"`` (custom-code models like MolmoAct2),
    also copies every ``*.py`` so the quantized repo stays loadable via
    ``trust_remote_code`` — the ``auto_map`` in ``config.json`` points at
    those modules. Without them, ``from_pretrained`` on the nf4 repo fails
    to import the model class.

    The full source weights (the sharded fp32 ``model-XXXXX-of-YYYYY.safetensors``
    + their ``model.safetensors.index.json``, and a single-file
    ``model.safetensors``) are NEVER copied: the nf4 pack we just wrote is
    the only weight file the rSkill ships. Shipping the fp32 shards would
    both bloat the repo by ~20 GiB and shadow the nf4 pack at load time
    (transformers prefers a sharded index over a single ``model.safetensors``).
    """
    print(f"[quantize] cloning {source_repo} architecture metadata...", flush=True)
    allow = ["*.json", "*.md", "*.safetensors", ".gitattributes"]
    if loader == "transformers":
        allow.append("*.py")
    snapshot = Path(
        snapshot_download(
            repo_id=source_repo,
            allow_patterns=allow,
            ignore_patterns=["model*.safetensors"],
        )
    )
    for src in snapshot.iterdir():
        if not src.is_file():
            continue
        # `ignore_patterns` only suppresses re-downloads; if a previous
        # `snapshot_download` already pulled the full source repo, the
        # weight files are still symlinked in the snapshot dir and would
        # clobber the nf4 blob we just wrote in `_save_state` (single-file
        # `model.safetensors`) or shadow it (sharded `model-XXXXX-*` + the
        # `model.safetensors.index.json` that points at them). Skip every
        # source weight artefact explicitly.
        if src.name in {"model.safetensors", "model.safetensors.index.json"}:
            continue
        if src.name.startswith("model-") and src.suffix == ".safetensors":
            continue
        if src.suffix in {".json", ".md", ".safetensors", ".py"} or src.name == ".gitattributes":
            shutil.copy(src, out_dir / src.name)


def _upload(out_dir: Path, target_repo: str, *, token: str, scheme: str) -> None:
    """Create the target repo (idempotent) and upload the contents of ``out_dir``."""
    api = HfApi(token=token)
    print(f"[quantize] ensuring HF repo {target_repo} exists...", flush=True)
    api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True, private=False)

    print(f"[quantize] uploading {out_dir} -> {target_repo}...", flush=True)
    t0 = time.perf_counter()
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=target_repo,
        repo_type="model",
        commit_message=f"add {scheme}-quantized weights (OpenRAL rSkill)",
    )
    print(f"[quantize] upload: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def main() -> int:
    """CLI entry point. See module docstring for usage."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="outputs/run_artifacts/r365_pi05_ckpt_lerobot",
        help="Source HF Hub repo (must be loadable by the policy class).",
    )
    parser.add_argument(
        "--target",
        default="OpenRAL/rskill-pi05-robocasa365-human300-nf4",
        help="Target HF Hub repo for the quantized artefact.",
    )
    parser.add_argument(
        "--policy-class",
        default="lerobot.policies.pi05.modeling_pi05.PI05Policy",
        help=(
            "Fully qualified dotted path to the lerobot policy class "
            "(loader=lerobot only). "
            "Examples: lerobot.policies.pi05.modeling_pi05.PI05Policy, "
            "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy."
        ),
    )
    parser.add_argument(
        "--loader",
        default="lerobot",
        choices=["lerobot", "transformers"],
        help=(
            "How to load the source checkpoint. 'lerobot' (default) uses "
            "--policy-class.from_pretrained; 'transformers' uses "
            "AutoModelForImageTextToText for custom-code action-reasoning "
            "models (e.g. allenai/MolmoAct2-LIBERO). The quantize / save / "
            "upload steps are loader-agnostic."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Forwarded to the transformers loader (loader=transformers). "
            "Required for custom_code models such as MolmoAct2. "
            "SECURITY: this EXECUTES code shipped in the source repo (RCE). "
            f"Only the {sorted(_DEFAULT_TRUSTED_REMOTE_CODE_ORGS)} org(s) are trusted by "
            f"default; add others via {_TRUSTED_ORGS_ENV} (comma-separated) or set "
            f"{_ALLOW_REMOTE_CODE_ENV}=1 to acknowledge an untrusted repo for one run."
        ),
    )
    parser.add_argument(
        "--transformers-auto-class",
        default="AutoModelForImageTextToText",
        choices=["AutoModel", "AutoModelForImageTextToText"],
        help=(
            "Transformers auto class used when --loader=transformers. "
            "MolmoAct2-style repos use AutoModelForImageTextToText; "
            "LocateAnything registers its custom code under AutoModel."
        ),
    )
    parser.add_argument(
        "--scheme",
        default="nf4",
        choices=["nf4", "int8"],
        help=(
            "Quantization scheme to apply. 'nf4' is loader-backed; "
            "'int8' is upload-only (no runtime fast-path yet)."
        ),
    )
    parser.add_argument(
        "--min-params",
        type=int,
        default=4_000_000,
        help=(
            "Minimum Linear weight elements to quantize. Smaller heads stay in "
            "compute dtype. Defaults to 4M (PaliGemma / SmolVLA paper threshold)."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for the bnb pack (must be CUDA for nf4).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Leave the staging directory intact after upload (debug).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only quantize + save locally; do not touch the Hub.",
    )
    args = parser.parse_args()

    # Resolve a write token from env first, then from the cached
    # `huggingface-cli login` credential (`~/.cache/huggingface/token`).
    # Letting the cached login work means contributors don't need to
    # re-export HF_TOKEN every shell session just to push a new rSkill.
    token = os.environ.get("HF_TOKEN") or get_token()
    if not args.skip_upload and not token:
        print(
            "ERROR: no HF token found; export HF_TOKEN=<write-token> or "
            "run `huggingface-cli login` before uploading.",
            file=sys.stderr,
        )
        return 1

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("ERROR: --device=cuda but no CUDA device available.", file=sys.stderr)
        return 1

    # The transformers loader ignores --policy-class; only resolve the dotted
    # path when we're actually going to call it.
    policy_class: type | None = None
    if args.loader == "lerobot":
        try:
            policy_class = _resolve_policy_class(args.policy_class)
        except (ImportError, AttributeError, ValueError) as exc:
            print(
                f"ERROR: could not resolve --policy-class={args.policy_class!r}: {exc}",
                file=sys.stderr,
            )
            return 1

    policy, revision = _build_policy_and_quantize(
        args.source,
        device=args.device,
        policy_class=policy_class,
        loader=args.loader,
        trust_remote_code=args.trust_remote_code,
        transformers_auto_class=args.transformers_auto_class,
        scheme=args.scheme,
        min_params=args.min_params,
    )

    out_dir = Path(tempfile.mkdtemp(prefix="openral-rskill-quant-"))
    try:
        # The lerobot loader pins the dotted policy class so the runtime
        # adapter knows how to rebuild the graph; the transformers loader has
        # no such class, so record the loader sentinel instead.
        policy_class_dotted = (
            args.policy_class
            if args.loader == "lerobot"
            else f"transformers:{args.transformers_auto_class}"
        )
        _save_state(
            policy,
            out_dir,
            source_repo=args.source,
            revision=revision,
            policy_class_dotted=policy_class_dotted,
            scheme=args.scheme,
            min_params=args.min_params,
        )
        _copy_source_config(args.source, out_dir, loader=args.loader)
        # Stamp the bnb quantization_config so the Hub auto-tags the mirror
        # (bitsandbytes / nf4 / 4-bit). Must run AFTER _copy_source_config,
        # which brings the source's bf16 config.json into the staging dir.
        # Gated to the transformers loader inside the function (lerobot configs
        # reject the field — see _stamp_quantization_config).
        _stamp_quantization_config(out_dir, scheme=args.scheme, loader=args.loader)
        if args.skip_upload:
            print(f"[quantize] --skip-upload; bundle left at {out_dir}", flush=True)
            return 0
        assert token  # narrowed above
        _upload(out_dir, args.target, token=token, scheme=args.scheme)
        print(f"[quantize] done. Repo: https://huggingface.co/{args.target}")
        return 0
    finally:
        if not args.keep_temp and not args.skip_upload:
            shutil.rmtree(out_dir, ignore_errors=True)
        else:
            print(f"[quantize] staging dir kept: {out_dir}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
