"""Merge a PEFT LoRA adapter into a loaded policy's weights, without ``peft``.

A LoRA rSkill (``RSkillManifest.base_weights_uri`` set) ships only the
low-rank factors: ``adapter_config.json`` + ``adapter_model.safetensors``
holding ``base_model.model.<module>.lora_{A,B}.weight`` pairs over a base
checkpoint. The adapter loads the base, then this module folds each pair in
place, ``W += (lora_alpha / r) * B @ A``, computed in fp32 and cast once to
the parameter's dtype. The merged model is an ordinary dense policy, so
every downstream path (quantization, CUDA graph, TensorRT) sees plain
weights and the forward costs nothing extra.

Only plain LoRA is accepted: DoRA, rsLoRA, per-module rank/alpha patterns,
trained biases, ``modules_to_save`` and an adapter trained on a different
base are refused (``ROSConfigError``), as is any key that does not pair up,
names no live ``Linear`` weight, falls outside ``target_modules``, has the
wrong shape, or targets a tied (storage-shared) weight; and every module
``target_modules`` selects must receive a pair. A partial merge is a
different policy, so nothing is applied until every check passes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from openral_core.exceptions import ROSConfigError

_log = structlog.get_logger(__name__)

__all__ = [
    "LoraConfigSummary",
    "fetch_adapter_dir",
    "merge_lora_into_policy",
    "read_lora_config",
    "resolve_base_weights",
]

_log = structlog.get_logger(__name__)

_PREFIX = "base_model.model."
_SUFFIXES = (".lora_A.weight", ".lora_B.weight")

# adapter_config.json keys whose non-default value changes the update rule.
_REFUSED: dict[str, Any] = {
    "use_dora": False,
    "use_rslora": False,
    "rank_pattern": {},
    "alpha_pattern": {},
    "bias": "none",
    "lora_bias": False,
    "fan_in_fan_out": False,
    "layers_to_transform": None,
    "use_qalora": False,
}


@dataclass(frozen=True)
class LoraConfigSummary:
    """The fields of ``adapter_config.json`` the merge uses.

    Attributes:
        r: LoRA rank.
        alpha: ``lora_alpha``; the merge scale is ``alpha / r``.
        target_modules: A regex (``re.fullmatch`` on the module name) or a
            list of module-name suffixes, as PEFT interprets them.
        base_model: ``base_model_name_or_path`` the adapter was trained on.
    """

    r: int
    alpha: float
    target_modules: str | list[str]
    base_model: str | None

    @property
    def scale(self) -> float:
        """``lora_alpha / r``."""
        return self.alpha / self.r

    def targets(self, module_name: str) -> bool:
        """Whether PEFT would wrap ``module_name`` (regex fullmatch or suffix list).

        Example:
            >>> LoraConfigSummary(r=2, alpha=4, target_modules=["q_proj"], base_model=None).targets(
            ...     "layers.0.q_proj"
            ... )
            True
        """
        if isinstance(self.target_modules, str):
            return re.fullmatch(self.target_modules, module_name) is not None
        return any(module_name == t or module_name.endswith(f".{t}") for t in self.target_modules)


def read_lora_config(adapter_dir: str | Path) -> LoraConfigSummary:
    """Parse and vet ``<adapter_dir>/adapter_config.json``; plain LoRA only.

    Raises:
        ROSConfigError: Missing file, malformed JSON or not an object, a
            ``target_modules`` regex that does not compile, not ``peft_type: LORA``, a bad rank /
            alpha, or any of DoRA, rsLoRA, rank/alpha patterns, trained bias,
            ``fan_in_fan_out``, ``layers_to_transform``, QA-LoRA,
            ``modules_to_save``.
    """
    path = Path(adapter_dir) / "adapter_config.json"
    if not path.is_file():
        raise ROSConfigError(f"lora: no adapter_config.json in {adapter_dir}")
    try:
        cfg = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ROSConfigError(f"lora: {path} is not valid JSON: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ROSConfigError(f"lora: {path} must hold a JSON object")
    if cfg.get("peft_type") != "LORA":
        raise ROSConfigError(f"lora: peft_type {cfg.get('peft_type')!r} is not LORA")
    for key, default in _REFUSED.items():
        if cfg.get(key, default) not in (default, None, False, {}, []):
            raise ROSConfigError(
                f"lora: adapter_config {key}={cfg.get(key)!r} is not plain LoRA; refusing to merge"
            )
    if cfg.get("modules_to_save"):
        raise ROSConfigError(
            f"lora: modules_to_save={cfg['modules_to_save']!r} ships full module copies; "
            "only plain low-rank pairs are merged"
        )
    r, alpha, targets = cfg.get("r"), cfg.get("lora_alpha"), cfg.get("target_modules")
    if not isinstance(r, int) or r <= 0 or not isinstance(alpha, int | float) or alpha <= 0:
        raise ROSConfigError(f"lora: bad rank/alpha r={r!r} lora_alpha={alpha!r}")
    if not (isinstance(targets, str) or (isinstance(targets, list) and targets)):
        raise ROSConfigError(f"lora: target_modules must be a regex or a list, got {targets!r}")
    if isinstance(targets, str):
        try:
            re.compile(targets)
        except re.error as exc:
            raise ROSConfigError(f"lora: target_modules is not a valid regex: {exc}") from exc
    return LoraConfigSummary(
        r=r,
        alpha=float(alpha),
        target_modules=targets,
        base_model=cfg.get("base_model_name_or_path"),
    )


def merge_lora_into_policy(
    policy: Any,
    adapter_dir: str | Path,
    *,
    torch: Any,
    base_repo_id: str | None = None,
) -> dict[str, float | int]:
    """Fold every LoRA pair in ``adapter_dir`` into ``policy``'s Linear weights, in place.

    PEFT keys ``base_model.model.<module>.lora_{A,B}.weight`` name modules
    relative to the policy object PEFT wrapped (lerobot: the ``PI05Policy``),
    so ``<module>`` is looked up in ``policy.named_modules()`` as is.

    Args:
        policy: The loaded ``torch.nn.Module`` (real storage, not meta).
        adapter_dir: Directory with ``adapter_config.json`` +
            ``adapter_model.safetensors``.
        torch: The ``torch`` module (lazy import convention).
        base_repo_id: The repo the base weights were loaded from; refused when
            the adapter names a different ``base_model_name_or_path``.

    Returns:
        ``{"modules": n, "params": n_lora_params, "scale": alpha/r,
        "max_rel_delta": max ||ΔW||/||W||}`` — also logged as ``lora.merged``.

    Raises:
        ROSConfigError: Any refusal listed in the module docstring.
    """
    from safetensors.torch import load_file

    cfg = read_lora_config(adapter_dir)
    if base_repo_id is not None and cfg.base_model not in (None, base_repo_id):
        raise ROSConfigError(
            f"lora: adapter was trained on {cfg.base_model!r}, base weights are {base_repo_id!r}"
        )
    weights_path = Path(adapter_dir) / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise ROSConfigError(f"lora: no adapter_model.safetensors in {adapter_dir}")
    tensors = load_file(str(weights_path))

    pairs: dict[str, dict[str, Any]] = {}
    for key, value in tensors.items():
        if not key.startswith(_PREFIX) or not key.endswith(_SUFFIXES):
            raise ROSConfigError(f"lora: unexpected adapter key {key!r} (not a lora_A/B weight)")
        module, _, ab = key[len(_PREFIX) :].rpartition(".lora_")
        pairs.setdefault(module, {})[ab[0]] = value

    modules = dict(policy.named_modules())
    targeted = {
        name
        for name, mod in modules.items()
        if cfg.targets(name) and isinstance(mod, torch.nn.Linear)
    }
    # Storage shared by more than one parameter name is a tie: merging one would
    # silently move the other.
    owners: dict[int, int] = {}
    for _, p in policy.named_parameters(remove_duplicate=False):
        owners[p.data_ptr()] = owners.get(p.data_ptr(), 0) + 1

    plan: list[tuple[str, Any, Any, Any]] = []
    for module, pair in sorted(pairs.items()):
        if set(pair) != {"A", "B"}:
            raise ROSConfigError(f"lora: {module!r} has lora_{''.join(pair)} without its pair")
        mod = modules.get(module)
        if mod is None or not isinstance(mod, torch.nn.Linear):
            raise ROSConfigError(f"lora: {module!r} names no Linear module in the policy")
        if module not in targeted:
            raise ROSConfigError(f"lora: {module!r} is outside target_modules")
        weight = mod.weight
        a, b = pair["A"], pair["B"]
        if a.shape != (cfg.r, weight.shape[1]) or b.shape != (weight.shape[0], cfg.r):
            raise ROSConfigError(
                f"lora: {module!r} A{tuple(a.shape)} B{tuple(b.shape)} do not fit "
                f"weight {tuple(weight.shape)} at r={cfg.r}"
            )
        if owners.get(weight.data_ptr(), 0) > 1:
            raise ROSConfigError(f"lora: {module!r}.weight is tied to another parameter")
        plan.append((module, weight, a, b))
    missing = sorted(targeted - set(pairs))
    if missing:
        raise ROSConfigError(
            f"lora: {len(missing)} target module(s) have no adapter weights, e.g. {missing[:3]}"
        )

    max_rel = 0.0
    with torch.no_grad():
        for _module, weight, a, b in plan:
            w32 = weight.detach().float()
            delta = cfg.scale * (b.to(w32.device).float() @ a.to(w32.device).float())
            norm = float(w32.norm())
            if norm > 0.0:
                max_rel = max(max_rel, float(delta.norm()) / norm)
            weight.copy_((w32 + delta).to(weight.dtype))
    stats: dict[str, float | int] = {
        "modules": len(plan),
        "params": sum(int(t.numel()) for t in tensors.values()),
        "scale": cfg.scale,
        "max_rel_delta": max_rel,
    }
    _log.info("lora.merged", adapter_dir=str(adapter_dir), base=cfg.base_model, **stats)
    return stats


def resolve_base_weights(uri: str) -> tuple[str, str | None]:
    """Split a manifest ``base_weights_uri`` into ``(repo_id_or_path, revision)``.

    Example:
        >>> resolve_base_weights("hf://lerobot/pi05_base@22c30e5")
        ('lerobot/pi05_base', '22c30e5')
    """
    if uri.startswith("hf://"):
        repo_id, _, revision = uri[len("hf://") :].partition("@")
        if not revision:
            _log.warning("rskill.unpinned_weights", repo=repo_id, field="base_weights_uri")
        return repo_id, (revision or None)
    if uri.startswith("local://"):
        path = Path(uri[len("local://") :]).resolve()
        if not path.is_dir():
            raise ROSConfigError(f"lora: base_weights_uri {uri!r} is not a directory: {path}")
        return str(path), None
    raise ROSConfigError(f"lora: base_weights_uri {uri!r} is neither hf:// nor local://")


def fetch_adapter_dir(repo_id: str, revision: str | None) -> Path:
    """Local directory holding the adapter's two files (a dir as is, else the HF cache)."""
    if Path(repo_id).is_dir():
        return Path(repo_id)
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id, "adapter_config.json", revision=revision)
    return Path(hf_hub_download(repo_id, "adapter_model.safetensors", revision=revision)).parent
