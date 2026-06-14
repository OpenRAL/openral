"""Per-family scaffold defaults and HF Hub config introspection.

This module powers the two ``ral skill new`` UX upgrades:

- ``--family <act|smolvla|pi05|xvla|diffusion>`` — drives manifest
  defaults so a freshly-scaffolded ACT skill doesn't ship a
  pi0.5-shaped baseline that the user then has to rewrite by hand.
- ``--from-hf <owner/repo>`` — fetches the checkpoint's ``config.json``
  (and ``policy_preprocessor.json`` when present) from the Hub, infers
  the policy family, chunk size, image-feature names, and proprio /
  action dims, and folds those into the scaffold patch. Bypasses the
  family menu when present.

Both paths produce a `RSkillPatch` dict that
`openral_cli._rskill_scaffolder.scaffold_rskill` applies on top
of the on-disk template, so the rest of the scaffolder stays oblivious
to family-specific knobs.

No mocks, no network unless ``--from-hf`` is requested.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

RSkillFamily = Literal["act", "smolvla", "pi05", "xvla", "diffusion"]

#: Min ``shape`` length for HxW extraction from an ``input_features``
#: entry. Anything shorter triggers the template-default fallback.
_CHW_DIMS = 2

#: The five families OpenRAL ships sim policy adapters for. Mirrors
#: the keys of :data:`openral_sim.registry.POLICIES` minus the mock
#: entries (``zero`` / ``random``) which are scene-side aids, not
#: packageable skills.
RSKILL_FAMILIES: tuple[RSkillFamily, ...] = ("act", "smolvla", "pi05", "xvla", "diffusion")


class RSkillPatch(TypedDict, total=False):
    """Subset of manifest fields the scaffolder will overlay on the template.

    Empty / missing keys mean "leave the template value untouched". The
    scaffolder applies the patch *after* the rename / license rewrite so
    keys that come from CLI flags (name, license, embodiment_tags,
    weights_uri) stay authoritative.
    """

    model_family: str
    chunk_size: int
    quantization: dict[str, Any]
    latency_budget: dict[str, float]
    min_vram_gb: dict[str, float] | None
    n_action_steps: int | None
    image_preprocessing: dict[str, Any] | None
    state_contract: dict[str, Any] | None
    # ADR-0019: per-checkpoint action contract. Required for any rSkill
    # that wants to write through the dataset bridge — the bridge reads
    # ``action_contract.dim`` to bind the LeRobot v3 ``action`` feature
    # shape (cf. ``state_contract`` for ``observation.state``).
    action_contract: dict[str, Any] | None
    sensors_required: list[dict[str, Any]]
    weights_uri: str
    source_repo: str
    description: str


def family_defaults(family: RSkillFamily) -> RSkillPatch:
    """Return the manifest patch for a given family.

    Numbers match the published reference manifests under ``rskills/``
    (``act-aloha``, ``smolvla-libero``, ``pi05-libero-nf4``, ``xvla-libero``,
    ``diffusion-pusht``). Keeping them in one place avoids drift between
    the scaffolder and the in-tree reference skills.
    """
    if family == "act":
        # ACT (Zhao et al., 2023) — light ResNet-18 backbone, chunk=100,
        # plain chunked replay. fp32 only (the published checkpoints
        # were not trained with bf16 norm stats).
        return {
            "model_family": "act",
            "chunk_size": 100,
            "quantization": {"dtype": "fp32", "backend": "pytorch"},
            "latency_budget": {
                "per_chunk_ms": 100.0,
                "warmup_ms": 5000.0,
                "load_ms": 15000.0,
            },
            "min_vram_gb": None,
            "n_action_steps": None,
            "image_preprocessing": None,
            "state_contract": None,
        }
    if family == "smolvla":
        # SmolVLA paper config — chunk=16, bf16 on a desktop GPU.
        return {
            "model_family": "smolvla",
            "chunk_size": 16,
            "quantization": {"dtype": "bf16", "backend": "pytorch"},
            "latency_budget": {
                "per_chunk_ms": 150.0,
                "warmup_ms": 8000.0,
                "load_ms": 30000.0,
            },
            "min_vram_gb": None,
            "n_action_steps": None,
        }
    if family == "pi05":
        # Physical Intelligence π0.5 — 3B PaliGemma backbone, bf16,
        # chunk=50 with half-chunk replan (n_action_steps=25). Keeps the
        # min_vram_gb block so `openral doctor` can warn on 8 GB GPUs.
        return {
            "model_family": "pi05",
            "chunk_size": 50,
            "quantization": {"dtype": "bf16", "backend": "pytorch"},
            "latency_budget": {
                "per_chunk_ms": 200.0,
                "warmup_ms": 15000.0,
                "load_ms": 60000.0,
            },
            "min_vram_gb": {"fp32": 14.0, "bf16": 7.0},
            "n_action_steps": 25,
        }
    if family == "xvla":
        # xVLA (Florence-2 backbone). Same LIBERO 8-D state contract,
        # but action space is 20-D internally (collapsed to 7-D by the
        # adapter's postprocessor).
        return {
            "model_family": "xvla",
            "chunk_size": 30,
            "quantization": {"dtype": "bf16", "backend": "pytorch"},
            "latency_budget": {
                "per_chunk_ms": 200.0,
                "warmup_ms": 10000.0,
                "load_ms": 30000.0,
            },
            "min_vram_gb": {"fp32": 10.0, "bf16": 5.0},
            "n_action_steps": None,
        }
    if family == "diffusion":
        # Diffusion Policy — single camera, chunk=16, fp32 CPU-friendly.
        return {
            "model_family": "diffusion",
            "chunk_size": 16,
            "quantization": {"dtype": "fp32", "backend": "pytorch"},
            "latency_budget": {
                "per_chunk_ms": 100.0,
                "warmup_ms": 3000.0,
                "load_ms": 10000.0,
            },
            "min_vram_gb": None,
            "n_action_steps": 8,
            "image_preprocessing": None,
            "state_contract": None,
        }
    raise ValueError(f"unknown skill family: {family!r}")


_CONFIG_TYPE_TO_FAMILY: dict[str, RSkillFamily] = {
    "act": "act",
    "smolvla": "smolvla",
    "pi05": "pi05",
    "xvla": "xvla",
    "diffusion": "diffusion",
    "diffusion_policy": "diffusion",
}


def introspect_hf(
    repo_id: str, *, default_family: RSkillFamily | None = None
) -> tuple[RSkillFamily, RSkillPatch]:
    """Probe a HF Hub repo for ``config.json`` and derive a scaffold patch.

    Builds on `family_defaults` — the introspected fields
    (chunk_size, sensors, state_contract, weights_uri) override the
    family baseline so the scaffold lands as close to the actual
    checkpoint contract as possible.

    Args:
        repo_id: Hub identifier, e.g. ``"Deepkar/libero-test-act"`` or
            ``"hf://Deepkar/libero-test-act"`` (the ``hf://`` prefix is
            tolerated for symmetry with ``weights_uri``).
        default_family: Family to fall back to when ``config.json``
            ``type`` field is missing or unrecognized. ``None`` raises
            on an unknown type.

    Returns:
        ``(family, patch)`` — the inferred family and a manifest patch
        ready for the scaffolder. ``patch["weights_uri"]`` and
        ``patch["source_repo"]`` are pinned to ``hf://<repo_id>``.

    Raises:
        ValueError: ``config.json`` cannot be fetched, doesn't parse,
            or has an unknown ``type`` and no ``default_family`` was
            given.
    """
    repo_id = repo_id.removeprefix("hf://")
    config = _fetch_hf_json(repo_id, "config.json")
    if not isinstance(config, dict):
        raise ValueError(f"{repo_id}/config.json did not parse as a JSON object")

    raw_type = str(config.get("type", "")).lower()
    family = _CONFIG_TYPE_TO_FAMILY.get(raw_type, default_family)
    if family is None:
        valid = ", ".join(sorted(_CONFIG_TYPE_TO_FAMILY))
        raise ValueError(
            f"{repo_id}/config.json declares type={raw_type!r}; recognized: {valid}. "
            "Pass --family explicitly to override, or use a checkpoint with one "
            "of the supported policy types."
        )

    patch: RSkillPatch = dict(family_defaults(family))  # type: ignore[assignment]
    if "chunk_size" in config:
        patch["chunk_size"] = int(config["chunk_size"])

    sensors = _sensors_from_input_features(config.get("input_features"))
    if sensors:
        patch["sensors_required"] = sensors

    state_dim = _state_dim_from_input_features(config.get("input_features"))
    if state_dim is not None:
        patch["state_contract"] = {"dim": state_dim}

    aliases = _aliases_from_input_features(config.get("input_features"))
    if aliases:
        ip = dict(patch.get("image_preprocessing") or {})
        ip.setdefault("flip_180", False)
        ip["aliases"] = aliases
        patch["image_preprocessing"] = ip

    patch["weights_uri"] = f"hf://{repo_id}"
    patch["source_repo"] = f"hf://{repo_id}"
    return family, patch


def _fetch_hf_json(repo_id: str, filename: str) -> Any:  # noqa: ANN401  # reason: returns parsed JSON of arbitrary shape
    """Download a single JSON file from a HF Hub repo, return parsed body.

    Uses ``huggingface_hub.hf_hub_download`` (same path the loader uses
    for ``rskill.yaml``) so cache + auth behave identically. Raises on
    network / parse error so the caller can surface a clear message.
    """
    import json  # noqa: PLC0415  # reason: keep top-level imports minimal

    # reason: optional dep, only needed for --from-hf
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:
        raise ValueError(
            "huggingface_hub is required for --from-hf; install with `uv sync`."
        ) from exc

    try:
        local = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as exc:
        raise ValueError(f"could not fetch {repo_id}/{filename}: {exc}") from exc

    with open(local) as f:
        return json.load(f)


def _sensors_from_input_features(input_features: Any) -> list[dict[str, Any]]:  # noqa: ANN401  # reason: parsed-JSON shape varies per checkpoint
    """Build ``sensors_required`` from ``config.input_features``.

    Only emits entries for keys that start with ``observation.images.``;
    state / proprio features stay out (they're declared via
    ``state_contract`` instead). Each image feature becomes a
    ``camera<N>`` sensor on the scene side, with the checkpoint's
    ``image*`` / ``wrist_image`` key landing in
    ``image_preprocessing.aliases`` separately.
    """
    if not isinstance(input_features, dict):
        return []
    sensors: list[dict[str, Any]] = []
    counter = 0
    for key, spec in input_features.items():
        if not key.startswith("observation.images."):
            continue
        if not isinstance(spec, dict):
            continue
        shape = spec.get("shape") or [3, 224, 224]
        # ``input_features`` shapes are CHW; the last two dims are HxW.
        # Length < ``_CHW_DIMS`` means we couldn't parse the shape, so we
        # fall back to the template's 224x224 minimum.
        h = int(shape[-2]) if len(shape) >= _CHW_DIMS else 224
        w = int(shape[-1]) if len(shape) >= _CHW_DIMS else 224
        counter += 1
        sensors.append(
            {
                "modality": "rgb",
                "vla_feature_key": f"observation.images.camera{counter}",
                "min_width": w,
                "min_height": h,
            }
        )
    return sensors


def _state_dim_from_input_features(input_features: Any) -> int | None:  # noqa: ANN401  # reason: parsed-JSON shape varies per checkpoint
    """Pull the proprio state dim from ``observation.state.shape``."""
    if not isinstance(input_features, dict):
        return None
    state = input_features.get("observation.state")
    if not isinstance(state, dict):
        return None
    shape = state.get("shape")
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    return int(shape[0])


def _aliases_from_input_features(input_features: Any) -> dict[str, str]:  # noqa: ANN401  # reason: parsed-JSON shape varies per checkpoint
    """Map ``camera<N>`` source keys → checkpoint image-feature names.

    Walks ``observation.images.<feature_name>`` entries in declaration
    order and pairs them with the ``camera<N>`` slots emitted by
    `_sensors_from_input_features`. The result is suitable for
    ``image_preprocessing.aliases`` — the in-tree adapters rename the
    scene-side ``camera<N>`` keys into the checkpoint's input-feature
    names at step time.

    Returns an empty dict when the checkpoint's image keys already
    match ``camera<N>`` (no rename needed).
    """
    if not isinstance(input_features, dict):
        return {}
    image_keys = [k for k in input_features if k.startswith("observation.images.")]
    aliases: dict[str, str] = {}
    for i, key in enumerate(image_keys, start=1):
        feature_name = key.removeprefix("observation.images.")
        cam = f"camera{i}"
        if feature_name == cam:
            continue
        aliases[cam] = feature_name
    return aliases
