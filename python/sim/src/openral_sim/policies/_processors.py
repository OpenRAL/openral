"""Manifest-first per-file resolver for policy processor JSON sidecars.

Used by ``openral_sim.policies.diffusion``, ``openral_sim.policies.xvla`` and
``openral_sim.policies.pi05`` for the ``policy_preprocessor.json`` /
``policy_postprocessor.json`` sidecars that lerobot's ``make_pre_post_processors`` /
``PolicyProcessorPipeline.from_pretrained`` consume. Prefers per-file download via
``RSkillManifest.processors``; falls back to ``snapshot_download`` for explicit-scheme URIs (e.g.
``hf://lerobot/diffusion_pusht``) that predate that contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openral_rskill._vla_core import materialize_processor_dir

if TYPE_CHECKING:
    from openral_core import VLASpec

__all__ = ["resolve_processor_dir"]


def resolve_processor_dir(spec: VLASpec | Any, repo_id: str) -> str:
    """Return a local directory containing the processor JSON sidecars.

    Resolution order:

    1. If ``spec.weights_uri`` is a bare rSkill reference (no explicit
       scheme) **and** the resolved manifest declares a ``processors``
       block, call
       ``materialize_processor_dir`` —
       per-file ``hf_hub_download`` of exactly the two URIs
       (``preprocessor_uri`` / ``postprocessor_uri``).
    2. Otherwise fall back to ``snapshot_download(repo_id)``. This path
       is what explicit-scheme URIs (e.g. ``hf://lerobot/diffusion_pusht``,
       which predates the per-file processor contract) still rely on.

    Args:
        spec: A sim-layer ``VLASpec`` (or duck-typed equivalent with a
            ``weights_uri`` attribute).
        repo_id: The Hugging Face Hub repo id resolved from
            ``spec.weights_uri`` — used by the snapshot fallback.

    Returns:
        Absolute path to a directory containing
        ``policy_preprocessor.json`` and ``policy_postprocessor.json``
        either symlinked in by the manifest-first path or downloaded by
        the snapshot fallback.
    """
    weights_uri = str(getattr(spec, "weights_uri", "") or "")
    if not weights_uri.startswith(("hf://", "local://", "file://", "http://", "https://")):
        from openral_rskill.loader import load_rskill_manifest

        manifest = load_rskill_manifest(weights_uri)
        if manifest.processors is not None:
            return materialize_processor_dir(manifest)

    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo_id=repo_id, ignore_patterns=["*.md"]))
