"""Private helper for the manifest-first per-file processor-dir resolution.

Closes the three sister TODOs in :mod:`openral_sim.policies.diffusion`,
:mod:`openral_sim.policies.xvla`, and :mod:`openral_sim.policies.pi05` —
each previously called ``huggingface_hub.snapshot_download`` to fetch the
``policy_preprocessor.json`` / ``policy_postprocessor.json`` sidecars
needed by lerobot's ``make_pre_post_processors`` /
``PolicyProcessorPipeline.from_pretrained``. The SmolVLA and modern-ACT
adapters already migrated to the per-file URI contract declared on
``RSkillManifest.processors`` (ADR-0013); this helper extends the same
pattern to the remaining three adapters.

The function deliberately accepts the sim-layer ``VLASpec`` (and a raw
``repo_id`` fallback) rather than requiring callers to pre-load the
manifest themselves — the snapshot fallback is a real path that
explicit-scheme URIs still rely on (e.g. ``hf://lerobot/diffusion_pusht``,
which predates the per-file processor contract).
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
       :func:`~openral_rskill._vla_core.materialize_processor_dir` —
       per-file ``hf_hub_download`` of exactly the two URIs
       (``preprocessor_uri`` / ``postprocessor_uri``).
    2. Otherwise fall back to ``snapshot_download(repo_id)``. This path
       is what explicit-scheme URIs (e.g. ``hf://lerobot/diffusion_pusht``,
       which predates ADR-0013) still rely on.

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
