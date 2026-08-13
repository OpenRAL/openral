"""Manifest → segmenter-backend dispatch for ``kind: segmenter`` rSkills.

The sibling of :mod:`~openral_runner.backends.gstreamer.detector_factory`, and
GStreamer-free for the same reason: the manifest↔backend selection stays
unit-testable without a live pipeline or a model load.

Dispatch keys on the manifest's ``segmenter.engine``, which is REQUIRED — there
is no legacy ``runtime``-keyed fallback to preserve for this kind, so unlike
:func:`~openral_runner.backends.gstreamer.detector_factory.build_manifest_detector`
this never has to guess:

* ``engine: sam2_hf`` → the in-process Transformers promptable segmenter
  (:class:`~.sam2_segmenter.Sam2Segmenter`). No ``onnx_path``; the model loads
  under the runtime's own ``transformers``.
"""

from __future__ import annotations

from openral_core.exceptions import ROSConfigError
from openral_core.schemas import RSkillManifest, SegmenterEngine

from openral_runner.backends.gstreamer.detector_factory import weights_source_from_manifest
from openral_runner.backends.gstreamer.sam2_segmenter import Sam2Segmenter

__all__ = ["build_manifest_segmenter"]


def build_manifest_segmenter(manifest: RSkillManifest) -> Sam2Segmenter:
    """Build the segmenter backend for a ``kind: segmenter`` manifest.

    Args:
        manifest: A validated manifest with ``kind == "segmenter"``.

    Returns:
        The backend instance, which exposes
        ``segment(frame_bgr, width, height, *, positive_points, negative_points)``
        and must be :meth:`~.sam2_segmenter.Sam2Segmenter.warm_up`-ed at node
        activate.

    Raises:
        ROSConfigError: If the manifest is not a ``kind: segmenter`` with a
            ``segmenter`` block, or names an engine with no backend.

    Example:
        >>> from openral_core import RSkillManifest
        >>> from pathlib import Path
        >>> import yaml
        >>> path = Path("rskills/rskill-sam2_1-any-grasped_object_mask-bf16/rskill.yaml")
        >>> m = RSkillManifest.model_validate(yaml.safe_load(path.read_text()))
        >>> build_manifest_segmenter(m).model_id
        'OpenRAL/rskill-sam2_1-any-grasped_object_mask-bf16'
    """
    if manifest.kind != "segmenter" or manifest.segmenter is None:
        raise ROSConfigError(
            f"build_manifest_segmenter: manifest {manifest.name!r} is not a "
            f"kind:segmenter with a segmenter block (kind={manifest.kind!r})."
        )
    contract = manifest.segmenter
    if contract.engine is not SegmenterEngine.SAM2_HF:
        raise ROSConfigError(
            f"build_manifest_segmenter: manifest {manifest.name!r} names engine "
            f"{contract.engine.value!r}, which has no backend."
        )
    # `weights_source_from_manifest` is shared with the detector factory rather
    # than reimplemented (CLAUDE.md §1.13): it applies the same `hf://` stripping
    # and the same immutable-revision requirement for `trust_remote_code`
    # checkpoints. Its `source_repo` fallback is unreachable here — the schema
    # makes `weights_uri` REQUIRED for `kind: segmenter`.
    return Sam2Segmenter(
        model_id=manifest.name,
        weights_source=weights_source_from_manifest(manifest),
        max_prompt_points=contract.max_prompt_points,
        multimask=contract.multimask,
        min_mask_area_px=contract.min_mask_area_px,
    )
