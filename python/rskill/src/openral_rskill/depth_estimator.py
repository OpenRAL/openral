"""The ``kind: depth_estimator`` rSkill a depth provider may run, checked before anything runs.

One copy of the rule for ``openral deploy`` (before launch), ``deploy_e2e.launch.py`` and
``depth_provider_node`` (at configure): where a skill reference points, and whether that
skill may serve the requested mode, license gate included (``rSkill.from_yaml``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openral_core.exceptions import ROSConfigError

from openral_rskill.loader import rSkill

if TYPE_CHECKING:
    from openral_core import DepthEstimatorContract


def depth_rskill_manifest(ref: str, rskills_dir: str | Path) -> Path:
    """The absolute ``rskill.yaml`` a depth-skill reference names.

    ``ref`` is a directory name under ``rskills_dir`` or a path to an ``rskill.yaml``; a
    relative path resolves against the checkout that holds ``rskills_dir`` (its parent),
    the same root a directory name resolves under.

    Raises:
        ROSConfigError: no manifest at the resolved path.

    Example:
        >>> path = depth_rskill_manifest("rskill-da3_small-any-depth-bf16", "rskills")
        >>> path.is_absolute(), path.parts[-2:]
        (True, ('rskill-da3_small-any-depth-bf16', 'rskill.yaml'))
        >>> depth_rskill_manifest(
        ...     "rskills/rskill-da3_small-any-depth-bf16/rskill.yaml", "rskills"
        ... ) == path
        True
    """
    root = Path(rskills_dir).resolve()
    if ref.endswith(".yaml"):
        path = Path(ref) if Path(ref).is_absolute() else root.parent / ref
    else:
        path = root / ref / "rskill.yaml"
    if not path.is_file():
        raise ROSConfigError(f"depth_estimator rSkill {ref!r}: no manifest at {path}")
    return path


def depth_skill(manifest_path: str | Path, *, metric: bool) -> tuple[DepthEstimatorContract, str]:
    """The depth_estimator contract and model repo a provider serves, refused if unusable.

    Loaded through ``rSkill.from_yaml``, so a non-commercial skill (Depth Pro 2,
    ``apple-amlr``) needs ``OPENRAL_ALLOW_NONCOMMERCIAL=1``. Metric mode (world collision)
    needs a ``metric`` / ``canonical_focal`` model; the SLAM mode (no RGB ``CameraInfo``)
    needs one that predicts its own camera. The repo is the ``weights_uri`` without
    ``hf://`` and revision: what every sidecar reply must name.

    Raises:
        ROSConfigError: not a depth_estimator, the license blocks it, or it does not fit
            the mode.

    Example:
        >>> root = "rskills/rskill-da3metric_large-any-metric_depth-bf16/rskill.yaml"
        >>> import contextlib, io
        >>> with contextlib.redirect_stdout(io.StringIO()):  # the provenance warning
        ...     contract, repo = depth_skill(root, metric=True)
        ...     try:
        ...         depth_skill(root, metric=False)
        ...     except ROSConfigError as exc:
        ...         slam_refusal = str(exc)
        >>> contract.output.value, repo, "predicts no camera" in slam_refusal
        ('canonical_focal', 'depth-anything/DA3METRIC-LARGE', True)
    """
    manifest = rSkill.from_yaml(manifest_path).manifest  # schema + license gate
    contract = manifest.depth_estimator
    if contract is None or manifest.weights_uri is None:
        raise ROSConfigError(f"depth_provider: {manifest.name} is not a depth_estimator")
    if metric and contract.output.value == "relative":
        raise ROSConfigError(
            f"depth_provider: {manifest.name} predicts relative depth; metric mode needs a "
            "metric or canonical_focal depth_estimator"
        )
    if not metric and not contract.predicts_intrinsics:
        raise ROSConfigError(
            f"depth_provider: {manifest.name} predicts no camera; the SLAM mode needs a "
            "depth_estimator with predicts_intrinsics"
        )
    return contract, manifest.weights_uri.removeprefix("hf://").partition("@")[0]
