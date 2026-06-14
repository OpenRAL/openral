"""Scaffolder for new rSkill packages — backs ``ral skill new``.

Copies ``rskills/template/`` to a target directory and rewrites a small,
fixed set of placeholders (manifest ``name`` / ``license`` /
``embodiment_tags``), then re-validates the result through
`RSkillManifest.from_yaml` and `rSkill.from_yaml` so any
malformed scaffold fails at scaffold-time rather than lazily on first
load.

Used by:

- ``ral skill new <id>`` (Typer command in :mod:`openral_cli.main`).
- ``tools/rskill_scaffolder.py`` (standalone argparse wrapper).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import structlog
import yaml
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    EmbodimentTag,
    RSkillLicensePosture,
    RSkillManifest,
)
from openral_rskill.loader import rSkill
from pydantic import ValidationError

from openral_cli._rskill_intel import RSkillFamily, RSkillPatch, family_defaults

log = structlog.get_logger(__name__)

_TEMPLATE_DIR_NAME = "template"
_MANIFEST_FILENAME = "rskill.yaml"
_README_FILENAME = "README.md"


def _default_template_dir() -> Path:
    """Resolve the in-tree ``rskills/template/`` directory.

    Walks upward from this file's location until it finds a ``rskills/``
    sibling, then returns ``rskills/template``. This makes the helper
    work both when invoked from inside the installed CLI wheel and when
    invoked via ``tools/rskill_scaffolder.py`` against an editable
    workspace.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "rskills" / _TEMPLATE_DIR_NAME
        if candidate.is_dir():
            return candidate
    raise ROSConfigError(
        "Could not locate rskills/template/ relative to "
        f"{Path(__file__).resolve()}; pass template_dir= explicitly."
    )


def scaffold_rskill(
    rskill_id: str,
    *,
    out_dir: Path,
    owner: str,
    license_: RSkillLicensePosture,
    embodiment_tag: EmbodimentTag,
    family: RSkillFamily | None = None,
    patch: RSkillPatch | None = None,
    template_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Scaffold a new local rSkill package from ``rskills/template/``.

    Args:
        rskill_id: Local skill id, convention ``<policy>-<task>`` (e.g.
            ``pi05-pick-cube``). Becomes the ``<repo>`` segment of the
            generated manifest ``name`` field.
        out_dir: Destination directory. The scaffold lands here directly
            (no extra ``rskill_id`` segment is appended).
        owner: HF Hub owner segment for the manifest ``name`` field
            (e.g. ``"your-org"``).
        license_: License posture to write into the manifest.
        embodiment_tag: Canonical embodiment tag to declare in
            ``embodiment_tags``.
        family: Optional :data:`RSkillFamily`. When set,
            `openral_cli._rskill_intel.family_defaults` provides
            the manifest baseline (``model_family``, ``chunk_size``,
            ``quantization``, latency budget, …) so a fresh ACT scaffold
            doesn't ship pi0.5 numbers. ``None`` leaves the template
            (currently pi0.5-shaped) as the baseline.
        patch: Optional manifest patch to overlay on top of the family
            defaults — typically built by
            `openral_cli._rskill_intel.introspect_hf` from a
            checkpoint's ``config.json``. Wins over both ``family``
            defaults and template values, but loses to the rename /
            license rewrite.
        template_dir: Override the in-tree template location. Defaults
            to ``rskills/template/`` relative to this file.
        overwrite: If ``True``, an existing ``out_dir`` is removed
            before scaffolding. If ``False`` (default), a pre-existing
            ``out_dir`` raises `ROSConfigError`.

    Returns:
        The resolved ``out_dir`` path.

    Raises:
        ROSConfigError: ``out_dir`` already exists and ``overwrite`` is
            ``False``; or the template directory cannot be located; or
            the generated manifest fails Pydantic / loader validation
            (a partial scaffold is removed before re-raising).

    Example:
        >>> # from openral_cli._rskill_scaffolder import scaffold_rskill
        >>> # scaffold_rskill(
        >>> #     "pi05-pick-cube",
        >>> #     out_dir=Path("rskills/pi05-pick-cube"),
        >>> #     owner="your-org",
        >>> #     license_=RSkillLicensePosture.APACHE_2_0,
        >>> #     embodiment_tag="franka_panda",
        >>> # )
    """
    resolved_out = out_dir.resolve()
    src = (template_dir or _default_template_dir()).resolve()

    if not (src / _MANIFEST_FILENAME).is_file():
        raise ROSConfigError(f"template directory {src} is missing {_MANIFEST_FILENAME}")

    if resolved_out.exists():
        if not overwrite:
            raise ROSConfigError(
                f"refusing to overwrite existing path {resolved_out}; "
                "pass overwrite=True (or --overwrite on the CLI) to replace it."
            )
        shutil.rmtree(resolved_out)

    shutil.copytree(src, resolved_out)

    try:
        _rewrite_manifest(
            resolved_out / _MANIFEST_FILENAME,
            rskill_id=rskill_id,
            owner=owner,
            license_=license_,
            embodiment_tag=embodiment_tag,
            family=family,
            patch=patch,
        )
        _rewrite_readme(
            resolved_out / _README_FILENAME,
            rskill_id=rskill_id,
            owner=owner,
        )
        _validate_scaffold(resolved_out)
    except (ROSConfigError, ValidationError):
        # Clean up the partial scaffold so a re-run doesn't trip the
        # "out_dir already exists" guard above.
        shutil.rmtree(resolved_out, ignore_errors=True)
        raise

    log.info(
        "rskill_scaffolder.created",
        rskill_id=rskill_id,
        out_dir=str(resolved_out),
        owner=owner,
        license=license_.value,
        embodiment_tag=embodiment_tag,
        family=family,
        from_hf=(patch or {}).get("weights_uri"),
    )
    return resolved_out


def _rewrite_manifest(
    manifest_path: Path,
    *,
    rskill_id: str,
    owner: str,
    license_: RSkillLicensePosture,
    embodiment_tag: EmbodimentTag,
    family: RSkillFamily | None,
    patch: RSkillPatch | None,
) -> None:
    """Rewrite the scaffolded fields in ``rskill.yaml`` in place.

    Application order (lowest → highest precedence):

    1. The on-disk template values (currently a pi0.5-shaped baseline).
    2. Family-aware defaults from
       `openral_cli._rskill_intel.family_defaults` when
       ``family`` is set — overrides ``model_family`` /
       ``chunk_size`` / ``quantization`` / latency budget / etc.
    3. The explicit ``patch`` (typically from ``--from-hf``
       introspection) — overrides family defaults with checkpoint-side
       facts (real chunk_size, real image-feature names, real state dim).
    4. CLI-driven rename / license rewrite — always authoritative.

    Round-trips through ``yaml.safe_load`` → mutate → ``yaml.safe_dump``
    so inline template comments are stripped (documented in the
    template header).
    """
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ROSConfigError(f"template manifest {manifest_path} did not parse as a mapping")

    # Layer (2): family defaults.
    if family is not None:
        _apply_patch(raw, dict(family_defaults(family)))

    # Layer (3): introspection patch.
    if patch:
        _apply_patch(raw, dict(patch))

    # Layer (4): the always-authoritative CLI rewrite. Re-apply *after*
    # the patch so a stale ``weights_uri`` left in the patch (e.g.
    # placeholder template URL) never wins.
    raw["name"] = f"{owner}/rskill-{rskill_id}"
    raw["license"] = license_.value
    raw["embodiment_tags"] = [embodiment_tag]
    if not raw.get("weights_uri", "").startswith("hf://") or "TEMPLATE" in str(raw["weights_uri"]):
        raw["weights_uri"] = f"hf://{owner}/{rskill_id}"
    if "source_repo" in raw and (
        not str(raw["source_repo"]).startswith("hf://") or "TEMPLATE" in str(raw["source_repo"])
    ):
        raw["source_repo"] = f"hf://{owner}/{rskill_id}"

    manifest_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _apply_patch(raw: dict[str, object], patch: dict[str, object]) -> None:
    """Overlay ``patch`` keys onto ``raw``, dropping keys explicitly set to ``None``.

    Setting a manifest key to ``None`` in a patch means "remove this
    optional block from the template" (e.g. the ACT family clears
    ``min_vram_gb`` / ``n_action_steps`` since neither applies). Any
    other value (including empty dicts / lists) overwrites in place.
    """
    for key, value in patch.items():
        if value is None:
            raw.pop(key, None)
        else:
            raw[key] = value


def _rewrite_readme(readme_path: Path, *, rskill_id: str, owner: str) -> None:
    """Replace ``TEMPLATE_ORG`` / ``TEMPLATE_ID`` sentinels in the README."""
    if not readme_path.is_file():
        return
    text = readme_path.read_text(encoding="utf-8")
    text = text.replace("TEMPLATE_ORG/rskill-TEMPLATE_ID", f"{owner}/rskill-{rskill_id}")
    text = text.replace("TEMPLATE_ORG", owner)
    text = text.replace("TEMPLATE_ID", rskill_id)
    readme_path.write_text(text, encoding="utf-8")


def _validate_scaffold(scaffold_dir: Path) -> None:
    """Re-validate the generated manifest through the full loader path."""
    manifest_path = scaffold_dir / _MANIFEST_FILENAME
    # Pydantic-level validation first — gives the cleanest error message
    # when the rewrite produced an invalid YAML shape.
    RSkillManifest.from_yaml(str(manifest_path))
    # Full rSkill loader path — runs the license guard + walks eval/*.json.
    # An empty eval/ directory is OK (loader's _validate_eval_jsons
    # early-returns on an empty glob).
    rSkill.from_yaml(manifest_path)
