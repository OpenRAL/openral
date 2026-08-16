"""Build an HF model-card README for an rSkill from its manifest.

The ``rskill.yaml`` manifest is the single source of truth (CLAUDE.md §1.3). The
HF model-card **front-matter** (license, tags, ``pipeline_tag``, ``library_name``,
the ``base_model`` tree, ``datasets``) is DERIVED from it here so every published
repo — **private or public** — carries a consistent, discoverable model card
without hand-maintaining YAML front-matter across ~30 READMEs. The human-written
prose **body** is kept verbatim; only the front-matter is (re)generated.

This is the "best of both" reconciliation: the OpenRAL-specific tags + accurate
license posture that the in-tree READMEs already carried, unioned with the HF
model-card fields (``library_name``, ``pipeline_tag``, ``inference``,
``base_model`` / ``base_model_relation``, ``datasets``) that only some hand-edited
HF cards had — now emitted uniformly from the manifest.

Example:
    >>> from openral_core.schemas import RSkillManifest
    >>> m = RSkillManifest.from_yaml("rskills/act-aloha/rskill.yaml")
    >>> card = build_rskill_readme(m, "# body\\n\\n## License\\nMIT.\\n")
    >>> card.startswith("---\\n")
    True
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from openral_core.schemas import RSkillLicensePosture, RSkillManifest

__all__ = ["build_rskill_frontmatter", "build_rskill_readme", "render_frontmatter"]

# lerobot ``PolicyProcessorPipeline`` families — these load through lerobot, so the
# HF ``library_name`` widget should point there. Other families (gr00t / rldx run
# out-of-process; molmoact2 is a transformers custom-code model) don't.
_LEROBOT_FAMILIES = frozenset({"smolvla", "act", "pi05", "diffusion", "xvla"})

# License posture → (HF license identifier, optional license_name). HF only knows
# a fixed SPDX-ish set; anything non-standard maps to ``other`` + a descriptive
# ``license_name`` (mirrors the accurate posture the in-tree READMEs carried).
_LICENSE_MAP: dict[RSkillLicensePosture, tuple[str, str | None]] = {
    RSkillLicensePosture.APACHE_2_0: ("apache-2.0", None),
    RSkillLicensePosture.MIT: ("mit", None),
    RSkillLicensePosture.BSD: ("bsd-3-clause", None),
    RSkillLicensePosture.PERMISSIVE_RESEARCH: ("other", "permissive-research"),
    RSkillLicensePosture.NVIDIA_NON_COMMERCIAL: ("other", "nvidia-non-commercial"),
    RSkillLicensePosture.NVIDIA_OPEN_MODEL: ("other", "nvidia-open-model-license"),
    RSkillLicensePosture.RLWRLD_NON_COMMERCIAL: ("other", "rlwrld-non-commercial"),
    RSkillLicensePosture.PROPRIETARY: ("other", "proprietary"),
    RSkillLicensePosture.UNKNOWN: ("other", None),
}

_NON_QUANTIZED_DTYPES = frozenset({"fp32", "bf16", "fp16", "float32", "float16"})

# rSkill kind → HF ``pipeline_tag`` / extra model-card tags. Table-driven so a new
# perception kind is one row, not another ``elif`` in the front-matter builder.
# Kinds absent from ``_KIND_PIPELINE_TAG`` fall back to ``robotics``; the ROS
# wrapper kinds are handled separately (their tags depend on the skill name).
_KIND_PIPELINE_TAG: dict[str, str] = {
    "detector": "object-detection",
    "segmenter": "mask-generation",
}
_KIND_EXTRA_TAGS: dict[str, list[str]] = {
    "vla": ["vision-language-action"],
    "detector": ["detector", "object-detection"],
    "segmenter": ["segmenter", "mask-generation", "promptable-segmentation"],
}


def _hf_repo(uri: str | None) -> str | None:
    """``hf://org/repo@sha`` → ``org/repo``; ``None``/non-hf → ``None``."""
    if not uri or "://" not in uri:
        return None
    scheme, rest = uri.split("://", 1)
    if scheme != "hf":
        return None
    return rest.split("@", 1)[0] or None


_FRONTMATTER_RE = re.compile(r"^---\n(?P<fm>.*?)\n---\n+", re.DOTALL)


def _strip_frontmatter(body: str) -> str:
    r"""Drop a leading ``---\\n…\\n---\\n`` YAML front-matter block if present."""
    m = _FRONTMATTER_RE.match(body)
    return body[m.end() :] if m else body.lstrip("\n")


def _parse_frontmatter(body: str) -> dict[str, Any]:
    """Parse a leading YAML front-matter block into a dict (``{}`` if none/invalid)."""
    m = _FRONTMATTER_RE.match(body)
    if not m:
        return {}
    try:
        loaded = yaml.safe_load(m.group("fm"))
        return loaded if isinstance(loaded, dict) else {}
    except yaml.YAMLError:
        return {}


def _dedup(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_rskill_frontmatter(manifest: RSkillManifest) -> dict[str, Any]:
    """Derive the HF model-card front-matter dict from an rSkill manifest.

    Deterministic and pure (no network, no HF read), so the publisher builds the
    same card whether the repo is private or public.
    """
    kind = str(getattr(manifest.kind, "value", manifest.kind) or "")
    family = manifest.model_family
    family_s = str(getattr(family, "value", family)) if family else None
    is_lerobot = family_s in _LEROBOT_FAMILIES

    fm: dict[str, Any] = {"language": ["en"]}

    lic_id, lic_name = _LICENSE_MAP.get(manifest.license, ("other", None))
    fm["license"] = lic_id
    if lic_name:
        fm["license_name"] = lic_name

    if is_lerobot:
        fm["library_name"] = "lerobot"
    fm["pipeline_tag"] = _KIND_PIPELINE_TAG.get(kind, "robotics")

    tags = ["OpenRAL", "rskill"]
    if family_s:
        tags.append(family_s)
    if is_lerobot:
        tags.append("lerobot")
    if kind in {"ros_action", "ros_service"}:
        tags += ["ros2", "moveit"] if "moveit" in manifest.name else ["ros2"]
    else:
        tags += _KIND_EXTRA_TAGS.get(kind, [])
    q = manifest.quantization
    if q is not None:
        dt = str(getattr(q.dtype, "value", q.dtype))
        if dt not in _NON_QUANTIZED_DTYPES:
            tags.append("nf4" if dt == "int4" else dt)
            tags.append("4-bit" if dt == "int4" else "8-bit" if dt == "int8" else "quantized")
    tags += list(manifest.embodiment_tags)
    fm["tags"] = _dedup(tags)

    base = _hf_repo(getattr(manifest, "source_repo", None))
    own = _hf_repo(manifest.weights_uri)
    if base and base != own:
        fm["base_model"] = [base]
        # relation reflects what the packaging did to the upstream: quantized when
        # a real quantization block is present, else a plain finetune/wrapper.
        dtype = str(getattr(q.dtype, "value", q.dtype)) if q is not None else ""
        is_quantized = q is not None and dtype not in _NON_QUANTIZED_DTYPES
        fm["base_model_relation"] = "quantized" if is_quantized else "finetune"

    ds = _hf_repo(getattr(manifest, "dataset_uri", None))
    if ds:
        fm["datasets"] = [ds]

    # rSkills are packaging manifests, not HF-inference-servable checkpoints.
    fm["inference"] = False
    return fm


def render_frontmatter(fm: dict[str, Any]) -> str:
    """Render a front-matter dict as a ``---``-fenced YAML block (stable order)."""
    body = yaml.safe_dump(fm, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return f"---\n{body}---\n"


def build_rskill_readme(manifest: RSkillManifest, body: str) -> str:
    """Return the full README = merged front-matter + the prose ``body``.

    "Best of both": the manifest-derived model-card fields are authoritative, but
    hand-curated extras already on ``body``'s front-matter are preserved — its
    ``tags`` are unioned onto the derived tags, and any scalar/list field the
    derive did not emit (e.g. a curated ``datasets`` or ``license_name``) is kept.
    Derived fields win on conflict so the card stays consistent with the manifest.
    """
    derived = build_rskill_frontmatter(manifest)
    existing = _parse_frontmatter(body)
    if existing:
        extra_tags = [t for t in existing.get("tags", []) or [] if isinstance(t, str)]
        derived["tags"] = _dedup([*derived.get("tags", []), *extra_tags])
        for k, v in existing.items():
            if k != "tags" and k not in derived and v not in (None, "", []):
                derived[k] = v
    return f"{render_frontmatter(derived)}\n{_strip_frontmatter(body)}"


def _demo() -> None:
    """Runnable self-check: front-matter derives the expected best-of-both fields."""
    root = Path(__file__).resolve()
    while root != root.parent and not (root / "rskills").is_dir():
        root = root.parent
    m = RSkillManifest.from_yaml(str(root / "rskills" / "pi05-libero-nf4" / "rskill.yaml"))
    fm = build_rskill_frontmatter(m)
    assert fm["library_name"] == "lerobot", fm
    assert fm["pipeline_tag"] == "robotics", fm
    assert fm["license"] == "other" and fm["license_name"] == "permissive-research", fm
    assert fm["base_model"] == ["lerobot/pi05_libero_finetuned_v044"], fm
    assert fm["base_model_relation"] == "quantized", fm
    assert "4-bit" in fm["tags"] and "OpenRAL" in fm["tags"], fm
    assert fm["inference"] is False
    # merge: hand-curated extra tags + local-only scalars survive; derived wins
    card = build_rskill_readme(
        m, "---\ntags: [OpenRAL, hand-tag]\ncustom_field: keep\n---\n\n# body\n\n## License\nx\n"
    )
    merged = _parse_frontmatter(card)
    assert "hand-tag" in merged["tags"] and merged.get("custom_field") == "keep", merged
    assert merged["library_name"] == "lerobot", "derived field lost in merge"
    n_fences = card.count("---\n")
    assert n_fences >= 2 and "\n# body" in card, n_fences  # noqa: PLR2004  # reason: yaml front-matter has exactly two fences
    # detector uses the object-detection pipeline
    d = RSkillManifest.from_yaml(str(root / "rskills" / "rtdetr-coco-r18" / "rskill.yaml"))
    assert build_rskill_frontmatter(d)["pipeline_tag"] == "object-detection"
    print("ok: _rskill_readme self-check passed")


if __name__ == "__main__":
    _demo()
