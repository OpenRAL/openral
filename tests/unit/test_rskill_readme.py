"""HF model-card generation for rSkill READMEs (`openral_cli._rskill_readme`).

The manifest is the single source of truth; the model-card front-matter is
derived from it so every published repo — private or public — carries a
consistent card. These tests pin the derivation + the best-of-both merge against
real in-tree manifests (no placeholders).
"""

from __future__ import annotations

import pathlib

import yaml
from openral_cli._rskill_readme import (
    build_rskill_frontmatter,
    build_rskill_readme,
)
from openral_core.schemas import RSkillManifest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _manifest(name: str) -> RSkillManifest:
    return RSkillManifest.from_yaml(str(_ROOT / "rskills" / name / "rskill.yaml"))


def test_vla_frontmatter_derives_model_card_fields() -> None:
    fm = build_rskill_frontmatter(_manifest("act-aloha"))
    assert fm["library_name"] == "lerobot"
    assert fm["pipeline_tag"] == "robotics"
    assert fm["license"] == "mit"
    assert fm["inference"] is False
    assert {"OpenRAL", "rskill", "act", "vision-language-action"} <= set(fm["tags"])


def test_nf4_derives_base_model_tree_and_quant_tags() -> None:
    fm = build_rskill_frontmatter(_manifest("molmoact2-so101-nf4"))
    # base_model comes from source_repo (weights_uri self-hosts the NF4 pack)
    assert fm["base_model"] == ["allenai/MolmoAct2-SO100_101"]
    assert fm["base_model_relation"] == "quantized"
    assert {"nf4", "4-bit"} <= set(fm["tags"])
    assert fm["license"] == "apache-2.0"


def test_detector_uses_object_detection_pipeline() -> None:
    fm = build_rskill_frontmatter(_manifest("rtdetr-coco-r18"))
    assert fm["pipeline_tag"] == "object-detection"
    assert "detector" in fm["tags"]


def test_readme_merge_is_best_of_both_and_idempotent() -> None:
    m = _manifest("act-aloha")
    body = "---\ntags: [OpenRAL, hand-curated]\ncustom: keep\n---\n\n# body\n\n## License\nMIT.\n"
    card = build_rskill_readme(m, body)
    fm = yaml.safe_load(card.split("---\n")[1])
    # hand-curated extras survive; derived fields still present and win
    assert "hand-curated" in fm["tags"]
    assert fm["custom"] == "keep"
    assert fm["library_name"] == "lerobot"
    # body preserved, stale front-matter not duplicated
    assert card.count("custom: keep") == 1 and "\n# body" in card
    # regenerating the built card changes nothing
    assert build_rskill_readme(m, card) == card


def test_every_intree_readme_is_current_and_valid() -> None:
    """Each in-tree README already equals its generated form (CI guard) + validates."""
    from openral_cli._rskill_doc_validator import validate_rskill_docs

    stale = []
    for mf in sorted(_ROOT.glob("rskills/*/rskill.yaml")):
        d = mf.parent
        if d.name == "template":
            continue
        m = RSkillManifest.from_yaml(str(mf))
        rp = d / "README.md"
        if not rp.exists():
            stale.append(f"{d.name}: missing README")
            continue
        body = rp.read_text(encoding="utf-8")
        if build_rskill_readme(m, body) != body:
            stale.append(f"{d.name}: README front-matter out of date (regenerate)")
        report = validate_rskill_docs(d, m)
        errs = [i for i in report.issues if "error" in str(getattr(i, "severity", "")).lower()]
        assert not errs, f"{d.name}: doc-validator errors {errs}"
    assert not stale, "stale READMEs (run the model-card regen): " + "; ".join(stale)


def test_catalog_tags_and_openral_block_contract() -> None:
    """The reserved tags + ``openral:`` block the website / gated search read (exact shape)."""
    m = _manifest("act-aloha")
    fm = build_rskill_frontmatter(m)
    reserved = [t for t in fm["tags"] if ":" in t]
    assert reserved == [
        f"rskill-kind:{m.kind}",
        f"rskill-role:{m.role}",
        *[f"embodiment:{t}" for t in m.embodiment_tags],
        f"rskill-license:{m.license.value}",
    ]
    assert not any(t.startswith("license:") for t in fm["tags"])
    block = fm["openral"]
    assert list(block) == ["description", "min_vram_gb", "latency_ms", "runtime", "dtype"]
    assert block["description"] == " ".join(m.description.split())
    assert block["latency_ms"] == m.latency_budget.per_chunk_ms
    assert block["runtime"] == m.runtime.value
    assert block["dtype"] == m.quantization.dtype.value
    # deterministic: same manifest, same card
    assert build_rskill_readme(m, "# b\n") == build_rskill_readme(m, "# b\n")


def test_openral_min_vram_gb_per_dtype_map() -> None:
    """Per-dtype ``min_vram_gb`` → the active dtype's entry, else the smallest; unset → null."""
    m = _manifest("pi05-libero-int8")  # {fp32: 16.1, bf16: 8.1, int8: 6.4}, dtype int8
    assert build_rskill_frontmatter(m)["openral"]["min_vram_gb"] == 6.4
    assert m.min_vram_gb is not None
    no_active = m.model_copy(
        update={"min_vram_gb": {k: v for k, v in m.min_vram_gb.items() if k.value != "int8"}}
    )
    assert build_rskill_frontmatter(no_active)["openral"]["min_vram_gb"] == 8.1
    unset = m.model_copy(update={"min_vram_gb": None})
    assert build_rskill_frontmatter(unset)["openral"]["min_vram_gb"] is None
