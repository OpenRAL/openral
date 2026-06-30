#!/usr/bin/env python3
"""Generate a standard agent-skill ``SKILL.md`` *discovery view* from each
``rskills/<id>/rskill.yaml`` manifest.

An rSkill is an executable robot-policy package (weights + a Pydantic-validated
``rskill.yaml`` contract + license/capability gates), **not** a prose agent
skill. This tool emits a derived ``SKILL.md`` so tools that read the standard
agent-skill format (`name` + `description` YAML frontmatter) can *discover* and
reason about OpenRAL rSkills. The ``rskill.yaml`` manifest remains the single
source of truth (CLAUDE.md §1.3); the generated ``SKILL.md`` is discovery-only
and never executes a policy — that always goes through ``rSkill.from_pretrained``
and the robot HAL.

Usage::

    python tools/generate_rskill_skillmd.py            # all rskills/<id>/
    python tools/generate_rskill_skillmd.py pi05-libero-nf4 smolvla-libero
    python tools/generate_rskill_skillmd.py --check     # fail if any are stale

This is a deterministic projection of the manifest: re-running it overwrites the
generated files. Do not hand-edit ``SKILL.md``; edit ``rskill.yaml`` and
regenerate. CLAUDE.md §1.11: real manifests, no placeholders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RSKILLS_DIR = _REPO_ROOT / "rskills"

# Weight licenses that are NOT fully permissive open source — the generated
# SKILL.md surfaces a warning so a discovering agent does not assume free
# commercial use. Mirrors RSkillLicensePosture semantics (third-party weights).
_PERMISSIVE_WEIGHT_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "nvidia_open_model"}

_KIND_NOUN = {
    "vla": "Vision-Language-Action policy",
    "detector": "object detector",
    "vlm": "vision-language model",
    "reward": "task-progress / reward monitor",
    "ros_action": "ROS action skill (weightless)",
    "ros_service": "ROS service skill (weightless)",
    "wam": "World Action Model",
    "playbook": "decision-procedure playbook (weightless)",
}


def _collapse(text: str | None) -> str:
    """Collapse a multi-line manifest string into one clean line."""
    if not text:
        return ""
    return " ".join(str(text).split())


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _sensor_summary(sensors: Any) -> list[str]:
    out: list[str] = []
    for s in sensors or []:
        if isinstance(s, dict):
            mod = s.get("modality", "?")
            key = s.get("vla_feature_key") or s.get("feature_key") or ""
            out.append(f"{mod}:{key}" if key else str(mod))
        else:
            out.append(str(s))
    return out


def _quant_summary(quant: Any) -> str:
    if not isinstance(quant, dict):
        return ""
    dtype = quant.get("dtype", "")
    backend = quant.get("backend", "")
    return f"{dtype}/{backend}".strip("/")


def _yaml_inline(value: Any) -> str:
    """Render a small scalar/list/dict as compact inline YAML for metadata."""
    return yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()


def _build_metadata(m: dict[str, Any], skill_name: str, rel_manifest: str) -> str:
    """Emit the ``metadata:`` block, including only fields present in the manifest."""
    license_str = str(m.get("license", "")).strip()
    weights_permissive = license_str.lower() in _PERMISSIVE_WEIGHT_LICENSES
    lines: list[str] = ["metadata:"]

    def add(key: str, value: Any, *, inline: bool = False) -> None:
        if value in (None, "", [], {}):
            return
        lines.append(f"  {key}: {_yaml_inline(value) if inline else value}")

    lines.append("  openral_rskill: true            # generated discovery view of an rSkill")
    add("schema_version", str(m.get("schema_version", "")).strip('"'))
    add("rskill_id", m.get("name"))
    lines.append(f"  manifest: {rel_manifest}")
    add("role", m.get("role"))
    add("kind", m.get("kind"))
    add("model_family", m.get("model_family"))
    add("embodiment_tags", _as_list(m.get("embodiment_tags")), inline=True)
    add("actions", _as_list(m.get("actions")), inline=True)
    add("objects", _as_list(m.get("objects")), inline=True)
    add("scenes", _as_list(m.get("scenes")), inline=True)
    sensors = _sensor_summary(m.get("sensors_required"))
    add("sensors_required", sensors, inline=True)
    if isinstance(m.get("state_contract"), dict):
        add("state_dim", m["state_contract"].get("dim"))
    if isinstance(m.get("action_contract"), dict):
        add("action_dim", m["action_contract"].get("dim"))
        add("action_representation", m["action_contract"].get("representation"))
    add("runtime", m.get("runtime"))
    quant = _quant_summary(m.get("quantization"))
    add("quantization", quant)
    add("min_vram_gb", m.get("min_vram_gb"), inline=True)
    add("chunk_size", m.get("chunk_size"))
    add("n_action_steps", m.get("n_action_steps"))
    if isinstance(m.get("latency_budget"), dict):
        add("latency_budget", m["latency_budget"], inline=True)
    lines.append("  license_code: Apache-2.0")
    if license_str:
        warn = "" if weights_permissive else "   # NOT permissive — see License section"
        lines.append(f"  license_weights: {license_str}{warn}")
    add("weights_uri", m.get("weights_uri"))
    add("source_repo", m.get("source_repo"))
    add("paper_url", m.get("paper_url"))
    return "\n".join(lines)


def _frontmatter_description(m: dict[str, Any], skill_name: str) -> str:
    kind = str(m.get("kind", "")).strip()
    role = str(m.get("role", "")).strip().upper()
    noun = _KIND_NOUN.get(kind, "rSkill")
    desc = _collapse(m.get("description"))
    head = f"{role} {noun}." if role else f"{noun}."
    verbs = ", ".join(_as_list(m.get("actions"))[:6])
    objs = ", ".join(_as_list(m.get("objects"))[:6])
    cap = ""
    if verbs:
        cap = f" Capabilities: {verbs}"
        if objs:
            cap += f" on {objs}"
        cap += "."
    body = (
        f"{head}{cap} {desc} Discovery view of an OpenRAL rSkill — NOT directly "
        f"runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL."
    )
    return _collapse(body)


def render_skill_md(manifest_path: Path) -> str:
    m = yaml.safe_load(manifest_path.read_text())
    if not isinstance(m, dict):
        raise ValueError(f"{manifest_path} did not parse to a mapping")
    skill_name = manifest_path.parent.name
    rel_manifest = "./rskill.yaml"
    license_str = str(m.get("license", "")).strip()
    weights_permissive = license_str.lower() in _PERMISSIVE_WEIGHT_LICENSES
    kind = str(m.get("kind", "")).strip()
    noun = _KIND_NOUN.get(kind, "rSkill")

    front = (
        "---\n"
        f"name: {skill_name}\n"
        f"description: >-\n  {_frontmatter_description(m, skill_name)}\n"
        f"{_build_metadata(m, skill_name, rel_manifest)}\n"
        "---\n"
    )

    actions = _as_list(m.get("actions"))
    objects = _as_list(m.get("objects"))
    scenes = _as_list(m.get("scenes"))
    embod = _as_list(m.get("embodiment_tags"))

    cap_lines = []
    if actions:
        cap_lines.append(f"- **Verbs:** {' · '.join(actions)}")
    if objects:
        cap_lines.append(f"- **Objects:** {' · '.join(objects)}")
    if scenes:
        cap_lines.append(f"- **Scenes:** {' · '.join(scenes)}")
    if embod:
        cap_lines.append(f"- **Embodiments:** {' · '.join(embod)}")
    cap_block = "\n".join(cap_lines) if cap_lines else "_See the manifest for capabilities._"

    if weights_permissive or not m.get("weights_uri"):
        lic_block = (
            f"- **Code:** Apache-2.0.\n"
            f"- **Weights:** `{license_str or 'n/a'}` — permissive / commercial-use OK"
            if m.get("weights_uri")
            else "- **Code:** Apache-2.0. This is a weightless rSkill (the manifest *is* the artifact)."
        )
    else:
        lic_block = (
            f"- **Code:** Apache-2.0.\n"
            f"- **Weights:** `{license_str}` — **NOT** fully permissive. The loader surfaces this "
            f"posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where "
            f"applicable. Commercial use may require a separate upstream agreement. This is third-party "
            f"weight lineage; OpenRAL's own code is Apache-2.0."
        )

    # from_pretrained takes the canonical rSkill id (manifest `name`), NOT the
    # weights_uri (which often points at the upstream model the rSkill wraps).
    install_id = str(m.get("name") or f"OpenRAL/rskill-{skill_name}").removeprefix("hf://")

    body = f"""
# {skill_name} — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **{noun}** (`role: {m.get("role", "?")}`, `kind: {kind or "?"}`). {_collapse(m.get("description"))}

## Capabilities

{cap_block}

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract{", model weights," if m.get("weights_uri") else ""}
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

{lic_block}

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("{install_id}")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
"""
    return front + body


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("ids", nargs="*", help="rSkill ids (dir names). Default: all under rskills/.")
    p.add_argument(
        "--check", action="store_true", help="Fail if any SKILL.md is missing or stale (no writes)."
    )
    args = p.parse_args(argv)

    if args.ids:
        dirs = [_RSKILLS_DIR / i for i in args.ids]
    else:
        dirs = sorted(d for d in _RSKILLS_DIR.iterdir() if (d / "rskill.yaml").exists())

    stale: list[str] = []
    wrote = 0
    for d in dirs:
        manifest = d / "rskill.yaml"
        if not manifest.exists():
            print(f"[skip] {d.name}: no rskill.yaml", file=sys.stderr)
            continue
        try:
            content = render_skill_md(manifest)
        except Exception as e:  # report per-file failure, keep going
            print(f"[FAIL] {d.name}: {e}", file=sys.stderr)
            stale.append(d.name)
            continue
        out = d / "SKILL.md"
        if args.check:
            if not out.exists() or out.read_text() != content:
                stale.append(d.name)
            continue
        out.write_text(content)
        wrote += 1
        print(f"[ok] {out.relative_to(_REPO_ROOT)}")

    if args.check and stale:
        print(f"\nSTALE/MISSING SKILL.md for: {', '.join(stale)}", file=sys.stderr)
        print("Run: python tools/generate_rskill_skillmd.py", file=sys.stderr)
        return 1
    if not args.check:
        print(f"\nGenerated {wrote} SKILL.md file(s).")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
