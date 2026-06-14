# rSkill Packaging Reference

Use this reference after `SKILL.md` is loaded and the task is specifically about packaging, quantizing, validating, or publishing an rSkill.

## Context Blueprint

Collect these facts before editing files or running publishing tools:

| Area | Required facts | Evidence source |
| --- | --- | --- |
| Identity | rSkill ID, HF owner, target repo, package path | user request, `rskills/`, HF repo |
| Source | checkpoint path/repo, revision, loader, policy class | HF metadata, local files, model card |
| Robot | robot ID, embodiment tags, control mode | `robots/<id>/robot.yaml`, compatibility docs |
| Sensors | camera keys, resolutions, modalities, VLA feature keys | manifest, checkpoint config, docs |
| Contracts | state dim, action dim, chunk size, action semantics | checkpoint sidecars, adapter docs, existing manifests |
| Runtime | backend, dtype, quantization scheme, min VRAM, device | manifest, `quantization_metadata.json`, tool output |
| License | code license, weights license, commercial-use posture | model card, docs, manifest |
| Eval | benchmark, episodes, seed, reproduced locally, CLI | `eval/*.json`, benchmark output |

Unknown contract or license facts are blockers. Do not infer them from package naming alone.

## Tool Map

| Tool | Purpose | Notes |
| --- | --- | --- |
| `openral rskill new` | Preferred scaffold command when CLI is available | Supports HF introspection and family defaults |
| `tools/rskill_scaffolder.py` | Standalone scaffold wrapper | Use `--out-dir rskills/<id>` explicitly |
| `tools/quantize_rskill.py` | One-shot quantize + optional upload helper | Use `--skip-upload` first unless upload is intended |
| `tools/rskill_publisher.py` | Validate and publish local rSkill folders | Dry-run by default; publishes private repos only |
| `openral rskill check` | Compatibility check for installed/in-tree rSkills | Use with `--robot` when checking a target robot |
| `openral sim run` | Produces real sim evidence and summaries | Pair scene config with `--rskill` |

Do not move these project tools into the skill `scripts/` directory. They are canonical repo tooling, referenced by docs and usable outside the agent system.

## Command Catalog

Scaffold:

```bash
openral rskill new pi05-pick-cube --from-hf <owner>/<repo>
uv run python tools/rskill_scaffolder.py pi05-pick-cube \
  --out-dir rskills/pi05-pick-cube \
  --owner <owner> \
  --license apache-2.0 \
  --embodiment-tag franka_panda
```

Validate local package:

```bash
uv run python tools/rskill_publisher.py rskills/pi05-pick-cube/
openral rskill check pi05-pick-cube --robot robots/franka_panda/robot.yaml
```

Quantize locally first:

```bash
uv run python tools/quantize_rskill.py \
  --source <source-repo-or-path> \
  --target <owner>/rskill-<id>-nf4 \
  --policy-class <module.PolicyClass> \
  --scheme nf4 \
  --device cuda \
  --skip-upload
```

Transformers custom-code path:

```bash
OPENRAL_ALLOW_REMOTE_CODE=1 uv run python tools/quantize_rskill.py \
  --source <source-repo> \
  --target <owner>/rskill-<id>-nf4 \
  --loader transformers \
  --trust-remote-code \
  --scheme nf4 \
  --device cuda \
  --skip-upload
```

Publish after dry-run passes:

```bash
uv run python tools/rskill_publisher.py rskills/pi05-pick-cube/ --bump-revision --publish
```

If a command needs an HF token, the user must type the secret into the terminal or rely on cached Hugging Face login. Never ask for a token in chat.

## Manifest Checklist

- `name` uses the target Hub repo ID, usually `<owner>/rskill-<id>`.
- `version` matches the package intent.
- `license` reflects weights posture, not only code license.
- `commercial_use_allowed` is consistent with `license`.
- `role` is correct: `s1` for fast VLA policy, `s2` for reasoner-like skill, `s0` only for bounded low-level control.
- `kind` matches the package behavior, such as VLA or detector.
- `embodiment_tags` intersect the target robot capabilities.
- `sensors_required` maps to real VLA feature keys and camera geometry.
- `actuators_required` and action semantics match the robot/control mode.
- `state_contract` and `action_contract` are present when dataset bridge or action-contract consumers need them.
- `weights_uri` points to the intended artifact and is pinned for stable publication.
- `runtime`, `quantization`, and min VRAM match the load path.
- `latency_budget` is evidence-backed.
- `schema_version` remains `"0.1"` while OpenRAL is pre-publish.

## Quantization Checklist

- CUDA availability checked before `--device cuda`.
- Source repo/path and revision recorded.
- Loader chosen correctly: `lerobot` or `transformers`.
- `--policy-class` dotted path resolves for lerobot policies.
- `--trust-remote-code` used only with trusted-org or explicit env gate.
- `--scheme nf4` preferred when runtime loading is needed.
- `--scheme int8` treated as upload-only until runtime adapter support exists.
- `quantization_metadata.json` reviewed before publication.
- Source weight shards were not copied into the target package.

## Publish Gate

Before publication:

- Manifest validates as `RSkillManifest`.
- README passes doc validator and has no template sentinels.
- Eval artifacts are real or clearly labeled as unreproduced paper values.
- License and provenance warnings are visible in README/manifest.
- Repo is private at upload time.
- Upstream `weights_uri` is pinned when appropriate.
- Docs and tests travel with any code/schema changes.

## Hard Stops

- Missing license evidence.
- Unknown state/action dimension.
- Unknown camera key or preprocessing for a vision policy.
- Attempt to publish restricted weights as permissive/open commercial.
- Attempt to describe sigstore signing or provenance verification as implemented.
- User asks to bypass unsafe pickle, remote-code, capability, license, or embodiment gates.