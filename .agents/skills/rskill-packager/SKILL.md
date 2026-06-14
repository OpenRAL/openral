---
name: rskill-packager
description: 'Use when creating, quantizing, validating, or publishing OpenRAL rSkills: rskill.yaml, rskills/template README, eval artifacts, rskill_scaffolder, rskill_publisher, quantize_rskill, Hugging Face Hub, VLA adapters, embodiment tags, license gates, latency budgets, nf4, int8, model.safetensors.'
argument-hint: 'rSkill id/path, source checkpoint, target HF repo, robot target, or publish goal'
---

# rSkill Packager

Create, quantize, validate, and publish OpenRAL rSkills without violating the schema, license, safety, or reproducibility rules. An rSkill is a Hugging Face Hub-shaped package: `rskill.yaml`, `README.md`, optional `eval/*.json`, and model/runtime assets or pointers.

## When to Use

- Scaffolding a new `rskills/<id>/` package.
- Filling or reviewing `rskill.yaml` for a VLA, detector, controller, or policy adapter.
- Quantizing a lerobot or transformers checkpoint into `model.safetensors` plus `quantization_metadata.json`.
- Publishing an rSkill package to Hugging Face Hub.
- Checking README quality, eval artifacts, license posture, unsafe pickle, remote-code, non-commercial, embodiment, capability, latency, or provenance gates.

## Required Context

Read only the context needed for the task:

1. `CLAUDE.md` for rSkill packaging, VLA licenses, safety, tests, docs, and no-mocks rules.
2. `docs/tutorials/rskill/write-and-publish-an-rskill.md` for the end-to-end workflow.
3. `docs/reference/rskills.md` for shipped packages and package conventions.
4. `docs/reference/vla_compatibility.md` for embodiment tags, dimensions, camera keys, model limitations, and license caveats.
5. `rskills/template/README.md` and `rskills/template/rskill.yaml` for the canonical scaffold shape.
6. `tools/rskill_scaffolder.py`, `tools/rskill_publisher.py`, and `tools/quantize_rskill.py` before recommending exact commands.
7. `python/core/src/openral_core/schemas.py` when manifest or eval schema details are unclear.

## References

- Load [rskill-workflow.md](./references/rskill-workflow.md) when you need the full packaging context blueprint, tool map, quantization checklist, publish gate, or command catalog.
- Do not copy or move `tools/rskill_scaffolder.py`, `tools/rskill_publisher.py`, or `tools/quantize_rskill.py` into this skill. They are canonical repo tools and should stay in `tools/`.

## Workflow

1. Establish the source and target.
   - Source may be a local checkpoint, HF repo, existing rSkill folder, lerobot policy, transformers custom-code model, detector, or adapter.
   - Target must include rSkill ID, HF owner/repo, embodiment tags, robot/sim target, and intended role: `s0`, `s1`, or `s2`.
   - Record the exact upstream revision whenever possible.

2. Scaffold the package.
   - Prefer `openral rskill new <id>` when the CLI is installed.
   - The standalone wrapper is available as `uv run python tools/rskill_scaffolder.py <id> --out-dir rskills/<id> ...`; pass `--out-dir` explicitly.
   - Use `--from-hf` or family-aware defaults when available, then inspect the generated manifest manually.

3. Fill the manifest from evidence.
   - Verify state dimension, action dimension, control mode, camera keys, image sizes, preprocessing, normalization sidecars, chunk size, runtime, and latency budget from checkpoint metadata, compatibility docs, or existing package patterns.
   - Match `embodiment_tags` with `RobotCapabilities.embodiment_tags`.
   - Preserve `schema_version: "0.1"` while OpenRAL is pre-publish.
   - Do not invent manifest fields. Search nearby `rskills/*/rskill.yaml` and the Pydantic schema first.

4. Set license and provenance gates honestly.
   - GR00T-derived weights are non-commercial.
   - RLDX-1 weights are non-commercial and sidecar-based.
   - pi0/pi0.5 weights are permissive research, not full Apache-2.0.
   - `*.pt`/pickle weights require `OPENRAL_ALLOW_UNSAFE_PICKLE=1`; prefer `model.safetensors`.
   - `trust_remote_code` requires `OPENRAL_ALLOW_REMOTE_CODE=1` or a trusted org path.
   - Sigstore signing and provenance verification are planned, not implemented. Never call an rSkill signed or verified.

5. Quantize when requested.
   - Use `tools/quantize_rskill.py` for one-shot packaging of post-quantization weights.
   - The default path loads a lerobot policy via `--policy-class`, rewrites large `Linear` modules to nf4, writes `model.safetensors`, writes `quantization_metadata.json`, copies metadata sidecars, and uploads to the target repo unless `--skip-upload` is set.
   - Use `--loader transformers --trust-remote-code` only for custom-code models that need it, and respect the trusted-org / `OPENRAL_ALLOW_REMOTE_CODE=1` gate.
   - `--scheme nf4` is runtime-loader-backed. `--scheme int8` is upload-only unless the runtime adapter has an int8 fast path.
   - `--device cuda` requires CUDA. Check the local GPU before falling back or skipping.

6. Validate docs and evals.
   - Fill every README section from `rskills/template/README.md`; template sentinels are publish-blocking.
   - Add real `eval/<benchmark>.json` artifacts when available and ensure they validate as `SkillEvalResult`.
   - Paper-cited values must be labeled as not locally reproduced and include a reproduction command.

7. Publish safely.
   - Dry-run first: `uv run python tools/rskill_publisher.py rskills/<id>/`.
   - Publish only after validation passes: `uv run python tools/rskill_publisher.py rskills/<id>/ --publish`.
   - Use `--bump-revision` to pin upstream `weights_uri` to a commit SHA before upload.
   - The publisher creates private repos only; do not weaken this gate.
   - A write token is required via `HF_TOKEN`, cached Hugging Face login, or `--token` depending on the tool path.

8. Update docs and tests.
   - Update `docs/reference/rskills.md`, `docs/reference/vla_compatibility.md`, package README, and the matching `docs/methods/` file when relevant.
   - Add real-fixture tests for manifest loading, compatibility, eval loading, or runtime adapter behavior. Do not add mocks, fake manifests, or placeholder robot names.

## Command Patterns

```bash
openral rskill new pi05-pick-cube --from-hf <owner>/<repo>
uv run python tools/rskill_scaffolder.py pi05-pick-cube --out-dir rskills/pi05-pick-cube --owner <owner> --embodiment-tag franka_panda
uv run python tools/rskill_publisher.py rskills/pi05-pick-cube/
uv run python tools/rskill_publisher.py rskills/pi05-pick-cube/ --publish --bump-revision
HF_TOKEN=<token> uv run python tools/quantize_rskill.py --source <source> --target <owner>/rskill-<id> --policy-class <module.Class>
HF_TOKEN=<token> uv run python tools/quantize_rskill.py --source <source> --target <owner>/rskill-<id>-nf4 --loader transformers --trust-remote-code --skip-upload
```

## Output Checklist

Report the package path or HF repo, source checkpoint and revision, target robot/sim, embodiment tags, sensors, state/action dimensions, license/provenance caveats, quantization scheme, eval evidence, docs/tests changed, validation commands, and remaining blockers.

## Stop Conditions

Stop before publishing, quantizing, or documenting claims when required metadata is unknown, safety/license/provenance gates would be bypassed, remote code would execute without an explicit gate, or benchmark results would need to be guessed.