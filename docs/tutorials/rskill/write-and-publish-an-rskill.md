# Write an rSkill and publish it to the Hugging Face Hub

An **rSkill** is OpenRAL's packaging format for a robot skill: a Hugging Face
Hub repo containing an `rskill.yaml` manifest, a `README.md`, optional
`eval/<benchmark>.json` results, and (usually) a pointer to model weights that
live in an upstream repo. The manifest adds capability checking, license
surfacing, latency budgets, and local-registry integration on top of raw
weights — so installing a skill is as repeatable as installing a model.

This tutorial takes you from nothing to a published, installable rSkill. It
uses only real commands; nothing here is a mock.

## Prerequisites

```bash
just bootstrap && just sync   # toolchain + Python workspace (always `just sync`,
                              # never bare `uv sync` — see docs/contributing/toolchain.md)
openral doctor                # confirm the CLI and GPU/runtime
```

You'll also want a Hugging Face account and a write token (`HF_TOKEN`) if you
intend to publish.

## 1. Scaffold the package

`openral rskill new` copies the canonical template in
[`rskills/template/`](https://github.com/OpenRAL/openral/blob/master/rskills/template/)
and round-trips it through the schema, so a malformed scaffold fails
immediately rather than on first load. There are three modes:

```bash
# (a) Most intuitive — introspect a published checkpoint and pre-fill
#     model_family / chunk_size / sensors_required / state_contract /
#     image_preprocessing.aliases / weights_uri from its config.json:
openral rskill new pi05-pick-cube --from-hf <owner>/<repo>

# (b) Family-aware defaults without Hub introspection:
openral rskill new pi05-pick-cube --family pi05 --embodiment-tag franka_panda

# (c) Interactive — prompts for owner / license / embodiment / family:
openral rskill new pi05-pick-cube
```

Valid `--family` values: `act | smolvla | pi05 | xvla | diffusion`.
`--license` is one of `apache-2.0 | mit | bsd | permissive_research |
nvidia_non_commercial | proprietary | unknown`. `--embodiment-tag` must be a
canonical `EmbodimentTag` literal (e.g. `so100_follower`, `franka_panda`,
`aloha`). This writes `rskills/pi05-pick-cube/{rskill.yaml,README.md,eval/}`.

## 2. Fill in the manifest

Open `rskills/<id>/rskill.yaml`. The fields that matter most for consumers
(modeled on the in-tree
[`rskills/pi05-libero-nf4/rskill.yaml`](https://github.com/OpenRAL/openral/blob/master/rskills/pi05-libero-nf4/rskill.yaml)):

| Field | What it does |
| --- | --- |
| `name` | `<owner>/rskill-<id>` — the Hub repo id. |
| `license` | License **posture**; drives `commercial_use_allowed`. |
| `role` | `s1` (fast policy), `s2` (reasoner), or `s0` (cerebellar). |
| `kind` | `vla` for a learnable policy; detector kinds also exist. |
| `embodiment_tags` | Must match a robot's `RobotCapabilities.embodiment_tags`. |
| `sensors_required` | Modality + `vla_feature_key` + min resolution per camera. |
| `actuators_required` | Each entry needs `control_mode_semantics` (e.g. `mode: absolute`). |
| `runtime` / `quantization` | `pytorch` / `onnx` / `tensorrt`; `dtype` + `min_vram_gb`. |
| `weights_uri` | `hf://<owner>/<repo>` — the rSkill does **not** copy weights. |
| `chunk_size` / `n_action_steps` | Action-chunk size and replan cadence. |
| `latency_budget.per_chunk_ms` | Contractual — enforced by sim-tier latency tests. |
| `actions` / `objects` / `scenes` | ADR-0022 vocabulary the reasoner's LLM palette uses to pick the skill. |

The full schema is
[`openral_core.schemas.RSkillManifest`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py).
Strip the pi0.5-shaped knobs (`image_preprocessing`, `state_contract`,
`processors`) if your policy is a simpler ACT/Diffusion model that doesn't
need them.

## 3. Write the README

The scaffolded `README.md` is full of `<TODO>` / `TEMPLATE_ID` markers. These
are **publish-blocking**: the documentation validator
(`openral_cli._rskill_doc_validator.validate_rskill_docs`) refuses to upload a
README that still contains them. Fill in every section — what the skill does,
the observation→action contract, how it was trained, supported robots, sensors
required, and the license posture (keep `commercial_use_allowed` consistent
with `rskill.yaml::license`).

## 4. Check it runs on your robot

```bash
openral detect                           # auto-provisions ./robot.yaml
openral rskill check pi05-pick-cube --robot robot.yaml
```

`rskill check` prints a per-section breakdown — embodiment, capability flags,
GPU runtime, GPU dtype, sensors, actuators — and tells you whether the skill
will run on the current host. Run `openral rskill check` with no id to walk
every in-tree and installed rSkill.

## 5. (Optional) Produce reproducible eval results

If your skill ships `eval/<benchmark>.json`, the canonical producer is a sim
run against a paired scene config:

```bash
openral sim run \
  --config scenes/<your-config>.yaml \
  --rskill rskills/pi05-pick-cube
```

Results validate against `openral_core.SkillEvalResult`. Paper-cited numbers
you haven't reproduced locally are allowed with `reproduced_locally: false`
plus a `reproduction_cli` so others can rerun them.

## 6. Publish to the Hub

Validate first (dry-run is the default — no upload):

```bash
python tools/rskill_publisher.py rskills/pi05-pick-cube/
```

This validates the manifest **and** the README/eval docs and prints a report.
When it's clean, publish:

```bash
export HF_TOKEN=hf_...        # token with repo.write scope
python tools/rskill_publisher.py rskills/pi05-pick-cube/ --publish
# pin the exact upstream weights commit before uploading:
python tools/rskill_publisher.py rskills/pi05-pick-cube/ --publish --bump-revision
```

Two things to know:

- **Repos are created private.** The publisher always creates the Hub repo as
  private; you flip it public on the Hub when you're ready. This is deliberate
  for license-restricted weights.
- **Provenance is not yet signed.** Sigstore signing/verification is the
  *planned* control (ADR-0006) but is **not implemented**. Until it lands,
  `rSkill.from_pretrained` / `from_yaml` emit an `rskill.unverified_provenance`
  warning. Consumers can fail closed with `OPENRAL_REQUIRE_SIGNED_SKILLS=1`.
  `*.pt` weights are treated as untrusted pickle and need
  `OPENRAL_ALLOW_UNSAFE_PICKLE=1` to load — prefer `model.safetensors`. Do not
  describe your skill as "signed" or "verified" until the control exists.

## 7. Install and use it

Anyone (including you, on another host) can now install it like a model:

```bash
openral rskill search pick-cube          # discover it on the OpenRAL Hub org
openral rskill install <owner>/rskill-pi05-pick-cube   # always org-qualified
openral rskill list                      # see it in the local registry
```

`rskill install` needs the full `owner/name` id — a bare name fails fast with an
`OpenRAL/…` suggestion. Use `rskill search [QUERY] [--kind/--role/--embodiment/--license]`
when you don't already know the id.

In a `SimEnvironment` or `RobotEnvironment` YAML, reference it by its
bare rSkill reference in `vla.weights_uri` (see the
[deploy tutorial](../deploy/deploy-run-and-dashboard.md)).

## See also

- [`rskills/template/README.md`](https://github.com/OpenRAL/openral/blob/master/rskills/template/README.md) — the canonical README layout the validator enforces.
- [`docs/reference/vla_compatibility.md`](../../reference/vla_compatibility.md) — VLA × Robot × Sim matrix.
- [Run a deployment and open the dashboard](../deploy/deploy-run-and-dashboard.md).
- [CLAUDE.md §3](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) — the rSkill packaging contract and VLA license matrix.
