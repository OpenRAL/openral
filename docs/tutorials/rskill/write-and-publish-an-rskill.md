# Write an rSkill and publish it to the Hugging Face Hub

An **rSkill** is OpenRAL's packaging format for a robot skill: a Hugging Face
Hub repo containing an `rskill.yaml` manifest, a `README.md`, optional
`eval/<benchmark>.json` results, and (usually) a pointer to model weights that
live in an upstream repo. On top of raw weights, the manifest adds capability
checking, license surfacing, latency budgets, and local-registry integration,
so installing a skill is as repeatable as installing a model.

This tutorial goes from nothing to a published, installable rSkill.

## Prerequisites

```bash
just bootstrap && just sync   # toolchain + Python workspace (always `just sync`,
                              # never bare `uv sync` — see docs/contributing/toolchain.md)
openral doctor                # confirm the CLI and GPU/runtime
```

Publishing also needs a Hugging Face account and a write token (`HF_TOKEN`).

## 1. Scaffold the package

`openral rskill new` copies the canonical template in
[`rskills/template/`](https://github.com/OpenRAL/openral/blob/master/rskills/template/)
and round-trips it through the schema, so a malformed scaffold fails
immediately rather than on first load. Three modes:

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
`--license` is any `RSkillLicensePosture` value: `apache-2.0 | mit | bsd |
permissive_research | nvidia_non_commercial | nvidia_open_model |
rlwrld_non_commercial | cc-by-nc-sa-4.0 | proprietary | unknown`. `--embodiment-tag` must be a
tag some `robots/*/robot.yaml` declares, or `any` / `custom` / `multi`
(e.g. `so100_follower`, `franka_panda`, `aloha`). This writes `rskills/pi05-pick-cube/{rskill.yaml,README.md,eval/}`.

## 2. Fill in the manifest

Open `rskills/<id>/rskill.yaml`. The fields that matter most to consumers
(modeled on the in-tree
[`rskills/pi05-libero-int8/rskill.yaml`](https://github.com/OpenRAL/openral/blob/master/rskills/pi05-libero-int8/rskill.yaml)):

| Field | What it does |
| --- | --- |
| `name` | `<owner>/rskill-<model>-<robot>-<task>-<quant>` — the Hub repo id (enforced; see [Naming convention](#naming-convention)). |
| `license` | License **posture**; drives `commercial_use_allowed`. |
| `role` | `s1` (fast policy), `s2` (reasoner), or `s0` (cerebellar). |
| `kind` | `vla` for a learnable policy; detector kinds also exist. |
| `embodiment_tags` | Must match a robot's `RobotCapabilities.embodiment_tags`. |
| `sensors_required` | Modality + `vla_feature_key` + min resolution per camera. |
| `actuators_required` | Each entry needs `control_mode_semantics` (e.g. `mode: absolute`). A skill that sends gripper commands — a `gripper_*` action slot, or a `representation` ending in a gripper — also needs a `gripper_position` entry whose `gripper_convention` names the encoding it emits (e.g. `normalized_close_symmetric` for robosuite/LIBERO data, `normalized_open_unit` for 0 = closed / 1 = open). The runner refuses the skill on a robot or scene that consumes a different encoding. |
| `runtime` / `quantization` | `pytorch` / `onnx` / `tensorrt`; `dtype` + `min_vram_gb`. See [Quantize an rSkill](quantize-an-rskill.md). |
| `weights_uri` | `hf://<owner>/<repo>` — the rSkill does **not** copy weights. |
| `chunk_size` / `n_action_steps` | Action-chunk size and replan cadence. |
| `latency_budget.per_chunk_ms` | Contractual — enforced by sim-tier latency tests. |
| `actions` / `objects` / `scenes` | Vocabulary the reasoner's LLM palette uses to pick the skill. |
| `sensors_required` / `image_preprocessing.aliases` | Each RGB entry's `vla_feature_key` suffix (`observation.images.camera1` → `camera1`) is a **VLA slot**. `aliases` keys are slots — never robot sensor names (`top`) or scene camera names — mapped to the checkpoint's view names (`camera1: image`). `openral sim run`, `deploy sim` and `deploy run` all feed the policy slot-keyed images, so the same alias means the same camera everywhere; a key no `sensors_required` entry declares fails the load. |

The full schema is
[`openral_core.schemas.RSkillManifest`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py).
A simpler ACT/Diffusion policy can drop the pi0.5-shaped knobs
(`image_preprocessing`, `state_contract`, `processors`).

For a normalized Cartesian controller, declare
`action_contract.cartesian_delta_scale` as its per-axis physical range.
Predictive safety applies `clip(raw, -1, 1) * scale`; the unchanged raw
command goes to the native controller. RoboSuite `OSC_POSE` uses
`[0.05, 0.05, 0.05, 0.5, 0.5, 0.5]` (metres, then radians). Omit the field
when the policy already emits physical deltas.

Declare `action_contract.control_freq_hz` when the checkpoint was trained at a
known rate (DROID 15 Hz, ALOHA 50 Hz, …). The skill runner executes one chunk
row per tick at the robot's `action_spec.control_freq_hz` and has no resampler,
so it rejects the goal when the two differ rather than run the policy at the
wrong speed. Omit it only if the rate is unknown.

### Naming convention

The `name` is **enforced** at publish time. Hyphens are ONLY the segment
separators; every token uses underscores internally, so a name parses by a
plain `split("-")`. Two shapes:

```
<owner>/rskill-<model>-<robot>-<task>-<quantization>   # weight-bearing kinds
<owner>/rskill-<model>-<robot>-<task>                  # ros_action / ros_service
<owner>/rskill-playbook-<name>                          # kind: playbook
```

| Axis | Vocabulary | Example |
| --- | --- | --- |
| `<model>` | `CANONICAL_MODEL_TOKENS` — a versioned checkpoint token | `smolvla`, `gr00t_n17`, `lingbot_vla2`, `omdet_turbo` |
| `<robot>` | a robot-declared embodiment tag, or `any` / `multi` (>1 concrete robot); shape-checked by the publisher, vocabulary-checked in CI | `franka_panda`, `aloha_agilex`, `any`, `multi` |
| `<task>` | **author-chosen**, validated by shape `^[a-z0-9][a-z0-9_]*$` only | `libero_spatial`, `pen`, `pick_place_pen`, `locator` |
| `<quantization>` | `{fp32, fp16, bf16, int8, nf4}` (schema `int4` → `nf4`); **omitted** for weightless ROS wrappers | `bf16`, `int8`, `nf4` |

For a VLA the `<model>` token must match `model_family` (a `smolvla`
checkpoint can't be named `pi05`). A ROS-wrapper skill (`ros_action` /
`ros_service`) has no weights and drops the quant segment, e.g.
`OpenRAL/rskill-moveit-multi-eef_pose`.

`<task>` is validated by shape, not against `evaluated_tasks`, so two skills on
one benchmark can carry distinct slugs (`pen` vs `pick_place_pen`). The other
segments must use canonical tokens. Example:
`OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16`.

Validation is `openral_core.schemas.repo_name_is_canonical`; on a mismatch the
publisher prints a suggested name (from `expected_repo_name`). To rewrite an
existing manifest in one step (then refine the suggested `<task>` slug if you
like):

```bash
uv run python tools/rskill_publisher.py rskills/<id>/ --fix-name
```

**Weight-mirror repos** (an HF repo holding only weights, with no `rskill.yaml`)
are out of publisher scope. Recommended non-enforced shape:
`<model>-<benchmark>-<quantization>` (e.g. `lingbot_vla-robotwin-nf4`).

## 3. Write the README

The scaffolded `README.md` is full of `<TODO>` / `TEMPLATE_ID` markers. They
are **publish-blocking**: the documentation validator
(`openral_cli._rskill_doc_validator.validate_rskill_docs`) refuses to upload a
README that still contains them. Fill in every section: what the skill does,
the observation→action contract, how it was trained, supported robots, sensors
required, and the license posture (keep `commercial_use_allowed` consistent
with `rskill.yaml::license`).

## 4. Check it runs on your robot

```bash
openral detect                           # auto-provisions ./robot.yaml
openral rskill check pi05-pick-cube --robot robot.yaml
```

`rskill check` prints a per-section breakdown (embodiment, capability flags,
GPU runtime, GPU dtype, sensors, actuators) and whether the skill will run on
this host. With no id it walks every in-tree and installed rSkill.

## 5. (Optional) Produce reproducible eval results

`eval/<benchmark>.json` is written by the **benchmark** tier. `openral sim run`
is an ad-hoc rollout that never touches your rSkill package; use it to check
the pairing, then produce the number with one of:

```bash
# Multi-scene suite (the canonical producer) — writes
# rskills/<dir>/eval/<suite_id>.json.
openral benchmark run --suite libero_spatial --rskill rskills/pi05-pick-cube

# Single paper-protocol scene — writes
# rskills/<dir>/eval/scene_<scene_id>.json.
openral benchmark scene --config scenes/benchmark/<your-scene>.yaml \
  --rskill rskills/pi05-pick-cube
```

Both also write the resulting `avg_success_rate` into your manifest at
`benchmarks.<id>` (a surgical, comment-preserving edit); `--no-update-manifest`
skips that, and `--no-write-eval` (`benchmark scene` only) writes nothing.
Results validate against
[`openral_core.RSkillEvalResult`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py)
(search for `class RSkillEvalResult`). Roll them up with
`openral benchmark report [--json]`.

[Run a benchmark](../benchmark/run-a-benchmark.md) covers the opt-outs, suite
invariants, the `evaluated_tasks` gate, `trace_id`, and what
`reproduced_locally: true` / `false` + `reproduction_cli` commit you to.

## 6. Publish to the Hub

Validate first (dry-run is the default; no upload):

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

- **Repos are created private** unless you pass `--public`, which the
  publisher refuses for non-commercial weights. Otherwise flip it public on
  the Hub when ready.
- **Provenance is not yet signed.** Sigstore signing/verification is the
  *planned* control but is **not implemented**. Until it lands,
  `rSkill.from_pretrained` / `from_yaml` emit an `rskill.unverified_provenance`
  warning. Consumers can fail closed with `OPENRAL_REQUIRE_SIGNED_SKILLS=1`.
  `*.pt` weights are treated as untrusted pickle and need
  `OPENRAL_ALLOW_UNSAFE_PICKLE=1` to load; prefer `model.safetensors`. Do not
  describe your skill as "signed" or "verified" until the control exists.
- **Publishing is what makes it discoverable.** The publisher stamps the
  `OpenRAL` + `rskill` tags into the model card front matter; `openral rskill
  search` lists the `OpenRAL` org filtered on the `rskill` tag, so a repo
  shows up only once it carries that tag *and* is public.

## 7. Install and use it

Anyone, on any host, can now install it like a model:

```bash
openral rskill search pick_cube          # discover it on the OpenRAL Hub org
openral rskill install <owner>/rskill-pi05-franka_panda-pick_cube-bf16
openral rskill list                      # see it in the local registry
```

Two identifiers: `pi05-pick-cube` is only the **local directory** under
`rskills/` you chose in §1. You install the **Hub repo id**, the manifest's
canonical `name` ([Naming convention](#naming-convention)). For the running
example that is `rskill-pi05-franka_panda-pick_cube-bf16`, not
`rskill-pi05-pick-cube`, which parses as a four-segment ROS-wrapper name and
drops the embodiment.

`rskill install` needs the full `owner/name` id; a bare name fails fast with an
`OpenRAL/…` suggestion. To find an id, use
`rskill search [QUERY] [--kind/--role/--embodiment/--license/--family/--limit/--json]`.
`QUERY` is matched locally and
case-insensitively (every whitespace-separated token must appear somewhere
in the repo id, manifest name/description, family, kind, role, embodiment
tags, or Hub tags), so `rskill search pick cube` and `rskill search
pi05-pick-cube` both work.

Run it by passing the bare rSkill reference to `--rskill`. The policy always
travels on the CLI: scene YAMLs reject a `vla:` block
(`_reject_legacy_vla_block` raises `ROSConfigError`; see the
[deploy tutorial](../deploy/deploy-run-and-dashboard.md)).

## See also

- [`rskills/template/README.md`](https://github.com/OpenRAL/openral/blob/master/rskills/template/README.md) — the canonical README layout the validator enforces.
- [`docs/reference/vla_compatibility.md`](../../reference/vla_compatibility.md) — VLA × Robot × Sim matrix.
- [Run a deployment and open the dashboard](../deploy/deploy-run-and-dashboard.md).
- [CLAUDE.md §3](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) — the rSkill packaging contract and VLA license matrix.
