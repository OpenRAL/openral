# Write an rSkill and publish it to the Hugging Face Hub

**Goal:** go from nothing to a published, installable rSkill.

An **rSkill** is OpenRAL's packaging format for a robot skill: a Hugging Face
Hub repo with an `rskill.yaml` manifest, a `README.md`, optional
`eval/<benchmark>.json` results, and (usually) a pointer to weights that live in
an upstream repo. The manifest adds capability checking, license surfacing,
latency budgets and registry integration, so installing a skill is as repeatable
as installing a model.

## Prerequisites

```bash
just bootstrap && just sync   # toolchain + Python workspace (see docs/contributing/toolchain.md)
uv run openral doctor         # confirm the CLI and GPU/runtime
```

Publishing also needs a Hugging Face account and a write token (`HF_TOKEN`).
Skills that use `int8`, `fp8`, NF4 or `nvfp4` weights load only with OpenRAL Pro;
the open path is `fp32`, `fp16` and `bf16` (see
[Quantize an rSkill](quantize-an-rskill.md)).

## 1. Scaffold the package

`openral rskill new` copies the template in
[`rskills/template/`](https://github.com/OpenRAL/openral/blob/master/rskills/template/)
and round-trips it through the schema, so a malformed scaffold fails
immediately. Three modes:

```bash
# (a) Introspect a published checkpoint and pre-fill model_family, chunk_size,
#     sensors_required, state_contract, image_preprocessing.aliases, weights_uri:
uv run openral rskill new pi05-pick-cube --from-hf <owner>/<repo>

# (b) Family-aware defaults without Hub introspection:
uv run openral rskill new pi05-pick-cube --family pi05 --embodiment-tag franka_panda

# (c) Interactive: prompts for owner / license / embodiment / family:
uv run openral rskill new pi05-pick-cube
```

- `--family` is one of `act | smolvla | pi05 | xvla | diffusion`.
- `--license` is any `RSkillLicensePosture` value: `apache-2.0 | mit | bsd |
  permissive_research | nvidia_non_commercial | nvidia_open_model |
  rlwrld_non_commercial | cc-by-nc-sa-4.0 | flux_non_commercial | apple-amlr |
  lfm1.0 | openmdw-1.1 | proprietary | unknown`.
- `--embodiment-tag` must be a tag some `robots/*/robot.yaml` declares, or `any`
  / `custom` / `multi` (for example `so100_follower`, `franka_panda`, `aloha`).

You should see `rskills/pi05-pick-cube/{rskill.yaml,README.md,eval/}`.

## 2. Fill in the manifest

Open `rskills/<id>/rskill.yaml`. The fields that matter most to consumers (see
the in-tree
[`rskills/smolvla-libero/rskill.yaml`](https://github.com/OpenRAL/openral/blob/master/rskills/smolvla-libero/rskill.yaml)
for a bf16 example):

| Field | What it does |
| --- | --- |
| `name` | `<owner>/rskill-<model>-<robot>-<task>-<quant>`, the Hub repo id (enforced; see [Naming convention](#naming-convention)). |
| `license` | License **posture**; drives `is_commercial_use_allowed`. |
| `kind` | One of `vla`, `wam`, `ros_action`, `ros_service`, `detector`, `segmenter`, `depth_estimator`, `vlm`, `reward`, `playbook`. |
| `role` | Optional. The slot the `kind` runs in (`RSKILL_ROLE_BY_KIND`), filled in when omitted: `s1` for `vla` / `wam` / `ros_action` / `ros_service` / `detector` / `segmenter` / `depth_estimator`, `s2` for `vlm` / `reward` / `playbook`. A declared value that disagrees fails validation under `schema_version: "0.2"` (a `0.1` file is migrated with a warning); `s0` is never an rSkill. |
| `embodiment_tags` | Must match a robot's `RobotCapabilities.embodiment_tags`. |
| `sensors_required` | Modality, `vla_feature_key` and minimum resolution per camera. |
| `actuators_required` | Each entry needs `control_mode_semantics`; see [Manifest details](#manifest-details). |
| `runtime` / `quantization` | `pytorch` / `onnx` / `tensorrt`; `dtype` plus `min_vram_gb`. |
| `weights_uri` | `hf://<owner>/<repo>`. The rSkill does **not** copy weights. |
| `chunk_size` / `n_action_steps` | Action-chunk size and replan cadence. |
| `latency_budget.per_chunk_ms` | Contractual; enforced by sim-tier latency tests. |
| `actions` / `objects` / `scenes` | Vocabulary the reasoner's LLM palette uses to pick the skill. |
| `image_preprocessing.aliases` | Keys are VLA **slots** (the suffix of a `vla_feature_key`, e.g. `observation.images.camera1` is `camera1`), never robot sensor or scene camera names; values are the checkpoint's view names (`camera1: image`). `sim run`, `deploy sim` and `deploy run` all feed the policy slot-keyed images. A key no `sensors_required` entry declares fails the load. |

The full schema is
[`openral_core.schemas.RSkillManifest`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py).
A simple ACT or Diffusion policy can drop the pi0.5-shaped knobs
(`image_preprocessing`, `state_contract`, `processors`).

### Naming convention

The `name` is **enforced** at publish time. Hyphens are only segment
separators; every token uses underscores internally, so a name parses by a plain
`split("-")`:

```text
<owner>/rskill-<model>-<robot>-<task>-<quantization>   # weight-bearing kinds
<owner>/rskill-<model>-<robot>-<task>                  # ros_action / ros_service
<owner>/rskill-playbook-<name>                         # kind: playbook
```

| Axis | Vocabulary | Example |
| --- | --- | --- |
| `<model>` | `CANONICAL_MODEL_TOKENS`, a versioned checkpoint token | `smolvla`, `gr00t_n17`, `lingbot_vla2`, `omdet_turbo` |
| `<robot>` | a robot-declared embodiment tag, or `any` / `multi` (more than one concrete robot) | `franka_panda`, `aloha_agilex`, `any`, `multi` |
| `<task>` | author-chosen, shape `^[a-z0-9][a-z0-9_]*$` only | `libero_spatial`, `pen`, `pick_place_pen`, `locator` |
| `<quantization>` | `{fp32, fp16, bf16, int8, nf4}` (schema `int4` becomes `nf4`); omitted for weightless ROS wrappers | `bf16`, `nf4` |

For a VLA the `<model>` token must match `model_family` (a `smolvla` checkpoint
cannot be named `pi05`). A ROS-wrapper skill has no weights and drops the quant
segment, for example `OpenRAL/rskill-moveit-multi-eef_pose`. A full example:
`OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16`.

On a mismatch the publisher prints a suggested name (`expected_repo_name`). To
rewrite an existing manifest in one step, then refine the `<task>` slug:

```bash
uv run python tools/rskill_publisher.py rskills/<id>/ --fix-name
```

## 3. Write the README

The scaffolded `README.md` is full of `<TODO>` and `TEMPLATE_ID` markers. They are
**publish-blocking**: the documentation validator
(`openral_cli._rskill_doc_validator.validate_rskill_docs`) refuses to upload a
README that still contains them. Fill in every section: what the skill does, the
observation-to-action contract, how it was trained, supported robots, sensors
required and the license posture (keep `is_commercial_use_allowed` consistent
with `rskill.yaml::license`). Follow
[`rskills/template/README.md`](https://github.com/OpenRAL/openral/blob/master/rskills/template/README.md).

## 4. Check it runs on your robot

```bash
uv run openral detect                           # auto-provisions ./robot.yaml
uv run openral rskill check pi05-pick-cube --robot robot.yaml
```

You should see a per-section breakdown (embodiment, capability flags, GPU
runtime, GPU dtype, sensors, actuators) and whether the skill will run on this
host. With no id it walks every in-tree and installed rSkill. Options: `--json`
and `--rskills-dir`.

## 5. (Optional) Produce reproducible eval results

`eval/<benchmark>.json` is written by the **benchmark** tier. `openral sim run` is
an ad-hoc rollout that never touches your rSkill package; use it to check the
pairing, then produce the number with one of:

```bash
# Multi-scene suite (the canonical producer): writes rskills/<dir>/eval/<suite_id>.json
uv run openral benchmark run --suite libero_spatial --rskill rskills/pi05-pick-cube

# Single paper-protocol scene: writes rskills/<dir>/eval/scene_<scene_id>.json
uv run openral benchmark scene --config scenes/benchmark/<your-scene>.yaml \
  --rskill rskills/pi05-pick-cube
```

Both also write `avg_success_rate` into the manifest at `benchmarks.<id>` (a
comment-preserving edit). `--no-update-manifest` skips that; `--no-write-eval`
(`benchmark scene` only) writes nothing. Results validate against
[`openral_core.RSkillEvalResult`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py).
Roll them up with `uv run openral benchmark report [--json]`.
[Run a benchmark](../benchmark/run-a-benchmark.md) covers the suite invariants,
the `evaluated_tasks` gate, and what `reproduced_locally: true` / `false` plus
`reproduction_cli` commit you to.

## 6. Publish to the Hub

Validate first. Dry-run is the default and uploads nothing:

```bash
uv run python tools/rskill_publisher.py rskills/pi05-pick-cube/
```

It validates the manifest **and** the README/eval docs and prints a report. When
it is clean, publish:

```bash
export HF_TOKEN=hf_...        # token with repo.write scope
uv run python tools/rskill_publisher.py rskills/pi05-pick-cube/ --publish --public
# pin the exact upstream weights commit before uploading:
uv run python tools/rskill_publisher.py rskills/pi05-pick-cube/ --publish --public --bump-revision
```

You should see the repo created on the Hub with the `OpenRAL` and `rskill` tags
stamped into the model card.

- **Repos are private unless you pass `--public`.** A private repo is invisible
  to `rskill search` and to everyone else, so step 7 fails for them. Visibility
  has no license gate; the manifest's license still binds: a commercial
  deployment (`OPENRAL_COMMERCIAL_DEPLOYMENT=1`) refuses a non-commercial skill
  unless `OPENRAL_ALLOW_NONCOMMERCIAL=1`. `--gated` (exclusive with `--public`)
  publishes a premium rSkill: a public repo whose files sit behind Hugging Face
  manual access approval, set and verified before any upload. Other flags:
  `--fix-name`, `--token`.
- **Provenance is not signed yet.** Sigstore signing is planned but not
  implemented, so `rSkill.from_pretrained` / `from_yaml` emit an
  `rskill.unverified_provenance` warning. Consumers can fail closed with
  `OPENRAL_REQUIRE_SIGNED_SKILLS=1`. `*.pt` weights are untrusted pickle and need
  `OPENRAL_ALLOW_UNSAFE_PICKLE=1`; prefer `model.safetensors`. Do not describe a
  skill as "signed" or "verified". See
  [CLAUDE.md §3](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md).
- **Discovery needs the tag and a public repo.** `openral rskill search` lists
  the `OpenRAL` org filtered on the `rskill` tag.

## 7. Install and use it

Anyone, on any host, can now install it like a model:

```bash
uv run openral rskill search pick_cube     # discover it on the OpenRAL Hub org
uv run openral rskill install <owner>/rskill-pi05-franka_panda-pick_cube-bf16
uv run openral rskill list                 # see it in the local registry
```

Two identifiers: `pi05-pick-cube` is only the **local directory** under
`rskills/` you chose in step 1. You install the **Hub repo id**, the manifest's
canonical `name`. For this example that is
`rskill-pi05-franka_panda-pick_cube-bf16`, not `rskill-pi05-pick-cube` (which
parses as a ROS-wrapper name and drops the embodiment).

`rskill install` needs the full `owner/name`; a bare name fails with an
`OpenRAL/...` suggestion. `rskill search [QUERY]` takes
`--kind/--role/--embodiment/--license/--family/--limit/--json`. `QUERY` is matched
locally, case-insensitively, against the repo id, manifest name and description,
family, kind, role, embodiment tags and Hub tags (every token must appear), so
`rskill search pick cube` and `rskill search pi05-pick-cube` both work.

Run it by passing the bare reference to `--rskill`. The policy always travels on
the CLI: scene YAMLs reject a `vla:` block (`_reject_legacy_vla_block` raises
`ROSConfigError`). `uv run openral deploy run` also installs the rSkills on a
robot's openral.com list automatically; see the
[deploy tutorial](../deploy/deploy-run-and-dashboard.md).

## Manifest details

**Actuators.** Each `actuators_required` entry needs `control_mode_semantics`
(for example `mode: absolute`). A skill that sends gripper commands (a
`gripper_*` action slot, or a `representation` ending in a gripper) also needs a
`gripper_position` entry whose `gripper_convention` names the encoding it emits:
`normalized_close_symmetric` for robosuite/LIBERO data, `normalized_open_unit`
for 0 = closed and 1 = open. The runner refuses the skill on a robot or scene
that consumes a different encoding.

**Cartesian scale.** For a normalized Cartesian controller, declare
`action_contract.cartesian_delta_scale` as the per-axis physical range.
Predictive safety applies `clip(raw, -1, 1) * scale`; the raw command goes to the
native controller. RoboSuite `OSC_POSE` uses `[0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`
(metres, then radians). Omit it when the policy already emits physical deltas.

**Control rate.** Declare `action_contract.control_freq_hz` when the checkpoint
was trained at a known rate (DROID 15 Hz, ALOHA 50 Hz). The runner executes one
chunk row per tick at the robot's `action_spec.control_freq_hz` and has no
resampler, so it rejects the goal when the two differ. Omit it only if the rate is
unknown.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Publisher rejects the name | Run it with `--fix-name`, then refine the `<task>` slug. |
| Publisher rejects the README | Remove every `<TODO>`, `TEMPLATE_ID` and `TEMPLATE_ORG` marker. |
| `rskill new` rejects `--embodiment-tag` | Use a tag some `robots/*/robot.yaml` declares, or `any` / `custom` / `multi`. |
| `rskill search` does not list your repo | The repo is private or lacks the `rskill` tag; republish with `--public`. |
| `rskill install` says the id is not found | Use the full `owner/name` from the manifest, not the local directory name. |
| Runner refuses the skill on a robot | Check `rskill check`; likely a `gripper_convention` or `control_freq_hz` mismatch. |

## Next

- [Quantize an rSkill](quantize-an-rskill.md)
- [Run a benchmark](../benchmark/run-a-benchmark.md)
- [Run a deployment and open the dashboard](../deploy/deploy-run-and-dashboard.md)
- [VLA compatibility](../../reference/vla_compatibility.md): the VLA × robot × sim matrix
- [rSkills reference](../../reference/rskills.md)
