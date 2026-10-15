# `rskills/` — OpenRAL rSkill worked examples

This directory holds the **manifests** for a curated set of rSkills OpenRAL
ships as worked examples and test fixtures (exercised by this repo's tests,
`openral deploy sim` scenes, and the docs). It is **not** the catalog: that is
the [`OpenRAL` org on the Hugging Face Hub](https://huggingface.co/OpenRAL), and
nothing needs to be added here to use a skill published there. Each
subdirectory is one rSkill: a `rskill.yaml` manifest, `README.md`, a
discovery-only `SKILL.md`, and an `eval/` folder. No model weights: the
manifest's `weights_uri` points at the Hub repo, pulled on first load.

> **One rSkill ⇄ one HF repo.** Every entry below maps 1:1 to an
> `OpenRAL/rskill-<name>` repo on the Hub. The in-tree manifest is the
> source of truth; `tools/generate_rskill_skillmd.py` mirrors the
> discovery `SKILL.md` to each HF repo, and the org card counts are
> derived from this directory.

## Find rSkills on the Hub

`openral rskill search` queries the `OpenRAL` Hub org (server-side, filtered on
the `rskill` model-card tag), fetches each hit's manifest concurrently, and
matches your query locally and case-insensitively against the repo id,
manifest name/description, family, kind, role, embodiment tags, and Hub tags:

```bash
openral rskill search libero                                   # any field mentioning "libero"
openral rskill search --embodiment so101_follower               # facet filter, no query
openral rskill search --kind detector --license apache-2.0      # combine facets
openral rskill search --family pi05 --json                      # scriptable output
```

Each row's `local` column tells you what's already on this machine:
`in-tree` (a manifest with that name lives under `rskills/`), `installed`
(present in the local rSkill registry after `openral rskill install`), or
`—` (Hub-only; install it to use it). Install and run it like any other
skill:

```bash
openral rskill install OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16
openral sim run --config scenes/sim/libero_spatial.yaml \
  --rskill OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16
```

`--rskill` accepts a bare in-tree name, an `rskills/<id>` path, or a Hub
repo id interchangeably; a Hub skill needs no local manifest.

## How an rSkill resolves its weights

| `weights_uri` scheme | Where the weights live | Example |
| --- | --- | --- |
| `hf://OpenRAL/...` | Hugging Face Hub, fetched + cached on first load | every VLA / detector / VLM / reward skill |
| `local://rskills/...` | An ONNX file inside the rSkill dir, **gitignored** and reproduced via a `tools/export_*.py` script (also mirrored to HF) | `rtdetr-coco-r18` |

`rtdetr-coco-r18` is the deliberate `local://` exception: the GStreamer
perception path is ONNX-file-based, and `openral deploy sim` uses
`rskills/rtdetr-coco-r18/model.onnx` as its offline detector fallback. The
binary is gitignored (`model.onnx`, `model.onnx.data`) and regenerated locally
on demand. The heavier `rtdetr-v2-r50vd` variant (`runtime: tensorrt`) moved to
the private `openral-pro` repo, which hosts the TensorRT engine runtime it
depends on as an OpenRAL Pro plugin.

## Catalog

**Policies (`kind: vla`) — task skills.** Embodiment must match the scene's
`RobotCapabilities.embodiment_tags`.

| rSkill | family | embodiment |
| --- | --- | --- |
| `act-aloha` / `act-aloha-insertion` | act | aloha |
| `act-libero` | act | franka_panda |
| `diffusion-pusht` | diffusion | pusht |
| `3d-diffuser-actor-rlbench` | diffuser_actor | franka_panda |
| `gr00t-n17-libero` | gr00t | franka_panda |
| `gr00t-n17-b1k-turning-on-radio` | gr00t | r1pro |
| `rskill-internvla_n1-mobile_base-vln-nf4` | internvla_n1 | mobile_base |
| `lingbot-va-galaxea-a1-fruit-placement` | lingbot_va_a1 | galaxea_a1 |
| `lingbot-vla-4b-robotwin` | lingbot_vla | aloha_agilex |
| `lingbot-vla2-robotwin` | lingbot_vla2 | aloha_agilex |
| `molmoact2-libero-nf4` | molmoact2 | franka_panda |
| `molmoact2-so101-nf4` | molmoact2 | so100/so101_follower |
| `openvla-oft-simpler-widowx-nf4` | openvla | widowx |
| `pi05-libero-int8` | pi05 | franka_panda |
| `rldx1-ft-gr1-nf4` | rldx | gr1 |
| `rldx1-ft-libero-nf4` | rldx | franka_panda |
| `rldx1-ft-rc365-nf4` | rldx | panda_mobile |
| `xr1-robocasa` | xr1 | panda_mobile |
| `xr1-robocasa365` | xr1 | panda_mobile |
| `xr1-vlabench` | xr1 | franka_panda |
| `rldx1-ft-simpler-widowx-nf4` | rldx | widowx |
| `smolvla-libero` | smolvla | franka_panda |
| `smolvla-maniskill-franka` | smolvla | franka_panda |
| `smolvla-metaworld` | smolvla | sawyer |
| `smolvla-robotwin` | smolvla | aloha_agilex |
| `rskill-smolvla-so101-eraser_place-bf16` | smolvla | so101_follower |
| `smolvla-vlabench` | smolvla | franka_panda |
| `xvla-libero` | xvla | franka_panda |

**Auxiliary skills — run alongside a policy or on deploy scenes,
embodiment-agnostic.**

| rSkill | kind |
| --- | --- |
| `locateanything-3b-nf4` | detector (open-vocab VLM) |
| `omdet-turbo-indoor` / `omdet-turbo-locator` | detector (open-vocab) |
| `rtdetr-coco-r18` | detector (ONNX, `local://`) |
| `qwen35-4b-nf4` | vlm |
| `robometer-4b` / `topreward-qwen3vl-4b-nf4` | reward |
| `rskill-sam2_1-any-grasped_object_mask-bf16` | segmenter (SAM 2.1) |
| `rskill-moveit-eef-pose` / `rskill-moveit-joints` / `rskill-moveit-look-at` | ros_action (MoveIt) |
| `rskill-nav2-navigate-to-pose` | ros_action (Nav2) |
| `clarify-ambiguity` / `decompose-mission` / `find-object` / `preflight-reach` / `stage-for-manipulation` / `verify-outcome` | playbook (S2 reasoner) |

> The full, verified scene↔rSkill compatibility matrix lives in the team's
> `sim_rskill_matches.xlsx` tracker.

### Not on the Hub yet

Four in-tree manifests have no `OpenRAL/rskill-*` repo on the Hub, so
`openral rskill search` cannot see them and they cannot be installed by repo id
(loading by in-tree name still works wherever their weights resolve). Each needs
only the wrapper repo published; their weights are third-party public
checkpoints, as with `act-aloha` and `smolvla-libero`:

| in-tree skill | weights | to publish |
| --- | --- | --- |
| `lingbot-vla-4b-robotwin` | `robbyant/lingbot-vla-4b-posttrain-robotwin` | wrapper only |
| `lingbot-vla2-robotwin` | `robbyant/lingbot-vla-v2-6b` | wrapper only |
| `rskill-internvla_n1-mobile_base-vln-nf4` | `InternRobotics/InternVLA-N1-DualVLN` | wrapper only |
| `rskill-sam2_1-any-grasped_object_mask-bf16` | `facebook/sam2.1-hiera-small` | wrapper only |

Two OpenRAL-hosted NF4 prequant mirrors are also unpublished
(`lingbot-vla-4b-robotwin-nf4`, `lingbot-vla-v2-6b-nf4`). Both LingBot manifests
point at the fp32 upstreams and pack NF4 at load; republishing the mirrors would
shrink the download and needs only a `weights_uri` repoint.

## Add your own rSkill

You don't need an entry here to run your own skill; install it from any HF
repo at load time. To scaffold a new local rSkill from
[`template/`](template/):

```bash
openral rskill new my-skill --family pi05 --embodiment-tag franka_panda
# or wrap an existing HF checkpoint:
openral rskill new my-skill --from-hf <owner>/<repo>
```

Then edit `rskills/my-skill/rskill.yaml`, `README.md`, and `SKILL.md`
(the publish validator rejects leftover `TEMPLATE_ID` / `TODO:` markers),
and publish with `tools/rskill_publisher.py`. See
[`template/README.md`](template/README.md) for the per-field walkthrough.

Publishing to the Hub and making the repo public is what makes a skill
discoverable: the publisher stamps the `OpenRAL` + `rskill` tags into the
model-card front matter, and `openral rskill search` filters the org on the
`rskill` tag. There is no separate catalog step; adding the skill here is only
worthwhile if it should also serve as an in-tree test fixture or `deploy sim`
example.
