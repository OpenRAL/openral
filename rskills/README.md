# `rskills/` — the OpenRAL rSkill catalog

This directory holds the **manifests** for the rSkills OpenRAL ships as
worked examples. Each subdirectory is one rSkill: a `rskill.yaml` manifest
plus `README.md`, a discovery-only `SKILL.md`, and an `eval/` folder. The
subdirectories do **not** contain model weights — the manifest's
`weights_uri` points at a Hugging Face repo under
[`OpenRAL`](https://huggingface.co/OpenRAL), and the weights are pulled on
first load.

> **One rSkill ⇄ one HF repo.** Every entry below maps 1:1 to an
> `OpenRAL/rskill-<name>` repo on the Hub. The in-tree manifest is the
> source of truth; `tools/generate_rskill_skillmd.py` mirrors the
> discovery `SKILL.md` to each HF repo, and the org card counts are
> derived from this directory.

## How an rSkill resolves its weights

| `weights_uri` scheme | Where the weights live | Example |
| --- | --- | --- |
| `hf://OpenRAL/...` | Hugging Face Hub, fetched + cached on first load | every VLA / detector / VLM / reward skill |
| `local://rskills/...` | An ONNX file inside the rSkill dir, **gitignored** and reproduced via a `tools/export_*.py` script (also mirrored to HF) | `rtdetr-coco-r18`, `rtdetr-v2-r50vd` |

The two `rtdetr-*` ONNX skills are the deliberate `local://` exception: the
GStreamer perception path is ONNX-file-based, and `openral deploy sim` uses
`rskills/rtdetr-coco-r18/model.onnx` as its offline detector fallback. Those
binaries are listed in the repo `.gitignore` (`model.onnx`, `model.onnx.data`)
— the clone stays small; the files are regenerated locally on demand.

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
| `molmoact2-libero-nf4` | molmoact2 | franka_panda |
| `molmoact2-so101-nf4` | molmoact2 | so100/so101_follower |
| `openvla-oft-simpler-widowx-nf4` | openvla | widowx |
| `pi05-libero-nf4` | pi05 | franka_panda |
| `pi05-robocasa365-human300-nf4` | pi05 | panda_mobile |
| `rldx1-ft-gr1-nf4` | rldx | gr1 |
| `rldx1-ft-libero-nf4` | rldx | franka_panda |
| `rldx1-ft-rc365-nf4` | rldx | panda_mobile |
| `rldx1-ft-simpler-widowx-nf4` | rldx | widowx |
| `smolvla-libero` | smolvla | franka_panda |
| `smolvla-maniskill-franka` | smolvla | franka_panda |
| `smolvla-metaworld` | smolvla | sawyer |
| `smolvla-robotwin` | smolvla | aloha_agilex |
| `xvla-libero` | xvla | franka_panda |

**Auxiliary skills — run alongside a policy or on deploy scenes,
embodiment-agnostic.**

| rSkill | kind |
| --- | --- |
| `locateanything-3b-nf4` | detector (open-vocab VLM) |
| `omdet-turbo-indoor` / `omdet-turbo-locator` | detector (open-vocab) |
| `rtdetr-coco-r18` / `rtdetr-v2-r50vd` | detector (ONNX, `local://`) |
| `qwen35-4b-nf4` | vlm |
| `robometer-4b` | reward |
| `rskill-moveit-eef-pose` / `rskill-moveit-joints` / `rskill-moveit-look-at` | ros_action (MoveIt) |
| `rskill-nav2-navigate-to-pose` | ros_action (Nav2) |

> The full, verified scene↔rSkill compatibility matrix lives in the team's
> `sim_rskill_matches.xlsx` tracker.

## Add your own rSkill

You don't need to add an entry here to run your own skill — install it from
any HF repo at load time. To scaffold a new local rSkill from
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
