# `rskills/` — OpenRAL rSkill worked examples

This directory holds the **manifests** for a curated set of rSkills OpenRAL
ships as worked examples and test fixtures (exercised by this repo's tests,
`openral deploy sim` scenes, and the docs). It is **not** the catalog: that is
the [`OpenRAL` org on the Hugging Face Hub](https://huggingface.co/OpenRAL), and
nothing needs to be added here to use a skill published there. Each
subdirectory is one rSkill: a `rskill.yaml` manifest, `README.md`, a
discovery-only `SKILL.md`, and an `eval/` folder. No model weights: the
manifest's `weights_uri` points at the Hub repo, pulled on first load.

> **One rSkill is one HF repo.** A published skill maps 1:1 to an
> `OpenRAL/rskill-<name>` repo on the Hub. The in-tree manifest is the source of
> truth, and `tools/generate_rskill_skillmd.py` mirrors the discovery `SKILL.md`
> to each HF repo.

## Find rSkills on the Hub

`openral rskill search` queries the `OpenRAL` Hub org (server-side, filtered on
the `rskill` model-card tag), fetches each hit's manifest concurrently, and
matches your query locally and case-insensitively against the repo id,
manifest name/description, family, kind, role, embodiment tags, and Hub tags:

```bash
uv run openral rskill search libero                                   # any field mentioning "libero"
uv run openral rskill search --embodiment so101_follower               # facet filter, no query
uv run openral rskill search --kind detector --license apache-2.0      # combine facets
uv run openral rskill search --family pi05 --json                      # scriptable output
```

Each row's `local` column tells you what's already on this machine:
`in-tree` (a manifest with that name lives under `rskills/`), `installed`
(present in the local rSkill registry after `openral rskill install`), or
`—` (Hub-only; install it to use it). Install and run it like any other
skill:

```bash
uv run openral rskill install OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16
uv run openral sim run --config scenes/sim/libero_spatial.yaml \
  --rskill OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16
```

`--rskill` accepts a bare in-tree name, an `rskills/<id>` path, or a Hub
repo id interchangeably; a Hub skill needs no local manifest.

## How an rSkill resolves its weights

| `weights_uri` scheme | Where the weights live | Example |
| --- | --- | --- |
| `hf://<owner>/<repo>[@<revision>]` | Hugging Face Hub, fetched and cached on first load. The owner is the upstream publisher (`lerobot`, `robbyant`, `makermods`, ...), not always `OpenRAL`. | most VLA, VLM and reward skills |
| `local://rskills/...` | An ONNX file inside the rSkill directory, **gitignored** and rebuilt by a `tools/export_*.py` script | `rtdetr-coco-r18`, `rtdetr-v2-r50vd` |

The two RT-DETR detectors are the deliberate `local://` exceptions: the GStreamer
perception path is ONNX-file-based, and `openral deploy sim` uses
`rskills/rtdetr-coco-r18/model.onnx` as its offline detector fallback. With
OpenRAL Pro installed, the NVMM aggregator tier builds a TensorRT engine from the
same `model.onnx`.

## Catalogue

The in-tree skills, their kind, family, embodiment and quantization are listed in
the [rSkills reference](../docs/reference/rskills.md). `ls rskills/` is the
authority for what exists; each subdirectory's `rskill.yaml` is its source of
truth. A manifest here does not imply a published Hub repo: check with
`openral rskill search`.

## Add your own rSkill

You don't need an entry here to run your own skill; install it from any HF
repo at load time. To scaffold a new local rSkill from
[`template/`](template/):

```bash
uv run openral rskill new my-skill --family pi05 --embodiment-tag franka_panda
# or wrap an existing HF checkpoint:
uv run openral rskill new my-skill --from-hf <owner>/<repo>
```

Then edit `rskills/my-skill/rskill.yaml` and `README.md` (the publish
validator rejects leftover `TEMPLATE_ORG` / `TODO:` markers), regenerate the
discovery `SKILL.md` with `python tools/generate_rskill_skillmd.py my-skill`
(never hand-edit it), and publish with `tools/rskill_publisher.py`. See
[`template/README.md`](template/README.md) for the per-field walkthrough.

Publishing to the Hub and making the repo public is what makes a skill
discoverable: the publisher stamps the `OpenRAL` + `rskill` tags into the
model-card front matter, and `openral rskill search` filters the org on the
`rskill` tag. There is no separate catalog step; adding the skill here is only
worthwhile if it should also serve as an in-tree test fixture or `deploy sim`
example.
