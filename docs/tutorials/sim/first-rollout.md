# Your first sim rollout

**Goal:** run a trained policy in a simulated robot with one command, on a laptop with no GPU.

The pairing is [Diffusion Policy](https://arxiv.org/abs/2303.04137) on PushT, the lightest one. gym-pusht is 2-D `pymunk` with no MuJoCo render context, the checkpoint runs on CPU, and both ship in the `sim` dependency group. For the GPU flagship pairing, jump to [Move to a MuJoCo scene](#move-to-a-mujoco-scene).

## Prerequisites

```bash
git clone https://github.com/OpenRAL/openral && cd openral
just bootstrap                # uv + ROS 2 + system deps
just sync --group sim         # workspace + the sim group
uv run openral doctor
```

Never run bare `uv sync`: it trips on the `hf-libero` metadata and leaves the venv half-built. Rules: [Managing the Python environment & dependency groups](../../contributing/toolchain.md#managing-the-python-environment-dependency-groups). If you installed with the one-liner instead of a clone, use `openral install sim` in place of `just sync --group sim`.

A GPU row reading `absent` in the doctor output is fine here. Row meanings: [quickstart](../../quickstart/index.md).

---

## 1. Pick the two halves

Every `openral sim run` composes a **scene** (`--config`, a `SimScene` YAML carrying robot × scene × task) and an **rSkill** (`--rskill`, the policy). Both are required.

```bash
uv run openral sim list        # scene configs, paste into --config
uv run openral rskill list     # rSkills, in-tree and installed, paste into --rskill
```

Neither builds a sim or touches the GPU. This page uses:

| Half | Value | What it is |
| --- | --- | --- |
| `--config` | `scenes/sim/pusht.yaml` | The PushT scene and the `pusht/0` task, 300 steps |
| `--rskill` | `rskills/diffusion-pusht` | Diffusion Policy, about 263 M parameters, fp32 |

The scene fixes its robot to `pusht_2d` (a synthetic 2-DoF pusher), so there is no `--robot` to pass.

## 2. Dry-run it

`--dry-run` resolves the config and the rSkill and runs the embodiment and sensor compatibility gate a real rollout hits, without building the sim or downloading weights:

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht --dry-run
```

You should see the resolved header (the same block as in step 3), then:

```
  max_ticks: 301
  dry-run: resolved config + rSkill, compatibility checked; sim not built
```

and exit `0`. An incompatible pairing prints `compat error: ...` and exits `1` before any download:

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/smolvla-libero --dry-run     # exits 1: embodiment mismatch
```

## 3. Run it

On CPU, cap the episode. The scene's default is 300 steps, and Diffusion Policy runs 100 denoising steps per chunk, so a full episode takes a long time without a GPU:

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht --max-steps 16
```

With a GPU, add `--device cuda:0` and drop `--max-steps`. The first run downloads the checkpoint (about 1 GB) into your HF cache. You should see a header, one line per episode, then the success rate:

```
============================================================
  openral sim run
  robot : pusht_2d
  scene : pusht  [mujoco]
  task  : pusht/0
  vla   : diffusion  (rskills/diffusion-pusht)
  seed  : 0  episodes=1
============================================================

  ep0: success=False steps=16 reward=4.474 mean_lat=6684.2ms budget_viol=2

  success_rate: 0/1 = 0%
```

Real CPU timings, so yours will differ. A run with tracing active also prints a `trace_id:` line, and one with `--save-dir` prints `wrote .../summary.json`.

- **`vla`** is the policy adapter id, taken from the manifest's `model_family`. Several rSkills can share one adapter.
- **`[mujoco]`** is the scene's declared `PhysicsBackend`. For PushT it is cosmetic: gym-pusht is `pymunk`.
- **`mean_lat`** is mean per-step latency. The manifest records 1756 ms for a warm full chunk on its reference GPU; the figure above is the same work on CPU.
- **`budget_viol`** counts steps over the manifest's `latency_budget.per_chunk_ms`. Some are normal because the first chunk of an episode pays warm-up. It is a signal, not a failure.
- The exit code does not depend on success. A failed episode still exits `0`; you set the threshold.

---

## 4. Get something back from it

### A JSON summary

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht --max-steps 16 --save-dir out/
```

`out/summary.json` holds the resolved `config`, an `episodes` array (`success`, `steps`, `total_reward`, `mean_step_latency_ms`, `max_step_latency_ms`, `contact_stats`) and the aggregate `success_rate`. `contact_stats` is filled by Isaac scenes only. Assert on this file in CI.

It is a rollout summary, not an eval result, and it never writes into the rSkill package. `eval/<benchmark>.json` comes from the benchmark tier: [Run a benchmark](../benchmark/run-a-benchmark.md).

### A video

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht --max-steps 16 --save-video example_videos
```

The default `--video-style debug` stacks two bands per episode: an image band on top and an `observation state` line plot below. The top band is the policy's stitched camera grid when it has several cameras; on PushT the rollout and policy views are the same picture and it is labelled `rollout / policy view`. `--video-style world` writes a clean single-view render.

### A live dashboard

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht --max-steps 16 --dashboard
```

This boots `openral dashboard` as a child process, points OpenTelemetry at it and stops it on exit. Open <http://localhost:4318> for live spans, latency histograms and camera thumbnails ([dashboard quickstart](../../quickstart/dashboard.md)).

### A live viewer

MuJoCo-backed scenes open a passive viewer window when a display is available. PushT is `pymunk`, so it has none. On a MuJoCo scene use `--view`, and do not set `MUJOCO_GL=egl`: the two are incompatible.

---

## Next

### Sweep the same pairing

`--task`, `--instruction`, `--max-steps`, `--n-episodes`, `--seed` and `--device` overlay the YAML:

```bash
for s in 0 1 2; do
  uv run openral sim run --config scenes/sim/pusht.yaml \
    --rskill rskills/diffusion-pusht --seed "$s" --n-episodes 5
done
```

### Move to a MuJoCo scene

LIBERO with SmolVLA is the flagship pairing: a 7-DoF Franka Panda, real MuJoCo physics and a language-conditioned VLA. It needs a GPU, the `libero` group and an offscreen GL backend. Use the recipe; it adds two workarounds (a duplicate `hf-libero` egg-info and LIBERO's stale absolute-path config):

```bash
just sync --group libero
just sim-libero
```

The recipe runs `MUJOCO_GL=egl uv run --all-packages --group libero openral sim run --config scenes/sim/libero_spatial.yaml --rskill rskills/smolvla-libero --save-video example_videos`. Keep `--all-packages`: without it `uv run --group libero` evicts the `openral` script.

LIBERO and RoboCasa cannot share a venv because they pin conflicting robosuite versions. Swap with `just sync --group libero` or `just sync --group robocasa` per task.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `compat error: ...`, exit 1 | The rSkill's embodiment or sensors do not match the scene's robot. Pick another rSkill or scene. |
| `--view` errors out | Unset `MUJOCO_GL=egl`, or drop `--view` |
| First run is slow | It is downloading the checkpoint; later runs read the HF cache |
| Run takes very long | CPU inference. Add `--max-steps 16`, or use `--device cuda:0` |
| `Failed to spawn: openral` | A `uv run --group` without `--all-packages`; use the `just` recipe |

## Further reading

- [Create a sim environment](create-a-sim-environment.md): write your own scene, robot manifest or adapter.
- [Write & publish an rSkill](../rskill/write-and-publish-an-rskill.md): package your own policy for `--rskill`.
- [Sim environments reference](../../reference/sim-environments.md): every scene across the three tiers.
- [VLA × Robot × Sim compatibility](../../reference/vla_compatibility.md): which policy runs on which robot in which simulator.
