# Your first sim rollout

Get a real policy driving a real simulated robot, in one command, on a laptop
with no GPU. This is the shortest path from a fresh clone to a rollout you can
watch, measure and replay.

It uses **Diffusion Policy × PushT** deliberately: gym-pusht is a 2-D `pymunk`
rigid-body environment with no MuJoCo render context to configure, the
checkpoint runs on CPU, and both ship in the `sim` dependency group. Nothing
here is a mock — it is the real
[Diffusion Policy](https://arxiv.org/abs/2303.04137) checkpoint from the
LeRobot Hub against the real benchmark environment.

Already have a GPU and want the flagship pairing instead? Skip to
[Step 6](#6-where-to-go-next).

---

## 1. Install

```bash
git clone https://github.com/OpenRAL/openral && cd openral
just bootstrap                # uv + ROS 2 + system deps
just sync --group sim         # workspace + the sim group
```

`just sync` wraps `uv sync --all-packages` plus a repair step. **Never run bare
`uv sync`** — it trips on the `hf-libero` metadata and leaves the venv
half-built. The full rules are in [Managing the Python environment & dependency
groups](../../contributing/toolchain.md#managing-the-python-environment-dependency-groups).

Installed via the one-liner rather than a clone? Use `openral install sim`
instead of `just sync --group sim`.

Check the host before going further:

```bash
uv run openral doctor
```

A GPU row reading `absent` is fine for this page. See the
[quickstart](../../quickstart/index.md) for what each row means.

---

## 2. Pick the two halves

Every `openral sim run` composes exactly two things: a **scene** (`--config`, a
`SimScene` YAML carrying robot × scene × task) and an **rSkill** (`--rskill`,
the policy). Both are required. Each has its own listing command:

```bash
openral sim list        # scene configs — paste into --config
openral rskill list     # rSkills, in-tree and installed — paste into --rskill
```

Neither builds a sim, touches the GPU, or emits telemetry, so both are safe on
any host. For this page the two halves are:

| Half | Value | What it is |
| --- | --- | --- |
| `--config` | `scenes/sim/pusht.yaml` | The PushT scene + the `pusht/0` task, 300 steps |
| `--rskill` | `rskills/diffusion-pusht` | Diffusion Policy, ~263 M params, fp32 |

The scene hard-fixes its robot to `pusht_2d`, so there is no `--robot` to pass
— the PushT "robot" is a synthetic 2-DoF pusher, not a physical embodiment.

---

## 3. Dry-run it first

`--dry-run` resolves the config, resolves the rSkill, and runs the same
embodiment / sensor compatibility gate a real rollout hits — without building
the sim or downloading a single weight. It is the cheapest way to check a
pairing:

```bash
uv run openral sim run \
  --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht \
  --dry-run
```

You get the resolved plan and a `0` exit code. An incompatible pairing exits
non-zero here instead of after a multi-gigabyte download — try it by pointing
`--rskill` at a skill for another embodiment:

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/smolvla-libero --dry-run    # exits 1: embodiment mismatch
```

---

## 4. Run it

```bash
uv run openral sim run \
  --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht
```

The first run downloads the checkpoint from the Hub into your HF cache;
later runs start from cache. The run prints a resolved header, then one line
per episode, then the success rate. The shape, with the per-run values
elided:

```
============================================================
  openral sim run
  robot : pusht_2d
  scene : pusht  [mujoco]
  task  : pusht/0
  vla   : diffusion  (rskills/diffusion-pusht)
  seed  : 0  episodes=1
============================================================

  ep0: success=<bool> steps=<n> reward=<f> mean_lat=<f>ms budget_viol=<n>

  success_rate: <k>/<n> = <p>%
```

Reading that output:

- **`vla`** is the *adapter* id, taken from the manifest's `model_family`. It is
  not the rSkill name; several rSkills share one adapter.
- **`[mujoco]`** is the scene's declared `PhysicsBackend`, and for PushT it is
  cosmetic — gym-pusht is `pymunk`. The field is carried for schema
  uniformity, not dispatch.
- **`mean_lat`** is mean per-step latency. Diffusion Policy runs 100 DDPM
  denoising steps per chunk, so it is the slowest adapter in the tree by
  design: the manifest records 1756 ms for a warm full chunk on its
  reference host (RTX 4070 Laptop, CUDA 12.8, PyTorch 2.10). Expect
  meaningfully worse on CPU.
- **`budget_viol`** counts steps that blew the manifest's
  `latency_budget.per_chunk_ms`. Expect violations on CPU; the budget is
  pinned to a GPU reference host.
- The exit code is **not** gated on success rate. A failed episode still
  exits `0` — you decide the threshold.

Slow on CPU? Cut the episode short rather than waiting:

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht --max-steps 30
```

With a GPU, add `--device cuda:0`.

---

## 5. Get something back from it

### A JSON summary

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht --save-dir out/
```

Writes `out/summary.json`: the fully-resolved `config`, an `episodes` array
(`success`, `steps`, `total_reward`, `mean_step_latency_ms`,
`max_step_latency_ms`) and the aggregate `success_rate`. This is the file to
assert on in CI.

> This is a rollout summary, not an eval result. It does **not** write into the
> rSkill package. `eval/<benchmark>.json` is produced by the benchmark tier —
> see [Write & publish an rSkill §5](../rskill/write-and-publish-an-rskill.md).

### A video

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht --save-video example_videos
```

The default `--video-style debug` writes a 3-panel montage per episode: what
the policy saw, the rollout view, and a joint-position plot. Passing the flag
also switches on per-step frame capture. Use `--video-style world` for a clean
single-view render instead.

### A live dashboard

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht --dashboard
```

Boots `openral dashboard` as a child process, points OpenTelemetry at it, and
shuts it down on exit. Open <http://localhost:4318> to watch spans, latency
histograms and camera thumbnails stream in live. Details in the
[dashboard quickstart](../../quickstart/dashboard.md).

### A live viewer

MuJoCo-backed scenes open a passive viewer window automatically when a display
is available. PushT is `pymunk`, so it has none — `--view` is for the MuJoCo
scenes in the next section.

---

## 6. Where to go next

### Sweep the same pairing

The per-axis flags overlay the YAML, so one config drives many runs. `--task`,
`--instruction`, `--max-steps`, `--n-episodes`, `--seed` and `--device` all
work this way:

```bash
for s in 0 1 2; do
  uv run openral sim run --config scenes/sim/pusht.yaml \
    --rskill rskills/diffusion-pusht --seed "$s" --n-episodes 5
done
```

### Move to a MuJoCo scene

LIBERO with SmolVLA is the flagship pairing — a 7-DoF Franka Panda, real MuJoCo
physics, a language-conditioned VLA. It needs a GPU, the `libero` group, and an
offscreen GL backend:

```bash
just sync --group libero
MUJOCO_GL=egl uv run openral sim run \
  --config scenes/sim/libero_spatial.yaml \
  --rskill rskills/smolvla-libero
```

`just sim-libero` wraps exactly that, plus two workarounds you would otherwise
hit by hand (a duplicate `hf-libero` egg-info, and LIBERO's stale absolute-path
config). Prefer the recipe.

> **LIBERO and RoboCasa cannot share a venv** — they pin conflicting robosuite
> versions, and `[tool.uv].conflicts` refuses the mix. Swap with
> `just sync --group libero` / `just sync --group robocasa` per task.

### Then

- **[Create a sim environment](create-a-sim-environment.md)** — author your own
  `SimScene` YAML, add a robot manifest, or write a scene / policy adapter.
- **[Write & publish an rSkill](../rskill/write-and-publish-an-rskill.md)** —
  package your own policy so `--rskill` can load it.
- **[Sim environments reference](../../reference/sim-environments.md)** — the
  full scene catalogue across all three tiers.
- **[VLA × Robot × Sim compatibility](../../reference/vla_compatibility.md)** —
  which policy runs on which robot in which simulator.
- **[`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md)**
  — the cookbook, including the scene-fixed-robot table.
