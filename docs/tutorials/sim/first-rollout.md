# Your first sim rollout

A trained policy driving a simulated robot, in one command, on a laptop with
no GPU.

The pairing is [Diffusion Policy](https://arxiv.org/abs/2303.04137) × PushT,
picked because it asks the least of your machine: gym-pusht is 2-D `pymunk`
with no MuJoCo render context to configure, the checkpoint runs on CPU, and
both ship in the `sim` dependency group.

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

The first run downloads the checkpoint (~1 GB) from the Hub into your HF
cache; later runs start from cache. The run prints a resolved header, then one
line per episode, then the success rate:

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

That episode line is a real CPU run capped at `--max-steps 16` — short enough
to fail, which is what makes it worth reading.

Reading that output:

- **`vla`** is the *adapter* id, taken from the manifest's `model_family`. It is
  not the rSkill name; several rSkills share one adapter.
- **`[mujoco]`** is the scene's declared `PhysicsBackend`, and for PushT it is
  cosmetic — gym-pusht is `pymunk`. The field is carried for schema
  uniformity, not dispatch.
- **`mean_lat`** is mean per-step latency. Diffusion Policy runs 100 DDPM
  denoising steps per chunk, so it is the slowest adapter in the tree by
  design: the manifest records 1756 ms for a warm full chunk on its
  reference host (RTX 4070 Laptop, CUDA 12.8, PyTorch 2.10). The 6684 ms
  above is the same work on CPU only.
- **`budget_viol`** counts steps that blew the manifest's
  `latency_budget.per_chunk_ms`. Expect some on any host: the first chunk of an
  episode pays warm-up, so a 137-step GPU rollout still logged 18. They are a
  signal to read, not a failure.
- The exit code is **not** gated on success rate. A failed episode still
  exits `0` — you decide the threshold.

### Budget your patience on CPU

The scene's default episode is 300 steps. At the CPU latency above that is
roughly half an hour for one episode — fine to leave running, surprising if you
are watching it. Cap it while you are finding your feet:

```bash
uv run openral sim run --config scenes/sim/pusht.yaml \
  --rskill rskills/diffusion-pusht --max-steps 16
```

That took about two minutes end to end on a 20-core laptop CPU, including a
~10 s policy load from a warm cache. With a GPU, add `--device cuda:0` and use
the full episode.

You will also see a `rskill.unpinned_weights` warning: this rSkill's
`weights_uri` names a branch rather than a commit, so the load is not
byte-reproducible. Harmless here, worth pinning for anything you publish.

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

The default `--video-style debug` stacks two bands per episode: an image band
on top, and an `observation state` line plot underneath. The top band shows the
policy's stitched camera grid when the policy has more than one input camera;
otherwise, as on PushT, the rollout and policy views are the same picture and it
is labelled `rollout / policy view`. Passing the flag also switches on per-step
frame capture. Use `--video-style world` for a clean single-view render
instead.

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

### Further reading

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
