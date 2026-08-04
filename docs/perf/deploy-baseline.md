# Deploy bringup baseline

Measured on the reference host, `refactor/deployment_optimization` @ 2026-08-04.

**Host:** RTX 4070 Laptop (8 GB), SO-101 follower on `/dev/ttyACM0`, two USB
bench cameras, ROS 2 Jazzy, warm HF cache.

Both paths were run end to end as a full ROS graph — not a component harness.
Numbers come from the `deploy.bringup` spans and the dashboard's own
`/api/state`, i.e. the same surface an operator sees.

---

## `openral deploy sim` — `scenes/deploy/so101_box.yaml` (MuJoCo twin)

| node | transition | duration |
|---|---|---|
| `openral_hal_so100` | `on_activate` | **733.8 ms** |
| `openral_hal_so100` | `on_configure` | **267.5 ms** |
| `openral_world_state` | `on_configure` | 71.3 ms |
| `openral_world_state` | `on_activate` | 0.4 ms |
| `openral_skill_runner` | `on_configure` | 17.8 ms |
| `openral_skill_runner` | `on_activate` | 0.0 ms |
| `openral_reasoner` | `on_configure` | 0.6 ms → **FAILURE** (no LLM key on this host) |

## `openral deploy run` — `scenes/deploy/so101_bench.yaml` (real arm)

Run with `--no-enable-reward-monitor` to isolate these numbers from the known
6.5× reward-monitor load starvation, which is tracked separately.

| node | transition | duration |
|---|---|---|
| `openral_hal_so100` | `on_configure` | **111.7 ms** |
| `openral_hal_so100` | `on_activate` | 43.2 ms |
| `openral_world_state` | `on_configure` | 32.3 ms |
| `openral_world_state` | `on_activate` | 0.6 ms |
| `openral_skill_runner` | `on_configure` | 18.1 ms |
| `openral_reasoner` | `on_configure` | 0.8 ms → **FAILURE** (no LLM key) |

Skill dispatch (`rskill-smolvla-so101-eraser_place-bf16`, real arm actuating):
**34.4 s wall** for a 20 s execution budget, i.e. **~14 s** of load. Resident
GPU after load: **1676 MiB** (SmolVLA only). Teardown returned the card to
**16 MiB**.

### The headline: the 300 s timeouts are ~1000× the real cost

`sim_e2e.launch.py` gives the HAL and reasoner a **300 s** transition timeout,
justified in a comment by "HAL `on_configure` takes ~6 s, or ~27 s on a cold
robocasa kitchen". For an SO-101 the measured cost is **267 ms in sim and
112 ms on the real arm**. The comment is not wrong — it describes robocasa —
but until now there was no way to know it did not describe *your* robot. That
is the entire point of the `deploy.bringup` span.

Bringup is not where deploy latency lives. **Model load is**: ~14 s against
~0.9 s of total lifecycle transitions.

---

## Event-log noise

One second of a 30 Hz two-camera deploy previously produced **121 `info`
rows/s**, cycling the 200-slot ring every 1.65 s. Measured live after the
span-band inversion, on both paths:

| | debug | info | warn | error |
|---|---|---|---|---|
| `deploy sim` | 200 | 6 | 0 | 1 |
| `deploy run` | 200 | 6 | 0 | 1 |

Every `info` row is a `deploy.bringup` line; the single `error` is the
reasoner's failed configure. Routine 30 Hz traffic contributes **zero** rows to
the operator's INFO view.

---

## Three bugs only the live run could find

Each of these passed unit tests and would have shipped.

1. **The warm-up never ran.** `warm_up_lerobot_policy` called
   `policy.select_action` on a raw batch, but SmolVLA reads
   `observation.language.tokens`, which the *preprocessor* produces by
   tokenising `task`. Live result: `KeyError: 'observation.language.tokens'`.
   The non-fatal guard worked exactly as designed — a warning, deploy
   continued — which is precisely why it went unnoticed until the log was
   read. Fixed by running the batch through the adapter's own preprocessor,
   as `step()` does. Confirmed on the real arm: zero `warmup_failed`.

2. **`info` rows could not survive the flood.** Demoting the per-tick spans
   took info *generation* to zero, but the main ring is one FIFO shared with
   the debug stream: 193 of 201 rows were `hal.read_state`, about 7 s of
   history. Every `deploy.bringup` row was evicted within seconds of being
   emitted; only the reasoner's ERROR row survived, because errors were the
   only thing mirrored into the protected lane. An info row that cannot
   outlive the flood is no more useful than one that was never emitted. The
   lane now mirrors everything above debug.

3. **`rskill.execute` is emitted at two rates.** `rskill_runner_node` opens
   one per dispatched goal; `rSkillBase.step()` opens one per *tick*, under
   the same name. Promoting it to a headline put **60 rows into a 20 s
   window** — reproducing the exact flood the allow-list exists to prevent,
   and evicting the real headlines from the protected lane. Demoted. It
   cannot be a headline until the per-step emitter is renamed.

---

## Not captured

`phase_timer` load phases (`smolvla_imports` / `from_pretrained` /
`to_device`) are emitted at INFO through the structlog→OTel bridge, so they
reach the collector but not the console, and they had cycled out of the
dashboard ring by the time it was queried. The ~14 s load total above is
wall-clock around the dispatch, not a phase breakdown. For a per-phase
breakdown use the offline profiler, which renders the same phases directly:

```bash
just profile-load rskills/rskill-smolvla-so101-eraser_place-bf16
```

Also not covered here: the reasoner tick path (this host has no LLM key, so
`on_configure` fails by design) and the reward-monitor leg.
