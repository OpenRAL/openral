# ADR-0057 — `kind: reward` rSkills: robotic reward models as parallel task-progress monitors

- **Status:** Accepted 2026-06-15. Phases 0/2/3 validated empirically on an
  8 GB GPU (load, NF4 quantization, working ZMQ sidecar). Schema + manifest +
  this ADR land here; production runner backend, Reasoner tool, and live sim
  test follow in the same branch.
- **Date:** 2026-06-15
- **ADR number:** `0057`. `0056` is claimed by the in-flight
  `feat/multi-detector-locate` branch (on-demand detectors as reasoner tools);
  the integer is not load-bearing — cross-refs use filenames.
- **Related:**
  - ADR-0047 — `kind: vlm` scene VLM as a read-only `query_scene` Reasoner tool.
    The reward monitor is the same shape (read-only, S2-cadence, advisory,
    out-of-process sidecar) but emits **scalars** (progress/success) instead of
    free-form text. `query_scene` is the *escalation target* when the reward
    signal is ambiguous.
  - ADR-0056 — on-demand detectors as prompt-able Reasoner tools; same
    "auxiliary perception runs parallel to the VLA and feeds the Reasoner"
    pattern, with `locate_in_view` as a sibling read-only tool.
  - ADR-0046 — GR00T out-of-process ZMQ sidecar; the reward sidecar reuses the
    same process-isolation + msgpack scaffold.
  - ADR-0037 — `kind: detector` + the GStreamer perception bus / tee that
    supplies frames on real hardware.
  - ADR-0018 §4 — the Reasoner has no actuation authority over read-only tools;
    the reward signal is advisory (CLAUDE.md §1.1).

## Context

A `kind: vla` policy emits action chunks but carries no notion of whether it is
*succeeding*. Today the Reasoner infers success indirectly — absence of errors,
`query_scene` text answers, world-state changes — but has no continuous,
normalized per-frame progress/success signal. Without one, a stalled or failing
rollout runs to a timeout instead of triggering replanning.

[`robometer/Robometer-4B`](https://huggingface.co/robometer/Robometer-4B) (paper
*Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory
Comparisons*, arXiv 2603.02115, **Apache-2.0**) is a Qwen3-VL-4B reward
foundation model that, given a rollout's frames + a task instruction, predicts
per-frame **progress** and per-frame **success** probability. That is exactly
the missing signal. The question this ADR answers: how does such a model live in
OpenRAL, and how does its output reach the Reasoner?

### What was validated before deciding (gating spike)

- **Load** (Phase 0): the on-disk `config.json` says `architectures: ["RFM"]`
  but the class is `RBM`, with **no `auto_map` and no Hub-side modeling code** —
  vanilla `AutoModel` cannot load it. It loads only via the upstream `robometer`
  package (`load_model_from_hf`), which **requires `transformers==4.57.1`** (5.x
  changes processor kwargs and drops `input_ids`). Discrete (binned) mode yields
  per-frame progress ∈ [0,1] + per-frame success ∈ [0,1]; continuous mode yields
  raw regression values.
- **Quantize** (Phase 2): NF4 (the repo's `Linear.numel ≥ 4M → Linear4bit` rule)
  takes 8.91 GB bf16 → **3.33 GB resident / 3.56 GB peak** (8-frame forward),
  output intact — **4.44 GB headroom** on an 8 GB GPU.
- **Run in parallel** (Phase 3): a working ZMQ sidecar streamed a real rollout
  video and produced **progress 0.21 → 0.88 with success spiking to 0.90 at task
  completion**. Parallel-to-VLA on 8 GB is feasible alongside a small NF4 VLA.

## Decision

1. **Add a new rSkill kind, `reward`** (`RSkillKind`), with a `RewardContract`
   manifest block (`progress_range`, `success_threshold`, `preference`,
   `frame_window_s`, `target_fps`, `num_bins`, `instruction_required`). It joins
   `detector` / `vlm` as a perception kind: embodiment-agnostic, `actuators_required`
   empty, no action/state contract, no VLA preprocessing — enforced in
   `RSkillManifest._check_kind_consistency`. A new `RSkillAction.MONITOR` verb
   labels it. Backward-compatible additive change (no `schema_version` bump, no
   migrator) — every existing manifest still validates.

2. **Run it out-of-process** as a long-lived ZMQ sidecar (reusing the
   GR00T/scene-VLM scaffold), loading the NF4 model via the pinned `robometer`
   package in an isolated venv (`transformers==4.57.1`). Weights resolve from the
   HF cache; no per-run `git clone`.

3. **Abstract the frame source** so the same skill works in **sim and real**:
   the sidecar subscribes to the same `sensor_msgs/Image` camera topic the
   co-active VLA consumes — GStreamer tee on real hardware, sim HAL camera
   publisher in `deploy-sim`. Not GStreamer-bound.

4. **Surface it as a read-only Reasoner tool** (`QueryTaskProgressTool` /
   `query_task_progress`), not an `ExecuteSkill`. The Reasoner co-activates the
   reward rSkill with a VLA; the sidecar continuously ingests frames into a
   rolling window; the Reasoner queries it on demand for the windowed assessment
   (`progress_now`, `success_now`, trends, `stalled`) and uses it to continue,
   escalate to `query_scene`, advance, or enter the replanning ladder. **The
   signal is advisory only** — it never actuates and never suppresses a
   `ROSSafetyViolation`.

## Alternatives considered

- **Reuse `kind: vlm` + `query_scene`.** Rejected for semantic clarity: a reward
  model emits structured per-frame scalars with a contractual range/threshold,
  not free-form text. Folding it into `vlm` would overload that kind's
  open-vocab-QA meaning and lose the typed `RewardContract`. The two are
  complementary — `query_scene` is the escalation target when reward is ambiguous.
- **Continuous push topic** (sidecar publishes a progress stream the Reasoner
  subscribes to). Rejected in favor of continuous-ingest + on-demand query: the
  Reasoner pulls the windowed assessment when it wants context, which matches its
  event-driven cadence and avoids a high-rate topic the Reasoner would have to
  debounce. The rolling buffer still gives it history ("over the last X s").
- **In-process with the VLA.** Rejected: a 4 B VLM contends with the VLA on the
  GPU step loop and needs a different `transformers` pin; process isolation keeps
  the control path clean and makes CPU/2nd-GPU/cloud placement transparent.

## Consequences

- New `reward` kind + `RewardContract` + `MONITOR` action in `openral_core`
  (additive). `_EMBODIMENT_AGNOSTIC_KINDS` and `_PERCEPTION_KINDS` gain `reward`.
- A new runner backend (`openral_runner.backends.reward`) + sidecar + Reasoner
  tool + co-activation wiring.
- 8 GB co-residency is real but workable: NF4 both models, keep the frame window
  bounded (activation peak scales with window / resolution / `num_bins`), or
  place the sidecar on CPU / a 2nd GPU / the cloud.
- The upstream `robometer` package is executed in the sidecar (not an
  OpenRAL-trusted org); it is pinned by commit and isolated. `transformers` is
  pinned to `4.57.1` in the sidecar venv. `tools/quantize_rskill.py
  --loader transformers` does **not** work for this model (no `auto_map`);
  packaging must quantize via the `robometer` loader path.
- Reward output is advisory; it can never gate motors or be on the control path.
