# ADR-0046 — NVIDIA Isaac GR00T as an out-of-process VLA backend

- **Status:** Accepted — PR1 (packaging + license posture) + PR2 (runtime adapter + sidecar) implemented 2026-06-10; only the live LIBERO sim-eval remains operator-run on a Python-3.10 GPU host
- **Date:** 2026-06-10
- **Related:** ADR-0010 (RLDX-1 ZMQ sidecar — the architectural precedent this
  reuses), ADR-0006 (HF-Hub rSkill packaging + license guard), ADR-0012
  (open-core licensing), ADR-0019 (state/action contract dims). Slot 0043 is
  reserved by ADR-0044; 0043=locate-in-view, 0045=isaac-sim-backend; 0045 is now taken by the Isaac Sim backend ADR, so this one takes 0046.

## Context

NVIDIA's Isaac GR00T line has moved well past where the repo's docs assumed it
sat. As of 2026-06, five generations exist, all ~3B-class, all on HF Hub:

| Model | HF repo | `model_type` | Weights license |
|---|---|---|---|
| GR00T N1 (2B) | `nvidia/GR00T-N1-2B` | `gr00t_n1` | OneWay **Noncommercial** |
| GR00T N1.5 | `nvidia/GR00T-N1.5-3B` | `gr00t_n1_5` | OneWay **Noncommercial** |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | `Gr00tN1d6` | OneWay **Noncommercial** |
| **GR00T N1.7** | `nvidia/GR00T-N1.7-3B` | `Gr00tN1d7` | **NVIDIA Open Model License — commercial OK** |
| GR00T N2 | — | — | Announced, not yet released |

Two facts drive this ADR:

1. **License split.** The repo's `RSkillLicensePosture.NVIDIA_NON_COMMERCIAL`
   guard and the CLAUDE.md §3 matrix treated *all* GR00T as non-commercial.
   That is correct for N1 / N1.5 / N1.6 but **wrong for N1.7**, which NVIDIA
   re-licensed under the commercially-permissive **Open Model License**.
   Posture must be keyed on the model *version*, not the family.

2. **Python-version incompatibility.** Isaac-GR00T targets **Python 3.10** on
   x86 dGPU / Jetson Orin (3.12 only on Jetson Thor / DGX Spark) and pins
   `flash-attn==2.7.4.post1` + CUDA. This repo is **Python 3.12-only**
   (`pyproject.toml` `>=3.12,<3.13`). A direct in-process import is the same
   break that already forced the lerobot `GR00TN15Config` stub in
   `tests/sim/conftest.py`. GR00T therefore cannot load in-process.

## Decision

Integrate GR00T as a new `ModelFamily = "gr00t"` that runs **out-of-process via
a ZMQ + msgpack sidecar** in its own Python 3.10 venv — reusing the exact
architecture of the RLDX-1 adapter (ADR-0010). This is a natural fit: **RLDX-1
is itself a GR00T-N1.5 finetune**, so the sidecar lifecycle, msgpack ndarray
wire codec, per-embodiment observation builders (LIBERO 8-D state + two RGB
views; GR1 humanoid), and chunk-replay are already proven in
`policies/rldx.py` + `tools/rldx_sidecar.py`.

License posture becomes version-aware:

- `RSkillLicensePosture.NVIDIA_NON_COMMERCIAL` — GR00T N1 / N1.5 / N1.6. The
  loader continues to block commercial deployment unless
  `OPENRAL_ALLOW_NONCOMMERCIAL=1`.
- `RSkillLicensePosture.NVIDIA_OPEN_MODEL` (new) — GR00T N1.7+. Added to
  `_LICENSES_ALLOWING_COMMERCIAL`; `is_commercial_use_allowed` returns True and
  the guard does not fire.

## Scope split

This is shipped in two PRs because the live runtime cannot be honestly verified
without a lab GPU host (GR00T N1.7-3B in bf16 ≈ 6 GB weights + the Cosmos-Reason
VLM does not fit the 8 GB reference laptop without NF4; and the exact LIBERO
observation key layout must be confirmed against the checkpoint's
`experiment_cfg/metadata.json` on that host).

**PR1 (this change) — packaging, schema, license; fully unit-tested:**

- `ModelFamily` gains `"gr00t"`; docstring notes the out-of-process design.
- `RSkillLicensePosture.NVIDIA_OPEN_MODEL` + membership in
  `_LICENSES_ALLOWING_COMMERCIAL`.
- `policy_deps.py`: `gr00t` install hint / group / required-imports. The
  required-imports probe targets `openral_sim.policies.gr00t` (the PR2 adapter
  module), so the rSkill is **gracefully dropped** from a live policy palette
  with an install hint until PR2 lands — exactly as `xvla` is dropped when its
  package is absent. No half-wired runtime, no silent `KeyError` at dispatch.
- `pyproject.toml`: opt-in `gr00t` extras group (pyzmq + msgpack), mirroring
  `rldx`.
- `rskills/gr00t-n17-libero/` — manifest + README for `nvidia/GR00T-N1.7-LIBERO`
  (`license: nvidia_open_model`, `model_family: gr00t`, `franka_panda`, 8-D
  state / 7-D `delta_ee_6d_plus_gripper`). Validates against the real schema and
  passes the publish-readiness gate.
- Docs: CLAUDE.md §3 matrix, `vla_compatibility.md`, `rskills.md`, METHODS, and
  the repo-state map; plus the pre-existing `OPENRAL_ACCEPT_NONCOMMERCIAL` →
  `OPENRAL_ALLOW_NONCOMMERCIAL` doc typo fix.

**PR2 (this change) — runtime adapter + sidecar; openral side unit-tested:**

- `python/sim/src/openral_sim/policies/gr00t.py` — `@POLICIES.register("gr00t")`
  factory reusing `_Gr00tFamilySidecarAdapter` (`family="gr00t"`,
  `OPENRAL_GR00T_*` env namespace). The LIBERO obs/action keys
  (`state.x…`/`state.gripper`, `action.x…`, agentview/wrist video, task
  annotation) were confirmed against Isaac-GR00T's
  `gr00t/eval/sim/LIBERO/libero_env.py` and match the family adapter's existing
  `_build_libero_obs` exactly — so the reuse is contractually correct. Default
  embodiment tag is `LIBERO_PANDA` (enum value `libero_sim`), the tag the LIBERO
  finetune exposes.
- The shared sidecar scaffolding lives in `openral_sim/_sidecar_common.py`; the
  adapter (`_Gr00tFamilySidecarAdapter`, formerly `_RLDXSidecarAdapter` — alias
  kept) drives both families via a backward-compatible `family` field. RLDX
  behavior and tests are unchanged; `tools/rldx_sidecar.py` is untouched in
  substance.
- `tools/gr00t_sidecar.py` — clones `NVIDIA/Isaac-GR00T` into an isolated 3.10
  venv and runs `gr00t/eval/run_gr00t_server.py` (`Gr00tPolicy` + `PolicyServer`,
  `--use-sim-policy-wrapper`) over the same ZMQ + msgpack wire, with an NF4
  wrapper for ≤ 8 GB hosts.
- `policy_deps` probe flipped to `("zmq", "msgpack")` (the runtime now exists);
  `tests/unit/test_gr00t_adapter_auto_spawn.py` exercises the real factory,
  manifest, and sidecar-script locator with a real socket — no live model.

**Live NF4 verification (2026-06-10, RTX 4070 Laptop 8 GB).** The NF4 quant
path was run for real against `nvidia/GR00T-N1.7-3B` (Cosmos-Reason2-2B backbone,
which is HF-gated — accept its license first). `Gr00tPolicy` loaded as **468
`Linear4bit` modules at 3.31 GB peak VRAM**, and a **full `get_action` forward
ran end-to-end in 683 ms at 2.75 GB peak** — comfortably inside 8 GB. Running it
surfaced **two** real NF4 bugs the wrapper was missing, both now fixed in
`tools/gr00t_sidecar.py`:

1. `gr00t_policy.py` casts the model with `model.to(device, dtype=bf16)` after
   load, which transformers FORBIDS on a bnb-4bit model. Patch `PreTrainedModel.to`
   to strip the float dtype when quantized (keep the device move).
2. `dit.py`'s `TimestepEncoder.forward` infers its compute dtype from
   `next(self.parameters()).dtype` → `uint8` for 4-bit-packed weights → SiLU
   crashes on `Byte`. Hard-pin the timestep projection to bf16. **This is the
   identical bug + fix the rldx sidecar already carries** (RLDX-1 is a GR00T
   finetune) — concrete confirmation the two families share internals, not just
   a wire.

The CUDA-13.2 host was a non-issue: uv resolves Isaac-GR00T's torch to a `cu128`
wheel, so `flash-attn==2.7.4.post1` installed from a prebuilt wheel. (The base
3B exposes only pretrain tags — `OXE_DROID…` / `XDOF` — so the smoke ran under
`OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`; the LIBERO rSkill uses `LIBERO_PANDA`.)

**Live LIBERO rollout attempt (2026-06-10) — a third bug + an open design gap.**
Booting the shipped `tools/gr00t_sidecar.py` against the local
`GR00T-N1.7-LIBERO/libero_spatial` checkpoint (`--embodiment-tag LIBERO_PANDA
--quantization nf4`) and driving it with `openral sim run --config
scenes/sim/franka_libero_pnp.yaml --rskill rskills/gr00t-n17-libero` got the
**full stack to connect**: the `run_gr00t_server` PolicyServer loaded the LIBERO
checkpoint NF4 (3.5 GB), the gr00t adapter connected (`mode=existing`), and the
LIBERO MuJoCo env initialized. Two outcomes:

3. **Wrapper bug (fixed):** `_safe_to` / the TimestepEncoder patch referenced
   `torch` but it was imported only inside `_patched` — `NameError`. Hoisted
   `import torch` to the top of the quant block. After this the server boots and
   loads the LIBERO checkpoint cleanly.
4. **Codec mismatch (FOUND + FIXED + verified):** the rollout first failed at
   the obs with `Video key 'image' must be a numpy array. Got <class 'dict'>`.
   The shared adapter encodes arrays with `np.save` bytes
   (`{"__ndarray_class__": …}`, the codec RLDX's `run_rldx_server` speaks) while
   `gr00t/policy/server_client.py`'s `MsgSerializer` uses **`msgpack_numpy`**
   (`{nd: True, …}`, `allow_pickle=False`) — same framing, different ndarray
   encoding, so arrays reached GR00T as undecoded dicts. **Fix:** the sidecar
   wrapper repoints `MsgSerializer.to_bytes`/`from_bytes` at the adapter's
   `np.save` codec, so `run_gr00t_server`'s PolicyServer speaks the adapter's
   wire (keys, the `{observation, options}` envelope, and the
   `ping`/`get_action`/`reset` endpoints already matched). Chosen over a custom
   server because it reuses GR00T's robust server + sim-wrapper.
5. **Obs-shape nuance (FIXED):** GR00T's `LIBERO_PANDA` video modality is
   single-frame (horizon 1); the adapter sent RLDX-1's 4-frame history. Added a
   per-family `video_offsets` field (`(0,)` for gr00t, the 4-frame default for
   rldx).

**Net (verified end-to-end):** with the codec bridge + `video_offsets` fix, the
**closed-loop LIBERO rollout runs** — `openral sim run` on
`franka_libero_pnp.yaml` connects to the NF4 sidecar, the GR00T policy returns
7-D action chunks (`action_dim=7`, ~0.2 s/inference), and a 50-step episode
completes (`budget_violations=0`, `mean_lat_ms≈39`, video + `summary.json`
written). The smoke episode reports `success=False` — expected: it is a single
50-step episode with an empty task instruction, not a tuned benchmark. The
graded eval (`tests/sim/test_franka_gr00t_libero.py` → `eval/libero.json`, with
the task instruction set + multiple episodes) is the remaining work; paper
numbers (arXiv:2503.14734) are not faked — `reproduced_locally: false` until run
(§1.2, §1.11).

## Alternatives considered

- **Direct in-process load.** Rejected: Python 3.10 / transformers / flash-attn
  pins are incompatible with the 3.12-only workspace (§ Context 2).
- **A bespoke GR00T sidecar protocol.** Rejected: the RLDX sidecar already
  speaks a working msgpack-ndarray contract over ZMQ for GR00T-architecture
  policies; reusing it avoids a second wire format and a second 1000-line
  adapter.
- **Treat all GR00T as non-commercial (status quo).** Rejected: factually wrong
  for N1.7 and blocks a legitimately commercial-licensed model (§1.2, §1.9).

## Consequences

- A commercial deployment can now use GR00T N1.7 without the non-commercial
  guard firing; N1/N1.5/N1.6 remain correctly blocked. License lineage stays
  enforced and is now version-accurate.
- GR00T checkpoints carry their own normalization (`experiment_cfg`) rather than
  lerobot processor JSONs, so `gr00t` is **not** in `_MODERN_PROCESSOR_FAMILIES`
  and its manifests declare no `processors` block.
- Until PR2, a `gr00t` rSkill packages, validates, and publishes but does not
  dispatch — surfaced explicitly via the palette install hint, never silently.
