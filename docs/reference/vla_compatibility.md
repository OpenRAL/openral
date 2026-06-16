# VLA × Robot × Simulation Compatibility Matrix

This document is the canonical reference for which Vision-Language-Action models run on which robots under which simulators in the OpenRAL ecosystem. It is derived from upstream model cards, published papers, and checkpoint inspection. Entries marked **TBD** have not been locally verified; contributions welcome via PRs that include checkpoint inspection evidence.

See also: `CLAUDE.md §7.4` for the normative license matrix and `CLAUDE.md §6.4` for the rSkill packaging format.

---

## 1. Robots (Currently Integrated)

| Robot | Embodiment tags | DoF | Control mode | HAL module | Sim env |
|---|---|---|---|---|---|
| SO-100 (LeRobot) | `so100_follower` | 6 arm + 1 gripper | `joint_position` | `openral_hal.so100_follower` | SO-100 digital twin (MuJoCo, in-process) |
| Franka Panda (LIBERO sim only) | `libero`, `franka_panda` | 7 + gripper | `cartesian_delta` (6-D EEF + axis-angle) | LiberoEnv (lerobot) | LIBERO (MuJoCo via robosuite) |

Hardware-in-loop tested:
- **SO-100**: `tests/hil/` gate label `[self-hosted, lab-so100]`. USB tether required.
- **Franka Panda**: simulation only at this time; real-hardware HAL is planned (`packages/openral_hal_franka/`).

---

## 2. Embodiment Tag Registry

Embodiment tags are short strings that appear in `rskill.yaml` under `embodiment_tags` and in `RobotCapabilities.embodiment_tags`. The skill loader refuses to activate a skill whose tags do not intersect the target robot's capability set.

| Tag | Robot / Platform | DoF | Source dataset / paper | Notes |
|---|---|---|---|---|
| `so100_follower` | LeRobot SO-100 arm | 6 | [lerobot/so100](https://huggingface.co/datasets/lerobot/so100) | Follower arm in leader-follower teleoperation setup |
| `so101_follower` | LeRobot SO-101 arm | 6 | [lerobot/so101](https://huggingface.co/datasets/lerobot/so101) | Updated hardware revision of SO-100 |
| `libero` | Franka Panda on LIBERO benchmark | 7 + gripper | LIBERO (Yuke Zhu et al., NeurIPS 2023) | Simulation-only tag for LIBERO benchmark training |
| `franka_panda` | Franka Panda (real + sim) | 7 + gripper | Standard industry robot; widespread in BridgeData / Open X | Broader tag; use `libero` when targeting LIBERO-specific checkpoints |
| `widowx` | WidowX 250s | 6 | [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) | Low-cost research arm; common in Open X-Embodiment |
| `gr1` | Unitree GR1 humanoid | 23 | [NVIDIA Arena dataset](https://huggingface.co/nvidia) | Full humanoid; requires S0 cerebellar layer |
| `aloha` | Aloha bimanual teleoperation setup | 2 × 7 | [ACT paper](https://arxiv.org/abs/2304.13705) (Stanford / Toyota) | Bimanual; two Viperx arms with overhead + wrist cameras |
| `koch` | Koch arm | 6 | [lerobot/koch](https://huggingface.co/datasets/lerobot/koch) | Low-cost leader-follower arm |
| `piper` | Agilex Piper arm | 6 | ISdept dataset | Mid-range research arm from Agilex |

---

## 3. VLA Compatibility Matrix

Columns:
- **VLA (HF ID)** — canonical Hugging Face model ID
- **Sim env** — benchmark / simulator
- **Robot tag** — required embodiment tag(s)
- **State dim** — observation state vector
- **Cameras** — image inputs (resolution + any pre-processing)
- **Norm stats in checkpoint** — whether normalisation statistics are bundled
- **rSkill** — local skill stub path (if exists)
- **License** — SPDX expression for the *weights* (code license may differ)
- **Notes**

### 3.1 LIBERO (Franka Panda, MuJoCo via robosuite)

> The OpenRAL embodiment for LIBERO is `franka_panda` — see
> [`robots/franka_panda/`](https://github.com/OpenRAL/openral/tree/master/robots/franka_panda). The
> sim-imposed observation/action contract (8-D EEF state, 7-D
> delta-EEF action, 180° image flip) lives in the LIBERO scene
> adapter ([ADR-0007](../adr/0007-robot-sim-split.md)).


| VLA (HF ID) | Sim env | Robot tag | State dim | Cameras | Norm stats in ckpt | rSkill | License | Notes |
|---|---|---|---|---|---|---|---|---|
| `lerobot/smolvla_libero` | LIBERO | `libero` | **8-D** `eef_pos(3)+axisangle(3)+gripper_qpos(2)` ✓ | `image`→`camera1` + `image2`→`camera2` (256×256, flip 180°) ✓ | Yes — `step_5_normalizer_processor.safetensors` (state=[8], action=[7]) ✓ | `rskills/smolvla-libero/` | Apache-2.0 | Paper: Spatial 90% / Object 96% / Goal 92% / Long 71% (avg 87.3%). `scenes/benchmark/libero_spatial.yaml` (with `--rskill rskills/smolvla-libero`) |
| `HuggingFaceVLA/smolvla_libero` | LIBERO | `libero` | 8-D (same as above) | same as above | Yes (assumed same as above) | — | Apache-2.0 | Community mirror. Not locally verified. |
| `lerobot/pi05_libero_finetuned_v044` | LIBERO | `libero`, `franka_panda` | **8-D** same as smolvla ✓ | `image`+`image2` (256×256, flip 180°) + `empty_camera_0` (224×224 zeros) ✓ | Yes — `step_2_normalizer_processor.safetensors` (state=[8], action=[7]) ✓ | `rskills/pi05-libero-nf4/` | **Permissive research** (weights) / Apache-2.0 (code) | π0.5 (PaliGemma 3B backbone); requires ≥8 GB VRAM. `scenes/benchmark/libero_spatial.yaml` (with `--rskill rskills/pi05-libero-nf4`). **Non-commercial weights — see §5** |
| `lerobot/pi0_libero_finetuned_v044` | LIBERO | `libero`, `franka_panda` | 8-D (same format as pi05 — unverified) | same 3-camera format as pi05 (unverified) | Yes (assumed same format) | — | **Permissive research** (weights) / Apache-2.0 (code) | π0 (same license caveat). Not locally verified. |
| `lerobot/xvla-libero` | LIBERO | `libero`, `franka_panda` | **8-D** same `eef_pos+axisangle+gripper_qpos`; padded to max_state_dim=20 internally ✓ | `image`+`image2` (**224×224**, flip 180°) + `empty_camera_0` (224×224 zeros) ✓ | IDENTITY norm (no stats file) ✓; action output [20] (first 7 elements = LIBERO 7-D) ✓ | `rskills/xvla-libero/` | Apache-2.0 | xVLA (Florence-2 backbone, flow-matching). `scenes/benchmark/libero_spatial.yaml` (with `--rskill rskills/xvla-libero`) |
| `ar0s/groot_libero` | LIBERO | `libero`, `franka_panda` | TBD | TBD | TBD | — | Apache-2.0 (fine-tune) | GR00T on LIBERO; base model is NVIDIA AI Foundation **non-commercial** — guard required |

### 3.2 MetaWorld (Sawyer, MuJoCo)

> The OpenRAL embodiment for MetaWorld is `sawyer` — see
> [`robots/sawyer/`](https://github.com/OpenRAL/openral/tree/master/robots/sawyer). The MetaWorld benchmark
> simulates a Rethink Sawyer; some upstream checkpoints carry a
> `franka_panda` tag, but the actual robot is Sawyer
> ([ADR-0007](../adr/0007-robot-sim-split.md)).


| VLA (HF ID) | Sim env | Robot tag | State dim | Cameras | Norm stats in ckpt | rSkill | License | Notes |
|---|---|---|---|---|---|---|---|---|
| `lerobot/smolvla_metaworld` | MetaWorld MT50 | `franka_panda`, `manipulator` | **4-D** `agent_pos` (XYZ + gripper) ✓ | `observation.image`→`camera1` (256×256, flip+resize from 480×480) ✓ | Yes — `step_5_normalizer_processor.safetensors` (state=[4], action=[4]) ✓ | `rskills/smolvla-metaworld/` | Apache-2.0 | Action: 4-D delta (XYZ + gripper). Sawyer robot in MetaWorld (not Franka despite tag). `scenes/benchmark/metaworld_push.yaml` (with `--rskill rskills/smolvla-metaworld`) |

### 3.3 RoboCasa (Franka Panda, MuJoCo)

| VLA (HF ID) | Sim env | Robot tag | State dim | Cameras | Norm stats in ckpt | rSkill | License | Notes |
|---|---|---|---|---|---|---|---|---|
| `lerobot/smolvla_robocasa` | RoboCasa | `franka_panda`, `manipulator` | TBD | TBD | TBD | — | Apache-2.0 | Kitchen manipulation; no rSkill stub yet |

### 3.4 SO-100 / SO-101 (real robot or sim)

| VLA (HF ID) | Sim env | Robot tag | State dim | Cameras | Norm stats in ckpt | rSkill | License | Notes |
|---|---|---|---|---|---|---|---|---|
| `chamborgir/smolvla_pickplace_20k` | SO-101 real | `so101_follower` | TBD | TBD | TBD | — | Apache-2.0 | 20k steps pick-and-place fine-tune |
| `TakuyaHiraoka/act_so101_pick_diverse_objects` | SO-101 real | `so101_follower` | TBD | TBD | TBD | — | Apache-2.0 | ACT policy; diverse object pick task |
| `edge-inference/smolvla-so101-pick-orange` | Isaac Sim | `so101_follower` | TBD | TBD | TBD | — | Apache-2.0 | Isaac Sim backend; requires Isaac Sim license for reproduction |
| `HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b` | `so101_box` (MuJoCo) | `so101_follower` | **6-D** joint positions ✓ | `top`+`wrist`+`front` (224×224); scene `oak_top`→`top`, `wrist`→`wrist`, `front` zero-padded via image mask ✓ | Yes — `policy_{pre,post}processor` sidecars (state=[6], action=[6]) ✓ | `rskills/pi05-so101-pickplace-nf4/` (nf4 mirror at `OpenRAL/rskill-pi05-so101-pickplace-nf4`) | Apache-2.0 | π0.5 (4.14 B); nf4 fits 8 GB. Pick-place finetune; validated to load + step on `so101_box` (not insertion-trained — expect drift on the tube task). `scenes/sim/so101_tube_insertion.yaml` |

### 3.5 Other platforms

| VLA (HF ID) | Sim env | Robot tag | State dim | Cameras | Norm stats in ckpt | rSkill | License | Notes |
|---|---|---|---|---|---|---|---|---|
| `nvidia/smolvla-arena-gr1-microwave` | NVIDIA Arena | `gr1` | TBD | TBD | TBD | — | Apache-2.0 | Unitree GR1 humanoid, microwave-opening task |
| `ISdept/smolvla-piper` | Piper real | `piper` | TBD | TBD | TBD | — | Apache-2.0 | Agilex Piper arm; community fine-tune |

---

## 4. Sim Environment Reference

| Sim env | Backend | Install | Robot(s) | Task suites | Camera setup |
|---|---|---|---|---|---|
| LIBERO | MuJoCo (robosuite) | `CC=/usr/bin/gcc uv sync --group libero` + fix `~/.libero/config.yaml` to point at conda/pip libero data dirs | Franka Panda | libero_spatial, libero_object, libero_goal, libero_10 (= LIBERO-Long) | agentview + wrist 256×256; raw keys `image`/`image2` renamed to `camera1`/`camera2` by stored preprocessor; flip 180° |
| MetaWorld | MuJoCo | `uv run pip install metaworld==3.0.0 --no-deps` | Sawyer (MT50) | MT50 (50 tasks, v3) | 1 camera `corner2` 480×480 → resize to 256×256; `observation.image` key renamed to `camera1` |
| RoboCasa | MuJoCo | TBD | Franka Panda | Kitchen manipulation | TBD |
| SO-100 Digital Twin | MuJoCo (in-process, `python/sim/`) | `uv sync --group sim` | SO-100 | Smoke-test only (no task suite) | None — joint-space smoketest |
| SO-101 Box (`so101_box`) | MuJoCo (raw, `python/sim/src/openral_sim/backends/so101_box/`) | `uv sync --group sim` | SO-101 | tube-insertion (geometric success: tube vertical + lower tip ≥ 10 mm below the slotted-block hole top) — both block and tube spawn at random (x, y, yaw) on the floor each `reset()` | OAK-D Pro overhead (RGB + depth, default 640×480) + wrist RGB parented to the gripper body |
| NVIDIA Arena | Isaac Sim | Requires NVIDIA Isaac Sim license | GR1 | microwave | TBD |

### 4.1 LIBERO eval CLI

The lerobot `lerobot-eval` CLI drives LIBERO natively. Verified against `huggingface/lerobot` main as of 2026-05-05:

```bash
# Single suite
lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --eval.use_async_envs=true \
  --policy.device=cuda

# All four LIBERO suites
lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --eval.use_async_envs=true \
  --policy.device=cuda
```

Suite max steps: `libero_spatial` 280, `libero_object` 280, `libero_goal` 300, `libero_10` 520.

Note: `libero_10` is the lerobot/upstream name for LIBERO-Long. `LiberoProcessorStep` is injected automatically by `lerobot.envs.LiberoEnv` — no separate LIBERO gym install is required beyond the lerobot extras.

---

## 5. Known Limitations

- **Checkpoint normalisation requires `snapshot_download`**: `lerobot/smolvla_libero` bundles normalisation statistics in `policy_preprocessor_step_5_normalizer_processor.safetensors`. A bare `from_pretrained` call that only fetches `model.safetensors` + `config.json` will fail at inference time. Use `snapshot_download(repo_id="lerobot/smolvla_libero")` or `hf_hub_download` for the preprocessor file explicitly.

- **GR00T weights — license is version-specific** (ADR-0046): GR00T **N1 / N1.5 / N1.6** ship under the NVIDIA OneWay Noncommercial License. Any checkpoint that builds on those bases (e.g., `ar0s/groot_libero`) inherits the non-commercial restriction even if the fine-tune layer is Apache-2.0 — the rSkill manifest sets `license: nvidia_non_commercial` and the loader requires `OPENRAL_ALLOW_NONCOMMERCIAL=1` for a commercial deployment. GR00T **N1.7+** ship under the **NVIDIA Open Model License**, which permits commercial use — those manifests set `license: nvidia_open_model` (e.g., `rskills/gr00t-n17-libero`) and load without the guard. GR00T runs out-of-process via a ZMQ sidecar (the runtime adapter lands in ADR-0046 PR2).

- **π0 / π0.5 weights are "permissive research", not full Apache-2.0**: The code under `lerobot/` is Apache-2.0; the *weights* for `pi0` and `pi05` checkpoints carry a Physical Intelligence permissive-research license that is not equivalent to Apache-2.0 for commercial deployment. The corresponding rSkill manifests set `commercial_use_allowed: false`. See `CLAUDE.md §7.4` for the full VLA license matrix.

- **Reward monitor (`rskills/robometer-4b`, ADR-0057) co-residency on 8 GB**: The Robometer-4B reward monitor (`kind: reward`) runs in parallel with a VLA to score per-frame progress/success. At NF4 it is ~3.33 GB resident / 3.56 GB peak (8-frame window) on the 8 GB reference GPU, leaving ~4.4 GB — enough for a **small NF4 VLA** (e.g. SmolVLA ≈ 1.5–2 GB) but **not** a 3–4 GB π0.5/GR00T checkpoint simultaneously. When the VLA already saturates the card, place the reward sidecar on CPU, a second GPU, or a cloud host (the ZMQ transport makes location transparent), or shrink the reward `frame_window_s` / `num_bins` (activation peak scales with both). It is an **S2-cadence** monitor (~0.2–1 Hz over a frame window), not a per-control-step signal, and is **advisory-only** (never gates motors). In `deploy-sim`, the signal is only available on camera-rendering robots (the monitor needs `sensor_msgs/Image` frames). Apache-2.0; commercially usable.

- **MetaWorld, RoboCasa, and most SO-101 community entries are TBD**: RoboCasa and SO-101 community entries have not been locally verified. MetaWorld and the four LIBERO entries (smolvla, pi05, xvla, pi0) are now fully verified — see ✓ markers in §3.

- **Isaac Sim entries require a separate license**: `edge-inference/smolvla-so101-pick-orange` was trained in NVIDIA Isaac Sim. Reproducing its eval requires an Isaac Sim license and is not covered by the standard `uv sync --group sim` environment.

- **Embodiment tag `libero` implies simulation only**: The `libero` tag is defined for the LIBERO benchmark Franka Panda setup. Do not apply it to real Franka Panda deployments without verifying that action normalisation and camera geometry match your physical setup.

- **smolvla_libero state is 8-D, not 6-D**: The checkpoint's normalizer safetensors has `observation.state` stats for shape [8] (`eef_pos(3)+axisangle(3)+gripper_qpos(2)`), not [6]. The earlier config.json entry of shape [6] was a documentation error in the checkpoint. Always verify against the safetensors file, not config.json.

- **xvla action output is 20-D (padded)**: xVLA pads actions to `max_state_dim=20`. LIBERO's env.step expects 7-D. Slice `action_np = action_tensor.squeeze(0).cpu().numpy()[:7]` to extract the real 7-D action.

- **xvla is LIBERO-engine-only**: the xVLA adapter's env preprocessor (`LiberoProcessorStep`) consumes the nested LiberoEnv observation that the scene must expose as `observation['raw']`. Non-LIBERO scenes (e.g. the Isaac Sim Franka scenes) do not populate it, so `xvla` raises `ROSCapabilityMismatch` on the first step. Run xvla only on LIBERO scenes (`libero_spatial`, `franka_libero_pnp`, …).

- **GR00T / RLDX sidecars have no single-camera fallback**: these checkpoints read a fixed number of *distinct* camera streams positionally — LIBERO=2 (agentview+wrist), RC365=3, GR1/Simpler=1 — set by the manifest's `state_contract.layout`. Unlike the in-process lerobot adapters (smolvla / pi05 / act), which resolve their camera list from `scene.cameras` and adapt, the `gr00t` / `rldx` factories reject a scene that declares **fewer** cameras than the layout needs with an upfront `ROSCapabilityMismatch` (before the multi-minute sidecar boot). A scene that omits `cameras:` is the adapter default (LIBERO renders camera1+camera2 itself) and is never rejected. Example: `gr00t-n17-libero` runs on `isaac_franka_bowl_plate` (`cameras: [camera1, camera2]`) but not `isaac_franka_lift` (`cameras: [camera1]`).

- **π0.5 requires ≥8 GB VRAM**: The PaliGemma-3B backbone requires more memory than the 7-class GPU can provide in typical shared use. Use `--device cpu` for slow inference or a dedicated A100/H100 for production eval.

- **MetaWorld uses Sawyer, not Franka**: Despite the `franka_panda` embodiment tag in the lerobot metaworld dataset metadata, MetaWorld MT50 uses the Sawyer arm. The tag refers to the broader manipulation skill class, not the physical robot. Do not use smolvla_metaworld weights on a real Franka without re-training.

- **LIBERO `~/.libero/config.yaml` must point at the data files**: After installing `hf-libero` via pip, the config file at `~/.libero/config.yaml` pins absolute paths computed at first import and is never refreshed when you switch venv / workspace path. The next `just sim-libero` / `just sim-xvla-libero` / `just sim-pi05-libero` run then crashes inside `lerobot.envs.libero.get_task_init_states` with a `FileNotFoundError` on `<stale-path>/init_files/<task>.pruned_init`. The `_ensure-libero-config` private recipe (chained off every libero `just sim-*` target) invokes [`tools/fix_libero_config.py`](https://github.com/OpenRAL/openral/blob/master/tools/fix_libero_config.py) to detect + rewrite the file when stale; idempotent. Run it manually any time with `uv run --group libero python tools/fix_libero_config.py --verbose`, or set `LIBERO_CONFIG_PATH` to a project-local dir to bypass `~/.libero` entirely.
