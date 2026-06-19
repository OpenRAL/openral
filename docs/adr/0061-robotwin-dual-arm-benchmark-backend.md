# ADR-0061 — RoboTwin 2.0 dual-arm benchmark backend (SAPIEN, out-of-process sidecar)

- **Status:** Accepted 2026-06-19. Scene backend + sidecar + task-matched rSkill landed,
  unit-validated, and **live reset/step verified on the 8 GB RTX 4070 reference host** with
  the externally-provisioned RoboTwin assets + CuRobo sidecar (see **Live verification**).
  Full SmolVLA scored eval/video is still not reproduced locally.
- **Date:** 2026-06-19
- **ADR number:** `0061`. `0060` (benchmark task-data gate) is the previous entry; the
  integer is not load-bearing — cross-refs use filenames.
- **Related:**
  - **ADR-0045** — Isaac Sim scene backend. RoboTwin reuses the exact same out-of-process
    ZMQ/msgpack sidecar machinery (`openral_sim.sidecar.SidecarClient`,
    `openral_sim._sidecar_common`).
  - **ADR-0060** — `evaluated_tasks` task-data gate. The RoboTwin rSkill declares
    `evaluated_tasks: ["robotwin"]` so the gate accepts it on every RoboTwin scene.
  - **ADR-0009** — `openral benchmark run|scene` producers this backend plugs into.
  - **ADR-0019** — `state_contract` / `action_contract` dims (14-D aloha-agilex).
  - Issue #54 (follow-up to PR #48 "clean benchmark tier to official params").

## Context

PR #48 cleaned `scenes/benchmark/` so every benchmark uses **official parameters** and at
least one **task-matched rSkill that actually runs** (ADR-0060 gate). OpenRAL's existing
benchmark tier is single-arm (Franka/Panda LIBERO, ManiSkill, SimplerEnv WidowX) or the
gym-aloha bimanual cube tasks (MuJoCo, only 2 tasks). It has no **large-scale bimanual**
benchmark to back the ALOHA-bimanual / GR1-humanoid story.

**RoboTwin 2.0** (Chen et al., [arXiv 2506.18088](https://arxiv.org/abs/2506.18088), MIT) is
the natural fit: 50 dual-arm collaborative-manipulation tasks, 5 embodiments, 100k+ public
expert trajectories, strong domain randomization. Its benchmark is well-defined and has a
public leaderboard.

The hard part is the simulator. RoboTwin runs on **SAPIEN** (not MuJoCo) and its install
chain (SAPIEN, CuRobo, mplib, pytorch3d, multi-GB assets, py3.10, CUDA 12.1) is incompatible
with the openral workspace's `>=3.12,<3.13` pin and its VLA torch stack — the same class of
problem ADR-0045 solved for Isaac Sim.

### Authoritative protocol facts (pinned; CLAUDE.md §1.2)

| Property        | Value                                                                    |
| --------------- | ------------------------------------------------------------------------ |
| Simulator       | SAPIEN                                                                    |
| Tasks           | 50 dual-arm (snake_case, e.g. `beat_block_hammer`, `stack_blocks_two`)   |
| Embodiment      | **aloha-agilex** bimanual, 14-DOF (7/arm); action 14-D joint-space [-1,1] |
| Cameras         | `head_camera`, `left_camera`, `right_camera`                             |
| Episode horizon | `episode_length = 300`, `fps = 25` (LeRobot RoboTwin env defaults)        |
| Eval protocol   | 100 episodes/task, seeds 0/1/2, 50 `demo_clean` train, sim built-in success |
| Settings        | Easy (`demo_clean`) / Hard (`demo_randomized`)                           |
| Dataset         | `lerobot/robotwin_unified` (LeRobot v3, Apache-2.0)                       |
| `open_laptop`   | Broken upstream (`check_success` uses unset `arm_tag`) — omitted          |

Baseline averages (Easy/Hard): Pi0 46.4/16.3, DP3 55.2/5.0, RDT 34.5/13.7, ACT 29.7/1.7,
DP 28.0/0.6.

**LeRobot ships a native `robotwin` env** (`lerobot-eval --env.type=robotwin
--env.task=<task>`), which OpenRAL already depends on for its other lerobot policies. That is
the cleanest authoritative way to drive the SAPIEN env — we wrap it rather than re-implement
the task logic.

## Decision

Add RoboTwin 2.0 as a **SAPIEN scene backend run out-of-process via a sidecar venv**,
structurally identical to the Isaac Sim backend.

1. **`PhysicsBackend.SAPIEN`** — new enum value (additive, backward-compatible, no
   `schema_version` bump). Isaac and Genesis each got their own slot; for truthfulness we do
   not reuse the `mujoco` slot the way the older ManiSkill3 scenes did.

2. **Scene backend** `openral_sim/backends/robotwin.py` —
   `@SCENES.register("robotwin", fixed_robot="aloha_agilex")`. `_RoboTwinSimSidecar` is a
   `SimRollout` that marshals `reset`/`step`/`render`/`close` to the sidecar over
   `SidecarClient`; `action_dim` (14) comes from the `ping` reply. The scene is **fixed** to
   the aloha-agilex embodiment because the LeRobot integration, dataset, and public
   checkpoints are all aloha-agilex; other RoboTwin embodiments are a future relaxation.

3. **Sidecar** `tools/robotwin_sidecar.py` — runs under the externally-provisioned
   venv, constructs LeRobot's `robotwin` gym env for the requested task, and serves the same
   `ping/reset/step/render/close` msgpack+ndarray protocol the openral side speaks.
   Observations cross the wire in the eval-layer shape (`images` dict of the three RoboTwin
   cameras, 14-D `state`, `task` text).

4. **Robot manifest** `robots/aloha_agilex/robot.yaml` — a real `RobotDescription` for the
   AgileX dual-arm (14-DOF, 3 cameras). Like `aloha_bimanual` it ships **no on-disk URDF/MJCF**
   (the sidecar owns the SAPIEN robot); the manifest exists so the eval layer can resolve the
   action/state contract and run the embodiment/sensor gate. Auto-registered by the
   `robots/` directory scan.

5. **Dependencies** — a `robotwin` dependency-group carrying only the openral-side wire
   (`pyzmq`, `msgpack`), plus a `robotwin_client` install plan in `openral_sim._deps`
   (mirrors `isaac_client` / `rldx_client`). The heavy SAPIEN+RoboTwin stack lives only in
   the sidecar venv, provisioned out-of-band (opt-in `OPENRAL_ROBOTWIN_AUTO_PROVISION=1`,
   else a typed `ROSConfigError` carries the manual recipe).

6. **Benchmark + scenes** — `scenes/benchmark/robotwin_<task>.yaml` for a representative
   subset and a `benchmarks/robotwin.yaml` suite, all at the official horizon
   (`max_steps: 300`, `n_episodes: 100`, `success_key: is_success`, seed 0).

7. **Task-matched rSkill** — `rskills/smolvla-robotwin/` wraps the official
   **`lerobot/smolvla_robotwin`** checkpoint (450M, Apache-2.0, `smolvla` family — already
   supported; trained on `robotwin_unified`). `evaluated_tasks: ["robotwin"]` satisfies the
   ADR-0060 gate on every RoboTwin scene. `2toINF/X-VLA-RoboTwin2` (`xvla` family) is a
   ready second policy noted for a follow-up.

## Consequences

- **First large-scale bimanual benchmark** in the OpenRAL tier; backs the ALOHA-bimanual
  story with 50 tasks instead of 2, on the same eval producers and JSON result schema.
- **No new policy adapter** is needed: the strongest publicly-available, license-clean,
  8 GB-fitting checkpoint (`smolvla_robotwin`) maps onto the existing `smolvla` adapter. The
  issue-#54 suggestion of RDT/pi0 adapters is **deferred** — RDT-1B (1B params) and pi0 need
  new families + checkpoints that do not fit the 8 GB reference host and add adapter surface
  with no benchmark-tier benefit today.
- **License posture** (CLAUDE.md §1.9): RoboTwin (MIT), SAPIEN (MIT), LeRobot (Apache-2.0),
  `smolvla_robotwin` weights (Apache-2.0), dataset (Apache-2.0) — all permissive. Nothing is
  vendored: the SAPIEN+RoboTwin stack is an externally-provisioned sidecar venv.
- **Cost:** a second SAPIEN install path on disk (the sidecar venv, multi-GB with assets) and
  a second long-boot sidecar. Mitigated by the shared sidecar machinery and per-scene port
  derivation (no cross-scene adoption).

### Live verification

The scene backend, sidecar, robot manifest, dep plan, scenes/suite, and rSkill are landed
and unit-tested (manifest validation, scene/suite YAML validation, ADR-0060 gate accepts the
rSkill, SAPIEN `PhysicsBackend` enum). Hosts without the externally-provisioned sidecar venv
still `pytest.skip` (CLAUDE.md §1.11); hosts with it run a real SAPIEN reset/step.

**Reference-host attempt (2026-06-19, 8 GB RTX 4070 Laptop):**

- ✅ **SAPIEN works here.** A bounded probe installed `sapien==3.0.3`, created a
  `Scene`, and rendered a 256×256 frame offscreen (`cam.take_picture()` →
  `(256, 256, 4)` float32).
- ✅ **RoboTwin assets + CuRobo sidecar works here.** The sidecar cache was fully
  provisioned under `~/.cache/openral/robotwin-sidecar`: LeRobot `main`, the
  `RoboTwin-Platform/RoboTwin` checkout/assets (~16 GB), and NVLabs CuRobo `v0.7.8`
  built from source against Torch cu128 using `cuda-toolkit=12.8.2` + GCC 13.
- ✅ **Live `lift_pot` reset/step succeeded.** `tools/robotwin_sidecar._RoboTwinEnv`
  built the native LeRobot `RoboTwinEnv`, reported `action_dim=14`, reset with
  camera1/camera2/camera3 RGB observations plus a `(14,)` float32 state, and executed
  one zero-action step (`reward=0.0`, `is_success=False`). The OpenRAL wrapper test
  then passed end-to-end through ZMQ auto-spawn:
  `pytest tests/unit/test_robotwin_backend.py tests/sim/test_aloha_agilex_smolvla_robotwin.py -q`
  → 19 passed.
- ⚠️ **Full SmolVLA scored eval/video still not reproduced.** The live verification is
  simulator/sidecar/wire/action-contract verification, not a policy success number. The
  `lerobot/smolvla_robotwin` checkpoint and eval JSON remain to be run via
  `openral benchmark scene --config scenes/benchmark/robotwin_lift_pot.yaml --rskill
  rskills/smolvla-robotwin` on a host with enough free disk for the policy cache and video
  artifacts. No benchmark metric is invented.

## Alternatives considered

- **In-process SAPIEN import** — rejected: SAPIEN/RoboTwin pin py3.10 + a torch/CUDA build
  that clashes with the 3.12 VLA stack (same reason ADR-0045 chose a sidecar for Isaac).
- **Reuse `aloha_bimanual` for `robot_id`** — rejected: the gym-aloha ViperX manifest has one
  top camera, not RoboTwin's three; the embodiment is genuinely different. A dedicated
  `aloha_agilex` manifest keeps the contract honest.
- **Reuse the `mujoco` `PhysicsBackend` slot** (as ManiSkill3 does) — rejected for
  truthfulness; RoboTwin is SAPIEN, and Isaac/Genesis set the precedent of a dedicated slot.
- **Build RDT/pi0 adapters now** — deferred (see Consequences).
