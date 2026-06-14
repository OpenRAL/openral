# ADR-0031: Explicit simulation / real HAL separation with deterministic command routing

- Status: **Proposed**
- Date: 2026-06-01
- Related: [ADR-0023](0023-data-driven-mujoco-hal.md) (`MujocoArmHAL.from_description`);
  [ADR-0029](0029-unified-hal-lifecycle-node.md) (one robot.yaml-driven lifecycle node);
  [ADR-0025](0025-reasoner-managed-background-services.md) (panda_mobile lifecycle node,
  `SimAttachedHAL`); CLAUDE.md §1.3 (types are the contract), §1.4 (explicit beats implicit),
  §3 (HAL is layer 0).

## Context

A robot's simulation HAL and real-hardware HAL are not declared consistently, and the choice
of HAL **type** leaks out of the manifest into environment config and runtime parameters.

1. **`RobotDescription.sdk_entry` is overloaded.** A single `str | None` field names a **sim**
   HAL for some robots (`so100`/`so101` → `SO100DigitalTwin`, `rizon4` →
   `Rizon4MujocoHAL`) and a **real** HAL for others (`franka` → `FrankaPandaRealHAL`, `ur5e` →
   `UR5eRealHAL`, `aloha` → `AlohaHAL`, `sawyer` → `SawyerRealHAL`). It is never imported at
   runtime today — it is dead metadata — and for `so100`/`so101` it points at
   `SO100DigitalTwin`, which is not even a HAL (it is a lerobot `Robot` device that plugs *into*
   `SO100FollowerHAL`).

2. **`deploy sim` decides sim-vs-real at runtime.** The so100 lifecycle node branches on an
   injected `sim_robot_yaml` param; the panda_mobile node branches on `sim_env_yaml`. The CLI
   registry (`_ROBOT_HAL_REGISTRY`) carries per-robot `supports_sim_env_yaml` /
   `supports_sim_robot_yaml` flags that inject those params.

3. **`deploy run` decides sim-vs-real from env config.** The in-process `HardwareRunner`
   (so100-only) selects a digital twin vs real serial from a `hal.transport.digital_twin`
   boolean in the `RobotEnvironment` YAML.

The net effect: the same command can boot a different HAL class depending on YAML, violating
"types are the contract" and "explicit beats implicit".

## Decision

1. **Schema.** Replace `RobotDescription.sdk_entry` with a structured submodel
   `hal: HalEntrypoints { sim: str | None, real: str | None }`. `sdk_kind` (license posture) is
   orthogonal and kept. Each field is a `"module:Class"` import string or `None`.

2. **Sim derivation.** When `hal.sim is None` **and** a `sim:` block is present, the sim HAL is
   `MujocoArmHAL.from_description(description)` (ADR-0023). So every plain arm leaves `hal.sim`
   null and the derivation provides the twin; only non-generic sim HALs (e.g.
   `panda_mobile` → `PandaMobileHAL`, which has no `sim:` block) name `hal.sim` explicitly.

3. **One resolver.** `openral_hal.build_hal(description, *, mode: Literal["sim","real"],
   transport=None) -> HAL` is the sole HAL-construction seam. The ROS lifecycle nodes, the
   `deploy run` runner factory, and `deploy sim` all route through it.

4. **Deterministic command → mode.**
   - `openral deploy sim` ⇒ `mode="sim"`.
   - `openral deploy run` ⇒ `mode="real"` — and only works against connected hardware (the real
     HAL's `connect()` fails otherwise). No env-config flag selects HAL type.
   - `openral sim run` / `openral benchmark run` are unchanged: scene backends own the robot,
     no HAL is constructed.

5. **Missing HAL is a typed error.** `build_hal(mode)` raises `ROSCapabilityMismatch` when a
   robot lacks the HAL for the requested mode (e.g. `sawyer` for sim, `rizon4` for real). A
   robot may legitimately declare neither HAL (scene-only robots: `gr1`, `widowx`,
   `google_robot`, `pusht_2d`); the error fires only when `build_hal` is *called* for the
   missing mode.

6. **The "sim device behind a real HAL" (`SO100DigitalTwin` in `SO100FollowerHAL`) is not a
   HAL type** — it is a fake transport. Because `deploy run` now requires real hardware, that
   path is removed from `deploy run`; its no-hardware CI coverage migrates to `deploy sim` (or a
   clearly-labelled test-only twin harness), never an env-config branch.

## Alternatives considered

- **Two flat fields** `sim_hal_entry` + `real_hal_entry` — rejected; the submodel groups them
  and extends to a future `mock`/`replay` mode.
- **Keep `transport.digital_twin`, gate by command** — rejected; leaves the type-deciding flag
  in env config, the exact entanglement being removed.
- **Make real HALs `from_description`-constructible like sim HALs** — rejected; real HALs
  legitimately take transport-specific kwargs (`robot_ip`, `fci_ip`, serial `port`) and embed
  their own safety-envelope constants. The resolver bridges the two conventions instead.

## Consequences

- `deploy run` against a digital twin is no longer possible (use `deploy sim`).
- Sim-only (`rizon4`/`openarm`/`g1`/`h1`/`panda_mobile`) and real-only (`sawyer`) robots are
  enforced by `ROSCapabilityMismatch`; several real HALs remain skeletons (Franka/UR M3) and
  will raise "real HAL not implemented" until their driver lands — the honest state.
- Adding a new robot needs only a manifest declaring both HALs — no per-command routing code.
- Layer touch: layer 0 (HAL) + the `openral_core` schema. `schema_version` stays `"0.1"` (no
  migrators, CLAUDE.md §6); every `robots/*/robot.yaml` is updated in the same change.

## Roadmap (out of scope here — separate ADRs/PRs)

- **`deploy run` → ROS graph:** converge `deploy run` onto the `deploy sim` launch graph
  (kernel / sensors / dashboard / nav) with `hal_mode:=real`; retire the in-process
  `HardwareRunner` deploy path. Subsumes ADR-0029's unification.
- **`sim run` HAL-driven native harness:** run native MuJoCo scenes over the sim HAL (no ROS)
  for fast robot + rSkill iteration; generalize `SimAttachedHAL` so a scene takes a `robot_id`
  and attaches `build_hal(mode="sim")`. Benchmark/external scenes stay scene-backend.
- **Config reorg:** `examples/sim` → `scenes/`, `examples/robot` → `deployments/`.
