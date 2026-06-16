# OpenRAL Safety Hazard Log

> Per CLAUDE.md §3: every PR that touches `packages/openral_safety/`,
> `packages/openral_safety_watchdog/`, `packages/openral_human_estop/`, or
> `cpp/openral_safety_kernel/` must add an entry here documenting (a) what
> changed, (b) the hazard or non-hazard analysis, and (c) that the change is
> at least as conservative as what it replaces.

---

## Entry 001 — `try_shutdown` sweep for e-stop/watchdog nodes (issue #290)

**Date:** 2026-06-12
**PR:** #290 (try_shutdown sweep — 4 safety-path nodes)
**Files changed:**
- `packages/openral_safety/openral_safety/supervisor_node.py`
- `packages/openral_human_estop/openral_human_estop/forwarder_node.py`
- `packages/openral_safety_watchdog/openral_safety_watchdog/deadman_watchdog_node.py`
- `packages/openral_safety_watchdog/openral_safety_watchdog/hardware_estop_node.py`

### What changed

All four `main()` entry points replaced bare `rclpy.shutdown()` with
`rclpy.try_shutdown()` (idempotent — no-op when the context is already shut
down) and added `except (KeyboardInterrupt, ExternalShutdownException): pass`
around `rclpy.spin(node)`.

### Hazard analysis

**No change to enforcement behaviour.** This PR modifies only the process
teardown path — the `main()` function that starts and stops the node process.
It does not modify:

- Any envelope check, threshold, or limit.
- Any topic publish/subscribe surface.
- Any estop firing logic (`_handle_violation`, `_fire_estop`, `_on_human_estop`).
- Any deadman deadline or watchdog arming/disarming logic.
- Any service callback (`/openral/estop_reset`).
- The C++ safety kernel (`cpp/openral_safety_kernel/`).

**Before:** `rclpy.shutdown()` in the `finally` block crashed with
`RCLError: rcl_shutdown already called on the given context` on every
operator Ctrl-C (SIGINT), because rclpy's SIGINT handler already shut the
context down before the `finally` ran. This replaced `KeyboardInterrupt`
with a confusing `RCLError` traceback and stalled the launch supervisor's
wait-for-children past the 30 s `shutdown_grace` window.

**After:** `rclpy.try_shutdown()` is idempotent (no-op if already shut down).
The `except (KeyboardInterrupt, ExternalShutdownException): pass` is scoped
exclusively to normal-teardown signals — it does NOT catch `Exception`,
`ROSError`, or `ROSSafetyViolation`. An E-stop condition or safety-path
failure that propagates up to `main()` is still not silently swallowed.

**Cannot leave motors energised:** These are process entry-points, not
actuation control loops. By the time `main()` is exiting:
- The safety supervisor has already published on `/openral/estop` for any
  in-flight violation (the `_handle_violation` path is unaffected).
- The deadman watchdog has already fired its estop via `_fire_estop`.
- The hardware estop node has already published on SIGINT-triggered edge.
- The C++ safety kernel (ADR-0020) owns the actuation gate independently and
  is not affected by Python process teardown.

**Conservatism:** The new behaviour is strictly at least as conservative as
the old: the enforcement path is byte-identical; only the teardown-failure
mode is repaired.

### Tests (structural regression guards)

Four AST-structural guards added — one per node:
- `packages/openral_safety/test/test_supervisor_node_sigint_shape.py`
- `packages/openral_human_estop/test/test_forwarder_node_sigint_shape.py`
- `packages/openral_safety_watchdog/test/test_deadman_watchdog_node_sigint_shape.py`
- `packages/openral_safety_watchdog/test/test_hardware_estop_node_sigint_shape.py`

Each asserts: (a) `try_shutdown` is used and bare `rclpy.shutdown()` is NOT
present in `main`, (b) the spin is wrapped catching exactly
`(KeyboardInterrupt, ExternalShutdownException)`, (c) the except does NOT
catch `Exception`/`ROSError`/`ROSSafetyViolation` (the "does not mask E-stop"
proof).

### Safety-WG reviewer gate

**This PR still requires explicit sign-off from a safety-WG reviewer before
merge**, per CLAUDE.md §3. The hazard analysis above and the structural test
suite are the author's contribution; the reviewer must independently verify
the no-enforcement-change claim.

---

## Entry 002 — Standardized description assets: relocate lowering inputs (ADR-0058)

**Date:** 2026-06-16
**ADR:** [ADR-0058](../adr/0058-standardized-description-assets.md) (standardized
robot description assets — URDF / xacro / MJCF / SRDF)
**PR:** _pending_ (implementing PR for ADR-0058; this entry is authored with the
ADR per CLAUDE.md §3 and links the regression test below as its mitigation)
**Files to change (safety-relevant subset):**
- `packages/openral_safety/openral_safety/urdf_lowering.py` — delete the
  divergent `_load_urdf_model`; route URDF/SRDF reads through the new
  `openral_core.assets.resolve_asset` resolver.
- `python/core/src/openral_core/assets.py` — new single resolver (the file
  locator the lowering tool now calls).
- The 16 `robots/<id>/robot.yaml` manifests — migrated to the `assets:` block;
  `ur5e`/`ur10e`/`rizon4`/`openarm` gain vendored `robots/<id>/<id>.urdf`.

### What changed

This change replaces four divergent asset-resolution mechanisms (two of them
URDF loaders) with **one** resolver, `resolve_asset(ref, kind)`, and folds the
asset references into a structured `RobotDescription.assets` block. For the
xacro-only robots (`ur5e`/`ur10e`/`rizon4`) and `openarm`, the lowering tool now
reads a **vendored, pre-expanded URDF** instead of expanding upstream xacro
in-process.

It changes **only how the source URDF/SRDF/MJCF files are located** — not their
contents, not the lowering algorithm, not the ACM sampling seed.

### Hazard analysis

**The C++ safety kernel does not read URDF/SRDF/MJCF at runtime.** It reads only
the lowered `collision_geometry` + `allowed_collision_pairs` from the manifest
(`collision_params_from_description`). URDF/SRDF/MJCF are *inputs to the offline
lowering tool* (ADR-0030), which produces those lowered fields at authoring time.

This PR does **not** modify:

- Any kernel check, threshold, capsule-distance test, or ACM lookup.
- The lowering geometry algorithm (mesh→capsule fit, primitive bounds).
- The ACM derivation or its deterministic sampling seed
  (`_RNG_SEED = 20260610`, `_N_SAMPLES = 2000`).
- The committed `collision_geometry` / `allowed_collision_pairs` values in any
  manifest.

**Same input bytes → same lowered output.** The upstream URDF/SRDF/MJCF reach the
lowering tool unchanged; the vendored URDFs are the *expanded* form of the same
upstream xacro the divergent loader expanded before. Therefore the lowered
geometry and ACM are byte-identical.

**Conservatism:** identical geometry and an identical ACM are, by construction,
at least as conservative as what they replace (CLAUDE.md §3). The change cannot
make any pair newly *allowed* (less safe) without changing the ACM bytes — which
the regression test forbids.

**Cannot leave motors energised:** no actuation path, no E-stop logic, and no
process-teardown path is touched; this is an authoring-time file-locator change.

### Mitigation — byte-identical lowering regression test (release blocker)

For every robot carrying `collision_geometry` in its manifest, re-run lowering
through the new resolver and assert the output is **identical** to the committed
values: byte-for-byte for the ACM pairs, geometric equality for the capsules.
**A diff blocks the release.** This is the primary mitigation. It is backed by
the unchanged existing safety suite:
`packages/openral_safety/test/test_urdf_lowering_fk.py` (incl.
`test_franka_acm_uses_srdf_when_srdf_path_set`), the `mjcf_lowering` tests, the
envelope-loader tests, the kernel integration tests, and the fleet guard
`tests/unit/test_collision_lowering_fleet.py`.

### Safety-WG reviewer gate

**This change requires explicit sign-off from a safety-WG reviewer before
merge**, per CLAUDE.md §3. The reviewer must independently verify (a) the
"kernel never reads these files / this only relocates them" claim and (b) the
byte-identical regression evidence across the fleet, including that the vendored
`ur5e`/`ur10e`/`rizon4`/`openarm` URDFs lower to the same geometry the in-process
xacro path produced.

- [ ] **PENDING: safety-WG reviewer sign-off** (human gate — not author-clearable).
