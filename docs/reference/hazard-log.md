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
