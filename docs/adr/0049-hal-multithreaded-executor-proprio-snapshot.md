# ADR-0049: Dedicated HAL publisher thread + proprio snapshot (odom-starvation fix)

- Status: **Proposed**
- Date: 2026-06-11
- Related: [ADR-0034](0034-deploy-sim-scene-attach-for-arms.md) (deploy-sim scene-attach + the
  idle stepper; this ADR **keeps its single-thread contention rule** and extends it);
  [ADR-0029](0029-unified-hal-lifecycle-node.md) (unified HAL lifecycle node);
  [ADR-0027](0027-rskill-state-contract-bindings.md) (TF + state assembly rates);
  ADR-0048 (deploy-sim clock domain — the sibling nav-leg fix that exposed this).

## Context

`openral deploy sim` for `panda_mobile` (RoboCasa kitchen) runs the manifest-driven HAL
lifecycle node, which spins **single-threaded** (`rclpy.spin(node)` in
`make_lifecycle_main_from_manifest`). That one thread services every HAL timer/subscription:

| Callback | Rate | Cost | Touches `MjData`? |
|---|---|---|---|
| `_idle_step_tick` → `idle_step` → `env.step`(+render) | 10 Hz | ~15–80 ms | **writes** |
| `_on_safe_action` → `send_action` → `env.step`(+render) | ≤20 Hz | ~15–80 ms | **writes** |
| `_publish_scan` (2-D raycast `synthesize_laser_scan_2d`) | 10 Hz | ~10–30 ms | reads |
| `_publish_images` / `_publish_depth` (serve cached obs) | 10 Hz | small | reads cache |
| `_publish_joint_state` (`hal.read_state`) | 30 Hz | small | **reads** |
| `_publish_odom` (`hal.base_pose_6dof`) | 20 Hz | small | **reads** |

Because everything serializes on one thread, the expensive `env.step`+render and scan raycast
saturate it, and the **cheap, control-critical** publishers starve. Measured live on an
RTX 4070: `odom→base_link` TF was published at **~1.8 Hz** (configured 20 Hz; slam's `map→odom`
by contrast was a clean 20 Hz). At a ~0.55 s odom interval > Nav2's costmap
`transform_tolerance` (0.3 s), the costmap dropped fresh `/scan` messages ("extrapolation into
the future") and Nav2's progress checker could not confirm base motion → `"Failed to make
progress"`. (This was masked by the ADR-0048 frozen-clock bug, which broke nav entirely; fixing
the clock exposed it.)

### Why not a `MultiThreadedExecutor` (tried; rejected, with live evidence)

The obvious fix — a `MultiThreadedExecutor` with the env work in one
`MutuallyExclusiveCallbackGroup` and the publishers in another — **does not work for the
in-process MuJoCo HAL**, for two reasons:

1. **MuJoCo's EGL/GL context is thread-affine.** A `MutuallyExclusiveCallbackGroup` prevents
   *concurrency* but **not** *thread-hopping*: the executor's worker pool dispatches successive
   sim-group callbacks on whatever thread is free. So `env.step`'s render runs on a different
   thread than the one that created the GL context, and crashes. Observed live (this exact
   wiring, `deploy sim` on an RTX 4070): `idle stepper disabled after error:
   SimAttachedHAL.idle_step: env.step failed: EGLError`. The MuJoCo viewer is thread-affine too.
2. **`MjData` is not thread-safe.** Even setting rendering aside, any callback reading `MjData`
   (`read_state`, `base_pose`, the raycast) concurrent with an `env.step` write is a data race.
   ADR-0034's *Contention rule* ("exactly one writer to `env.step` at a time") assumed a single
   thread.

Conclusion: **keep all MuJoCo/GL work on one thread.** The decoupling must come from moving the
*publishing* off that thread — not from multi-threading the executor.

## Decision

Keep the HAL node **single-threaded** (all `env.step` / render / raycast on one thread, so
MuJoCo's GL context never moves) and move only the *publishing* of odom / joint_state / TF onto
a **dedicated publisher thread** that reads a plain-data proprio snapshot.

### 1. Single executor thread + a dedicated publisher thread

- The node spins single-threaded (`rclpy.spin`). All env / `MjData` / GL callbacks
  (`_on_safe_action`→`send_action`, `_idle_step_tick`→`idle_step`, the scan raycast, the camera
  / depth publishers serving `_last_obs`, the viewer sync) keep running there, mutually
  exclusive — ADR-0034's "exactly one `env.step` at a time" is preserved unchanged, and the GL
  context is only ever used on this thread.
- A `threading.Thread` started at activate (sim-attached HALs only) loops at the joint-state
  rate (~30 Hz) and publishes `joint_state` + `odom` + `odom→base_link` TF from the snapshot.
  It touches **only** the snapshot (thread-safe) and rclpy publishers (`publish()` /
  `sendTransform()` are thread-safe) — never `MjData`/GL — so it runs truly concurrently with
  the executor thread's heavy sim work. Stopped (signalled + joined) at deactivate, before the
  publishers it writes to are torn down.

For sim-attached HALs the legacy joint_state / odom **timers are not created** (the thread owns
publishing). Real HALs keep the timers and read encoders directly (`_proprio is None`).

### 2. Proprio snapshot seam (so the publisher thread never touches `MjData`)

A small `ProprioSnapshot` holder (one `threading.Lock` + an immutable `ProprioFrame`: the last
`JointState`, planar + 6-DoF base pose, base twist) owned by the HAL node:

- **Captured only on the executor thread** — immediately after each `env.step`
  (`_capture_proprio()` in `_send_action_traced` after `hal.send_action`, and via the
  `SimSensorBridge` `on_step` hook after `hal.idle_step`), plus once at activate. The capture
  calls `hal.read_state()` + `hal.base_pose_6dof()` (safe `MjData` read — same thread as the
  step) and stores **plain data**; the lock guards only the reference swap.
- **Read by the publisher thread** — `_publish_joint_state` / `MobileBaseBridge._publish_odom`
  read `snapshot.latest()` (a sub-microsecond locked read), then publish with a fresh
  wall-clock (or ADR-0048 sim-time) stamp. They never call into `MjData`.

The snapshot is captured at the step rate (≥10 Hz: idle stepper idle, ≤20 Hz under active nav)
and re-emitted by the thread at ~30 Hz, giving Nav2 a steady high-rate `odom→base_link` stream
(TF interpolates between captures). **Verified live:** odom rose from ~1.8 Hz → **~28 Hz**, the
costmap stopped dropping scans, and `navigate_to_pose` no longer aborts on the progress checker.

### 3. Scope

HAL layer only — `make_lifecycle_main_from_manifest` (stays `rclpy.spin`), the lifecycle base
(snapshot + capture + publisher thread + conditional joint_state timer), `MobileBaseBridge`
(`publish_from_snapshot` + conditional odom timer), `SimSensorBridge` (the `on_step` capture
hook). **No** change to `sim_attached.py`'s sim semantics (so it composes cleanly with ADR-0048
Phase 1, which also edits that file), the safety kernel, or the real-hardware HAL (no
`idle_step` → `_proprio is None` → unchanged timer-based publishing). `schema_version` stays
`"0.1"`.

## Consequences

**Positive**
- `odom→base_link` (and joint_state TF) publish at ~30 Hz regardless of per-step sim cost →
  Nav2 costmap stops dropping scans, progress checker works, the base navigates. Verified live
  (~1.8 Hz → ~28 Hz).
- ADR-0034's single-writer / single-thread guarantee is preserved **and** MuJoCo's GL context
  stays on one thread (no EGL breakage — the trap a multi-threaded executor falls into).
- Lower control-loop jitter generally (sensor/render work no longer head-of-line-blocks odom).

**Negative / risk**
- Correctness rests on the invariant *"only the executor thread touches `MjData`/GL; the
  publisher thread touches only the snapshot + publishers."* A future publisher-thread change
  that reads `MjData` would reintroduce a race — guarded by comments and the snapshot's
  HAL-agnostic design (it physically cannot reach the HAL).
- The snapshot adds one copy per step + a tiny lock; negligible vs `env.step`.
- A raw thread needs clean lifecycle handling (start at activate, signal+join at deactivate,
  daemon so a crash can't wedge shutdown) — implemented.

## Alternatives considered

1. **`MultiThreadedExecutor` + callback groups (the original plan).** Rejected — implemented and
   tested live, it crashes `env.step` with `EGLError` because MuJoCo's GL context is thread-affine
   and a `MutuallyExclusiveCallbackGroup` does not pin a thread (see §Why not a MultiThreadedExecutor).
2. **Single lock around all `MjData` access, one executor.** The publishers would still block for
   the full `env.step` (~15–80 ms) whenever they race a step — only partial relief. The
   snapshot + separate thread removes the block entirely.
3. **Publish odom/joint_state from inside the step callback.** Couples publish rate to step rate
   (10 Hz idle) — too slow for Nav2's 20 Hz loop, and bursts during active nav. Rejected.
4. **Raise Nav2 `transform_tolerance` to ~1.0 s (config-only band-aid).** Lets the costmap
   tolerate the laggy odom, but uses stale transforms for a moving base (mis-placed obstacles)
   and doesn't fix the progress checker, which needs real odom updates. Interim mitigation, not
   the root-cause fix. (Documented so the band-aid isn't mistaken for the solution.)
5. **Reduce camera/render load (fewer cameras, lower rate).** Treats the symptom; the cameras
   are a product requirement (perception bus). Rejected.

## Tests

- Unit: `ProprioSnapshot` under contention (a writer thread swapping frames while readers spin)
  never yields a torn frame; the holder is HAL-agnostic so a reader cannot touch `MjData`. Real
  `JointState` data, no mocks. (`test_proprio_snapshot.py`, incl. a 20k-iteration race.)
- Live (deploy-sim, reference host — done): under the full robocasa+nav+slam+octomap load,
  `odom→base_link` ≈ 28 Hz (was 1.8 Hz), **zero** `EGLError`, **zero** costmap scan-drops, and
  `navigate_to_pose` actuates the base without the odom-starvation "Failed to make progress"
  abort. Node-level wiring can't be unit-tested off a real ROS graph (no rclpy/GL under the test
  venv), so the live run is the verification of record.
