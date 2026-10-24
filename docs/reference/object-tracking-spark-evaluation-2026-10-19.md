# Object tracking: Spark evaluation, 2026-10-19

## Scope and status

Evaluate the #418 source snapshot using plain `sim run` as the policy benchmark
and `deploy sim` as the production-safety evaluation. The real Isaac orange
scene integration tests now pass (2 tests). One full plain policy benchmark and one direct-dispatch deploy attempt have
run. The deploy attempt stopped on a kernel world-collision refusal. Scene grading tests, CPU validation and builds do not
establish policy task success, contact safety or force calibration.

The isolated Spark checkout is
`/home/allopart/workspace/openral-418-eval-20261019`, branch
`feat/418-object-tracking`, based on
`85a8c812852d064b516c6e8e2945445a9ecfa071` plus the working changes.
The initial source manifest verifies SHA-256 hashes for all 86 changed/new source
files. The final manifest covers 91 files including compatibility changes and this
report. Evaluation artifacts are under
`/tmp/openral-418-spark-eval-20261019/` on Spark.

## Environment and isolation

- Spark: NVIDIA GB10, Linux ARM64, ROS 2 Jazzy; Isaac interpreter
  `/opt/isaac-sim/python.sh` (Python 3.12.13); OpenRAL host Python 3.12.
- Separate checkout, build/install directories and real virtual environment;
  existing worktrees and their shared virtual environment were not modified.
- Environment created with `just sync --frozen --group sim --group isaacsim`.
- ROS underlay: `/home/allopart/workspace/openral-418-render/install`, overlaid
  by this checkout's rebuilt packages. ROS domain 216. This is an existing
  underlay, not a clean-room build of every ROS dependency.
- No restock assets were copied into the public repository.

`just sync --frozen --group sim --group isaacsim --group scene-usd` failed:
the pinned `usd-core==26.8` distribution has no Linux ARM64 wheel/source package.
The explicit optional `object_catalog.reader_python` compatibility change lets
an independently provisioned USD interpreter extract metadata without starting
SimulationApp. The host still validates the catalog. This finding does not
justify substituting Isaac ground-truth poses for sensor observations.

## Completed checks

| Check | Result | Artifact |
| --- | --- | --- |
| Rebuild `openral_msgs`, `openral_world_state`, `openral_safety_kernel` | 3 packages succeeded | `build.log` |
| Rebuild OctoMap bridge against changed nested IDL | succeeded | `bridge-build.log` |
| Install updated reasoner/rSkill ROS packages and launch files | 2 packages succeeded | `ros-python-build.log` |
| Native model/runtime/calibration tests, isolated environment | 38 passed | `cpu-validation-isolated.log` |
| Earlier CPU batch including catalog/runtime checks | 42 passed, 11 skipped without host `pxr` | `cpu-validation.log` |
| Explicit bundled USD reader, public orange/bench catalog | loaded `tutorial-v1`; wrong revision refused | `usd-reader-validation.log` |
| Deploy orange, tracker shadow, direct dispatch | E-stop at tick 88; task failed | `orange-deploy-shadow-r2/` |
| Plain orange policy, seed 1, 3600 steps | task failed; 60 sim s; 72.889 wall s | `orange-policy-seed1/` |
| Real Isaac orange scene integration | 2 passed, 14.95 s | `orange-scene-20261019/pytest.log` |
| Final external-reader boundary tests on Spark | 3 passed, 6 skipped requiring host `pxr` | `catalog-reader-tests.log` |

Local final catalog/runtime/schema regression: 23 passed, including real
standalone USD equivalence and timeout descendant cleanup. Strict mypy and
Ruff/format pass for the reader changes. Schema exports, method coverage,
repository map and ROS topic graph checks pass. The full test suite was not
repeated for this compatibility change.

The USD-only check uses an explicit executable wrapper at
`/tmp/openral-418-spark-eval-20261019/usd-reader.sh`. This installation requires
`PYTHONPATH` to include
`/opt/isaac-sim/extscache/omni.usd.libs-1.0.3+00c488ae.la64.r.cp312` and
`LD_LIBRARY_PATH` to include its `bin` directory. Merely selecting
`/opt/isaac-sim/python.sh` fails to import `pxr`; the wrapper provisions those
paths. The loaded USD reports version `(0, 25, 11)`. No SimulationApp is created.
These installation paths belong to evaluation configuration, not the public
catalog or an automatic interpreter discovery path.

The first attempt to install the Python ROS packages selected a non-system
interpreter lacking `catkin_pkg`; explicitly selecting `/usr/bin/python3.12`
resolved the build. The isolated pytest run reports an unknown `asyncio_mode`
configuration warning because plugin autoload was disabled; the selected tests
are synchronous.

## Resource gate and remaining runs

The read-only capacity check observed 3–4 existing Isaac interpreter jobs owned
by another user, with roughly 66–74 GiB available system memory. The existing
session policy allows at most two Isaac instances total and requires at least
80 GiB available before starting another instance. The evaluation initially
started no simulator and terminated no existing job.
A later check observed no Isaac jobs, 0% GPU utilization and 107.9 GiB available.
After acquiring the lock, a fresh check confirmed zero jobs and 107.8 GiB, and
the public orange scene test was launched successfully. It verifies initial
objects/robot pose, image rendering, initial failure grading and success grading
after evaluator-only object teleportation. It does not execute a learned policy,
tracker/reasoner recovery or the deploy kernel stack. The earlier occupied
capacity snapshot is stored in `capacity-check.json`.

Before any launch, reacquire `/home/allopart/livesim/isaac.lock` and recheck
actual interpreter count and available memory. A prior successful poll does
not reserve capacity. Keep every evaluation output in a new directory.

Pending evidence:

1. Run matched seed/config plain benchmarks and deploy evaluations for orange,
   banana and the externally owned restock scene. Include declared/undeclared
   dimensions, location known/unknown, compatible partial views, calibrated
   contradictions, moved objects, occlusions and stale observations.
2. Verify UUID-scoped failure delivery/cancellation in the running deploy stack.
   Any run with reasoner dispatch disabled cannot establish live reasoner recovery.
3. Record task outcome, kernel refusals, query evidence, timings and video per
   run. Do not disable kernel checks or infer a calibrated external wrench from
   Isaac joint efforts.

Neither declared geometry tightening nor continuous calibrated contact permission
is implemented by these changes. Safety-WG review and independent calibration
remain prerequisites for those capabilities and real deployment.

## First full orange policy benchmark

The pinned SmolVLA policy ran one plain `sim run` episode with seed 1 and the
training instruction `Grab orange and place into plate`. It used the shipped
60 Hz scene and the full 3,600-step budget: 60 seconds of simulated time.
End-to-end wall time was 72.889 seconds including startup, teardown and video
writing. The task did not complete (`success=False`, reward 0). Exit status 0
means the evaluation ran successfully, not that the robot completed its task.
The run reported mean policy latency 3.46 ms and one latency budget violation.
This single seed establishes neither a success rate estimate nor a regression.

Artifacts: `orange-policy-seed1/command.json`, `timing.json`, `run.log`,
`artifacts/summary.json`, and `video/isaac_leisaac_pick_orange.mp4`. The external
LeIsaac stage emits PhysX warnings/errors for some environment colliders,
static joints and mass properties; they remain visible in the log and have not
been repaired or ruled out as affecting the benchmark.

The earlier 14.95-second pytest result comprises two short scene tests (at most
about 2.5 seconds of physics steps across the assertions, with independent
resets), not a policy episode. Its success case explicitly teleports oranges;
it must not be described as autonomous picking.

## First deploy attempt with tracking shadow

The first setup attempt (`orange-deploy-shadow/`) was interrupted after discovering
that the isolated environment lacked `timm`, required by the locator. It does not
count as a safety result. `just sync --frozen --group sim --group isaacsim --group
omdet` supplied the declared optional dependency. No shared environment was changed.

The second attempt (`orange-deploy-shadow-r2/`) reached the running HAL, preloaded
the pinned orange policy and dispatched the same instruction with a 60-graph-second
deadline. It preserved the shipped scene's safety configuration, enabled tracking
in `shadow` mode with an 8-second grounding wait, and used the OmDet locator.
There are no LLM credentials on this evaluation setup: the reasoner was explicitly
disabled and the existing validation dispatch tool sent the skill goal. This does
not test live reasoner recovery. No catalog or optional surface-model calibration
was added to the scene, so this attempt does not establish those capabilities.

Startup to readiness took 28.008 wall seconds. The goal returned after 11.9 wall
seconds, with first action feedback after 7.82 seconds, latest chunk/tick 87, and
an abort waiting for tick 88. The kernel latched a world-collision E-stop:
`attached:grasped_payload:gripper` against `voxel_128185`. Reported signed distance
was -0.0042287 m and sweep distance -0.0418301 m. Those are model/map distances,
not measured physical penetration. Whether this is an actual impending contact or
a conservative geometry refusal remains unresolved. Tracker logs also contain
candidate-split, border-truncation and missing startup pose/mask refusals; these
are recorded observations, not established causes of the collision.

The dispatch result is `success=false`, action status 6. Total harness wall time
including shutdown was 43.787 seconds. Launch exit status 1 includes interrupted
ROS node shutdown; it is not a task-success code. The first attempt required ROS
SIGTERM escalation during shutdown. Neither attempt relaxed safety limits or reset
the E-stop to continue. The plain benchmark used seed 1; no explicit equivalent
deploy seed was configured, so these runs are not a controlled matched-seed
regression comparison, despite the scene's authored fixed object layout.

Artifacts include exact commands/config, launch log, goal JSON and timing JSON
under each attempt directory. The complete declared/undeclared, movement,
contradiction, occlusion and stale-observation matrix across all three simulations
remains outstanding.

## Recorded shadow and tracker-on follow-through

Both subsequent runs required a receiving two-camera video recorder before goal
dispatch and preserved the kernel and shipped safety settings. They used the same
direct-dispatch harness (no live LLM recovery), with fresh process startup per run.

| Run | Readiness / total wall s | Goal result | Video artifact |
| --- | --- | --- | --- |
| `orange-deploy-shadow-r3video` | 30.009 / 69.594 | collision E-stop waiting tick 578; latest 577 | `cams.mp4` |
| `orange-deploy-on-r4video` | 28.009 / 49.230 | collision E-stop waiting tick 169; latest 168 | `cams.mp4` |

R3 collided in the model between `attached:grasped_payload:gripper` and
`voxel_127824`, signed distance -0.0186019 m. Workspace contact was denied
for speed (measured 0.541458 m/s versus the 0.5 m/s ceiling). The reported cell
center [-0.1425, -0.2025, 0.0375] m lay outside the evaluator's actual held-orange
bounds and was consistent with the plate footprint. Its later disappearance
does not establish payload ownership: it was beyond the ordinary legacy
current-pose clear reach, and its removal cause is unresolved. Snapshot TF is
measured state, whereas collision checks can evaluate proposed action poses;
there is no demonstrated collision-math defect.

R3 also emitted a stale region target identity `obj:orange_1`, although evaluator
ground truth identified Orange003 between the jaws. A region proximity test does
not establish physical identity. R4 in `on` mode bound tracker identity `trk:3`,
then stopped on `gripper` versus `voxel_137847` at signed distance -0.0209806 m.
Workspace denial code was 5. These distances describe model/map overlap, not
measured physical penetration. Neither run completed the task.

The shared-code audit independently found that legacy region/segmentation
primitive proximity can authorize published-grid deletion without cell ownership,
including in real deploy. The conservative correction retires that authority and
legacy raw sweep requests while preserving the existing guarded identity path.
This finding is not a causal explanation of the particular R3 cell disappearance.
Revision acknowledgments remain processing acknowledgments; waiting for one does
not justify ignoring an occupied cell. Private ADR0119 and the hazard log record
the decision before implementation. Safety-WG review is still outstanding.

## Full banana policy benchmark

The pinned `pi05-droid-jointpos-arena` policy ran the shipped banana scene with
seed 1 and `pick up the banana and place it in the bowl`, without the deploy
stack. It exhausted 1,050 steps at 15 Hz (70 simulated seconds), in 98.987 wall
seconds including startup/teardown and video writing. Task success was false,
reward 0; mean policy latency was 23.2 ms with one latency budget violation.
The run exit code was 0 because evaluation completed. This single episode is
not a success-rate estimate and cannot attribute failure to the tracking changes,
which this benchmark does not run.

Artifacts: `banana-policy-seed1/command.json`, `run.log`, `timing.json`,
`artifacts/summary.json`, and `video/isaac_arena_droid_banana_in_bowl.mp4`.
The recorded video is exactly 70 seconds. Policy inputs and state traces are
visible; videos are shared with the maintainer after each completed run.

Post-audit local regression across surface model/runtime, free-space configuration
and catalog runtime/declarations: 59 passed in 11.21 s using the real native PCL
module. The focused subagent batch also passed strict mypy and Ruff.

The plain banana benchmark also reports simulator contacts: distal finger/table
contacts from steps 68–69 onward (up to approximately 35 mm reported penetration),
and `panda_link7`/bowl contact over steps 394–558. These are upstream physics
contact statistics, not a calibrated real-world torque witness. They make it
especially inappropriate to require the deploy path to reproduce the benchmark
trajectory by relaxing collision rules or extending gripper allowances to the wrist.

Residual identity-clearing limitation: a validated track binding and known support/other-track/static keep-outs still cannot prove ownership of every raw occupancy cell inside its fitted shape. An unknown touching obstacle remains a concern. This fix removes legacy authority without expanding identity clearing; typed contribution ownership remains unfinished. Do not treat these runs as full production-safety acceptance.

## Recorded banana deploy evaluation

`banana-deploy-on-r1video` ran the shipped deploy scene with tracker `on`,
the pinned pi05 policy and the same prompt, using direct dispatch with a
70-graph-second deadline. Readiness took 34.009 wall seconds; goal duration
was 23.5 wall seconds, first feedback 12.71 seconds, latest tick 93. The kernel
latched world-collision E-stop waiting for tick 94: `right_distal_phalanx` versus
`voxel_1165795`, model signed distance -0.00858132 m, workspace denial code 5.
No contact limit or collision check was relaxed. Task success was false,
action status 6. Total harness time was 65.703 seconds; launch exit code 0
reflects teardown, not task success. No catalog/surface calibration was enabled.

Artifacts include `cams.mp4`, exact command/config, `monitor.jsonl`, snapshots,
evaluator-only ground truth, launch/goal logs and timings. The attempted failure
topic observer exited before discovery because its message type was omitted;
this run does not establish failure-bus delivery. Subsequent harnesses name
`openral_msgs/msg/FailureTrigger` explicitly. The reasoner remained disabled.
This run and the earlier orange/baseline runs precede the latest audit fixes.

The banana collision snapshot has no attached payload. Voxel 1165795 is occupied
at [0.4725, -0.0975, 0.0525] m in `panda_link0`. The nearby evaluator snapshot
places the banana bounds around [0.463626, -0.105503, 0.052569] m with half-extents
[0.069839, 0.097363, 0.035422] m, so the cell is consistent with banana occupancy.
This is not proof of exact surface identity or physical penetration. Logs contain
pre-measurement refusals for missing self-filtered clouds at the required mask/depth
skew. The kernel reported no active grasp exemption; no basis exists to bypass it
simply because the benchmark touched the target. Better grounded identity and
geometry must precede any additional permitted contact.

Maintainer review correction: the banana policy nearly completed its goal before
the banana fell out; the meaningful review window is the first 300 steps. The
1,050-step result above is the executed full-budget result, not evidence that the
early behavior was uniformly unsuccessful. Future banana benchmarks use a
300-step cap (20 simulated seconds), and deploy evaluation uses a corresponding
20-graph-second deadline. A trimmed first-300-step clip and a deploy clip with
the first 16 wall seconds removed were shared while retaining the full originals.
The initial deploy video included startup/grounding; first action feedback arrived
12.71 wall seconds after dispatch. Its stop was the explicit collision E-stop,
not an episode success or time budget exhaustion.

## External restock plain benchmark

Assets/configuration and cached model files remain outside the public repository.
A private scratch `SimScene` reused the external deploy scene block, mounting pose,
unit and seed 103, with its exact rubric instruction. The pinned weighted local
model directory resolved directly, avoiding a second Hub download.

`restock-policy-seed103` used a 2,100-step maximum and stopped successfully at
step 1,077 (reward 1). At 30 Hz this is 35.9 simulated seconds; total harness time
was 53.431 wall seconds. Mean policy latency was 7.04 ms, maximum 965.95 ms.
Subsequent preflight found that the source plain config uses a 900-step budget
(30 simulated seconds). Therefore this is a successful extended-budget run,
not a canonical-budget pass. Future restock runs use 900 steps / 30 graph seconds.

The exact private evaluation command/config and `artifacts/summary.json` remain
in the external artifact directory; `video/restock-plain-scene.mp4` was shared
with the maintainer. This is a single episode, not a success-rate estimate.

## Bridge audit correction validation

Retired the geometry-only published-grid and raw-sweep clearing path. Region,
segmentation, unknown, pinch-fallback and gripper-closure evidence retain occupancy.
The existing guarded identity path remains, with its residual provenance limit.
Real node regressions assert retained cells, no raw clear request, and processing
revision acknowledgment; link mappings/support witnesses do not grant clearing.
Local package build succeeded, and three focused native executables passed all
18 gtests. `git diff --check` passed. `clang-format` was unavailable locally.
The candidate snapshot containing all five bridge changes was hash-verified on
Spark and the bridge rebuilt successfully there before any post-fix rollout.
Full Spark bridge test results follow when complete.

Full Spark OctoMap bridge package tests after the legacy-clearing correction: **157 tests, 0 errors, 0 failures, 0 skipped** (`bridge-audit-tests.log`, `bridge-audit-test-results.log`). This validates the package regression boundary, not complete per-cell ownership or hardware safety.

The adjacent upstream `RobotSelfFilter` had the same unowned-evidence gap: it
appended all attachment primitives to its point mask. The follow-through fix
reuses the shared identity validation before suppressing payload returns;
robot-body masking is unchanged. Real-node tests first reproduced the defect,
then verified retained unowned/invalid/ambiguous points, stale-state refusal,
and the positive valid tracked case. Full local package suite after both fixes:
160 tests, 0 errors, 0 failures, 0 skipped. The same per-point ownership residual
remains for validated tracks. The private ADR/hazard amendment preceded the fix.

The banana scope-ordering regression passes 38 focused tracker session/grounding
tests, strict package/test mypy, Ruff and format checks. The test delivers a real
late result from the earlier deployment query after a grasp invocation opens,
and verifies exact scoped readiness, goal replacement and retraction. Competing
live declaration scopes leave tracks unowned. No kernel equality check changed.
Next ON evaluations explicitly require grounding before motion; this is an
intentional evaluation configuration change from the earlier permissive runs.

Post-fix combined local tracker/model/catalog batch: 97 passed in 28.50 s.
The same Spark batch: 85 passed, 12 skipped for missing host `pxr`, in 21.49 s;
standalone reader compatibility remains covered separately and the skipped catalog
cases passed locally. The 99-file scope/self-filter snapshot was hash-verified
on Spark before rebuilding. Method inventory refresh/check passed afterward.
The deploy 20/30-graph-second deadlines include grounding time and are not claims
of exactly 300/900 executed policy ticks; actual tick counts are reported separately.


## Corrected banana evaluations and force-source search

The complete bridge suite on Spark after the self-filter correction passed:
160 tests, 0 errors, 0 failures, 0 skipped. Logs are
`bridge-scope-self-filter-tests.log` and `bridge-scope-self-filter-test-results.log`.

`banana-deploy-on-fixed-r2video` used the verified 99-file snapshot, required
grounding, and a 20-graph-second deadline. Both World State and the runner reported
ready; the runner waited 6.00 seconds for usable observations. The kernel then
reported one eligible tracked target. This verifies the scope-ordering correction
in a live deployment. Later the track split, eligibility fell to zero of two,
and the kernel stopped at tick 85 (latest feedback 84) on
`attached:pinch:finger_joint@14483334089` versus `voxel_1259123`, signed/swept
distance -0.023536 m. Task success was false, status 6. No safety check was relaxed.
Readiness took 32.012 wall seconds; total harness time was 59.437 seconds.
The failure-topic observer used the explicit message type in this run.
`action.mp4` omits 9.9 seconds using dispatch/feedback wall stamps; full `cams.mp4`
and the offset provenance remain available. Grounding success is not task success.

`banana-policy-seed1-300` is a new plain benchmark, not a trim of the older run.
It executed exactly 300 steps (20 simulated seconds) with seed 1 and the pinned
policy. Success was false, reward 0; total harness time 38.123 wall seconds.
Mean policy latency was 24.02 ms, maximum 1080.40 ms. Its complete video is
`video/isaac_arena_droid_banana_in_bowl.mp4` in the named artifact directory.

A subsequent read-only branch/worktree search found no live calibrated force
witness publisher in the inspected local or Spark trees, including Spark's
`origin/feat/373-force-contact-layer` at `42d22066`. The real OpenArm HAL forwards
upstream JointState effort if supplied and otherwise leaves it empty; its manifest
marks torque sensors absent (`377275d3`). Deploy sim publishes actuator effort
(MuJoCo `qfrc_actuator`, Isaac measured joint effort; `8dcd7b5b7`). These values
are not calibrated external contact-force residuals. The MuJoCo
`probe_contact_force()` helper has tests but no runtime caller. The existing
kernel consumes embedded `ContactForceWitness` evidence, so missing continuous
production remains an implementation gap, not permission to infer contact force
from actuator commands. The search did not fetch or mutate remote branches.


## External restock deployment with current audit fixes

`restock-deploy-on-fixed-r1video` used tracking ON, required grounding, the
99-file source snapshot, external scene/assets and pinned policy. The goal had
a 30-graph-second deadline. Readiness took 66.013 wall seconds; total harness time
was 110.085 seconds. Both tracker and runner reported grounding ready (runner
wait 6.70 seconds); one of three tracks became kernel-eligible. The goal stopped
on a collision E-stop waiting for action-group tick 105 (latest feedback 104):
`openarm_left_link7` versus `voxel_285076`, signed distance -0.00199577 m,
swept distance -0.00761643 m, workspace denial 5 (occupied cell not explained by workspace evidence). Link7
is the manifest gripper carrier, corresponding to URDF `openarm_left_ee_base_link`,
and belongs to the derived workspace hand-link set. The earlier description
that attributed this stop to a disallowed wrist link was incorrect. Task success was false, status 6; first feedback took
26.23 wall seconds and the goal lasted 35.9 wall seconds. No collision check
was relaxed. The declared static map contains 93,898 kernel cell centers and
14,979,607 input surface points; startup overhead is separate from policy time.

The full three-camera recording, frame wall stamps, evaluator-only ground truth,
collision snapshots, exact config/command and logs remain in the external artifact
directory. `action.mp4` starts half a second before first feedback according to
recorded wall stamps; the full recording is retained. The reasoner was disabled,
so this run does not establish automatic recovery. It also does not establish
calibrated contact-force permission or declaration-based geometry tightening.


`restock-policy-seed103-900` repeated the plain benchmark with an explicit
900-step cap, seed 103 and the same pinned local weights. It exhausted all
900 steps without success (reward 0). Total harness time was 45.002 wall seconds;
mean policy latency 7.21 ms, maximum 919.85 ms. The complete
`video/restock-plain-scene.mp4` was shared. This establishes a canonical-budget
failure for this episode, separately from the prior extended-budget success.


Independent scope review found two defensive edges after the live ordering fix:
a duplicate active goal ID with changed queries retained the old goal, and a track
with no matching live query could retain a previous scope key. Both now close or
clear the obsolete scope. Regressions cover query replacement, active empty-query
retirement and unmatched tracks. Focused session tests: 23 passed; Ruff and strict
World State/test-file mypy passed. Exact kernel keys remain unchanged. The private
ADR/hazard update preceded the edits. The subsequent 99-file snapshot is
`source-manifest-scope-hardened.json`; earlier run manifests remain immutable.


## Orange ON after scope hardening

`orange-deploy-on-fixed-r5video` used the verified hardened 99-file snapshot,
tracking ON, required grounding, pinned SmolVLA and a 60-graph-second goal deadline.
Readiness took 28.008 wall seconds; total harness time was 51.849 seconds. Tracker
and runner reported ready (runner wait 1.03 seconds). The goal lasted 16.4 wall
seconds, first feedback 6.65 seconds. The kernel stopped waiting for action tick
218 (latest feedback 217): gripper versus `voxel_157861`, signed and swept
distance -0.00264462 m, hand speed 1.28486 m/s, workspace denial 2. That exceeds
the configured 0.5 m/s workspace-contact budget. Success was false, status 6.
No safety checks or contact budgets changed. The action clip and full recording
were shared with the maintainer, with wall-stamp clipping provenance retained.

Final focused local batch after scope hardening: 90 passed in 27.37 seconds
(surface model/runtime, catalog declaration/runtime, tracker session/grounding).
Native PCL was loaded from the real local build; no skips occurred in this batch.
The methods inventory coverage check and whitespace check passed.


Restock collision attribution review: voxel 285076 is occupied at
[0.1725, 0.0975, -0.4725] m in `openarm_base` (15 mm grid). At the nearby
14.483334-s evaluator sample its volume is adjacent to/overlaps the approximate
bounds of a neighboring carton, while the target stock object is well away.
This is a plausible dynamic source, not a certified attribution; the binary grid
has no backing identity. No nearby declared static box was identified, and mesh
fixture contribution was not resolved. No exact link7/contact witness is available
in the saved GT. Last-step PhysX contacts cannot establish absence of a predicted
collision. Keep the stop and the ownership gap explicit.


Banana post-stop replay: the stored fitted box at sim 11.416 s has center
[0.5154, -0.1325, 0.0494] m and half extents [0.1380, 0.0980, 0.0494] m. The
track remained confirmed through 14.083 s and split at 14.35 s. Applying that
box to the saved later grid reproduces `map_split` with components [101, 14, 1].
Most cells in the major component are below the later banana bounds; this suggests
support/background or stale occupancy, not proof of a second banana. The exact
transition grid was not captured. The scene supplies no authoritative static cells.
Late binding refused the split candidate on occupied-coverage grounds, then used
pinch fallback. Neither replay nor evaluator contact proves a safe rigid-carry
identity. No threshold was changed. Independently declared table geometry is the
next evidence check; it must actually cover an entire offending component before
it can explain this split. A second non-static body must still trigger refusal.


Authoritative Maple USD follow-up rules out the proposed static-table workaround.
Standalone USD composition identifies an enabled non-kinematic rigid body, which
existing static-collider extraction intentionally excludes. The tabletop reaches
z=0.003001 m, whereas the replayed offending component has cell centers at
z=0.0225..0.0975 m. Declaring this movable table static would be incorrect and
would not cover the entire component. No static table cells were added. The
remaining problem is identifying stale/mixed occupancy versus a separate body,
not an established missing static mask.

## Orange with source-stamped diagnostic capture

`orange-deploy-on-sensor-r6video` used the same hardened source snapshot and
configuration, plus a ROS bag of depth/CameraInfo, raw and self-filtered clouds,
joint states, TF, clock, tracked-object state, voxel updates and safety failures.
Readiness took 28.008 wall seconds; total time 55.059 seconds. Grounding passed
in 0.82 seconds. The goal lasted 18.8 wall seconds, first feedback 5.13 seconds,
and stopped waiting for tick 319 (latest feedback 318): untracked pinch fallback
versus `voxel_128123`, signed distance -0.00653839 m, swept distance -0.0477025 m,
workspace denial 4. Success was false, status 6. This is a different collision
from r5, so its sensor capture does not retroactively establish r5 cell ownership.
The action clip was shared; full recording and the bag remain available on Spark.


The diagnostic bag is MCAP, 1.0 GiB / 43.328 wall seconds, 7,753 messages:
75 depth frames, camera infos and raw clouds; 66 filtered clouds; 277 voxel grids;
79 tracked sets; 290 TF batches and one TF-static batch. The selected World State
topic had no recorded messages, so only the listed tracked/voxel topics establish
that part of the capture. Recorder shutdown flushed the cache normally.


R6 sensor trace: voxel 128123 decodes to [39, 29, 25], center
[0.0825, -0.1425, 0.0375] m in `base`. It is already occupied in the first
recorded grid (header 1.216666730 s, source 1.133333333 s) and stays occupied
through the final grid (header 8.500000443 s, source 8.466666666 s). This does
not establish its first-ever insertion. Reprojected raw and filtered endpoints
both support it: 50 each at source 1.133333333 s and 99 each at 8.333333333 /
8.466666666 s. The endpoints form a plane at z approximately 0.03022 m. This
supports a persistent observed world surface, likely counter/table, rather than
an unsupported stale cell; exact collider attribution remains unavailable.
At attachment, the binder explicitly reports candidates `trk:3` and `trk:4`,
then outcome `ambiguous` with pinch fallback. Keep the geometry refusal; neither
the ambiguity nor the observed surface justifies proximity-based map clearing.
This result does not retroactively identify the different R5 collision cell.


R6 binding review identifies two real orange tracks, not a demonstrated duplicate
track of one object. Their fitted boxes are only 3.2 mm apart along y; the
one-voxel-expanded pinch test accepts both. One fit is older (last camera fit
4.833 s versus ATTACH at 8.450 s), while the other was fit at 6.933 s. Evaluator
contacts show the jaw and gripper touching Orange003, but runtime evidence does
not uniquely distinguish it from the adjacent fitted box. `no_hold_read` simply
prevents late binding; ordinary ATTACH then correctly returns ambiguity. No
containment or freshness threshold was relaxed.

Query matching is label/role scoped, not certified instance association. Here the
instruction's rubric intentionally covers all three oranges, so the class-wide
query is appropriate; the legacy first-object declaration key does not prove a
named instance identity. Explicit named-object/SKU tasks still need that separate
association contract before declaration-based tightening.

Final documentation correction updates both shipped scene comments, launch help
and schema descriptions to describe guarded tracked-only clearing rather than
legacy untracked boxes. Both real DeployScene YAML files validate. Generated
schemas have no drift; methods coverage, whitespace, Ruff lint and formatting
checks pass. These final edits change descriptions/formatting only; rollout
source manifests remain preserved at their executed revisions.

The R6 `diagnostics/` directory preserves the analyzer, exact stdout, structured
summary and transform/indexing method in both local and Spark artifact copies;
all four file hashes match. The 1 GiB sensor bag remains on Spark.


## Follow-up: runtime names and collision causes

The next banana policy run uses **600 steps (40 simulated seconds at 15 Hz)**,
as requested after review. The earlier 300-step result remains historical; this
is an evaluation override, not a change to the canonical benchmark scene budget.
A deploy comparison must separately report grounding time and applied action ticks.

The robot self-model now includes manifest joint names, simulator aliases and
parent/child links. It is seeded independently of skill palette search paths.
Collision prompt summaries retain both actors, signed distance and measured versus
predicted check phase without the former 120-character whole-record truncation.
Joint vectors remain on the ROS wire for replay and are summarized by count in
the prompt. These diagnostic changes do not establish physical collision truth
or relax a contact decision. User-requested gripper-base/carton contact needs
object-scoped evidence; existing workspace permission cannot identify a carton
by itself. No new simulator run was performed for this diagnostic change.

This follow-up's Spark status/SSH checks timed out; Tailscale reported the peer
at `100.109.35.76` offline (last seen `2026-10-19T17:00:00.1Z`). No additional
simulator was launched. The next 600-step banana run remains unexecuted.

Diagnostic follow-up verification: 104 Python reasoner/self-model/schema-fuzz
checks pass; 22 generated-ROS-IDL constant-mirror checks pass after sourcing the
current 418 overlay (the older main-worktree underlay lacks KIND_OBJECT_QUERY);
10 live ROS reasoner grounding checks pass, including configured robot names in
context without skill search paths. Two real kernel lifecycle publisher tests
pass and preserve captured predictive and attached-collision payloads as schema
fixtures. The attached payload fixture carries pinch-fallback ambiguity through
the existing FailureTrigger topic. These checks exercise ROS publication and
reasoner consumption separately; no new Isaac rollout or automatic recovery
claim is made.

## 2026-10-20: Spark restored, banana benchmark and deploy comparison

Spark became reachable again. The isolated `openral-418-eval-20261019` checkout
was restored at `85a8c812852d064b516c6e8e2945445a9ecfa071`, with 111 changed source
files recorded by SHA-256 in `/tmp/openral-418-spark-eval-20261019/source-manifest-20261020.json`.
Artifacts below live under that same `/tmp` directory on Spark and locally.

- Plain benchmark `banana-policy-seed1-600-20261020-r2`: budget 600 steps, seed 1;
  benchmark predicate reported success and ended at step 276 (18.4 simulated
  seconds, 45.556 wall seconds including startup). This is not a 600-step
  retention test. Video: `video/isaac_arena_droid_banana_in_bowl.mp4`.
- The first plain attempt failed before rollout because `accelerate` was absent;
  no video frames were produced. The complete evaluation environment now uses
  `just sync --group dev --group sim --group isaacsim --group omdet --group sam2`.
- Deploy `banana-deploy-on-20261020-r1video`: no action chunk was applied. Detector
  initialization lacked `timm`; grounding timed out. The detector incorrectly
  represented the exception as an empty detection, motivating explicit
  `LocateInView.failure_reason` propagation. Video: `cams.mp4` (idle startup).
  A short native test ran in the same ROS domain before deployment readiness;
  early test diagnostics in this bag must not be attributed to the rollout.
- Deploy `banana-deploy-on-20261020-r2video`: grounding completed; first chunk
  arrived 10.34 wall seconds after dispatch. Chunks through 43 applied; chunk 44
  timed out after the kernel rejected stale world voxels (data age 1533 ms,
  budget 1500 ms). This was not a contact rejection. Video: `action.mp4`, with
  full startup recording in `cams.mp4`. No native test shared its ROS domain.

The R2 ROS bag contains 408 joint messages and 222 unique timestamps. Two-tick
intervals were 133333340–341 ns; the filter's nominal 15 Hz truncation derived
133333332 ns as its two-period bound. This incorrectly rejects valid intervals.
The proposed correction uses the actual validated simulator tick period and
rounds the one/two-period nanosecond boundaries upward; it does not increase the
world-data freshness budget. Both detector failure reporting and timebase
corrections require a recorded deploy rerun before claiming resolution.

R2 failure payloads were decoded from the ROS bag and validated with the real
`FailureEvidence` Pydantic union. The final controller evidence contained
`field=voxel_stale joint=65535 value=0 limit=0`; 65535 is the internal no-joint
sentinel, not a robot joint. Removing that fictitious joint identity and naming
real configured joints is a diagnostic correction, with no safety decision change.
These direct-dispatch trials disable the reasoner and do not verify autonomous
replanning. They do not establish calibrated torque contact permission or exact
carton-to-robot collision attribution. Safety-WG review remains required.

### Timing/error-reporting follow-up verification and recorded reruns

The final candidate source snapshot is `source-manifest-20261020-r3.json`
(120 SHA-256 file entries, verified on Spark). Rebuilding messages, kernel,
world state and bridge took 2 min 7 s. Spark checks passed: 22 pure filter tests,
17 ROS filter-node tests, 12 launch configuration tests, and two kernel
publisher regressions. The corresponding earlier local node crash was caused by
old generated message libraries in `.m-build/install`; the consistent Spark
overlay passed the same real node suite. Local methods inventory check passed.
The typed detector-failure change additionally passed 19 tracker unit tests,
three live reasoner/object-query tests and eight detector structural tests.

- `banana-deploy-on-20261020-r3video`: first chunk 9.66 wall seconds after dispatch;
  118 chunks applied, then a world collision at `left_distal_phalanx`, cell
  `voxel_1258815`, signed distance -0.0148083 m. Cause: `cell_unexplained`, with
  workspace permission active. Video: `action.mp4`; full recording `cams.mp4`.
  This run used the render underlay's old installed launch package, so it does
  **not** verify the new HAL cadence handoff. That package was subsequently
  rebuilt into the isolated evaluation overlay.
- `banana-deploy-on-20261020-r4video`: the isolated installed launch package was
  verified before launch. The HAL logged admission of its declared simulation
  period, 0.0666666666667 s. First chunk 11.93 wall seconds after dispatch;
  117 chunks applied, then the same category of world collision, now cell
  `voxel_1258971`, signed distance -0.00479894 m. No stale-world final stop.
  Video: `action.mp4`; full recording `cams.mp4`. The reported period remains
  nominal despite the measured clock quantization; producer timing consistency
  still needs investigation before claiming complete resolution.

Both collision payloads validate against the actual `FailureEvidence` schema;
configured joint names, actor, signed distance, sweep distance and workspace
refusal reason are present. The monitor's collision snapshot contains no attached
payload; the later HAL stop snapshot in R4 already contains a pinch-fallback
attachment. That race must not be read as proof that no grasp occurred. R4's
PhysX report lists both distal fingers contacting the banana at small positive
separations, with no reported penetration; R3 lists no robot/world contacts.
Neither report identifies the source of the specific rejected voxel. R3's rejected
cell center is [0.4125, -0.0375, 0.1125] m in `panda_link0` at 15 mm resolution;
this lies above the last recorded banana bounds. This is evidence for checking
residual robot occupancy, not proof that the contact may safely be ignored.
No collision allowance, force threshold or freshness deadline was relaxed.

Read-only follow-up distinguishes the two stops. R3's cell is near the left
finger's collision geometry while the banana is still on the table; heavy
joint-gap/mask drops make a retained robot return plausible, but the recorded bag
cannot attribute it. In R4, the cell is inside the banana's nearby ground-truth
AABB, while the track still uses its old table-level box from sim time 13.15 s.
The attachment is ambiguous `pinch_fallback` / `untracked_grasp`; the self-filter
explicitly retains its returns. This is consistent with refusing unexplained
banana contact rather than granting contact from an unproven object identity.

The next diagnostic recording adds raw depth and camera intrinsics, raw and
filtered clouds, robot mask, world voxels and candidate actions to the existing
TF/joint/tracker/failure capture. Earlier bags lack those sensor/grid streams and
cannot establish which raw point first occupied the rejected cell.

### PhysX clock-cadence correction and full sensor capture

Inspection of Spark's installed Isaac `SimulationContext` confirms that
`current_time` accumulates the physics callback's `step_size`, whereas
`get_physics_dt()` reports the configured reciprocal step rate. Scene reset now
reports the observed callback cadence times the unchanged substep count; the
Isaac client refreshes that value before HAL validation. Its reset hook first
caches action dimensions so a later ping cannot overwrite observed cadence.
Missing, null, non-finite or zero reset cadence fails closed. No physics steps or
warmup counts changed. The R5 source manifest contains 128 files.

The complete Spark wire/scene checks passed: **111 passed, eight skipped**. The
eight skips require `pxr` in that test interpreter; the real Isaac sidecar uses
its installed USD runtime. Earlier attempts exposed incomplete test reset
fixtures and a socket-bind readiness race; those fixtures now carry the new
metadata and wait for the network boundary to bind before client connection.
Ruff, production strict mypy, compilation and methods inventory checks passed;
full repository CI and automatic reasoner recovery were not run.

- `banana-deploy-on-20261020-r5video`: actual period admission is now
  **0.0666666701436 s**. No actions applied: grounding exceeded the eight-second
  evaluation wait and resolved after goal rejection. Video: `cams.mp4` (idle
  grounding). The expanded sensor bag is about 796 MB. Offline joint stamps
  contain two-tick gaps of 133333340–341 ns, now within the exact upward-rounded
  bound 133333341 ns. The clock correction is active; this trial is not a policy
  rollout result.
- `banana-deploy-on-20261020-r6video`: evaluation-only grounding wait increased
  to 15 s, still requiring grounding before motion. Grounding completed after
  8.93 wall seconds at the runner gate; first chunk at 14.96 s after dispatch.
  87 chunks applied, then tick 88 was stopped for a world collision at
  `left_distal_phalanx`, `voxel_1258819`, -0.010753 m, with
  `workspace_denial_reason=cell_unexplained`. The recorded configured joint names
  and collision cause validate against `FailureEvidence`. Video: `action.mp4`;
  full recording: `cams.mp4`. The full sensor/grid ROS bag stays on Spark;
  smaller artifacts and videos are copied locally.

R6 PhysX reports both distal fingers and proximal fingers in contact with the
banana, with small positive separation/contact-offset values and no reported
penetration. This does not identify the exact rejected cell. The kernel contact
rule, freshness budget and unknown-object refusal remain unchanged. The run
therefore verifies the timing/reporting path, not successful production-safe
placement. R6 sensor forensics now traces the first recorded occupancy transition for that
cell to grid header 17.550000915 s, source stamp 17.483333333 s. The same-stamped
raw and self-filtered clouds contain exactly the same 14 XYZ points in the cell;
all 14 lie inside the nearby ground-truth banana AABB. The nearest joint state
is only 0.9 microseconds from the cloud stamp. This supports retained depth
returns in the banana volume, consistent with the logged ambiguous
`pinch_fallback` attachment and deliberately retained payload returns. It does
not establish a unique tracked-object binding. The mask has no exact-stamp
sample (nearest 17.2167 and 17.75 s), so mask-level pixel attribution is not
claimed. Keep the conservative denial. Reproducible analyzer, point arrays and
`r6-cell-forensics-summary.json` are saved beside the R6 local video; the full
bag remains on Spark.


### Query observation availability diagnostics

The R5 zero-attempt query failure exposed a separate reporting defect: missing
camera pose or robot mask prevented detection, but the query had no associated
observation and escalated as `object_not_found`. Unusable frames now record their
bounded camera/stamp and exact refusal against pending queries. No observation
falls back to `no_usable_observation`; a completed negative locate still reports
`object_not_found`. The reasoner explicitly states that unavailable visibility
does not verify absence. Geometry, query eligibility and safety decisions are
unchanged. ADR-0119 and the private hazard log record this diagnostic change.

Focused grounding/recovery tests: **32 passed**. Existing live ROS query-to-
reasoner tests: **2 passed**, isolated on domain 219. These checks do not establish
automatic physical recovery or successful production-safe placement. This final
Python diagnostic slice was tested locally after R6; it was not part of the R6
source snapshot or a new Isaac rollout.


## 2026-10-20: banana observation and binding repairs

R6 binding review proves the immediate failure was no geometric candidate, not
multiple candidates: one banana track retained its 11.15 s visual fit while map
confirmation advanced to 16.817 s. At attach 17.483334245 s its old box did not
contain the hold pinch centre. The unresolved bowl query could monopolize the
scheduler ahead of the confirmed banana refresh; individual requests were not
logged in R6, so that scheduling explanation is supported but not a complete
request-by-request reconstruction.

The follow-through adds exact-source-stamp LocateInView requests/responses,
bounded detector image history, query/refresh fairness, separate fit freshness
for late binding, and bounded exact-stamp TF retry for robot masks. Issued
tracker requests now identify source stamp, service, query and track. The optical
TF producer also uses measured backend state time rather than a lagging ROS clock
callback; it does not retag a current pose as a delayed image capture.

The TF regression failed before the producer change (TF stamp zero while backend
state was 2 s); all 41 wire tests then passed on Spark. Preliminary combined
Spark checks passed 90 tests, and its rebuilt native filter passed all 21 ROS
node cases. Local native geometry passed 22 tests. These preliminary checks used
`source-manifest-20261020-r7-preliminary.json` (135 files). Subsequent normal-bind
freshness and segmentation-header checks require their final snapshot and
regressions before rollout acceptance.

No new Isaac rollout has run for these repairs: the shared-compute checks show
Jonathan's active policy/evaluation GPU work, with available RAM below the
existing 80 GiB evaluation gate. No process owned by another user was altered.
Videos for earlier runs remain linked above; there is no new simulation video.

### Final source checks before the next recorded rollout

Normal-bind freshness and late-bind exclusion passed the full 198-test tracker
suite; the strengthened RELEASED regrasp fixture subsequently passed its 47-test
focused suite. Exact segment-mask camera/stamp/frame checks and generic failure
propagation passed 53 focused unit tests plus both live ROS query-to-reasoner
cases on isolated domain 220. Strict ROS-package mypy and method coverage passed.

The latest capacity check still shows Jonathan's policy server and three Isaac
processes, with GPU utilization at 95%. Available RAM has recovered to around
80 GiB, but the existing maximum-process gate still prevents our rollout. No new
simulation has run, and no new video is claimed.

Final Spark source verification matched all **139 file hashes** in
`source-manifest-20261020-r7.json`. The combined tracker, active-search, source-frame
and HAL wire suite passed **267 tests in 89.10 s** (`r7-final-cpu.log`). The native
filter's frozen implementation had already passed all 21 real ROS node cases.
Spark's live query repetition passed one case and skipped the other because the
evaluation venv lacks `pxr.Usd`; both cases passed locally as reported above.
`r7-final-live-query.log` records that limitation. These result paragraphs were
appended after the frozen source snapshot; runtime source was not changed.

The final capacity probe saw three existing Isaac processes and 75.5 GiB available
RAM. `banana_deploy_r7_trial.py` is prepared against the final manifest, but has
not launched. Production-safe banana placement and automatic physical recovery
remain unverified.

## 2026-10-20: recorded R7–R9 follow-through

Spark became idle (0% GPU, no Isaac jobs, 106 GiB available RAM). With user
authorization, a one-hour compute reservation was taken. Every attempt recorded
`cams.mp4` under its immutable `banana-deploy-on-20261020-r{7,8,9}video` folder.

- **R7:** stopped during startup, no goal dispatched. The source snapshot was
  updated, but ROS launched the old detector executable directly from the render
  underlay. Its replies omitted `source_stamp_ns`; the tracker refused them.
  Rebuilt `openral_perception_ros` into the evaluation overlay and verified both
  detector and segmenter executable hashes against their source files.
- **R8:** policy preload completed; required grounding failed, **zero applied
  actions**. The locator stayed INACTIVE after its configure response timed out.
  The typed action result was `ROSPerceptionStale: grounding_incomplete`. Switched
  locator startup to the existing bounded one-shot state-poll lifecycle helper;
  it exits at ACTIVE and does not override later reasoner deactivation. Governance
  was recorded before the launch change. Seven lifecycle helper tests passed.
- **R9:** rebuilt `openral_rskill_ros`; installed launch/source SHA-256 hashes
  matched. Snapshot `source-manifest-20261020-r9.json` contains 139 files. Locator,
  segmenter and tracker reported ACTIVE. Required grounding completed at graph
  time about 9.6 s. The policy applied **73 actions**, then the kernel stopped
  before tick 74: `left_distal_phalanx` versus `voxel_1606131`, penetration
  0.0193692 m, `cell_center_outside_workspace`. Result was an explicit
  `safety_estop`, not successful banana placement.

R9's first offending occupied grid has header stamp 14216667408 ns and source
stamp 14216666666 ns. Its 15 mm cell centre is (0.3525, 0.0375, 0.3375) m. The
same-stamped raw cloud contains 16 points in that cell; the self-filter removes
13 but retains three identical XYZ records. Ground truth places the distal
finger nearby and the banana near its initial tabletop pose, far below the cell.
This supports a robot-return leakage investigation; it is not yet proof of the
exact geometry/kinematics defect. No workspace expansion, map clearing, collision
bypass or padding increase was used. `r9-cell-forensics.json` records the cloud
points, grid and joint-state evidence; the full sensor bag stays on Spark.

R9 startup and exact-frame query transport now work, but the approach has **not**
passed production-safe placement. Next investigation is the distal-finger
self-filter geometry/pose at the offending capture, followed by another recorded
run. The earlier R6 stale binding failure remains a distinct downstream scenario
that R9 did not reach.

## 2026-10-20: R9 cause reproduced, R10 bounded-state repair

The production-linked R9 regression reproduces exactly 3/16 robot returns kept
by the extrapolated history available at cloud receipt; adding the matching
state that arrived 5.1 ms later removes all 16. The actual left_bar collision mesh
is within 0.055–0.151 mm of the three points at that state. Existing 20 mm padding
and ideal mimic geometry suffice; no larger removal envelope was introduced.

`RobotSelfFilter` now waits for history to cover each cloud/mask capture stamp.
Pending clouds are bounded to two buffers of at most 32 MiB each; the existing
skew deadline and shared 5 ms retry timer bound waiting. Expiry/overflow refuse
the frame. Interpolation gap/skew checks remain. Both forward and backward
extrapolation of removal poses are excluded at the node boundary.

Local and Spark checks passed **23 geometry tests** and **26 live ROS node tests**
(including five new arrival/interpolation/expiry/overflow/backward-state cases).
Spark verified all 140 source hashes in `source-manifest-20261020-r10.json`, built
the native bridge in 44.7 s, and repeated both suites successfully.

R10 recorded `banana-deploy-on-20261020-r10video/cams.mp4` with tracking ON,
required grounding and kernel checks enabled. It passed the old action-74
failure and applied **245 actions**, stopping before tick 246 on
`left_distal_phalanx`/`voxel_1143148`, penetration 0.0127772 m, denial
`cell_unexplained`. Grounding completed; no held-object binding occurred.
At stop, the kernel reports zero tracked objects (`n=0 of=0`). The legacy
pre-grasp producer is intentionally disabled in tracker ON mode; its diagnostic
refusals do not explain this failure.

The new cell centre is (0.4875, -0.0525, 0.0375) m. It was already occupied in
the first recorded grid (source 1.95 s), before motion. All **11 raw points**
in that cell are preserved identically by the filter and lie inside the initial
evaluation-only banana bounds. This is distinct from the R9 moving-finger ghost.
It does not authorize blanket world clearing or contact exemption from ground truth.

R10 was capacity-gated at 104.9 GiB available RAM with one existing Jonathan
`robot_diff.py` Isaac process; our own was the second. Runtime executable hashes
and process context are saved in `runtime-provenance.txt`. Timing is not an
isolated performance benchmark. Full sensor bags remain on Spark; local copies
include videos, small logs, tracks and the R9/R10 forensic summaries.

No successful production-safe placement or physical reasoner recovery is
claimed. The reasoner was disabled for these direct-dispatch validation runs.

### R10 tracker publication gap: corrected diagnosis and bounded repair

The bag has a tracker arrival at wall 1792497687.4207 s, followed by
`world_state_fast` at 1792497689.0476 s with `tracked_objects_valid=false` and
zero objects: a 1.627 s gap exceeds the existing 1.5 s receipt deadline. At
1792497690.1019 s tracking resumes with the same two confirmed objects. The
kernel's zero-object log and collision fall within that gap. Grounded tracker
boxes already replace the legacy grasp region in ON mode.

The backup fill timer had its own callback group but used ROS time. Jazzy's
/clock subscription shares the expensive fit callback group, freezing the
node's ROS clock during that work. A real ROS reproduction held that group
for 1.2 wall seconds: its ROS timer stalled 1.22 s while an otherwise identical
steady timer continued. The exact expensive callback in R10 was not profiled.

Only the backup timer now uses STEADY_TIME. Cached tracked messages retain all
stamps/revisions; attachment headers still use node ROS time and can therefore
remain old and be conservatively refused. Existing five-second silence/joint
bounds and kernel deadlines are unchanged. The real lifecycle regression
first grounds the SO-101 orange fixture, freezes /clock, requires identical
cached sets, then proves publication stops and world state expires. It fails
before the fix and passes after it. Both lifecycle tests plus all six cache
unit tests pass locally (8 total). The service-boundary fixture now respects
the optical prompt frame and exact requested mask timestamp.

### R11 recorded follow-through

Spark repeated all eight targeted tests successfully and verified 142 source
hashes. Runtime module SHA-256 is
`444662713a41305514db80ecc59eadf46018d2456930e0d277324e018a667162`.
R11 startup took 38.008 wall seconds; first chunk 11.37 s after dispatch.
It applied **272 actions** and refused action 273 at graph 28.483334819 s:
`left_distal_phalanx` / `voxel_1096930`, penetration into the voxel 8.486 mm,
`cell_unexplained`. There was no held-object binding or successful placement.
Video: `banana-deploy-on-20261020-r11video/cams.mp4`.

Bag audit finds **342/342 valid world-state tracked sets** after two-object
grounding through the stop, with zero invalid intervals. The largest tracker
receive gap is 1.401 s (inside the unchanged 1.5 s deadline); equal source stamp
and revision demonstrate cached republication. This validates the repaired path
in this run, not a guaranteed scheduling latency bound under arbitrary load.

The new cell is (0.3375, -0.0525, 0.0075) m, index (98,72,47), resolution 15 mm.
It is occupied from the first recorded grid, source 1.75 s, and remains occupied.
Initial raw endpoints and later filtered endpoints lie at z approximately
4.4 mm, consistent with the tabletop. The latest accepted banana box puts the
cell 91.4 mm beyond its oriented XY footprint, outside the existing 50 mm support
reach; the bowl gap is 184.4 mm. This explains the refusal with a live tracker.
It is neither the R10 empty-table failure nor sufficient evidence of physical
penetration: collision against a complete 15 mm voxel is conservative.

The configured tabletop workspace is a search/contact boundary, not identity
for every occupied cell in it. The scene explicitly identifies the table as a
movable rigid body and therefore correctly excludes it from static colliders.
The next contact-model work should represent that dynamic support as an observed,
tracked workspace with explicit allowed links and measured bounded contact
witnesses. Do not increase support reach, clear table voxels, or mark a moving
support static merely to pass this rollout. Compare the contact geometry and
measured force/torque before attributing this to physical collision.

Banana last-fit geometry is 11.53 s old at stop, though map confirmation is
current. This is separate from publication freshness and deserves follow-up on
refit refusals and moving-object detection. No new measurements were fabricated
by the fill. Reasoner recovery remains untested here (direct dispatch).

Artifacts: `r11-tracker-stream-audit.json`, `cell-1096930-forensics.log`, video,
source manifest, runtime provenance and bag in the R11 directory; full bag stays
on Spark. The independent ROS scheduling reproduction is saved as
`heartbeat-clock-reproduction.py` and `.log` in the parent artifact directory.
Local relay-expiry test also passes; Ruff, strict mypy, method coverage and
`git diff --check` pass for this repair. Full repository CI was not rerun.

### R11 cell persistence across the full recording

Follow-up read-only bag analysis (`r11_cell_persistence.py`, output
`banana-deploy-on-20261020-r11video/cell-persistence.json`) counted every recorded
cloud and grid, rather than only the initial and final windows:

- 225/225 raw clouds contain exactly 13 points in voxel 1096930.
- 224/224 self-filtered clouds contain 1–13 points there; none are empty.
- 546/546 grids mark it occupied, from source 1.75 s through 28.75 s.
- Raw endpoint z spans 0.004393935–0.004412472 m over the whole recording
  (less than 0.019 mm total variation); filtered endpoints match this surface.

Thus this recording provides persistent camera support for the occupied cell;
it is not merely a transient hit retained in octomap. The observed planar patch
is consistent with tabletop, but exact USD render/collision mesh alignment has
not yet been measured. The 15 mm voxel spans z=0..15 mm while returns lie near
4.4 mm, adding about 10.6 mm above the measured surface. A finger/voxel overlap
alone therefore cannot establish physical contact. Next compare capture-time
finger collision geometry against the depth surface and simulator contact/mesh
geometry, including calibration and collision lowering. No workspace widening,
voxel removal, new simulation or contact permission was introduced.

### R11 finger-to-table clearance investigation (no new simulation)

The existing stop log **does** include a PhysX contact-band record. At the
post-stop snapshot (sim 28.550001 s), left_distal_phalanx vs
`/World/environment/table/table_01/top` has three proximity points, minimum
separation **+3.352613 mm**, zero impulse and zero reported penetration. Positive
separation is a near miss, not touching. This is the last physics substep only,
66.7 ms after the collision evidence source stamp; it does not simulate the
refused command. Authored contact/rest offsets are null (PhysX-derived).

Independently reconstruct the URDF collision STL and the manifest tight hull
using the existing mimic-aware `link_transform`. The stop snapshot's measured
joint configuration puts the link origin within approximately 0.25 micrometres
of the sidecar's independently recorded link origin at sim 28.550001489 s.
This validates FK/frame/normalised-jaw alignment at that pose. It does not prove
that PhysX's cooked contact shape equals the original STL or resolve its default
offsets; do not equate the two clearance values.

At the **refused action 273** configuration (collision evidence joint vector):

| geometry | minimum z | vertical clearance above maximum measured cell surface |
| --- | --- | --- |
| source URDF collision STL | 6.874497 mm | **2.462025 mm** |
| manifest tight hull | 6.744709 mm | **2.332236 mm** |
| coarse outer box | 0.244623 mm | below the measured surface; it is not the tight hull |

The STL's lowest vertex is (0.341499762, -0.051245988, 0.006874497) m, inside the
implicated cell's XY footprint. The measured height bound is 4.412472 mm.
The voxel instead occupies z=0..15 mm, extending approximately 10.59 mm above
that observed patch. This explains how kernel voxel overlap of 8.486 mm can
coexist with positive source-mesh/surface clearance. No additional world margin
was needed to cause the overlap.

A 101-sample joint-linear path from the recorded measured state at source
28.483334819 s to the refused joint command has minimum STL clearance 2.462025 mm
and hull clearance 2.332236 mm, both at the endpoint. This sampled diagnostic is
not a certified continuous sweep, a whole-table plane measurement, or a PhysX
rollout of the refused action. The measured configuration starts at 10.650442 mm
STL clearance relative to the same local height bound.

Conclusion: evidence supports a conservative voxel/surface mismatch for this
specific refusal, not an actual measured finger strike. Keep the real tabletop
occupancy and existing contact checks. Any refinement must preserve bounded
surface/pose uncertainty and refuse on unsupported geometry; a 2.46 mm inferred
clearance is too small to treat as a real-hardware safety certificate. No blanket
workspace allowance, occupancy deletion, new simulation, or production change.

Reproducible local artifacts in the R11 directory: extracted
`estop_ground_truth_snapshot.json`, `estop_ground_truth_evidence.json`,
`joints-near-stop.json`, `finger-table-clearance.json` and
`finger-table-sweep.json`. Scripts `finger_table_clearance.py` and
`finger_table_sweep.py` are in the parent artifact directory. All calculations
use the active #418 manifest and cached source collision STL.
