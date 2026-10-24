# Object tracking implementation plan

Agreed with the maintainer on 2026-10-19. Applies to issue #418 on
`feat/418-object-tracking`. Research: [mapping survey](object-centric-mapping-survey.md).
This plan records intended behavior; entries below are not claims of implementation.

1. Load external USD declarations into common typed records usable by `deploy sim`
   and `deploy run` without starting Isaac. Restock assets and annotations remain
   external; only public tutorial fixtures belong in OpenRAL. Distinguish object
   shape/dimensions from search regions, workspace geometry and location priors.
2. Prefer declarations over free-form estimates. Validate units, frames, source
   revision, identity and calibrated tolerances. Contradictory surface evidence
   immediately suspends dependent tightening and contact permissions; missing
   depth and occlusion are unknown. Reobserve and explicitly revalidate before
   recovery. No arbitrary universal mismatch tolerance.
3. Preserve geometry in an object's local frame while updating observed world
   pose. Begin with rigid objects and bounded registration/model history. Preserve
   unknown extent; shape refinement is not a proof of complete geometry. Keep
   object occupancy contributions distinct from supports and other objects.
4. Reuse effort telemetry, but enforce only calibrated external effort/contact
   measurements with explicit provenance, freshness and robot/unit hard ceilings.
   USD task budgets may lower those ceilings. Missing/stale feedback cannot grant
   force-dependent contact permission. Allowed pairs are gripper–tote,
   gripper–carton and carried-carton–tote, with no wrist/other-link extension.
5. Return insufficient geometry and mismatch evidence to the reasoner through the
   existing typed failure bus. Include the cause and observations attempted. Permit
   bounded search/replanning and automatic recovery after revalidation, without
   human handoff for these failures. Exhaustion leaves manipulation stopped;
   repeated reasoner calls must not reset the recovery budget indefinitely.
6. Preserve K4 during carry; tighten before binding first. Keep kernel/E-stop reset
   requirements. Use `sim run` for the benchmark and `deploy sim` for production
   safety evaluation. Real deployment uses the same contracts, with separate
   calibration and hardware validation gates.
7. Validate changes with real schemas/public fixtures and regression tests. Apply
   the paired matrix in the survey to PickOrange, DROID and external restock,
   including declaration absence, motion, occlusion, identity ambiguity, stale
   sensing and feedback failure. Record containment, age, ownership/clearing and
   reasoner recovery as well as task success. Do not claim rollout acceptance
   before those evaluations pass.

Implementation should proceed in reviewable stages, retaining conservative behavior
where a required capability or calibrated bound is unavailable. Safety-WG review and
hazard updates remain required for safety behavior changes; maintainer agreement
on the plan is not a fabricated review result.

## First implementation stage — 2026-10-19

Implemented foundations:

- External USD catalog import, typed records, revision checks and the shared
  `deploy sim` / `deploy run` launch path. The tracker stores metadata only.
- Opt-in native PCL registration and bounded object-local observed surface
  voxels. Accepted poses relocate observations; rejected registration preserves
  the last accepted model. This observer does not change collision geometry.
- Query-scoped refusal evidence and reasoner cancellation followed by bounded
  recovery, with exhausted-task state persisted and no perception-failure human
  handoff. Invocation scoping protects replacement goals from delayed failures.
- Freshness checks for the existing carried-payload/place-target force refusal.
  Missing or invalid feedback leaves geometry checks in force; no contact
  permission or new general torque monitor is introduced.

Remaining after the first stage (progress below):

1. Associate catalog identities with observed tracks; add calibrated mismatch
   gates and explicit revalidation. No declaration currently tightens a box.
2. Evaluate known-asset pose initialization, symmetry/identity ambiguity and
   pose uncertainty. The PCL observer starts from observed surfaces only.
3. Add occupancy contribution ownership and independently observed free-space
   clearing. Object-local observations currently do not modify the world map.
4. Supply a real calibrated external-force producer, authoritative robot ceilings
   and lower task budgets, and validate the approved contact pairs in the kernel.
5. Run the paired PickOrange, DROID and external-restock evaluation matrix,
   followed by Safety-WG review and separate real-hardware validation.

Targeted tests and native builds are development evidence. They are not sim
rollout acceptance or evidence that declared dimensions safely bound an object.

### Development validation

- Tracker, native PCL surface model, query evidence/recovery, persistence and USD
  catalog batch: 234 passed. Native registration also exercises rotated and
  translated surfaces, partial views, rejected registration and buffer validation.
- CLI and deploy-scene schema batch: 191 passed; the opt-in live deployment smoke
  test was not enabled.
- Real ROS declaration lifecycle and schema batch: 93 passed.
- Real ROS query-failure integration: 2 passed, including tutorial catalog
  configuration and a grounded declaration restamped by the runner. UUID identity,
  early cancellation, stopped-at-cap state and absence of a handoff prompt are checked.
- Four launch mode/consumer checks passed, including direct-launch refusal of
  catalog metadata while tracking is off.
- Kernel collision suite: 249 passed; validator and allocation checks passed.
  All 165 lifecycle tests passed directly in 82.9 seconds with a live telemetry
  collector. The aggregate CTest lifecycle target still exceeds its 60-second
  timeout; the full CTest invocation is not green.
- Final native surface-model batch after buffer hardening: 9 passed.
- Strict mypy passed for the affected pure-Python packages and three ROS packages;
  changed Python files pass Ruff. Schema exports, method inventory, repository map
  and ROS topic graph are checked alongside the source.

The pre-existing test setup cached structlog loggers across processor replacement,
making capture assertions depend on collection order; the test-only setup now
disables that cache. Existing launch-mode expectations and heartbeat method entries
were also stale and have been corrected. These baseline check corrections are
separate from the feature behavior and should precede it when committing.

## Second implementation stage — 2026-10-19

- Catalog candidates associate only when the normalized label is unique on both
  the declared and observed sides. With explicit attributed-point calibration,
  a measured axis span greater than the maximum primitive diameter plus twice
  the point error latches a contradiction. This necessary condition needs no CAD
  pose, but cannot detect every shape mismatch or prove identity/containment.
- Contradictions immediately make dependent standing queries unresolved and
  emit causal failure evidence, including the offending span and allowed limit.
  The existing UUID-scoped cancellation/recovery path handles the failure.
  A smaller partial view, association ambiguity or lifecycle epoch change cannot
  erase the contradiction. Complete-geometry revalidation is still unavailable.
- Surface contributions have independent object ownership. Accepted rigid motion
  relocates them; optional `free_space_calibration` enables bounded depth-ray
  retirement of only local model cells. The whole uncertainty-grown voxel must
  be independently observed free at the accepted pose/source stamp. Missing,
  robot-masked, occluded, out-of-range and stale depth retains cells. Rotating
  bounded work prevents later cells from starving. Raw OctoMap remains unchanged.
- The existing additive place-force refusal uses the lower of the task budget
  and robot force ceiling. Invalid ROS ceiling overrides refuse configuration.
  This grants no contact permission and does not turn actuator effort into a
  calibrated external wrench.
- The public USD tutorial contains only an orange and generic bench metadata;
  restock assets remain external.

Remaining capabilities: certified identity/CAD pose and declaration-based geometry
tightening; complete-geometry revalidation; typed occupancy contribution transport
and independently justified clearing in the safety map; a continuous calibrated
force producer and approved pair contact enforcement. MuJoCo contact probing is
not connected to a continuous witness stream: republishing attachment state can
rearm retired support exemptions. Isaac's available joint efforts are not that
stream. These interfaces require separate handling, not attachment heartbeat reuse.

The new depth-ray and catalog tests use analytic rendered depth with the three
scene camera rigs. They are not Isaac rollouts or hardware calibration evidence.

Second-stage development validation:

- Combined tracker, native model, catalog/runtime, calibration, failure evidence
  and reasoner regression batch: 279 passed.
- CLI/deploy schemas: 191 passed, 1 opt-in live deployment test not enabled.
- Live ROS query recovery: 2 passed with the updated tracker node.
- Full native collision suite: 250 passed. Focused force gate: 9 passed;
  lifecycle invalid-ceiling check: 1 passed; allocation checks: 10 passed;
  robot-description/envelope batch: 85 passed.
- Strict type checks passed for the affected pure modules and World State ROS
  package. Ruff passed for all 34 changed/new Python files; generated schemas,
  method inventory, repository map and topic graph have no drift.

The full lifecycle suite was not repeated for this second stage; its first-stage
direct pass and aggregate CTest timeout remain recorded above. No remote simulator
rollout or hardware actuation was performed.

### Stage 2 follow-through — 2026-10-19

- A size contradiction blocks the same track ID at its recorded or any later
  lifecycle epoch until explicit revalidation; an older epoch remains stale.
- PCL inputs are sampled to an exact bounded count. If voxel coalescing or
  free-space retirement leaves too few registration cells, the next observation
  discards the earlier diagnostic surface and starts a fresh model from the
  current tracker fit pose instead of mixing unregistered points or leaving the
  model stuck. This may lose prior partial observations and grants no collision
  geometry or world-map clearing authority.
- Focused catalog, tracker and native surface tests: 46 passed. The native
  World State package built for the test run; strict mypy and Ruff passed for
  the affected package, tests and formatting.

A read-only Spark inspection found the available Isaac per-joint drive/constraint
effort path and an offline force-observer feasibility tool, but no running
calibrated external-wrench producer in the inspected #418/385 worktrees. No
remote files were changed and no GPU/simulator work was started. A reference to
any separately added torque producer is still needed before integrating it.

## Spark evaluation preparation — 2026-10-19

The subsequent authorized evaluation uses an isolated Spark worktree, virtual
environment and ROS overlay. Native messages, World State, safety kernel,
OctoMap bridge and reasoner/rSkill ROS packages build successfully. The isolated
native surface/runtime/configuration batch passes all 38 tests. After foreign
jobs finished and the capacity gate cleared, the real Isaac orange scene tests
passed both cases (14.95 s). One full plain orange policy episode then exhausted
3,600 steps (60 sim s, 72.889 wall s) without task success. A direct-dispatch
deploy attempt with tracking shadow stopped on a kernel carried-payload/world
collision at tick 88. These are initial integration evidence, not acceptance of
the full three-scene matrix or calibrated contact behavior.
The Linux ARM64 `usd-core` packaging gap requires an explicit USD interpreter
option; this changes metadata extraction only, with host validation retained.
See [Spark evaluation evidence](object-tracking-spark-evaluation-2026-10-19.md)
for the exact scope, artifacts and outstanding rollout matrix.


### Live evaluation follow-through

The post-audit Spark bridge suite passes all 160 tests. Recorded ON banana and
external restock runs both pass the tracker/runner grounding check with exact
invocation keys. Banana subsequently loses eligible identity after a split and
stops on a pinch fallback/world collision; restock stops on a non-gripper link
collision. These results verify scoped grounding and conservative refusal, not
completion of the production-safety matrix. A new plain banana run is capped at
300 steps and reports no task success. Detailed budgets, timing, source snapshots,
videos and residual occupancy ownership limits are in the evaluation report.

A broader read-only search of local/Spark branches and worktrees, including the
force-contact branch, found measured simulator actuator efforts and the tested
MuJoCo force helper but no continuous calibrated contact-force witness producer.
Real OpenArm effort depends on its upstream driver and the manifest declares no
torque sensors. Do not arm external-force contact permission from these efforts.


### Next force integration boundary

The existing MuJoCo helper can sample carried-payload to declared-target normal
loads, but continuous samples must not revise attachment geometry or refresh the
support attestation stamp. A separate typed dynamic evidence channel, merged into
the existing nested force witness, is the next shared sim/run boundary to design.
Isaac additionally needs payload/target pair contact reports, simulation timestamps
and physics step duration; current robot-contact diagnostics cannot supply that
witness. Neither an unset calibration nor missing upstream OpenArm effort may be
replaced by an invented force scale. Hardware external-force residual calibration
and independently justified contact budgets remain prerequisites.

### Follow-up after video review: gripper-base contact and current identity

The next banana policy evaluation uses a 600-step override (40 simulated seconds
at 15 Hz). The canonical benchmark scene budget is unchanged.

`openarm_left_link7` is the manifest's gripper carrier, corresponding to URDF
`openarm_left_ee_base_link`; `left_joint7` drives it. User intent permits its
contact with cartons under the agreed scoped contact rule. The restock stop's
existing denial code 5 means `cell_unexplained`, not `not_hand_link`.

Remaining implementation must authorize an exact configured link and live target
identity, not every voxel in a workspace. Binary occupancy currently loses source
identity: preserve per-cell contributors (track ID/epoch or authored collider ID,
unknown and multiple contributors) through projection before authorizing a
specific carton/tote pair. Mixed or unknown contributors retain collision checks.
Existing calibrated force witnesses remain additive vetoes; no calibrated torque
producer has been established for OpenArm. This permission is not implemented by
the diagnostic enrichment.

Orange R6 illustrates why map confirmation cannot substitute for current visual
fit/identity: two real tracks retained old camera boxes while map confirmation
advanced. A fresh, synchronized unique identity association at attach remains
required; ground truth and nearest-center selection are not runtime proofs. Until
then, ambiguity keeps the conservative pinch fallback and is reported in collision
failure evidence for reasoner/debugging.

### Joint-state cadence evidence correction

The self-filter's two-period interpolation bound now uses the actual connected
simulation tick supplied by `SimAttachedHAL.clock_authority().timestep_s` for
Isaac camera deployments. R2 emitted regular 15 Hz joint samples separated by
`133333340` ns; its nominal period produces a truncated two-period limit of
`133333332` ns, while the simulator's float32 `1/60` physics step repeated four
times gives an actual two-period limit of `133333341` ns after conservative
nanosecond rounding. The filter starts closed until the HAL atomically admits
that cadence and clears its prior joint history when the cadence changes. Real
and non-Isaac configurations retain the manifest cadence. This repairs normal
clock quantization at the existing two-period rule; it adds no timing margin.


## Remaining work audit — 2026-10-20

The R6 banana recording narrows the immediate blocker: ATTACH had zero eligible
candidates (`untracked_grasp`), not several competing banana tracks. The recorded
banana fit remains at 11.15 s while attachment occurs at 17.483334245 s. Map confirmation advanced to 16.817 s without moving that box. The late-bind
hold-centre geometry predicate then failed despite recent map confirmation. Grid
confirmation is not a fresh visual pose. The first blocking-cell occupancy is
backed by 14 identical raw/self-filtered points inside the synchronized banana
bounds; clearing it without proved identity would be unjustified.

Prioritized investigation/implementation queue (not completed acceptance):

1. **Synchronize observation inputs.** `LocateInView` serves latest RGB without a
   source-image stamp, while its tracker callback consumes the captured depth
   frame. Add a bounded source-stamp contract and reject/retry mismatches; do not
   apply a box from an unverified RGB/depth pair. This interface defect is proven
   by code, but its contribution to R6 is not established.
2. **Preserve same-stamp mask/TF availability.** R6 repeatedly drops masks because
   camera TF is one tick behind. The Isaac RGB-D publisher emits images before
   optical TF; the self-filter mask path drops a frame on unavailable transform.
   Trace capture-time versus publication-time pose and use bounded pending work
   or correctly authored fixed mounts. Never substitute latest TF for a moving
   camera or widen freshness merely to accept missing data.
3. **Refresh the manipulated object before binding.** Investigate whether
   unresolved destination queries starve confirmed-object refresh (the scheduler
   always ranks unresolved queries first), plus service timeout/cold-start effects.
   Use visual-fit age separately from map-confirmation age for binding; require
   current synchronized visual evidence for identity and expose precise candidate
   rejection reasons. A missing binding should reach bounded reasoner
   pause/reobserve/replan; retain kernel reset requirements after an E-stop.
4. **Verify banana placement and retention.** After these fixes, repeat recorded
   plain benchmark and production-safety evaluation. The 600-step benchmark
   stopped at its success predicate after 276 steps; it does not prove retention
   over 600 steps. Direct-dispatch R6 does not prove automatic recovery.
5. **Activate paired external declarations.** Banana, orange and external restock
   canonical configs do not enable `object_catalog`. Test absent/correct/wrong
   dimensions, shape-only versus location, moving targets, occlusion and ambiguity
   using the survey matrix. Restock assets/catalog remain external. Its canonical
   tracking default is off; previous ON evaluation was an explicit override.
6. **Finish geometry and contact authority.** Certified identity/CAD pose,
   declaration-based tightening, full revalidation, safety-map contributor
   transport/clearing, continuous calibrated pair force evidence and exact
   link-to-target contact authorization remain incomplete. Existing local surface
   models are diagnostic; actuator effort is not external-contact calibration.
7. **Prove recovery and deploy parity.** Run reasoner-enabled recovery and
   persistent-budget cases across all three scenes. Advanced tracking/contact/
   clearing remain sim-only; shared catalog loading alone is not `deploy run`
   safety parity. Safety-WG review and hardware calibration/validation remain due.

This audit changed documentation only. No new simulation or test was run for it;
existing R6 streams and current source were inspected. Detailed rollout artifacts
and limitations remain in the linked Spark evaluation report.


### Banana evidence-path implementation follow-through — 2026-10-20

- `LocateInView` now carries requested and actual source-image nanoseconds. Zero
  requests retain latest-image behavior for reasoner reads; tracker requests use
  the captured depth stamp and reject missing/mismatched replies. A fixed 32-frame
  detector history serves only exact requested frames or reports unavailability.
- Tracker locate replies must echo the selected camera. Every segmentation mask
  must carry the exact captured stamp and optical frame before decoding or resizing;
  service and frame failures retain their bounded cause in query evidence.
- Dynamic optical TF uses the measured backend state time rather than a lagging
  ROS clock callback, and is published before the raster. Captured images/clouds
  retain their own older stamps; current poses are not retagged as old captures.
- Robot masks retain at most two depth images per camera for exact-stamp TF retry,
  bounded by the existing skew interval. Expiry/overflow refuses the image; no
  latest-transform substitution or widened joint interpolation is introduced.
- When both unresolved queries and visible confirmed refreshes wait, scheduler
  service alternates between them. Without a waiting query, existing priorities
  apply. Issued requests log their frame and target identifiers for replay.
- Late binding tests visual-fit age, independently of map-confirmation time.
  Review also found that normal attachment could otherwise accept a stale fit
  after that late guard refused it. Normal tracked attachment now also requires
  a nonfuture fit within the existing `TrackerParams.max_age_s` bound; stale
  fits use pinch fallback and regrounding, and a late stale-fit refusal excludes
  those tracks from the subsequent normal-bind path. RELEASED fall-column
  regrasp retains its separate eligibility rule. Regression coverage includes
  fresh map confirmation with stale geometry and the late-bind fall-through.

The preceding inspection remains the explanation for the queue; these changes are
implementation evidence, not a successful production-safe banana placement.
Continuous calibrated force/contact publishing, occupancy contributor authority,
reasoner physical recovery and the three-scene paired rollout matrix remain open.

### Recorded follow-through: R9 identifies a new pre-grasp blocker

R7 exposed an evaluation installation mismatch; R8 exposed a dropped locator
configure response. Own perception/launch overlays now match source, and locators
use the existing bounded one-shot state-poll helper. R9 completed required
grounding and applied 73 actions, then stopped on a distal-finger/world cell
collision before grasp. Raw/self-filtered evidence shows 16/3 points in the
offending cell near the finger. Investigate capture-time self-filter geometry
and joint mapping before changing any envelope. The stale-binding repairs remain
unit/ROS verified but are not yet accepted by a successful physical rollout.

### R9 self-return cause reproduced — capture-time state arrival

Native real-manifest replay exactly reproduces the three surviving gripper
points using the joint history available at cloud arrival. The matching state
arrives 5.1 ms later and removes all 16 points in that cell. No mesh, mimic,
padding or declared-workspace change is needed for this reproduction.

The self-filter now retains at most two pending clouds (32 MiB each), using the
existing skew deadline and shared 5 ms retry timer. Cloud and robot-mask removal
requires joint history to cover the capture stamp; original interpolation
gap/skew checks still apply. Expiry and overflow refuse the sensor frame.
Native recorded-data and real ROS arrival-order regressions cover the repair;
the next recorded banana test must establish how far execution now progresses.

R10 validation: 23 geometry + 26 live ROS tests pass locally and on Spark. The
recorded run passes the R9 action-74 failure and reaches 245 applied actions. Its
new refusal is a cell occupied before motion with 11 unchanged sensor points
inside evaluation-only banana bounds. Bag evidence corrects the earlier legacy
region diagnosis: tracker ON intentionally omits that region; world state instead
expired its entire tracked-object table during a 1.627 s receive gap. The backup
fill now uses steady time because expensive default-group work also blocks the
ROS /clock subscription. Cached evidence stamps and all expiry rules remain.
A live lifecycle regression fails before and passes after the timer change;
identical cached sets stop after five seconds and world state then invalidates.
R11 recorded follow-through reaches 272 actions. All 342 post-grounding world
samples retain valid tracks; the longest receive gap is 1.401 s. The next stop
is a preexisting tabletop voxel, 91.4 mm outside the banana footprint and beyond
the 50 mm support-contact reach. The table is movable and cannot be declared
static to bypass that refusal. Next: observed dynamic support/workspace contact
identity and bounded force witnesses; distinguish voxel conservatism from
physical contact. Banana last-fit age (11.53 s with fresh map confirmation) also
needs investigation. No successful placement or reasoner recovery is claimed.
