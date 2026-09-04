# Collision-safety alternatives survey — is the hand-rolled kernel the right shape?

> **Status: research survey, analysis only.** Nothing on this page has landed and
> nothing here changes `cpp/openral_safety_kernel/` or
> `packages/openral_safety/`, both safety-WG gated. This is the evidence a
> reviewer would need to decide whether the mm-scale-discrimination problem the
> [validation evidence](collision-validation-evidence.md) documents should be
> solved inside the kernel or by adopting an established external stack.
>
> Companions: the [collision-primitive study](collision-primitive-study.md)
> (link-envelope geometry), the
> [validation evidence ledger](collision-validation-evidence.md) (the failure
> census and its standing caveats),
> and [world-map fidelity](world-map-fidelity.md) (the live map's own error
> terms).
>
> **Round-3 (2026-08-30, evening pass):** every file:line citation and external
> claim below was independently re-verified; corrections are folded inline and
> marked *(round-3)* — none changed a verdict. The same pass ran a second
> research sweep over methods outside the original shortlist (CAPT point-tree
> checking, Tesseract CCD/per-pair margins, learned distance fields,
> velocity-damper layers, proprioceptive contact discrimination); it is folded
> in as **§§13–23** so this file stays the single source of truth (the interim
> note this material started as was never committed to the repo, and is
> superseded by this file).
>
> **Method.** External claims were checked against primary sources
> (2026-08-28…30): official docs, and — where it mattered — the actual source
> files fetched from the upstream repositories. Throughout, "docs claim" means
> a statement from a project's documentation; "source shows" means the code was
> read. In-repo claims cite files in this tree. §2 (the PR ledger) is built
> from the 54 merged PR bodies of 2026-08-11…29, cross-checked against the
> evidence ledger's standing caveats; §7 (community) cites ~16 threads/issues
> found 2026-08-30, with maintainer replies marked. §12 is a second
> verification round (same date): academic literature via the Papers With Code
> catalog + arXiv with a logged search trail, plus upstream source and issue
> trackers — its corrections are folded into §4.1, §6.1 and §10.

---

## 1. The problem being shopped for

OpenRAL runs VLA policies (XR-1, π0.5, GR00T, …) that emit raw action chunks
with no collision awareness of their own, and a custom C++ safety kernel vets
every chunk: self-collision via per-link OBBs lowered from the URDF with an
SRDF-derived 16-pair allowed-collision list
(`robots/panda_mobile/robot.yaml` — `collision_geometry` for `panda_link1..7`
only, `allowed_collision_pairs` generated `source: srdf`), world collision via
an octomap-derived 25 mm voxel grid, with a staged 26-DOP → convex-hull-GJK
narrow phase for links declaring `tight_geometry`
(`cpp/openral_safety_kernel/README.md`, "Link geometry for the world-voxel
check").

Two weeks and ~30 PRs of calibration later, the measured state:

- **Contact-rich tasks E-stop before the grasp completes.** The latest two
  drawer-opening runs: run 1 stopped on "self collision `panda_link2` vs
  `panda_link5` at −5.34 mm" while offline mesh-to-mesh adjudication at the
  *recorded* joints shows the links **+53 mm apart** — the verdict was for a
  predicted horizon configuration the artifact does not store (the same
  reconstruction gap [issue #172](https://github.com/OpenRAL/openral/issues/172)
  records for payload stops). Run 2 stopped on "world collision `panda_link7`
  vs `voxel_219914` at −2.96 mm" — the voxel's own 27-ray ground-truth probe hit
  nothing, the cell's position resolves 2.2 m from link7, outside the grid's own
  r = 1.05 m coverage, and link7 was +40 mm clear of the nearest real surface by
  certified GJK.
- **The five-round battery** (`advisory-band-1..5`, 2026-08-26, commit
  `7cb2376`): of 15 stops, **11 classified `within-quantization`** — real
  contacts of ~1 mm reading as up to 15 mm of penetration because 25 mm voxels
  inflate thin geometry (the 2026-08-13 baguette run: −15.70 mm reported against
  six MuJoCo contacts of −0.87…−1.37 mm). Nine of the fifteen are a robot link
  against world occupancy at −0.29…−11.34 mm — "the whole spread sits inside
  what map discretisation alone explains"
  ([validation evidence](collision-validation-evidence.md)).
- **OBB corner slop is 27–76 mm of protrusion per link** (45–88 mm measured as
  corner slop), on top of the 21.7 mm voxel half-diagonal
  ([primitive study §4.2](collision-primitive-study.md)).

The design constraints are not in question: the kernel stays ("Python proposes,
C++ disposes", CLAUDE.md §3), E-stop latching on real contact is wanted, and
the evidence trail (FailureTrigger + latched SafetyStatus + OTel `safety.check`
spans) stays. What is in question is **mm-scale discrimination near contact**:
legitimate grasping — touching a drawer handle, holding an object, placing into
a shelf — reads as collision. Framed against what the survey found, the kernel
lacks exactly three things established stacks have:

1. **distance-based speed scaling** — the kernel's only verdict is
   accept / drop / latch at a fixed margin;
2. **mesh-level checking** — the world side is voxels vs primitives (the hull
   GJK narrow phase exists, but only against 25 mm cells);
3. **touch-links / attached-object semantics as a first-class contract** — the
   gripper is simply absent from `collision_geometry` (the `robot.yaml` comment:
   "checking it against the world would veto every grasp"), and intended-contact
   scoping was built bespoke (ADR-0092 witness, ADR-0097 declaration).

---

## 2. What the last two weeks built — PR ledger and verdicts

54 merged PRs (2026-08-11 → 08-29) form the collision / safety / sim-evidence
line this survey exists to judge. Two (#128, #129) are re-lands of #104/#117
after a squash-into-branch accident, so the distinct work is ~52 changes. Every
"taken back" claim below is cross-checked against the
[validation evidence ledger](collision-validation-evidence.md) "Standing
caveats" (8 entries, 3 of them withdrawals) and
[world-map fidelity](world-map-fidelity.md); judgments align with §9–§10 below.
Sources: each PR's own body (`gh pr view`), which in this repo carries the
measured rationale.

The one-paragraph version: the *instruments* (certified distance, branch-and-
bound ACM proof, fail-closed refusals) and the *voxel-grid fidelity* line
earned their cost — each found a trusted mechanism wrong in the unsafe
direction, and the oriented grid (#178) is the best-evidenced single
improvement in the corpus. The *link-envelope* line spent four PRs proving
perfect link geometry recovers only 27/72 stops. The *place-allowance*
(ADR-0097) line is armed but binding on 2 of 15 stops, and its advisory band
fired 0 times in 20 runs. Roughly ten days of adjudication output was voided by
the pipeline's own later corrections, and the Nav2 payload-footprint publisher
was built, never live-verified, and deleted six days later. The harness work
was recovery, not capability — every failure it fixed was a verification
failure, not a collision one.

### 2.1 PR ledger

| PR | date | theme | what | why | pros | cons |
|---|---|---|---|---|---|---|
| [#101](https://github.com/OpenRAL/openral/pull/101) | 08-11 | frame/FK | Scale normalized Cartesian predictions; exact voxel cubes; atomic multi-slot commit; PandaMobile FK "correction" | XR-1 RoboCasa365 deploy read normalized controller units as physical | Killed a whole class of unit-mismatch stops; exact cubes replaced circumscribed spheres | Its FK change was wrong (joint 1 1.033 -> 0.333 m, putting every kernel link 0.7 m too low) and cost #103 + #117 + #129 to undo |
| [#103](https://github.com/OpenRAL/openral/pull/103) | 08-15 | frame/FK | Restore FK root to ground-level base; replace 7 hand capsules with mesh-derived OBBs | #101 put every kernel link 0.7 m too low; capsules over-reported clearance | OBBs are sub-mm on faces; the geometry every later study measures | 1.033 root was itself wrong, reverted by #117/#129. Squash deleted a predictive test (#140 restored it) |
| [#104](https://github.com/OpenRAL/openral/pull/104) | 08-15 | ADR-0092 | `AttachedCollisionObject` schema + IDL + aggregator storage | Grasped payload must move from world occupancy into robot geometry | The contract every later attachment leg reuses | Never reached `master` (squash-into-branch); re-landed as #128. Pure process waste |
| [#110](https://github.com/OpenRAL/openral/pull/110) | 08-13 | frame/FK | Stop publishing `world→base_link` for mobile robots; new `describes_mobile_base` | Guard read `footprint_radius` off the wrong model, so it was dead code since 2026-07-09 | Fixed a two-parent TF break that timed out Nav2's global costmap | Latent for 5 weeks with no test; a typo-class bug that ate a debugging round |
| [#111](https://github.com/OpenRAL/openral/pull/111) | 08-13 | world map | Depth synth casts one `mj_ray` per pixel instead of batched `mj_multiRay` | `mj_multiRay`'s BVH cull skips visual-only geoms; free space reported where a surface is | Correct against MuJoCo's own GL depth; 947 mm median error on a real g1 scene | ~1.9× cost (6.0→18.6→55.6 ms/frame). #180 later filters exactly those geoms out, so the premium may now buy nothing |
| [#114](https://github.com/OpenRAL/openral/pull/114) | 08-13 | evidence | `CollisionEvidence.horizon_step` accepts the kernel's `-1` reactive sentinel | `ge=0` rejected every Cartesian-mode stop; reasoner fell back to raw-JSON truncation | Schema now honest about the kernel's actual wire format | Schema and kernel had disagreed since the reactive check shipped; nobody noticed until the payload work |
| [#115](https://github.com/OpenRAL/openral/pull/115) | 08-13 | observability | `safety_abort_getter` seam: apply-wait names a latched stop | 3 of 4 E-stopped runs reported a generic apply-timeout instead of the safety abort | Typed `ROSEStopRequested` reaches the dispatch result | Only fixed one wait; #121 had to widen it. Its own validation showed 1/4 coverage |
| [#117](https://github.com/OpenRAL/openral/pull/117) | 08-15 | frame/FK | ADR-0095 atomic: camera TF against `base_link`'s real body + FK 1.033→0.333 | Grid was stamped `base_link` but world-referenced; kernel self-cancelled the error | Nav2/SLAM/dashboard finally read obstacles at the right height; bit-identical collision outcomes | Also lost to the squash bug; re-landed as #129. Explicitly did not fix the false E-stops |
| [#118](https://github.com/OpenRAL/openral/pull/118) | 08-13 | observability | Typed `failure_kind` uint8 on `ExecuteRskill.Result` | Replanning ladder classified failures by substring-matching prose | Ten kinds mirroring CLAUDE.md §5; reasoner can no longer be re-routed by a reworded log line | Adds a mirror to maintain; string fallback retained as deprecated-in-place |
| [#119](https://github.com/OpenRAL/openral/pull/119) | 08-13 | observability | ADR-0096 latched `SafetyStatus` topic, both implementations + dashboard card | `/openral/estop` carries no reason; `envelope_unconfigured` was silent end-to-end | `TRANSIENT_LOCAL` means a late subscriber learns the truth on connect; fail-toward-unsafe staleness rule | Implemented twice (C++ kernel and `SafetyPassthroughNode`), a permanent double-maintenance tax |
| [#121](https://github.com/OpenRAL/openral/pull/121) | 08-14 | observability | Route every blocking dispatch wait through #115's seam | Post-#115 validation: 1 of 4 stops surfaced; single-slot policies reached the seam never | Rollout loop, post-reset wait and MoveIt approach all now abort named | One deliberate behaviour widening (silent safety publisher aborts a healthy goal); `_poll_future` still uninstrumented |
| [#128](https://github.com/OpenRAL/openral/pull/128) | 08-15 | ADR-0092 | Re-land of #104 + kernel payload-vs-world + coherent evidence cell | #104 never reached `master`; `min_distance` and the named cell could describe different cells | Evidence reporting became trustworthy; `sweep_min_distance` separated out | A merge-strategy failure cost a 6-PR re-stack |
| [#129](https://github.com/OpenRAL/openral/pull/129) | 08-16 | frame/FK | Re-land of #117 | ditto | Validated in vivo: a stop's distance held to 6 dp while its cell moved exactly 28 voxels | ditto |
| [#130](https://github.com/OpenRAL/openral/pull/130) | 08-16 | evidence | Ground-truth E-stop snapshot on every stop + three `mj_geomDistance` near-miss probes | A margin-based stop usually leaves `ncon == 0`, so a contact list cannot adjudicate it | Made every later verdict decidable; also restored the depth self-filter's clearing rays | Built the entire adjudication corpus on `mj_geomDistance`, which #170 later proved unusable — caveat 8 withdraws all of it |
| [#131](https://github.com/OpenRAL/openral/pull/131) | 08-16 | ADR-0092 | `SupportContactWitness`, payload occupancy clearing, and the `withheld ⊆ exempt` partition | Payload's own silhouette tripped the stop; then the witness starved on the occupancy it needs | Object-frame attestation cannot decorrelate; 53,872-probe invariant sweep; retired two weaker mechanisms | Three cross-package predicate mirrors to hand-maintain; 12.5 mm discrimination floor accepted |
| [#132](https://github.com/OpenRAL/openral/pull/132) | 08-16 | ADR-0097 | Place-phase witness, attach-sweep window, barrier release keyed on `(object_id, evidence_ref)` | Place into an enclosed container had no exemption path; then a successful attestation aborted its own goal | Kernel stays producer-blind; task-success signal decidable from artifacts for the first time | The barrier deadlock was self-inflicted by #128's own design and shipped for a full round |
| [#133](https://github.com/OpenRAL/openral/pull/133) | 08-16 | ADR-0097 | Approach allowance (`PlaceRegion` margin reduction), clock-domain fix, two one-voxel headrooms | Witness earned by touching, but the kernel stopped the payload before it could touch | The only two full task completions in the entire corpus (baguette, runs 2 and 3) | Caveat 1: that acceptance predates #134/#135 and was never reproduced. Headrooms are maintainer-directed, not derived |
| [#134](https://github.com/OpenRAL/openral/pull/134) | 08-16 | ADR-0092 | SAM 2.1 segmenter rSkill + `VisionAttachmentBridge` for real-hardware attachment evidence | Attachment evidence existed only in sim (MuJoCo ground truth) | Every gate is geometric, never the model score; a 0.977-score tablecloth mask ships as a rejection test | Opt-in, default off, ~17 commits, zero hardware validation, gripper-effort readback unvalidated. Nothing in-tree exercises it |
| [#135](https://github.com/OpenRAL/openral/pull/135) | 08-16 | world map | Rasterize octree→grid by cube overlap instead of centre sampling | Field: a cabinet panel published a full voxel closer to the robot, zero cells where it actually is | Strictly additive (mechanised superset proof); 7.9–37× *cheaper* | Bought +½ cell of forward reach — which #171 later measured as 48 % of live stops, undone by #178 |
| [#138](https://github.com/OpenRAL/openral/pull/138) | 08-22 | evidence | Close the stack's stale-claim debt; create the validation-evidence ledger | Four documented claims were provably wrong; a body of evidence existed only on the Spark | The ledger is now the single most useful artifact of the whole effort | Docs-only; it also recorded that several 08-13/08-14 rounds have no written summary at all |
| [#139](https://github.com/OpenRAL/openral/pull/139) | 08-22 | validation | `sim.estop_initial_configuration` guard + fridge-scene KNOWN DEFECT block | Fridge scene E-stopped at `latest_chunk: 0` and the artifacts read like a mid-task stop | Named a stop class that had wasted a whole round | Its diagnosis (pin a bottom-freezer layout) was **falsified by #154** — every bottom-freezer layout fails |
| [#140](https://github.com/OpenRAL/openral/pull/140) | 08-22 | tests | Restore the multi-step clear-trajectory predictive accept test | #103's squash repurposed the test away and left the false-positive direction uncovered | A look-ahead that stops on everything now fails a test | Pure debt repayment from a squash three weeks earlier |
| [#141](https://github.com/OpenRAL/openral/pull/141) | 08-22 | observability | Mirror `KIND_COLLISION` in `failure_bus.py`; contract-test the four-way IDL mirror | The ROS-free mirror stopped at kind 9 while everything else carried 10 | Mirror drift now fails a test, not an audit | Confirms the mirror pattern itself is the liability; #138 had already recorded three more mirrors |
| [#142](https://github.com/OpenRAL/openral/pull/142) | 08-22 | ADR-0097 | Declare place targets in 3 RoboCasa scenes; refuse a dispatch-supplied `PlaceRegion` | ADR-0097 machinery was merged and inert — `place_allowance_active` logged zero times, ever | Closed a real fail-open (a hand-written box could arm the allowance); fail-closed at three boundaries | 2 of 3 declared targets were guesses and **both were wrong** (#146). Six weeks of machinery had been dead code |
| [#143](https://github.com/OpenRAL/openral/pull/143) | 08-23 | Nav2 | Payload footprint publisher + lidar robot self-filter + the Nav2/kernel boundary | No repo path masked a real lidar's own returns; footprint was static while carrying | Self-filter is genuinely fail-closed (every failure leaves *more* obstacles); measured the `consider_footprint` cost | The footprint publisher it built is **deleted by #186** six days later. `consider_footprint` deferred on a +8.1 ms figure #185 could not reproduce |
| [#144](https://github.com/OpenRAL/openral/pull/144) | 08-22 | evidence | `evidence_voxel_backing` + `adjudication_budget` (`corner_slop + voxel half-diagonal`) | The 08-22 round called two stops false positives with no way to see what backed the cell | Refuted its own round's leading hypothesis with its own artifacts; corner-slop table (28.3–88.2 mm) | Caveat 6: **every `false-positive` verdict before this PR is withdrawn** and cannot be re-derived. That is ~10 days of adjudication voided |
| [#145](https://github.com/OpenRAL/openral/pull/145) | 08-23 | harness | Land the 4-scene validation matrix as one versioned command with typed verdicts | ~17 rounds over ~10 days were driven by scripts that lived only on the Spark | A round can no longer end without a written summary; 8 Pydantic verdict models | Its own first live round died in <1 s on a nonexistent CLI flag, and reported exit 0. Recovery, not new capability |
| [#146](https://github.com/OpenRAL/openral/pull/146) | 08-22 | ADR-0097 | Correct the sink and fridge place targets against a live env; raise backstop 120→300 s | 2 of 3 #142 targets named no MuJoCo body; the 120 s backstop expired before the place phase | The fail-closed refusal caught both wrong names — cost a round, not a collision | Also found #142's documented lookup procedure (`env.fixture_refs[...].name`) does not exist. Guessing cost two PRs |
| [#147](https://github.com/OpenRAL/openral/pull/147) | 08-23 | evidence | `attached_payload_mesh_slop`; extend `adjudication_budget` to the self-collision case | A real, correctly-attributed payload stop was called a defect **twice** because OBB inverts the mesh ordering | Kernel arithmetic reproduced to 8 dp; the ordering inversion is now a gtest | The trigger value (−4.63 mm) is **not reproducible** from the round's artifacts — the snapshot kept the wrong tick |
| [#148](https://github.com/OpenRAL/openral/pull/148) | 08-23 | harness | `chmod +x` six ROS node files (2 breaking master, 4 latent) | `--symlink-install` inherits the source mode; launch abandons the whole description | Repo-wide guard test over every `install(PROGRAMS ...)` target | `openral deploy sim` could not launch **at all** on published master. Three of the latent four are the defence-in-depth E-stop path |
| [#149](https://github.com/OpenRAL/openral/pull/149) | 08-23 | harness | Four harness defects from a 24-run round: deaf monitor, launch-abort-as-deadline, visual-mesh `real-contact` | All 24 monitor logs held only start/stop; a purge of `/dev/shm` killed the monitor's participant | The collidable-mask fix; monitor gated on a `dds_transport_ready` marker | Caveat 5: **every `real-contact` verdict before 08-23 is withdrawn**. A 24-run round produced nothing |
| [#150](https://github.com/OpenRAL/openral/pull/150) | 08-23 | world map | nvblox height band measures a box's exact z-extent instead of reading `radius_m` | `AttributeError: 'BoxShape' has no attribute 'radius_m'` — the band could not be computed at all | Exact and closed-form at both edges; unhandled shape raises rather than guesses | Broken since #103 (8 days) because `robots/**` triggered only `tests/unit`. A test-selection hole, not a logic one |
| [#151](https://github.com/OpenRAL/openral/pull/151) | 08-23 | world map | Refuse an unplaceable collision volume; seed the manifest's URDF-root bridge | 5 of 11 geometry-bearing manifests silently truncated their band; `g1`'s excluded its whole upper body | Partial failure is now loud; `ur5e`/`ur10e` bands agree with the live TF tree | Another manifest-parsing class bug; the "5 not 3" correction shows the initial report was itself wrong |
| [#153](https://github.com/OpenRAL/openral/pull/153) | 08-25 | ACM/self | Refuse a disconnected collision graph; new `FixedAttachment` schema | Every rigid mount looked parentless and was modelled as a second robot base at the origin | Kernel FK now matches each robot's own normative source to 1e-16 m; 4 manifests corrected | A retracted commit inside it ("g1 transcription slips") was circular verification — caught, dropped, and pinned |
| [#154](https://github.com/OpenRAL/openral/pull/154) | 08-23 | validation | Pin fridge/utensil kitchens; validate and enforce a layout pin | `Kitchen._load_model` redraws layout on every reset, so `seed: 1` does not identify a kitchen | ~85 % of this task's kitchens have the defect — an unpinned round mostly tested the spawn | **Falsified #139's candidate fix.** Its own layout-30 pin was then falsified by #159/#171 — verified against mesh clearance, not the kernel criterion |
| [#157](https://github.com/OpenRAL/openral/pull/157) | 08-23 | link geometry | Measure what the kernel's link envelopes enclose; score sphere-swept boxes | Was the OBB the reason kitchen start states read in-collision? | Result #1 killed the premise: `base_link` has no collision geometry, so a chassis primitive buys zero. Recovers **no** characterised stop | Its swept-box protrusion row is **provably unattainable** (#158). Recommendation was "hold" — the study's own answer was don't ship it |
| [#158](https://github.com/OpenRAL/openral/pull/158) | 08-23 | link geometry | Render the OBB-vs-mesh geometry; independently re-derive #157's fit table | A corner-slop claim is hard to believe from a number | Found #157's swept row falls below a proven Lipschitz lower bound on 6 of 7 links | Ten PNGs (3.5 MB) for a candidate the same page recommends against. Two PRs to conclude "don't do this" |
| [#159](https://github.com/OpenRAL/openral/pull/159) | 08-23 | evidence | 240-start-state census: name the limb behind the RoboCasa stops | "If the base is excluded, what limb is stopping the scenes?" | `link1`+`link2` = 83 % of stops; `link3`/`link4` have the worst slop and dominate nothing. `UNEXPLAINED` = 0/72 | Its recommended layouts 18/21 were **refuted live** by #171. Its proposal-2 recovery numbers were corrected by #161 |
| [#160](https://github.com/OpenRAL/openral/pull/160) | 08-23 | evidence | Settle self-occupancy (**no**); refuse a re-lower that loosens hand-tightened geometry | The chassis is outside `collision_geometry`, so nothing knows it is the robot | 13,288/65,536 rays hit the robot unfiltered, **0** survive the self-filter. Hypothesis closed by measurement | Two of four stops resolved to cells containing **nothing at all**; the 120.9 mm baguette residual left explicitly unexplained |
| [#161](https://github.com/OpenRAL/openral/pull/161) | 08-24 | link geometry | Price tight link geometry; find the world grid holds 45 of 72 stops | Is the envelope the binding constraint? | **Perfect link geometry recovers 27/72 worst case.** Crossover at ~10 mm. Recommends staged 26-DOP → hull on link1/link2 only | Corrects #159 and #157 and its own prior hypothesis (15-axis SAT bound: 0.16 mm, not 3–6 mm). Four docs PRs to size one change |
| [#164](https://github.com/OpenRAL/openral/pull/164) | 08-25 | CI | Fix two independent ways `select-and-test` reported success over failure | A missing `local rc` discarded every lane but the last; a full-run diff ran **zero** lanes | Five named runs were green while carrying 12–13 lane errors; #166's own run had verified nothing | The strict lane gate had been decorative for longer than the issue tracking it existed |
| [#165](https://github.com/OpenRAL/openral/pull/165) | 08-25 | schemas | Make `CollisionShape` a real discriminated union; close two fail-open shape branches | Union resolved by pydantic *structural* matching; an untagged mapping silently became a sphere | `envelope_loader` no longer lowers an unknown primitive as an under-sized capsule | Emitted JSON Schema stays looser than runtime; a documented "never dump with `exclude_defaults`" landmine remains |
| [#166](https://github.com/OpenRAL/openral/pull/166) | 08-25 | link geometry | Staged 26-DOP → exact convex hull for links declaring `tight_geometry` | OBB over-reports link1/link2 reach by 53.3 / 46.8 mm — slack in a fitted box | Exact hull → 0.00 mm excess; 1.04–1.11× *faster*; containment definitional, not fitted | Scoped to 2 links. `link1`'s hull (1588 verts) was too slow to ship. Recovers 26 % of stops; the map holds the rest |
| [#169](https://github.com/OpenRAL/openral/pull/169) | 08-25 | ACM/self | Prove "always-colliding" by branch-and-bound instead of sampling; delete inscribed-sphere lowering | #155 claimed the sweep under-approximates a box — true, but its conclusion was inverted | `link5`↔`link7` genuinely interpenetrates in 914/14641 poses; the committed ACM was **exempting a real check** | No shipped ACM byte changed, so the unsafe exemption is still in the manifests "under protest" |
| [#170](https://github.com/OpenRAL/openral/pull/170) | 08-25 | instruments | Replace `mj_geomDistance` on the evidence path with a certified `convex_geom_distance` | Two silent failure modes; a `+0.148 mm` truth read as `0.000 mm`, knife-edge at 1 pm displacement | Self-checking (witness-clearance + separating-axis certificate); every probe now attests `certified_pairs` | Caveat 8: **no verdict from any probe before 08-25 is citable, of any kind.** Four `0.000 m` readings re-measure at +14.8 … +107.9 mm |
| [#171](https://github.com/OpenRAL/openral/pull/171) | 08-25 | world map | Re-pin the fridge kitchen against the *kernel* criterion; characterise the live map | #154 and #159 both verified pins against the wrong criterion | 61 live captures; layout 47 clears by +19.34 mm across two maps 68 % apart. **Split: 26 % link-side, 74 % world-side** | Refuted its two predecessors' pins. Found the census's "live map ⊆ ideal grid" assumption is **false** |
| [#175](https://github.com/OpenRAL/openral/pull/175) | 08-25 | evidence | File the two world-side terms as issues #173/#174; record the re-verification | Both would otherwise have been rediscovered | Named the dilation term as the largest single world-side contributor (48 % of live stops) | Docs-only bookkeeping; the finding was already in #171 |
| [#177](https://github.com/OpenRAL/openral/pull/177) | 08-27 | evidence | Payload pose (`xquat`) in the E-stop record; carry backing when evidence loses the race | The record could not replay the geometry it adjudicates — 14 of 15 battery stops | Also caught a stale-cell attribution: a stop was handed the *previous* stop's cell | Closes only half of the reconstruction gap — the survey's motivating run 1 still lacks the tripping horizon configuration |
| [#178](https://github.com/OpenRAL/openral/pull/178) | 08-27 | world map | Publish `/openral/world_voxels` on the OctoMap's own lattice, with an orientation | #173: cube-overlap dilation = 29–35 mm median extra reach, 48 % of live stops | Grid is now the octree cell-for-cell (proved both directions). Also closed a 124 mm coverage hole and a fail-*open* empty-grid path | 1555/826 lines over the 800 bar. **Safety-WG reviewer, hazard entry and sign-off all still unchecked.** Undoes half of #135's cost |
| [#179](https://github.com/OpenRAL/openral/pull/179) | 08-27 | graded outcome | `CollisionHit::advisory`: a declared place's own contact refuses the chunk without latching | A 1.4 mm physical touch read as −38.22 mm and ended the run | Five independent bounds, any one failing gives today's latch; `place_advisory_max_consecutive: 0` is an exact rollback | **Fired 0 times in 20 scene runs.** 9/15 stops are link-vs-world, outside its scope. Safety-WG path also unchecked |
| [#180](https://github.com/OpenRAL/openral/pull/180) | 08-27 | world map | Filter geoms with neither `contype` nor `conaffinity` out of the depth cast | #149 taught the *probe* to ignore intangible geoms; nothing taught the *perception path* | Per-ray monotonicity verified exhaustively (16,384 rays × 4 scenes); 234 fridge rays travel 25–815 mm further | Inverts two of #111's own assertions. Sim-only term — "no hardware counterpart", so it fixes evaluation fidelity, not safety |
| [#183](https://github.com/OpenRAL/openral/pull/183) | 08-27 | Nav2/CI | Build `openral_nav2_bringup` in the deploy image | The live Nav2 tests imported a package the image never built | Found a test that passed **vacuously** — "the payload never marked the costmap" is true when nothing reaches the costmap | The #143 Nav2 work had had no working CI verification since it merged |
| [#185](https://github.com/OpenRAL/openral/pull/185) | 08-29 | Nav2 | Land the carry-while-navigating scene; time the whole MPPI loop | #143's footprint had never run against a base that translates while carrying | Loop fits in every arm at ~20 % of a 50 ms cycle, 0 dropped cycles | **The flag delta measured +0.53 ms, not #143's +8.1 ms** — an unexplained 15× discrepancy, so the deferral was priced wrong |
| [#186](https://github.com/OpenRAL/openral/pull/186) | 08-29 | Nav2 | Delete the payload footprint publisher; turn `consider_footprint` on; fix the lidar frame | The grown polygon forbade the approach poses every RoboCasa place task requires | Removed a node that protected against nothing (payload rides 0.28 m above the costmap slice) | Reverses #143 six days later; moves payload avoidance to a 3-D **E-stop instead of an avoidance**. ADR + hazard line still owed |


### 2.2 Theme verdicts

**Frame and FK alignment (#101, #103, #110, #117/#129).** Net effect: correct. The
grid is `base_link`-referenced for every consumer, the FK root matches the
simulator, and #129's differential evidence (distance identical to six decimals,
cell moved exactly 28 voxels = 0.700 m) is the cleanest proof in the corpus. The
cost is that #101 introduced the error, #103 half-fixed it in the wrong direction,
and #117 fixed it properly — three PRs and one ADR for one constant. **Done, stop.**
Nothing in later evidence points back here.

**Exact geometry and instruments (#153, #157, #158, #161, #165, #166, #169, #170).**
Two very different halves. The *instrument* half is the highest-value work of the
two weeks: #170's certified `convex_geom_distance` and #169's branch-and-bound ACM
proof each found that a previously-trusted mechanism was wrong in the *unsafe*
direction (an ACM exempting a pair that interpenetrates 48 mm; a probe reporting
0.000 m where truth is +107.9 mm). #153 and #165 are the same class — refusals
replacing silent wrong states. The *link-envelope* half is a dead end reached
honestly: #157 → #158 → #161 spent four docs PRs to establish that perfect link
geometry recovers **27 of 72** stops and the crossover is ~10 mm, then #166 shipped
the answer for two links and stopped. §9 (point 2) confirms it — "the binding
term is the map, so mesh-level matters mainly for self-collision". **Instruments:
keep. Link envelopes: done, stop** — do not extend `tight_geometry` past link1/link2
without a new measurement showing the map term has been removed first.

**Voxel grid / octomap fidelity (#111, #135, #150, #151, #178, #180).** This is
where the measured yield actually is, and it is also the theme that churned hardest:
#135 fixed a half-voxel phase error by adding a half-voxel of dilation; #171
measured that dilation as 48 % of live stops; #178 removed it by changing the
message format rather than the rule. The oriented-grid A/B is the only paired round
in the corpus that shows across-the-board improvement — 3.5–6.3× less reported
penetration on the three link-side stops, one real false positive eliminated, all
four scenes progressing further — while the evidence page is explicit that one round
per arm with a diverging rollout proves direction, not magnitude. #180 closes the
remaining sim-only term (38–40 % of geoms intangible, 18.8 % of occupied cells
backed only by decoration). **Keep going** — this theme is the only one whose
measurements keep saying "the next thing here is worth doing". The two terms it
tracked, #173 (bridge dilation) and #174 (non-collidable geometry in the map),
are both closed as completed (2026-08-27) — #178 removed the dilation and #180
landed the `noncollidable_geom_ids` filter, so the continuation now needs a new
issue, not a pointer to those two.

**Place-phase allowance, ADR-0097 (#132, #133, #142, #146, #179).** Net effect after
churn: a scoped, fail-closed, evidence-gated allowance that is **armed on three of
four scenes and binding on 2 of 15 stops**. The trajectory is bad. #133 produced the
corpus's only two task completions, and caveat 1 says that acceptance was never
reproduced. #142 discovered the machinery had been inert since it merged; #146
discovered two of three targets were wrong and the documented lookup procedure did
not exist; #179's advisory band fired zero times in 20 runs and its own PR body says
so. §9 (point 1) names the actual shape of the fix: the declaration "names the
target; it does not yet give the kernel its geometry" — Path B. **Stop extending the
margin-reduction mechanism**; the next ADR-0097 work should be shipping the declared
target's geometry, not another bound on the region box.

**Evidence and adjudication pipeline (#114, #130, #138, #144, #147, #159, #160, #171,
#175, #177).** Honest, thorough, and enormously expensive. Three of the eight standing
caveats are withdrawals of verdicts this same pipeline produced: caveat 5 (all
`real-contact` before 08-23), caveat 6 (all `false-positive` before #144), caveat 8
(**all** verdicts from any probe before 08-25). Two independent PRs (#159, #160)
found the `mj_geomDistance` defect on the same day without coordinating. The
pipeline is now genuinely sound — certified distances, collidability attestation,
per-link slop budgets, payload pose in the record — but roughly ten days of
adjudication output was voided to get there, and §24 notes the two
motivating drawer runs still have no in-tree fixture. **Keep, but freeze the
contract.** The remaining gap worth closing is the one the survey calls zero-cost:
record the tripping *horizon configuration*, not just the measured one. #177 did the
payload half; the joint half is still open.

**Validation harness (#139, #145, #148, #149, #154, #164, #183).** The harness works
now and did not before. Its first live use ran nothing and reported exit 0; its
second produced 24 runs with empty monitor logs; master could not launch at all for
part of that window; CI's lane gate had been decorative; the deploy image did not
build the package under test. Every one of those is a *verification* failure, not a
collision failure, and each cost a round. The compensating value is real — a round
can no longer end without a written summary, and #164/#183 each caught a green result
that had verified nothing. **Done, stop building.** The marginal harness PR is now
lower-value than a marginal map PR.

**ACM / self-collision (#153, #169).** Small, clean, and the only theme with no
retraction. #169 inverted the framing of its own issue with measurement and refused a
sampling threshold on principle; #153 refused a forest and re-derived four manifests
from their own normative sources. **Done, stop** — with one caveat: #169 changed no
manifest, so `panda_link5`↔`panda_link7` is still exempted in
`robots/panda_mobile/robot.yaml` *(verified: line 751)* despite being proven a
real, checkable pair.

**Nav2 boundary (#143, #148, #183, #185, #186).** Net effect after six days: the
payload footprint publisher was built (#143), never verified in CI (#183), never run
against a translating base (#185), and deleted (#186). What survives is the lidar
self-filter, the ADR-0040 boundary docs, a `findCircumscribedCost` correction, and
`consider_footprint: true`. #185 also measured the flag at +0.53 ms where #143's
isolated benchmark predicted +8.1 ms — a 15× discrepancy neither PR explained,
and whose mechanism §12/G3 identifies (`CostCritic::score()` `continue`s on
free-space points before `inCollision` is reached) though the numeric split is
still uninstrumented — meaning the deferral that shaped #143's merge was priced on a number
nobody can reproduce. The direction #186 chose is defensible but strictly less
protective: a payload clipping a tall thin obstacle is now an E-stop rather than an
avoidance. **Stop.** Do not add more Nav2-side payload modelling; the boundary
decision (payload obstacle avoidance is the kernel's, in 3-D) is now explicit and
should be left alone until the ADR is recorded.

---

## 3. MoveIt 2

Primary sources: `github.com/moveit/moveit2` `main` (source files fetched raw),
moveit.picknik.ai. BSD-3-Clause; first-class Jazzy LTS (2.10, June 2024;
releases through 2.14.1 on docs.ros.org).

### 3.1 PlanningScene collision checking is mesh-level, FCL by default

Source shows (`moveit_core/planning_scene/src/planning_scene.cpp` L200) the
default detector is FCL; Bullet is switchable at runtime. Source shows
(`moveit_core/collision_detection_fcl/src/collision_common.cpp` L900–922) URDF
collision **meshes are loaded as real triangle meshes** into
`fcl::BVHModel<OBBRSS>` — every triangle copied, not convexified — primitives
map to FCL primitives, octomaps to `fcl::OcTree`. `checkCollision` = padded
robot-vs-world + unpadded self-collision (`planning_scene.cpp` L436–455).
`distanceRobot`/`distanceToCollision` return the minimum robot-vs-world
distance with closest pairs and nearest points; signed distance is available
(`DistanceRequest::enable_signed_distance`, penetration via an FCL collide
pass, `collision_common.cpp` L636–660).

Caveats, all from source: the FCL path is **discrete-only** ("Continuous
collision not implemented", `collision_env_fcl.cpp` L340–346); Bullet has CCD
but **no distance queries** (`collision_env_bullet.cpp` L241–250 logs "not
implemented") and is documented as not thread-safe; a distance query is a
separate, much slower pass than a boolean check.

### 3.2 Attached objects and touch_links

`moveit_msgs/AttachedCollisionObject` carries `link_name`, the object, and
`touch_links` — the links allowed to keep touching the attached body. Source
shows the semantics live in the FCL narrowphase callback
(`collision_common.cpp` L130–183): a ROBOT_LINK vs ROBOT_ATTACHED contact is
always allowed if the link is in the body's `touch_links`; two bodies attached
to the same link never collide; the ACM is checked separately and is **not**
modified on attach. Attached bodies are appended to the robot's FCL object set
and are checked against both the world and the robot in every check
(`collision_env_fcl.cpp` L212–240).

### 3.3 Octomap self-filtering also masks attached objects

Source shows (`moveit_ros/planning/planning_scene_monitor/src/planning_scene_monitor.cpp`):
`excludeRobotLinksFromOctree()` (L886), **`excludeAttachedBodiesFromOctree()`**
(L958) and `excludeWorldObjectsFromOctree()` (L988) register every link shape,
attached-body shape, and known world object with the occupancy-map updater's
`ShapeMask`, re-run on attach/detach. The point-cloud updater
(`pointcloud_octomap_updater.cpp` L285, L336–345) classifies each point against
the padded/scaled shapes (`padding_offset`, `padding_scale`) and only `OUTSIDE`
points become occupied cells. This is the upstream equivalent of
`openral_octomap_bridge`'s payload clearing — with the same structural
consequence this repo found the hard way (the 2026-08-14 witness/clearing
partition): masking the payload's cells also masks the support surface cells it
rests on.

### 3.4 MoveIt Servo: proximity-based velocity scaling, not a binary stop

Source shows (`moveit_ros/moveit_servo/src/collision_monitor.cpp`) a dedicated
thread polling padded robot-vs-world and unpadded self-collision with
`request.distance = true` at `collision_check_rate` (default **10 Hz**;
`servo_parameters.yaml` warns "Collision-checking can easily bog down a CPU if
done too often"). The scaling formula, quoted from source:

```cpp
// If collision detected scale velocity to 0, else start decelerating exponentially.
// velocity_scale = e ^ k * (collision_distance - threshold)
// k = - ln(0.001) / collision_proximity_threshold
scene_collision_scale = std::exp(scene_velocity_scale_coefficient *
    (scene_collision_result_.distance - servo_params_.scene_collision_proximity_threshold));
collision_velocity_scale_ = std::min(scene_collision_scale, self_collision_scale);
```

Scale is 1.0 at the threshold (defaults: self 0.01 m, scene 0.02 m), decays to
0.001 at distance 0, and is hard 0.0 only on actual collision. Octomap checking
is **off by default** (`check_octomap_collisions: false`). Servo is a slowdown
assist for teleop/servoing; nothing in the docs claims it is a certified safety
function.

### 3.5 Rate and the standalone-filter pattern

The in-tree pattern for "checkCollision on streamed joint states as a filter"
is exactly Servo's monitor — and its default is 10 Hz, not 100. Published
throughput (GSoC Bullet benchmark, Panda,
[moveit/moveit#1427](https://github.com/ros-planning/moveit/issues/1427#issuecomment-514541239)):
boolean self-check ~9 µs (FCL) / ~3.7 µs (Bullet); 100 world meshes ~29 µs
clear, but **~1.25 ms (FCL) with 4 meshes in collision** — and a
`distance=true` query costs a further full distance pass on top. 20–100 Hz on a
7-DoF arm vs meshes is plausible; vs a dense octomap with distance enabled it
is exactly the load Servo's defaults avoid. No official MoveIt octomap-latency
benchmark was found.

### 3.6 The contact-rich pattern: phase-scoped ACM edits

The canonical MoveIt answer to "the gripper must touch the object" is two
mechanisms, both scoped, neither global: `touch_links` on attach (§3.2), and
MoveIt Task Constructor's `ModifyPlanningScene` stage, which mutates the ACM
**per planning-scene snapshot for one stage** — the pick-and-place tutorial's
sequence is "allow collision (hand,object)" → close gripper → attach → lift,
and on place, allow object↔support-surface, open, forbid, detach
(`moveit_task_constructor/core/src/stages/modify_planning_scene.cpp` L79–145;
[MTC tutorial](https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html)).
**This is ADR-0097's "place declaration" pattern, established upstream**: a
typed, phase-scoped allowance for a named object against a named surface, with
everything else still checked. The differences are that MTC scopes at plan time
(a stage boundary) where ADR-0097 scopes at run time (a declaration with a
timeout, gated further by measured contact), and that ADR-0097 fails toward
*less* permission — a bad region grants nothing. The in-repo mechanism is the
more conservative of the two.

---

## 4. FCL and coal (ex hpp-fcl)

Primary sources: `github.com/flexible-collision-library/fcl`,
`github.com/coal-library/coal` (the 2024 rename of hpp-fcl). Both BSD-3-Clause;
coal is maintained by Gepetto/LAAS-CNRS and INRIA and is the collision engine
of Pinocchio 3.

- **FCL's signed distance is demonstrably broken in exactly the mm regime**:
  with BVH mesh models, `min_distance` returns 0 in penetration even with
  `enable_signed_distance` set (issues
  [#221](https://github.com/flexible-collision-library/fcl/issues/221),
  [#574](https://github.com/flexible-collision-library/fcl/issues/574),
  [#575](https://github.com/flexible-collision-library/fcl/issues/575)); box-box
  signed distance can hang on a libccd EPA assertion (#578 — *(round-3)*
  closed 2024-02 by its own reporter for inactivity, with the maintainer's
  acknowledged root cause never fixed; closed-unfixed, not resolved). MoveIt's default
  detector inherits this.
- **coal has its own GJK+EPA** (no libccd), computes true signed
  distance/penetration depth (`DistanceRequest::enable_signed_distance`
  default true), and offers a `security_margin` on collision checks plus a free
  `distance_lower_bound` (source shows,
  [`include/coal/collision_data.h`](https://github.com/coal-library/coal/blob/devel/include/coal/collision_data.h)).
  Recommended solver tolerance ≥1e-6 — sub-micron, comfortably inside 1 mm vs
  15 mm discrimination *for convex pairs*. EPA robustness was an active fix
  area through 3.0 ("Fixed EPA returning nans…", CHANGELOG); use coal ≥ 3.0.
- **Performance** (source shows, Montaut et al., RSS 2022,
  [arXiv:2205.09663](https://arxiv.org/abs/2205.09663), the paper behind coal's
  accelerated GJK): distance queries on ShapeNet convex meshes run 0.8–2.5 µs
  per pair near contact at ε = 1e-8. Arithmetic, not a citation: 7 links × 20
  world objects ≈ 140 pairs ≈ 0.15–0.5 ms/cycle single-threaded — >20×
  headroom at 100 Hz. No published number for octree-distance latency.
- **Octrees work directly**: source shows
  [`src/distance_func_matrix.cpp`](https://github.com/coal-library/coal/blob/devel/src/distance_func_matrix.cpp)
  registers distance functions for OcTree vs every primitive, convex, and BVH
  variant. But the answer is distance to occupied *cells* — axis-aligned boxes
  at octree resolution — so **the world-side accuracy is still bounded by the
  voxel size**, exactly as the in-tree grid is.
- **Self-collision pipeline exists upstream**: Pinocchio 3 (built on coal)
  ships `addAllCollisionPairs` → `pinocchio::srdf::removeCollisionPairs` (SRDF
  `<disable_collisions>`, the same file format the in-tree ACM is generated
  from) → `computeDistances` per active pair
  ([Pinocchio collisions example](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/md_doc_b-examples_e-collisions.html)).

Worth stating for perspective: the kernel already *has* an in-house convex-hull
GJK with certified conservative fallbacks
(`cpp/openral_safety_kernel/README.md`, staged narrow phase), and the HAL has a
certified separating-axis instrument (`openral_hal.convex_distance`). coal is
what those would have been as a dependency; adopting it buys maturity and
mesh-mesh self-collision distance, not a capability class the repo lacks the
ability to build.

---

### 4.1 Round-2 verification (2026-08-30): coal is not a free swap

A second, independent pass against coal's own tracker and the ROS index
downgrades the "optionally swap to coal" recommendation:

- **Adoption is real** — Pinocchio 4.1.0 depends on `coal` directly
  ([package.xml](https://raw.githubusercontent.com/stack-of-tasks/pinocchio/master/package.xml))
  and coal 3.0.3 is buildfarm-released in ROS Jazzy's `distribution.yaml`.
  **But MoveIt 2 has not migrated**: `moveit_core` still depends on plain
  `libfcl-dev`, and a search of `moveit/moveit2` issues+PRs for "coal" returns
  zero results — a genuine negative, not a search failure.
- **Open accuracy issues in exactly the near-contact regime this repo needs**:
  [#823](https://github.com/coal-library/coal/issues/823) witness points go
  wrong once meshes *penetrate* (open, 2026-03); [#755](https://github.com/coal-library/coal/issues/755)
  a `security_margin` check silently misses (open, 2025-09); [#603](https://github.com/coal-library/coal/issues/603)
  octree-vs-octree distance returned `inf` for ~2 years; [#636](https://github.com/coal-library/coal/issues/636)
  NaN contacts from mesh-shape traversal (open).
- **The fixes this survey leaned on are unreleased.** Source shows the
  GJK/EPA-NaN fix and the octree-octree fix (#811) sit under `[Unreleased]`
  in [CHANGELOG.md@devel](https://raw.githubusercontent.com/coal-library/coal/devel/CHANGELOG.md);
  the latest tag is 3.0.4 (2026-06-29). Pinning `coal>=3.0` does not get them.
- **Performance is not uniformly better**: an independent migration benchmark
  ([#857](https://github.com/coal-library/coal/issues/857)) measures coal's
  broadphase ~1.8× *slower* than FCL 0.6.1 (distance ~1.08×, narrowphase
  ~1.14× slower, numerically identical); and
  [#649](https://github.com/coal-library/coal/issues/649) reports EPA
  allocations on the hot path — in tension with the kernel's no-alloc rule.

**Revised verdict:** keeping the in-house certified GJK is *lower-risk today*
than adopting coal for the near-contact narrow phase. coal remains the right
watch item (BSD-3, active, Pinocchio-backed) — re-evaluate when the
`[Unreleased]` fixes ship in a tagged release.

---

## 5. cuRobo + nvblox + Isaac ROS cuMotion

Primary sources: curobo.org, `github.com/NVlabs/curobo`,
`github.com/nvidia-isaac/nvblox`, nvidia-isaac-ros.github.io. The repo already
emits cuRobo collision spheres from the kernel's own lowered geometry
(`packages/openral_safety/openral_safety/cumotion_config.py`).

- **Usable as a pure checker, batched**: `RobotWorld` exposes
  `get_world_self_collision_distance_from_joints(q)` on `(batch, dof)` tensors
  and a trajectory variant on `(batch, horizon, dof)` — world + self distances
  on arbitrary configurations, no planner
  ([API docs](https://curobo.org/_api/curobo.wrap.model.robot_world.html)).
  Throughput (docs claim, RTX 6000 Ada, batch 2000): full collision-*checked IK
  solves* at 0.05–0.11 ms/query; the paper
  ([arXiv:2310.17274](https://arxiv.org/abs/2310.17274)) claims >7000
  collision-free IK queries/s. Vetting a VLA chunk's whole horizon in one batch
  is well inside this envelope.
- **But the distance is clamped**: docs state the signed-distance query returns
  a positive value only inside `activation_distance`, and **zero beyond it**
  ([Collision World Representation](https://curobo.org/get_started/2c_world_collision.html)) —
  fine for a margin gate, no true clearance for graded scaling beyond η.
- **Self-collision is spheres, cm-conservative**: source shows
  `curobo/content/configs/robot/franka.yml` carries **61 collision spheres**
  with explicit per-link buffers. Sphere sets over-approximate flanged links —
  the same class of envelope slop the
  [primitive study](collision-primitive-study.md) measured for capsules
  (96–112 mm protrusion on panda links). Not a route to mm self-collision
  fidelity.
- **nvblox ESDF is voxel-bounded**: Isaac ROS default `voxel_size` = 0.05 m; no
  doc claims sub-voxel accuracy anywhere. **Coarser than the in-tree 25 mm
  grid**, not finer.
- **The robot segmenter is real and directly relevant**: docs claim the
  cuMotion robot segmenter FKs the sphere model and masks depth pixels within
  sphere radius + `distance_threshold` (default 0.05 m), publishing a
  robot-free depth image; the object-attachment node adds a grasped object's
  spheres to the robot model so it is checked against the world *and* masked
  from the depth stream, with its voxels cleared from the nvblox SDF
  ([Isaac ROS manipulation concepts](https://nvidia-isaac-ros.github.io/concepts/manipulation/index.html)).
  This is the GPU equivalent of the depth self-filter + ADR-0092 payload
  clearing already in-tree.
- **Licenses, verified from source, one trap**: cuRobo relicensed to
  **Apache-2.0** with the V2 release (source shows: LICENSE at `main` vs the
  2023 non-commercial NVIDIA License at the initial commit) — "cuRobo is
  non-commercial" is stale. nvblox is Apache-2.0. **`nvblox_torch` — the
  cuRobo↔nvblox bridge — is still NVIDIA non-commercial** (source shows §3.3
  "research or evaluation purposes only"), which gates the depth-camera BLOX
  path for commercial deployment. Isaac ROS cuMotion is docs-claimed compatible
  with ROS 2 Jazzy (Jetson Thor/Orin, x86_64 Ampere+).

---

## 6. Safety filters for learned policies — the field

- **No flagship VLA stack ships any collision layer.** Verified by cloning and
  grepping at HEAD (2026-08-30): `Physical-Intelligence/openpi` (π0/π0.5)
  contains zero hits for collision/e-stop/deadman/safety — the DROID example's
  only action processing is `np.clip(action, -1, 1)`; `octo-models/octo` has
  none; `NVIDIA/Isaac-GR00T` ships *advice* (soft joint/EEF-pose limits plus an
  e-stop hotkey in `getting_started/real_world_deployment.md`), not code;
  `huggingface/lerobot` is the most careful and is still purely kinematic
  bounds (`max_relative_target` per-step delta caps, an EE workspace box). A
  GitHub code search of `isaac-sim/IsaacLab` for "safety filter" returns zero
  results. The de-facto contact safety under research Franka deployments is the
  arm's own reflex layer (`Robot::setCollisionBehavior` torque/force
  thresholds, libfranka docs). **The in-tree kernel is ahead of the field here,
  not behind it.**
- **The academic framing is projection and speed modulation, not binary
  stops.** ATACOM (Liu et al., CoRL 2021; T-RO version
  [arXiv:2404.09080](https://arxiv.org/abs/2404.09080)) projects any policy's
  action onto the tangent space of a constraint manifold; the 2025 follow-up
  ([arXiv:2505.10219](https://arxiv.org/abs/2505.10219)) applies it explicitly
  as a safety layer *after a foundation policy*, arguing safety should not be
  expected to emerge from data scale. CBF filters minimally modify the command
  via a QP on a distance-based barrier — continuous slowdown near the boundary.
  For chunked policies specifically, PACS
  ([arXiv:2511.06385](https://arxiv.org/abs/2511.06385)) verifies the *whole
  predicted chunk* by set-based reachability and brakes along the policy's own
  intended path, reporting up to 68 % higher task success than reactive CBF
  filtering — evidence that graded responses on the chunk beat point-wise
  binary vetoes.
- **The intended-contact problem has two published answers.** Geometric:
  exempt the intended target and check everything else — the attention-guided
  VLA filter ([arXiv:2606.09749](https://arxiv.org/abs/2606.09749)) derives the
  target from the model's own attention and notes plainly that VLAs "offer no
  guarantees against collisions with task-irrelevant objects"; ADR-0097 is the
  same shape with an explicit declaration instead of inferred attention.
  Force/energy-based: **ISO/TS 15066** power-and-force limiting
  ([iso.org/standard/62996.html](https://www.iso.org/standard/62996.html)) is
  the industry answer for permitted contact — bound contact force/pressure, not
  forbid contact; SARA-shield
  ([arXiv:2412.10180](https://arxiv.org/abs/2412.10180)) applies exactly this
  to *learned* manipulation, verifying contact kinetic energy against per-body
  thresholds. Near contact, force is the observable that discriminates a 1 mm
  touch from a crush; geometry at any voxel resolution is not.


### 6.1 Round-2 correction: the layer now exists in the literature (Dec 2025 – Jun 2026)

The first pass said no established collision layer for VLAs exists. That was
true of vendor stacks and is **now false of the literature** — four groups
published external safety layers over *frozen* VLAs within seven months, three
with code, converging on exactly the in-tree kernel's shape (external plug-in
filter, distance/CBF-shaped):

- **VLSA / AEGIS** ([arXiv:2512.11891](https://arxiv.org/abs/2512.11891),
  Tsinghua, Dec 2025, code `THU-RCSCT/vlsa-aegis`): plug-and-play CBF-QP layer
  over unmodified VLAs — VL module localizes the critical obstacle, QP corrects
  unsafe actions. **+59.16 % obstacle avoidance *with* +17.25 % task success**
  on its SafeLIBERO benchmark. \[abstract + project page]
- **Attention-guided safety filter** ([arXiv:2606.09749](https://arxiv.org/abs/2606.09749),
  Jun 2026): training-free — attention heads localize the intended object, the
  rest is obstacle, CBF-filtered. Its stated motivation — VLM-inferred obstacle
  identification is *too slow for the control loop* — is an independent
  endorsement of ADR-0097's **explicit declaration** over inferred targets.
  *(round-3)* Its dynamic-obstacle extension reports a **43 % average
  improvement** over the oracle baseline on a moving-obstacle SafeLIBERO
  variant, using a lightweight real-time tracker.
- **SafeVLA** ([arXiv:2503.03480](https://arxiv.org/abs/2503.03480), NeurIPS
  2025, code): the training-time lane (constrained RL), +83.58 % safety,
  +3.85 % performance.
- **PACT** ([arXiv:2606.08414](https://arxiv.org/abs/2606.08414), ICML 2026):
  post-training constraint projection, −31.0 % violations, +30.7 % success.

Plus three benchmarks now measure VLA safety violations directly — SafeLIBERO,
**LIBERO-Safety** ([arXiv:2606.23686](https://arxiv.org/abs/2606.23686):
19,664 collision-free demos, eight VLAs evaluated) and ForesightSafety-VLA
([arXiv:2606.27079](https://arxiv.org/abs/2606.27079)) — off-the-shelf
evaluation harnesses that could replace part of the bespoke five-round battery.
Runtime *failure* monitors reading VLA internals (SAFE,
[arXiv:2506.09937](https://arxiv.org/abs/2506.09937), code; Hide-and-Seek,
[arXiv:2605.30834](https://arxiv.org/abs/2605.30834)) are the natural producer
of the replanning ladder's hand-off signal. The field's own survey
([arXiv:2604.23775](https://arxiv.org/abs/2604.23775)) still names "unified
runtime safety architectures" an *open problem* — so the corrected claim is:
**no vendor ships one and no standard exists, but the layer is actively
populated and the in-tree kernel matches the published shape.**

---

## 7. What the ROS community reports

Community threads and issues (searched 2026-08-30), as distinct from the
official docs and source in §3–§6: what practitioners actually hit and what
they do about it. The load-bearing discussion for this problem class lives in
GitHub issues (moveit/moveit, moveit_ros, moveit2, NVlabs/curobo) and the
legacy ROS Answers / moveit-users archives — ROS Discourse itself was thin.
Threads with maintainer replies are marked; pre-2022 threads are flagged where
they may be stale.

### 7.1 What practitioners hit

**Octomap doesn't clear cleanly around a collision object you're about to touch/grasp — a long-running, still-open MoveIt bug.**
- ["Octomap updates for task planning"](https://github.com/orgs/moveit/discussions/3613) — moveit/moveit discussion, opened Nov 19, 2025. User running ROS 2 Humble, octomap resolution 0.015, hits planning failures because voxels overlapping an added collision object aren't cleared immediately during a pick task; worse at finer resolution. **Maintainer reply (rhaschke, MoveIt maintainer)** labels it "a bug with MoveIt's octomap" and moves it for tracking — i.e. still unresolved as of late 2025.
- ["Too many cells of the octomap are cleared when a collision object is added"](https://github.com/moveit/moveit/issues/3221) — Sept 22, 2022, no maintainer reply, still open. Inverse symptom (over-clearing) on the same underlying clearing logic.
- ["\[perception\] octomap not properly cleared when collision model is added"](https://github.com/moveit/moveit_ros/issues/315) — Sept 24, 2013 (repo archived 2017, pre-2022/stale but the exact same defect resurfaces in 2022 and 2025 above — this is not a fixed-and-forgotten issue, it's chronic).

**Depth-camera/octomap self-filtering sees the robot's own arm/gripper (or its grasp target) as an obstacle.**
- ["Octomap detect arm as obstacle"](https://github.com/ros-planning/moveit/issues/3210) — Sept 7, 2022. Arm re-entering camera FOV after a planned move gets flagged as collision; user tried both `DepthImageOctomapUpdater` and `PointCloudOctomapUpdater` with `padding_scale`/`padding_offset` tuning, no fix, no maintainer reply, still open.
- ["Moveit! self_filter from octomap working for robotiq gripper but not UR5/UR10 body"](https://answers.ros.org/question/307247/moveit-self_filter-from-octomap-working-for-robotiq-gripper-but-not-ur5ur10-body) — self-filtering is inconsistent across links even on a stock arm/gripper combo.

**MoveIt Servo's collision checking is too blunt for contact-rich/streamed commands.**
- ["moveit servo collision checker precheck"](https://github.com/ros-planning/moveit2/issues/409) — April 2, 2021 (pre-2022, likely stale — no activity since). Reporter: both collision-checking modes ("threshold_distance" and "stop_distance") either slow the robot even while it's retreating from the obstacle, or just halt outright — no directional/contact-aware nuance. Assigned to a maintainer (AndyZe) but no follow-up visible; feature never landed.

**Self-collision ACM/padding tuning is a known pain point, not a precision fix.**
- ["\[perception\] padding parameters difficult to tune"](https://github.com/moveit/moveit_ros/issues/342) — Oct 16, 2013 (archived repo, stale by date but frequently re-cited): "`padding_offset` and `padding_scale`... are very difficult/impossible to tune"; small offsets (0.01–0.02 m) didn't help, larger ones made it worse.
- ["\[perception\] padding scale parameters inverted for collision objects"](https://github.com/moveit/moveit_ros/issues/537) and ["fixed padding collision attached objects" PR #2721](https://github.com/moveit/moveit/pull/2721) — attached objects didn't even inherit link padding/scale correctly until this fix; another sign the padding knob has a rough implementation history.
- A New Software Tool for Generating and Visualizing Robot Self-Collision Matrices (arXiv, [2512.23140](https://arxiv.org/html/2512.23140)) — not a community thread but documents the underlying defect the threads above complain about: MoveIt's sampling-based ACM generator (MoveIt Setup Assistant) is "susceptible to omissions" and leaves both false-active and over-conservative pairs because rare/edge-case configurations are under-sampled.

**nvblox/cuRobo voxel collision world has the same self-vs-world confusion, and dense-obstacle scenes make it worse.**
- ["Can I set collision ignore space near robot when I use nvblox?"](https://github.com/NVlabs/curobo/discussions/148) — Jan 31–Feb 23, 2024, **NVIDIA maintainer replies**: "We currently do not have a way to segment out the robot" at first; then suggests geometric masking of the depth image, then points to the official [robot-segmentation doc](https://curobo.org/source/advanced_examples/4_robot_segmentation.html). A later comment (Sept 2025) reports the masking approach also deletes real nearby obstacle voxels — not a clean fix.
- [cuRobo Known Issues](https://curobo.org/get_started/6_known_issues.html): "Collision-free planning with the world represented by camera perception is an open research problem"; nvblox integration "works well in sparse obstacle environments. But as we increased the density of obstacles, the occlusions in perception can cause many failures." Also flags voxel sizes <1 cm hitting GPU memory limits fast — a direct tension with wanting finer voxels to fix quantization.

### 7.2 What practitioners do

**Commonly recommended, well-established:**
- **`touch_links` / dynamic `AllowedCollisionMatrix` entries for the grasp target, not tighter geometry.** MoveIt's own pattern (via `moveit_users` group threads and MoveIt Task Constructor docs) is: during the grasp/attach phase, explicitly allow collision between the gripper links and the target object (`ModifyPlanningScene` + `allowCollisions`, or `touch_links` on the attach call) rather than trying to make the octomap/box envelope precise enough to avoid a false positive. See [Collision checking for attached object during planning](https://groups.google.com/g/moveit-users/c/34cvq4mtCoE) and the [Pick and Place with MTC](https://moveit.picknik.ai/humble/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html) tutorial. This is the standard answer across many threads, not a one-off hack.
- **Attach the object to the end-effector link the instant the grasp closes**, so world-collision checks against the octomap treat it as part of the robot (with touch_links suppressing gripper↔object false positives) rather than as world geometry the octomap has to precisely carve out.

**Reported working but with caveats (not universally recommended):**
- **Force/torque-threshold contact classification instead of geometric collision** — Franka's `setCollisionBehavior` (see [franka_ros wiki](https://github.com/EUREKA-CardiffMet/EUREKA_Wiki/wiki/Franka-ROS), [franka_hw source](http://docs.ros.org/en/melodic/api/franka_hw/html/franka__hw_8cpp_source.html)) distinguishes "contact" (F/T inside a lower/upper band, robot keeps moving) from "collision" (reflex trip above the upper band), with separate `_nominal`/`_accel` thresholds. This is a first-party vendor mechanism, widely used, but it's arm-specific (Franka only) rather than a general ROS pattern.
- **Admittance/compliant control instead of pure collision-avoid-then-stop.** PickNik's [ros2_control admittance controller](https://picknik.ai/ros/robotics/moveit/2022/02/07/admittance-control-in-ROS2.html) (Andy Zelenak & Denis Stogl, Feb 2022) was built explicitly for "realtime contact tasks such as tool insertion," enforcing kinematic limits while bounding interaction force — i.e., shifting the safety question from "did geometry overlap" to "is force within bounds." [CRISP](https://discourse.openrobotics.org/t/announcing-crisp-closing-the-gap-between-ros-2-and-robot-learning/49625) (TU Munich, Aug 2025) extends this with torque/effort-based `ros2_control` controllers aimed specifically at deploying learned/VLA policies — a maintainer-adjacent, community-visible project, but its discourse post has no reply thread discussing collision/safety specifics, so treat its safety posture as unverified from this thread alone.
- **Robot self-segmentation before voxelization** (cuRobo/nvblox): mask the robot out of the depth image (via sim segmentation mask, or a geometric zero-box) before feeding nvblox, per NVIDIA's own guidance in discussion #148 above. Explicitly NVIDIA-recommended, but the Sept 2025 follow-up shows it's imperfect (can delete real nearby obstacles too) — call this "commonly recommended by the vendor, imperfect in practice," not a solved problem.

**One person's hack / not broadly validated:**
- Increasing `padding_scale`/`padding_offset` to compensate for quantization — repeatedly reported as fiddly-to-useless by individual users (moveit_ros#342, 2013); no thread shows this working as a general fix.
- Bespoke self-collision-matrix visualization tooling (the arXiv paper above) — a research prototype building on top of MoveIt's sampler, not (yet) an adopted community practice.

### 7.3 Directly actionable for this repo

1. **Use `touch_links`/dynamic ACM entries for the specific link-object pair expected to contact during a skill's grasp/place phase, instead of trying to shrink voxel/box quantization error.** [MoveIt attached-object collision threads](https://groups.google.com/g/moveit-users/c/34cvq4mtCoE) — this is the community's standard answer to exactly your failure mode (E-stop on expected millimeter-scale contact): stop treating the contact geometry as unknown-world-collision and instead declare it allowed for the duration of the contact-rich phase, gated by skill manifest / reasoner state so it's still an explicit, logged exception rather than a blanket disable.
2. **Don't chase perfect octomap-clearing fidelity as the fix — it's a chronic, maintainer-confirmed-unresolved MoveIt defect as of Nov 2025** ([discussion #3613](https://github.com/orgs/moveit/discussions/3613)). Investing engineering time in "make the voxel grid clear precisely around the grasped/contacted object" is fighting upstream's own unsolved problem; the ACM/attached-object-exclusion route (#1) is the load-bearing mitigation the community actually relies on.
3. **Adopt a force/torque or contact-force threshold as the arbiter for "is this a real hazardous collision" during contact phases, separate from the geometric octomap/box check** — mirrors Franka's `setCollisionBehavior` nominal/accel band pattern. Fits your failure mode directly: real ~1 mm contact should be classified by measured force, not by voxel/box penetration depth, during a skill-declared contact-rich window.
4. **If a compliant-controller layer is in scope, look at PickNik's admittance controller pattern** ([blog](https://picknik.ai/ros/robotics/moveit/2022/02/07/admittance-control-in-ROS2.html)) for the grasp/place execution phase — bounding interaction force rather than treating any geometric proximity as terminal, which is the structural fix for "contact-rich task E-stops on quantization artifacts."
5. **Treat `padding_scale`/`padding_offset` tuning as a low-value lever, not a fix path** — long, consistent community history of it being unpredictable (moveit_ros#342, #537) and never resolving quantization-vs-real-contact ambiguity; don't sink time there.
6. **If moving to a voxel/SDF world representation beyond octomap (e.g. nvblox/cuRobo), budget for robot self-segmentation as a required, imperfect component**, not a one-time fix — NVIDIA's own guidance (discussion #148) is a workaround with a documented 2025 regression report, so plan verification/tests around it rather than assuming it's solved.

### 7.4 Dead ends the community already tried

- **Tuning `padding_scale`/`padding_offset` to eliminate false-positive contact detection.** Reported "very difficult/impossible to tune" as far back as 2013 ([moveit_ros#342](https://github.com/moveit/moveit_ros/issues/342)), plus a separate bug where the scale parameter was inverted for collision objects ([moveit_ros#537](https://github.com/moveit/moveit_ros/blob/master/)) and attached objects didn't inherit padding at all until a later fix ([PR #2721](https://github.com/moveit/moveit/pull/2721)). No thread shows this knob reliably solving quantization-vs-contact ambiguity.
- **Relying on octomap/self-filter to robustly exclude the robot's own body or grasp target from world collision.** Chronic, unresolved across a 12-year span: [moveit_ros#315](https://github.com/moveit/moveit_ros/issues/315) (2013) → [moveit#3221](https://github.com/moveit/moveit/issues/3221) and [moveit#3210](https://github.com/ros-planning/moveit/issues/3210) (2022) → [discussion #3613](https://github.com/orgs/moveit/discussions/3613) (2025, maintainer-confirmed still-a-bug). The community has not solved general self/attached-object voxel clearing; it routes around it with `touch_links`/ACM instead of fixing the sensor-side geometry.
- **MoveIt Servo's built-in collision checking as a nuanced contact-aware safety layer.** [moveit2#409](https://github.com/ros-planning/moveit2/issues/409) (2021, no resolution) shows it's binary/blunt — slows or halts regardless of whether the robot is approaching or retreating from the "collision" — and the feature request to add directional awareness was never implemented. Not fit for contact-rich streamed/learned-policy commands as-is.
- **Geometric depth-image masking to segment the robot out of an nvblox/cuRobo voxel map.** Works initially per NVIDIA's guidance ([discussion #148](https://github.com/NVlabs/curobo/discussions/148), 2024) but a Sept 2025 follow-up in the same thread reports it also deletes real nearby obstacle voxels — an incomplete workaround, still unresolved in the thread.

### 7.5 Coverage notes

- ROS Discourse / discourse.openrobotics.org yielded little beyond the CRISP announcement — most of the load-bearing technical discussion for this exact problem class lives in GitHub issues/discussions (moveit/moveit, moveit/moveit_ros, moveit/moveit2, NVlabs/curobo) and the legacy ROS Answers / moveit-users Google Group archives, not Discourse threads. Said plainly rather than padded.
- Robotics Stack Exchange results largely overlapped with the archived ROS Answers content already cited above; no additional distinct threads worth citing separately were found.
- No thread was found specifically about a custom safety-kernel design vetting VLA/learned-policy actions pre-execution (the exact shape of this repo's problem) — the closest is CRISP (learned-policy ROS 2 controllers) and the general VLA-safety-filter literature (arXiv, not community threads), so this repo's approach appears to be ahead of what's discussed in ROS community venues rather than following an established pattern.

---

## 8. Comparison table

"mm near contact?" asks whether the approach can tell a ~1 mm intended touch
from a ~15 mm penetration against a *depth-sensed* world.

| approach | self-collision fidelity | world-collision source | near-contact behavior | runtime rate | ROS 2 integration effort | GPU | maturity / license | mm near contact? |
|---|---|---|---|---|---|---|---|---|
| **in-tree kernel** (today) | OBB + 26-DOP/hull GJK vs voxels; 16-pair ACM | octomap → 25 mm grid | binary margin → drop/latch; ADR-0097 margin reduction + advisory band | 30–200 Hz, allocation-free | — (it is the ROS 2 integration) | no | 2 weeks in-repo; Apache-2.0 | **no** — voxel + OBB slop 67–110 mm combined (45.3–88.2 mm corner slop + 21.7 mm half-diagonal) |
| **MoveIt 2 PlanningScene** | true URDF mesh vs mesh (FCL BVH) | meshes + primitives + octomap (self- and attached-filtered) | signed distance available; discrete only; distance pass is 10–100× a boolean check | Servo monitor default 10 Hz; boolean checks µs-scale, octomap+distance is the slow path | large (PlanningSceneMonitor, TF, URDF/SRDF pipeline) — replaces the kernel's world model | no | BSD-3; Jazzy LTS; the reference implementation | **vs known meshes yes; vs octomap no** (voxel-bounded) |
| **MoveIt Servo collision monitor** | via PlanningScene | via PlanningScene | **exponential velocity scale** from distance; 0 only on collision | 10 Hz default, tunable | moderate if only the *formula* is adopted | no | BSD-3; part of Servo, "slowdown assist" not a rated safety function | n/a — modulates, doesn't discriminate |
| **coal (hpp-fcl)** ≥ 3.0 | convex-hull signed distance, µs/pair, SRDF pair exclusion via Pinocchio | primitives, convex meshes, BVH, **octree (distance supported)** | true signed distance, EPA penetration depth, `security_margin` | 0.8–2.5 µs/convex pair → 100 Hz trivial for O(100) pairs | small as a library (C++, header-consumable); one dependency into the kernel | no | BSD-3; LAAS-CNRS/INRIA, active | **convex-pair yes; vs octree no** (voxel-bounded) |
| **cuRobo RobotWorld** | 61 spheres (Franka) + buffers — cm-conservative | primitives, meshes, nvblox ESDF, voxel grid | distance **clamped at activation_distance**; batched full-horizon checks | >10⁴ configs/ms batched | moderate (Python/torch sidecar; `cumotion_config.py` already emits the robot config) | **yes** | Apache-2.0 since V2 (verified); curobo.org docs still v0.7 paths | **no** — spheres + ESDF are cm-scale |
| **nvblox + robot segmenter** | n/a (perception side) | ESDF from depth, 0.05 m default voxels; robot + attached object masked from depth | n/a | real-time (GPU) | moderate (Isaac ROS, Jazzy-supported) | yes | nvblox Apache-2.0; **nvblox_torch bridge non-commercial** | **no** — coarser than the in-tree grid |
| **CBF / ATACOM / PACS filters** | whatever distance function you give them (spheres/SDFs) | ditto | **QP projection / chunk-consistent braking** — graded by construction | policy-rate | research code; no ROS 2 turnkey | varies | academic; no rated deployments found | inherits its geometry's resolution |
| **force limiting (ISO/TS 15066, Franka reflexes, SARA-shield)** | n/a | n/a — measures contact, not proximity | **discriminates touch from crush by force/energy**, permits intended contact | control-rate (kHz on Franka) | small on hardware that reports external torque; needs a sim analogue | no | ISO TS + shipping arm firmware; SARA-shield is research | **yes — the only row that is** |

---

## 9. Assessment

**1. Nothing off-the-shelf clears the voxel wall.** Every surveyed stack that
checks against a *depth-derived* world model is bounded by its cell size:
MoveIt's octomap path (FCL `OcTree` = occupied boxes), coal's octree distance
(same), nvblox's ESDF (0.05 m default — coarser than the in-tree 25 mm grid).
The 11-of-15 `within-quantization` stops are not a defect a library fixes; they
are what a 25 mm occupancy grid *is* near thin geometry, and the in-tree
evidence already proved the point from the other side — the support-contact
witness exists precisely because a ~1 mm contact reads as ~15 mm of cube
penetration (`cpp/openral_safety_kernel/README.md`, ADR-0092 D6). Where
established stacks do reach mm scale is against **known, modeled geometry**:
MoveIt checks real URDF meshes, coal gives µs certified signed distance on
convex pairs. The route to mm world-side discrimination is therefore not a
better checker but a better *world model* — objects the robot intends to touch
promoted from anonymous voxels to posed meshes/primitives (which is also what
ADR-0097's declaration already half-does: it names the target; it does not yet
give the kernel its geometry). *(round-3 qualifier)* Tesseract's 2026
implicit-SDF path (§17.3) is the one checker found that consumes a distance
field lazily instead of resampling it onto a grid — the "voxel wall" claim
stays true of every *mapper*, but no longer of every *checker*.

**2. The three missing capabilities are all established, and two of the three
are cheap.** (a) *Distance-based speed scaling*: MoveIt Servo's exponential
scale-from-distance is shipped, source-readable, and three lines of math; the
chunk-screening literature (PACS) independently finds graded braking beats
binary vetoes for learned policies. The kernel already computes
`min_distance`/`sweep_min_distance` per chunk — the number exists; only the
graded response does not. (b) *Touch-links / attached-object semantics*: MoveIt
formalizes what the repo built ad hoc — `touch_links` ≈ the finger-pair
exclusion, `AttachedCollisionObject` checked against world and robot ≈
ADR-0092, `excludeAttachedBodiesFromOctree` ≈ the octomap bridge's payload
clearing, MTC's phase-scoped ACM ≈ ADR-0097. **The bespoke mechanisms are
independently converged re-derivations of the established pattern, mostly
stricter** (evidence-gated, fail-closed, capped). That is a validation of the
design, and simultaneously the honest cost accounting: roughly two weeks were
spent rebuilding, with better evidence discipline, semantics MoveIt ships.
(c) *Mesh-level checking*: available (coal, or the kernel's own hull GJK
extended to self-pairs), and the [primitive study §6](collision-primitive-study.md)
already showed that better *primitives* (sphere-swept box, multi-OBB) recover
none of the characterised stops, and the tight-geometry census puts the ceiling
for *perfect* link geometry at 27 of 72 states
([tight geometry §7.3](collision-tight-geometry.md))
— the binding term is the map, so mesh-level matters mainly for self-collision
(the `panda_link5`↔`panda_link7` pair exempted "under protest" because boxes
false-positive on 79.6 % of the interpenetration band).

**3. The verdict-for-a-predicted-config evidence gap is in-house and no
library touches it.** Run 1's −5.34 mm self-collision verdict against links
+53 mm apart at the recorded joints is the artifact not storing the horizon
configuration the sweep actually tripped on — the same class as
[#172](https://github.com/OpenRAL/openral/issues/172) (payload pose absent from
the snapshot). Run 2's voxel resolving 2.2 m from the link, outside the grid's
own coverage, is an evidence-decode or indexing defect on the reporting path.
Both would persist unchanged under MoveIt, coal, or cuRobo. Fix the recorder
regardless of any architecture decision.

**4. On intended contact, geometry runs out and the field's answer is force.**
At the moment of a grasp the true clearance *is* ~0 mm; no geometric margin at
any resolution can pass "touch the drawer handle" while stopping "crush the
drawer handle". ISO/TS 15066 power-and-force limiting is the industry answer
for permitted contact, Franka's reflex thresholds are the shipping
implementation, and SARA-shield applies it to learned policies. ADR-0097 scopes
*where* contact is allowed; a force bound is what would say *how much* — the
axis the stack currently does not have at all.

**5. What real VLA deployments do: nothing — but the literature has caught up.**
openpi, Octo, GR00T, LeRobot ship clipping, joint bounds, and an e-stop key
(§6). No vendor product exists to adopt, and CRISP — the one ROS 2
learned-policy controller stack — ships **zero** collision/contact safety
(verified from source, §12). But §6.1's round-2 correction stands: four
2025–26 groups published plug-in safety layers over frozen VLAs, three with
code, and the strongest (VLSA/AEGIS) reports safety *and* success improving
together. The premise "surely everyone else solved this" is still false for
shipping software; the in-tree kernel now has published company on its
architecture.

**Bottom line.** The hand-rolled kernel is the right *shape* — a typed,
evidence-logging, fail-closed external checker is what the literature says a
learned policy needs and what no VLA vendor ships. What it got wrong is not
buildable-vs-buyable but two policy choices established stacks made
differently: a **binary stop where the field uses graded slowdown**, and a
**voxel-only world model where mm-scale work needs the intended-contact target
as modeled geometry (or a force signal)**. Both are adoptable as bounded
changes to the existing kernel; neither requires replacing it.

---

## 10. Candidate paths

All three keep: the E-stop latch and manual `estop_reset`, the topic contract
(`/openral/candidate_action` → `/openral/safe_action`, `SafetyStatus`,
`FailureTrigger`), the OTel evidence trail, the allocation-free hot path,
ADR-0092/0097 machinery, and the ACM. All three start with the zero-cost fix:
**record the tripping horizon configuration (and payload pose, #172) in the
stop evidence**, so the next false-positive investigation adjudicates the
config the kernel actually judged.

### Path A — graded response inside the existing kernel (recommended first)

Adopt Servo's shape, not Servo: compute a velocity scale
`exp(k·(min_distance − threshold))` from the sweep the kernel already runs, and
scale the outgoing chunk (or shrink the accepted horizon) in the band between
`margin + proximity_threshold` and `margin`; the latch at true penetration is
untouched.

> **Correction (2026-09-04). The "converts the 9-of-15 stops" claim below was
> wrong and is withdrawn.** Those stops are recorded at **−0.29…−11.34 mm**,
> and `hit.min_distance` is the reported pair's *true surface distance*
> (`collision.hpp:428`, `lifecycle_kernel.cpp:774`), while both
> `world_collision_margin_m` and `world_voxel_margin_m` default to **0.0**
> (`lifecycle_kernel.cpp:195,201`). The proposed band is therefore
> `[0, proximity_threshold]` in positive surface distance, and every one of
> those nine stops sits **below** it. Path A as specified does not convert
> them; they latch exactly as before.
>
> What Path A can honestly claim is that it slows the *approach* to such a
> stop, which may reduce how often one is reached. That is an empirical claim
> for the five-round battery to settle, not something derivable here.
>
> Converting them would require running the graded band into **negative**
> surface distance — continuing to command motion while two surfaces already
> interpenetrate by up to ~11 mm. That is a materially different proposal, it
> sits precisely on the quantization-vs-real-contact ambiguity this survey
> elsewhere records as unresolved, and it needs Safety-WG sign-off and a
> hazard-log entry rather than a parameter choice. Do not fold it into Path A
> silently.

Anything that keeps closing still stops. Path A generalises the
#176 advisory band (payload-only, place-only, depth-capped) into the graded
outcome the field uses, with the same severity ladder (`fold_pair` already
ranks severity over depth). *Replaced*: the binary-only verdict.
*Effort*: kernel-internal, no new dependencies, one new scaling parameter
family + hazard-log entry; the action-scaling seam already exists at the
accept/republish point. *Risk*: scaling a chunk changes what the policy's next
observation sees — needs the same battery treatment as the advisory band.

**Binding precondition: gate the scaling on the action's declared semantics.**
Servo's formula scales a *velocity*. `ActionChunk` carries a `control_mode`
mirroring `openral_core.ControlMode`, which includes **absolute** spaces —
`JOINT_POSITION`, `CARTESIAN_POSE`, `JOINT_TRAJECTORY`, `GRIPPER_POSITION`,
`FOOT_PLACEMENT`, `DEX_HAND_JOINT` — alongside the velocity and delta ones.
Multiplying an absolute joint-position target by 0.5 does not halve a speed; it
commands the arm halfway to its **zero configuration**, which is an arbitrarily
large motion executed in the name of caution, toward the obstacle as readily as
away from it. The repo already carries the discriminator this needs:
`ControlModeSemantics.mode` is `Literal["absolute", "delta"]`, declared per
`ActuatorRequirement`. Path A must scale only `delta` / velocity / twist modes
and, for absolute modes, **truncate the horizon or refuse the chunk** rather
than scale it. This is not a refinement to add later — a first implementation
that scales unconditionally is unsafe on the majority of the shipped adapters.

**Round-2 evidence (2026-08-30) — the direct experiment exists.** PACS
([arXiv:2511.06385](https://arxiv.org/abs/2511.06385), ICRA 2026, full text
read) ran exactly this comparison on robomimic LIFT/CAN/SQUARE with a dynamic
obstacle, 100 rollouts each:

| method | safe? | avg success |
|---|---|---:|
| unfiltered policy (same pipeline) | ✗ | 0.70 |
| reactive CBF projection | ✓ | **0.04** |
| single-action braking (SSM) | ✓ | 0.41 |
| **chunk-level graded braking (PACS-PFL)** | ✓ | **0.72** |

Graded slowdown along the policy's own path is *free* (0.72 vs 0.70);
reactive/binary filtering is what destroys the task (0.04). Reproduced on real
FR3 hardware, and on SmolVLA — not a diffusion-policy-only result. Two design
riders transfer directly: (1) **scale the whole chunk, not per action** (+28 %
in their ablation — the kernel's `sweep_min_distance` is already chunk-level);
(2) the published mitigation for the distribution-shift risk this path's own
*Risk* names is **observation-side**: PACS deliberately excludes velocity from
the policy's observations so slowing is not itself OOD. **Binding
precondition: audit every deployed adapter's observation space for velocity /
rate / step-index signals before shipping Path A** — a velocity-conditioned
policy inverts the OOD argument and PACS's result does not transfer.
Corroborating cost of the *un*-graded alternative: SARA-shield
([arXiv:2412.10180](https://arxiv.org/abs/2412.10180)) measures plain SSM at
59.2 %/55.9 % of unrestricted throughput (sim/real) vs 92.6 %/93.7 % for a
contact-classified graded bound. No manipulation-side chattering/deadlock
report was found in six query formulations — unestablished in either
direction, so the five-round battery remains the arbiter. Ecosystem check
(§12): **no ROS 2 component ships arm-side distance-graded scaling to reuse**
— Nav2's collision monitor is base-only and threshold-triggered, PILZ's SSM
never left ROS 1, ros2_controllers has nothing SSM-shaped — so hand-rolling
the formula into the kernel is confirmed as the only route, not a
reinvention.

### Path B — promote the declared target to modeled geometry

The mm problem is confined to the object the robot intends to touch; ADR-0097
already names it. Extend the declaration's producer to ship the target's
geometry (sim: the declared body's meshes/primitives, the same subtree it
already measures the `PlaceRegion` box from; real: a fitted primitive from
perception, the seam the README already reserves), and have the kernel check
link hulls / payload primitives against *that* at mesh resolution — its staged
GJK narrow phase already does exactly this query class against cells — while
the voxel grid keeps covering everything undeclared. The narrow-phase engine should stay the in-house certified GJK: the
round-2 check (§4.1) found coal's open near-penetration witness/margin issues
(#823, #755), its EPA hot-path allocations (#649), and that the fixes this
survey originally leaned on are unreleased — so the earlier "optionally swap
to coal" option is downgraded to a watch item until those ship in a tagged
release. This is the touch-links/AttachedCollisionObject semantics, arrived at
via the declaration the repo already has. *Replaced*: voxels as the sole world
representation near the grasp target. *Effort*: moderate — a new geometry field
on the declaration path (dispatch → HAL → World State → kernel), kernel ingest
+ bounds checks mirroring `ingest_place_region`, safety-WG review.

### Path C — force-based contact gating for the declared phase

Add the missing axis: during a live ADR-0097 declaration, gate the
intended-contact region by measured force/energy rather than geometry — sim:
MuJoCo contact forces the HAL already reads for ground truth; real Franka: the
reflex thresholds (`setCollisionBehavior`) plus external-torque estimates,
which is where ISO/TS 15066 points and what SARA-shield demonstrates for
learned policies. Geometry remains the authority everywhere undeclared; force
bounds the one place geometry cannot discriminate. *Replaced*: nothing —
purely additive conservatism-shaping inside the already-scoped allowance.
*Effort*: largest — a new sensor contract through Layer 1/2 into the kernel (a
layer-boundary decision), a sim/real seam, and force-threshold calibration with
its own hazard entries. Do it after A, and only if A + B leave contact-phase
stops on the table.

**Round-2 evidence (2026-08-30) — the numbers exist, the maturity does not.**
ISO/TS 15066:2016 Table A.2 (read from the standard): hands/fingers **140 N
quasi-static, 300 N/cm² peak pressure, ×2 transient**; Table A.3 body model
K = 75 N/mm, m_H = 0.6 kg for the hand. §A.3.4 states the standard's own
actuation knob for a force bound *is robot velocity* — **A and C are one lever
seen from two ends**, which is why C composes after A rather than competing
with it. A hazard entry should cite **ISO 10218-2:2025** *(round-3: the FDIS was
finalized and published; cite the 2025 standard, which absorbed the PFL
content)* alongside TS 15066 for the Annex A body-model tables. Realised budgets vary by an order of
magnitude for the same body region — SARA-shield uses 0.49 J
(constrained-blunt hand) where PACS's HANDOVER task uses 0.014 J — so **the
threshold must be a per-declaration field, not a config default**. Nearest
published instances of this path's exact shape: CompliantVLA-adaptor
([arXiv:2601.15541](https://arxiv.org/abs/2601.15541), F/T-bounded impedance
around frozen RDT/π0.5/OpenVLA-OFT, code) and FORGE
([arXiv:2408.04587](https://arxiv.org/abs/2408.04587), force-threshold-
conditioned assembly, >1000 real trials, 15 N snap-fit) — though FORGE
conditions the policy rather than gating it externally. Two sober caveats:
SARA-shield is one lab, one arm, no published repo URL, no ROS 2, no rated
deployment; and **no paper validates MuJoCo contact-force *magnitude* against
real F/T sensors** — FORGE re-tunes its threshold on hardware, which is itself
the finding. The sim side must ship a logged calibration parameter mapping
MuJoCo force to nominal Newtons, defaulted conservatively — a knob, not an
equivalence claim.

**Not recommended**: adopting MoveIt PlanningScene wholesale as the runtime
checker (10 Hz-class with distance + octomap, discrete-only FCL, and it would
replace working, better-instrumented machinery to gain semantics Paths A/B add
piecemeal), or the cuRobo/nvblox stack as the safety path (cm-scale by
construction; its right role here is the one `cumotion_config.py` already
targets — plan-time checking and, on GPU hosts, the robot segmenter for the
depth stream — noting the `nvblox_torch` non-commercial gate).

---


### Code map — where each addition lands (verified against `master`, 2026-08-30)

Every location below was re-checked by grep on the date above, not carried
over from the PR bodies.

**The zero-cost evidence fix (do first, all paths).** Record the tripping
configuration alongside the verdict:
- Kernel: the `report` lambda (`cpp/openral_safety_kernel/src/lifecycle_kernel.cpp:780`)
  is where every hit is named; the configurations it should capture are
  `q_check_` (measured seed, checked with sentinel `-1` at `:974`) and
  `q_predict_` (integrated per step in the predictive loop around `:1040`).
- Schema: `CollisionEvidence` (`python/core/src/openral_core/schemas.py:10056`,
  whose docstring already documents the `-1` sentinel) grows the joint row.
  There is no IDL mirror to update: `CollisionEvidence` crosses the wire as
  serialized JSON inside `FailureTrigger.evidence_json`
  (`packages/msgs/msg/FailureTrigger.msg:46`), so the schema change is the
  whole wire change. (`WorldCollision.msg` is the kernel's *input* — world
  capsules — not an evidence mirror.)
- HAL recorder: `sim.estop_ground_truth_snapshot`
  (`python/hal/src/openral_hal/sim_sensor_bridge.py:1377` — *(round-3)* the
  definition; `:126` is only the `__all__` entry) — #177 added the payload
  pose here; the joint half lands next to it.

**Path A — graded response.**
- The scale/shrink seam: the accepted chunk is republished at
  `safe_pub_->publish(*msg)` (`lifecycle_kernel.cpp:1057`); the distance to
  grade on is already folded per pair by `fold_pair`
  (`cpp/openral_safety_kernel/src/collision.cpp:949`) into
  `sweep_min_distance`.
- The machinery to generalize, not rebuild: the advisory band's non-latching
  path (#176, shipped by PR #179) —
  `CollisionHit::advisory` (`collision.hpp:431`; doc block at `:414`), the depth bound
  `place_advisory_depth` (`collision.hpp:290–293`), and the consecutive-refusal
  cap (`lifecycle_kernel.cpp:243`, decision at `:793`, param re-read at
  `:1399`). The new scaling-parameter family sits beside
  `place_advisory_max_consecutive`.
- **Precondition audit, preliminary result:** grepping every policy adapter in
  `python/sim/src/openral_sim/policies/` for velocity terms finds hits in
  exactly one adapter — `xvla.py:203` feeds **gripper `qvel`** (2-DoF) and
  *(round-3)* `xvla.py:207` also feeds **arm `joints["vel"]`** into its
  observation.
  XR-1 / π0.5 / the state assemblers
  (`packages/openral_rskill_ros/openral_rskill_ros/`,
  `openral_hal/proprio_snapshot.py` and the per-robot HAL modules)
  assemble position-only proprio; no arm `qvel` reaches any observation.
  Caveats: this is a grep-level pass for velocity terms only — step-index /
  elapsed-time signals are not yet audited, and XVLA must be excluded from
  Path A or re-analyzed before it ships there.

**Path B — declared-target geometry.**
- Declaration: `PlaceDeclaration` (`schemas.py:2592`) + its IDL
  (`packages/msgs/msg/PlaceDeclaration.msg`) grow the geometry field; the wire
  for the measured region already exists (`packages/msgs/msg/PlaceRegion.msg`).
- Kernel ingest: mirror `ingest_place_region` (defined at
  `cpp/openral_safety_kernel/src/collision.cpp:1734`, declared at
  `collision.hpp:877`, called from `lifecycle_kernel.cpp:2005`) — same
  bounds-checked shape (finite, positive, <= 1.5 m per side, <= 8 m3).
- Narrow phase to reuse (in-house, per §4.1's coal downgrade): the staged
  26-DOP → exact-hull check already in `collision.cpp` (~`:370–426` containment,
  `:1250` hull-to-cell distance) — Path B points it at the declared body's
  primitives instead of cells.

**Path C — force gating.** Two seams already exist and are *dormant*:
- `SafetyEnvelope.contact_force_threshold_n` (`schemas.py:897`, default
  30.0 N) is loaded and min-folded by
  `packages/openral_safety/openral_safety/envelope_loader.py:394` — and
  *(round-3, corrected)* it is plumbed all the way into the C++ envelope
  (ROS param at `lifecycle_kernel.cpp:143`, field at `envelope.hpp:70`,
  loaded at `envelope.cpp:80`) but **no check ever reads it**: zero
  enforcement consumers, so the value dies one hop later than first stated.
  Wiring that last hop is the core of Path C. (The 30.0 N default
  predates the ISO reading in §10; it is conservative against the 140 N hand
  limit but should be re-derived per-declaration when wired.)
- `WorldState.contact_forces` (`schemas.py:3323`) — the transport is declared
  in the contract with **zero producers and zero consumers** today.
- Sim force source: the HAL already walks MuJoCo contacts for
  `sim.estop_ground_truth_snapshot`; adding `mj_contactForce` extraction plus
  the calibration knob (§10's Path C caveat) happens in
  `python/hal/src/openral_hal/sim_sensor_bridge.py`, feeding the dormant
  `WorldState.contact_forces`.
- Per-declaration threshold: a new field on `PlaceDeclaration`, not a config
  default — per the order-of-magnitude budget spread in §12/Q2.

---

## 11. What to remove, replace, or stop

Consolidated from the PR history (§2), the alternatives assessment (§9), and
the community's dead ends (§7.4). "Stop" verdicts on whole themes are in §2.2;
this section is the component-level list, each with its dependency check.
Locations re-verified against `master` 2026-08-30; corrections to the original
citations are marked *(verified)* inline.

**Stop investing in (per §2.2 and §7.4):** octomap-clearing fidelity as the
false-positive fix (12-year-unsolved upstream, maintainer-confirmed Nov 2025 —
route around it with declared/attached geometry instead); extending
`tight_geometry` past link1/link2 (27/72 ceiling measured, the map term
dominates); extending the ADR-0097 margin-reduction mechanism (binding on 2/15
stops — ship the declared target's geometry instead, §10 Path B); further
harness building; further Nav2-side payload modelling; padding/offset-style
tuning knobs anywhere (community: "very difficult/impossible to tune" since
2013, never resolved quantization-vs-contact ambiguity).

1. **Per-pixel `mj_ray` depth cast (#111) — replace with the batched `mj_multiRay`,
   now that #180 has landed.**
   *What:* `_cast_depth_rays` in `python/sim/src/openral_sim/backends/depth_camera.py`
   *(verified: defined at `:118`; callers `synthesize_depth_pointcloud:344` and
   `synthesize_depth_frame:440`)* casts one `mj_ray` per strided pixel.
   *Why:* its entire justification was that `mj_multiRay`'s body-BVH cull skips
   visual-only geoms and so reports free space where a surface is. #180 now filters
   exactly those geoms out of the cast before it runs, and #180's own "Not done here"
   section says so: *"since the `mj_multiRay` body cull only ever mis-skipped
   non-collidable geoms, the ~1.9× premium the synth pays for per-pixel `mj_ray` may
   now be recoverable."* The premium is measured: 6.0 / 18.6 / 55.6 ms per frame at
   4 / 301 / 1201 geoms, versus a 5 Hz budget.
   *Depends on it:* `synthesize_depth_pointcloud`, `synthesize_depth_image`, and the
   two `test_depth_camera_synth.py` assertions #180 already inverted.
   `synthesize_laser_scan_2d` in `robocasa.py:1392` *(round-3)* casts per-beam for a
   different reason (self-hit re-casting) and must not be touched.
   *Risk:* medium. The claim "the cull only ever mis-skipped non-collidable geoms"
   is stated in #180 but not proved there; it needs the same per-ray exhaustive
   comparison #180 itself ran (16,384 rays × 4 scenes) before the swap. The failure
   direction is unsafe (missing geometry), so this is a measure-first change.

2. **`range_min_m: 0.55` on `panda_mobile`'s `base_scan`
   (`robots/panda_mobile/robot.yaml:471` *(verified)*) — remove or reduce to the
   sensor's real minimum.**
   *What:* a blunt radial cutoff on the 2-D lidar.
   *Why:* #143 documented it precisely — *"a blunt radial cutoff that deletes every
   real obstacle inside 0.55 m in every direction, to hide a chassis whose
   circumscribed radius is 0.43 m"* — and then shipped `payload_scan_filter_node`,
   which removes self-returns by proving they lie inside the manifest's own chassis
   polygon, fail-closed. The polygon filter supersedes the cutoff; the cutoff was
   never lowered afterwards. The comment block above it (lines 446–448) still
   describes it as the self-filter, which #144 corrected for the *depth* sensor but
   not for the lidar.
   *Depends on it:* `RobotDescription.lidar_sensor` → HAL `scan_*` params and
   `sim_e2e.launch.py`. `panda_mobile_vslam` declares no 0.55 cutoff, so the change
   is one manifest line plus whatever pins it. *(round-3)* Note the same file
   contradicts itself: the `front_depth` block's comment (`robot.yaml:487–504`)
   explicitly says `range_min_m` is **not** the self-filter (identity-based
   exclusion runs first), while the `base_scan` block (`:446–448`) still
   describes it as one — resolve both comments when the value changes.
   *Risk:* low-to-medium and in the *safe* direction — lowering it puts more
   obstacles into SLAM and the collision monitor, which is the conservative
   direction, but it will surface chassis returns wherever the polygon filter is not
   running (a manifest-less bringup degrades to "remove nothing" by design). Land it
   with a live round on the #185 carry scene.

3. **`CollisionHit::advisory` / the place advisory band (#176, shipped by PR #179) — replace with the
   survey's Path A (graded response scoped to the majority class).**
   *What:* `advisory` on `CollisionHit`, `place_advisory_depth`,
   `place_advisory_max_consecutive`, and the eight `PlaceAdvisoryBand.*` gtests in
   `cpp/openral_safety_kernel/{include,src,test}` *(verified: `collision.hpp:285–293`;
   `advisory` field at `:431`, doc at `:414`; `lifecycle_kernel.cpp:243`, `:793`, `:1399`)*.
   *Why:* it is measured inert — zero firings across 20 scene runs, and the same
   battery shows 9 of 15 stops are a robot link against world occupancy at −0.29 to
   −11.34 mm, i.e. entirely inside map discretisation and entirely outside this
   mechanism's scope. §10 Path A proposes exactly the generalisation: a
   velocity scale from the `min_distance` / `sweep_min_distance` the kernel already
   computes, applied in the band between `margin + threshold` and `margin`, with the
   latch at true penetration untouched. That covers the majority class the advisory
   band cannot reach.
   *Depends on it:* `lifecycle_kernel.cpp` (the non-latching drop path and the
   `safety.collision_advisory` log/span), `fold_pair`'s severity ranking, and the
   kernel README's topic contract.
   *Risk:* **do not delete outright.** The non-latching-drop path and the severity
   ranking are the two hard parts of Path A and PR #179 already built and tested them;
   deleting them would mean rebuilding them. The right move is to widen the five
   bounds rather than remove the mechanism — and note that the band has never had its
   safety-WG reviewer, hazard entry or sign-off, so any change here re-enters that
   queue regardless. `place_advisory_max_consecutive: 0` is the documented exact
   rollback if the WG wants it off meanwhile.

4. **`mj_geomDistance` on the support-contact attestation path
   (`python/hal/src/openral_hal/_sim_attachment_evidence.py:649` *(verified — the
   call is live inside `_probe_support_hits`, defined at `:616`)*) — replace with
   `openral_hal.convex_distance.convex_geom_distance`.**
   *What:* the probe that measures payload↔support penetration and produces the
   `SupportContactWitness` the kernel exempts on.
   *Why:* #170 established that `mj_geomDistance` returns confidently wrong values
   for RoboCasa-fixture-vs-`panda_mobile`-mesh pairs in two silent modes, replaced it
   on the evidence path, and withdrew every verdict that rested on it (caveat 8). The
   witness path was not converted. Unlike the evidence path this one is a **safety
   input** — an attestation earns an exemption in the kernel — and #170's measured
   failure direction is *toward closer*, which on this path means attesting a contact
   that is not there.
   *Depends on it:* `_probe_support_hits` and its callers in the same module; the caps
   `_SUPPORT_MAX_PENETRATION_M` (0.01) and `_SUPPORT_PROBE_GAP_M` (0.001), which are
   also the producer half of a hand-maintained kernel mirror
   (`docs/methods/14-duplication-watch.md`).
   *Risk:* low to convert, but the *finding* matters more than the fix: this is a
   known-defective instrument on a path that grants exemptions, and nothing in the
   evidence ledger flags it. Two honest caveats on my own claim — the probe runs at a
   1 mm window rather than the 0.1 m windows #170 characterised, and #170 measured
   1101 of 1102 pairs agreeing at the shipped windows, so the exposure is likely
   small. But #170 is explicit that *"no `distmax`, no 'only trust it below N mm'
   rule and no `ncon` cross-check separates the good answers from the bad"*, which
   is precisely the reasoning a narrow window would rely on.

5. **`SafetyPassthroughNode` (`packages/openral_safety/openral_safety/supervisor_node.py`)
   — reduce to a test double, or retire.**
   *What:* the Python safety implementation that sits "behind the same topic surface"
   as the C++ kernel *(round-3: `supervisor_node.py:143`; 39 references across
   8 Python files, 54 across 20 files counting docs/msgs)*.
   *Why:* every ADR-0096-class change is implemented twice. #119 says so explicitly
   ("so the node the C++ kernel replaces behind the same topic surface really does
   expose the same surface"), and #138 recorded three further hand-synchronised
   mirrors in the same stack. Duplication of a safety decision is the one place the
   playbook's own §1.13 argument cuts hardest.
   *Depends on it:* 20 files repo-wide, including `test_rskill_runner_node.py`,
   `test_safety_status_latched_topic.py`, `test_so100_digital_twin_e2e.py` and
   `test_safety_supervisor_per_mode.py` — it is the real component several live-ROS
   tests use precisely because it is not the kernel.
   *Risk:* high, and this is the weakest of the five. The tests that use it are
   real-component tests by design (CLAUDE.md §1.11) and the kernel is not always buildable on a
   dev host (several PR bodies note `opentelemetry_cpp_vendor` failures). Flagging it
   as the standing maintenance tax rather than proposing removal today.

**Not recommended for removal, against first appearances:** `tight_geometry` /
the 26-DOP+hull path (#166) — it is measurably faster than what it replaced and its
containment is definitional; it is simply scoped smaller than the effort that sized
it. The SAM 2.1 vision-attachment leg (#134) — it is opt-in, default-off, costs
nothing unused, and is the only real-hardware attachment path; the criticism it
deserves is that it was built before any hardware existed to run it, not that it
should be deleted. `_footprint_geometry.py` — checked *(round-3, corrected)*:
`payload_scan_filter_node.py` (which lives in `packages/openral_nav2_bringup/`,
not `python/sim/`) still imports `base_footprint_polygon` (used at `:432`)
after #186 removed the publisher, and `base_footprint_polygon` itself calls
`convex_hull_2d` internally (`_footprint_geometry.py:121,124`) — so nothing
there is orphaned, though `convex_hull_2d` is not imported by the node
directly.

---

## 12. Round-2 verification (2026-08-30) — eight questions, eight verdicts

Before committing to a direction, the open questions from §24 were pressed
against additional independent sources: the academic literature (Papers With
Code catalog + arXiv, with a logged search trail), and a second ecosystem pass
(ros.org/REPs, rosdistro, upstream source, issue trackers). Detailed evidence
is folded into §4.1, §6.1 and §10 above; this is the scorecard.

| # | question | verdict | strongest evidence |
|---|---|---|---|
| Q1 | Does graded slowdown rescue task completion? | **Strengthens Path A** — with a binding precondition | PACS: 0.04 (reactive CBF) → 0.72 (chunk-graded) vs 0.70 unfiltered ([2511.06385](https://arxiv.org/abs/2511.06385)) |
| Q2 | Force-gating thresholds and maturity | **Strengthens the axis, weakens maturity** | ISO/TS 15066 Table A.2: 140 N / 300 N/cm² / ×2 for hands; budgets vary 0.014–0.49 J per task |
| Q3 | Any VLA-specific safety layers? | **Corrects §6** — field is no longer empty | VLSA/AEGIS: +59.16 % avoidance *and* +17.25 % success, code public ([2512.11891](https://arxiv.org/abs/2512.11891)) |
| G1 | Is coal a lower-risk narrow phase? | **No — downgraded to watch item** | Near-penetration issues open (#823/#755), fixes unreleased at tag 3.0.4 (§4.1) |
| G2 | Reusable ROS 2 arm-side SSM? | **None exists** — hand-rolling confirmed | Nav2 monitor base-only+threshold-triggered; PILZ SSM never left ROS 1; ros2_controllers has nothing |
| G3 | The 15× `consider_footprint` discrepancy (#143 vs #185) | **Mechanism found, number not** | `CostCritic::score()` `continue`s on free-space points (cost < 1) before `inCollision` is ever reached — [cost_critic.cpp@jazzy](https://github.com/ros-navigation/navigation2/blob/jazzy/nav2_mppi_controller/src/critics/cost_critic.cpp); #143's microbenchmark assumed all 56 000 points reach the call. Quantifying needs live call-count instrumentation neither PR did |
| G4 | Does CRISP (or any ROS 2 learned-policy stack) ship safety? | **No** — verified from source | `learnsyslab/crisp_controllers`: soft joint-limit repulsion + torque-rate slew only; *(round-3)* no functional collision/safety code — a stray "set collision behavior" docstring and an unrelated topic-hygiene log string are the only grep hits, and `setCollisionBehavior` is never called |
| G5 | Any REP/standard to align with? | **None** | Full REP index read; only REP-2006 (cyber disclosure) is safety-adjacent; REP-2005 is a package list |

**Coverage honesty:** the ROS Discord (~5.5 k members) has no public archive or
mirror — genuinely unreachable, stated rather than approximated. Reddit r/ROS
and r/robotics: nothing relevant found. The `pwc` catalog 404'd six real arXiv
papers found via web search — **an absence claim must never rest on `pwc`
alone**. Every load-bearing citation is reproduced inline above; the academic
queries behind the absence claims, so they can be re-run:

> `pwc`: "speed and separation monitoring human robot collaboration
> productivity" (no usable hits); "ISO/TS 15066 power and force limiting
> collaborative robot" (zero relevant — pwc does not index standards work);
> "safety filter vision-language-action policy" (→ SafeVLA, VLSA, 2604.23775,
> SAFE); "safety filter causes distribution shift degrades learned policy
> performance" (→ PACS); "control barrier function deadlock oscillation near
> obstacle boundary" (→ 2411.02186; no manipulation deadlock evidence);
> "freezing robot problem conservative safety overly cautious task failure"
> (no relevant hits); "force torque contact detection distinguish intended
> contact from collision insertion"; "learned manipulation policy force
> threshold monitor contact-rich peg-in-hole safety" (→ FORGE, 2503.00287);
> "MuJoCo simulated contact force fidelity sim-to-real contact-rich
> manipulation" (→ no magnitude-validation paper). All run 2026-08-30.

---

---

*Sections 13–23 below are the **round-3 second pass** (2026-08-30): primary-source
research over methods outside §§3–7's shortlist, targeting the same three
measured gaps — (a) mm near-contact, (b) dynamic worlds, (c) self-collision.
They were briefly a separate, never-committed note, now folded in so this
file stays the single source of truth. Same conventions as §12: **"docs claim"**
= paper abstract / README / official docs; **"source shows"** = the repository
file (LICENSE, header, changelog) was fetched and read; a number that could not
be verified against a primary source is marked **unverified** and supports no
verdict. Nothing below is a plan or an ADR — every "adopt" is a candidate with a
stated cost.*

## 13. What the kernel actually needs, restated as test criteria

From the §1 and
the [validation evidence](collision-validation-evidence.md), the kernel's two
slop sources are additive and both are **structural**, not tuning errors:

1. **World-side quantisation** — a 25 mm occupancy grid, half-diagonal
   21.7 mm. A ~1 mm real contact reads as up to ~15 mm of cube penetration
   (the 2026-08-13 baguette run: −15.70 mm reported against six MuJoCo contacts
   of −0.87…−1.37 mm).
2. **Robot-side over-approximation** — per-link OBB corner slop of 27–76 mm
   protrusion ([primitive study §4.2](collision-primitive-study.md)).

So every method below is scored on three questions, and a method that fixes
only one of them fixes only half the problem:

- **(a) mm near-contact** — can it tell a ~1 mm intended touch from a real
  penetration, *against a sensed world*?
- **(b) dynamic worlds** — moving obstacles, carried payloads, base + arm.
- **(c) self-collision** — link-vs-link, at better than OBB fidelity.

Two structural constraints from CLAUDE.md bound every "adopt": the hot path is
**allocation-free C++** at 30–200 Hz (CLAUDE.md §1.5 and §2), and new dependencies
must be Apache-2.0 / MIT / BSD (CLAUDE.md §4.4). A GPU-only or Python-only method is not
disqualified as an *idea*, but it cannot be the thing that decides whether
motors stay energised.

Where a candidate would land is concrete: the kernel already exposes
`WorldModel`, `VoxelGrid`, `check_voxel_collision`, `hull_cell_distance`
(hull-vs-cell GJK with a witness) and an `AttachedModel` with support-contact
witnesses
(`cpp/openral_safety_kernel/include/openral_safety_kernel/collision.hpp`), fed
from `/openral/world_voxels`. A different world representation is a **sibling
of `VoxelGrid`**, not a rewrite.

---

## 14. Learned and analytic distance fields for the robot body

This is the direction that attacks slop source #2 — the OBBs.

### 14.1 RDF — robot geometry as a Bernstein-polynomial distance field

**What it is.** Each link's SDF is encoded as a product of Bernstein polynomial
basis functions and composed along the kinematic chain by forward kinematics,
giving `d(p, q)` with an analytic gradient in `p`. It is not an MLP; it is a
closed-form basis expansion, which is why the model is 24 KB.

**Primary sources.** Li, Zhang, Calinon et al., *Representing Robot Geometry as
Distance Fields*, ICRA 2024 — <https://arxiv.org/abs/2307.00533>; repo
<https://github.com/yimingli1998/RDF>.

**License.** **MIT** — *source shows* (`LICENSE` present; GitHub licence API
returns `MIT`). Last push 2025-07-26.

**Accuracy and rate.** *Docs claim*: whole-body mean absolute distance error
**1.41 mm** at basis order N=24 — and, unusually, **better near the surface
(1.71 mm near / 1.18 mm far)**; Chamfer distance 0.40 mm mean. The same table
puts a sphere approximation at 5.91 mm and Neural-JSDF at 23.0 mm. Query cost
0.21 ms (N=8) / 0.54 ms (N=24) on an RTX 3060.

**ROS 2 maturity.** None. Pure Python/PyTorch, no C++, no ROS package —
*source shows* (repo is 15 Python files plus model weights).

**Addresses.** (a) **yes**, at 1.4 mm on the robot side. (b) no — this models
the *body*, not the scene. (c) partially: the paper demonstrates dual-arm
mutual avoidance; single-arm self-collision is not its evaluated claim.

**Verdict: adopt as a body model, but you write the port.** 1.4 mm against
27–76 mm of OBB corner slop is the right order of magnitude, the licence is
compatible, and evaluating a fixed-order polynomial sum is exactly the shape
that survives an allocation-free C++ rewrite. You inherit nothing runnable —
no C++, no ROS 2, no realtime path — and a learned field is not conservative
(see §14.6).

### 14.2 CDF — configuration-space distance fields

**What it is.** Distance measured in *joint* space: `d(p, q)` is the minimal
joint motion in radians needed for the robot to touch point `p`. Because that
metric is even in joint space, projection/IK is one gradient step.

**Primary sources.** Li, Chi, Calinon et al., RSS 2024 —
<https://arxiv.org/abs/2406.01137>; repo <https://github.com/idiap/cdf>.

**License.** **MIT** — *source shows* (GitHub licence API `MIT`; README states
it). Python only.

**Numbers.** *Docs claim*: MAE **1.39–1.64 cm**, RMSE 2.09–2.80 cm on Franka —
an order of magnitude worse than RDF, because CDF optimises gradient
consistency, not surface precision. Inference 0.37 ms CPU / 0.49 ms GPU at
batch 1; >700,000 IK solutions/s on an RTX 3090; a reactive QP demo at 200+ Hz.

**Addresses.** (a) **no** — centimetre-scale. (b) yes, at the controller level.
(c) not addressed.

**Verdict: watch, do not adopt.** CDF's win is planning/IK throughput; the
kernel's problem is millimetres. Follow-ups worth tracking, neither with code
found: CDFlow (<https://arxiv.org/abs/2509.13771>) and time-varying CBFs over
CDF (<https://arxiv.org/abs/2412.16456>).

### 14.3 Neural-JSDF

<https://github.com/epfl-lasa/Neural-JSDF> (Koptev, Figueroa, Billard, RA-L
2022). **No licence file at all** — *source shows*: the repo contains only
`.gitignore`, `README.md`, `learning`, and the GitHub licence endpoint returns
404. Default all-rights-reserved, which is a hard blocker under CLAUDE.md §1.9.
Accuracy is reported by RDF's comparison table at 23.0 mm MAE; a competing
figure of 10.40 mm RMSE appears in a third-party paper and could **not** be
verified against the paywalled original — **unverified**.
**Verdict: irrelevant** — unlicensed, and superseded by RDF from adjacent work.

### 14.4 Learned body SDF + SVM self-collision (Zhu et al. 2024)

<https://arxiv.org/abs/2409.14955> — many tiny MLPs in parallel for the robot
SDF, **plus SVMs specifically for self-collision**, kept differentiable, with a
reactive controller for dynamic obstacles. Architecturally the closest match to
what the kernel needs (body field + self-collision + dynamics in one). *Docs
claim* up to 5× faster inference than prior methods; the abstract gives **no
error figure**, and a "0.19 cm RMSE" attribution found in search snippets could
not be located in the paper — **unverified, not used**. No code found.
**Verdict: watch** — nothing to adopt without code or a verified error bound.

### 14.5 ReDSDF — regularised deep SDFs

<https://arxiv.org/abs/2203.04739> (Liu, Tateo, Chalvatzaki et al., IROS 2022).
The contribution is far-field regularisation: a vanilla neural SDF degrades
outside its training shell, which is exactly where a safety margin lives. No
official code release found; two same-named GitHub repos exist with no licence
and no confirmed authorship — *source shows* (GitHub search API).
**Verdict: watch (concept only).**

### 14.6 The honest constraint on all of the above — Neural Implicit Swept Volumes

<https://arxiv.org/abs/2402.15281> (Joho, Schwinn, Safronov; ICRA 2024; KUKA).
A neural implicit SDF of the *swept volume* of a motion parameterised by start
and goal — i.e. a learned continuous-collision check. *Docs claim*: MAE
**3.08–4.38 mm**, 93.1 % classification at a 5 mm safety margin, false
positives 6.3 % / false negatives 0.6 %, 0.49–5.55 ms on an RTX 3090.

The load-bearing part is what the authors do next: they state the network
**cannot be guaranteed conservative**, and pair it with a geometric checker so
the guarantee is restored — with the net speedup dropping to 25–49 %.

**This is the template any learned field must follow inside this kernel**: the
learned distance is a fast *filter*, the geometric check keeps veto authority.
A network with a 0.6 % false-negative rate cannot be the thing that decides
whether motors stay energised (CLAUDE.md §1.1).

### 14.7 Scene neural SDFs (iSDF and successors) — the negative result

iSDF (<https://arxiv.org/abs/2204.02296>, Meta, RSS 2022; repo
<https://github.com/facebookresearch/iSDF>, **MIT**, *source shows* — but
**archived, last commit 2023-02-14**). *Docs claim*: **< 6 cm** SDF error,
33 ms per training step on an RTX 2080, static-scene assumption.

**6 cm is worse than the 25 mm grid the kernel already has.** The 2025–26
successors found (Neural NMPC through SDF encoding,
<https://arxiv.org/abs/2511.21312>; SAMP, <https://arxiv.org/abs/2509.11185>)
are navigation/MPC-flavoured, with no mm claim and no verified permissive repo.

**Verdict: irrelevant.** Neural *scene* SDF fusion is currently a navigation
accuracy class, not a near-contact one. Worth stating plainly, because it is the
direction that sounds most promising and is not.

---

## 15. Point clouds instead of voxels — CAPT, and the SIMD planning family

This is the direction that attacks slop source #1 — the 25 mm grid. It is the
single most relevant finding on this page.

### 15.1 CAPT — collision-affording point trees

**What it is.** A spatial data structure that is an **exact** representation of
a point cloud (no voxelisation, no occupancy inference), supporting
SIMD-parallel sphere-vs-cloud collision queries. Built once per cloud with a
legal radius range; queried as `collides(center, radius) -> bool`.

**Primary sources.** Ramsey, Kingston, Thomason, Kavraki, *Collision-Affording
Point Trees: SIMD-Amenable Nearest Neighbors for Fast Collision Checking*,
RSS 2024 — <https://arxiv.org/abs/2406.02807>, PDF
<https://www.roboticsproceedings.org/rss20/p038.pdf>. Rust implementation
<https://github.com/KavrakiLab/captree-rs>; the C++ implementation ships inside
VAMP as `src/impl/vamp/collision/capt.hh`.

**License.** **Apache-2.0** for both `captree-rs` and `vamp` — *source shows*
(GitHub licence API returns `Apache-2.0` for each; `vamp/LICENSE.txt` is the
Apache-2.0 text). `captree-rs` last push 2026-07-10, `vamp` 2026-08-27 — both
alive.

**Query semantics — *source shows*, not inferred.** The `captree-rs` README's
own usage example is `Capt::new(&points, radius_range)` then
`captree.collides(&[x, y], r)`. It is a **boolean** sphere query, not a signed
distance. The paper's §IV formalises it: the robot is a set of spheres with
known `r_min`/`r_max`, and a query sphere is tested against a leaf's
*affordance set* — a conservative superset of points that could collide with
any sphere centred in that cell.

**Numbers — *source shows* (extracted from the RSS PDF).**

- **9.89 ns** mean per query, versus 309 ns for NanoFLANN and **0.01 ms for an
  OctoMap backed by FCL** — over three orders of magnitude.
- Benchmark clouds: up to 50,000 points, **mean dispersion 7 mm – 2.2 cm**.
- Real sensor run (RealSense D455, UR5): 166,587 raw points filtered to 2,732
  at `r_filter = 2 cm`, `r_min = 1.5 cm`; **median 7.166 ms** end-to-end from
  cloud through filter, CAPT construction, planning and simplification —
  ~140 Hz, faster than the 60 FPS camera.
- **Construction is the bottleneck, not query**: the paper states construction
  scales superlinearly and that "point cloud filtering and CAPT construction
  account for the lion's share of planning time", and that CAPT construction is
  faster than OctoMap construction on the same data.
- Dynamic-scene demo: UR5 dodging human-moved pool noodles, rebuilding on every
  new cloud, at or above 60 FPS, with *no* obstacle modelling or forecasting.

**Conservativeness — *source shows*.** Lemma V.1: the tree does not alter the
collision status of any query sphere, so exactness versus brute force is
preserved. Lemma V.2: there exists `r_filter <= r_min - dispersion` such that
filtering inserts no gap larger than the minimum sphere diameter; alternatively
every query radius is padded by `r_filter`. So the filter is *provably
conservative* — but only by inflating, which reintroduces cm-scale margin
exactly where the kernel cannot afford it.

**ROS 2 maturity.** **None official.** VAMP has no ROS package and its README
never mentions ROS — *source shows*. One third-party prototype exists,
<https://github.com/anilzeybek/vamp-moveit-plunning-plugin> (MIT, 3 stars,
created 2026-06-15); its own README states it does **not** translate MoveIt
collision objects, attached objects, or octomap geometry, and that the robot
model is a **compile-time C++ type**. Treat it as a code sample.

**Addresses.** (a) **partially, and this is the key nuance** — CAPT removes the
*voxel* quantisation entirely (the world is the measured points), but the robot
is still spheres and the filter still pads. The residual error becomes
`sensor dispersion (7 mm–2.2 cm measured) + r_filter + sphere slop`, not
`21.7 mm half-diagonal + OBB corner slop`. (b) **yes** — rebuild-per-frame is
the demonstrated mode. (c) no — it is a world-side structure only.

**Verdict: adopt — the strongest candidate on this page for the world side.**
A `PointCloudWorld` sibling to `VoxelGrid` behind a `check_point_cloud_collision`
entry point shaped like the existing `check_voxel_collision`
(`collision.hpp:766`) — not `check_world_collision`, which consumes a
`WorldModel` of base-frame capsules (`collision.hpp:715`) — fed from the depth
`PointCloud2` the HAL
already publishes (`python/hal/src/openral_hal/sim_sensor_bridge.py`), removes
one of the two structural slop sources with an Apache-2.0 C++ header. The costs
are real and must be stated in any ADR: it is boolean not signed-distance (the
kernel's advisory band and witness machinery need distance — recoverable by
bisecting on radius, at ~10 ns per probe, but that is new code); construction is
per-frame and superlinear; and a point cloud is a *surface sample* with no
free-space/unknown distinction, which is a different failure mode from a grid,
not a strictly smaller one.

### 15.2 VAMP — vectorised sampling-based motion planning

<https://github.com/KavrakiLab/vamp> (**Apache-2.0**, *source shows*; 473
stars, active). *Docs claim* median **35 µs** to solve Franka MotionBenchMaker
problems on one core.

Three facts the survey brief asked to check, all *source shows*:

- **SIMD, not GPU** — `src/impl/vamp/vector/isa/` contains exactly `avx.hh`,
  `neon.hh`, `wasm.hh`; there is no CUDA in the tree. Confirmed.
- **Spheres** — `src/impl/vamp/robots/panda.hh` declares `n_spheres = 59`,
  `min_radius = 0.012`, `max_radius = 0.08` m. Mesh collision is an unchecked
  box under "Planned Features". So VAMP's accuracy floor (12–80 mm sphere radii)
  is the **same order as the kernel's OBB slop** — it is fast, not precise.
- **Discrete, not continuous** — `planning/validate.hh` builds a SIMD-wide rake
  of interpolated configurations at `n = max(ceil(distance / rake * resolution),
  1)`; Panda uses `resolution = 32`. It is a raked discrete resolution check,
  not a swept volume.

**Verdict: irrelevant as a planner, adopt its CAPT.** The kernel vets chunks, it
does not plan. Also note VAMP's robot model is generated C++ per robot — a poor
fit for a URDF-driven fleet.

### 15.3 pRRTC and Foam

pRRTC (<https://arxiv.org/abs/2503.06757>, ICRA 2026, CoMMA Lab;
<https://github.com/CoMMALab/pRRTC>, **Apache-2.0** *source shows*) is a
GPU-parallel RRT-Connect. *Docs claim* up to 10× speedup on constrained
reaching. Collision is spheres + a discrete `granularity` parameter along each
edge — same accuracy floor as VAMP, no CCD. **Verdict: irrelevant** (planner,
sphere-bounded).

**Foam** (<https://github.com/CoMMALab/foam>, MIT, arXiv 2503.13704) converts a
URDF into a spherical approximation. **Verdict: watch / cheap fallback** — if
the OBBs are to be replaced without a learned field, a fine sphere
decomposition is strictly tighter than an OBB and trivially allocation-free.
Sibling repos `batch-cc` and `SPaSM` are **unlicensed** — *source shows*.

### 15.4 RTCollisionDetection — hardware ray-traced discrete *and* continuous CD

<https://arxiv.org/abs/2409.09918> (Sui, Sentis, Bylard; ICRA 2025);
<https://github.com/Ssz990220/RTCollisionDetection>, **MIT** — *source shows*.
Built on NVIDIA OptiX / RTX ray-tracing cores. *Docs claim*: mesh-to-mesh
discrete CD **and** mesh-to-swept-volume CCD along piecewise-linear or
quadratic B-spline paths; up to 3× (discrete) and 9× (CCD) over GPU
sphere-based baselines, at 24k-triangle robot meshes against 190k+-triangle
obstacles. No ROS 2.

**Addresses.** (a) yes on the robot side (exact meshes, no primitive slop);
(b) via re-query; (c) yes.

**Verdict: watch, with a hard caveat.** B-spline swept CCD is precisely the
right *shape* for vetting an action chunk rather than a pose. But it binds the
safety path to OptiX — a closed NVIDIA SDK, which under CLAUDE.md §1.9 sits
behind a licence guard and an env var, and rules out a portable aarch64 path —
plus GPU latency jitter on a 200 Hz deadline.

### 15.5 MuJoCo Warp / Newton — explicitly disqualified by its own docs

<https://github.com/google-deepmind/mujoco_warp> (**Apache-2.0**, *source
shows*; active 2026-08-29). The collision pipeline **is** usable standalone —
*source shows*: `mujoco_warp/__init__.py` publicly exports `collision`,
`nxn_broadphase`, `sap_broadphase`, `primitive_narrowphase`, `sdf_narrowphase`,
with pluggable user SDF geoms.

Two disqualifiers, both *docs claim* from the official MJWarp page: it is
"optimized for throughput" not latency, aimed at RL sampling; and it is
**non-deterministic** — the docs answer the determinism question with "No.
There may be ordering or small numerical differences between results computed by
different executions of the same code."

**Verdict: irrelevant on the vetting path.** A check the vendor documents as
non-deterministic cannot decide whether motors stay energised. Fine for offline
scene validation.

### 15.6 Other GPU/CCD entries checked

- **Scalable-CCD** (<https://github.com/Continuous-Collision-Detection/Scalable-CCD>,
  **Apache-2.0**, *source shows*, active 2026-07) — GPU sweep-and-tiniest-queue
  plus tight-inclusion CCD, built for deformable/FEM simulation (IPC lineage),
  triangle-mesh pairs, no articulated FK. **Watch**: its tight-inclusion
  interval arithmetic is the only *provably conservative* CCD primitive found in
  this pass.
- **NeuralSVCD** (<https://arxiv.org/abs/2509.00499>, CoRL 2025, Son/Jung/Kim;
  project page <https://neuralsvcd.github.io/>) — neural encoder-decoder for
  swept-volume CD exploiting shape and temporal locality. *Docs claim* it beats
  prior SVCD on both accuracy and compute; **the abstract gives no numbers**.
  The project page links `github.com/neuralsvcd/PointObjRep`, which returned 404
  for `README.md` and `LICENSE` on 2026-08-30 — **no public code verified**.
  **Watch.**
- **RoboGPU** (<https://arxiv.org/abs/2603.01517>, March 2026, UBC) — a proposed
  GPU *hardware* unit simulated in Vulkan-Sim, which tests link OBBs against
  octree AABBs with a 15-axis SAT, i.e. the kernel's exact representation, and
  claims 3.1×/14.8× over RT/CUDA baselines. No code. **Irrelevant for adoption**,
  but a useful outside signal that OBB-vs-octree SAT is a defensible design
  point rather than an oddity.
- **OMPL GPU forks** — none found; VAMP's `scripts/cpp/ompl_integration.cc`
  bridge is the state of the art.

---

## 16. Dynamic-world mapping — what exists, and why none of it clears the wall

Scored on the same three questions. Licences below were read from the
repository's own file, not from a paper.

### 16.1 Dynablox

<https://github.com/ethz-asl/dynablox> (Schmid et al., RA-L 2023,
<https://arxiv.org/abs/2304.10049>). **BSD-3-Clause** — *source shows*
(`LICENSE`, ETHZ ASL 2023). *Docs claim* 86 % IoU at 17 FPS.

**ROS 2: no.** *Source shows* — the README states development on "Ubuntu 20.04
using ROS Noetic", the CI badge is Noetic, and the build instructions are
`catkin`. The only ROS 2 port found (`Stachq1/dynablox_ros2`) has **no licence
file at all** — unusable under CLAUDE.md §4.4; another fork is GPL-3.0.

Output is occupancy plus a *high-confidence free-space* layer: anything
observed inside known free space is declared dynamic.

**Verdict: watch the algorithm, ignore the package.** Addresses (b) only, at
LiDAR scale and 17 FPS. Its free-space conservatism is actively wrong for
near-contact — it widens the "unknown means obstacle" band. The idea already
ships inside nvblox (§16.5).

### 16.2 wavemap — the right representation, the wrong plumbing

<https://github.com/ethz-asl/wavemap> (Reijgwart et al., RSS 2023,
<https://www.roboticsproceedings.org/rss19/p065.pdf>). **BSD-3-Clause** —
*source shows*. Last push 2024-12-30.

**Resolution: it does go below 25 mm.** *Source shows* — the shipped config
`wavemap_livox_mid360_pico_flexx.yaml` sets `min_cell_width: {meters: 0.01}`
with `max_update_resolution: 0.01` for the depth camera and `0.08` for LiDAR.
It is multi-resolution: 1 cm cells only near surfaces, coarse elsewhere, which
is exactly the memory argument a uniform 25 mm grid cannot make.

**It has a true Euclidean SDF, which the README does not advertise** —
*source shows*: `core/utils/sdf/full_euclidean_sdf_generator.h` and
`quasi_euclidean_sdf_generator.h`, plus `examples/cpp/planning/occupancy_to_esdf.cc`.
**But it is a batch, whole-map, `const` one-shot generator** — no update-in-place
API. At 30–200 Hz that is a hard blocker.

**ROS 2: no.** *Source shows* — `main` has `interfaces/ros1/` only; an unmerged
`feature/ros2` branch exists whose HEAD is 2023-12-04 and whose
`wavemap_ros/package.xml` *still* declares `catkin`, `roscpp` and `rosbag`. The
port was abandoned with only the library converted.

**Addresses.** (a) at the representation level, yes — the only mapper here that
does. (b) no. (c) no.

**Verdict: watch.** The ROS-agnostic `library/cpp/` is portable and BSD, but
adopting it means porting it *and* writing an incremental SDF. That is a
project, not an integration.

### 16.3 Bonxai — a faster wrong answer

<https://github.com/facontidavide/Bonxai>, active (2026-07-17), 866 stars.

**Licence correction: MPL-2.0, not Apache** — *source shows* (`LICENSE` is the
Mozilla Public License 2.0 text). File-level weak copyleft; linking into
Apache-2.0 code is permitted but it is **not on the allowlist** and needs TSC
review. Worse, the repo-root `package.xml` (which *is* `bonxai_ros`) declares
`<license>TODO: License declaration</license>` — *source shows*. An undeclared
package licence is a provenance defect, not a formality.

Default resolution `0.02` m with a log-odds sensor model — an octomap clone.
*Docs claim* 22× faster creation, 4.5× faster update, 5.6× faster iteration
than octomap. No SDF anywhere in the tree. README states it is "primarily for
educational purposes" and is **not thread-safe for writes** — *source shows*.
No CI, not on the ROS index.

**Verdict: irrelevant.** Addresses none of (a)/(b)/(c) — it makes the existing
representation faster, and speed is not the reported bottleneck.

### 16.4 vdb_mapping — best ROS 2 hygiene here, wrong output type

<https://github.com/fzi-forschungszentrum-informatik/vdb_mapping> +
`vdb_mapping_ros2`. **Apache-2.0** — *source shows* (core repo licence;
`vdb_mapping_ros2/package.xml` declares `Apache-2.0`). **The only mapper in
this section with verified Jazzy CI** — *source shows*: the GitLab CI matrix
covers humble / iron / **jazzy** / rolling.

**A stale-fact correction that matters beyond this page: OpenVDB relicensed
from MPL-2.0 to Apache-2.0 in v12.0.0 (2024-10-31)** — *source shows*
(`AcademySoftwareFoundation/openvdb` `LICENSE` is Apache-2.0; `CHANGES` states
the relicense). Caveat: Ubuntu 24.04's `libopenvdb-dev` is 10.x/11.x and
therefore **still MPL** — a rosdep install can silently pull the copyleft
version. Pin ≥ 12.

Output is log-odds occupancy in a VDB grid; the ROS 2 wrapper exposes **no
SDF**. Version 0.0.1, README warns interfaces will change. Voxel floor and
update rate could **not** be verified from source.

**Verdict: watch.** (b) partially, (a) and (c) no.

### 16.5 nvblox dynamics — production-grade, and pointed the wrong way

<https://github.com/nvidia-isaac/nvblox> (branch `public`, 2026-07-03) and
<https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox> v4.6.0.

**Licence:** GitHub reports `NOASSERTION` for `nvblox`, but `LICENSE.md` is
**Apache-2.0** plus a **BSD-3-Clause** block covering voxblox-derived files —
*source shows*; both halves are allowlist-clean. `isaac_ros_nvblox` is cleanly
Apache-2.0 (`SPDX-License-Identifier` in `package.xml`) — *source shows*.
**But the cuRobo↔nvblox bridge is a separate repo, `NVlabs/nvblox_torch`, and
its `LICENSE` at `main` is the NVIDIA License whose §15.3 restricts use to
"research or evaluation purposes only"** (*source shows*, read 2026-08-30) —
§5's non-commercial flag on `nvblox_torch` stands. A
`nvblox_torch/` path inside the `nvblox` repo is not that bridge.

**What it does with dynamics — *source shows*,
`nvblox/include/nvblox/dynamics/dynamics_detection.h` (©2025 NVIDIA,
Apache-2.0), class docstring verbatim:** "A class for detecting dynamic
objects. It takes a depth frame and compares it to the freespace layer. If any
surface seen on the depth image falls into freespace it is assumed to be
dynamic." That is Dynablox ported to CUDA. The architecture is a dual mapper —
a static TSDF + freespace map and a separate decaying occupancy map with its
own ESDF for dynamic objects.

**Numbers — *source shows* from the shipped configs:** `voxel_size: 0.05`
(50 mm), `update_esdf_rate_hz: 10.0`, `esdf_mode: "2d"`,
`occupied_region_half_width_m: 0.15` — **dynamic obstacles are inflated by
150 mm**.

**Verdict: watch — solves (b) properly and makes (a) strictly worse.** It is
the only production-grade, Apache-2.0, Jazzy, GPU dynamic mapper found, but
every default is navigation-shaped. Near a gripper the 15 mm phantom
penetration would become ~150 mm. (c): no robot model at all. Reasonable for
the mobile base's world; keep it out of the manipulation volume.

### 16.6 The rest, briefly

- **Voxfield** (<https://github.com/VIS4ROB-lab/voxfield>, BSD-3, *source
  shows*) — non-projective ESDF, ROS 1 catkin, **dead since 2023-05**. The
  non-projective correction fixes *bias*, not *resolution*. **Irrelevant.**
- **GPU-Voxels** (FZI) — dormant since 2023-08, and `LICENSE.txt` states
  *source shows* that GPU-Voxels itself and the `icl_core` helper are
  **CDDL**, with only the build system BSD. Not on the allowlist, contested
  Apache compatibility, no ROS 2 packaging. **Irrelevant — dead and CDDL.**
  Ironic, since it is the only one originally designed for *manipulator* voxel
  collision.
- **ROG-Map** (<https://github.com/hku-mars/ROG-Map>) — **GPL-3.0**, *source
  shows*. **Rejected on licence.** The EGO-Planner mapping family shares that
  lineage and framing.
- **DSP-Map** (<https://github.com/g-ch/DSP-map>, MIT repo but
  `<license>TODO</license>` in `package.xml`, ROS 1 catkin only, *source
  shows*) — particle-based continuous occupancy with **predicted future
  occupancy**. MAV-scale, occupancy-only, no SDF. **Irrelevant to adopt**; the
  future-occupancy idea is worth reading.
- **Sub-cm mapping for manipulation specifically**: nothing released. The
  closest primary sources are **ParaMaP** (<https://arxiv.org/abs/2512.22575>,
  GPU EDT + sampling MPC on a 7-DoF arm — **no resolution or rate in the
  abstract, no code found**) and **DB-TSDF**
  (<https://arxiv.org/abs/2509.20081>, CPU TSDF whose cost is claimed constant
  in voxel resolution — **no repo, no licence found**). Both **unverified**.
- An independent corroboration of the diagnosis, worth recording: CADGrasp
  (<https://arxiv.org/abs/2601.15039>) reports *docs claim* an ablation
  choosing 5 mm voxels, with 2.5 mm giving "only marginal gains" and 10 mm
  causing "a clear performance drop". **10 mm is already too coarse for
  grasp-level geometry** — the kernel is at 25 mm.

**Section verdict.** No 2023–2026 mapper clears the mm wall at 30–200 Hz. The
two with the right ingredients pull in opposite directions: wavemap has the
resolution and a true Euclidean SDF but no ROS 2 and no incremental update;
nvblox has the engineering and the dynamics but 50 mm cells, a 10 Hz 2D ESDF
and 150 mm inflation. **§9's "nothing off-the-shelf clears the voxel wall" conclusion (point 1) survives this pass** — with the caveat that §17 finds
a way to stop resampling onto a grid at all.

---

## 17. Tesseract — the one library that beats MoveIt's discrete FCL path

<https://github.com/tesseract-robotics/tesseract>, v0.35.0, active
(2026-08-24).

**Licence.** Multi-licensed **Apache-2.0 + BSD-2 + BSD-3**, marked per file —
*source shows* (`LICENSE`, plus `LICENSE.Apache-2.0` / `LICENSE.BSD-*`); every
collision and geometry header read carried the Apache-2.0 block. GitHub's
`NOASSERTION` is an artefact of the multi-licence root. **Caveat:**
`tesseract_gui` and `tesseract_qt` are **LGPL-3.0** — GUI only, never link.

### 17.1 Continuous collision checking — confirmed in source

`collision/core/include/tesseract/collision/continuous_contact_manager.h`
declares `ContinuousContactManager` with
`setCollisionObjectsTransform(id, pose1, pose2)` — a swept hull needs both
endpoints. Backends *source shows*: `bullet_cast_bvh_manager.h` and
`bullet_cast_simple_manager.h` implement swept/cast BVH; the FCL backend has
`fcl_discrete_managers.h` **only — FCL contributes no continuous manager**.
The shipped plugin config makes `BulletDiscreteBVHManager` the discrete default
and `BulletCastBVHManager` the continuous default.

Results carry swept metadata: `cc_time[2]`, `cc_type[2]`
(`CCType_None/Time0/Time1/Between`) and `cc_transform[2]`, with the
documentation telling you to interpolate on `cc_time` to locate the contact.

**Why this matters here specifically:** at 30–200 Hz a VLA action chunk moves
the tool tens of millimetres between samples. A discrete check tunnels through
thin geometry — a failure mode the kernel's staged 26-DOP → GJK pipeline
inherits. And `cc_time` says *when* along the chunk the contact occurs, which
is a truncation point rather than an E-stop.

### 17.2 Per-pair margins — the correct shape for the intended-contact problem

*Source shows*, from the manager interface: `setCollisionMarginData`,
`setCollisionMarginPairData`, `setDefaultCollisionMargin`,
`setCollisionMarginPair(id1, id2, margin)`, `incrementCollisionMargin`, plus
`setContactAllowedValidator(...)` — a generalisation of the SRDF ACL. Contacts
closer than the margin are "in collision", order-independent.

`ContactResult` carries `distance` (negative = penetration), `nearest_points`
and `nearest_points_local`, `transform[2]`, and a separation normal;
`ContactRequest` has `ContactTestType {FIRST, CLOSEST, ALL, LIMITED}` and
`calculate_penetration`.

This is strictly richer than MoveIt's binary ACM plus a single
`contact_distance`, and it is exactly what the kernel lacks: a negative margin
on *one* declared pair (gripper finger vs the target object) while every other
pair stays conservative. §1 item 3 — "touch-links /
attached-object semantics as a first-class contract" — is this feature.

### 17.3 New in 2026: implicit-SDF collision and an SDF geometry type

Not in the docs; found by reading the tree.

`collision/core/include/tesseract/collision/implicit_sdf_collision_solver.h`
(©2026, Apache-2.0) — *source shows*, docstring verbatim: "This adapts
MuJoCo's multi-start SDF collision strategy: deterministic Halton seeds are
optimized over the overlap of the shapes' margin-expanded AABBs using a
composite collision objective and backtracking gradient descent. Contact
distance and nearest points are then recovered by projecting the converged
point onto both zero level sets." Analytic support for box, sphere, cylinder,
cone, capsule and signed distance field, with a small config
(`initial_sample_count{16}`, `max_iterations{20}`, `max_contacts{4}`,
`contact_margin`).

`geometry/include/tesseract/geometry/impl/signed_distance_field.h`
(©2026-06-10) — a first-class `SignedDistanceField : public Geometry` with a
**lazy, function-backed** mode:
`using BatchedSignedDistanceFunction = std::function<std::vector<double>(const std::vector<Eigen::Vector3d>&)>`.
Its own docstring names the intended sampler: "Use this overload to vectorize /
offload the evaluation (e.g. a batched nvblox ESDF GPU query) instead of paying
a per-point call", and "No grid is sampled up front: sampler is the field's
source of truth, so queries and each collision backend evaluate it directly
(exact, no resampling)."

**That is a door in the voxel wall.** The world stops being resampled onto a
25 mm grid; the collision query evaluates the field where it actually needs it.

Two hard constraints, both verbatim from that header — *source shows*:

- "The Bullet and FCL discrete collision backends consume this geometry through
  the shared implicit collision solver. **It is concave, so it is not supported
  by continuous/cast managers.**" → **SDF geometry and CCD are mutually
  exclusive.** You get mm-exact discrete checks *or* swept checks, not both.
- Serialization and equality force `discretize()`.

VDB interop exists (`writeSignedDistanceFieldVDB` / `...NVDB`) backed by a
**vendored `third_party/tinyvdb`, Apache-2.0** — *source shows*
(`UPSTREAM.md` pins `syoyo/tinyvdb@c0ecb7d`, retrieved 2026-07-23) — so no
dependency on AcademySoftwareFoundation/openvdb at all.

### 17.4 Octomap worlds, ROS 2 status, and real-time honesty

**Octomap:** supported (`geometry/impl/octree.h`, `OctreeSubType {BOX,
SPHERE_INSIDE, SPHERE_OUTSIDE}`, `resolution_{0.01}` default member, a custom
prune, `liboctomap-dev` as a hard dep) — but each cell becomes a Bullet box or
sphere at the *source* resolution. **Feeding the kernel's 25 mm octomap to
Tesseract reproduces the 15 mm phantom penetration exactly.** The win only
materialises via `SignedDistanceField`.

**ROS 2:** core `tesseract` is ROS-agnostic plain CMake (no `rclcpp` anywhere in
core — *source shows*). **No ROS 2 binary exists**: `index.ros.org/r/tesseract`
lists v0.35.0 with releases for **Noetic only**, "No version for distro" on
humble/jazzy/kilted/rolling — *source shows*. ROS 2 lives in
<https://github.com/tesseract-robotics/tesseract_ros2>, whose CI matrix builds
**humble, jazzy, kilted** — so Jazzy is genuinely built, source-only via
`dependencies.repos`. For a kernel that wants to link
`libtesseract_collision_bullet`, that is fine.

**Real-time:** `tesseract_monitoring/contact_monitor.h` is a *sensor-rate
monitor and visualiser*, not a hard gate — it holds a **discrete** manager, a
100 mm default margin, a `condition_variable` driven by incoming
`JointState`, and publishes markers on the same path. Structurally against
CLAUDE.md §2: managers are clone-per-thread, `contactTest` fills an allocating
`ContactResultMap`, and margin setters **throw** (`std::runtime_error`,
`std::out_of_range`) — exceptions across what would be the safety-kernel
boundary, which the C++ standard in this repo forbids. **No published latency
benchmark was found**; `tesseract_collision_benchmarks` exists and is active but
ships no fetchable results.

### 17.5 Verdict

**Adopt selectively: link `tesseract_collision` (Bullet cast BVH + per-pair
margins) and `tesseract_geometry::SignedDistanceField`; do not adopt
`tesseract_ros2`.**

- **(a) mm near-contact — partially, and better than anything else in either
  survey.** Per-pair margins plus the implicit-SDF exact path. Its *octomap*
  path is bounded exactly as today's is.
- **(b) dynamic worlds — no.** Tesseract consumes a world, it does not build
  one; the `BatchedSignedDistanceFunction` hook is how a mapper would wire in.
- **(c) self-collision — yes, cleanly.** `ContactAllowedValidator` generalises
  the SRDF ACL, per-pair margins let tight link pairs be tuned individually
  instead of against one global threshold, and CCD catches the self-collisions
  discrete sampling skips between chunk waypoints.

The prototype gate is a single number nobody has published: `contactTest`
latency against the kernel's chunk budget, on this repo's hardware, with an
SDF-backed world. That measurement is the whole decision, and it has to be
produced locally.

---

## 18. Reactive control layers — five independent stacks, one shared inequality

The single strongest convergence in this whole pass. Five well-cited,
independently developed reactive layers all express collision avoidance as the
*same* constraint: a signed distance mapped to an **allowed velocity along the
contact normal**, not a boolean.

| stack | the inequality | licence |
|---|---|---|
| NEO / Robotics Toolbox | `nᵀJ q̇ ≤ ξ(d−dₛ)/(dᵢ−dₛ) + nᵀv_obs` | MIT |
| mink | `nᵀ(J₂−J₁) q̇ ≤ gain·(d−d_min)/dt + relax` | Apache-2.0 |
| pink | CBF on `h = d − d_min` | Apache-2.0 |
| OCS2 | `d − d_min` as an MPC constraint | BSD-3 |
| fabrics | Finsler geometry leaves | **GPL-3.0 — rejected** |

All five inequalities need the same two inputs: a signed distance **and the
contact normal `n`**, out to an influence band `dᵢ`.

> **Correction (2026-09-04): the "~15 lines of C++ over the GJK output
> `hull_cell_distance` already produces" estimate was wrong and is
> withdrawn.** That routine returns a single `double`
> (`collision.hpp:691`) — a supporting-hyperplane *lower bound* on the
> distance. It emits **no contact normal** (`seed_dir` is an *input*, stage 1's
> winning separating axis), it **early-exits** as soon as the bound clears
> `margin`, and on overlap it returns the caller's `fallback` instead of a
> depth. So beyond `margin` the value is not the true distance and there is no
> `n` at all. Supplying both is a change to the narrow phase's **output
> contract**, not 15 lines over an existing one, and it has to preserve the
> conservatism argument that routine is built around: every value it can return
> today is a lower bound, so the early exit and the overlap case can only make
> the answer *more* conservative. A normal-carrying variant must inherit that
> property or it is a regression.
>
> The certified instrument added by #170/#204 already reports exactly this pair
> of outputs — `ConvexDistance.distance_m` plus `ConvexDistance.direction`,
> the latter exact even at a flush contact — but it is offline Python. It is
> the right **reference** for a C++ mirror, and the right oracle to test one
> against; it is not the hot path.

The formulation still dissolves the reported failure mode directly: a grasp at
3 mm gets a *small allowed approach speed* instead of an E-stop. This is the
same conclusion §3.4 reached from
MoveIt Servo's scaling formula, arrived at independently by five
more codebases — which is the strongest evidence on either page that graded
response is the right shape.

### 18.1 mink `CollisionAvoidanceLimit` — the best-verified near-contact geometry

<https://github.com/kevinzakka/mink>, **Apache-2.0** — *source shows*
(`LICENSE`; `pyproject.toml` `license = "Apache-2.0"`), v1.3.0, pushed
2026-08-20, 1514 stars.

*Source shows*, `src/mink/limits/collision_avoidance_limit.py`:

```python
dist = mujoco.mj_geomDistance(model, data, geom1_id, geom2_id, distmax, fromto)
row  = compute_contact_normal_jacobian(...)
if dist > min_dist: upper_bound[idx] = (gain*(dist - min_dist)/dt) + relaxation
else:               upper_bound[idx] = relaxation
sign = -1.0 if dist >= 0 else 1.0     # penetration flips the row
```

Defaults `gain=0.85`, `minimum_distance_from_collisions=0.005` (5 mm),
`collision_detection_distance=0.01` (10 mm). **`min_dist` may be negative** —
the docstring says "A negative distance allows the geoms to penetrate by the
specified amount". That is a directly usable knob for "this grasp is allowed
to touch", expressed per pair.

**Self-collision is first-class** — *source shows*: `_construct_geom_id_pairs`
applies exactly the three filters the SRDF ACL encodes (`_is_welded_together`,
`_are_geom_bodies_parent_child`, `_is_pass_contype_conaffinity_check`) and then
dedups, i.e. it *derives* an allowed-collision list from the model. Broadphase
mirrors MuJoCo's `mj_filterSphere` and its docstring is explicit that it is a
**strict** pre-filter: "it only discards pairs that would not produce a
constraint, so the resulting constraint is identical to the unfiltered
computation". Scratch buffers are pre-allocated in `__init__`.

No published rate; a circulated "~12× faster" figure could **not** be found in
the CHANGELOG — **unverified**. No ROS 2, and none intended.

**Addresses.** (a) **yes — the strongest here**: exact signed convex distance
with negative-margin support. (b) partially: geoms move with `data`, but there
is **no obstacle-velocity feed-forward** — a fast approach is seen only next
tick. (c) yes, cleanly, including the ACL equivalent.

**Verdict: adopt as design reference. NOT as an oracle.** ~330 lines of
Apache-2.0 that specify what the kernel should return instead of a boolean, and
that half of the verdict stands: mirror the algebra in C++ over the existing
GJK, and do not put Python in the 200 Hz path.

The oracle half is **withdrawn**. It rested on "mink plus MuJoCo native CCD
offline as the mm-scale regression oracle", and the code block above shows why
that cannot work: mink's distance *is* `mujoco.mj_geomDistance`, on the native
CCD path, which §22.1 and §11 item 4 record as returning confidently wrong
values on exactly the fixture-vs-link pair class such an oracle would be
pointed at. Adopting mink's formulation does not require adopting its
instrument. The oracle that belongs under this staging is
`openral_hal.convex_distance.convex_geom_distance` (§22.1), which certifies
every answer and refuses rather than guess.

Note also that mink's per-pair velocity-damper row consumes the **contact
normal** (`compute_contact_normal_jacobian`), which the certified instrument
reports directly as `ConvexDistance.direction` — exact even at a flush contact,
where differencing two coincident witness points yields nothing. Any C++ mirror
of this algebra needs that output, and today's `hull_cell_distance` does not
provide it (see §18's cost note).

### 18.2 NEO and the holistic controller (Haviland & Corke)

<https://arxiv.org/abs/2010.08686> (RA-L 2021); repos
`petercorke/robotics-toolbox-python`, `jhavl/swift`, `jhavl/spatialgeometry` —
all **MIT**, all pushed within the last two days of this research.

*Source shows*, `Robot.py::link_collision_damper`: the QP row is
`norm_h @ Je` bounded by `xi*(d - ds)/(di - ds) + dp`, where **`dp = norm_h @
shape.v` is the obstacle's own velocity**. That term is the only mechanism
found anywhere in this pass that makes a *moving* obstacle a first-class part
of the constraint rather than a re-plan trigger — directly relevant to (b).
Defaults in `examples/neo.py`: influence 0.3 m, stop 0.05 m.

**2026 update worth recording**: `spatialgeometry` now dispatches to **coal**,
not PyBullet — *source shows* (`CollisionShape.py` imports `coal`, holds a
`coal.CollisionObject`, `closest_point()` returns `(d, p1, p2)`). So NEO's
damper already rides the same GJK/EPA family the kernel uses.

**Honest caveats, both *source shows*:** the published holistic base+arm
examples (`holistic_mm_non_holonomic.py`, `holistic_mm_omni.py`) contain
**only** `joint_velocity_damper` — no obstacles, no `link_collision_damper`.
NEO's avoidance and the holistic base+arm QP are demonstrated *separately*,
never together. And there is **no self-collision helper**; `link_collision_damper`
takes a single external shape. Rate: "a few ms" per QP (*docs claim*); examples
step at 40–100 Hz. No ROS 2.

**Verdict: adopt the formulation, not the package.** MIT means the code may
even be lifted. The `+ nᵀv_obs` term is the piece to take for dynamic worlds.

### 18.3 pink `SelfCollisionBarrier`

<https://github.com/stephane-caron/pink>, **Apache-2.0** — *source shows*
(SPDX headers). The collision work lives in `barriers/`, not `limits/` —
`ConfigurationLimit` is joint-space only, which corrects a common
misattribution. `self_collision_barrier.py` builds a CBF on
`h(q) = d(p¹,p²) − d_min` (default 20 mm) over the N closest pairs from
hpp-fcl/coal via Pinocchio, with an SRDF shipped in the examples.

Docstring caveat, quoted: "Note that for non-smooth collision geometries
behaviour is undefined." Boxes and hulls *are* non-smooth, and the mm regime is
exactly where that bites.

**Verdict: watch.** (c) yes and it is the only surveyed package whose *primary*
product is self-collision; (a) partially; (b) no velocity term. Weaker fit than
mink for a kernel that must be conservative on OBBs and hulls.

### 18.4 Fabrics, RMPflow — rejected on licence and on maintenance

- **TU Delft `fabrics`** (<https://github.com/tud-amr/fabrics>, T-RO 2023
  <https://arxiv.org/abs/2205.08454>) is **GPL-3.0** — *source shows*
  (`pyproject.toml` `license = "GPL-3.0-or-later"`, `LICENSE` is GPLv3
  verbatim). **Disqualified** under CLAUDE.md §4.4 without TSC review.
  *Docs claim* up to 500 Hz replanning on a 7-DoF arm and a 10-DoF mobile
  manipulator dodging a moving human. Two things are worth stealing as *ideas*:
  `ESDFGeometryLeaf`, whose contract is "give me φ, J, J̇ as external symbols"
  — i.e. a collision term that consumes distance+gradient from someone else's
  field rather than re-deriving geometry; and the observation that
  `set_self_collision_avoidance()` is **commented out** of the default problem
  path (*source shows*), so the self-collision leaf exists but is not exercised.
  The ROS wrapper `maxspahn/fabrics_ros` is ROS 1 catkin; `tud-amr/fabrics-ros`
  is a 404.
- **NVIDIA geometric fabrics**: no open implementation found. The only shipped
  NVIDIA fabrics/RMPflow is inside Isaac Sim's **Lula**, which publishes API
  docs only — no public source, **no published licence** (stated that way
  deliberately: "no licence found", not "closed"). *Docs claim* Lula's RMPflow
  "uses collision spheres internally".
  <https://arxiv.org/abs/2405.02250> ("Geometric Fabrics: a Safe Guiding Medium
  for Policy Learning") is the closest published framing of *wrapping a learned
  policy in a medium rather than gating it* — worth reading before redesigning
  the verdict semantics.
- **`rmp2`** (<https://github.com/UWRobotLearning/rmp2>, MIT) — last push
  2021-06-26. Dead. **Verdict: irrelevant.** No maintained, permissively
  licensed RMPflow exists in 2026.

### 18.5 Pinocchio 3 + coal derivatives, and the zero-distance degeneracy

`pinocchio` is **BSD-2**, and — notably — is the **only** thing in this entire
pass with released ROS 2 Jazzy binaries: index.ros.org lists 4.1.0 for
humble / jazzy / kilted / lyrical / rolling, updated 2026-08-28, with `coal` as
a dependency.

**An honest correction to a common assumption:** Pinocchio advertises analytic
derivatives of RNEA/ABA, **not** of collision distance. Everyone — OCS2, pink —
builds the distance gradient from `nearest_points` plus joint Jacobians, which
is an approximation that **degenerates exactly at `min_distance == 0`**. OCS2's
own source carries the admission: `// TODO(perry): is there a way to calculate
a correct jacobian for the case of distanceVector = 0?` — *source shows*.

That degeneracy is precisely the regime the kernel keeps landing in. The one
paper found attacking it is **iDCOL** (<https://arxiv.org/abs/2602.03250>,
Feb 2026), "Collision Detection with Analytical Derivatives of Contact
Kinematics" — analytic derivatives of contact distance, location and normal,
explicitly regularising degenerate geometries, with a claimed open-source C++
implementation. **Repo URL and licence could not be verified from the abstract
page.** **Verdict: watch closely** — it is the only thing surveyed that attacks
the specific numerical failure at the heart of the mm problem.

---

## 19. Whole-body mobile manipulation, base + arm coupled

Short version: **there is no shipped ROS 2 stack for this, and the published
art is coarser than what the kernel already has.**

### 19.1 OCS2 — mine the self-collision algebra, do not adopt the stack

<https://github.com/leggedrobotics/ocs2>, **BSD-3** — *source shows*
(per-file headers), pushed 2026-07-20.

- **Self-collision via pinocchio/hpp-fcl: confirmed.** *Source shows*,
  `ocs2_self_collision/src/SelfCollision.cpp`:
  `violations[i] = distanceArray[i].min_distance - minimumDistance_`, with the
  gradient from `getJointJacobian(..., LOCAL_WORLD_ALIGNED)` translated to the
  nearest points via `skewSymmetricMatrix`, sign-flipped on penetration. A real
  unit test exists (`testSelfCollision.cpp`). This is the reference
  implementation of the Jacobian one would otherwise write from scratch.
- **ESDF obstacle cost: weaker than assumed.** `ocs2_perceptive` provides a
  `DistanceTransformInterface` (`getValue`, `getProjectedPoint`,
  `getLinearApproximation`) — but *source shows* the **only** constraint
  consuming it is `EndEffectorDistanceConstraint`, one row per **end-effector
  frame**. There is **no whole-body ESDF constraint in-tree**. World collision
  is end-effector-only.
- **ROS 2: community forks only.** Upstream install docs mention Ubuntu 20.04
  and `ros-noetic-*` and never mention ROS 2. `manumerous/ocs2_ros2` (BSD-3,
  2025-10-16) and `zhengxiang94/ocs2_ros2` (BSD-3, Humble, 2024-09-22) are
  **not** maintained by ETH RSL. No Jazzy fork verified. No published MPC rate.

**Verdict: watch — steal `ocs2_self_collision`, inherit its zero-distance TODO
as a known open defect.**

### 19.2 RMMI — the closest published answer, and an order of magnitude too slow

<https://arxiv.org/abs/2408.16206> (Marticorena, Fischer, Haviland,
Sünderhauf; IROS 2025). A **neural SDF** map giving a continuous differentiable
geometry, consumed by NEO's velocity-damper QP extended to a coupled base+arm
Jacobian, on a Panda + Omron LD-60.

*Source shows* (arXiv HTML): "fixed step time of 0.05 s (i.e, a control loop of
20 Hz)"; body sampling ablated at 82 spheres / 2358 points / 9476 points, with
real-world deployment using the 2358-point model because it "offered the
fastest controller frequency". Result: +25 % success on cluttered reaching.
No QP-solve or SDF-query timing published. Benchmark repo
`nmarticorena/frankie_planner` has **no LICENSE at `main` or `master`** —
*source shows* (404 on both) — so it is all-rights-reserved.

**Verdict: watch strongly, adopt nothing.** Take the architecture (one shared
QP over `[base_dof, arm_dof]`); the code is unlicensed, 20 Hz is an order of
magnitude under the kernel's floor, and a 2358-point body model is *coarser*
near contact than the 25 mm grid.

### 19.3 The rest of the base+arm field, and a calibrating data point

- **Reactive Base Control for On-The-Move Mobile Manipulation**
  (<https://arxiv.org/abs/2309.09393>, QUT) — shared QP over base+arm, 20 Hz,
  48 % real-world task-time reduction. But the obstacle model is a **2D lidar
  occupancy grid**, the gripper is queried as a **single point**, and the paper
  assumes "all obstacles detected by the lidar are tall enough that the arm
  should avoid them". No code.
  **Verdict: irrelevant to adopt — valuable as calibration.** The field's
  *deployed* answer to base+arm coupling is coarser than this repo's current
  3D voxel grid.
- **Zheng et al. 2025** (<https://arxiv.org/abs/2501.02815>) — polytopic free
  regions + AL-DDP, per-link polynomial inequalities. **No rate published**, no
  self-collision, and the repo has **no licence** — *source shows*.
  **Irrelevant.** The one takeaway is the idea of convex-decomposing *free
  space* rather than enumerating obstacles, which is a planning-layer idea.
- **Chen et al.** (<https://arxiv.org/abs/2409.14775>) — base+arm CBF-QP with
  self-collision. No rate, no code. **Irrelevant** (CBF already covered).
- **AutoMoMa** (<https://arxiv.org/abs/2604.12565>) — GPU trajectory
  *dataset generation*, 5,000 episodes/GPU-hour. Offline throughput, not a
  runtime layer; CC BY-NC-SA. **Irrelevant** — and worth naming explicitly so
  its 80× number is not mistaken for a control-rate claim.
- **Perceptive MPC** (RA-L 2020, <https://ieeexplore.ieee.org/document/9145591>)
  — ancestor of `ocs2_perceptive`; ROS 1 research code, repo metadata could
  **not** be verified. **Watch, cite, do not build on.**

**Is there any shipped ROS 2 stack for coupled base+arm avoidance?** No —
verified to the extent a negative can be. OCS2 has no official ROS 2; RMMI,
Zheng and Chen ship no released packages; `moveit_servo` is arm-only. The only
released ROS 2 Jazzy packages in this space are the **geometry libraries**
(`pinocchio` + `coal`), not a controller.

**Verdict for the kernel: build, don't adopt.** On this axis the repo is not
behind the state of the art — the published art is 20 Hz Python QPs over point
clouds and 2D occupancy grids.

---

## 20. Predictive safety filters for action chunks — and why none of them fix mm

**The headline for this section is a negative result, and it should be stated
before any of the individual methods:** every predictive, reachability, CBF or
flow-matching filter found **inherits the resolution of whatever geometry it is
handed**. Replacing a 25 mm voxel grid with an HJ value function or a barrier
function does not change that the 25 mm voxel says "penetration" when a
fingertip touches. This section is therefore mostly *watch*, with two
exceptions that are architectural rather than geometric.

### 20.1 The exception worth adopting: mode-switched constraint sets

<https://arxiv.org/abs/2608.00600> — "Grasp Execution Without a Planner:
Configuration-Space Grasp Distance Fields with Certified Safety & Guaranteed
Quality" (Enwerem et al., Aug 2026).

A four-mode hybrid controller — **reach → close → hold → lift** — with
**hysteresis** on the number of fingers in contact (band `N⁻ < N⁺` plus dwell
limits). The constraint set changes per mode:

- *reach*: the target object is a full obstacle;
- *close*: **coarse finger-link constraints drop out of the active set**, palm
  constraints shrink, and **one fingertip clearance constraint per finger
  enters at ZERO margin**;
- *hold*: arm frozen, with a wrench-quality CBF holding the risk-adjusted
  force-closure margin within `k_wq = 0.02` of its value at hold onset.

*Docs claim*: convex hulls substituted for collision meshes at load
(**0.1 ms** per query vs **330 ms** for exact mesh distance), CBF-CLF QP solved
in **0.09 ms inside a 20 ms interval**, minimum observed obstacle clearance
**6.7 mm**, median **94 %** of the synthesised grasp-quality margin retained.
Pinocchio + URDF. Licence and code: **not stated — could not verify.**

**Addresses.** (a) **structurally yes** — it does not achieve mm *sensing*, it
**stops asking the question** during grasp phases. (b) no. (c) yes, via convex
hulls.

**Verdict: adopt the idea, not the code.** The kernel's binary
accept/drop/latch at a *fixed* margin is the root cause of the 15 mm false
positive. The smallest-diff fix is a per-link, per-phase margin table with
hysteresis, and the target object's ACL entry toggled by the declared task
phase — the same shape as the existing SRDF allowed-collision list, made
phase-dependent. This is exactly what ADR-0092/ADR-0097 built bespoke; the
paper is external evidence that the shape is right, plus the hysteresis detail
the in-tree version does not have.

### 20.2 Flow-matching and diffusion filters — watch

- **SafeFlow / flow-matching barrier functions**
  (<https://arxiv.org/abs/2504.08661>, Dai et al.) — barriers constraining the
  generated trajectory over the **whole planning horizon**, training-free at
  deployment, evaluated on planar navigation and a 7-DoF arm. Licence and ROS 2:
  **could not verify.** Structurally the closest analogue to vetting a π0.5 /
  GR00T flow-matching chunk as a whole rather than per step — but it does
  nothing the kernel cannot get by evaluating its existing check across the
  whole chunk. **Watch.** Sibling: SafeFlowMatcher
  (<https://arxiv.org/abs/2509.24243>).
- **Neuro-Symbolic Safety Guidance via Constrained Flow Matching**
  (<https://arxiv.org/abs/2607.01378>, English/Zheng/Ewetz, July 2026) —
  **new since the round-2 pass (§6.1)**. Formulates safety as a minimum-norm
  constrained optimisation that corrects violations *during denoising*, i.e.
  predictive rather than reactive. *Docs claim* 82.8 % collision avoidance /
  81.6 % task success on SafeLIBERO, +6.3 / +19.8 points over single-step
  baselines, largest gains on long-horizon tasks. Collision representation is
  **not stated in the abstract**; no code confirmed. **Watch.**
- **Individual-CBF-guided diffusion** (<https://arxiv.org/abs/2606.12640>) —
  multi-agent offline RL, CC BY-NC-ND. **Irrelevant.**

### 20.3 Reachability and MPSF — irrelevant at manipulator scale

- **refineCBF** (<https://arxiv.org/abs/2204.12507>;
  <https://github.com/UCSD-SASLab/refineCBF>) — **no LICENSE file**, and the
  ROS wrapper `refinecbf_ros` is "tested solely with ROS Noetic", targeting
  Crazyflie / Turtlebot3 / Jackal. *Source shows*. The usable piece underneath
  is `StanfordASL/hj_reachability` (**MIT**, JAX). **Verdict: irrelevant** —
  ROS 1, unlicensed, 2D platforms; HJ value functions in a 7-DoF configuration
  space are not tractable and none of the papers claim otherwise.
- **Language-conditioned latent safety filters**
  (<https://arxiv.org/abs/2608.00315>, July 2026) — an HJ actor/critic
  conditioned on natural-language constraints. Read honestly, the abstract
  claims *reduced* violations and *partial* transfer, i.e. evidence, not proof.
  **Latent filters are less metrically precise than a voxel grid, not more.**
  **Verdict: irrelevant to the mm problem; watch as a possible S2-layer
  complement** — it can express "don't touch the hot pan", which no metric
  kernel can.
- **UPSi** (<https://arxiv.org/abs/2604.26836>) — reachable sets over a
  probabilistic NN ensemble with a certainty constraint. **Safe-RL benchmarks
  only, no physical robot.** **Irrelevant.**
- **Towards Safe Robot Foundation Models**
  (<https://arxiv.org/abs/2503.07404>) — a generalist policy wrapped in an
  ATACOM safe action space, demonstrated on air hockey. **Watch** (the
  foundation-model instantiation of ATACOM, which §6
  already covers generically).
- **SQ-CBF** (<https://arxiv.org/abs/2602.11049>) — superquadric SDFs via GJK
  to avoid the numerical instability of direct implicit-gradient evaluation.
  Runtime, code and licence **not verifiable**. **Watch**: a continuous SDF
  gives sub-voxel gradients, but the superquadric *fit* error to a cluttered
  scene becomes the new bottleneck and nobody quantified it.

**Standing correction:** §6 treats PACS as the
state of the art for chunk-consistent filtering. PACS is now **accepted to
ICRA 2026** (<https://arxiv.org/abs/2511.06385>, v2 March 2026); code/licence
still **could not be verified**. That is the only new fact.

### 20.4 Real-Time Chunking — a liveness guarantee, not a safety one

<https://arxiv.org/abs/2506.07339> (Black et al., Physical Intelligence,
NeurIPS 2025). RTC poses asynchronous chunking as **inpainting**, with
inference delay `d := ⌊δ/Δt⌋` and execution horizon `s`.

**Read the guarantee precisely** — *source shows*, from the PDF: "so long as
`d ≤ H − s`, this strategy will satisfy the real-time constraint and guarantee
that an action is always available when it is needed." That is an
**availability guarantee**. The paper is equally explicit that without
inpainting the transition between chunks "may be arbitrarily discontinuous and
out-of-distribution". §3.2's soft masking ramps guidance weight from 1 to 0
across the frozen prefix; hard masking underperforms, especially at small `d`.

Latency numbers from the paper, worth recording because they bound any
in-kernel design: π0 (3B) spends **46 ms on KV-cache prefill alone** on an
RTX 4090 against a 20 ms tick; the real-robot setup (π0.5, H=50, Δt=20 ms,
5 denoising steps) measures **76 ms baseline / 97 ms RTC** model latency, LAN
adding 10–20 ms, giving **d ≈ 6**.

Code: <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>
(licence **could not be verified**); RTC is documented in LeRobot
(<https://huggingface.co/docs/lerobot/rtc>).

**The kernel-relevant consequence.** The frozen prefix of length `d` — roughly
6 steps, ~120 ms at π0.5 rates — is a window in which the policy **cannot**
re-plan but the world **can** move. Nothing in RTC re-checks it. So the frozen
prefix is a *safety obligation*: it must be re-vetted against fresh perception
every cycle, with braking authority inside it. RTC and PACS are a coherent
pair; RTC alone is not. **Verdict: adopt RTC as an execution strategy, and
treat its frozen prefix as an explicit kernel contract.**

Newer chunk-execution work, listed for tracking, not fetched in depth: PACE
(<https://arxiv.org/abs/2606.00537>), action-prior denoising
(<https://arxiv.org/abs/2605.25537>), DREAM-Chunk
(<https://arxiv.org/abs/2606.18589>), adaptive inference-time chunking
(<https://arxiv.org/abs/2604.04161>).

### 20.5 Runtime monitoring — one cheap signal worth taking

- **VLA-FAIL** (<https://arxiv.org/abs/2606.21386>, Seligmann et al., KIT,
  June 2026) — two failure detectors needing **no failure data**: last-layer
  Mahalanobis distance, and — the useful one — **Action Chunk Consistency
  (ACC)**: flag a failure when *consecutive chunks become inconsistent*. Also
  introduces AUCPDT, a threshold-independent precision/recall/detection-time
  metric. Code **could not be verified**.
- **FAIL-Detect** (<https://arxiv.org/abs/2503.08558>) — distil policy
  inputs/outputs into scalar signals, then conformal prediction for
  statistically-guaranteed thresholds; best signal is a flow-based density
  estimator. **Watch.**
- **KnowNo** (<https://arxiv.org/abs/2307.01928>, CoRL 2023) — conformal
  prediction over an LLM planner's option set to decide when to ask a human.
  This is an **S2 / Reasoner-layer** control, not an actuation-path one.

**Verdict: adopt ACC as a third verdict.** It is nearly free — consecutive
chunks are already held in the RTC buffer — and it expresses something the
current binary verdict cannot: *"the policy is confused, escalate to S2"* is a
different event from *"geometry says collision, latch"*. That maps onto the
existing replanning ladder rather than the E-stop path.

**Also checked and not relevant:**
<https://arxiv.org/abs/2604.23775> ("Vision-Language-Action Safety: Threats,
Challenges, Evaluations, and Mechanisms") is an **adversarial-security** survey
— poisoning, backdoors, patches, jailbreaks — not physical safety. It does
**not** supersede this survey. By contrast,
<https://arxiv.org/abs/2512.11908> ("Safe Learning for Contact-Rich Robot
Tasks: A Survey", IIT, Dec 2025 / v2 Jan 2026) is the nearest thing to a 2026
superset of this survey and is worth reading in full; only its
abstract was verified here, and that abstract does **not** mention momentum
observers or intent discrimination.

---

## 21. Proprioceptive contact discrimination — where the mm problem is actually solvable

**This is the most important section on the page.** The kernel is trying to
infer *intent* from *geometry* at a resolution geometry cannot deliver.
Force does deliver it — with the crucial caveat that force measures **contact,
not penetration depth**.

### 21.1 What a Franka-class arm can actually measure

*Source shows* — text extracted from the official **Franka Emika Panda**
datasheet (<https://download.franka.de/Datasheet-EN.pdf>):

| quantity | value |
|---|---|
| force resolution | **< 0.05 N** |
| force repeatability | 0.15 N |
| **force noise (RMS)** | **0.035 N** |
| torque resolution | 0.02 Nm |
| **relative torque accuracy** | **0.15 Nm** |
| torque repeatability | 0.05 Nm |
| torque noise (RMS) | 0.005 Nm |
| pose repeatability | < ±0.1 mm (ISO 9283) |

**Correction that matters for this repo's hardware claims: the FR3 datasheet
does not publish any of these numbers.** *Source shows* — the full text of
<https://download.franka.de/Franka-Research-3_Datasheet_v1.1_August2022.pdf>
contains only "Force/Torque sensing: link-side torque sensor in all 7 axes" and
"Guiding force ~2.5 N". No force resolution, no noise figure. **Anyone citing
"FR3 has 0.05 N force resolution" is citing the Panda sheet.** The FR3 sheet
does give a worst-case safe Cartesian position accuracy for stopping functions
of **50 mm**, PL d / Cat. 3 on E-stop, and PFH < 1e-7/h.

**How `tau_ext_hat_filtered` is computed is undisclosed.** *Source shows* —
<https://github.com/frankaemika/libfranka/issues/91>, the only substantive
reply from Franka's own maintainer (2024-09-26): "this data is streamed from
the backend and computed within." No model, no filter spec, no accuracy figure.
Treat it as an uncharacterised black box. An independent measurement exists
(Petrea & Bertoni, IEEE 2021,
<https://ieeexplore.ieee.org/document/9589424/>) but is paywalled — its
quantitative error figures **could not be verified**, and it is the single most
valuable missing number for this decision.

**Can this discriminate a ~1 mm intended touch from a collision?**

- **On magnitude: comfortably yes, for detection.** A 0.035 N RMS noise floor
  against a 1–5 N deliberate fingertip touch is 30–140× margin.
- **On intent: no, not from magnitude alone.** A 3 N accidental brush and a
  3 N deliberate touch are the same number.
- **A caveat that must be carried:** the binding constraint is *relative torque
  accuracy 0.15 Nm*, not the 0.005 Nm noise RMS. At a ~0.5 m effective lever
  that is roughly **0.3 N of unmodelled force error** — 6× the headline
  "< 0.05 N", i.e. an order of magnitude. *(This lever-arm conversion is derived arithmetic, not
  a Franka claim.)*
- **The honest framing:** proprioception measures **force, not penetration**.
  It cannot tell 1 mm from 15 mm. It *can* tell "contact is occurring and it is
  gentle" — which is the discriminator actually needed, and it is orthogonal to
  and vastly more precise than a 25 mm voxel grid.

### 21.2 The momentum-observer lineage, and the number that bounds it

**Haddadin, De Luca, Albu-Schäffer, "Robot Collisions: A Survey on Detection,
Isolation, and Identification", T-RO 33(6), 2017** — open PDF
<http://www.diag.uniroma1.it/~labrob/pub/papers/TRO_Collision_Dec2017.pdf>.

Still canonical, and it names the missing stage: the pipeline is detection →
isolation → identification → **classification** → reaction → post-collision,
where classification means "accidental or intentional … light or severe …
permanent, transient, or repetitive". The survey also says plainly that this
decision "cannot be done purely at the control level, as global environmental
information and reasoning is certainly needed". It further establishes
(§IV-B) that the contact point and force vector can be estimated **from
proprioception alone**, with the momentum observer usable "in place of a wrist
force-torque sensor". It is honest about thresholds absorbing unmodelled
friction, quantisation and model error — its sensitivity study models ±0.3 Nm
joint torque noise, 0.2 Nm hysteresis, 0.2 Nm harmonic-drive ripple and ±0.5 Nm
friction.

**The kernel today has detection wired straight to latch, with no
classification stage at all.** That is the gap, and the reference architecture
for closing it is nine years old and uncontroversial.

**Birjandi & Haddadin, RA-L 2020** — open PDF
<http://iliad-project.eu/wp-content/uploads/papers/ModelAdaptiveHighSpeedCollisionDetectionForSerialChainRobotManipulators.pdf>.
*Source shows*, quoted verbatim: "it is due to this error that the collision
threshold is typically set to **1 Nm** in the momentum observer … the error of
the proposed solution **does not exceed 0.1 Nm**. Meaning that small amplitude
external forces (≥ 0.1 Nm) can be effectively detected. With such 0.1 Nm
threshold, the collision is **detected within 1.2 ms**." Detection delays
against a force plate: ≈3.4 / 2.5 / 0.8 ms at 0.2 / 0.5 / 2 m/s.

**But the cost is one IMU per link**, which an FR3 does not have, and no code
or licence was found. **Verdict: watch, do not build.** Start from stock
`tau_ext_hat_filtered`; revisit only if the stock signal proves too noisy.

Modern successors (variational-Bayesian Kalman external-torque estimation,
Sensors 2025 <https://doi.org/10.3390/s25206315>; momentum observer + LSTM;
deep-Lagrangian-network observers; finite-time and super-twisting observers;
SafePR <https://arxiv.org/abs/2501.17773>) are all incremental
threshold-reduction papers on the same 2017 observer. **None ships ROS 2. None
claims intent discrimination. Watch, do not chase.**

### 21.3 The highest-value line: intent from the torque *spectrum*

**"Tactile Gesture Recognition with Built-in Joint Sensors for Industrial
Robots"** — <https://arxiv.org/abs/2508.12435> (Aug 2025). Franka Emika
Research robot, **built-in joint sensors only** — no skin, no vision. *Docs
claim* **over 95 % accuracy** in both contact detection and gesture
classification via STFT-2D-CNN / STT-3D-CNN, with the key finding that
**time-frequency (spectrogram) representations significantly outperform raw
torque signals**. Class count, force levels, code and licence: **could not
verify.**

**Why this is the most on-point result in either survey.** It shows the
*temporal-spectral signature* of a torque transient — not its amplitude —
carries the intent information. A grasp closure and an accidental impact at the
same peak force have different spectra: the impact is a broadband impulse, the
grasp is a ramp. That is precisely the discriminator the kernel lacks, and a
~10 ms STFT over 7 channels at 1 kHz is a small, allocation-free C++ addition.

**Complementary, and directly about "intended vs unintended":**
**Aim-Aware Collision Monitoring** (Proper, Kurdas, Abdolshah, Haddadin,
Saccon; **RA-L 8(8):4609–4616, Aug 2023**, DOI 10.1109/LRA.2023.3284371;
<https://research.tue.nl/en/publications/aim-aware-collision-monitoring-discriminating-between-expected-an/>).
An online framework that discriminates **expected from unexpected post-impact
behaviour** by comparing an idealised rigid robot-object-environment response
against the measured one, with a **causal envelope filter** generating
classification error bounds to absorb joint and environmental flexibility,
driven by a **bandpass momentum observer**. This is impact-*aware* manipulation
— contact is intended by design — which is exactly the regime the kernel keeps
mis-classifying. **Verdict: adopt the framing** (compare against a predicted
contact response, not against a fixed margin).

### 21.4 Contact localisation — set expectations correctly

- **Contact Particle Filter** (Manuelli & Tedrake, IROS 2016) — a convex QP
  inside a particle filter finding the contact point(s) that best explain the
  measured external joint torque. Simulation on a humanoid. The classic worth
  understanding.
- **UniTac** (<https://arxiv.org/abs/2507.07980>, Brown IVL, July 2025) —
  whole-robot touch sensing from **proprioception only**, ~2000 Hz on an
  RTX 3090, localisation **within 8.0 cm on a Franka arm** / 7.2 cm on a Spot.
  Code <https://github.com/julia-fu0528/UniTac> has **no LICENSE file** —
  all-rights-reserved, **do not vendor**.
- **Iskandar, Albu-Schäffer & Dietrich, Science Robotics 9(93):eadn4008,
  Aug 2024** (DLR) — high-resolution redundant joint force-torque sensing, no
  skin, localising touch *trajectories* well enough to read handwritten letters
  and place virtual buttons on the robot surface. **The paper's spatial
  resolution and force sensitivity figures are paywalled — could not verify**,
  and no number is asserted here on the strength of the abstract alone.

**Verdict: budget for mm-accurate contact *detection*, not mm-accurate contact
*localisation*.** The published open number for proprioceptive localisation is
7–8 cm. Detection is a much easier and much better-served problem.

### 21.5 The plumbing already exists — this is the cheap rung

*Docs claim*, franka_ros2 (Jazzy):
<https://frankarobotics.github.io/docs/doc/franka_ros2_jazzy/franka_hardware/doc/index.html>
— `franka_hardware` exposes a **`ForceTorqueSensor` named
`<arm_prefix><robot_type>_tcp`** carrying **`K_F_ext_hat_K`** (estimated
external wrench in the stiffness frame) as six state interfaces
`force.x/y/z`, `torque.x/y/z`, alongside per-joint effort. The **Gazebo plugin
mirrors the same tcp wrench interfaces**, so a sim and a hardware controller
activate identically — which matters directly for this repo's sim-first test
tiers. The fetched docs do **not** list `tau_ext_hat_filtered` or
`O_F_ext_hat_K` as separate state interfaces; that is "docs do not show it",
not "it does not exist" — check `franka_msgs/FrankaRobotState`.

`ros2_control`'s `force_torque_sensor_broadcaster` (Apache-2.0) publishes
`WrenchStamped` with filter-chain support, and in-controller access goes
through `semantic_components::ForceTorqueSensor` — **no topic hop on the hot
path**.

**One landmine, and it maps onto a rule this repo already has:** the Universal
Robots driver's broadcast wrench is expressed relative to `base_link` and is
"almost always incorrect as the robot's pose changes"
(<https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/issues/235>),
with a `wrench_transformer_node` as the workaround. An external wrench in the
wrong frame is worse than none — CLAUDE.md §2's "TF2 is the only source of
coordinate frames" applies to wrenches too.

**Verdict: adopt — and note the plumbing costs nothing.** For an FR3, the
external wrench arrives as a standard `ForceTorqueSensor` semantic component
inside `ros2_control`, consumable in a C++ controller at kernel rate with **zero
new dependencies**. No momentum observer needs to be written, and no new
package needs to be created.

### 21.6 Gripper-level grasp confirmation — a phase signal, not a safety signal

*Docs claim*, libfranka `Gripper::grasp()`
(<https://frankarobotics.github.io/libfranka/0.15.0/classfranka_1_1Gripper.html>):
returns true iff `(width − epsilon_inner) < d < (width + epsilon_outer)`, with
**defaults `epsilon_inner = epsilon_outer = 0.005` m — a ±5 mm width band**.
Force is a *commanded* parameter, not a measured confirmation.

So it is a **width predicate**: "something roughly the expected size is between
my fingers". It cannot discriminate a 1 mm touch (its own tolerance is 5 mm)
and says nothing about arm-link contact. **Verdict: adopt as a *phase* signal
only** — it is a perfectly good input to the §20.1 mode switch ("we are in
`hold`, relax the target-object constraint"), and worthless as a contact-force
measurement.

Adjacent 2026 work, both with **unverifiable code/licence**: "Current as
Touch" (<https://arxiv.org/abs/2607.03529>) and **FACTR 2**
(<https://arxiv.org/abs/2606.12406>), whose NEXT component learns external
joint torque from **motor current on commodity arms with no torque sensors**
from 10 minutes of free-motion data. **Watch** — relevant only if OpenRAL ever
targets a current-only arm.

### 21.7 MuJoCo contact-force fidelity — an honest negative, and a blocker

**No 2023–2026 paper was found validating MuJoCo or MJX contact-force
*magnitudes* against real force-torque measurements.** Five search phrasings
were tried. What *was* established:

- **MuJoCo's own documentation frames its contact model as a deliberate
  approximation, not a calibrated force model** — *docs claim*
  (<https://mujoco.readthedocs.io/en/stable/computation/index.html>): it "drops
  the strict complementarity constraint at the heart of the LCP formulation",
  so "force and velocity in the contact normal direction can be simultaneously
  positive", justified because "all physical materials allow some deformation".
  The docs nowhere claim forces are in calibrated real Newtons, and put the
  burden of physical validity on the user's `solref`/`solimp` choices.
- The strongest indirect evidence is **"Direction Matters: Learning Force
  Direction Enables Sim-to-Real Contact-Rich Manipulation"**
  (<https://arxiv.org/abs/2602.14174>, 2026), whose entire premise is that
  force *magnitudes* are "highly sensitive to simulation inaccuracies" while
  force *directions* "remain robust across the sim-to-real gap". It is an
  argument *from* the magnitude gap, not a measurement of it, and it gives no
  quantitative magnitude error.

**Verdict: record this as a blocker.** The sim path cannot produce a
trustworthy real-Newton analogue today, and no published work says otherwise.
The correct move is to have the sim path emit **contact-occurring boolean plus
force direction**, and to calibrate any magnitude threshold on hardware only.
Anything else fabricates a number, which CLAUDE.md §1.2 forbids. This
strengthens — and gives a citation to — §10 Path C's standing caveat that its sim seam ships "a calibration knob, not a claim".

---

## 22. Opportunistic finds outside the eight directions

### 22.1 `mj_geomDistance` is already installed — and must NOT be the oracle

**This subsection's original verdict is withdrawn.** It read "adopt as the
offline oracle first", on the reasoning that MuJoCo ships a mesh-exact signed
distance and is already a dependency. The premise is true and the conclusion is
wrong, for a reason that was already in this repository when the section was
written and is recorded in §11 item 4 and in standing caveat 8 of
[the validation-evidence ledger](collision-validation-evidence.md): **that call
is the defective instrument PR #170 removed from the evidence path and PR #204
(issue #190) removed from the last safety path.** The corrected verdict is at
the end of this subsection. The description below is kept because the facts are
right and the trap is instructive.

**MuJoCo ships `mj_geomDistance`, and MuJoCo is already in this repo's
dependency set** (`pyproject.toml` pins `mujoco>=3.8.0`; the only other
geometry dependency in the tree is `trimesh`).

*Docs claim*, MuJoCo API reference
(<https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html>):

```c
mjtNum mj_geomDistance(const mjModel* m, mjData* d, int geom1, int geom2,
                       mjtNum distmax, mjtNum fromto[6]);
```

"Returns the smallest signed distance between two geoms and optionally the
segment from geom1 to geom2" — i.e. a **signed distance plus a witness
segment**, bounded above by `distmax`. Positive is separation, negative is
penetration.

*Source shows*, upstream `doc/changelog.rst`:

- `mj_geomDistance` was added in **3.1.6 (2024-06-03)**, together with three
  MJCF **sensors** — `distance`, `normal`, `fromto` — so a model can *emit*
  the signed distance between two named geoms or bodies every step, with a
  `cutoff`.
- The `nativeccd` flag (GJK/EPA replacing MPR) was added in **3.2.3
  (2024-09-16)**, recommended in 3.2.5, and **became the default in 3.3.0
  (2025-02-26)**: "The native convex collision detection pipeline introduced in
  3.2.3 and enabled by the `nativeccd` flag, is now the default." The
  `mj_geomDistance` docs warn that "distances are inaccurate when using the
  legacy CCD pipeline, and its use is discouraged".

**And that last inference is exactly the trap.** "The legacy path is
inaccurate, therefore the pinned default is accurate" is the reasoning this
survey originally made, and #170 falsified it by measurement: under
`mujoco 3.8.0` on the **default native-CCD path** the call returns
`+0.000000` for `robot0_link7_collision` against
`fridge_right_group_freezer_door_main` on `robocasa_fridge_drawer` layout 9,
where the certified truth is `+0.148512 mm`, writing a 126.264 mm witness
segment whose endpoints lie 526.6 mm and 432.9 mm outside the two geoms it
claims to touch. Displacing the link by one picometre returns the right
answer, so it is a degenerate *configuration*, not a distance regime — and a
scene's reset pose is where such configurations live, because fixtures are
placed on exact axis-aligned numbers. #170 states it flatly: **no `distmax`,
no "only trust it below N mm" rule and no `ncon` cross-check separates the
good answers from the bad.** The libccd path is worse and unbounded in
`distmax` (−57 mm through a 48 mm panel). Re-measuring the checked-in rounds
found four recorded `0.000 m` readings whose certified values are +14.8,
+82.2, +98.8 and +107.9 mm, the last on a solid↔solid pair.

**Why it looked compelling.** The kernel's per-link OBBs exist because there
was no cheap mesh-level distance available, and here is one that is in-tree,
Apache-2.0 and mesh-exact. The original text also called its `fromto` witness
"the same shape as the existing `GjkWitness`". **That comparison is wrong**:
`GjkWitness` (`collision.hpp:661`) stores hull vertex *indices* and cube-corner
sign codes as a warm-start hint — its own comment says "never Minkowski
difference points" — so it is a cache, not a nearest-point pair, and the two
are not interchangeable.

**Corrected verdict: do NOT adopt `mj_geomDistance` as the oracle, or as
anything else on an adjudication or permission path.** The requirement it was
proposed for is real and still stands — the 26-DOP → hull staging does need an
mm-scale regression oracle — but the instrument for it already exists in-tree
and is the one that replaced this call:
**`openral_hal.convex_distance.convex_geom_distance`**. It answers the same
question about the same convex bodies MuJoCo would collide, and it *proves*
each answer: separated pairs carry a separating-axis certificate, overlapping
pairs are solved by exact SAT, round types with no ball form are bracketed by
inscribed and circumscribed polytopes, and anything it cannot certify is
returned as an explicit refusal rather than a plausible number. It costs ~1 ms
per mesh↔box pair against ~0.8 µs, which is affordable precisely because an
oracle runs offline. It adds **no** dependency, so the "rung 5 of the ladder,
not a new stack" argument survives intact — it simply points at a different
function.

`mj_geomDistance` keeps exactly one legitimate use in this repo, and it is
already in the tree: as the *subject* of the two regression tests that assert
the defect still reproduces
(`tests/sim/safety/test_geom_distance_instrument_robocasa.py`,
`tests/sim/safety/test_support_probe_instrument_robocasa.py`), so that an
upstream fix is noticed rather than silently assumed.

The two limits the original verdict listed are now moot for the oracle
question, but are recorded because they apply to any future MuJoCo-backed
proposal: it needs `mjModel`/`mjData`, so the *world* side would have to exist
as MuJoCo geoms — fine for the robot and declared objects, not for a
depth-derived world, where runtime geom insertion is not supported; and it is a
CPU per-pair call with no published rate on this repo's hardware.

### 22.2 Contact-implicit MPC — contact as a decision variable, not a violation

**C3 / C3+ (consensus complementarity control)**, Posa's DAIR lab. Push
Anything (<https://arxiv.org/abs/2510.19974>, project
<https://dairlab.github.io/push-anything/>) reports *docs claim* a **99.9 %
(700/701)** single-object and **92.5 % (210/227)** multi-object success rate,
with control rates of **~14 Hz single-object and 8–15 Hz multi-object**, over
meshes reconstructed by BundleSDF. Code lives in `DAIRLab/dairlib`, **MIT** —
*source shows* (`LICENSE`, "MIT License, Copyright (c) 2023 Michael Posa") —
though it is built on Drake.

**Verdict: irrelevant to the kernel, important as a framing.** 8–15 Hz is far
below the 30–200 Hz floor and this is a controller, not a filter. But it is the
clearest existence proof that a whole research line treats contact as a
*decision variable* rather than a *safety violation*. A kernel whose only
verdict on contact is "latch" is structurally at odds with contact-rich
manipulation, and that is the same conclusion §20.1 reaches from the grasp side.

### 22.3 A cautionary "sub-millimetre" claim, checked

**"Learning Fast, Tool-aware Collision Avoidance for Collaborative Robots"**
(<https://arxiv.org/abs/2508.20457>, RA-L, DOI 10.1109/LRA.2025.3579207) is
frequently summarised as achieving "sub-millimetre accuracy". *Source shows*,
from the paper's HTML: the sub-millimetre figure is **task-space position
tracking error** in nominal operation ("open areas consistently exhibit errors
around 0.01 mm") — **not** collision discrimination. The actual geometry is "a
voxel grid with 0.05 m resolution" over a 0.6 × 1.0 × 0.6 m workspace, 50 Hz
control, ~6 ms per action, on an Indy7 with a RealSense D435; all authors are
at Neuromeka; no code or ROS package is mentioned.

**Verdict: irrelevant, and recorded deliberately.** 50 mm voxels. This is
exactly the class of claim that must be followed to the primary source before
it is allowed to influence a safety decision (CLAUDE.md §1.2).

### 22.4 Self-filtering the robot and its payload out of depth — a ROS 2 option

§3.3 covers MoveIt's octomap self-filter. A
standalone ROS 2 alternative exists: **`leggedrobotics/robot_self_filter`**, a
ROS 2 port of the classic self-filter — *source shows*, `package.xml` declares
`<license>BSD-3-Clause</license>`, `ament_cmake`, `rclcpp`, `tf2`, `urdf`. The
widely cited `ctu-vras/robot_body_filter` (clip / contains / **shadow** tests,
organized-cloud NaN marking) remains **ROS 1**. No binary Jazzy release was
verified for the ROS 2 port — the distro names on its index.ros.org page are
site navigation, not confirmed releases.

**Verdict: watch.** Marginal on its own, but the *shadow test* — removing
points seen *through* a robot link — is the mechanism most directly relevant to
the carried-payload false positives, and no equivalent exists in-tree.

### 22.5 The force-limiting standard §8 cites has been superseded

§8's comparison table rests its only "yes" on
**ISO/TS 15066**. The 2025 revision of the ISO 10218 series is published and
folded ISO/TS 15066's power-and-force-limiting requirements in
(**ISO 10218-1:2025** and **ISO 10218-2:2025**), retiring the TS as a
standalone document — confirmed in the round-3 verification pass (the same one
behind §10 Path C's citation update). Cite ISO 10218-2:2025 in hazard-log
entries, with TS 15066 kept for the Annex A body-model tables (Table A.2/A.3
values re-verified against the standard's own PDF, §12).

---

## 23. Comparison table — scored against the kernel's three needs

"mm near contact?" asks whether the method can distinguish a ~1 mm intended
touch from a real penetration, **against a sensed world**. Licences are as
verified above.

| method | what it changes | (a) mm near-contact | (b) dynamic world | (c) self-collision | rate / resolution | ROS 2 | licence | verdict |
|---|---|---|---|---|---|---|---|---|
| **CAPT** (§15.1) | world: points instead of voxels | **partially** — no voxel quantisation; sensor dispersion 7 mm–2.2 cm + sphere/filter slop remains | **yes** — rebuild per frame, 60 FPS demoed | no | 9.89 ns/query; 7.2 ms end-to-end at 2.7k points | none (3★ prototype only) | Apache-2.0 | **adopt** (world side) |
| **Tesseract collision** (§17) | per-pair margins + swept CCD + SDF geometry | **partially** — best available; per-pair margins + implicit-SDF exact path; its octomap path is not | no (consumes a world) | **yes** — `ContactAllowedValidator` + per-pair margins + CCD | no published latency | source-only, Jazzy in CI | Apache-2.0 (+BSD) | **adopt selectively** |
| **velocity damper** (§18, NEO/mink/pink/OCS2) | verdict semantics: distance → allowed normal velocity | **yes** — graded, negative margins expressible | **yes** with NEO's `+nᵀv_obs` term | yes (mink derives the ACL) | ~15 lines over existing GJK | none | MIT / Apache-2.0 / BSD-3 | **adopt the formulation** |
| **MuJoCo `mj_geomDistance`** (§22.1) | mesh-exact signed distance + witness | **no — measured wrong on this pair class** (#170) | modelled objects only | n/a | CPU per-pair, ~0.8 µs | n/a (already a dep) | Apache-2.0 | **REJECTED — verdict withdrawn; use `openral_hal.convex_distance` instead** |
| **external wrench** (§21.5) | measures contact, not proximity | **yes for detection** — 0.035 N RMS vs 1–5 N touch; cannot measure depth | n/a | n/a | control rate, zero new deps | **shipped** (`franka_ros2` Jazzy, `ros2_control`) | Apache-2.0 | **adopt** |
| **torque-spectrum intent classification** (§21.3) | separates intended from accidental at equal force | **yes, structurally** | n/a | n/a | ~10 ms STFT, 7 ch @ 1 kHz | none | unverified | **adopt the idea** |
| **phase-switched constraint sets** (§20.1) | stops asking the geometric question during grasp | **yes, structurally** | no | yes | QP 0.09 ms in 20 ms | none | unverified | **adopt the idea** |
| **RDF Bernstein body field** (§14.1) | robot: 1.41 mm field replaces OBBs | **yes** (robot side only) | no | partially | 0.21–0.54 ms/query on GPU | none | MIT | **adopt, port yourself** |
| **wavemap** (§16.2) | 1 cm adaptive cells + true Euclidean SDF | at representation level, yes | no | no | batch SDF generation only | **no** (dead ROS 2 branch) | BSD-3 | watch |
| **nvblox dynamics** (§16.5) | freespace-intrusion dynamic detection | **no** — 50 mm cells, 150 mm inflation | **yes** | no | 10 Hz ESDF, 2D sliced | **yes**, Jazzy | Apache-2.0 (+BSD-3) | watch (base only) |
| **RTCollisionDetection** (§15.4) | ray-traced mesh CCD along B-splines | yes (mesh-exact) | by re-query | yes | 3×/9× over GPU sphere baselines | none | MIT, **OptiX-bound** | watch |
| **iDCOL** (§18.5) | gradient that survives zero distance | targets exactly this | — | yes | unpublished | none | **unverified** | watch closely |
| **RTC frozen prefix** (§20.4) | names a ~120 ms unre-plannable window | n/a | **it is the blind spot** | n/a | d ≈ 6 steps at π0.5 rates | via LeRobot | **unverified** (`real-time-chunking-kinetix` licence could not be verified; the Apache-2.0 finding is for `openpi`) | adopt as a contract |
| **VLA-FAIL ACC** (§20.5) | third verdict: "policy confused" | no | indirectly | no | free — chunks already buffered | none | unverified | adopt the signal |
| **VAMP / pRRTC** (§15.2–15.3) | fast planners | no — 12–80 mm spheres | via replanning | via spheres | 35 µs/plan | none | Apache-2.0 | irrelevant (planner) |
| **Bonxai** (§16.3) | faster octomap | no | no | no | 20 mm default | undeclared | **MPL-2.0** | irrelevant |
| **fabrics** (§18.4) | Finsler reactive layer | no (spheres/capsules) | yes | leaf exists, disabled | 500 Hz claimed | ROS 1 only | **GPL-3.0** | rejected |
| **MJWarp** (§15.5) | GPU collision pipeline | n/a | n/a | n/a | throughput, **non-deterministic** | no | Apache-2.0 | irrelevant on the safety path |
| **GPU-Voxels / ROG-Map / Neural-JSDF / UniTac / RMMI code** | — | — | — | — | — | — | **CDDL / GPL-3.0 / unlicensed** | rejected on licence |

---

---

## 24. What this survey does NOT establish

- No latency was measured in this tree: the µs-scale coal and MoveIt numbers
  are upstream benchmarks on other hardware; a decision to adopt either needs
  the number re-measured against this kernel's chunk-rate budget on the DGX
  Spark / target hosts.
- Servo's scaling formula is shipped and read from source, but no claim is made
  that it is a *rated* safety function — MoveIt's docs make no such claim
  either, and neither does this page for the in-tree kernel.
- The two motivating drawer-opening runs are cited from their run evidence and
  are consistent with recorded failure classes (#172, the phase-dilation family
  #171/#173), but they postdate the checked-in validation rounds and have no
  in-tree fixture yet; the predicted-config and out-of-coverage-voxel defects
  should get the same recorder-side pinning the earlier corpus got.
- Whether graded slowdown (Path A) converts *this repo's* within-quantization
  stop class into completed grasps: PACS's Table I (§10) now establishes the
  effect exists and is large on robomimic + real FR3, but the magnitude on the
  RoboCasa scenes remains for the five-round battery — and the Path A
  precondition (velocity-free policy observations) has not yet been audited
  per adapter.
- The VLSA-vs-PACS disagreement is unresolved: VLSA reports a CBF layer
  *raising* success (+17.25 %) where PACS's reactive-CBF baseline collapses it
  (0.04). Setups differ (static semantic obstacle vs moving obstacle with
  formal reachability); both are cited from their own experiments and neither
  invalidates the other yet.
- MuJoCo contact-force *magnitude* is unvalidated as a real-Newton analogue
  (no paper found; FORGE re-tunes on hardware). Path C's sim seam therefore
  ships a calibration knob, not a claim.
- G3's mechanism (the free-space `continue`) is read from source but the
  numeric split of the 15× is uninstrumented — closing it needs `inCollision`
  call counts against the live `DeliverStraw` scene.
- coal's near-contact fixes are on `devel`, unreleased at tag 3.0.4 — §4.1's
  downgrade should be re-checked against whatever tag exists when Path B's
  narrow-phase question is next opened.
- cuRobo's Apache-2.0 relicense was verified at `NVlabs/curobo@main`'s LICENSE
  file; anything pinned to a pre-V2 release (≤ 0.7.x) is still under the
  non-commercial NVIDIA License, and `nvblox_torch` remains non-commercial —
  re-verify before any dependency lands (CLAUDE.md §1.9).

**From the round-3 pass (§§13–23):**

- **No latency was measured in this tree.** Every rate quoted — CAPT's 9.89 ns,
  RDF's 0.54 ms, mink's QP, Tesseract's `contactTest` — is an upstream number
  on someone else's hardware, and for Tesseract there is no published number at
  all. The adoption decision for §15.1 and §17 turns on a measurement that does
  not yet exist: `contactTest` / CAPT query cost against this kernel's chunk
  budget on the DGX Spark and target hosts.
- **CAPT would not make the kernel mm-accurate on its own.** It removes voxel
  quantisation; it leaves sphere/OBB slop, a conservative filter radius, and
  the sensor's own 7 mm–2.2 cm dispersion. Nothing here establishes what the
  *combined* residual would be on this repo's scenes.
- **No learned distance field found is conservative.** Joho et al. state this
  explicitly and pair their network with a geometric checker to restore the
  guarantee. A 0.6 % false-negative rate is not admissible as a veto authority,
  so any adoption of §14.1 is as a *filter in front of* the geometric check, and
  this note does not establish the speedup that survives that pairing on this
  hardware.
- **Tesseract's SDF path and its CCD path are mutually exclusive** — its own
  header says the SDF geometry "is concave, so it is not supported by
  continuous/cast managers". Nothing here establishes which of the two the
  kernel should choose, or whether a split (cast managers for self-collision,
  implicit-SDF for the manipulation volume) is coherent in practice.
- **The Franka force numbers are Panda numbers.** The FR3 datasheet publishes
  none of them, `tau_ext_hat_filtered`'s computation is undisclosed by the
  vendor, and the one independent measurement found is paywalled. The 0.3 N
  lever-arm figure in §21.1 is derived arithmetic, not a manufacturer claim, and
  no in-tree measurement of the signal exists.
- **The torque-spectrum intent claim rests on one paper's abstract.** >95 % on
  a Franka with built-in sensors only is *docs claim*; class count, force
  levels, dataset, code and licence could not be verified, and nothing
  establishes that the same signal separates a *grasp* from a *collision* as
  opposed to separating deliberate human gestures.
- **MuJoCo contact-force magnitude remains unvalidated as a real-Newton
  analogue** — a negative result after five search phrasings, consistent with
  §10 Path C's existing caveat. Any sim-side force threshold is a
  calibration knob, not a measurement.
- **Several load-bearing repos have no licence at all** — Neural-JSDF, UniTac,
  RMMI's `frankie_planner`, Zheng et al., refineCBF, `dynablox_ros2`. Those are
  all-rights-reserved by default. Nothing in this note should be read as
  clearing any of them.
- **Nothing here has been run.** No code was built, no benchmark reproduced, no
  fixture added. This is a reading list with verdicts attached, and every
  "adopt" is a candidate for an ADR, not a decision.
