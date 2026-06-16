# ADR-0057: Standardized robot description assets (URDF / xacro / MJCF / SRDF)

- Status: **Proposed**
- Date: 2026-06-16
- Related: [ADR-0027](0027-rskill-state-contract-bindings.md) (the
  `urdf_root_frame` / `static_base_to_urdf_root_xyz_rpy`
  `robot_state_publisher` mount fields this ADR folds into a structured
  `assets.urdf` block); [ADR-0030](0030-geometric-safety-collision-checking.md)
  (the offline collision-lowering tool whose URDF / SRDF / MJCF *inputs* this
  ADR relocates — its lowered `collision_geometry` + `allowed_collision_pairs`
  *outputs* are the only thing the kernel reads, and are untouched here);
  [ADR-0007](0007-robot-sim-split.md) (the robot / sim manifest split that
  put `sim.mjcf_uri` where it lives today);
  [ADR-0023](0023-data-driven-mujoco-hal.md) (the data-driven MuJoCo HAL that
  consumes `sim.mjcf_uri`); CLAUDE.md §1.2 (truth over plausibility — no silent
  `None`), §1.3 (types are the contract), §1.6 (schemas evolve, never
  silently — N/A here: RobotDescription has no `schema_version`), §1.9 (license lineage),
  §3 Layer 0 / 6, §3 Safety (safety-WG review + hazard log).

> Design spec: [`docs/superpowers/specs/2026-06-16-standardize-robot-description-assets-design.md`](../superpowers/specs/2026-06-16-standardize-robot-description-assets-design.md)
> (per-robot tables, empirical timings, phasing). This ADR records the
> decision; the spec carries the working detail.

## Context

A `RobotDescription` points at four kinds of description asset — URDF (the
kinematic/visual model `robot_state_publisher` broadcasts as `/tf` and
`/robot_description`), the SRDF (its allowed-collision matrix), the MJCF (the
MuJoCo sim model), and — via ADR-0027 — a `robot_state_publisher` mount frame.
Today each is declared and resolved through a **separate, divergent
mechanism**:

- **`urdf_path`** — resolved by `openral_core.urdf_resolve.resolve_urdf_path`.
  Accepts `python:<module>:<attr>`, an absolute path, a relative path, and the
  `ros2://robot_description` dynamic-detection marker.
- **A second, divergent URDF loader** —
  `openral_safety.urdf_lowering._load_urdf_model`. Accepts `python:<…>`,
  `robot_descriptions:<name>`, and a file path — **and additionally expands
  xacro** in-process via `xacrodoc`. It exists because the collision-lowering
  tool (ADR-0030) needs to read xacro-only robots that `resolve_urdf_path`
  cannot.
- **`sim.mjcf_uri`** — resolved by
  `openral_hal._mujoco_arm.resolve_mjcf_uri`. Accepts
  `robot_descriptions:<m>`, `gym_aloha:<scene>`, `openarm_v2:bimanual`,
  `file:<…>`, and an absolute path.
- **`srdf_path`** — no URI scheme at all; a plain filesystem path parsed in
  place.

**Why this is a defect, not just untidiness.** The two URDF resolvers
disagree on grammar, which produces a *silent* failure:
`urdf_path: robot_descriptions:ur5e_description` is understood by the
lowering loader (so `openral collision lower` works) but **not** by
`resolve_urdf_path` (so the launch's `robot_state_publisher` gets `None` —
no `/tf`, no robot model, no error). A safety-adjacent asset that resolves
for one consumer and silently vanishes for another violates CLAUDE.md §1.2
(truth over plausibility) and §1.4 (explicit beats implicit). Separately,
`ur5e` / `ur10e` / `rizon4` ship **xacro only** upstream
(`robot_descriptions` exposes `XACRO_PATH`, no `URDF_PATH`); getting a usable
URDF from them requires the optional `lowering` dependency group (`xacrodoc` /
`yourdfpy`) at runtime — an install burden pushed onto every end user who just
wants `/tf`.

Empirically (see spec §2): xacro expansion is fast (0.17–0.29 s for ur5e), so
**time is not the constraint** — the *dependency* is. And `robot_descriptions`
auto-downloads (git-clones) upstream assets to a cache on first use, which is
fine as long as we make that download visible rather than surprising.

## Decision

### 1. One `assets:` block, one resolver, one grammar

Introduce a single structured `assets:` block on `RobotDescription` and a
single new resolver `openral_core.assets.resolve_asset(ref, kind)` that all
consumers call. The schema (full definition in the spec §4.2):

```python
class UrdfAsset(BaseModel):
    ref: str                                      # e.g. "rd:panda_description"
    root_frame: str | None = None                # was urdf_root_frame (ADR-0027)
    base_to_root_xyz_rpy: tuple[float, ...] | None = None  # was static_base_to_urdf_root_xyz_rpy

class AssetRefs(BaseModel):
    urdf: UrdfAsset | None = None                # OPTIONAL — sim-only robots omit it
    mjcf: str | None = None
    srdf: str | None = None

class RobotDescription(BaseModel):
    assets: AssetRefs = AssetRefs()
    # REMOVED: urdf_path, sim.mjcf_uri, srdf_path,
    #          urdf_root_frame, static_base_to_urdf_root_xyz_rpy
```

The ADR-0027 mount metadata (`root_frame`, `base_to_root_xyz_rpy`) folds *into*
`assets.urdf`, so the mount and the file it mounts travel together. A Pydantic
validator rejects any `ref` that does not match the grammar below — closing the
silent-`None` failure class **at parse time**.

### 2. The URI grammar — `resolve_asset(ref, kind)`

`resolve_asset(ref: str, kind: Literal["urdf", "mjcf", "srdf"]) -> Path` lives
in the new module `openral_core.assets`. One grammar serves all three kinds:

| Scheme | Meaning |
|---|---|
| `rd:<module>` | `robot_descriptions.<module>`. Picks `URDF_PATH` for `kind="urdf"`, `MJCF_PATH` for `kind="mjcf"`. **Cache-miss → emit a visible "Downloading `<module>` …" line, then fetch** (no silent network I/O). If `kind="urdf"` but the module exposes only `XACRO_PATH`, raise `ROSConfigError` directing the author to `openral robot vendor-urdf` — we **never** expand xacro at runtime. |
| `file:<relpath>` | Vendored file resolved against `robots/<id>/` (the manifest dir), then the repo root. Used for vendored URDF / SRDF / MJCF. |
| `gym_aloha:<scene>` · `openarm:<variant>` · `menagerie:<model>` | Robot-specific MJCF loaders (sim-only optional deps, lazy-imported). `kind="mjcf"` only. |
| `ros2://robot_description` | Dynamic-detection marker (URDF only) — not resolved to a file; passed through so the launch subscribes to a live `/robot_description`. |

**Dropped (this is the no-backcompat part):** bare
`robot_descriptions:<name>`, `python:<module>:<attr>`, plain-path SRDF,
`openarm_v2:bimanual` (→ `openarm:bimanual`), and the divergent
`_load_urdf_model`. The old `robot_descriptions:<name>` form now **fails the
validator loudly** instead of resolving for one consumer and not the other.

### 3. No backwards compatibility

The old fields (`urdf_path`, `sim.mjcf_uri`, `srdf_path`, `urdf_root_frame`,
`static_base_to_urdf_root_xyz_rpy`) and the divergent `_load_urdf_model` are
**removed outright** — not deprecated, not aliased. This is an explicit user
decision: the cost of carrying two grammars and two resolvers (the exact source
of the silent-`None` bug) is higher than the cost of a one-time migration of 16
manifests. **No migrator ships:** `RobotDescription` manifests carry no
`schema_version` field, and per the user's no-backwards-compatibility directive
old-format manifests are not supported. All 16 in-repo manifests are hand-migrated
in this PR; there are no other manifests to migrate. (CLAUDE.md §1.6's bump +
migrator requirement applies to on-disk formats that carry `schema_version` —
rSkill / scene / trace — none of which this change touches.)

### 4. xacro-only robots ship vendored, pre-expanded URDFs

`ur5e` / `ur10e` / `rizon4` (xacro-only upstream) and `openarm` (upstream xacro
is prefix-mismatched against the HAL joint names) get a **vendored
pre-expanded URDF** committed under `robots/<id>/<id>.urdf`. A new CLI,
`openral robot vendor-urdf <id>` (which uses the optional `lowering` group),
resolves the upstream xacro, expands it via `xacrodoc`, applies any joint-name
normalization (openarm: strip the `openarm_` prefix to match the HAL's
`left_joint1..7`), and writes the URDF with a provenance + license header. This
runs **once at authoring time**; the committed output is reproducible (not
hand-edited) and **runtime needs zero build tooling**. End users no longer
install `xacrodoc` / `yourdfpy` to get `/tf`.

**Joint-name-only vendoring for already-flat URDFs (`h1`).** Some robots ship a
flat upstream URDF whose *joint* names diverge from the manifest /
HAL-control-contract names, which makes `robot_state_publisher`'s `/tf` use
joint names that don't match the HAL's `/joint_states`. `h1` suffixes every
joint with `_joint` (`torso_joint`). For these, `vendor-urdf --raw-text` copies
the upstream URDF **text verbatim** and applies the joint renames with `re.sub`
directly on the raw XML — **no yourdfpy round-trip** (which would absolutize /
mangle `package://` and relative mesh paths, and rewrite CRLF). The renames
target **joint names only** (`<joint name="X"` and any `joint="X"` reference);
link names, geometry, inertials, mesh paths and line endings are preserved
byte-for-byte (verified: the only diff vs upstream is the renamed joint-name
attributes plus the inserted provenance comment). `h1` strips `_joint`.

**`so100_follower` / `so101_follower` / `gr1` are NOT yet vendored** — each
blocked for a distinct reason, all surfaced (not papered over):

- **`so100_follower` / `so101_follower` — relative-path collision meshes.**
  Their upstream URDF names joints `1`..`6` (SO-ARM motor convention → manifest
  semantic names `shoulder_pan`..`gripper`), which `vendor-urdf --raw-text`
  renames correctly. **But** their `<collision>` meshes are referenced by
  *relative* path (`assets/*.stl`), resolved by yourdfpy against the URDF file's
  own directory. Relocating the vendored URDF under `robots/<id>/` makes those
  meshes unreachable, so the safety collision-lowering router (§5) silently
  flips them from URDF-sampling to MJCF — a safety-source change the
  lowering-regression test correctly rejects (`so100_follower`'s MJCF-path ACM
  comes out empty; `so101_follower`'s happens to coincide but the silent flip is
  still rejected by the router's "no source flip" intent). Vendoring these two
  therefore *also* requires vendoring (or `package://`-rewriting) their mesh
  assets (3 MB / 16 MB) — a maintainer / safety-WG decision, deferred.
- **`gr1` — copy-left upstream.** Its URDF (Wiki-GRx-Models) is licensed
  **GPL-3.0**. CLAUDE.md §1.9 rejects copy-left from open-core without TSC
  review, so the GPL-3.0 URDF cannot be committed into this Apache-2.0 repo. A
  joint-name patch is mechanically trivial (strip `_joint`, collapse
  `*_elbow_pitch` → `*_elbow`) but the license, not the mechanics, is the
  blocker.

All three keep their `rd:` ref and their documented `/tf` joint-name xfail
(`test_asset_resolution._URDF_JOINT_NAME_MISMATCH`).

### 5. URDF stays optional

7 of the 16 robots are MuJoCo-sim-only (`aloha_bimanual`, `sawyer`, `widowx`,
`google_robot`, `pusht_2d`, and others per spec §4.3) and declare **no**
`assets.urdf`. The resolver and schema treat a missing URDF as a first-class
state, asserted explicitly in tests (a negative assertion — no silent
placeholder), not as an error.

## Safety (CRITICAL — read before reviewing)

**The C++ safety kernel never reads URDF, SRDF, or MJCF at runtime.** It reads
only the *lowered* outputs `collision_geometry` + `allowed_collision_pairs`
from the manifest, via
`openral_safety.envelope_loader.collision_params_from_description` (ADR-0030).
URDF / SRDF / MJCF are **inputs to the offline collision-lowering tool**
(ADR-0030's `urdf_lowering.py` / `mjcf_lowering.py`, driven by `openral
collision lower`), which *produces* those lowered fields at authoring time and
commits them to the manifest for human review.

**This change alters only how the source files are located — never the
geometry.** Same upstream URDF/SRDF/MJCF bytes, same lowering algorithm, same
ACM sampling seed (ADR-0030 `_RNG_SEED = 20260610`), therefore byte-identical
lowered output. The conservatism invariant (CLAUDE.md §3) holds **by
construction**: identical geometry and identical ACM cannot be less
conservative than what they replace.

Required guards on the implementing PRs (per CLAUDE.md §3 Safety — these are
mandates, not suggestions):

1. **Byte-identical lowering regression test, fleet-wide.** For every robot
   that carries `collision_geometry` in its manifest, re-run lowering through
   the new resolver and assert the output is **identical** to the committed
   values — byte-for-byte for the ACM pairs, geometric equality for the
   capsules. **A diff is a release blocker.** This is the concrete mitigation
   the hazard-log entry below references. **Implemented:**
   `packages/openral_safety/test/test_lowering_regression.py` — re-lowers every
   committed robot through the provenance-correct dispatcher
   `openral_safety.urdf_lowering.lower_robot_auto` (`select_lowering` picks SRDF /
   URDF-sampling / MJCF per the §5 rule), asserting ACM-set equality, exact ACM
   order, per-link geometric equality at committed (4-dp) precision, and
   byte-identical rendered blocks. 9/10 robots re-lower with **zero drift**
   (franka_panda, g1, h1, openarm, rizon4, so100_follower, so101_follower, ur5e,
   ur10e). **One documented exception — `panda_mobile`:** its committed
   `collision_geometry` is **hand-authored** (no `# GENERATED` header; clean
   round capsules) and it has no MJCF to keep that geometry, so *any* current
   lowering path re-fits geometry from the URDF and the ACM loses the
   hand-tuned `panda_link5 ↔ panda_link7` capsule-junction pair. This drift is
   **pre-existing** (the old `lower_robot`-for-everything CLI drifted it
   identically) and is marked **strict-xfail** so it stays loud. **Resolved
   (2026-06-16): keep the hand-authored geometry.** A candidate re-lower was
   performed and rejected as *not* at-least-as-conservative — it shrinks link7's
   capsule (0.060 → 0.053 m) and drops the hand-tuned `link5 ↔ link7` ACM
   exception (re-introducing a spurious E-stop in the stowed config). The hand
   geometry is authoritative; the strict-xfail records it as an accepted, not a
   pending, exception.
2. **All existing safety tests pass unchanged** —
   `packages/openral_safety/test/test_urdf_lowering_fk.py` (including
   `test_franka_acm_uses_srdf_when_srdf_path_set`), the `mjcf_lowering` tests,
   the envelope-loader tests, the kernel integration tests, and the fleet
   guard `tests/unit/test_collision_lowering_fleet.py` (ADR-0030).
3. **Safety-WG sign-off** — a **human gate** that the author cannot
   self-clear. Tracked as a pending checkbox:
   - [ ] **PENDING: safety-WG reviewer sign-off** on the lowering-regression
     evidence and the "locates files, never changes geometry" claim.
   - [x] **RESOLVED (2026-06-16): keep `panda_mobile`'s hand-authored geometry.**
     The candidate re-lower is less conservative (link7 capsule shrinks
     0.060 → 0.053 m) and drops the hand-tuned `link5 ↔ link7` ACM exception
     (spurious-E-stop risk in the stowed config), so it was rejected. No kernel
     geometry changes; the strict-xfail now records an accepted exception.
4. **Hazard-log update** — entry added below referencing the regression test as
   the mitigation (CLAUDE.md §3).
5. **TDD for the safety-touching edits** (`urdf_lowering.py`,
   `collision_params_from_description`), per CLAUDE.md §1.4.

The one behavioural detail to watch: `urdf_lowering.py` currently relies on
`_load_urdf_model`'s in-process xacro expansion for `ur5e`/`ur10e`/`rizon4`.
After this change those robots are lowered from their **vendored**
`robots/<id>/<id>.urdf`. The regression test (guard 1) is exactly what proves
the vendored URDF lowers to the same geometry the xacro path produced.

## License lineage (CLAUDE.md §1.9)

Vendoring pre-expanded URDFs means **committing third-party files** into the
repo. Per §1.9 each carries its upstream license, recorded version-specifically
(not family-wide). The `openral robot vendor-urdf` CLI writes a provenance +
license header into each generated URDF; the table below is the authoritative
record and **each entry must be confirmed against the upstream repo's
`LICENSE` at vendoring time** (truth over plausibility — §1.2; if an upstream
license is ambiguous, the vendoring is blocked, not guessed):

| Robot | Upstream source | License (confirm at vendor time) | Notes |
|---|---|---|---|
| `franka_panda` / `panda_mobile` (panda URDF) | `example-robot-data` (Gepetto/INRIA), originating from Franka Emika's `franka_ros` | BSD-2-Clause (example-robot-data) / Apache-2.0 (franka description) | URDF is currently `rd:panda_description`; vendoring only if a gap appears. |
| `ur5e` / `ur10e` | ROS-Industrial `universal_robots` / `ur_description` | BSD-3-Clause | Matches the `ur_robot_driver` BSD-3 posture already recorded in `robots/ur5e/README.md`. |
| `rizon4` | `flexivrobotics/flexiv_description` | Apache-2.0 | Upstream URDF (see `robots/rizon4/README.md`). |
| `openarm` | `enactic/openarm` (description) | Apache-2.0 | Upstream xacro is prefix-mismatched; vendored URDF is prefix-patched to the HAL joints. |
| `h1` | `unitreerobotics/unitree_ros` (`h1_description`) | BSD-3-Clause (confirmed: `unitree_ros/LICENSE`, © Unitree Robotics) | Already-flat upstream URDF; vendored via `vendor-urdf --raw-text` with `_joint` suffix stripped from joint names only (mesh `package://` paths verbatim). |

All vendored URDFs are permissive (Apache-2.0 / BSD), so §1.9's "Apache-2.0 /
MIT / BSD, no GPL without TSC review" constraint is satisfied. No copy-left, no
closed-SDK bundling. The third-party meshes these URDFs reference still resolve
via `package://` from the `robot_descriptions` cache (the prompted download) —
**full offline mesh vendoring is explicitly out of scope** (spec §8).

**Not vendored (blocked at vendor time — recorded per §1.2, truth over
plausibility):**

| Robot | Upstream source | License | Why not vendored |
|---|---|---|---|
| `gr1` | `Wiki-GRx-Models` (Fourier) | **GPL-3.0** (confirmed: `Wiki-GRx-Models/LICENSE`) | Copy-left — §1.9 rejects from open-core without TSC review. Keeps `rd:` ref + xfail. |
| `so100_follower` / `so101_follower` | `TheRobotStudio/SO-ARM100` | Apache-2.0 | Vendored joint-renamed URDF + the upstream Apache-2.0 mesh assets (relative-path) under `robots/<id>/assets/` (with the upstream `LICENSE`); lowering re-fits byte-identically from the vendored meshes. ~3 MB (so100) / ~16 MB (so101). |

## Consequences

**Benefits**

- **One grammar, one resolver** — `resolve_asset` is the single source of
  truth; the second URDF loader and the MJCF resolver are deleted.
- **Loud validation, no silent `None`** — a malformed or unsupported `ref`
  fails the Pydantic validator at parse time, not silently downstream
  (closes the `robot_descriptions:ur5e_description` → no-`/tf` defect).
- **End users need no xacro tooling** — vendored URDFs mean `/tf` works from a
  base install; `xacrodoc` / `yourdfpy` stay in the optional `lowering` group,
  used only by the authoring-time `vendor-urdf` CLI.
- **Asset + mount travel together** — ADR-0027's mount fields live inside
  `assets.urdf`, so they can't drift apart from the URDF they describe.

**Costs**

- **Breaking schema change** — old fields removed; all 16 manifests hand-migrated.
  No migrator (RobotDescription carries no `schema_version`; no old-format support).
- **Migrate all 16 manifests** to the `assets:` block (spec §4.3 has the
  canonical per-robot table).
- **Vendor 4 URDFs** (`ur5e`, `ur10e`, `rizon4`, `openarm`) — committing
  third-party files with license headers (lineage above).
- **Update every consumer** to call `resolve_asset` (spec §4.5):
  `sim_e2e.launch.py`, `isaac_sim.py`, the detect `assemble.py`
  `ros2://robot_description` marker, `openral_safety/urdf_lowering.py`
  (delete `_load_urdf_model`), the `openral collision lower|check` CLI, and
  `_mujoco_arm.py` (delete `resolve_mjcf_uri`).
- `docs/METHODS.md` (new `assets.py` symbols; removed resolvers) and the
  repo-state map updated in the implementing PR (CLAUDE.md §1.13–1.14, §4.3).

## Alternatives considered

- **Keep four mechanisms, just document them.** Rejected: documentation does
  not close the silent-`None` failure class — only a single resolver + a
  reject-at-parse validator does (§1.2, §1.4).
- **Unify the grammar but keep backward-compatible aliases** (accept the old
  forms, warn, resolve them). Rejected by explicit user decision: the alias
  layer *is* the second grammar, i.e. the bug surface we are removing. A clean
  break + migrator is cheaper to reason about than a dual-grammar resolver.
- **Expand xacro on the fly at runtime** (drop vendoring, resolve xacro-only
  robots live). Rejected: forces the optional `lowering` dependency group onto
  every end user just to get `/tf` (spec §2). Time isn't the constraint; the
  dependency is.
- **Vendor every robot's URDF + meshes for full offline operation.** Out of
  scope (spec §8): a much larger effort (mesh licensing, repo size). This ADR
  vendors only the xacro-only *gap*; meshes keep resolving from the
  `robot_descriptions` cache.

## Phasing

One logical change (likely >800 lines → maintainer pre-approval per §4.2.5).
Detailed in spec §7:

1. **This ADR** — grammar, vendor-vs-reference, no-backcompat, safety note,
   license lineage.
2. **Resolver** `openral_core/assets.py` + grammar validator + unit tests
   (TDD).
3. **Schema** `AssetRefs` / `UrdfAsset`; remove old fields (no `schema_version`
   bump/migrator — RobotDescription carries none); `hypothesis` round-trip.
4. **Vendoring CLI** `openral robot vendor-urdf`; run it → commit
   `robots/{ur5e,ur10e,rizon4,openarm}/<id>.urdf` with license headers.
5. **Migrate all 16 manifests** to `assets:`.
6. **Update every consumer** (spec §4.5); delete `_load_urdf_model` and
   `resolve_mjcf_uri`.
7. **Comprehensive + safety-regression tests** (Safety §, spec §6); `just lint
   && just test` and the safety suite green; **safety-WG sign-off** before
   merge.
