# ADR-0004: Single monorepo over poly-repo for the OpenRAL open-core

- Status: Accepted
- Date: 2026-05-24 (retroactive — documents a Week-1 decision already in code)
- Amended: 2026-05-24 (see Amendments below)

## Context

OpenRAL's open-core today spans **ten Python workspace members**
(`openral-core`, `openral-cli`, `openral-hal`, `openral-sensors`,
`openral-world-state`, `openral-rskill`, `openral-runner`, `openral-sim`,
`openral-detect`, `openral-observability`) plus **five ROS 2 packages**
(`openral_msgs`, `openral_world_state_ros`, `openral_hal_so100/franka/ur5e/ur10e`),
plus a `tools/` tree, an `examples/` tree, an `rskills/` catalogue, a
`robots/` catalogue, a `benchmarks/` catalogue, and the docs site. The
contracts in `openral_core` are normative and consumed by every other
package; the ROS IDL in `openral_msgs` is normative and consumed by every
ROS node.

A change that touches a schema typically touches **at least three trees in
the same commit**: the schema (`openral_core`), a real test fixture
(`robots/` or `rskills/`), and the consumer (e.g., `openral-rskill` or
`openral-runner`). While we are pre-publish the on-disk schemas sit at
`schema_version: "0.1"` and the surface evolves in place without
migrators (CLAUDE.md §1.6). CLAUDE.md §1.14 elevates the cross-tree
workflow to a rule — "Docs travel with the code" — and §1.13 makes
`docs/METHODS.md` update mandatory in the same PR. Cross-package atomic
commits are the default workflow, not the exception.

External, deliberately-separated repos exist and are documented in
CLAUDE.md §2:

- `huggingface.co/openral/skill-*` — skill weights & manifests (one HF
  Hub repo per published rSkill; license-gated).
- `huggingface.co/openral/dataset-*` — LeRobotDatasets.
- `openral/cloud` — BSL-1.1 hosted observability / fleet control plane.
- `openral/contrib-closed-shims` — NDA-restricted vendor adapters.
- `openral/awesome-ros` — community curation.

These are split precisely because they have **different licensing,
different release cadence, or different audiences** from the open core
— not because the open core wanted finer granularity.

## Decision

**The open core lives in one monorepo** (`openral/openral`) with three
build systems coexisting at the root:

1. **uv workspace** for Python — `pyproject.toml:13-14` declares
   `members = ["python/*"]`. New Python packages land by creating a
   directory and listing it in `[tool.uv.sources]`. One lockfile
   (`uv.lock`) at the root.
2. **colcon** for ROS 2 — `packages/<name>/` directories built with
   `ament_cmake` / `ament_python`. One `install/` after a colcon build.
3. **`just`** as the canonical task runner — every workflow that touches
   more than one package has a `just` recipe so contributors do not have
   to know which build system owns a given file.

Concrete rules:

1. **Atomic schema evolution.** A schema change touches `openral_core`,
   the real fixture under `robots/` / `rskills/` / `scenes/`, and
   the consumer in the same commit. The on-disk `schema_version` stays
   at `"0.1"` while we are pre-publish (CLAUDE.md §1.6).
2. **Single CI surface.** `.github/workflows/` exercises every
   workspace member from one place — there is no cross-repo CI to
   coordinate.
3. **Single ADR catalogue.** `docs/adr/` is the canonical record for
   the whole open core. ADRs that affect a deliberately-separated repo
   (e.g., `openral/cloud`) cross-reference but live here.
4. **Single CHANGELOG generator** — `release-please` consumes
   Conventional Commits across the workspace and produces one release.
5. **Closed / non-Apache pieces stay in their own repos.** The
   monorepo is Apache-2.0 for the open-core tier (CLAUDE.md §1.9 +
   ADR-0012). The PolyForm SBA tier (reasoner / wam / dispatcher /
   skill_catalog / fleet) plans to land **here** under a different
   per-directory `LICENSE`; the BSL cloud tier and the closed-shim
   repo do **not**.

## Consequences

- **Pros**
  - Cross-cutting changes (a schema bump, a layer rename, a Justfile
    recipe rewrite) land atomically. No "PR 4-of-7 stuck across
    repos" failure mode.
  - One CI dashboard, one lockfile, one `just test` to verify the
    whole open core.
  - New contributors clone one URL and have the entire normative
    surface in their editor's project root — including ADRs,
    schemas, METHODS.md, and the repo state map.
  - `docs/METHODS.md` (CLAUDE.md §1.13) is feasible: a flat,
    layer-ordered index over `python/`, `packages/`, and `tools/`.
    A poly-repo split would either give up the index or maintain it
    cross-repo.

- **Cons**
  - `git log` on `main` is high-volume; readers filter by path
    (`git log -- python/sim/`) more than by branch.
  - Workspace builds can be slow if every package is touched; uv's
    incremental resolver + `just test` filters mitigate this.
  - A contributor only interested in `openral-core` still clones the
    whole tree. Acceptable price for the atomicity guarantees above.
  - When the PolyForm SBA tier lands, the per-directory licensing
    requires reviewers to know which subtree they're editing. ADR-0012
    handles the license posture; a future tooling PR will surface it
    in CI (`ral check-license` exists today for installed weights, not
    for source-tree boundaries).

## Alternatives considered

- **One repo per workspace member (full poly-repo).** Rejected — every
  schema bump becomes a coordinated multi-PR landing across
  10+ repos; the atomicity guarantee disappears.
- **Two repos: `openral-python` and `openral-ros`.** Rejected — the
  Python ↔ IDL bridge (`packages/msgs/`) and the lifecycle nodes that
  wrap Python services (`packages/world_state/`) would still need
  cross-repo PRs. We'd inherit poly-repo pain without poly-repo
  ownership clarity.
- **Monorepo with sparse-checkout discipline.** Possible at scale but
  premature — the repo is ~50k LoC today; the ergonomics of a single
  checkout still win.
- **Subtree splits per workspace member (autoupdated mirror repos).**
  Considered for the eventual PyPI publishing story. Deferred — when
  PyPI trusted publishing is wired (roadmap "Org / publishing" item),
  per-package wheels publish from the monorepo directly; mirrored
  read-only repos are an option for visibility but not necessary for
  distribution.

## Why this ADR is retroactive

The monorepo decision is encoded in `pyproject.toml:13-14`
(`members = ["python/*"]`), in the colcon layout under `packages/`,
in the Justfile's recipe set, and in CLAUDE.md §2 (the repo map). This
ADR records the reasoning so future "should we split this out" proposals
have a paper trail to push against (CLAUDE.md §7.9).

## References

- CLAUDE.md §2 (repo map), §1.13 (METHODS.md), §1.14 (docs travel),
  §1.9 / ADR-0012 (open-core licensing).
- Root `pyproject.toml` — uv workspace + dependency-group conflicts.
- `packages/` — ROS 2 colcon workspace.
- `Justfile` — the canonical task runner.
- ADR-0012 — per-tier licensing inside the same monorepo.
