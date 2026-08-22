# Design decisions

OpenRAL records its architecture and design decisions as Architecture
Decision Records (ADRs). As of 2026-07-08 the ADR log itself lives in the
private [`OpenRAL/management`](https://github.com/OpenRAL/management) repo,
under `adr/` — not in this public repo.

This public repo **does** cite individual decisions by number. Code
comments, docstrings, tests, and docs carry bare `ADR-NNNN` identifiers as
shorthand for the decision that fixed a contract — as of `2edcf67` there
are 398 such citations on 334 lines across 98 files, spanning 14 distinct
numbers (most cited: `ADR-0097`, `ADR-0096`, `ADR-0092`, `ADR-0088`,
`ADR-0095`). Those identifiers are deliberately **not links**: the record
they name lives in the private repo, and a dead link would be worse than a
bare id. Prose around a citation is written to stand on its own, so a
reader without access can follow *what* the behavior is even when they
cannot read *why it was chosen*.

Two consequences worth stating plainly rather than leaving a reader to
discover:

- **The number is a handle, not a coordinate.** The ADR log is append-only
  but not gap-free, and cross-references inside it use filenames rather
  than integers. Do not infer an ordering, a date, or a dependency from a
  number's neighbours.
- **Some cited numbers are not on the private repo's `main` either.** The
  collision-stack citations `ADR-0092`, `ADR-0094` and `ADR-0095` live on
  an unmerged draft branch (`safety/xr1-deploy-sim`) of
  `OpenRAL/management`, and `ADR-0093` on `adr/0093-quantization-dtype-fp8`;
  the implementations they describe are merged here on `master` while the
  decision records that name them are not yet merged there. `ADR-0096` and
  `ADR-0097` are on `main`. So a maintainer resolving one of those four
  numbers has to look on the branch, not just the default branch.

Contributors without access to `OpenRAL/management` who want the full
rationale, alternatives considered, or history behind a specific piece of
behavior can ask by opening an issue in this repo, quoting the identifier.

The decision discipline is unchanged: adding, removing, renaming, or moving
a responsibility between the eight architecture layers (§3 of
[CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md))
requires recording a decision in the private log, written before the code
that implements it.

## Licensing & commercial boundary

Two of those decisions define OpenRAL's public/private boundary and are
worth stating here in full, since the maintainer wants this posture visible
even though the decision record that established it is now private:

- **The public `OpenRAL/openral` repo is uniformly Apache-2.0.** Every
  package in this repo ships under a single permissive license — no
  source-available tier, no BSL, no per-package license drift. Copy-left
  dependencies are rejected without TSC review.
- **Commercial capabilities live in a separate, private monorepo**
  (`OpenRAL/openral-pro`), not in this repo. That includes the TensorRT/NVMM
  zero-copy runtime fast path, WAM (World Action Model) implementations,
  fleet/cloud dispatch, and future premium rSkills. This repo retains the
  protocols and extension seams those capabilities plug into; it does not
  ship the implementations themselves.
- Third-party model **weights** (e.g. NVIDIA GR00T checkpoints) keep their
  own upstream license, independent of OpenRAL's code license — this is
  compliance for models OpenRAL doesn't own, not a statement about OpenRAL's
  own code.

These two points were decided in separate decision records — the original
uniform-Apache-2.0 posture, and a later decision establishing the OpenRAL
Pro commercial tier that superseded that record's earlier "no commercial
tier, ever" commitment while retaining its public-repo-is-Apache-2.0
posture. Full context, alternatives considered, and consequences are in the
private decision log.
