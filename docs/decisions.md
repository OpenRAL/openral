# Design decisions

OpenRAL records architecture and design decisions as Architecture Decision
Records (ADRs). The ADR log lives in the private
[`OpenRAL/management`](https://github.com/OpenRAL/management) repo, under
`adr/`, not in this public repo.

Code comments, docstrings, tests, and docs here cite decisions by bare
`ADR-NNNN` identifiers. They are deliberately **not links**: the record is
private, and a dead link is worse than a bare id. Prose around a citation
stands on its own, so a reader without access can follow *what* the behavior
is even without *why it was chosen*.

- **The number is a handle, not a coordinate.** The ADR log is append-only
  but not gap-free, and cross-references inside it use filenames rather
  than integers. Do not infer an ordering, a date, or a dependency from a
  number's neighbours.
- **Some cited numbers are not on the private repo's `main` either.**
  `ADR-0092`, `ADR-0094` and `ADR-0095` live on an unmerged draft branch
  (`safety/xr1-deploy-sim`) of `OpenRAL/management`, and `ADR-0093` on
  `adr/0093-quantization-dtype-fp8`. Their implementations are merged here
  on `master`; the records are not yet merged there. To resolve one of
  those four numbers, check that branch, not just `main`.

Without access to `OpenRAL/management`, ask for the rationale, alternatives,
or history behind a decision by opening an issue here that quotes the
identifier.

Adding, removing, renaming, or moving a responsibility between the seven
architecture layers (§3 of
[CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md))
requires a decision in the private log, written before the code that
implements it.

## Licensing & commercial boundary

Two decisions define OpenRAL's public/private boundary. They are stated here
in full so the posture stays visible although the records are private:

- **The public `OpenRAL/openral` repo is uniformly Apache-2.0.** Every
  package ships under that one license: no source-available tier, no BSL,
  no per-package license drift. Copy-left dependencies are rejected without
  TSC review.
- **Commercial capabilities live in a separate, private monorepo**
  (`OpenRAL/openral-pro`), not in this repo. That includes the TensorRT/NVMM
  zero-copy runtime fast path, WAM (World Action Model) implementations,
  fleet/cloud dispatch, and future premium rSkills. This repo keeps the
  protocols and extension seams they plug into, not the implementations.
- **The safety kernel stays open, and every check that can only refuse
  motion stays open with it.** A kernel rule that *relaxes* a verdict
  (lets a hand touch a cell, ends a payload binding) may move to OpenRAL Pro.
  It can only do so behind the kernel's open, fail-closed, sim-only
  exemption-provider interface
  (`cpp/openral_safety_kernel/include/openral_safety_kernel/exemption_provider.hpp`).
  The kernel enforces the provider's bounds itself (ADR-0121). The box-based rules
  a scene declares stay open (the ADR-0115 `search_box` grasp allowance, #349
  `workspace_box` contact); the tracker-based rules (#418 K2, K5, K7,
  payload-own cells) are the ones that move (ADR-0121 amendment, 2026-10-25).
- Third-party model **weights** (e.g. NVIDIA GR00T checkpoints) keep their
  own upstream license, independent of OpenRAL's code license. This is
  compliance for models OpenRAL doesn't own, not a statement about
  OpenRAL's code.

The first two points come from separate records: the original
uniform-Apache-2.0 posture, and a later decision establishing the OpenRAL
Pro commercial tier. The later one superseded the original's "no commercial
tier, ever" commitment and kept the public repo Apache-2.0. The safety-kernel
point is `ADR-0121`. It narrows that later record's "the safety kernel stays
open" line to the refuses/relaxes rule above. Context, alternatives, and
consequences are in the private decision log.
