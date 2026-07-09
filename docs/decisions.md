# Design decisions

OpenRAL records its architecture and design decisions as Architecture
Decision Records (ADRs). As of 2026-07-08 the ADR log itself lives in the
private [`OpenRAL/management`](https://github.com/OpenRAL/management) repo,
under `adr/` — not in this public repo.

This public repo no longer cites individual decisions by number. Code
comments, docstrings, and docs describe *why* in prose instead — what a
piece of behavior does and the concept behind it — without a citation a
reader here can't follow. Contributors without access to
`OpenRAL/management` who want the full rationale, alternatives considered,
or history behind a specific piece of behavior can ask by opening an issue
in this repo.

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
