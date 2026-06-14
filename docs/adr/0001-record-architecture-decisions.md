# ADR-0001: Record architecture decisions

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)

## Context

We need a lightweight, durable record of architectural choices that future contributors
(human and AI agents) can rely on when making changes to OpenRAL.

## Decision

Use Markdown Architecture Decision Records (ADRs) under `docs/adr/`. One file per decision,
monotonically numbered. Immutable once accepted; supersede with a new ADR — never edit an
accepted one.

## Amendments

### 2026-05-08 — Soften strict immutability

CLAUDE.md §7.9 has been relaxed. The Decision above stands with one
clarification: ADRs **may be amended in place** for factual corrections,
status updates, and reconciliation against the live repo state, provided
the amendment is additive and dated (e.g., as a subsection under
"Amendments" or a clearly-marked update note). The original Decision
text must be preserved.

**Reversing or contradicting** a prior decision still requires a new ADR
that marks the prior one Superseded; do not silently rewrite a Decision
under the same ADR number.

This brings ADRs in line with how `tests/README.md` and
`docs/architecture/repo-state-map.html` are maintained — hand-edited,
diff-able, but tracked over time rather than frozen.

## Consequences

- Pros: low friction, diff-able, lives next to the code, works with any Git host.
- Cons: requires discipline; new contributors must skim `docs/adr/` before proposing
  large changes.

## Template

```markdown
# ADR-XXXX: Title

- Status: Draft | Accepted | Superseded by ADR-YYYY
- Date: YYYY-MM-DD

## Context
...

## Decision
...

## Consequences
...
```
