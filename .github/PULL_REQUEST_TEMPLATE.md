## What
<!-- One sentence summary. -->

## Why
<!-- Link issue or describe motivation. -->

## How tested
<!-- Unit / integration / sim / hardware. Paste `pytest` summary. -->

## Checklist
<!-- Mirrors CLAUDE.md §4.4 — keep the two in sync when either changes. -->
- [ ] Conventional commit title; description has "What changed", "Why", "How tested"
- [ ] Schemas: real fixture under `robots/` / `rskills/` / `scenes/` validates; on-disk `schema_version` bumped + migrator shipped for any backward-incompatible change (post-publish — no longer frozen at `"0.1"`)
- [ ] Layer boundary crossed → decision recorded in the private management decision log (`OpenRAL/management`, `adr/`)
- [ ] Tests: unit + integration + sim where applicable; HIL if a HAL changed. No new mocks/stubs/smoke tests (CLAUDE.md §1.11)
- [ ] The matching `docs/methods/` file updated for every added/renamed/removed/moved public symbol; `tools/refresh_methods_linenos.py --check` clean
- [ ] Docs updated in the same PR (READMEs, `docs/`, ADRs) — no follow-up deferrals
- [ ] Pre-existing errors fixed in a separate prior `fix(...)` commit
- [ ] Repo state map (`docs/architecture/repo-state-map.html`) updated if a module was added, renamed, removed, or its `green` / `yellow` / `blue` / `red` status flipped (CLAUDE.md §4.3)
- [ ] `just lint` passes; `mypy --strict` clean; no new `# type: ignore` without `# reason: ...`; no new `try/except: pass`; no new global mutable state
- [ ] No new safety-disabling flag; no new `try/except: pass` on the actuation path
- [ ] Performance budgets met if relevant; no PII in fixtures/logs; new deps Apache-2.0 / MIT / BSD (no GPL without TSC review)
