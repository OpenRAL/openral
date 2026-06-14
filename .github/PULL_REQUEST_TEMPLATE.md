## What
<!-- One sentence summary. -->

## Why
<!-- Link issue or describe motivation. -->

## How tested
<!-- Unit / integration / sim / hardware. Paste `pytest` summary. -->

## Checklist
- [ ] Conventional commit title
- [ ] Schemas: on-disk `schema_version` stays at `"0.1"` (pre-publish); change validated against a real fixture under `robots/` / `rskills/` / `scenes/`
- [ ] Layer boundary: ADR added (if crossed)
- [ ] Tests added/updated
- [ ] Docs updated (if public surface changed)
- [ ] Repo state map (`docs/architecture/repo-state-map.html`) updated if a module was added, renamed, removed, or its `green` / `yellow` / `blue` / `red` status flipped (CLAUDE.md §4.3)
- [ ] No new `try/except: pass` on actuation path
- [ ] No new safety-disabling flag
- [ ] Licenses of new deps reviewed
