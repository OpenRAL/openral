# OpenRAL Governance

This document describes how the OpenRAL project is run and how decisions are
made. It is intentionally lightweight and reflects the project's current,
early-stage reality; it will grow as the community does.

## Roles

### Project Lead
OpenRAL is currently led by **Adrian Llopart** (@AdrianLlopart). The lead has
final say on direction, breaking changes, and releases, and is responsible for
keeping that authority accountable to the principles in
[CLAUDE.md](CLAUDE.md) and to the maintainers.

### Maintainers
Maintainers review and merge PRs, triage issues, and own areas of the codebase
via [`.github/CODEOWNERS`](.github/CODEOWNERS). They are listed in the
`@openral/maintainers` team. New maintainers are nominated by an existing
maintainer after a sustained track record of high-quality contributions and
confirmed by the project lead.

### Safety Working Group
Because OpenRAL drives physical actuators, the **Safety Working Group**
(`@openral/safety`) has mandatory review authority over everything under
`packages/openral_safety/` and `cpp/openral_safety_kernel/`. A safety-WG
reviewer, a hazard-log update, and tests proving the change is *at least as
conservative* are required for any safety-touching change (see CLAUDE.md §3).
Safety reviewers can block a change on safety grounds regardless of other
approvals. Reach the Safety Working Group at safety@openral.dev for hazard
reports or coordinated safety disclosure.

### Contributors
Anyone who opens an issue or PR. You do not need to be a maintainer to
contribute — see [CONTRIBUTING.md](CONTRIBUTING.md).

## How decisions are made

1. **Day-to-day changes** (bug fixes, features, docs) are decided by normal PR
   review: at least one maintainer approval, CODEOWNERS satisfied, CI green.
2. **Architectural changes that cross a layer boundary** require an
   **Architecture Decision Record** in [`docs/adr/`](docs/adr/) before
   implementation (CLAUDE.md §3, §4.2).
3. **Safety-critical changes** additionally require Safety Working Group
   sign-off as described above.
4. **Disagreements** are resolved by discussion first. If consensus cannot be
   reached, the project lead decides. As the project grows, a Technical
   Steering Committee (TSC) will be formed to take on this role; this section
   will be updated when that happens.

## Licensing & contributions

OpenRAL uses a two-tier open-core licensing model
([ADR-0012](docs/adr/0012-open-core-licensing.md)): the open core is
Apache-2.0, and future commercial layers are source-available under PolyForm
Small Business 1.0.0. Contributions are accepted under the Developer
Certificate of Origin (DCO) — see [CONTRIBUTING.md](CONTRIBUTING.md#developer-certificate-of-origin-dco).

## Code of Conduct

All participation is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md). Reports go to conduct@openral.dev.

## Contact

- General / project enquiries: hello@openral.dev
- Safety Working Group: safety@openral.dev
- Security vulnerabilities: see [SECURITY.md](SECURITY.md)
- Code of Conduct: conduct@openral.dev

## Changing this document

Changes to governance are made by PR and require project-lead approval. Once a
TSC exists, governance changes will require a TSC vote.
