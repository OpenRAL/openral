# Contributing to OpenRAL

Thank you for considering contributing to OpenRAL.

## Before you start

- Read [CLAUDE.md](CLAUDE.md) — this is the engineering playbook and single source of truth.
- Read the [Architecture overview](docs/architecture/overview.md).
- Skim the [roadmap](docs/roadmap/index.md) so your work lines up with what's already done, in flight, or planned.
- Open [`docs/architecture/repo-state-map.html`](docs/architecture/repo-state-map.html) in a browser to see at a glance which modules already exist, which are in flight, and which are still spec-only. PRs that change this set must update the map (CLAUDE.md §4.3).
- Check open issues for context, especially those labelled `safety`, `vla`, or `hardware`.

## Development setup

```bash
just bootstrap   # installs uv, ROS 2, system deps
uv sync          # install Python workspace
just test        # run all tests
```

## Workflow

1. Fork and create a feature branch.
2. Write tests first for anything on the actuation path.
3. Run `just lint && just test` before pushing.
4. Open a PR using the [PR template](.github/PULL_REQUEST_TEMPLATE.md).
5. PRs over 800 lines need maintainer pre-approval.

## Commit style

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(skill): add SmolVLA adapter
fix(safety): clamp ee_speed before publish
docs(adr): ADR-0002 Pydantic over dataclasses
chore: bump uv.lock
```

## Code standards

- Python 3.12 only (matches `pyproject.toml`'s `requires-python = ">=3.12,<3.13"`). `mypy --strict` must pass.
- Pydantic v2 for all schemas and interfaces.
- Ruff for linting and formatting (line length 100).
- Google-style docstrings on all public symbols.
- No `time.sleep` in async code. No blocking I/O on the event loop.
- All errors are typed `ROSError` subclasses (see CLAUDE.md §5).
- `ROSSafetyViolation` is **never** silently caught.

## Safety

Touching `packages/openral_safety/` or `cpp/openral_safety_kernel/` requires:
- Reviewer from the safety working group.
- Hazard-log update.
- Tests proving new behavior is at least as conservative.

Never add a feature flag that disables a safety check.

## License

By contributing, you agree that your contribution will be licensed under Apache-2.0.

## Developer Certificate of Origin (DCO)

We use the [Developer Certificate of Origin](DCO) (DCO 1.1). It is a simple
statement that you wrote the contribution or otherwise have the right to submit
it under the project's license. You affirm it by **signing off** each commit:

```bash
git commit -s -m "feat(skill): add SmolVLA adapter"
```

This appends a line to your commit message:

```
Signed-off-by: Your Name <you@example.com>
```

Use the same name and email as your Git author identity. If you forgot to sign
off, amend the last commit with `git commit --amend -s`, or rebase to sign off a
series with `git rebase --signoff <base>`. CI checks that every commit in a PR
is signed off.

**Tip — never think about `-s` again.** `just bootstrap` installs a git hook
(`.githooks/prepare-commit-msg`) that auto-appends the `Signed-off-by` trailer
to any commit missing one, using your Git identity. To enable it without a full
bootstrap, run `just install-hooks` (it just points `core.hooksPath` at
`.githooks/`). Installing the hook is itself your standing DCO affirmation.

No separate CLA is required. See [GOVERNANCE.md](GOVERNANCE.md) for how the
project is run.
