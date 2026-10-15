# Contributing to OpenRAL

## Before you start

- Read [CLAUDE.md](CLAUDE.md), the engineering playbook and single source of truth.
- Read the [Architecture overview](docs/architecture/overview.md).
- Skim the [roadmap](docs/roadmap/index.md): what is done, in flight, or planned.
- Open [`docs/architecture/repo-state-map.html`](docs/architecture/repo-state-map.html) in a browser to see which modules exist, are in flight, or are spec-only. PRs that change this set must update the map (CLAUDE.md §4.3).
- Check open issues for context, especially those labelled `safety`, `vla`, or `hardware`.

## Development setup

```bash
just bootstrap   # installs uv, ROS 2, system deps
just sync        # install Python workspace (never bare `uv sync`)
just test        # unit suite
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
docs: record the Pydantic-over-dataclasses decision
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

We use the [Developer Certificate of Origin](DCO) (DCO 1.1): a statement that
you wrote the contribution or otherwise have the right to submit it under the
project's license. Affirm it by **signing off** each commit:

```bash
git commit -s -m "feat(skill): add SmolVLA adapter"
```

This appends a line to your commit message:

```
Signed-off-by: Your Name <you@example.com>
```

Use the same name and email as your Git author identity. Forgot? Amend the last
commit with `git commit --amend -s`, or sign off a series with
`git rebase --signoff <base>`. CI checks that every commit in a PR is signed off.

**Tip:** `just bootstrap` installs the git hooks (`.githooks/`), including a
`prepare-commit-msg` hook that appends the `Signed-off-by` trailer, from your Git
identity, to any commit missing one. It recognises an existing sign-off by your
**email**, so a replayed commit (`reword`, `edit`, `rebase --continue`) never
gets a second trailer even if the recorded name differs from your `user.name`; a
`---` line in the body is treated as prose, not the end of the message.

Without a full bootstrap, run `just install-hooks`: it points `core.hooksPath` at
`.githooks/` and builds the pre-commit environments, which also wires up ruff,
ruff-format, mypy, codespell and the conventional-commit check (through the
committed `.githooks/pre-commit` and `.githooks/commit-msg` wrappers, since
`pre-commit install` refuses to run with `core.hooksPath` set). Installing the
hooks is itself your standing DCO affirmation.

No separate CLA is required. See [GOVERNANCE.md](GOVERNANCE.md) for how the
project is run.
