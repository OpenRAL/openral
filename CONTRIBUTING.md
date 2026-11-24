# Contributing to OpenRAL

[CLAUDE.md](CLAUDE.md) is the engineering playbook for humans and AI agents: coding standards, architecture rules, the PR checklist. Read it before you change code. For orientation, see the [architecture overview](docs/architecture/overview.md), the [roadmap](docs/roadmap/index.md) and the [repo state map](docs/architecture/repo-state-map.html) (PRs that change the module set must update it).

## Set up

```bash
just bootstrap   # uv, ROS 2, system deps, git hooks
just sync        # Python workspace (never bare `uv sync`)
just test        # unit suite
```

Other setups (dev container, Codespaces), the recipe table and troubleshooting: [development.md](docs/contributing/development.md). Environment rules and dependency groups: [toolchain.md](docs/contributing/toolchain.md).

## Workflow

1. Fork and create a feature branch.
2. Write tests first for anything on the actuation path.
3. Run `just lint && just test` before pushing, plus the relevant `just sim-*` recipe.
4. Open a PR with the [PR template](.github/PULL_REQUEST_TEMPLATE.md) and a [Conventional Commits](https://www.conventionalcommits.org/) title (`feat(skill): add SmolVLA adapter`, `fix(safety): clamp ee_speed before publish`). The type decides the next release's version bump ([releasing](docs/contributing/releasing.md)).
5. PRs over 800 lines need maintainer pre-approval.

Updating docs, `docs/methods/` and the repo state map is part of the change, not a follow-up (CLAUDE.md §4.4). How CI runs and what blocks a merge: [ci.md](docs/contributing/ci.md).

## Safety

Touching `packages/openral_safety/` or `cpp/openral_safety_kernel/` needs a safety working group reviewer, a hazard-log update, and tests proving the new behavior is at least as conservative. Never add a flag that disables a safety check.

## License and DCO

By contributing, you agree your contribution is licensed under Apache-2.0. We use the [Developer Certificate of Origin](DCO): sign off every commit.

```bash
git commit -s -m "feat(skill): add SmolVLA adapter"      # adds Signed-off-by from your Git identity
git commit --amend -s                                    # fix the last commit
git rebase --signoff <base>                              # fix a series
```

`just install-hooks` (run by `just bootstrap`) sets `git config core.hooksPath .githooks`, which installs a hook that adds the `Signed-off-by` trailer automatically, and the pre-commit checks. Without it, commits are silently unsigned and CI's `Verify Signed-off-by` check fails. Installing the hooks is itself your standing DCO affirmation.

No separate CLA is required. See [GOVERNANCE.md](GOVERNANCE.md) for how the project is run.
