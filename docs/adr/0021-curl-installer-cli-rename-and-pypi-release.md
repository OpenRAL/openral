# ADR-0021 — Curl-bash installer, CLI rename, and multi-package PyPI release scaffold

* **Status:** Accepted (build mode), 2026-05-24
* **Deciders:** Adrian Llopart (TSC)
* **Supersedes:** —
* **Related:** ADR-0004 (monorepo), ADR-0010 (inference runner), ADR-0011 (libero ↔ robocasa exclusion), CLAUDE.md §1.13/§1.14/§4

## Context

The OpenRAL workspace ships 13 distributable Python packages under
`python/*`, four optional simulator backends, and a sudo+apt ROS 2 system
bootstrap. The pre-existing installation flow is:

```
git clone … && cd openral
just bootstrap        # apt + uv + ROS 2 Jazzy + libusb
uv sync --all-packages
uv run ral …          # CLI never on $PATH
```

Three pain points fall out of this:

1. **No frictionless entry.** A user who wants to "try the CLI" must clone
   the repo, run a sudo-gated bash script, then prefix every command with
   `uv run`. There is no equivalent of `curl -fsSL https://claude.ai/install.sh | bash`.
2. **CLI name mismatch.** The console script is `ral`, which is a poor
   discoverability handle (`ral` is also the Polish word for "moray" — `git`
   grep collisions are common) and does not match the project name. The
   ergonomic name is `openral`.
3. **No PyPI presence.** `openral-core`, `openral-cli`, et al. are not
   published. The Tier-0 installer therefore has nothing to install from
   except git, and downstream consumers cannot `pip install openral-cli`.

## Decision

### 1. Rename the CLI to `openral` and add an interactive REPL

* `python/cli/pyproject.toml` `[project.scripts]` ships a single canonical
  entry point: `openral = "openral_cli.main:app"`. No `ral` alias, no
  deprecation banner.
* The Typer app uses `invoke_without_command=True`. When `openral` is
  invoked with **no subcommand**, the root callback drops into an
  interactive REPL (`openral_cli.main._run_repl`):
  * Prints an ASCII banner + the subtitle "The open-source agentic layer
    for physical AI".
  * Reads lines from `input()` (stdlib `readline` enabled when available
    for arrow-key history — no `prompt_toolkit` dependency, so the
    Tier-0 install stays at `uv tool install openral-cli`).
  * Each line is `shlex.split`-tokenised and re-dispatched through the
    same Typer app with `standalone_mode=False`, so subcommands run
    bare: `sim run --config foo.yaml` inside the REPL is equivalent to
    `openral sim run --config foo.yaml` on the shell.
  * `exit` / `quit` / `:q` / Ctrl-D leave the REPL; `help` / `?` runs
    `--help`.
  * `UsageError`, `Abort`, and `SystemExit` from subcommands are
    caught so a single bad invocation does not tear down the session.
* When `openral` is invoked **with** a subcommand the behaviour is
  unchanged: single one-shot run, no banner, OTel tracing scope opened
  per ADR-0010. This keeps `openral skill list | jq …` pipelines clean.
* All in-repo references (README, docs, Justfile, CLAUDE.md §4) use
  `openral` per CLAUDE.md §1.14.

### 2. Tiered curl-bash installer (`scripts/install.sh`)

The installer is explicitly **Tier-0 only** — it does what curl-bash can
honestly do without surprising the user with sudo or 10 GB of CUDA wheels:

| Step | Action | sudo? | Time |
|------|--------|-------|------|
| 1 | Detect OS / arch, refuse to run as root | no | <1 s |
| 2 | Install `uv` via `curl -LsSf https://astral.sh/uv/install.sh \| sh` if missing | no | ~5 s |
| 3 | `uv python install 3.12` (uv-managed CPython, no apt) | no | ~10 s |
| 4 | `uv tool install --python 3.12 openral-cli` | no | ~15 s |
| 5 | Verify `~/.local/bin/openral` exists; print PATH hint if needed | no | <1 s |
| 6 | Print the opt-in `openral install <group>` menu | no | <1 s |

Heavier groups (`sim`, `libero`, `metaworld`, `maniskill3`, `simpler-env`,
`robocasa`, `rldx`) layer in via the new `openral install <group>`
subcommand, which calls `uv pip install --python <tool-venv-python>` against
mirrored copies of the workspace `[dependency-groups]` table. The
sudo-gated `ros` group re-exec's `scripts/bootstrap_ubuntu.sh` /
`scripts/bootstrap_macos.sh` with a clear "this needs sudo" banner.

The libero ↔ robocasa mutual-exclusion declared in the root
`[tool.uv].conflicts` table (ADR-0011) is enforced inside
`openral install` as a typed `ROSConfigError` with a `--force` escape
hatch.

### 3. Multi-package PyPI release workflow

`.github/workflows/release-pypi.yml` is the **canonical, single** release
workflow (see the 2026-06-17 amendment for the consolidation):

* Two ways to publish, both via PyPI Trusted Publishing
  (`pypa/gh-action-pypi-publish@release/v1`, OIDC, no long-lived token):
  * `workflow_dispatch` with a `target` choice — `testpypi` (default, no
    confirmation) or `pypi` (requires `confirm=YES`).
  * tag push `v*.*.*` — production release to real PyPI; the tag is the
    deliberate confirmation.
* A `resolve` job picks the index + enforces the real-PyPI guard; a
  `precheck` job runs the same ruff + mypy + schema-drift + `mkdocs --strict`
  gate as the PR `quality` workflow, so a release cannot publish a broken tree.
* The publish matrix lists every distributable workspace member
  (14 packages today, incl. `openral-state-adapter`).
* The **TestPyPI** path is usable now (register a TestPyPI trusted publisher,
  or upload locally with twine). The **real-PyPI** path remains blocked until:
  1. Registering each `openral-*` name on PyPI under the openral org.
  2. Configuring the Trusted Publisher entry on PyPI for this repo +
     this workflow file.

Until then the Tier-0 installer ships pointing at PyPI (`spec=openral-cli`).
Operators bridging the pre-publish gap can override with
`OPENRAL_INSTALL_SOURCE=git+https://github.com/OpenRAL/openral`.

### 4. Version pin: all packages stay at 0.1.x

Per the feat/cleaner-cli directive, every `python/*/pyproject.toml` keeps
`version = "0.1.0"`. The first published release will tag `v0.1.0` after
the Trusted Publishing wiring lands.

## Consequences

* **Positive**
  * `openral` is on `$PATH` ≤30 s after a curl-bash install, no sudo, no
    clone.
  * The installation matrix (sim / libero / robocasa / ros) is explicit and
    enforced by typed errors, not by silently-failing `uv pip install`s.
  * Multi-package release pipeline is defined and reviewable today; flipping
    it on is a small follow-up PR after PyPI namespace setup.

* **Negative / accepted tradeoffs**
  * Two installation flows now exist (curl-bash for end users, `just
    bootstrap` for contributors). CLAUDE.md §4 keeps `just bootstrap`
    canonical for development; README.md leads with the curl one-liner for
    discovery.
  * The REPL adds a stateful interactive path on top of the one-shot
    surface, but the dispatcher reuses the same Typer app so there is
    one canonical command tree (no parallel REPL-only command set to
    keep in sync).
  * `openral install` duplicates the `[dependency-groups]` table from the
    workspace root pyproject — drift is caught by
    `tests/unit/test_install_command.py` which loads the root file when
    present and asserts the two stay in lockstep.

## Amendments

* _2026-05-19_ — initial Decision. Release workflow ships disabled; will
  be enabled in a follow-up after PyPI namespace + Trusted Publishing
  setup.
* _2026-05-23_ — §4 reaffirmed. All thirteen workspace packages
  (including `openral-core`) sit at `version = "0.1.0"`. A short-lived
  `openral-core 0.3.0` bump that landed alongside `SceneDefaults` /
  `TopCameraDefaults` was reverted to `0.1.0` to keep the entire
  workspace lockstep until the first PyPI publish. On-disk
  `schema_version` stays at `"0.1"` per master commit `39cb622` (drop
  schema migrators while pre-publish; on-disk shape evolves in place).
  Net directive: until the first publish, **neither package versions
  nor on-disk schema_version are bumped** — both stay at `"0.1"` /
  `"0.1.0"`, additions land in place, real-fixture tests prove the
  shape change.
* _2026-05-24_ — ADR index renumbering. ADR-0021 retains its number; former collisions renumbered to ADR-0022 (rSkill action vocabulary) and ADR-0023 (data-driven MuJoCo HAL).
* _2026-06-17_ — Release-pipeline consolidation. The separate
  `release.yml` (tag-triggered) was **dropped**: its root `uv build` was
  broken (the workspace root has no `[build-system]`, so it fell back to
  legacy setuptools and errored on multi-package discovery) and its PyPI half
  duplicated `release-pypi.yml` while building the wrong artifact (the
  meta-package). The ghcr runtime-image + cosign path it carried was
  non-functional (`Dockerfile.runtime` `COPY`s the gitignored colcon
  `install/` space) and is deferred to a future purpose-built
  `release-image.yml` once ROS-in-CI infra exists. `release-pypi.yml` is now
  the single source of truth (§3): tag-push trigger live, a `target` choice
  adds a TestPyPI path, a `precheck` gate guards against publishing a broken
  tree, and `openral-state-adapter` was added to the matrix (13 → 14).
