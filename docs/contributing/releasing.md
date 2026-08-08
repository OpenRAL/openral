# Releasing

OpenRAL ships **one lockstep SemVer version** across the whole workspace: the
root `pyproject.toml` and all 14 `python/*` packages carry the same number, and
every `openral-*` dependency pins it exactly. A release is a single `v*.*.*`
tag that publishes all 14 distributions to PyPI together.

Nobody picks that number by hand. It is derived from the Conventional Commits
merged since the last tag.

## The loop

```
merge PRs to master  →  release-please computes the bump  →  release PR
                                                                 │
                     release-pypi.yml  ←  tag v0.3.0  ←  merge the release PR
                             │
                             └→ 14 packages on PyPI (Trusted Publishing, OIDC)
```

1. **Every push to `master`** re-runs [`release-please.yml`](https://github.com/OpenRAL/openral/blob/master/.github/workflows/release-please.yml).
   It reads the commits since the tag recorded in `.release-please-manifest.json`
   and picks the segment:

   | Commit                              | Bump while `< 1.0`          | Bump at `>= 1.0` |
   | ----------------------------------- | --------------------------- | ---------------- |
   | `fix: …`                            | patch (`0.2.0` → `0.2.1`)   | patch            |
   | `feat: …`                           | minor (`0.2.0` → `0.3.0`)   | minor            |
   | `feat!: …` / `BREAKING CHANGE:`     | minor (`bump-minor-pre-major`) | major         |
   | `docs/test/style/ci/build/chore`    | none                        | none             |

   The highest-ranking commit in the range wins. A range containing only hidden
   types produces no release PR at all.

2. **The release PR** (`chore(release): vX.Y.Z`) is the human gate. It rewrites
   every version and every `openral-*==` pin, and regenerates `CHANGELOG.md`.
   Review the number and the notes there — the changelog body is editable, so
   curate it in the PR if a release deserves a narrative.

   Its CI is deliberately thin. `test-selective` short-circuits on the
   `release-please--*` branch and selects nothing: rewriting all 15 pyprojects
   matches the `pyproject.toml` full-run glob *and* marks every package
   changed, so the selector would otherwise run the entire suite plus every
   opt-in dependency lane — including `isaacsim` / `robotwin` / `gr00t`, whose
   sidecars a hosted runner cannot provision — for a diff that cannot change
   behaviour. Every commit the release covers already passed on its own PR.
   The job still *runs* and reports green, because it is a required check and
   a skipped required check is never reported. `quality` and `DCO` are
   unaffected, and `release-pypi.yml` re-runs the full `precheck` gate against
   the tag before anything reaches an index.

3. **Merging it** tags `vX.Y.Z` and publishes the GitHub release. The tag push
   triggers [`release-pypi.yml`](https://github.com/OpenRAL/openral/blob/master/.github/workflows/release-pypi.yml),
   which runs the same fast quality gate as PR CI and then publishes each
   package via PyPI Trusted Publishing.

## Check the release PR against `git log`

release-please parses each commit message with a conventional-commits grammar.
A message it cannot parse is **skipped silently** — no error, no entry in the
changelog, no contribution to the bump. Large squash-merges are the usual
casualty: GitHub composes the squash body from every sub-commit, and one
prose line containing source syntax is enough to fail the whole message.

A real example on this repo — `cc6182a` (PR #43), a 3658-line squash of ~150
sub-commits, fails to parse on the body line:

```
isinstance(x, (int, float)) admits bool
```

Everything in that PR is therefore missing from the generated changelog,
including a `refactor(reasoner)!` sub-commit with a `BREAKING CHANGE:` footer.
Nothing warns you.

So before merging a release PR, diff its changelog against the range it
covers:

```sh
git log --no-merges --format='%s' v<last>..origin/master
```

Anything in that list with a user-facing type (`feat`, `fix`, `refactor`,
`perf`) that is absent from the PR's changelog was dropped — add it by hand.
The changelog body is editable, and editing it is the intended fix; the merged
commit message cannot be corrected without rewriting `master`.

Two habits keep this rare: prefer several focused PRs over one very large
squash, and keep code snippets out of commit bodies (describe the fix in
prose, or fence the snippet in a PR comment instead).

## What you have to do

Write Conventional Commits (already required — [CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) §4.2).
That is the whole contract. Two escape hatches exist for the rare case where
the computed number is wrong:

- **Force a specific version** — put `Release-As: 0.4.0` in a commit footer.
- **Force a bump from an otherwise-silent change** — use `feat:` deliberately,
  or add a `BREAKING CHANGE:` footer describing the incompatibility.

Do **not** hand-edit a `version =` field to cut a release. Versions and pins
are rewritten by release-please via the `x-release-please-version` /
`x-release-please-start-version` annotations in each `pyproject.toml`; editing
around them puts the manifest and the tree out of sync.

## Adding a package

A new `python/<name>` package needs three edits, all enforced by
`tests/unit/test_lockstep_versions.py`:

1. Its `pyproject.toml` starts at the current lockstep version, with
   `# x-release-please-version` on the `version =` line and its `openral-*`
   deps grouped in an `x-release-please-start-version` / `x-release-please-end`
   block, pinned `==<current version>`.
2. Add `python/<name>/pyproject.toml` to `extra-files` in
   [`release-please-config.json`](https://github.com/OpenRAL/openral/blob/master/release-please-config.json).
3. Add `python/<name>` to the publish matrix in `release-pypi.yml`.

## Token setup (one time)

GitHub does not trigger workflows from events created with the default
`GITHUB_TOKEN`. Without a dedicated token, release-please's tag lands but
`release-pypi.yml` never fires.

Set the `RELEASE_PLEASE_TOKEN` repository secret to a **GitHub App
installation token** (preferred) or a fine-grained PAT with `contents:write`
and `pull-requests:write`. This also makes the release PR run the required
`quality` and `DCO` checks normally, so it merges without an admin bypass.

Without the secret the job still runs and says so with a `::warning::`. Recover
a stranded tag by dispatching `release-pypi.yml` manually with `target=pypi`,
`confirm=YES`.

## Trial runs

`release-pypi.yml` also accepts a `workflow_dispatch` with `target=testpypi`
(the default), which publishes to TestPyPI without any confirmation. Use it to
validate packaging changes without burning a version number.

## What is *not* versioned by this

`schema_version` on the on-disk Pydantic schemas is a **separate** field with
its own rules ([CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) §1.6): a backward-incompatible
schema change bumps it and ships a migrator, independent of the package
version.
