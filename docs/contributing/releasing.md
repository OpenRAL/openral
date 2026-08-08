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

   It also carries a second commit, `chore(release): refresh uv.lock for the
   version bump`, pushed by the same workflow. release-please only substitutes
   text in the files `extra-files` names, and `uv.lock` is generated — it
   records every workspace member's version, so a bump that stops at the
   pyprojects leaves the lock a release behind (v0.3.0 and v0.3.1 both did:
   master ran 0.3.1 pyprojects against a lock still saying 0.2.0, and
   `uv sync --frozen` installed that stale metadata). The step runs `uv lock`
   on the release branch; without `--upgrade` it rewrites only the 15 version
   lines and leaves third-party pins alone. `test_lockstep_versions.py` fails
   if the lock and the tree ever disagree again.

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

release-please expands a squash-merge body: GitHub composes it from every
sub-commit, and each `* type(scope): subject` line is parsed as its own
entry. When the body fails the conventional-commits grammar — one prose line
containing source syntax is enough — the expansion is **dropped silently**.
The commit's own subject still lands, so the failure looks like a normal
one-line entry rather than an error.

A real example on this repo: `cc6182a` (PR #43) is a 3658-line squash of ~150
sub-commits whose body fails to parse on the line

```
isinstance(x, (int, float)) admits bool
```

In the 0.3.0 changelog it appears as a single line under **Changed** — its
own subject, `perf(sensors): 9x faster skill load…`. The ~150 sub-commits are
absent, and so is the `refactor(reasoner)!` sub-commit's `BREAKING CHANGE:`
footer: 0.3.0 has no "⚠ BREAKING CHANGES" section even though the release
removes the `OPENRAL_REASONER_LLM_*` contract. Nothing warns you.

So before merging a release PR, diff its changelog against the range it
covers:

```sh
git log --no-merges --format='%s' v<last>..origin/master
```

Anything in that list with a user-facing type (`feat`, `fix`, `refactor`,
`perf`) that is absent from the PR's changelog was dropped — add it by hand.
Check large squashes line by line, and check for a missing "⚠ BREAKING
CHANGES" section against `git log --grep='BREAKING CHANGE'`. The changelog
body is editable, and editing it is the intended fix; the merged commit
message cannot be corrected without rewriting `master`.

A merge commit needs the same care in the other direction: keep its message
free of any `type(scope): subject` line, or release-please parses the merge
*and* the branch commit and the entry appears twice. The 0.3.0 changelog
carries exactly that duplicate for `feat(release)` (`f1d1b1f` and `5ba709a`).

Two habits keep this rare: prefer several focused PRs over one very large
squash, and keep code snippets out of commit bodies (describe the fix in
prose, or fence the snippet in a PR comment instead).

## When a release stalls between merge and tag

release-please can find a merged release PR it cannot build a release from, and
then decline to open a new one — logging `There are untagged, merged release
PRs outstanding - aborting`. It emits no outputs in that state, which is
byte-for-byte what "nothing releasable merged" looks like. v0.3.0 sat stalled
across two runs before this was noticed; no tag was created, so nothing
published.

The `summarise` step in [`release-please.yml`](https://github.com/OpenRAL/openral/blob/master/.github/workflows/release-please.yml)
now distinguishes them by asking the repository rather than the action: a
**merged** PR still labelled `autorelease: pending` is a release that stopped
between merging and tagging. The job fails loudly in that case.

If you see that error, the reason is in the same job's log, above the abort —
look for a line naming what could not be built. The one that bit us was:

```
⚠ PR component: undefined does not match configured component: openral
```

`getBranchComponent()` derives a component from `package-name` and, unlike
`getComponent()`, ignores `include-component-in-tag`; the release branch has no
component segment, so the two could never match. Removing `package-name` fixed
it. Once the config is right, a re-run tags the already-merged PR — the PR does
not need to be reopened or remade.

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

`release-pypi.yml` accepts a `workflow_dispatch` with `target=testpypi` (the
default), which publishes to TestPyPI without confirmation. Use it to validate
packaging without burning a version number — `skip-existing` means re-running
it against an already-published version is a no-op, so it costs nothing.

It is worth doing before any release that changes packaging metadata, because
it is the only way to see what installers will actually read. After it runs,
check the published requirements rather than the source:

```sh
curl -s https://test.pypi.org/pypi/openral-cli/<version>/json \
  | python3 -c "import sys,json;[print(r) for r in json.load(sys.stdin)['info']['requires_dist']]"
```

The 0.2.0 trial confirmed every `openral-*` sibling carries its `==` pin, with
extras and markers intact (`openral-observability[dashboard]==0.2.0`,
`pygobject>=3.42; extra == "gstreamer"`). Neither `uv lock --check` nor the
unit tests can show that — they read source, and `[tool.uv.sources]` is
stripped at build time.

### When the token exchange fails

A `invalid-publisher: valid token, but no corresponding publisher` error means
the OIDC claims do not match any registered trusted publisher — nothing is
uploaded. Register one per project with `owner: OpenRAL`, `repo: openral`,
`workflow: release-pypi.yml`, and **no** environment name (the workflow
declares none, so the claim is `environment: MISSING` and a publisher that
expects one will never match).

Trusted publishing also matches on the ref, so claims are not portable between
triggers: a tag push presents `refs/tags/vX.Y.Z`, a dispatch off master
presents `refs/heads/master`. A publisher with no ref restriction accepts
both; one restricted to tags rejects the dispatch. A green tag release
therefore does not imply a green dispatch, and vice versa.

## What is *not* versioned by this

`schema_version` on the on-disk Pydantic schemas is a **separate** field with
its own rules ([CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) §1.6): a backward-incompatible
schema change bumps it and ships a migrator, independent of the package
version.
