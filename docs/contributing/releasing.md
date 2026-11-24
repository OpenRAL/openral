# Releasing

OpenRAL ships one lockstep SemVer version: the root `pyproject.toml` and all 13 `python/*` packages carry the same number, and every `openral-*` dependency pins it exactly. One `v*.*.*` tag publishes all 13 distributions to PyPI. Nobody picks the number: [release-please](https://github.com/googleapis/release-please) derives it from the Conventional Commits merged since the last tag.

Do not hand-edit a `version =` field or an `openral-*==` pin; release-please rewrites them through the `x-release-please-*` annotations in each `pyproject.toml`. Nothing under `src/` carries a version either: each package's `__version__` reads `importlib.metadata`, and `tests/unit/test_lockstep_versions.py` rejects a literal.

## The loop

```
merge PRs to master  →  release-please computes the bump  →  release PR
                                                                 │
                     release-pypi.yml  ←  tag v0.3.0  ←  merge the release PR
                             │
                             └→ 13 packages on PyPI (Trusted Publishing, OIDC)
```

1. **Every push to `master`** runs [`release-please.yml`](https://github.com/OpenRAL/openral/blob/master/.github/workflows/release-please.yml), which reads the commits since the tag in `.release-please-manifest.json`. The highest-ranking commit wins; a range with only hidden types opens no release PR.

   | Commit | Bump while `< 1.0` | Bump at `>= 1.0` |
   | --- | --- | --- |
   | `fix: ...` | patch | patch |
   | `feat: ...` | minor | minor |
   | `feat!: ...` / `BREAKING CHANGE:` | minor (`bump-minor-pre-major`) | major |
   | `docs` / `test` / `style` / `ci` / `build` / `chore` | none | none |

2. **The release PR** (`chore(release): vX.Y.Z`) is the human gate. It rewrites every version and `openral-*==` pin and regenerates `CHANGELOG.md`; the changelog body is editable, so curate it there. A second commit, `chore(release): refresh uv.lock for the version bump`, runs `uv lock` (no `--upgrade`) so the lock's workspace versions match. The workflow pins uv to the version that wrote master's lock, because a newer uv re-locks in its own format and `tools/check_release_diff.py` refuses that as more than version churn.

3. **Merging it** tags `vX.Y.Z` and publishes the GitHub release. The tag triggers [`release-pypi.yml`](https://github.com/OpenRAL/openral/blob/master/.github/workflows/release-pypi.yml): the same quality gate as PR CI, then one Trusted Publishing upload per package.

## CI on the release PR

`test-selective` selects nothing on a `release-please--*` branch, because rewriting 14 pyprojects would otherwise run the full suite and every dependency lane for a diff that cannot change behaviour. The job still runs and reports green (a skipped required check is never reported). The skip applies only when the branch lives in this repo and [`tools/check_release_diff.py`](https://github.com/OpenRAL/openral/blob/master/tools/check_release_diff.py) finds the diff is only OpenRAL's own versions and pins moved in lockstep; any dependency edit, off-version pin or fork head selects tests normally. `quality` and `DCO` run as usual, and `release-pypi.yml` re-runs its `precheck` against the tag.

## What you have to do

Write Conventional Commits (already required, [CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) §4.2). For the rare wrong number:

- **Force a version:** put `Release-As: 0.4.0` in a commit footer.
- **Force a bump from a silent change:** use `feat:`, or add a `BREAKING CHANGE:` footer describing the incompatibility.

`schema_version` on on-disk Pydantic schemas is separate; it follows [CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) §1.6, not the package version.

## Before merging a release PR

release-please parses each `* type(scope): subject` line of a squash body as its own entry, and silently drops the whole expansion when a body line breaks the grammar (a prose line containing code such as `isinstance(x, (int, float))` is enough). The changelog then shows only the squash subject, and a dropped `BREAKING CHANGE:` footer can leave out the breaking-changes section.

- **If the changelog looks short:** compare it with `git log --no-merges --format='%s' v<last>..origin/master`. Add any missing `feat`, `fix`, `refactor` or `perf` entry by hand, and check `git log --grep='BREAKING CHANGE'` against the breaking-changes section. Edit the PR body; the merged commit cannot be fixed without rewriting `master`.
- **If you write a merge commit message:** keep any `type(scope): subject` line out of it, or the entry appears twice.
- **To avoid it:** prefer several focused PRs to one huge squash, and describe code in prose rather than pasting snippets into commit bodies.

`release-please-config.json` sets `commit-search-depth` to 2000 (the release-please default is 500). It counts every commit in the range, merged-branch commits included, and older commits drop out of the notes with no warning. `test_lockstep_versions.py` fails when the commits since the last tag reach the depth.

- **If that test fails:** cut a release (or raise the depth). Count with `git rev-list --count v<last>..origin/master`.

## When a release run fails

- **If the log shows `release-please failed:` with an HTML "Unicorn!" page (GitHub 5xx):** the workflow already retried once after 60 s. Re-run the job from the Actions tab or push anything to `master`.
- **If the `summarise` step fails loudly:** a merged PR still labelled `autorelease: pending` means a release stopped between merge and tag, and release-please aborts with `There are untagged, merged release PRs outstanding`. The cause is on a log line above the abort, such as `PR component: undefined does not match configured component`. Fix the config (do not set `package-name`; it breaks the component match) and re-run; the already-merged PR is tagged without being reopened.
- **If the tag landed but `release-pypi.yml` never fired:** the `RELEASE_PLEASE_TOKEN` secret is missing (events made with the default `GITHUB_TOKEN` trigger no workflows). Dispatch `release-pypi.yml` by hand with `target=pypi`, `confirm=YES`.
- **If PyPI says `invalid-publisher: valid token, but no corresponding publisher`:** the OIDC claims match no trusted publisher. Register one per project with `owner: OpenRAL`, `repo: openral`, `workflow: release-pypi.yml` and no environment name. Publishers match on the ref too: a tag push presents `refs/tags/vX.Y.Z`, a dispatch off `master` presents `refs/heads/master`, so a green tag release does not imply a green dispatch.

## Setup

**`RELEASE_PLEASE_TOKEN`** (repository secret, one time): a GitHub App installation token (preferred) or a fine-grained PAT with `contents:write` and `pull-requests:write`. It also makes the release PR run `quality` and `DCO`, so it merges without an admin bypass. Without it the job runs and warns.

**Adding a package.** A new `python/<name>` needs three edits, enforced by `test_lockstep_versions.py`:

1. Its `pyproject.toml` starts at the current version with `# x-release-please-version` on the `version =` line, and its `openral-*` deps sit in an `x-release-please-start-version` / `x-release-please-end` block pinned `==<current version>`.
2. Add `python/<name>/pyproject.toml` to `extra-files` in [`release-please-config.json`](https://github.com/OpenRAL/openral/blob/master/release-please-config.json).
3. Add `python/<name>` to the publish matrix in `release-pypi.yml`.

## Trial runs

A `workflow_dispatch` of `release-pypi.yml` with `target=testpypi` (the default) publishes to TestPyPI without confirmation; `skip-existing` makes a re-run a no-op. Do this before any release that changes packaging metadata, then check the published requirements rather than the source (`[tool.uv.sources]` is stripped at build time, so lock checks and unit tests cannot see it):

```sh
curl -s https://test.pypi.org/pypi/openral-cli/<version>/json \
  | python3 -c "import sys,json;[print(r) for r in json.load(sys.stdin)['info']['requires_dist']]"
```

Every `openral-*` sibling should carry its `==` pin, with extras and markers intact.
