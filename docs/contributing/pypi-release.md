# PyPI release & Trusted Publisher setup

How to publish the `openral-*` packages to PyPI. The release pipeline
([`release-pypi.yml`](https://github.com/OpenRAL/openral/blob/master/.github/workflows/release-pypi.yml),
ADR-0021 §3) is wired and tested against TestPyPI; the steps below are the
one-time **PyPI-side** setup that only a maintainer can do in the web UI.

!!! note "Why this is gated"
    Publishing to real PyPI is irreversible — a version number can never be
    reused and project names are claimed permanently. Real-PyPI publishing
    stays blocked until the OpenRAL org / Trusted Publisher are configured.
    Do **not** publish `openral-*` under a personal account: it squats the
    names and complicates the eventual org transfer.

## The Trusted Publisher config (used for every package)

Every entry uses these **identical** values — only the project name changes.
No API token is stored anywhere; the workflow exchanges a GitHub OIDC token
for a short-lived PyPI credential (`pypa/gh-action-pypi-publish`).

| Field | Value |
| --- | --- |
| Owner | `OpenRAL` |
| Repository name | `openral` |
| Workflow name | `release-pypi.yml` |
| Environment name | *(leave blank — the workflow defines none)* |
| PyPI Project Name | one of the 14 below |

The 14 distributable packages (must match the `release-pypi.yml` matrix):

```
openral-core      openral-cli        openral-observability  openral-detect
openral-sensors   openral-hal        openral-world-state    openral-runner
openral-sim       openral-rskill     openral-reasoner       openral-wam
openral-dataset   openral-state-adapter
```

## 1. Organization (optional)

You do **not** need an org to publish — `openral-*` are ordinary project
names. The org only groups/manages them. Two routes:

- **Wait for the org.** Request it at
  [pypi.org/manage/organizations](https://pypi.org/manage/organizations/) →
  *Create organization* (name `openral`, type *Community* = free, requires
  PyPI approval). Once approved, create the pending publishers from the org's
  Publishing page so projects land under the org.
- **Publish under your account now, transfer later.** Set up the pending
  publishers under your personal account (step 2), publish, then transfer each
  project to the org once approved (project → *Settings* → *Transfer to an
  organization*).

## 2. Pending Trusted Publishers (one per package)

Because the projects do not exist on PyPI yet, use **pending** publishers —
PyPI creates the project on first publish.

1. Go to
   [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/)
   (or the org's Publishing page for the org route).
2. Under **Add a pending publisher → GitHub**, fill in the config table above
   with **PyPI Project Name = `openral-core`** and click **Add**.
3. Repeat for all 14 project names. PyPI has no wildcard — one entry per
   project.

## 3. Release and verify

With the pending publishers in place, from a clean `master`:

```bash
git checkout master && git pull
git tag v0.1.0
git push origin v0.1.0
```

The tag fires `release-pypi.yml` → `resolve` (→ real PyPI) → `precheck` gate →
`build-and-publish` matrix publishes all 14 via OIDC. After it is green, the
curl one-liner is finally testable end-to-end:

```bash
curl -fsSL https://raw.githubusercontent.com/OpenRAL/openral/master/scripts/install.sh | bash
```

!!! warning "`v0.1.0` is one-shot"
    Once pushed, that version is permanent on PyPI. Ensure master is fully
    green and includes every release-readiness fix before tagging.

## TestPyPI (optional, tokenless CI path)

To run `workflow_dispatch → target=testpypi` without a token, repeat step 2 at
[test.pypi.org/manage/account/publishing](https://test.pypi.org/manage/account/publishing/)
(TestPyPI is a separate site/account). Not required — a local
`twine upload --repository testpypi dist/*` already validates a publish.

!!! info "TestPyPI cannot validate the full install"
    TestPyPI is squatted with placeholder builds of common dependencies (e.g.
    `rich`, `fastapi`), and uv's first-index guard will not fall through to
    real PyPI for them — so `OPENRAL_INSTALL_INDEX=https://test.pypi.org/...`
    cannot resolve a full `openral-cli` install. Use TestPyPI to verify a
    *publish*; verify a full *install* against real PyPI (or a local wheel
    build with `--find-links`).
