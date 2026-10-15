"""Start the opt-in heavy lanes on a PR once it is ready for them — no manual trigger.

``heavy-lanes`` is a required check that ``.github/workflows/heavy-lanes.yml``
reports, and that workflow runs only via ``workflow_dispatch``. A dispatched
run's own check runs are not linked to any PR (GitHub only lists a PR's
``pull_request`` runs), so the result is reported as a **commit status** named
``heavy-lanes`` on the PR head instead — commit statuses show on the PR and
satisfy the required check. This tool keeps that status current: ``pending``
with the reasons while it waits, ``pending`` "Running" once it dispatches, and
the lanes' gate job posts ``success`` / ``failure`` / ``error`` (cancelled). It
dispatches for an open, non-draft PR whose current head commit meets all of
these:

* **up to date with its base** — the compare API reports the head 0 commits
  behind the base branch (rebased, or base merged in);
* **every other check is green** — every check run on the head is completed
  as success / skipped / neutral, every commit status is ``success``, and the
  ``--require-check`` names (the other required checks) are present, so a
  workflow that has not reported yet does not read as green;
* **every review thread is resolved**.

It never dispatches twice for one head commit: a heavy-lanes run already
queued, running, or finished (other than ``cancelled``) for that SHA, or a
``heavy-lanes`` status saying so, is left alone. A failed lane is re-run from
the Actions page, not re-dispatched.

Fork PRs get the same treatment. ``workflow_dispatch`` cannot target a fork
branch, and a fork's ``heavy-lanes.yml`` is not trusted, so the lanes run from
``heavy-lanes/fork-<pr>``: a branch this tool points at a fresh commit with the
base branch's exact tree (reviewed workflow, own cache scope — fork code can't
plant a cache the base branch restores — and its checks don't land on a base
commit). The ``sha`` input makes the lane jobs check out the fork head, with
the same read-only token and no secrets a fork ``pull_request`` run gets; the
status-posting gate job never runs fork code. Branches of closed PRs are
deleted on the next full sweep.

Run (from ``.github/workflows/heavy-lanes-trigger.yml``)::

    python tools/heavy_lanes_trigger.py --repo OpenRAL/openral
    python tools/heavy_lanes_trigger.py --repo OpenRAL/openral --pr 302 --dry-run

Example:
    >>> pr = PullRequestState(
    ...     number=1,
    ...     draft=False,
    ...     head_ref="feat/x",
    ...     head_sha="abc",
    ...     base_ref="master",
    ...     same_repo=True,
    ...     behind_by=0,
    ...     checks=[
    ...         CheckRunState(name="select-and-test", status="completed", conclusion="success")
    ...     ],
    ...     statuses=[],
    ...     unresolved_threads=0,
    ...     lane_runs=[],
    ... )
    >>> readiness(pr, required_checks=["select-and-test"]).ready
    True
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pydantic import BaseModel

API = "https://api.github.com"
LANES_WORKFLOW = "heavy-lanes.yml"
# Check runs heavy-lanes.yml itself posts, plus the trigger workflow's own job
# (a workflow_run-triggered sweep posts it on the triggering PR's head) —
# never a precondition for starting the lanes.
OWN_CHECK_NAMES = frozenset({"lanes-select", "heavy-lanes", "lane", "heavy-lanes-trigger"})
OWN_CHECK_PREFIXES = ("lane (",)
GREEN_CONCLUSIONS = frozenset({"success", "skipped", "neutral"})
# The commit status that carries the heavy-lanes result onto the PR. Its name
# is the required check's name, so posting it is what satisfies the ruleset.
STATUS_CONTEXT = "heavy-lanes"
RUNNING = "Running"
WAITING = "Waiting: "
STATUS_DESCRIPTION_MAX = 140  # GitHub rejects longer status descriptions
FORK_BRANCH_PREFIX = "heavy-lanes/fork-"
DEFAULT_REQUIRED_CHECKS = ("select-and-test", "quality", "Verify Signed-off-by")


class GitHubAPIError(RuntimeError):
    """A GitHub response that is not usable: a GraphQL ``errors`` payload or a null result."""


class CheckRunState(BaseModel):
    """One check run on the PR head: ``name``, ``status``, ``conclusion``."""

    name: str
    status: str
    conclusion: str | None = None


class StatusState(BaseModel):
    """One commit status on the PR head: ``context``, ``state``, ``description``."""

    context: str
    state: str
    description: str = ""


class LaneRunState(BaseModel):
    """One ``heavy-lanes.yml`` workflow run for the PR head."""

    status: str
    conclusion: str | None = None


class PullRequestState(BaseModel):
    """Everything the readiness verdict reads about one PR, as fetched from GitHub."""

    number: int
    draft: bool
    head_ref: str
    head_sha: str
    base_ref: str
    same_repo: bool
    behind_by: int
    checks: list[CheckRunState]
    statuses: list[StatusState]
    unresolved_threads: int
    lane_runs: list[LaneRunState]


class Verdict(BaseModel):
    """``ready`` to dispatch, else the list of ``reasons`` why not."""

    ready: bool
    reasons: list[str]


def _is_own_check(name: str) -> bool:
    return name in OWN_CHECK_NAMES or name.startswith(OWN_CHECK_PREFIXES)


def own_status(pr: PullRequestState) -> StatusState | None:
    """The ``heavy-lanes`` status on ``pr``'s head, if any."""
    return next((s for s in pr.statuses if s.context == STATUS_CONTEXT), None)


def lanes_started(pr: PullRequestState) -> bool:
    """A run (or a status) says the lanes already ran or are running on this head.

    ``cancelled`` runs and the gate's ``error`` status (posted on cancel) don't
    count, so a cancelled dispatch is retried.
    """
    if any(r.status != "completed" or r.conclusion != "cancelled" for r in pr.lane_runs):
        return True
    status = own_status(pr)
    if status is None:
        return False
    return status.state in ("success", "failure") or status.description.startswith(RUNNING)


def readiness(pr: PullRequestState, required_checks: list[str]) -> Verdict:
    """Decide whether the heavy lanes should start on ``pr``'s head now.

    Pure: every fact comes from ``pr``. An empty ``reasons`` list means ready.
    """
    reasons: list[str] = []
    if pr.draft:
        reasons.append("draft PR")
    if lanes_started(pr):
        reasons.append("heavy lanes already started for this head commit")
    if pr.behind_by > 0:
        reasons.append(f"{pr.behind_by} commit(s) behind {pr.base_ref} — rebase first")
    others = [c for c in pr.checks if not _is_own_check(c.name)]
    present = {c.name for c in others}
    for name in required_checks:
        if name not in present:
            reasons.append(f"check {name!r} has not reported yet")
    for check in others:
        if check.status != "completed":
            reasons.append(f"check {check.name!r} is {check.status}")
        elif check.conclusion not in GREEN_CONCLUSIONS:
            reasons.append(f"check {check.name!r} concluded {check.conclusion}")
    for status in pr.statuses:
        if status.context != STATUS_CONTEXT and status.state != "success":
            reasons.append(f"commit status {status.context!r} is {status.state}")
    if pr.unresolved_threads:
        reasons.append(f"{pr.unresolved_threads} unresolved review thread(s)")
    return Verdict(ready=not reasons, reasons=reasons)


class GitHub:
    """Minimal GitHub REST + GraphQL client over ``urllib`` (the only network boundary)."""

    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self._token = token

    def _request(self, method: str, url: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None

    def get(self, path: str) -> Any:
        """GET ``/repos/<repo>/<path>`` and return the decoded JSON."""
        return self._request("GET", f"{API}/repos/{self.repo}/{path}")

    def paginate(self, path: str, key: str | None = None) -> list[Any]:
        """Collect every page of a list endpoint (``key`` picks the list out of an object)."""
        sep = "&" if "?" in path else "?"
        items: list[Any] = []
        page = 1
        while True:
            data = self.get(f"{path}{sep}per_page=100&page={page}")
            batch = data[key] if key else data
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1

    def unresolved_threads(self, number: int) -> int:
        """Count PR ``number``'s unresolved review threads (GraphQL ``reviewThreads``)."""
        owner, name = self.repo.split("/")
        query = """
        query($owner: String!, $name: String!, $number: Int!, $after: String) {
          repository(owner: $owner, name: $name) {
            pullRequest(number: $number) {
              reviewThreads(first: 100, after: $after) {
                nodes { isResolved }
                pageInfo { hasNextPage endCursor }
              }
            }
          }
        }"""
        count = 0
        after: str | None = None
        while True:
            variables = {"owner": owner, "name": name, "number": number, "after": after}
            data = self._request("POST", f"{API}/graphql", {"query": query, "variables": variables})
            # GraphQL reports rate limits, missing scopes and a vanished PR
            # as HTTP 200 + `errors` (with `data` null), not as an HTTPError.
            if data.get("errors") or not data.get("data"):
                raise GitHubAPIError(f"GraphQL: {data.get('errors') or 'no data'}")
            threads = data["data"]["repository"]["pullRequest"]["reviewThreads"]
            count += sum(1 for t in threads["nodes"] if not t["isResolved"])
            if not threads["pageInfo"]["hasNextPage"]:
                return count
            after = threads["pageInfo"]["endCursor"]

    def fetch(self, pr: dict[str, Any]) -> PullRequestState:
        """Gather everything :func:`readiness` needs about one PR from the REST/GraphQL API."""
        sha = pr["head"]["sha"]
        base = pr["base"]["ref"]
        compare = self.get(f"compare/{urllib.parse.quote(base, safe='')}...{sha}")
        checks = self.paginate(f"commits/{sha}/check-runs?filter=latest", key="check_runs")
        combined = self.get(f"commits/{sha}/status")
        # Only dispatched runs: a fork can add `on: pull_request` to its own copy
        # of heavy-lanes.yml, and such a run must neither block dispatch nor be
        # backfilled into the `heavy-lanes` status.
        runs = self.get(
            f"actions/workflows/{LANES_WORKFLOW}/runs?head_sha={sha}&event=workflow_dispatch"
        )
        head_repo = pr["head"]["repo"]
        return PullRequestState(
            number=pr["number"],
            draft=pr["draft"],
            head_ref=pr["head"]["ref"],
            head_sha=sha,
            base_ref=base,
            same_repo=bool(head_repo) and head_repo["full_name"] == self.repo,
            behind_by=compare["behind_by"],
            checks=[CheckRunState.model_validate(c) for c in checks],
            statuses=[StatusState.model_validate(s) for s in combined["statuses"]],
            unresolved_threads=self.unresolved_threads(pr["number"]),
            lane_runs=[LaneRunState.model_validate(r) for r in runs["workflow_runs"]],
        )

    def dispatch(self, pr: PullRequestState) -> None:
        """Start ``heavy-lanes.yml`` for the PR head via ``workflow_dispatch``.

        A same-repo PR runs on its own branch (so a PR that edits the workflow
        tests its edit); a fork PR runs the base branch's workflow on its head
        commit, because a fork's workflow file is not trusted.
        """
        ref = pr.head_ref if pr.same_repo else self.fork_branch(pr)
        self._request(
            "POST",
            f"{API}/repos/{self.repo}/actions/workflows/{LANES_WORKFLOW}/dispatches",
            {
                "ref": ref,
                "inputs": {"base": f"origin/{pr.base_ref}", "head": "HEAD", "sha": pr.head_sha},
            },
        )

    def fork_branch(self, pr: PullRequestState) -> str:
        """Point ``heavy-lanes/fork-<pr>`` at a new commit with the base branch's tree.

        Same files as the base head, so the reviewed workflow runs; a commit of
        its own, so the lanes' check runs don't land on a base-branch commit.
        """
        base = self.get(f"git/ref/heads/{urllib.parse.quote(pr.base_ref)}")
        base_sha = base["object"]["sha"]
        tree = self.get(f"git/commits/{base_sha}")["tree"]["sha"]
        commit = self._request(
            "POST",
            f"{API}/repos/{self.repo}/git/commits",
            {
                "message": f"heavy lanes for fork PR #{pr.number} @ {pr.head_sha[:7]}",
                "tree": tree,
                "parents": [base_sha],
            },
        )
        branch = f"{FORK_BRANCH_PREFIX}{pr.number}"
        try:
            self._request(
                "PATCH",
                f"{API}/repos/{self.repo}/git/refs/heads/{branch}",
                {"sha": commit["sha"], "force": True},
            )
        except urllib.error.HTTPError as exc:
            if exc.code != 422:  # 422: the branch does not exist yet
                raise
            self._request(
                "POST",
                f"{API}/repos/{self.repo}/git/refs",
                {"ref": f"refs/heads/{branch}", "sha": commit["sha"]},
            )
        return branch

    def delete_stale_fork_branches(self, open_prs: set[int]) -> None:
        """Delete ``heavy-lanes/fork-<n>`` branches whose PR is no longer open."""
        for ref in self.get(f"git/matching-refs/heads/{FORK_BRANCH_PREFIX}"):
            suffix = ref["ref"].removeprefix(f"refs/heads/{FORK_BRANCH_PREFIX}")
            if suffix.isdigit() and int(suffix) not in open_prs:
                self._request("DELETE", f"{API}/repos/{self.repo}/git/{ref['ref']}")

    def set_status(self, sha: str, state: str, description: str) -> None:
        """Post the ``heavy-lanes`` commit status on ``sha``."""
        self._request(
            "POST",
            f"{API}/repos/{self.repo}/statuses/{sha}",
            {
                "state": state,
                "context": STATUS_CONTEXT,
                "description": description[:STATUS_DESCRIPTION_MAX],
                "target_url": f"https://github.com/{self.repo}/actions/workflows/{LANES_WORKFLOW}",
            },
        )


def _describe(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"GitHub API {exc.code} — {exc.reason}"
    return str(exc)


def _report_waiting(gh: GitHub, pr: PullRequestState, verdict: Verdict) -> None:
    """Show on the PR why the lanes have not started, or backfill a finished run.

    A run dispatched before the lanes reported a commit status left none behind;
    its conclusion is copied over so that PR stops reading as "waiting".
    """
    status = own_status(pr)
    if lanes_started(pr):
        finished = [
            r.conclusion
            for r in pr.lane_runs
            if r.status == "completed" and r.conclusion and r.conclusion != "cancelled"
        ]
        if status is None and finished:
            state = "success" if finished[0] == "success" else "failure"
            _try_status(gh, pr, state, f"{finished[0].capitalize()} (backfilled from the run)")
        return
    description = WAITING + "; ".join(verdict.reasons)
    if status is None or status.description != description[:STATUS_DESCRIPTION_MAX]:
        _try_status(gh, pr, "pending", description)


def _try_status(gh: GitHub, pr: PullRequestState, state: str, description: str) -> None:
    try:
        gh.set_status(pr.head_sha, state, description)
    except urllib.error.URLError as exc:
        print(f"::warning::PR #{pr.number}: status not posted — {_describe(exc)}")


def main(argv: list[str] | None = None) -> int:
    """CLI: evaluate open PRs (or ``--pr``) and dispatch the heavy lanes where ready."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--pr", type=int, action="append", default=[], help="only these PRs")
    parser.add_argument(
        "--require-check",
        action="append",
        default=None,
        help=f"check that must have reported green (default: {', '.join(DEFAULT_REQUIRED_CHECKS)})",
    )
    parser.add_argument("--dry-run", action="store_true", help="report, do not dispatch")
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("::error::GITHUB_TOKEN is not set", file=sys.stderr)
        return 2
    gh = GitHub(args.repo, token)
    required = args.require_check or list(DEFAULT_REQUIRED_CHECKS)
    prs = [gh.get(f"pulls/{n}") for n in args.pr] if args.pr else gh.paginate("pulls?state=open")

    # One PR's API trouble is a warning, never a failed sweep: a red
    # heavy-lanes-trigger check would land on whichever PR's workflow_run
    # started this sweep, and the next sweep (≤10 min) retries anyway.
    for raw in prs:
        if raw["state"] != "open":
            continue
        # Drafts are rejected by `readiness` anyway; skipping them before
        # `fetch` saves its 4 REST + 1 GraphQL calls per PR, per sweep. A PR
        # whose fork was deleted has no head repo and nothing to test.
        if raw["draft"] or not raw["head"]["repo"]:
            print(f"PR #{raw['number']}: waiting — draft PR, or its fork was deleted")
            continue
        try:
            pr = gh.fetch(raw)
        except (urllib.error.URLError, GitHubAPIError) as exc:
            print(f"::warning::PR #{raw['number']}: {_describe(exc)}")
            continue
        verdict = readiness(pr, required)
        if not verdict.ready:
            print(f"PR #{pr.number} @ {pr.head_sha[:7]}: waiting — " + "; ".join(verdict.reasons))
            if not args.dry_run:
                _report_waiting(gh, pr, verdict)
            continue
        if args.dry_run:
            print(f"PR #{pr.number} @ {pr.head_sha[:7]}: ready (dry run, not dispatched)")
            continue
        # workflow_dispatch runs the branch TIP, so a push since `fetch` would
        # start the lanes on a commit nobody evaluated. The next sweep handles it.
        try:
            current_sha = gh.get(f"pulls/{pr.number}")["head"]["sha"]
        except urllib.error.URLError as exc:
            print(f"::warning::PR #{pr.number}: {_describe(exc)}")
            continue
        if current_sha != pr.head_sha:
            print(f"PR #{pr.number}: head moved since evaluation — deferring to the next sweep")
            continue
        try:
            gh.dispatch(pr)
            gh.set_status(pr.head_sha, "pending", f"{RUNNING} — dispatched by heavy-lanes-trigger")
        except urllib.error.URLError as exc:
            # e.g. 422 when the branch has no heavy-lanes.yml — unreachable for
            # a PR 0 behind a base that has it, but one PR must not end the sweep.
            print(f"::warning::PR #{pr.number}: dispatch failed — {_describe(exc)}")
            continue
        print(f"::notice::PR #{pr.number} @ {pr.head_sha[:7]}: ready — heavy lanes dispatched")
    if not args.pr and not args.dry_run:
        try:
            gh.delete_stale_fork_branches({raw["number"] for raw in prs if raw["state"] == "open"})
        except urllib.error.URLError as exc:
            print(f"::warning::fork branch cleanup: {_describe(exc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
