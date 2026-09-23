"""Start the opt-in heavy lanes on a PR once it is ready for them — no manual trigger.

``heavy-lanes`` is a required check that only ``.github/workflows/heavy-lanes.yml``
posts, and that workflow runs only via ``workflow_dispatch``. So on every PR
the check sits at "Expected — waiting" until this tool dispatches it, which it
does for an open, non-draft, same-repo PR whose current head commit meets all
of these:

* **up to date with its base** — the compare API reports the head 0 commits
  behind the base branch (rebased, or base merged in);
* **every other check is green** — every check run on the head is completed
  as success / skipped / neutral, every commit status is ``success``, and the
  ``--require-check`` names (the other required checks) are present, so a
  workflow that has not reported yet does not read as green;
* **every review thread is resolved**.

It never dispatches twice for one head commit: a heavy-lanes run already
queued, running, or finished (other than ``cancelled``) for that SHA is left
alone. A failed lane is re-run from the Actions page, not re-dispatched.

Fork PRs are reported and skipped: ``workflow_dispatch`` can only target a
branch of this repository, so nothing can post ``heavy-lanes`` on a fork head.
A maintainer runs the lanes for a fork PR by pushing its head to a branch of
this repository and dispatching ``heavy-lanes.yml`` there (the check lands on
the same commit SHA), or merges through the ruleset bypass.

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
DEFAULT_REQUIRED_CHECKS = ("select-and-test", "quality", "Verify Signed-off-by")


class GitHubAPIError(RuntimeError):
    """A GitHub response that is not usable: a GraphQL ``errors`` payload or a null result."""


class CheckRunState(BaseModel):
    """One check run on the PR head: ``name``, ``status``, ``conclusion``."""

    name: str
    status: str
    conclusion: str | None = None


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
    statuses: list[str]
    unresolved_threads: int
    lane_runs: list[LaneRunState]


class Verdict(BaseModel):
    """``ready`` to dispatch, else the list of ``reasons`` why not."""

    ready: bool
    reasons: list[str]


def _is_own_check(name: str) -> bool:
    return name in OWN_CHECK_NAMES or name.startswith(OWN_CHECK_PREFIXES)


def readiness(pr: PullRequestState, required_checks: list[str]) -> Verdict:
    """Decide whether the heavy lanes should start on ``pr``'s head now.

    Pure: every fact comes from ``pr``. An empty ``reasons`` list means ready.
    """
    reasons: list[str] = []
    if pr.draft:
        reasons.append("draft PR")
    if not pr.same_repo:
        reasons.append(
            "fork PR — workflow_dispatch cannot target a fork branch; push its head to a "
            "branch of this repo and dispatch heavy-lanes.yml there"
        )
    if any(r.status != "completed" or r.conclusion != "cancelled" for r in pr.lane_runs):
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
    for state in pr.statuses:
        if state != "success":
            reasons.append(f"a commit status is {state}")
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
        runs = self.get(f"actions/workflows/{LANES_WORKFLOW}/runs?head_sha={sha}")
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
            statuses=[s["state"] for s in combined["statuses"]],
            unresolved_threads=self.unresolved_threads(pr["number"]),
            lane_runs=[LaneRunState.model_validate(r) for r in runs["workflow_runs"]],
        )

    def dispatch(self, pr: PullRequestState) -> None:
        """Start ``heavy-lanes.yml`` on the PR's branch via ``workflow_dispatch``."""
        self._request(
            "POST",
            f"{API}/repos/{self.repo}/actions/workflows/{LANES_WORKFLOW}/dispatches",
            {"ref": pr.head_ref, "inputs": {"base": f"origin/{pr.base_ref}", "head": "HEAD"}},
        )


def _describe(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"GitHub API {exc.code} — {exc.reason}"
    return str(exc)


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
        # Drafts and forks are rejected by `readiness` anyway; skipping them
        # before `fetch` saves its 4 REST + 1 GraphQL calls per PR, per sweep.
        head_repo = raw["head"]["repo"]
        if raw["draft"] or not head_repo or head_repo["full_name"] != args.repo:
            print(f"PR #{raw['number']}: waiting — draft or fork PR")
            continue
        try:
            pr = gh.fetch(raw)
        except (urllib.error.URLError, GitHubAPIError) as exc:
            print(f"::warning::PR #{raw['number']}: {_describe(exc)}")
            continue
        verdict = readiness(pr, required)
        if not verdict.ready:
            print(f"PR #{pr.number} @ {pr.head_sha[:7]}: waiting — " + "; ".join(verdict.reasons))
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
        except urllib.error.URLError as exc:
            # e.g. 422 when the branch has no heavy-lanes.yml — unreachable for
            # a PR 0 behind a base that has it, but one PR must not end the sweep.
            print(f"::warning::PR #{pr.number}: dispatch failed — {_describe(exc)}")
            continue
        print(f"::notice::PR #{pr.number} @ {pr.head_sha[:7]}: ready — heavy lanes dispatched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
