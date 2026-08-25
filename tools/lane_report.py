"""Opt-in lane accounting — decide, record and attest what a lane actually ran.

Issue #163. The `test-selective` job used to report a lane's outcome by
grepping pytest's terse summary for ``N passed`` / ``N skipped``. Two silent
degradations lived in that scheme:

1. A *blast-radius* diff expanded to zero lanes, so the step exited 0 having
   executed nothing — indistinguishable from a run that executed every lane and
   passed. (Fixed in ``select_tests.py``; this tool makes the difference
   *visible* and *enforced*.)
2. "Any skip fails the lane" is the right rule for anything a runner can
   provide and the wrong rule for anything it cannot. A hosted ubuntu-24.04
   runner has no CUDA GPU, no Vulkan ICD and no proprietary sidecars, so
   CUDA/Vulkan/sidecar-gated tests skip forever and redden every PR that
   selects them — while a genuinely fixable gap ("panda.srdf not installed")
   looked exactly the same.

The policy implemented here, driven by ``[capability_gaps]`` in
``tools/test_selection.toml``:

* A skip explained by a DECLARED capability gap is *declared-not-run*: allowed,
  attributed to the gap, and counted in the ledger. It is never called
  "skipped" and never silently absent.
* Any other skip still FAILS the lane. Matching is fail-closed — reword a skip
  reason out of the declared patterns and the lane goes red, not green.
* A lane whose every test is explained by a declared gap is *declared-not-run*
  — reported, never silent. A lane that collected no tests at all still fails.
* ``attest`` cross-checks the ledger against the selector's own output: every
  selected lane must have produced a record. "Selected but never executed" is
  the exact shape of #163 and is now a hard failure.

Run::

    python tools/lane_report.py lane --lane sim --junit-xml a.xml --junit-xml b.xml
    python tools/lane_report.py attest --selection-json sel.json --summary "$GITHUB_STEP_SUMMARY"

Example:
    >>> _classify_skip("SmolVLA LIBERO sim test requires CUDA", _GAPS())
    'cuda'
    >>> _classify_skip("mujoco not installed", _GAPS()) is None
    True
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
# Import the selector's config schema rather than restating it (CLAUDE.md
# §1.13). Run as `python tools/lane_report.py`, sys.path[0] is `tools/`, so the
# repo root has to go on the path for the `tools.` package form — which is also
# the module name mypy resolves, keeping the real types instead of `Any`.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.select_tests import CapabilityGap, SelectionConfig, load_config  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "tools" / "test_selection.toml"
DEFAULT_LEDGER = REPO_ROOT / ".lane-ledger.jsonl"

# Status vocabulary. Deliberately NOT "skipped": a lane is either coverage we
# got, coverage we declared we cannot get here, or a failure.
STATUS_RAN = "ran"
STATUS_DECLARED_NOT_RUN = "declared-not-run"
STATUS_FAILED = "failed"


def _GAPS() -> dict[str, CapabilityGap]:  # noqa: N802  # reason: docstring-example helper
    """The declared capability gaps (used by this module's doctest)."""
    return load_config(DEFAULT_CONFIG).capability_gaps


class LaneRecord(BaseModel):
    """One lane's accounted outcome — the unit of the ledger."""

    lane: str
    status: str
    passed: int = 0
    failed: int = 0
    declared_skips: dict[str, int] = Field(default_factory=dict)
    undeclared_skips: list[str] = Field(default_factory=list)
    note: str | None = None

    @property
    def declared_total(self) -> int:
        """How many tests were not run because of a declared capability gap."""
        return sum(self.declared_skips.values())


def _classify_skip(reason: str, gaps: dict[str, CapabilityGap]) -> str | None:
    """Return the capability gap explaining ``reason``, or ``None`` if undeclared."""
    lowered = reason.lower()
    for name, gap in gaps.items():
        if any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in gap.skip_patterns):
            return name
    return None


def _skip_reason(skipped: ElementTree.Element) -> str:
    """Extract a skip's human reason from a junit ``<skipped>`` element.

    pytest writes the reason two different ways and BOTH must be read: a normal
    ``pytest.skip`` puts it in ``message=``, while a *collection* skip (a
    module-level ``importorskip``) sets ``message="collection skipped"`` and
    puts the real reason in the element text. Reading only ``message`` would
    classify every module-level skip as undeclared — the sort of plausible
    wrong answer this whole file exists to prevent.
    """
    message = (skipped.get("message") or "").strip()
    text = (skipped.text or "").strip()
    if message and message.lower() != "collection skipped":
        return message
    return text or message


def read_junit(paths: list[Path]) -> tuple[int, int, list[str]]:
    """Aggregate junit XML files into ``(passed, failed, skip_reasons)``.

    A lane may emit several reports — one batched run plus one per
    ``isolate_globs`` file — and they are summed into a single lane verdict.
    Missing files are ignored: pytest writes no report when it never starts.
    """
    passed = 0
    failed = 0
    reasons: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        root = ElementTree.parse(path).getroot()
        for case in root.iter("testcase"):
            skipped = case.find("skipped")
            if skipped is not None:
                reasons.append(_skip_reason(skipped))
            elif case.find("failure") is not None or case.find("error") is not None:
                failed += 1
            else:
                passed += 1
    return passed, failed, reasons


def build_record(  # noqa: PLR0911  # reason: one explicit early return per lane verdict
    lane: str,
    *,
    passed: int,
    failed: int,
    skip_reasons: list[str],
    config: SelectionConfig,
    exit_code: int | None = None,
    from_exit_code: bool = False,
) -> LaneRecord:
    """Apply the lane policy to one lane's raw counts (see module docstring).

    ``from_exit_code`` marks a lane with no junit evidence — ``simpler-env`` and
    ``robocasa-gr1`` run their pytest inside ``verify_test_envs.py``, so the
    exit code is all there is. Such a lane is judged solely on that code, and
    the ledger says so rather than implying per-test accounting it never had.
    """
    gaps = config.capability_gaps

    declared: Counter[str] = Counter()
    undeclared: list[str] = []
    for reason in skip_reasons:
        name = _classify_skip(reason, gaps)
        if name is None:
            undeclared.append(reason)
        else:
            declared[name] += 1

    record = LaneRecord(
        lane=lane,
        status=STATUS_RAN,
        passed=passed,
        failed=failed,
        declared_skips=dict(sorted(declared.items())),
        undeclared_skips=sorted(set(undeclared)),
    )

    if failed:
        record.status = STATUS_FAILED
        record.note = f"{failed} failing test(s)"
        return record
    if from_exit_code:
        # No per-test evidence exists for this lane; the exit code is the whole
        # verdict. Say that explicitly instead of applying rules to counts we
        # never measured.
        if exit_code:
            record.status = STATUS_FAILED
            record.note = f"verify_test_envs.py exited {exit_code}"
        else:
            record.note = "accounted by exit code only (verify_test_envs.py drives its own pytest)"
        return record
    if exit_code is not None and exit_code != 0 and passed == 0 and not skip_reasons:
        # The lane never got as far as running tests (e.g. a provisioning step
        # died). That is a failure, not an absence — do not let it look like a
        # lane that simply had nothing to do.
        record.status = STATUS_FAILED
        record.note = f"lane exited {exit_code} without running any test"
        return record
    if undeclared:
        record.status = STATUS_FAILED
        record.note = (
            f"{len(undeclared)} undeclared skip(s): "
            f"{'; '.join(record.undeclared_skips[:3])}"
            " — provision the missing env, or declare the capability in "
            "[capability_gaps] if the runner genuinely cannot provide it"
        )
        return record
    if passed == 0:
        if not skip_reasons:
            # Nothing passed and nothing skipped: the lane collected no tests at
            # all. That is a broken lane, not an absence of coverage.
            record.status = STATUS_FAILED
            record.note = "lane collected no tests at all"
            return record
        # Every test the lane ran is accounted for by a DECLARED gap — the
        # undeclared branch above has already returned. Nothing is hidden: the
        # gap is named and counted here and printed by the attest step.
        #
        # Judged on what the diff SELECTED, not on the lane's full potential. A
        # narrow diff can select a single fully-gated file out of an otherwise
        # well-covered lane: `sim` yields 162 passing tests when all seven of
        # its files are selected, but a diff touching only `rskills/act-aloha/**`
        # selects just `test_aloha_bimanual_act_aloha.py`, whose six tests are
        # all CUDA-gated (proof run 32815008771). Failing that would punish a PR
        # for touching a GPU-only file — the exact class of breakage this policy
        # exists to remove.
        record.status = STATUS_DECLARED_NOT_RUN
        record.note = "every selected test is gated by a declared capability gap"
        return record
    return record


def _describe(record: LaneRecord, config: SelectionConfig) -> str:
    """One-line human summary of a lane record."""
    bits = [f"{record.passed} passed"]
    if record.failed:
        bits.append(f"{record.failed} failed")
    for name, count in record.declared_skips.items():
        gap = config.capability_gaps.get(name)
        summary = gap.summary if gap else name
        bits.append(f"{count} not run (needs {summary})")
    if record.undeclared_skips:
        bits.append(f"{len(record.undeclared_skips)} UNDECLARED skip(s)")
    return ", ".join(bits)


def _append_ledger(ledger: Path, record: LaneRecord) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")


def _read_ledger(ledger: Path) -> list[LaneRecord]:
    if not ledger.is_file():
        return []
    return [
        LaneRecord.model_validate_json(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _render_summary(
    records: list[LaneRecord],
    selected: list[str],
    config: SelectionConfig,
    problems: list[str],
) -> str:
    """Markdown lane ledger for the GitHub job summary."""
    lines = ["## Opt-in dependency lane ledger", ""]
    if not selected:
        lines.append("No opt-in lane was selected by this diff — nothing to run.")
        return "\n".join(lines) + "\n"

    by_lane = {r.lane: r for r in records}
    lines.append("| Lane | Status | Detail |")
    lines.append("| --- | --- | --- |")
    for lane in selected:
        record = by_lane.get(lane)
        if record is None:
            lines.append(f"| `{lane}` | **NO RECORD** | selected but never executed |")
            continue
        lines.append(f"| `{lane}` | {record.status} | {_describe(record, config)} |")

    declared = sorted({name for r in records for name in r.declared_skips})
    if declared:
        lines.extend(["", "### Declared-not-run — coverage this runner cannot provide", ""])
        for name in declared:
            gap = config.capability_gaps.get(name)
            if gap is None:
                continue
            total = sum(r.declared_skips.get(name, 0) for r in records)
            lines.append(
                f"- **{name}** ({total} test(s)) — needs {gap.summary}. {gap.satisfied_by}"
            )
    if problems:
        lines.extend(["", "### Problems", ""])
        lines.extend(f"- {p}" for p in problems)
    return "\n".join(lines) + "\n"


def _cmd_lane(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    passed, failed, reasons = read_junit([Path(p) for p in args.junit_xml])
    record = build_record(
        args.lane,
        passed=passed,
        failed=failed,
        skip_reasons=reasons,
        config=config,
        exit_code=args.exit_code,
        from_exit_code=args.from_exit_code,
    )
    _append_ledger(Path(args.ledger), record)
    detail = _describe(record, config)
    if record.status == STATUS_FAILED:
        print(f"::error::opt-in lane {record.lane}: {record.note} ({detail})")
        return 1
    if record.status == STATUS_DECLARED_NOT_RUN:
        print(f"::notice::opt-in lane {record.lane} DECLARED-NOT-RUN — {record.note} ({detail})")
    else:
        print(f"opt-in lane {record.lane}: {detail}")
    return 0


def _cmd_attest(args: argparse.Namespace) -> int:
    """Cross-check the ledger against the selection. This is the anti-vacuity gate."""
    config = load_config(Path(args.config))
    selection = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
    full_run = str(selection.get("full_run", "false")).lower() == "true"
    selected = sorted(json.loads(selection.get("requirement_targets_json") or "{}"))
    records = _read_ledger(Path(args.ledger))

    problems: list[str] = []
    # Guard the #163 regression directly: a blast-radius diff that expands to no
    # lane at all means the selector stopped emitting them again.
    if full_run and not selected:
        problems.append(
            "full_run diff selected ZERO opt-in lanes — select_tests.py must expand "
            "every requirement glob on a full run (issue #163)"
        )
    recorded = {r.lane for r in records}
    for lane in selected:
        if lane not in recorded:
            problems.append(f"lane `{lane}` was selected but produced no ledger record")
    for record in records:
        if record.status == STATUS_FAILED:
            problems.append(f"lane `{record.lane}` failed: {record.note}")

    summary = _render_summary(records, selected, config, problems)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(summary)
    print(summary)
    for problem in problems:
        print(f"::error::{problem}")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    sub = parser.add_subparsers(dest="command", required=True)

    lane = sub.add_parser("lane", help="account for one lane's junit report(s)")
    lane.add_argument("--lane", required=True)
    lane.add_argument("--junit-xml", action="append", default=[])
    lane.add_argument("--exit-code", type=int, default=None)
    lane.add_argument(
        "--from-exit-code",
        action="store_true",
        help="lane produces no junit report; judge it by --exit-code alone",
    )
    lane.set_defaults(func=_cmd_lane)

    attest = sub.add_parser("attest", help="cross-check the ledger against the selection")
    attest.add_argument("--selection-json", required=True)
    attest.add_argument("--summary", default=None)
    attest.set_defaults(func=_cmd_attest)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
