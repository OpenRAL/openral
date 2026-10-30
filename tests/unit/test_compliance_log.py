"""The on-robot compliance log (issue #469): chain, tamper evidence, retention, ship, CLI.

Run with:
    uv run pytest tests/unit/test_compliance_log.py -v
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tarfile
from pathlib import Path

import pytest
from openral_cli.main import app
from openral_core import (
    Actor,
    InterventionRecord,
    RetentionClass,
    ROSRuntimeError,
    SafetyStopRecord,
    SessionRecord,
    SoftwareInventoryRecord,
)
from openral_observability.compliance import ComplianceEmitter, software_inventory
from openral_observability.compliance_log import (
    ComplianceLog,
    delete,
    export,
    hold,
    prunable,
    records,
    ship,
    verify,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from typer.testing import CliRunner

from tests.unit.fakes.fake_otlp_exporter import FlakyCollector, UnreachableCollector

_DAY = 86_400 * 10**9


def _emitter(
    root: Path, writer: str = "recorder"
) -> tuple[ComplianceEmitter, InMemoryLogRecordExporter]:
    exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]  # reason: OTel SDK leaves __init__ unannotated
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    log = ComplianceLog(root, writer=writer)
    return ComplianceEmitter(clock_source="ntp", logger_provider=provider, log=log), exporter


def _fill(root: Path) -> ComplianceEmitter:
    """A session, two stops over two days, a reset: three classes, one stream."""
    emitter, _ = _emitter(root)
    t0 = 1_761_700_000 * 10**9
    emitter.emit(SessionRecord, ts_ns=t0, start_ts_ns=t0, run_mode="hardware")
    emitter.emit(SafetyStopRecord, ts_ns=t0 + 1, trigger_source="kernel", latched=True)
    emitter.emit(SafetyStopRecord, ts_ns=t0 + _DAY, trigger_source="human", latched=True)
    emitter.emit(
        InterventionRecord,
        ts_ns=t0 + _DAY + 1,
        intervention_kind="estop_reset",
        actor=Actor(channel_label="pendant"),
        channel="pendant",
        legitimate=None,
    )
    return emitter


def _segment(root: Path, cls: RetentionClass) -> Path:
    return sorted((root / "recorder" / cls.value).glob("*.jsonl"))[0]


def test_an_intact_log_verifies_and_reads_back(tmp_path: Path) -> None:
    _fill(tmp_path)
    assert verify(tmp_path) == []
    stops = list(records(tmp_path, RetentionClass.MR_SAFETY_DECISION_1Y))
    assert [s.trigger_source for s in stops] == ["kernel", "human"]  # type: ignore[union-attr]
    assert len(list((tmp_path / "recorder" / "mr_safety_decision_1y").glob("*.jsonl"))) == 2


def test_an_edited_line_breaks_the_chain(tmp_path: Path) -> None:
    _fill(tmp_path)
    seg = _segment(tmp_path, RetentionClass.MR_SAFETY_DECISION_1Y)
    seg.write_bytes(seg.read_bytes().replace(b'"kernel"', b'"human"'))
    problems = verify(tmp_path)
    assert any("chain broken" in p for p in problems), problems


def test_a_deleted_line_shows_as_a_broken_chain_and_a_seq_gap(tmp_path: Path) -> None:
    _fill(tmp_path)
    seg = _segment(tmp_path, RetentionClass.MR_SAFETY_DECISION_1Y)
    seg.write_bytes(b"")  # the first day's stop is gone
    problems = verify(tmp_path)
    assert any("chain broken" in p for p in problems), problems
    assert any("seq [1] missing" in p for p in problems), problems


def test_a_torn_tail_is_quarantined_and_the_chain_continues(tmp_path: Path) -> None:
    _fill(tmp_path)
    seg = sorted((tmp_path / "recorder" / "mr_trace_5y").glob("*.jsonl"))[-1]
    with seg.open("ab") as f:
        f.write(b'{"prev_hash": "abc", "rec')  # power cut mid-write
    emitter, _ = _emitter(tmp_path)  # reopen
    emitter.emit(
        InterventionRecord,
        ts_ns=1_761_700_000 * 10**9 + _DAY + 2,
        intervention_kind="config_change",
        actor=Actor(channel_label="dashboard"),
        channel="dashboard",
        legitimate=True,
    )
    assert seg.with_suffix(".quarantine").read_bytes() == b'{"prev_hash": "abc", "rec'
    problems = verify(tmp_path)
    assert [p for p in problems if "chain broken" in p] == []
    assert any("quarantined" in p for p in problems)


def test_two_writers_keep_separate_chains(tmp_path: Path) -> None:
    _fill(tmp_path)
    runtime, _ = _emitter(tmp_path, writer="runtime")
    runtime.emit(SafetyStopRecord, trigger_source="supervisor", latched=False)
    assert verify(tmp_path) == []
    with pytest.raises(ValueError, match="writer"):
        ComplianceLog(tmp_path, writer="../escape")


def test_retention_keeps_floors_and_holds(tmp_path: Path) -> None:
    _fill(tmp_path)
    first_day = dt.date.fromisoformat(_segment(tmp_path, RetentionClass.AI_LOG_6M).stem)
    assert prunable(tmp_path, first_day + dt.timedelta(days=184)) == []
    after_6m = prunable(tmp_path, first_day + dt.timedelta(days=186))
    assert {p.parent.name for p in after_6m} == {"ai_log_6m"}
    extended = prunable(
        tmp_path, first_day + dt.timedelta(days=186), {RetentionClass.AI_LOG_6M: 400}
    )
    assert extended == []
    shortened = prunable(
        tmp_path, first_day + dt.timedelta(days=186), {RetentionClass.MR_TRACE_5Y: 1}
    )
    assert {p.parent.name for p in shortened} == {"ai_log_6m"}  # a floor never shrinks
    hold(tmp_path, on=True)
    assert prunable(tmp_path, first_day + dt.timedelta(days=4000)) == []
    hold(tmp_path, on=False)
    delete(after_6m)
    assert verify(tmp_path) == []  # pruning removes records by design; no gap reported


def test_an_exported_bundle_verifies_on_its_own(tmp_path: Path) -> None:
    _fill(tmp_path / "log")
    out = tmp_path / "bundle.tar.gz"
    n = export(tmp_path / "log", out, classes=[RetentionClass.MR_TRACE_5Y], since=None)
    assert n == 1
    with tarfile.open(out) as tar:
        tar.extractall(tmp_path / "authority", filter="data")
    assert verify(tmp_path / "authority", partial=True) == []
    assert verify(tmp_path / "authority") != []  # other classes of the stream are absent


def test_ship_resumes_after_an_outage_and_never_resends(tmp_path: Path) -> None:
    _fill(tmp_path)
    attrs = {"openral.unit.serial": "SO101-0042"}
    first = FlakyCollector(ok=2)
    assert ship(tmp_path, first, resource_attributes=attrs) == 2
    second = FlakyCollector(ok=100)
    assert ship(tmp_path, second, resource_attributes=attrs) == 2
    assert ship(tmp_path, FlakyCollector(ok=100), resource_attributes=attrs) == 0
    ids = {json.loads(b)["record_id"] for b in first.got + second.got}
    assert len(ids) == 4


def test_a_failed_log_write_raises_after_the_record_went_out(tmp_path: Path) -> None:
    emitter, exporter = _emitter(tmp_path)
    os.chmod(tmp_path / "recorder", 0o500)
    try:
        if os.access(tmp_path / "recorder", os.W_OK):
            pytest.skip("running as root: a read-only directory is still writable")
        with pytest.raises(ROSRuntimeError, match="compliance log write failed"):
            emitter.emit(SafetyStopRecord, trigger_source="kernel", latched=True)
    finally:
        os.chmod(tmp_path / "recorder", 0o700)
    assert len(exporter.get_finished_logs()) == 1  # still sent over OTLP


def test_inventory_names_the_installed_openral_software() -> None:
    names = {c.name for c in software_inventory()}
    assert "openral-observability" in names
    assert SoftwareInventoryRecord(
        record_id="0" * 32, seq=0, ts_ns=0, clock_source="ntp", components=software_inventory()
    )


def test_cli_verify_hold_export(tmp_path: Path) -> None:
    _fill(tmp_path)
    runner = CliRunner()
    root = ["--root", str(tmp_path)]
    assert runner.invoke(app, ["compliance", "verify", *root]).exit_code == 0
    assert runner.invoke(app, ["compliance", "hold", "true", *root]).exit_code == 0
    assert (tmp_path / "hold").exists()
    out = tmp_path / "b.tar.gz"
    result = runner.invoke(app, ["compliance", "export", "--out", str(out), *root])
    assert result.exit_code == 0 and out.exists()
    _segment(tmp_path, RetentionClass.MR_SAFETY_DECISION_1Y).write_bytes(b"")
    assert runner.invoke(app, ["compliance", "verify", *root]).exit_code == 1


def test_an_earlier_timestamp_never_goes_to_an_older_segment(tmp_path: Path) -> None:
    """Written after midnight with a stamp from before (or a clock step back): still intact."""
    emitter, _ = _emitter(tmp_path)
    t0 = 1_761_700_000 * 10**9
    emitter.emit(SafetyStopRecord, ts_ns=t0 + _DAY, trigger_source="kernel", latched=True)
    emitter.emit(SafetyStopRecord, ts_ns=t0, trigger_source="human", latched=True)
    assert verify(tmp_path) == []
    assert len(list((tmp_path / "recorder" / "mr_safety_decision_1y").glob("*.jsonl"))) == 1


def test_a_short_write_leaves_no_partial_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENOSPC half-way through a line: the fragment is truncated off, the chain stays whole."""
    emitter, _ = _emitter(tmp_path)
    emitter.emit(SafetyStopRecord, trigger_source="kernel", latched=True)
    real_write = os.write

    def half_then_full(fd: int, data: bytes) -> int:
        real_write(fd, bytes(data[: len(data) // 2]))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("os.write", half_then_full)
    with pytest.raises(ROSRuntimeError):
        emitter.emit(SafetyStopRecord, trigger_source="human", latched=True)
    monkeypatch.setattr("os.write", real_write)
    emitter.emit(SafetyStopRecord, trigger_source="deadman", latched=True)
    problems = verify(tmp_path)
    assert [p for p in problems if "chain" in p or "not a compliance" in p] == []
    assert problems == [p for p in problems if "seq [1] missing" in p]  # the lost one shows


def test_ship_to_an_unreachable_collector_sends_nothing_and_raises_nothing(
    tmp_path: Path,
) -> None:
    _fill(tmp_path)
    attrs = {"openral.unit.serial": "SO101-0042"}
    assert ship(tmp_path, UnreachableCollector(), resource_attributes=attrs) == 0
    assert ship(tmp_path, FlakyCollector(ok=100), resource_attributes=attrs) == 4


def test_on_log_error_keeps_the_record(tmp_path: Path) -> None:
    """With a callback, a failed local write is reported and the record still returned."""
    exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]  # reason: OTel SDK leaves __init__ unannotated
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    errors: list[str] = []
    emitter = ComplianceEmitter(
        logger_provider=provider,
        log=ComplianceLog(tmp_path, writer="recorder"),
        on_log_error=lambda record, error: errors.append(record.kind),
    )
    os.chmod(tmp_path / "recorder", 0o500)
    try:
        if os.access(tmp_path / "recorder", os.W_OK):
            pytest.skip("running as root: a read-only directory is still writable")
        record = emitter.emit(SafetyStopRecord, trigger_source="kernel", latched=True)
    finally:
        os.chmod(tmp_path / "recorder", 0o700)
    assert record.trigger_source == "kernel" and errors == ["safety_stop"]


def test_an_unreadable_safety_binary_is_listed_without_a_digest(tmp_path: Path) -> None:
    from openral_observability.compliance import _digest

    binary = tmp_path / "safety_kernel_node"
    binary.write_bytes(b"\x7fELF")
    binary.chmod(0o000)
    try:
        if os.access(binary, os.R_OK):
            pytest.skip("running as root: an unreadable file is still readable")
        assert _digest(binary) is None
    finally:
        binary.chmod(0o600)
