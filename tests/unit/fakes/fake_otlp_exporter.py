"""Fake OTLP collectors for the compliance-log ``ship`` tests (a network boundary)."""

from __future__ import annotations

from collections.abc import Sequence

from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult


class FlakyCollector(LogRecordExporter):
    """Accepts ``ok`` records, then refuses: a collector going away mid-ship."""

    def __init__(self, ok: int) -> None:
        self.ok = ok
        self.got: list[str] = []

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        if len(self.got) >= self.ok:
            return LogRecordExportResult.FAILURE
        self.got += [str(r.log_record.body) for r in batch]
        return LogRecordExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


class UnreachableCollector(LogRecordExporter):
    """Raises on every export, as an exporter does when the host does not resolve."""

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        raise ConnectionError("collector unreachable")

    def shutdown(self) -> None:
        return None
