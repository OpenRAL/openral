"""The on-robot compliance log: append-only, hash-chained JSONL (issue #469).

Every ``ComplianceRecord`` the robot emits is also written here, so the machine keeps its own
evidence with no network: EU Machinery Regulation 2023/1230 Annex III §1.2.1(f) wants the
intervention / safety-software trace kept for five years, producible on a competent
authority's reasoned request. Layout::

    <root>/<writer>/<retention_class>/<YYYY-MM-DD>.jsonl       one line per record
    <root>/<writer>/<retention_class>/<YYYY-MM-DD>.quarantine  a torn tail found at open
    <root>/<writer>/<retention_class>/pruned                   retention has deleted segments
    <root>/hold                                                present → nothing is deleted

One ``<writer>`` per process that records (``recorder``, ``runtime``), so two processes never
fork one chain. Each line is ``{"prev_hash": <sha256 of the previous line>, "record": {...}}``;
the chain runs across days within a writer's class, so deleting, editing or reordering a line
breaks it (``verify``). The first line chains to ``GENESIS``. Stdlib only.

Example:
    >>> import tempfile
    >>> from openral_core import SessionRecord
    >>> log = ComplianceLog(tempfile.mkdtemp(), writer="recorder")
    >>> log.append(
    ...     SessionRecord(
    ...         record_id="0" * 32,
    ...         seq=0,
    ...         ts_ns=0,
    ...         clock_source="ntp",
    ...         start_ts_ns=0,
    ...         run_mode="sim",
    ...     )
    ... )
    >>> verify(log.root)
    []
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import tarfile
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

from openral_core import RETENTION_FLOOR_DAYS, ComplianceRecord, RetentionClass
from opentelemetry.sdk._logs.export import LogRecordExporter
from pydantic import TypeAdapter, ValidationError

__all__ = [
    "COMPLIANCE_DIR_ENV",
    "GENESIS",
    "ComplianceLog",
    "default_compliance_dir",
    "delete",
    "export",
    "hold",
    "prunable",
    "records",
    "ship",
    "verify",
]

COMPLIANCE_DIR_ENV = "OPENRAL_COMPLIANCE_DIR"
"""Where the log lives; setting it turns the log on (a hardware deploy always sets it)."""

GENESIS = "0" * 64
"""``prev_hash`` of the first line of a writer's class."""

RAW_TELEMETRY_DAYS = 90
"""``telemetry_raw`` has no floor; it is kept this long by default."""

_FSYNC_CLASSES = frozenset(
    {
        RetentionClass.MR_SAFETY_DECISION_1Y,
        RetentionClass.MR_TRACE_5Y,
        RetentionClass.INCIDENT_10Y,
    }
)
"""Classes whose every record is fsync'd: a power cut must not lose a stop or a reset."""

_WRITER = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_ADAPTER: TypeAdapter[ComplianceRecord] = TypeAdapter(ComplianceRecord)


def default_compliance_dir() -> Path:
    """``$OPENRAL_COMPLIANCE_DIR``, else ``$XDG_STATE_HOME/openral/compliance``.

    Example:
        >>> default_compliance_dir().name
        'compliance'
    """
    if env := os.environ.get(COMPLIANCE_DIR_ENV):
        return Path(env)
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "openral" / "compliance"


def _line_hash(line: bytes) -> str:
    return hashlib.sha256(line).hexdigest()


def _segments(class_dir: Path) -> list[Path]:
    return sorted(class_dir.glob("*.jsonl"))


def _class_dirs(root: Path) -> Iterator[tuple[RetentionClass, Path]]:
    """Every ``(class, <root>/<writer>/<class>)`` that exists."""
    for writer in sorted(p for p in root.iterdir() if p.is_dir()):
        for cls in RetentionClass:
            if (writer / cls.value).is_dir():
                yield cls, writer / cls.value


class ComplianceLog:
    """One process's writer into the compliance log at ``root``.

    Thread-safe. A line is written with one ``os.write`` on an ``O_APPEND`` descriptor and,
    for safety-relevant classes, fsync'd before ``append`` returns.

    Args:
        root: The log directory; created if missing.
        writer: This process's subdirectory (``recorder``, ``runtime``…).

    Raises:
        ValueError: ``writer`` is not a short lowercase name.
    """

    def __init__(self, root: str | Path, *, writer: str) -> None:
        """Open the writer, quarantining a torn tail a crash left behind."""
        if not _WRITER.match(writer):
            raise ValueError(f"compliance log writer must match {_WRITER.pattern}: {writer!r}")
        self.root = Path(root)
        self._dir = self.root / writer
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._heads = {cls: self._recover(self._dir / cls.value) for cls in RetentionClass}

    @classmethod
    def from_env(cls, *, writer: str) -> ComplianceLog | None:
        """The log at ``$OPENRAL_COMPLIANCE_DIR``, or ``None`` when it is unset."""
        env = os.environ.get(COMPLIANCE_DIR_ENV)
        return cls(env, writer=writer) if env else None

    @staticmethod
    def _recover(class_dir: Path) -> str:
        """Head hash of a class; a torn last line moves to ``<segment>.quarantine``."""
        segments = _segments(class_dir)
        for i, seg in enumerate(reversed(segments)):
            data = seg.read_bytes()
            if i == 0 and data and not data.endswith(b"\n"):
                keep, _, torn = data.rpartition(b"\n")
                keep = keep + b"\n" if keep else b""
                seg.with_suffix(".quarantine").write_bytes(torn)
                seg.write_bytes(keep)
                data = keep
            lines = data.splitlines()
            if lines:
                return _line_hash(lines[-1])
        return GENESIS

    def append(self, record: ComplianceRecord) -> None:
        """Write one record to its class's segment for the record's UTC day.

        Raises:
            OSError: The write failed (disk full, read-only filesystem). The log never
                swallows it; the caller decides whether that stops the machine (#468).
        """
        cls = record.retention_class
        day = dt.datetime.fromtimestamp(record.ts_ns / 1e9, tz=dt.UTC).date().isoformat()
        with self._lock:
            line = json.dumps(
                {"prev_hash": self._heads[cls], "record": json.loads(record.model_dump_json())},
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            path = self._dir / cls.value / f"{day}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
            try:
                os.write(fd, line + b"\n")
                if cls in _FSYNC_CLASSES:
                    os.fsync(fd)
            finally:
                os.close(fd)
            self._heads[cls] = _line_hash(line)


def records(root: str | Path, cls: RetentionClass) -> Iterator[ComplianceRecord]:
    """Every record of a class, all writers, in file order (oldest segment first per writer)."""
    for c, class_dir in _class_dirs(Path(root)):
        if c is cls:
            for seg in _segments(class_dir):
                for line in seg.read_bytes().splitlines():
                    yield _ADAPTER.validate_python(json.loads(line)["record"])


def verify(root: str | Path) -> list[str]:
    """Walk every chain; one message per problem, empty when the log is intact.

    Reports a broken ``prev_hash`` link (an edited, deleted or reordered line), a line that
    is not a compliance record, a quarantined torn tail, and — unless retention has pruned
    something, which removes records by design — a ``seq`` gap within a ``stream_id``.

    Example:
        >>> import tempfile
        >>> verify(tempfile.mkdtemp())
        []
    """
    root = Path(root)
    problems: list[str] = []
    seqs: dict[str, set[int]] = {}
    pruned = False
    for _, class_dir in _class_dirs(root):
        pruned |= (class_dir / "pruned").exists()
        prev = None if (class_dir / "pruned").exists() else GENESIS
        for seg in _segments(class_dir):
            if seg.with_suffix(".quarantine").exists():
                problems.append(f"{seg}: a torn tail was quarantined at open")
            for n, line in enumerate(seg.read_bytes().splitlines(), 1):
                where = f"{seg}:{n}"
                try:
                    entry = json.loads(line)
                    record = _ADAPTER.validate_python(entry["record"])
                    link = entry["prev_hash"]
                except (ValueError, KeyError, TypeError, ValidationError):
                    problems.append(f"{where}: not a compliance record line")
                    prev = _line_hash(line)
                    continue
                if prev is not None and link != prev:
                    problems.append(f"{where}: chain broken (prev_hash does not match)")
                prev = _line_hash(line)
                if record.stream_id is not None:
                    seqs.setdefault(record.stream_id, set()).add(record.seq)
    if not pruned:
        for stream, seen in seqs.items():
            missing = sorted(set(range(min(seen), max(seen) + 1)) - seen)
            if min(seen) != 0:
                missing = [*range(min(seen)), *missing]
            if missing:
                problems.append(f"stream {stream[:8]}: seq {missing[:5]} missing")
    return problems


def hold(root: str | Path, *, on: bool) -> None:
    """Put the whole log on legal hold (nothing is deleted) or release it (AI Act Art. 73(6))."""
    marker = Path(root) / "hold"
    if on:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    else:
        marker.unlink(missing_ok=True)


def prunable(
    root: str | Path, today: dt.date, keep_days: dict[RetentionClass, int] | None = None
) -> list[Path]:
    """Segments past their class's retention; none while the log is on hold.

    ``keep_days`` may extend a class's floor (``RETENTION_FLOOR_DAYS``), never shorten it. A
    segment is kept until its whole day is older than the retention.
    """
    root = Path(root)
    if (root / "hold").exists():
        return []
    out: list[Path] = []
    for cls, class_dir in _class_dirs(root):
        days = max(RETENTION_FLOOR_DAYS[cls] or RAW_TELEMETRY_DAYS, (keep_days or {}).get(cls, 0))
        for seg in _segments(class_dir):
            if dt.date.fromisoformat(seg.stem) + dt.timedelta(days=days) < today:
                out.append(seg)
    return out


def delete(segments: list[Path]) -> None:
    """Delete segments ``prunable`` returned, marking their classes as pruned for ``verify``.

    Callers record the deletion as an ``InterventionRecord(intervention_kind=
    "retention_delete")`` (``openral compliance prune`` does).
    """
    for seg in segments:
        (seg.parent / "pruned").touch()
        seg.unlink()
        seg.with_suffix(".quarantine").unlink(missing_ok=True)


def export(
    root: str | Path, out: str | Path, *, classes: list[RetentionClass], since: dt.date | None
) -> int:
    """Write the chosen classes' segments (from ``since``) to a ``.tar.gz`` for an authority.

    Whole segments go in, so the recipient can re-run ``verify`` on the bundle. Returns the
    number of segments written.
    """
    root = Path(root)
    n = 0
    with tarfile.open(out, "w:gz") as tar:
        for cls, class_dir in _class_dirs(root):
            if cls not in classes:
                continue
            for seg in _segments(class_dir):
                if since is None or dt.date.fromisoformat(seg.stem) >= since:
                    tar.add(seg, arcname=str(seg.relative_to(root)))
                    n += 1
    return n


def ship(
    root: str | Path,
    exporter: LogRecordExporter,
    *,
    resource_attributes: dict[str, str],
) -> int:
    """Send every not-yet-shipped record to ``exporter``; return how many went.

    Resumes from ``<root>/ship.state`` (lines already accepted, per segment) and advances it
    only after the collector accepted each record, so an outage costs nothing: the next run
    picks up where this one stopped. The receiving side dedupes on ``record_id``, so a crash
    between "accepted" and "state saved" re-sends at most one record. Raises nothing for an
    unreachable collector; it returns what it managed to send.
    """
    from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
    from opentelemetry.sdk._logs.export import LogRecordExportResult, SimpleLogRecordProcessor
    from opentelemetry.sdk.resources import Resource

    from openral_observability.compliance import send_record

    root = Path(root)
    state_path = root / "ship.state"
    state: dict[str, int] = json.loads(state_path.read_text()) if state_path.exists() else {}
    results: list[LogRecordExportResult] = []

    class _Checked(LogRecordExporter):  # one export per record, result kept
        def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
            result = exporter.export(batch)
            results.append(result)
            return result

        def shutdown(self) -> None:
            exporter.shutdown()  # type: ignore[no-untyped-call]  # reason: OTel SDK leaves it unannotated

    provider = LoggerProvider(resource=Resource.create(resource_attributes))
    provider.add_log_record_processor(SimpleLogRecordProcessor(_Checked()))
    logger = provider.get_logger("openral.compliance")
    sent = 0
    try:
        for _, class_dir in _class_dirs(root):
            for seg in _segments(class_dir):
                key = str(seg.relative_to(root))
                lines = seg.read_bytes().splitlines()
                for line in lines[state.get(key, 0) :]:
                    send_record(logger, _ADAPTER.validate_python(json.loads(line)["record"]))
                    if results[-1] is not LogRecordExportResult.SUCCESS:
                        return sent
                    state[key] = state.get(key, 0) + 1
                    sent += 1
    finally:
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True))
        tmp.replace(state_path)
        provider.shutdown()
    return sent
