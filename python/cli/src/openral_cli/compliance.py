"""``openral compliance`` — the on-robot compliance log (issue #469).

The log the deploy writes under ``$OPENRAL_COMPLIANCE_DIR`` (see
``openral_observability.compliance_log``): verify its hash chains, show the installed
software, export a bundle for a competent authority, put it on legal hold, prune what is past
retention, and ship it to an OTLP collector.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import typer
from openral_core import (
    ORG_ID_ENV,
    UNIT_SERIAL_ENV,
    Actor,
    InterventionRecord,
    RetentionClass,
    SoftwareInventoryRecord,
)
from openral_observability._sdk import _load_exporters
from openral_observability.compliance import ComplianceEmitter, software_inventory
from openral_observability.compliance_log import (
    ComplianceLog,
    default_compliance_dir,
    delete,
    export,
    hold,
    prunable,
    records,
    ship,
    verify,
)
from rich.console import Console

__all__ = ["compliance_app"]

compliance_app = typer.Typer(
    name="compliance",
    help="The on-robot compliance log: verify, inventory, export, hold, prune, ship.",
    no_args_is_help=True,
)
_console = Console()


def _root(root: Path | None) -> Path:
    path = root or default_compliance_dir()
    if not path.is_dir():
        _console.print(f"[red]no compliance log at {path}[/red] (set $OPENRAL_COMPLIANCE_DIR)")
        raise typer.Exit(2)
    return path


_ROOT = typer.Option(None, "--root", help="Log directory (default: $OPENRAL_COMPLIANCE_DIR).")
_OUT = typer.Option(..., "--out", help="Bundle to write (.tar.gz).")
_CLASSES = typer.Option(
    list(RetentionClass), "--class", help="Classes to include (repeatable; default all)."
)


@compliance_app.command("verify")
def verify_command(root: Path | None = _ROOT) -> None:
    """Walk every hash chain; exit 1 on an edited, deleted or missing record."""
    problems = verify(_root(root))
    for problem in problems:
        _console.print(f"[red]✗[/red] {problem}")
    if problems:
        raise typer.Exit(1)
    _console.print("[green]✓[/green] every chain intact, no sequence gaps")


@compliance_app.command("inventory")
def inventory_command(
    root: Path | None = _ROOT,
    live: bool = typer.Option(False, help="Inspect this host now instead of the last record."),
) -> None:
    """The software installed on this unit (MR Annex III §1.1.9), safety software marked."""
    if live:
        components = software_inventory()
        source = "this host, now"
    else:
        latest = None
        for record in records(_root(root), RetentionClass.MR_TRACE_5Y):
            if isinstance(record, SoftwareInventoryRecord) and (
                latest is None or record.ts_ns > latest.ts_ns
            ):
                latest = record
        if latest is None:
            _console.print("no software_inventory record yet; use --live")
            raise typer.Exit(1)
        components = latest.components
        source = dt.datetime.fromtimestamp(latest.ts_ns / 1e9, tz=dt.UTC).isoformat()
    _console.print(f"[bold]installed software[/bold] ({source})")
    for c in components:
        mark = "[yellow]safety[/yellow] " if c.safety_relevant else ""
        digest = f"  sha256:{c.sha256[:16]}…" if c.sha256 else ""
        _console.print(f"  {mark}{c.name} {c.version}{digest}")


@compliance_app.command("export")
def export_command(
    out: Path = _OUT,
    retention_class: list[RetentionClass] = _CLASSES,
    since: str | None = typer.Option(None, help="First day to include, YYYY-MM-DD."),
    root: Path | None = _ROOT,
) -> None:
    """Bundle whole segments for a competent authority's reasoned request."""
    n = export(
        _root(root),
        out,
        classes=retention_class,
        since=dt.date.fromisoformat(since) if since else None,
    )
    _console.print(f"wrote {n} segment(s) to {out}; the recipient can run `verify` on it")


@compliance_app.command("hold")
def hold_command(
    on: bool = typer.Argument(..., help="true: hold (nothing is deleted); false: release."),
    root: Path | None = _ROOT,
) -> None:
    """Legal hold: freeze the whole log while an incident is investigated (AI Act Art. 73(6))."""
    hold(_root(root), on=on)
    _console.print("log on hold: nothing will be pruned" if on else "hold released")


@compliance_app.command("prune")
def prune_command(
    dry_run: bool = typer.Option(False, help="List what would go, delete nothing."),
    root: Path | None = _ROOT,
) -> None:
    """Delete segments past their retention floor; each deletion is itself recorded."""
    path = _root(root)
    doomed = prunable(path, dt.datetime.now(tz=dt.UTC).date())
    for seg in doomed:
        _console.print(f"{'would delete' if dry_run else 'delete'} {seg.relative_to(path)}")
    if dry_run or not doomed:
        return
    emitter = ComplianceEmitter(log=ComplianceLog(path, writer="cli"))
    for seg in doomed:
        emitter.emit(
            InterventionRecord,
            intervention_kind="retention_delete",
            actor=Actor(channel_label="openral compliance prune"),
            channel="system",
            legitimate=True,
            diff_ref=str(seg.relative_to(path)),
        )
    delete(doomed)


@compliance_app.command("ship")
def ship_command(
    endpoint: str = typer.Option(
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        help="OTLP collector (default $OTEL_EXPORTER_OTLP_ENDPOINT); headers from "
        "$OTEL_EXPORTER_OTLP_HEADERS, protocol from $OTEL_EXPORTER_OTLP_PROTOCOL.",
    ),
    serial: str = typer.Option(os.environ.get(UNIT_SERIAL_ENV, ""), help="Unit serial."),
    org: str = typer.Option(os.environ.get(ORG_ID_ENV, ""), help="Organisation id."),
    root: Path | None = _ROOT,
) -> None:
    """Send records not yet shipped; resumes where the last run stopped."""
    if not endpoint or not serial:
        _console.print("[red]--endpoint and --serial are required[/red] (or their env vars)")
        raise typer.Exit(2)
    _, _, log_exporter_cls = _load_exporters()
    is_http = "http" in os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "")
    exporter = log_exporter_cls(endpoint=f"{endpoint.rstrip('/')}/v1/logs" if is_http else endpoint)
    attributes = {"service.name": "openral.compliance", "openral.unit.serial": serial}
    if org:
        attributes["openral.org.id"] = org
    sent = ship(_root(root), exporter, resource_attributes=attributes)
    _console.print(f"shipped {sent} record(s)")
