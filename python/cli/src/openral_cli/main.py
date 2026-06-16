"""openral CLI entry point — ``openral`` command.

Two modes of use:

* **One-shot**: ``openral <subcommand> [args...]`` runs a single command and
  exits. Use this in scripts and CI. ``openral --help`` lists the surface.
* **Interactive REPL**: ``openral`` with no arguments drops into a prompt
  where subcommands run bare (``sim run --config …`` instead of
  ``openral sim run --config …``). Type ``help`` for the menu, ``exit`` or
  Ctrl-D to leave.

Sub-commands
------------
doctor              Diagnose the host environment (Python, OS, ROS 2, GPU, USB).
detect              Probe hardware and write a full RobotDescription robot.yaml.
connect             Open a HAL connection to a robot and verify it responds.
calibrate camera    Calibrate a camera sensor using ros2 camera_calibration.
install             Install opt-in dependency groups (sim, ros, libero, …) — see ADR-0021.
rskill search       Find installable rSkills on the OpenRAL HF Hub org (ADR-0055).
rskill install      Download an rSkill from the HF Hub and register it locally.
rskill list         List all locally installed rSkills.
rskill check        Report which installed rSkills will run on the current host.
rskill new          Scaffold a new local rSkill from rskills/template/.
collision lower     Lower a robot's URDF/SRDF into its self-collision model (ADR-0030).
collision check     Fail if a manifest drifts from its lowered collision model.

Run ``openral --help`` for full usage.
"""

from __future__ import annotations

import contextlib
import json as _json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from glob import glob
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, cast
from urllib.parse import urlparse

import click
import typer
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_observability import (
    cli_command_span,
    configure_observability,
    semconv,
)
from openral_sim.cli import sim_app
from rich.console import Console
from rich.table import Table

from openral_cli.collision import collision_app
from openral_cli.dataset import dataset_app
from openral_cli.deploy_sim import deploy_sim_command
from openral_cli.install import install_app
from openral_cli.prompt import prompt_command

if TYPE_CHECKING:
    from openral_core import BenchmarkScene, RSkillEvalResult, VLASpec
    from openral_core.schemas import RSkillManifest
    from openral_detect import CompatibilityReport, RSkillCompatRow

    from openral_cli._rskill_intel import RSkillFamily, RSkillPatch

app = typer.Typer(
    name="openral",
    help="OpenRAL — open-source robot agent harness for rSkill / VLA models",
    invoke_without_command=True,
)
console = Console()

# ── REPL ──────────────────────────────────────────────────────────────────────

BANNER = "\n".join(
    [
        "",
        " ▒▒         ▒░     ██████╗ ██████╗ ███████╗███╗   ██╗██████╗  █████╗ ██╗     ",
        " █▓         ▓█    ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗██╔══██╗██║     ",
        " ██▓░     ░▓██    ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██████╔╝███████║██║     ",
        " ▒█████ █████▒    ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██╗██╔══██║██║     ",
        "   ▒▓█████▓▒      ╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║██║  ██║███████╗",
        "     ▓███▓         ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝",
        "",
    ]
)
SUBTITLE = "The open-source agentic layer for physical AI"


def _print_banner() -> None:
    """Print the ASCII OpenRAL banner + subtitle to the REPL console."""
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")
    console.print(f"  [dim]{SUBTITLE}[/dim]\n")
    console.print("  Type [bold]help[/bold] for commands, [bold]exit[/bold] (or Ctrl-D) to quit.\n")


def _dispatch_repl_line(line: str) -> None:
    """Tokenise a REPL line and re-enter the Typer app as if invoked from a shell.

    Uses ``shlex.split`` so quoting works (``sim run --config 'path with
    spaces.yaml'``). The Typer app is invoked with ``standalone_mode=False``
    so ``typer.Exit`` and ``click.exceptions.UsageError`` don't tear down
    the REPL. Each line spawns its own top-level callback + tracing scope.
    """
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        console.print(f"[red]parse error:[/red] {exc}")
        return
    if not tokens:
        return
    head = tokens[0].lower()
    if head in {"exit", "quit", ":q"}:
        raise EOFError
    if head in {"help", "?"}:
        # Re-enter with --help so Typer prints the full surface.
        tokens = ["--help"]
    try:
        app(args=tokens, prog_name="openral", standalone_mode=False)
    except click.exceptions.UsageError as exc:
        exc.show()
    except click.exceptions.Abort:
        console.print("[yellow]aborted[/yellow]")
    except SystemExit:
        # Some commands still call sys.exit; swallow it so the REPL survives.
        pass
    except Exception as exc:  # reason: keep REPL alive on subcommand crashes
        console.print(f"[red]error:[/red] {exc}")


def _path_completer(text: str, state: int) -> str | None:
    """``readline``-shaped tab-completion function for filesystem paths.

    Expands a leading ``~`` against ``$HOME``, globs ``<text>*``, suffixes
    directory matches with ``/`` so a second Tab descends into them, and
    rewrites the home prefix back to ``~`` on return so a user who typed
    ``~/foo`` does not see their line buffer silently rewritten to an
    absolute path. ``state`` is readline's call-counter contract: state=0
    returns the first match, state=N returns the (N+1)-th, and we return
    ``None`` past the end to signal exhaustion.
    """
    import glob
    import os

    expanded = os.path.expanduser(text) if text else ""
    raw = sorted(glob.glob(expanded + "*"))
    matches = [m + "/" if os.path.isdir(m) else m for m in raw]

    if text.startswith("~"):
        home = os.path.expanduser("~")
        if home and home != "~":
            matches = [
                "~" + m[len(home) :] if m == home or m.startswith(home + os.sep) else m
                for m in matches
            ]

    if state < len(matches):
        return matches[state]
    return None


def _run_repl() -> None:
    """Run the interactive ``openral>`` shell until EOF or ``exit``.

    Uses stdlib ``input()`` + optional ``readline`` (stdlib) for arrow-key
    history and Tab path completion. Deliberately avoids a hard dependency
    on ``prompt_toolkit`` so the curl-bash Tier-0 install (uv +
    openral-cli only) is enough.
    """
    import contextlib

    with contextlib.suppress(ImportError):
        # readline is absent on Windows; REPL still works, just without
        # history or Tab completion.
        import readline

        readline.set_completer(_path_completer)
        # Shell-shaped delimiters: split on whitespace and shell
        # metacharacters only, so a path token like "~/foo/bar.yaml" is
        # passed to the completer whole instead of being chopped at "~",
        # "/", or ".".
        readline.set_completer_delims(" \t\n=;|&><")
        # macOS ships libedit-backed readline whose bind syntax differs.
        if "libedit" in getattr(readline, "__doc__", "") or "":
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")

    _print_banner()
    while True:
        try:
            line = input("openral> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()  # newline after ^D / ^C
            break
        if not line:
            continue
        try:
            _dispatch_repl_line(line)
        except EOFError:
            break


_RUN_MODE_BY_SUBCOMMAND: dict[str, str] = {
    "sim": semconv.RUN_MODE_SIM,
    "benchmark": semconv.RUN_MODE_BENCHMARK,
    "deploy": semconv.RUN_MODE_HARDWARE,
    "connect": semconv.RUN_MODE_HARDWARE,
}

# Hardware deployments at >=100 Hz over 24 h would emit millions of tick spans
# per day at ALWAYS_ON. ADR-0010 (2026-05-17 amendment) calls for a
# 10% ratio sampler on hardware mode and ALWAYS_ON for sim / benchmark
# / one-shot subcommands (doctor / detect / skill install / …) where the
# total volume is bounded by a single invocation.
_SAMPLE_RATIO_BY_MODE: dict[str, float] = {
    semconv.RUN_MODE_HARDWARE: 0.1,
}


@app.callback()
def _root(ctx: typer.Context) -> None:
    """Initialise tracing+logs, open the ``cli.command`` root span, or enter REPL.

    Bare ``openral`` invocations (no subcommand) drop into the interactive
    REPL where each entered line is re-dispatched through the Typer app as
    if typed on the shell. Subcommand invocations behave exactly as before:
    a single ``cli.command`` root span wraps the call and the sampler is
    chosen by ``openral.run.mode`` (hardware → 10% ratio, others → always-on)
    per ADR-0010's 2026-05-17 amendment. ``OPENRAL_OTEL_SAMPLE_RATIO``
    overrides for ad-hoc debugging.
    """
    if ctx.invoked_subcommand is None:
        # Configure tracing with always-on (REPL == bounded session), then
        # drop into the prompt. Each dispatched line re-enters this callback
        # with its own subcommand, so the per-command span tree stays intact.
        configure_observability(service_name="openral", sample_ratio=None)
        _run_repl()
        return

    subcommand = ctx.invoked_subcommand
    mode = _RUN_MODE_BY_SUBCOMMAND.get(subcommand)
    sample_ratio = _SAMPLE_RATIO_BY_MODE.get(mode) if mode is not None else None
    configure_observability(service_name="openral", sample_ratio=sample_ratio)
    ctx.with_resource(cli_command_span(subcommand, mode=mode))


# Status values used throughout; kept as plain str for JSON serialisation.
# Colour mapping: ok→green, absent/info→yellow, everything else→red.
_YELLOW_STATUSES = frozenset({"absent", "info", "warn"})


class CheckResult(NamedTuple):
    """One row in the ``openral doctor`` output table.

    Attributes:
        check: Short name of the thing being checked.
        status: One of ``ok``, ``fail``, ``missing``, ``absent``, ``info``, ``warn``.
        details: Human-readable detail string (path, version, device list, …).
    """

    check: str
    status: str
    details: str


# ── Individual check functions (each independently testable) ──────────────────


def _check_python() -> CheckResult:
    ok = sys.version_info >= (3, 10)
    return CheckResult("Python", "ok" if ok else "fail", platform.python_version())


def _check_platform() -> CheckResult:
    return CheckResult("Platform", "info", f"{platform.system()} {platform.release()}")


def _check_openral_core() -> CheckResult:
    try:
        v = version("openral-core")
        return CheckResult("openral-core", "ok", v)
    except PackageNotFoundError as exc:
        return CheckResult("openral-core", "fail", str(exc))


def _check_ros2() -> list[CheckResult]:
    """Return one or more rows covering the ROS 2 binary, distro, and RMW."""
    results: list[CheckResult] = []

    ros2_path = shutil.which("ros2")
    if not ros2_path:
        results.append(CheckResult("ROS 2 binary", "missing", "not found"))
        return results
    results.append(CheckResult("ROS 2 binary", "ok", ros2_path))

    # Distro — set by sourcing /opt/ros/<distro>/setup.bash
    distro = os.environ.get("ROS_DISTRO", "")
    if distro:
        results.append(CheckResult("ROS 2 distro", "ok", distro))
    else:
        installed = sorted(glob("/opt/ros/*/setup.bash"))
        if installed:
            names = [p.split("/")[3] for p in installed]
            results.append(
                CheckResult(
                    "ROS 2 distro",
                    "info",
                    f"installed: {', '.join(names)} — run: source /opt/ros/<distro>/setup.bash",
                )
            )
        else:
            results.append(
                CheckResult(
                    "ROS 2 distro",
                    "missing",
                    "ROS_DISTRO not set and no /opt/ros/* found",
                )
            )

    # RMW implementation
    rmw = os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp (default)")
    results.append(CheckResult("RMW", "info", rmw))

    return results


def _check_colcon() -> CheckResult:
    path = shutil.which("colcon")
    return CheckResult("colcon", "ok" if path else "missing", path or "")


def _check_gpu() -> list[CheckResult]:
    """Return one row per detected GPU / SoC accelerator.

    Delegates to `openral_detect.probes.probe_gpus` so the CLI
    and the auto-provisioning flow share one enumeration code path.
    """
    from openral_detect.probes import probe_gpus

    warnings: list[str] = []
    result = probe_gpus(warnings=warnings)
    rows: list[CheckResult] = []
    for gpu in result.nvidia:
        rows.append(
            CheckResult(
                f"GPU {gpu.index}",
                "ok",
                f"{gpu.name} ({gpu.vram_total_mib} MiB, "
                f"sm_{gpu.cuda_compute_capability[0]}{gpu.cuda_compute_capability[1]})",
            )
        )
    if result.jetson is not None:
        rows.append(
            CheckResult(
                "Jetson",
                "ok",
                f"{result.jetson.board} ({result.jetson.tops:.0f} TOPS, "
                f"{result.jetson.ram_gb:.0f} GB unified)",
            )
        )
    if result.apple_silicon is not None:
        rows.append(CheckResult("GPU", "info", f"Apple Silicon — {result.apple_silicon.chip}"))
    if not rows:
        if warnings:
            rows.append(CheckResult("GPU", "absent", warnings[0]))
        else:
            rows.append(CheckResult("GPU", "absent", "no accelerator detected"))
    return rows


def _check_usb() -> list[CheckResult]:
    """Return one row listing USB serial devices that could be robot controllers."""
    if platform.system() == "Linux":
        patterns = ["/dev/ttyUSB*", "/dev/ttyACM*"]
    elif platform.system() == "Darwin":
        patterns = ["/dev/cu.usbserial*", "/dev/cu.usbmodem*"]
    else:
        return [CheckResult("USB devices", "info", "enumeration not supported on this OS")]

    devices: list[str] = []
    for pattern in patterns:
        devices.extend(sorted(glob(pattern)))

    if devices:
        return [CheckResult("USB devices", "ok", ", ".join(devices))]
    return [CheckResult("USB devices", "info", "none found")]


def _check_just() -> CheckResult:
    path = shutil.which("just")
    # `just` is a developer-convenience task runner, not a runtime requirement
    # of `openral`; report absence with `warn` rather than `missing` so doctor
    # still exits 0 on hosts that only need to run skills.
    return CheckResult("just", "ok" if path else "warn", path or "not found")


# PROVIDER values whose endpoint enforces auth and so require
# OPENRAL_REASONER_LLM_API_KEY. Bare ``openai-compatible`` is the
# exception because a local Ollama / llama-server doesn't.
_REASONER_PROVIDERS_REQUIRING_KEY: frozenset[str] = frozenset({"anthropic", "openrouter"})

# Provider-default base URLs used when the user hasn't set
# OPENRAL_REASONER_LLM_BASE_URL. Mirrors tool_use.py constants but kept
# local to avoid forcing the CLI to import the (optionally-installed)
# reasoner package on every `openral doctor` invocation.
_REASONER_PROVIDER_DEFAULT_BASE_URL: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai-compatible": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _is_local_base_url(url: str) -> bool:
    """Return True when ``url``'s host resolves to a loopback name."""
    host = urlparse(url).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _probe_tcp(host: str, port: int, *, timeout_s: float = 0.2) -> bool:
    """Return True if a TCP connection to ``host:port`` succeeds quickly."""
    with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=timeout_s):
        return True
    return False


def _check_reasoner_llm() -> list[CheckResult]:
    """Return rows describing the reasoner LLM env configuration.

    Always emits a leading ``Reasoner LLM`` summary row. When the
    provider is set but the rest of the config is incomplete, follow-up
    rows name each missing variable so the user can read the table
    top-to-bottom and see exactly what to export.

    When the resolved endpoint is loopback (Ollama / local vLLM /
    llama-server), an additional ``Ollama`` row TCP-probes the port so
    a user trying the local baseline gets an immediate diagnosis if the
    daemon isn't running.

    The API key value is never printed — only ``set`` / ``unset``.
    """
    rows: list[CheckResult] = []
    provider_raw = os.environ.get("OPENRAL_REASONER_LLM_PROVIDER", "").strip()
    provider = provider_raw.lower()

    if not provider:
        rows.append(
            CheckResult(
                "Reasoner LLM",
                "absent",
                "OPENRAL_REASONER_LLM_PROVIDER unset — see "
                "packages/openral_reasoner_ros/README.md for the three baseline configs "
                "(anthropic / openrouter / openai-compatible).",
            )
        )
        return rows

    if provider not in _REASONER_PROVIDER_DEFAULT_BASE_URL:
        rows.append(
            CheckResult(
                "Reasoner LLM",
                "fail",
                f"OPENRAL_REASONER_LLM_PROVIDER={provider_raw!r}; expected one of "
                f"{sorted(_REASONER_PROVIDER_DEFAULT_BASE_URL)!r}.",
            )
        )
        return rows

    model = os.environ.get("OPENRAL_REASONER_LLM_MODEL", "").strip()
    api_key = os.environ.get("OPENRAL_REASONER_LLM_API_KEY", "").strip()
    base_url_env = os.environ.get("OPENRAL_REASONER_LLM_BASE_URL", "").strip()
    base_url = base_url_env or _REASONER_PROVIDER_DEFAULT_BASE_URL[provider]

    key_required = provider in _REASONER_PROVIDERS_REQUIRING_KEY
    key_status = "set" if api_key else "unset"
    parts = [
        f"provider={provider}",
        f"model={model or '<unset>'}",
        f"api_key={key_status}",
        f"base_url={base_url}",
    ]
    summary = " ".join(parts)

    incomplete: list[CheckResult] = []
    if not model:
        incomplete.append(
            CheckResult(
                "Reasoner MODEL",
                "missing",
                "OPENRAL_REASONER_LLM_MODEL unset — required for every provider.",
            )
        )
    if key_required and not api_key:
        incomplete.append(
            CheckResult(
                "Reasoner API_KEY",
                "missing",
                f"OPENRAL_REASONER_LLM_API_KEY unset — required for provider={provider}.",
            )
        )

    if incomplete:
        rows.append(CheckResult("Reasoner LLM", "warn", summary))
        rows.extend(incomplete)
    else:
        rows.append(CheckResult("Reasoner LLM", "ok", summary))

    # Ollama / local-endpoint probe. Only meaningful when the resolved
    # base_url is loopback; we never reach out to a cloud endpoint from
    # `openral doctor`.
    if _is_local_base_url(base_url):
        parsed = urlparse(base_url)
        port = parsed.port or 11434
        host = parsed.hostname or "localhost"
        if _probe_tcp(host, port):
            rows.append(
                CheckResult(
                    "Ollama",
                    "ok",
                    f"endpoint reachable at {host}:{port}",
                )
            )
        else:
            rows.append(
                CheckResult(
                    "Ollama",
                    "warn",
                    f"endpoint unreachable at {host}:{port} — "
                    "run `just bootstrap-ollama` or `ollama serve`.",
                )
            )

    return rows


def _gather_checks() -> list[CheckResult]:
    """Run all checks and return the combined result list.

    Returns:
        List of `CheckResult` in display order.
    """
    checks: list[CheckResult] = []
    checks.append(_check_python())
    checks.append(_check_platform())
    checks.append(_check_openral_core())
    checks.extend(_check_ros2())
    checks.append(_check_colcon())
    checks.extend(_check_gpu())
    checks.extend(_check_usb())
    checks.append(_check_just())
    checks.extend(_check_reasoner_llm())
    return checks


# ── CLI commands ──────────────────────────────────────────────────────────────


@app.command()
def doctor(
    json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Diagnose the host: Python, OS, ROS 2 distro, GPU, USB devices.

    Exits 0 when every check is ``ok``, ``absent``, or ``info``; exits 1 if
    any check has status ``fail`` or ``missing``.

    Example:
        >>> # openral doctor
        >>> # openral doctor --json
    """
    checks = _gather_checks()

    if json:
        result = [{"check": c.check, "status": c.status, "details": c.details} for c in checks]
        console.print_json(_json.dumps(result))
    else:
        table = Table(title="openral doctor")
        table.add_column("check", style="bold")
        table.add_column("status")
        table.add_column("details")
        for c in checks:
            style = (
                "green" if c.status == "ok" else "yellow" if c.status in _YELLOW_STATUSES else "red"
            )
            table.add_row(c.check, f"[{style}]{c.status}[/{style}]", c.details)
        console.print(table)

    fatal = {"fail", "missing"}
    if any(c.status in fatal for c in checks):
        raise typer.Exit(code=1)


@app.command()
def detect(
    output: Path = typer.Option(
        Path("robot.yaml"), "--output", "-o", help="Output robot.yaml path"
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Optional path to dump the raw DetectionReport as JSON.",
    ),
    dds_timeout: float = typer.Option(
        5.0, "--dds-timeout", help="DDS topic discovery timeout in seconds"
    ),
    include: str | None = typer.Option(
        None,
        "--include",
        help="Comma-separated probe names to run (default: all). "
        "Choices: usb, dds, gpu, cameras_v4l2, cameras_realsense, network.",
    ),
    no_write: bool = typer.Option(
        False, "--no-write", help="Print summary and skip writing robot.yaml"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Overwrite existing file without prompting"
    ),
) -> None:
    """Probe the host and emit a complete RobotDescription robot.yaml.

    Runs the auto-provisioning flow from ``openral_detect``:

    1. Probe USB / DDS / GPU / V4L2 / RealSense / network.
    2. Identify the rig (USB VID/PID match or DDS topology).  If a
       known robot is detected, load the canonical
       ``robots/<name>/robot.yaml`` directly; otherwise synthesize a
       minimal scaffold.
    3. Reverse-look up each detected sensor in the catalog so its
       ``SensorSpec`` carries **real** intrinsics, FOV, encoding, rate.
    4. Promote detected GPU / Jetson / Apple Silicon caps onto
       ``RobotCapabilities`` so ``openral rskill check`` can match
       ``RSkillManifest.runtime`` / ``quantization.dtype``.

    Example:
        >>> # openral detect
        >>> # openral detect --include gpu,network --no-write
    """
    import yaml as _yaml
    from openral_detect import (
        assemble_robot_description,
        detect_hardware,
    )

    include_set: set[str] | None = (
        {p.strip() for p in include.split(",") if p.strip()} if include else None
    )

    console.print("[bold]openral detect[/bold] — probing host …")
    detection = detect_hardware(dds_timeout_s=dds_timeout, include=include_set)

    _render_detection_summary(detection)

    if report is not None:
        report.write_text(detection.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {report} (raw DetectionReport)")

    description = assemble_robot_description(detection)
    yaml_text = _yaml.safe_dump(
        description.model_dump(mode="json"),
        sort_keys=False,
        default_flow_style=False,
    )

    if no_write:
        console.print("\n[dim]--no-write set — printing yaml to stdout:[/dim]\n")
        console.print(yaml_text)
        return

    if output.exists() and not yes:
        overwrite = typer.confirm(f"{output} already exists. Overwrite?", default=False)
        if not overwrite:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=0)

    output.write_text(yaml_text, encoding="utf-8")
    console.print(f"\n[green]Wrote[/green] {output} (RobotDescription, {description.name})")
    console.print(f"[dim]Next step:[/dim] openral rskill check --robot {output}")


def _render_detection_summary(detection: object) -> None:
    """Print a compact per-probe summary table of the detection report."""
    from openral_detect import DetectionReport

    assert isinstance(detection, DetectionReport)  # reason: typed input
    table = Table(title="openral detect")
    table.add_column("probe", style="bold")
    table.add_column("result")
    table.add_row(
        "usb", f"{len(detection.usb.devices)} device(s), {len(detection.usb.matches)} matched"
    )
    if detection.gpu.nvidia:
        table.add_row(
            "gpu (nvidia)",
            ", ".join(f"{g.name} ({g.vram_total_mib // 1024} GiB)" for g in detection.gpu.nvidia),
        )
    if detection.gpu.jetson is not None:
        table.add_row(
            "gpu (jetson)", f"{detection.gpu.jetson.board} ({detection.gpu.jetson.tops:.0f} TOPS)"
        )
    if detection.gpu.apple_silicon is not None:
        table.add_row("gpu (apple)", detection.gpu.apple_silicon.chip)
    if (
        not detection.gpu.nvidia
        and detection.gpu.jetson is None
        and detection.gpu.apple_silicon is None
    ):
        table.add_row("gpu", "[yellow]none detected[/yellow]")
    table.add_row(
        "cameras",
        f"v4l2={len(detection.cameras.v4l2)}, "
        f"realsense={len(detection.cameras.realsense)}, "
        f"orbbec={len(detection.cameras.orbbec)}",
    )
    inferred = detection.ros2.inferred_robot_type or "-"
    table.add_row(
        "ros2",
        f"{len(detection.ros2.topics)} topic(s), inferred={inferred}",
    )
    table.add_row(
        "network",
        f"{detection.network.hostname}, "
        f"{len(detection.network.interfaces)} iface(s), "
        f"route={detection.network.default_route or '-'}",
    )
    console.print(table)
    if detection.warnings:
        console.print("[dim]warnings:[/dim]")
        for w in detection.warnings:
            console.print(f"  [yellow]·[/yellow] {w}")


@app.command()
def connect(
    robot: str = typer.Option(..., help="Robot type (so100, g1, ur5e, …)"),
    port: str = typer.Option("", "--port", help="USB/serial port override, e.g. /dev/ttyUSB0"),
) -> None:
    """Open a HAL connection to a robot, read one joint state, and disconnect.

    Exits 0 on success; exits 1 with an error message on failure.

    Supported robots: so100

    Example:
        >>> # openral connect --robot so100
        >>> # openral connect --robot so100 --port /dev/ttyUSB1
    """
    if robot == "so100":
        _connect_so100(port or "/dev/ttyUSB0")
    else:
        console.print(f"[red]Unknown robot '{robot}'. Supported: so100[/red]")
        raise typer.Exit(code=1)


def _connect_so100(port: str) -> None:
    """Connect to an SO-100 follower arm, read state, and disconnect."""
    try:
        from openral_hal.so100_follower import SO100FollowerHAL
    except ImportError:
        console.print("[red]openral-hal is not installed. Run: uv sync --all-packages[/red]")
        raise typer.Exit(code=1)  # noqa: B904

    hal = SO100FollowerHAL(port=port)
    console.print(f"Connecting to SO-100 on [bold]{port}[/bold] …")
    try:
        hal.connect()
    except ROSConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1)  # noqa: B904
    except ROSRuntimeError as exc:
        console.print(f"[red]Runtime error:[/red] {exc}")
        raise typer.Exit(code=1)  # noqa: B904

    try:
        state = hal.read_state()
        joint_summary = ", ".join(
            f"{n}={v:.3f} rad" for n, v in zip(state.name, state.position, strict=True)
        )
        console.print(f"[green]Connected.[/green] Joint state: {joint_summary}")
    finally:
        hal.disconnect()
        console.print("Disconnected.")


# ── calibrate sub-app ─────────────────────────────────────────────────────────

calibrate_app = typer.Typer(
    name="calibrate",
    help="Sensor calibration helpers.",
    no_args_is_help=True,
)
app.add_typer(calibrate_app, name="calibrate")


@calibrate_app.command("camera")
def calibrate_camera(
    sensor: str = typer.Option(
        ...,
        "--sensor",
        "-s",
        help="Sensor name as it appears in robot.yaml (e.g. head_color).",
    ),
    topic: str = typer.Option(
        "",
        "--topic",
        help="Override the image topic (default: derived from sensor name).",
    ),
    chessboard_size: str = typer.Option(
        "8x6",
        "--chessboard-size",
        help="Internal corners COLSxROWS of the calibration target.",
    ),
    square_size: float = typer.Option(
        0.025,
        "--square-size",
        help="Physical size of one chessboard square in metres.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the command instead of executing it.",
    ),
) -> None:
    r"""Calibrate a camera sensor using the ROS 2 camera_calibration package.

    Builds and optionally runs::

        ros2 run camera_calibration cameracalibrator \
            --size COLSxROWS --square SIZE \
            --ros-args -r image:=TOPIC -r camera_info:=INFO_TOPIC

    Requires ``ros2_camera_calibration`` to be installed and ROS 2 sourced.

    Example:
        >>> # openral calibrate camera --sensor head_color --chessboard-size 8x6 --square-size 0.025
        >>> # openral calibrate camera --sensor head_color --dry-run
    """
    try:
        cols_str, rows_str = chessboard_size.lower().split("x")
        cols, rows = int(cols_str), int(rows_str)
    except ValueError:
        console.print(
            f"[red]Invalid --chessboard-size '{chessboard_size}'. "
            "Expected format: COLSxROWS, e.g. 8x6[/red]"
        )
        raise typer.Exit(code=1)  # noqa: B904

    # Derive topic names from sensor name if not overridden.
    image_topic = topic or f"/{sensor}/image_raw"
    info_topic = image_topic.replace("/image_raw", "/camera_info").replace(
        "/image_rect_raw", "/camera_info"
    )

    cmd = [
        "ros2",
        "run",
        "camera_calibration",
        "cameracalibrator",
        "--size",
        f"{cols}x{rows}",
        "--square",
        str(square_size),
        "--ros-args",
        "-r",
        f"image:={image_topic}",
        "-r",
        f"camera_info:={info_topic}",
    ]

    if dry_run:
        console.print("[bold]openral calibrate camera[/bold] — dry run:")
        console.print(" ".join(cmd))
        return

    ros2_bin = shutil.which("ros2")
    if ros2_bin is None:
        console.print(
            "[red]ros2 not found. Source your ROS 2 installation first:[/red]\n"
            "  source /opt/ros/<distro>/setup.bash"
        )
        raise typer.Exit(code=1)

    console.print(f"[bold]openral calibrate camera[/bold] — sensor: [cyan]{sensor}[/cyan]")
    console.print(f"  image topic : [dim]{image_topic}[/dim]")
    console.print(f"  camera info : [dim]{info_topic}[/dim]")
    console.print(f"  target size : [dim]{cols}x{rows} squares @ {square_size} m[/dim]")
    console.print("Running camera_calibration … (Ctrl+C to abort)")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        console.print(f"[red]cameracalibrator exited with code {result.returncode}.[/red]")
        raise typer.Exit(code=result.returncode)


if __name__ == "__main__":
    app()


# ── rskill sub-app ────────────────────────────────────────────────────────────

rskill_app = typer.Typer(
    name="rskill",
    help="rSkill package management — install and list robot skills from the HF Hub.",
    no_args_is_help=True,
)
app.add_typer(rskill_app, name="rskill")

#: Canonical HF Hub org for first-party rSkills (ADR-0055). Used to suggest a
#: repair when ``rskill install`` is handed an org-less id, and as the ``author``
#: filter for ``rskill search``.
_DEFAULT_RSKILL_ORG: Final[str] = "OpenRAL"


@rskill_app.command("install")
def rskill_install(
    hub_id: str = typer.Argument(
        ...,
        metavar="HUB_ID",
        help="HF Hub repository, e.g. OpenRAL/rskill-smolvla-libero",
    ),
    revision: str = typer.Option(
        "",
        "--revision",
        "-r",
        help="Git commit SHA or branch to pin (recommended for reproducibility).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-download even if cached files already exist.",
    ),
    non_commercial: bool = typer.Option(
        False,
        "--non-commercial",
        help="Declare non-commercial research intent (relaxes NVIDIA non-commercial guard).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt for proprietary or non-commercial licenses.",
    ),
) -> None:
    """Download an rSkill from the HF Hub, validate it, and register it locally.

    Fetches ``rskill.yaml`` from the repository, validates the manifest,
    surfaces the license to the terminal, then downloads the weights snapshot
    into the local HF Hub cache (``~/.cache/openral/rskills/``).

    The rSkill is registered in ``~/.local/share/openral/rskills.json`` and
    can be listed with ``openral rskill list``.

    Example:
        >>> # openral rskill install OpenRAL/rskill-smolvla-libero
        >>> # openral rskill install OpenRAL/rskill-smolvla-libero --revision abc1234
    """
    from openral_rskill.loader import rSkill

    # ── Step 0: a HF repo id needs an `org/name` shape. A bare name (the most
    # common paste mistake) otherwise 404s against a non-existent top-level repo
    # — fail fast with the canonical suggestion instead of a raw Hub stack trace.
    if "/" not in hub_id:
        suggestion = f"{_DEFAULT_RSKILL_ORG}/{hub_id}"
        console.print(
            f"[red]Not a Hub repo id:[/red] '{hub_id}' has no org prefix (expected `org/name`)."
        )
        console.print(f"  Did you mean:  [cyan]openral rskill install {suggestion}[/cyan]")
        console.print(f"  Or find it:    [cyan]openral rskill search {hub_id}[/cyan]")
        raise typer.Exit(code=1)

    console.print(f"[bold]openral rskill install[/bold] — fetching [cyan]{hub_id}[/cyan] …")

    # ── Step 1: fetch manifest only (to surface license before downloading weights)
    try:
        from huggingface_hub import hf_hub_download
        from openral_core.schemas import RSkillManifest
    except ImportError as exc:
        console.print(f"[red]Missing dependency:[/red] {exc}")
        raise typer.Exit(code=1)  # noqa: B904

    try:
        manifest_path = hf_hub_download(
            repo_id=hub_id,
            filename="rskill.yaml",
            revision=revision or None,
        )
        manifest = RSkillManifest.from_yaml(manifest_path)
    except Exception as exc:  # reason: surface any download/parse error to user
        console.print(f"[red]Failed to fetch manifest:[/red] {exc}")
        if "404" in str(exc) or "Repository Not Found" in str(exc):
            bare = hub_id.rsplit("/", 1)[-1]
            console.print(f"  Browse available skills: [cyan]openral rskill search {bare}[/cyan]")
        raise typer.Exit(code=1)  # noqa: B904

    # ── Step 2: display license + confirm if non-permissive
    _display_license_banner(manifest.name, manifest.license.value, manifest.version, console)

    _permissive = {"apache-2.0", "mit", "bsd"}
    if manifest.license.value not in _permissive and not yes:
        confirmed = typer.confirm("Proceed with installation?", default=False)
        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=0)

    # ── Step 3: full install (snapshot download + registry)
    console.print("  Downloading weights snapshot …")
    try:
        pkg = rSkill.from_pretrained(
            hub_id,
            revision=revision or None,
            force_download=force,
            commercial_use=not non_commercial,
        )
    except Exception as exc:  # reason: surface ROSConfigError + network errors
        console.print(f"[red]Install failed:[/red] {exc}")
        raise typer.Exit(code=1)  # noqa: B904

    console.print(
        f"[green]Installed[/green] [bold]{pkg.manifest.name}[/bold] "
        f"v{pkg.manifest.version} → {pkg.local_dir}"
    )
    if not revision:
        console.print(
            "[yellow]Tip:[/yellow] Pin a revision for reproducibility: "
            f"openral rskill install {hub_id} --revision <sha>"
        )


#: Max description chars rendered in the `rskill search` table before eliding.
_RSKILL_SEARCH_DESC_MAX: Final[int] = 60


def _load_hub_rskill_manifest(repo_id: str) -> RSkillManifest | None:
    """Fetch + validate one Hub repo's ``rskill.yaml``; ``None`` if absent/invalid.

    A repo with no manifest (or one that fails schema validation) is not an
    rSkill — the caller counts and surfaces these rather than failing the search.
    """
    from huggingface_hub import hf_hub_download
    from openral_core.schemas import RSkillManifest

    try:
        path = hf_hub_download(repo_id=repo_id, filename="rskill.yaml")
        return RSkillManifest.from_yaml(path)
    except Exception:  # reason: not an rSkill repo / invalid manifest — skip (counted by caller)
        return None


def _rskill_matches_filters(
    m: RSkillManifest, *, kind: str, role: str, embodiment: str, license_: str
) -> bool:
    """Return whether a manifest passes every non-empty facet filter."""
    if kind and m.kind != kind:
        return False
    if role and m.role != role:
        return False
    if embodiment and embodiment not in m.embodiment_tags:
        return False
    return not (license_ and m.license.value != license_)


def _render_rskill_search_results(
    rows: list[tuple[str, RSkillManifest]], skipped: int, query: str
) -> None:
    """Print the human-readable `rskill search` table (or a no-results notice)."""
    if not rows:
        suffix = f" ({skipped} repo(s) skipped — no valid manifest)" if skipped else ""
        console.print(
            f"[dim]No rSkills found in {_DEFAULT_RSKILL_ORG}/ for query "
            f"{query!r} with the given filters.{suffix}[/dim]"
        )
        return

    _permissive = {"apache-2.0", "mit", "bsd"}
    table = Table(title=f"rSkills on the Hub — {_DEFAULT_RSKILL_ORG}/")
    for col in ("repo_id", "kind", "role", "license", "embodiment_tags", "description"):
        table.add_column(col, style="cyan bold" if col == "repo_id" else None)
    for repo_id, m in rows:
        lic = m.license.value
        lic_color = "green" if lic in _permissive else "yellow"
        desc = (
            m.description
            if len(m.description) <= _RSKILL_SEARCH_DESC_MAX + 1
            else m.description[:_RSKILL_SEARCH_DESC_MAX] + "…"
        )
        table.add_row(
            repo_id,
            m.kind,
            m.role,
            f"[{lic_color}]{lic}[/{lic_color}]",
            ", ".join(m.embodiment_tags) or "—",
            desc,
        )
    console.print(table)
    if skipped:
        console.print(
            f"[dim]{skipped} {_DEFAULT_RSKILL_ORG} repo(s) skipped — no valid rskill.yaml.[/dim]"
        )
    console.print("[dim]Install one with:[/dim] [cyan]openral rskill install <repo_id>[/cyan]")


@rskill_app.command("search")
def rskill_search(
    query: str = typer.Argument(
        "",
        metavar="[QUERY]",
        help="Free-text query matched against rSkill repo ids on the Hub.",
    ),
    kind: str = typer.Option(
        "", "--kind", help="Filter by manifest kind (vla, ros_action, detector, …)."
    ),
    role: str = typer.Option("", "--role", help="Filter by control role (s0, s1, s2)."),
    embodiment: str = typer.Option(
        "", "--embodiment", help="Filter by embodiment tag (e.g. franka_panda)."
    ),
    license_: str = typer.Option(
        "", "--license", help="Filter by license posture (e.g. apache-2.0)."
    ),
    limit: int = typer.Option(50, "--limit", help="Max OpenRAL repos to inspect."),
    json: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
) -> None:
    """Search the OpenRAL HF Hub org for installable rSkills (ADR-0055 D4).

    Lists every ``OpenRAL/*`` repo whose ``rskill.yaml`` manifest validates and
    matches the optional facet filters, so the printed ids are paste-able into
    ``openral rskill install <repo_id>``. The HF Hub is the index — there is no
    bespoke catalog service. Repos without a valid manifest are skipped and the
    count surfaced.

    Example:
        >>> # openral rskill search aloha
        >>> # openral rskill search --kind detector --license apache-2.0
    """
    import json as _json_mod

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        console.print(f"[red]Missing dependency:[/red] {exc}")
        raise typer.Exit(code=1)  # noqa: B904

    try:
        models = list(
            HfApi().list_models(author=_DEFAULT_RSKILL_ORG, search=query or None, limit=limit)
        )
    except Exception as exc:  # reason: surface Hub/network errors to the user
        console.print(f"[red]Search failed:[/red] {exc}")
        raise typer.Exit(code=1)  # noqa: B904

    rows: list[tuple[str, RSkillManifest]] = []
    skipped = 0
    for model in models:
        manifest = _load_hub_rskill_manifest(model.id)
        if manifest is None:
            skipped += 1
        elif _rskill_matches_filters(
            manifest, kind=kind, role=role, embodiment=embodiment, license_=license_
        ):
            rows.append((model.id, manifest))

    if json:
        console.print_json(
            _json_mod.dumps(
                [
                    {
                        "repo_id": repo_id,
                        "name": m.name,
                        "version": m.version,
                        "kind": m.kind,
                        "role": m.role,
                        "license": m.license.value,
                        "embodiment_tags": list(m.embodiment_tags),
                        "description": m.description,
                    }
                    for repo_id, m in rows
                ]
            )
        )
        return

    _render_rskill_search_results(rows, skipped, query)


@rskill_app.command("list")
def rskill_list(
    json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """List every available rSkill — in-tree (``rskills/``) + HF-Hub-installed.

    Each row shows the source so users can tell at a glance which entries
    are paste-able as ``--rskill <name>`` (in-tree) versus which
    need a HF Hub install first. JSON output keeps the same fields.
    """
    import json as _json_mod

    from openral_rskill.loader import discover_intree_rskills, rSkill

    intree = discover_intree_rskills()
    try:
        installed = rSkill.list_installed()
    except Exception as exc:  # reason: surface corrupt registry error
        console.print(f"[red]Failed to read installed registry:[/red] {exc}")
        raise typer.Exit(code=1)  # noqa: B904

    def _bare_name_from_repo_id(repo_id: str) -> str:
        """Recover the bare rskill name that the loader would resolve to.

        Mirrors `_candidate_local_paths`: strip the org prefix and the
        `rskill-`/`rskill_` prefix so the URI matches the in-tree form.
        """
        tail = repo_id.rsplit("/", 1)[-1]
        return tail.removeprefix("rskill-").removeprefix("rskill_") or repo_id

    if json:
        data = [
            {
                "source": "in-tree",
                "name": name,
                "repo_id": m.name,
                "version": m.version,
                "model_family": m.model_family,
                "role": str(m.role),
                "license": m.license.value,
                "embodiment_tags": list(m.embodiment_tags),
                "uri": name,
            }
            for name, m in intree
        ] + [
            {
                "source": "installed",
                "name": _bare_name_from_repo_id(e.repo_id),
                "repo_id": e.repo_id,
                "version": e.version,
                "model_family": None,
                "role": e.role,
                "license": e.license,
                "embodiment_tags": list(e.embodiment_tags),
                "uri": _bare_name_from_repo_id(e.repo_id),
                "installed_at": e.installed_at,
            }
            for e in installed
        ]
        console.print_json(_json_mod.dumps(data))
        return

    if not intree and not installed:
        console.print(
            "[dim]No rSkills available. Drop one under rskills/<name>/ or run: "
            "openral rskill install <hub-id>[/dim]"
        )
        return

    table = Table(title="Available rSkills")
    table.add_column("source", style="dim")
    table.add_column("name / repo_id", style="cyan bold")
    table.add_column("version")
    table.add_column("family")
    table.add_column("license")
    table.add_column("embodiment_tags")
    table.add_column("paste-able --rskill")

    _permissive = {"apache-2.0", "mit", "bsd"}
    for name, m in intree:
        lic = m.license.value
        lic_color = "green" if lic in _permissive else "yellow"
        table.add_row(
            "in-tree",
            name,
            m.version,
            m.model_family or "—",
            f"[{lic_color}]{lic}[/{lic_color}]",
            ", ".join(m.embodiment_tags) or "—",
            name,
        )
    for entry in installed:
        lic_color = "green" if entry.license in _permissive else "yellow"
        bare = _bare_name_from_repo_id(entry.repo_id)
        table.add_row(
            "installed",
            entry.repo_id,
            entry.version,
            "—",
            f"[{lic_color}]{entry.license}[/{lic_color}]",
            ", ".join(entry.embodiment_tags) or "—",
            bare,
        )
    console.print(table)


_SECTION_DISPLAY: dict[str, str] = {
    "embodiment": "Embodiment",
    "capability_flags": "Capability flags",
    "gpu_runtime": "GPU runtime",
    "gpu_dtype": "GPU dtype",
    "sensors": "Sensors",
    "actuators": "Actuators",
}


def _render_single_rskill_table(row: RSkillCompatRow, robot_name: str) -> None:
    """Print the per-section table for `openral rskill check <rskill_id>`."""
    console.print(f"[bold]rSkill compatibility[/bold] for [cyan]{robot_name}[/cyan]")
    header = f"rSkill: [cyan bold]{row.repo_id}[/cyan bold]"
    if row.version:
        header += f"  v{row.version}"
    if row.role:
        header += f"  role={row.role}"
    console.print(header)

    if not row.sections:
        console.print(f"[red]✗ {row.failure_kind}[/red]  {row.reason or ''}")
        return

    section_table = Table(show_header=True, header_style="bold")
    section_table.add_column("Section")
    section_table.add_column("Status")
    section_table.add_column("Reason")
    for section in row.sections:
        label = _SECTION_DISPLAY.get(section.label, section.label)
        if section.informational:
            status = "[dim]· informational[/dim]"
        elif section.compatible:
            status = "[green]✓[/green]"
        else:
            status = f"[red]✗ {section.failure_kind or 'fail'}[/red]"
        section_table.add_row(label, status, section.reason or "")
    console.print(section_table)

    blocking = [s for s in row.sections if not s.informational and not s.compatible]
    if blocking:
        plural = "s" if len(blocking) != 1 else ""
        console.print(
            f"[red]Overall: ✗ incompatible ({len(blocking)} failing section{plural})[/red]"
        )
    else:
        console.print("[green]Overall: ✓ compatible[/green]")


def _render_walk_all_table(report: CompatibilityReport, robot_name: str) -> None:
    """Print the walk-all (no-arg) table for `openral rskill check`."""
    if not report.rows:
        console.print(
            "[dim]No rSkills evaluated. Install some with `openral rskill install <hub-id>` "
            "or pass `--rskills-dir rskills/`.[/dim]"
        )
    else:
        table = Table(title=f"rSkill compatibility for {robot_name}")
        table.add_column("repo_id", style="cyan bold")
        table.add_column("role")
        table.add_column("status")
        table.add_column("reason")
        for row in report.rows:
            if row.compatible:
                status = "[green]✓ compatible[/green]"
                reason = ""
            else:
                status = f"[red]✗ {row.failure_kind or 'fail'}[/red]"
                reason = row.reason or ""
            table.add_row(row.repo_id, row.role, status, reason)
        console.print(table)
    if report.incompatible:
        console.print(
            f"[yellow]{len(report.incompatible)} of {len(report.rows)} "
            "installed rSkill(s) cannot run on this host.[/yellow]"
        )


@rskill_app.command("check")
def rskill_check(
    rskill_id: str | None = typer.Argument(
        None,
        metavar="RSKILL_ID",
        help=(
            "rSkill to check — bare in-tree name, path (rskills/<name>), "
            "or HF Hub repo id (e.g. OpenRAL/rskill-smolvla-libero). "
            "Omit to walk every installed / in-tree rSkill (walk-all mode)."
        ),
    ),
    robot: Path = typer.Option(
        Path("robot.yaml"),
        "--robot",
        help="Path to a RobotDescription yaml (typically the output of `openral detect`).",
    ),
    rskills_dir: Path = typer.Option(
        Path("rskills"),
        "--rskills-dir",
        help=(
            "(Walk-all mode) in-tree rSkills directory to scan in addition to "
            "the installed registry. Skipped if it does not exist."
        ),
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output the CompatibilityReport as JSON."
    ),
) -> None:
    """Report whether one (or every) rSkill will run on the current host.

    Two modes:

    * ``openral rskill check <rskill_id>`` resolves the id the same way as
      ``openral rskill list`` / ``openral benchmark run --rskill <id>``
      (in-tree, installed registry, or HF Hub) and prints a per-section
      breakdown — embodiment, capability flags, GPU runtime, GPU dtype,
      sensors, actuators.
    * ``openral rskill check`` (no arg) walks every installed / in-tree
      rSkill via `openral_detect.check_installed_rskills` and prints a
      one-row-per-rSkill compatibility table.

    Exits 1 if any non-informational section fails (single-rSkill mode)
    or if any installed rSkill is incompatible (walk-all mode); exits 0
    otherwise.

    Example:
        >>> # openral rskill check smolvla-libero
        >>> # openral rskill check OpenRAL/rskill-smolvla-libero --robot /tmp/robot.yaml --json
        >>> # openral rskill check                                              # walk-all
    """
    import json as _json_mod

    from openral_core.schemas import RobotDescription
    from openral_detect import check_installed_rskills, check_single_rskill

    if not robot.exists():
        console.print(
            f"[red]Robot description not found:[/red] {robot}\n"
            "Run [bold]openral detect[/bold] first."
        )
        raise typer.Exit(code=1)

    description = RobotDescription.from_yaml(str(robot))

    if rskill_id is not None:
        single_report = check_single_rskill(rskill_id, description)
        single_row = single_report.rows[0]
        if json_output:
            console.print_json(_json_mod.dumps(single_report.model_dump(mode="json")))
        else:
            _render_single_rskill_table(single_row, description.name)
        if not single_row.compatible:
            raise typer.Exit(code=1)
        return

    # ── Walk-all (no-arg) ─────────────────────────────────────────────────────
    walk_dir = rskills_dir if rskills_dir.is_dir() else None
    report = check_installed_rskills(description, rskills_dir=walk_dir)
    if json_output:
        console.print_json(_json_mod.dumps(report.model_dump(mode="json")))
    else:
        _render_walk_all_table(report, description.name)

    if report.incompatible:
        raise typer.Exit(code=1)


_DEFAULT_OWNER = "your-org"
_DEFAULT_LICENSE = "apache-2.0"
_DEFAULT_EMBODIMENT = "franka_panda"


@rskill_app.command("new")
def rskill_new(
    rskill_id: str = typer.Argument(
        ...,
        metavar="ID",
        help="Local rSkill id, convention <policy>-<task> e.g. pi05-pick-cube.",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Destination directory. Defaults to rskills/<ID>/ under the cwd.",
    ),
    owner: str | None = typer.Option(
        None,
        "--owner",
        help="HF Hub owner segment for the manifest 'name' field.",
    ),
    license_: str | None = typer.Option(
        None,
        "--license",
        help=(
            "One of: apache-2.0 | mit | bsd | permissive_research | "
            "nvidia_non_commercial | proprietary | unknown."
        ),
    ),
    embodiment_tag: str | None = typer.Option(
        None,
        "--embodiment-tag",
        help="One of the canonical EmbodimentTag literals (see CLAUDE.md §6.4).",
    ),
    family: str | None = typer.Option(
        None,
        "--family",
        "-f",
        help=(
            "Policy family — one of act | smolvla | pi05 | xvla | diffusion. "
            "Sets model_family / chunk_size / dtype / latency budget from the "
            "matching in-tree reference manifest so a fresh ACT scaffold "
            "doesn't ship pi0.5 numbers. Inferred from --from-hf when set; "
            "interactively prompted otherwise."
        ),
    ),
    from_hf: str | None = typer.Option(
        None,
        "--from-hf",
        help=(
            "HF Hub repo id (e.g. 'Deepkar/libero-test-act' or "
            "'hf://Deepkar/libero-test-act'). Fetches the checkpoint's "
            "config.json, infers the family, and pre-fills chunk_size, "
            "sensors_required, state_contract.dim, image_preprocessing aliases, "
            "and weights_uri. Eliminates manual rewriting after scaffold."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip interactive prompts and accept the defaults (for scripting / CI).",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing destination directory instead of refusing.",
    ),
) -> None:
    """Scaffold a new local rSkill from ``rskills/template/``.

    Three modes:

    - ``--from-hf <owner/repo>`` — most intuitive. Fetches the
      checkpoint's ``config.json`` and pre-fills ``model_family`` /
      ``chunk_size`` / ``sensors_required`` / ``state_contract`` /
      ``image_preprocessing.aliases`` / ``weights_uri`` from the real
      Hub-side contract. No more hand-rewriting after scaffold.
    - ``--family <act|smolvla|pi05|xvla|diffusion>`` — family-aware
      defaults without Hub introspection. Use when you know the family
      but the weights live somewhere else (private repo, local mirror).
    - Neither — interactive. Prompts for ``--owner`` / ``--license`` /
      ``--embodiment-tag`` / ``--family`` (and offers ``--from-hf`` as
      a one-shot alternative). Pass ``--yes`` to skip all prompts and
      accept the defaults (your-org / apache-2.0 / franka_panda /
      no-family).

    The generated manifest is round-tripped through
    `RSkillManifest.from_yaml` and `rSkill.from_yaml`
    so a malformed scaffold fails at scaffold-time, not on first load.

    Example:
        >>> # openral rskill new act-libero --from-hf Deepkar/libero-test-act
        >>> # openral rskill new pi05-pick-cube --family pi05 --embodiment-tag franka_panda
        >>> # openral rskill new act-aloha-insertion --owner foo --embodiment-tag aloha
    """
    from typing import get_args

    from openral_core.schemas import EmbodimentTag, RSkillLicensePosture

    from openral_cli._rskill_scaffolder import scaffold_rskill

    valid_tags = list(get_args(EmbodimentTag))
    valid_licenses = [v.value for v in RSkillLicensePosture]

    resolved_owner = _resolve_or_prompt(
        owner,
        prompt=f"HF Hub owner (e.g. your username or org) [{_DEFAULT_OWNER}]",
        default=_DEFAULT_OWNER,
        skip_prompt=yes,
    )
    resolved_license = _resolve_or_prompt(
        license_,
        prompt=f"License posture [{_DEFAULT_LICENSE}]",
        default=_DEFAULT_LICENSE,
        skip_prompt=yes,
    )
    resolved_embodiment = _resolve_or_prompt(
        embodiment_tag,
        prompt=f"Embodiment tag (canonical robot id) [{_DEFAULT_EMBODIMENT}]",
        default=_DEFAULT_EMBODIMENT,
        skip_prompt=yes,
    )

    try:
        license_enum = RSkillLicensePosture(resolved_license)
    except ValueError as exc:
        console.print(
            f"[red]Invalid license:[/red] {resolved_license!r}. "
            f"Valid values: {', '.join(valid_licenses)}"
        )
        raise typer.Exit(code=1) from exc

    if resolved_embodiment not in valid_tags:
        console.print(
            f"[red]Invalid embodiment_tag:[/red] {resolved_embodiment!r}. "
            f"Valid values: {', '.join(valid_tags)}"
        )
        raise typer.Exit(code=1)

    resolved_family, intel_patch = _resolve_family_and_patch(
        family=family, from_hf=from_hf, yes=yes
    )

    resolved_out = out_dir if out_dir is not None else Path("rskills") / rskill_id

    try:
        result = scaffold_rskill(
            rskill_id,
            out_dir=resolved_out,
            owner=resolved_owner,
            license_=license_enum,
            embodiment_tag=cast(EmbodimentTag, resolved_embodiment),
            family=resolved_family,
            patch=intel_patch,
            overwrite=overwrite,
        )
    except ROSConfigError as exc:
        console.print(f"[red]Scaffold failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Scaffolded[/green] [cyan]{rskill_id}[/cyan] → {result}")
    if intel_patch is not None:
        console.print(
            "[dim]Next steps: edit description / README.md, optionally adjust "
            "image_preprocessing.flip_180 for your scene, add eval/<benchmark>.json "
            "results, then publish with tools/rskill_publisher.py.[/dim]"
        )
    else:
        console.print(
            "[dim]Next steps: edit rskill.yaml (weights_uri / chunk_size / "
            "sensors_required / image_preprocessing), update README.md, add "
            "eval/<benchmark>.json results, then publish with "
            "tools/rskill_publisher.py. Tip: pass `--from-hf <owner/repo>` next "
            "time to auto-fill the manifest from a published checkpoint.[/dim]"
        )


def _resolve_family_and_patch(
    *,
    family: str | None,
    from_hf: str | None,
    yes: bool,
) -> tuple[RSkillFamily | None, RSkillPatch | None]:
    """Resolve ``--family`` / ``--from-hf`` for ``openral rskill new``.

    Three paths in order of priority:

    1. ``--from-hf`` set → introspect the Hub config and derive both the
       family and a manifest patch carrying real chunk_size / sensors /
       state_contract / aliases / weights_uri.
    2. ``--family`` set → take its family defaults, no Hub call.
    3. neither + interactive → offer the menu. Skipped under ``--yes``
       so scripted callers keep the historical "no-family, template
       baseline" behaviour.

    Returns ``(resolved_family, patch)``; either or both may be ``None``.
    """
    from openral_cli._rskill_intel import (
        RSKILL_FAMILIES,
        introspect_hf,
    )

    typed_family: RSkillFamily | None = (
        cast("RSkillFamily", family) if family in RSKILL_FAMILIES else None
    )

    if from_hf is not None:
        try:
            resolved_family, intel_patch = introspect_hf(from_hf, default_family=typed_family)
        except ValueError as exc:
            console.print(f"[red]--from-hf failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        console.print(
            f"[green]Auto-detected[/green] family=[cyan]{resolved_family}[/cyan] "
            f"from [dim]{from_hf}[/dim]"
        )
        return resolved_family, intel_patch

    if family is not None:
        if family not in RSKILL_FAMILIES:
            console.print(
                f"[red]Invalid --family:[/red] {family!r}. "
                f"Valid values: {', '.join(RSKILL_FAMILIES)}"
            )
            raise typer.Exit(code=1)
        return family, None

    if yes:
        return None, None

    menu = " | ".join(RSKILL_FAMILIES)
    response = typer.prompt(
        f"Policy family [{menu}, empty for template baseline]",
        default="",
        show_default=False,
    ).strip()
    if not response:
        return None, None
    if response not in RSKILL_FAMILIES:
        console.print(
            f"[red]Invalid family:[/red] {response!r}. Valid values: {', '.join(RSKILL_FAMILIES)}"
        )
        raise typer.Exit(code=1)
    return cast("RSkillFamily", response), None


def _resolve_or_prompt(value: str | None, *, prompt: str, default: str, skip_prompt: bool) -> str:
    """Return ``value`` if provided, else prompt (or fall back to ``default``).

    Used by ``openral rskill new`` to drive the interactive prompts only when
    the user didn't pass the flag AND didn't request non-interactive
    mode with ``--yes``.
    """
    if value is not None:
        return value
    if skip_prompt:
        return default
    response: str = typer.prompt(prompt, default=default, show_default=False)
    return response


def _display_license_banner(
    name: str,
    license_value: str,
    version: str,
    con: Console,
) -> None:
    """Print a colour-coded license banner to the console.

    Args:
        name: rSkill name from the manifest.
        license_value: License posture value string.
        version: SemVer version string.
        con: Rich Console instance.
    """
    _permissive = {"apache-2.0", "mit", "bsd"}
    _warn = {"permissive_research", "unknown"}
    if license_value in _permissive:
        color, icon = "green", "✓"
    elif license_value in _warn:
        color, icon = "yellow", "!"
    else:
        color, icon = "red", "⚠"

    con.print(
        f"  [{color}]{icon} License:[/{color}] [bold]{license_value}[/bold]  ({name} v{version})"
    )


# ── sensor sub-app ────────────────────────────────────────────────────────────

sensor_app = typer.Typer(
    name="sensor",
    help="Sensor catalog browsing — list and inspect registered sensor specs.",
    no_args_is_help=True,
)
app.add_typer(sensor_app, name="sensor")


@sensor_app.command("list")
def sensor_list(
    vendor: str = typer.Option(
        "",
        "--vendor",
        help="Filter by vendor (lowercase, e.g. intel, orbbec, livox).",
    ),
    modality: str = typer.Option(
        "",
        "--modality",
        help="Filter by sensor modality (e.g. rgb, depth, lidar_2d, point_cloud).",
    ),
    kind: str = typer.Option(
        "",
        "--kind",
        help="Filter by kind: 'sensor' or 'bundle'.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON instead of a table.",
    ),
) -> None:
    """List every sensor registered in the openral sensor catalog.

    Example:
        >>> # openral sensor list
        >>> # openral sensor list --vendor intel
        >>> # openral sensor list --modality lidar_2d --json
    """
    # Imported lazily so `openral --help` doesn't pay the side-effect import cost.
    from openral_core.schemas import SensorModality
    from openral_sensors import CATALOG

    modality_enum: SensorModality | None = None
    if modality:
        try:
            modality_enum = SensorModality(modality)
        except ValueError:
            valid = ", ".join(m.value for m in SensorModality)
            console.print(f"[red]Unknown --modality {modality!r}.  Valid: {valid}[/red]")
            raise typer.Exit(code=1) from None

    kind_filter: str | None = None
    if kind:
        if kind not in ("sensor", "bundle"):
            console.print(f"[red]--kind must be 'sensor' or 'bundle', got {kind!r}.[/red]")
            raise typer.Exit(code=1)
        kind_filter = kind

    entries = CATALOG.filter(
        vendor=vendor or None,
        modality=modality_enum,
        kind=kind_filter,  # type: ignore[arg-type]  # reason: narrowed above
    )

    if json_output:
        payload = [
            {
                "id": e.id,
                "vendor": e.vendor,
                "model": e.model,
                "kind": e.kind,
                "modalities": [m.value for m in e.modalities],
                "description": e.description,
                "docs_url": e.docs_url,
            }
            for e in entries
        ]
        console.print_json(_json.dumps(payload))
        return

    if not entries:
        console.print("[yellow]No sensors match the requested filters.[/yellow]")
        return

    table = Table(title=f"openral sensor catalog ({len(entries)} entries)")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("kind")
    table.add_column("modalities", style="magenta")
    table.add_column("description")
    for e in entries:
        table.add_row(
            e.id,
            e.kind,
            ",".join(m.value for m in e.modalities),
            e.description,
        )
    console.print(table)


@sensor_app.command("show")
def sensor_show(
    sensor_id: str = typer.Argument(
        ...,
        metavar="SENSOR_ID",
        help="Catalog id, e.g. intel/realsense_d435i or slamtec/rplidar_a2",
    ),
    name: str = typer.Option(
        "sensor",
        "--name",
        help="Instance name passed to the factory (used as topic / frame prefix).",
    ),
    parent_frame: str = typer.Option(
        "base_link",
        "--parent-frame",
        help="tf2 parent frame for the static transform.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the resolved SensorSpec / SensorBundle as JSON.",
    ),
) -> None:
    """Resolve a catalog entry to a concrete ``SensorSpec`` / ``SensorBundle``.

    Example:
        >>> # openral sensor show intel/realsense_d435i --name head
        >>> # openral sensor show slamtec/rplidar_a2 --json
    """
    from openral_sensors import CATALOG

    try:
        entry = CATALOG.get(sensor_id)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(code=1) from None

    resolved = entry.factory(name=name, parent_frame=parent_frame)

    if json_output:
        console.print_json(resolved.model_dump_json(indent=2))
        return

    console.print(f"[bold cyan]{entry.id}[/bold cyan]  ({entry.kind})")
    console.print(f"  vendor      : [magenta]{entry.vendor}[/magenta]")
    console.print(f"  model       : [magenta]{entry.model}[/magenta]")
    console.print(f"  modalities  : {', '.join(m.value for m in entry.modalities)}")
    console.print(f"  description : [dim]{entry.description}[/dim]")
    if entry.docs_url:
        console.print(f"  docs_url    : [dim]{entry.docs_url}[/dim]")
    console.print()
    console.print("[bold]Resolved:[/bold]")
    console.print_json(resolved.model_dump_json(indent=2))


# ── openral benchmark ────────────────────────────────────────────────────────────

benchmark_app = typer.Typer(
    name="benchmark",
    help=(
        "Run a benchmark suite end-to-end (`openral benchmark run`), list available "
        "suites (`openral benchmark list`), or aggregate per-rSkill JSON results "
        "(`openral benchmark report`)."
    ),
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("list")
def benchmark_list(
    benchmarks_dir: Path = typer.Option(
        Path("benchmarks"),
        "--benchmarks-dir",
        help="Search directory for benchmark suite YAMLs.",
    ),
) -> None:
    """List every benchmark suite id available under ``benchmarks/*.yaml``.

    Each entry is a paste-able ``--suite`` value for ``openral benchmark run``.
    No rollout, no GPU.
    """
    if not benchmarks_dir.is_dir():
        console.print(f"[red]No benchmarks dir at {benchmarks_dir}[/red]")
        raise typer.Exit(code=1)
    suites = sorted(p.stem for p in benchmarks_dir.glob("*.yaml") if p.is_file())
    if not suites:
        print("<none>")
        return
    for suite in suites:
        print(suite)


@benchmark_app.command("run")
def benchmark_run(
    suite: str = typer.Option(
        ...,
        "--suite",
        help=(
            "Benchmark suite to evaluate — a bare ``list[BenchmarkScene]`` YAML "
            "(ADR-0042). Either a built-in id (resolved to "
            "`benchmarks/<id>.yaml`) or a direct YAML path."
        ),
    ),
    rskill: str = typer.Option(
        ...,
        "--rskill",
        help=(
            "rSkill reference — a bare name ('smolvla-libero'), a "
            "path ('rskills/smolvla-libero'), or an HF Hub repo id "
            "('OpenRAL/rskill-smolvla-libero'). "
            "Raw hf:// is rejected (weights must come from a manifest). "
            "The policy adapter id is read from the manifest's `model_family` "
            "field."
        ),
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help=(
            "Output path for the RSkillEvalResult JSON. Defaults to "
            "rskills/<dir>/eval/<suite_id>.json derived from the rSkill ref."
        ),
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Torch device override for the policy (cpu, cuda:0, mps, auto).",
    ),
    save_dir: Path | None = typer.Option(
        None,
        "--save-dir",
        help="Optional adapter-side artefact directory (videos, traces).",
    ),
    benchmarks_dir: Path = typer.Option(
        Path("benchmarks"),
        "--benchmarks-dir",
        help="Search directory for built-in benchmark suite YAMLs.",
    ),
    n_episodes: int | None = typer.Option(
        None,
        "--n-episodes",
        help=(
            "Override ``BenchmarkScene.n_episodes`` for every scene in the "
            "suite (lower for quick smoke runs). The published-protocol value "
            "lives in the suite YAML; this flag is for fast iteration, not "
            "for paper-comparison numbers."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Resolve the suite + VLA and print the planned (task x seed) "
            "matrix without running any rollouts. Useful in CI to "
            "validate config wiring."
        ),
    ),
    update_manifest: bool = typer.Option(
        True,
        "--update-manifest/--no-update-manifest",
        help=(
            "On success, write the avg_success_rate back into the rSkill "
            "manifest at `benchmarks.<suite_id>`. Surgical edit — "
            "preserves comments. Only fires for locally-resolvable rSkills. "
            "Disable for read-only paper-number runs."
        ),
    ),
    dashboard: bool = typer.Option(
        False,
        "--dashboard",
        help=(
            "Boot `openral dashboard` as a child process, point OTel at it, "
            "and shut it down on exit (same semantics as `openral sim run "
            "--dashboard`)."
        ),
    ),
    dashboard_port: int = typer.Option(
        4318,
        "--dashboard-port",
        help="Port for the spawned dashboard when --dashboard is set.",
    ),
) -> None:
    r"""Run a benchmark suite and write a validated `RSkillEvalResult` JSON.

    The runner iterates ``scenes x range(seed, seed + n_episodes)``,
    delegating each rollout to ``openral_sim.SimRunner`` so the
    rSkill compatibility check, OTel spans, and latency-budget reporting
    are identical to ``openral sim run``. Each :class:`BenchmarkScene`
    carries its own scene + task + robot (ADR-0041 / Task 10); ADR-0042
    deleted the ``BenchmarkSpec`` wrapper class so the suite is a bare
    list of scenes whose id is the YAML filename stem.

    Example:
        >>> # openral benchmark run --suite libero_spatial \\
        >>> #     --rskill smolvla-libero
    """
    scenes, suite_id = _resolve_benchmark_suite(suite, benchmarks_dir)
    vla_spec = _parse_rskill_cli_arg(rskill)

    # Apply --n-episodes override to every scene before dry-run or real run.
    if n_episodes is not None:
        scenes = [s.model_copy(update={"n_episodes": n_episodes}) for s in scenes]

    if dry_run:
        # Suite invariants (openral_core.raise_on_invalid_suite) guarantee
        # every BenchmarkScene shares robot_id / n_episodes / seed; read
        # from scenes[0] for the summary.
        first = scenes[0]
        eff_episodes = first.n_episodes
        # ``robot_id`` is non-None per raise_on_invalid_suite; coerce for printing.
        robot_id = first.robot_id or "<unset>"
        # When every scene shares one scene.id we print it; otherwise
        # show how many distinct scenes the suite covers.
        scene_ids = {scene.scene.id for scene in scenes}
        scene_summary = next(iter(scene_ids)) if len(scene_ids) == 1 else f"{len(scene_ids)} scenes"
        console.print(
            f"[cyan]suite[/cyan] {suite_id} — robot={robot_id} "
            f"scene={scene_summary} tasks={len(scenes)} "
            f"n_episodes={eff_episodes}"
        )
        console.print(f"[cyan]vla[/cyan]   id={vla_spec.id} weights={vla_spec.weights_uri}")
        console.print(
            f"[cyan]plan[/cyan]  {len(scenes) * eff_episodes} "
            f"episodes ({len(scenes)} tasks x {eff_episodes} reps)"
        )
        return

    out_path = out if out is not None else _default_benchmark_out_path(vla_spec, suite_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from openral_observability.dashboard import attached_dashboard
    from openral_sim.benchmark import run_benchmark

    # attached_dashboard is a no-op when enabled=False — wrap
    # unconditionally so the run_benchmark call is un-duplicated.
    with attached_dashboard(enabled=dashboard, port=dashboard_port):
        result, episodes = run_benchmark(
            scenes,
            suite_id=suite_id,
            vla=vla_spec,
            device=device,
            save_dir=str(save_dir) if save_dir is not None else None,
        )

    out_path.write_text(result.model_dump_json(indent=2))
    avg = result.results.get("avg_success_rate", 0.0)
    console.print(
        f"[green]wrote {out_path}[/green] — avg success "
        f"= {float(avg) if isinstance(avg, (int, float)) else avg:.3f} "
        f"over {len(episodes)} episodes"
    )

    if update_manifest:
        from openral_rskill.loader import resolve_rskill_local_dir
        from openral_sim.benchmark import update_rskill_benchmarks

        # Resolve to the in-tree dir so the surgical write hits
        # rskills/<name>/rskill.yaml even when the user supplied a bare
        # name or a Hub-style repo id. Falls through to a cwd-relative
        # path for Hub-only references with no in-tree shim — the
        # update is then a no-op (FileNotFoundError handled below).
        local_dir = resolve_rskill_local_dir(vla_spec.weights_uri)
        skill_dir = str(local_dir) if local_dir is not None else vla_spec.weights_uri

        try:
            manifest_path = update_rskill_benchmarks(
                skill_dir,
                suite_id,
                float(avg) if isinstance(avg, (int, float)) else 0.0,
            )
            console.print(
                f"[green]updated {manifest_path}[/green] — "
                f"benchmarks.{suite_id} = "
                f"{float(avg) if isinstance(avg, (int, float)) else avg:.3f}"
            )
        except FileNotFoundError as exc:
            console.print(
                f"[yellow]skipped manifest update:[/yellow] {exc} (eval JSON was still written)"
            )


def _resolve_benchmark_suite(
    suite: str,
    benchmarks_dir: Path,
) -> tuple[list[BenchmarkScene], str]:
    """Map a ``--suite`` argument to a validated ``(scenes, suite_id)`` tuple.

    ADR-0042: a benchmark suite is a bare ``list[BenchmarkScene]`` YAML;
    the suite id is the filename stem. Accepts either a built-in id
    (resolved to ``benchmarks/<id>.yaml``) or a direct path. Bare ids
    that don't resolve raise ``typer.BadParameter`` listing the catalogue
    entries that ARE present so typos are easy to fix. Per-scene Pydantic
    validation and suite-level invariant checks (uniformity, uniqueness,
    non-empty) run here; any failure surfaces as a ``typer.BadParameter``
    so the CLI exit code stays informative.
    """
    from openral_core import load_benchmark_suite, raise_on_invalid_suite
    from openral_core.exceptions import ROSConfigError

    candidate = Path(suite)
    if candidate.suffix in {".yaml", ".yml"} or candidate.exists():
        if not candidate.exists():
            raise typer.BadParameter(f"benchmark suite YAML not found: {candidate}")
        resolved_path = candidate
    else:
        resolved_path = benchmarks_dir / f"{suite}.yaml"
        if not resolved_path.exists():
            available = (
                sorted(p.stem for p in benchmarks_dir.glob("*.yaml") if p.is_file())
                if benchmarks_dir.is_dir()
                else []
            )
            raise typer.BadParameter(
                f"unknown benchmark suite {suite!r}; "
                f"available in {benchmarks_dir}/: {available if available else '<empty>'}"
            )

    suite_id = resolved_path.stem
    try:
        scenes = load_benchmark_suite(str(resolved_path))
        raise_on_invalid_suite(scenes, suite_id=suite_id)
    except ROSConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return scenes, suite_id


def _parse_rskill_cli_arg(raw: str) -> VLASpec:
    """Parse ``--rskill <reference>`` into a `VLASpec`.

    Accepts a bare rSkill reference — a name (``smolvla-libero``),
    a path (``rskills/smolvla-libero``), or an HF repo id
    (``OpenRAL/rskill-smolvla-libero``). The adapter id is read from
    the loaded manifest's ``model_family`` field.
    """
    from openral_core import VLASpec
    from openral_core.exceptions import ROSConfigError
    from openral_rskill.loader import _validate_skill_ref, load_rskill_manifest

    try:
        uri = _validate_skill_ref(raw)
    except ROSConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    manifest = load_rskill_manifest(uri)
    return VLASpec(id=manifest.model_family, weights_uri=uri)


def _default_benchmark_out_path(vla_spec: VLASpec, suite_id: str) -> Path:
    """Derive ``rskills/<vla>/eval/<suite_id>.json`` from a VLASpec + suite id.

    Resolves the rSkill to its in-tree directory via
    :func:`openral_rskill.loader.resolve_rskill_local_dir` so the JSON
    lands in the right place regardless of which URI form the user typed
    (bare name, ``rskills/<name>``, Hub repo id). Falls back to the
    library's :func:`openral_sim.benchmark.default_output_path` when no
    in-tree shim exists (Hub-only references).
    """
    from openral_rskill.loader import resolve_rskill_local_dir
    from openral_sim.benchmark import default_output_path

    local_dir = resolve_rskill_local_dir(vla_spec.weights_uri)
    if local_dir is not None:
        return local_dir / "eval" / f"{suite_id}.json"
    return Path(default_output_path(vla_spec.weights_uri, suite_id))


@benchmark_app.command("scene")
def benchmark_scene(
    config: Path = typer.Option(
        ...,
        "--config",
        help=(
            "Path to a BenchmarkScene YAML "
            "(`scenes/benchmark/<id>.yaml`). DeployScene and SimScene "
            "YAMLs are rejected with a redirect — `openral benchmark "
            "scene` accepts BenchmarkScene only (scene + task + "
            "`n_episodes` + `seed` + `metadata.paper` + "
            "`metadata.honest_scope`)."
        ),
    ),
    rskill: str = typer.Option(
        ...,
        "--rskill",
        help=(
            "rSkill reference — a bare name ('smolvla-libero'), a "
            "path ('rskills/smolvla-libero'), or an HF Hub repo id "
            "('OpenRAL/rskill-smolvla-libero'). "
            "Raw hf:// is rejected (weights must come from a manifest). "
            "The policy adapter id is read from the manifest's `model_family` "
            "field."
        ),
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help=(
            "Output path for the RSkillEvalResult JSON. Defaults to "
            "rskills/<dir>/eval/scene_<scene_id>.json derived from the "
            "rSkill ref."
        ),
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Torch device override for the policy (cpu, cuda:0, mps, auto).",
    ),
    save_dir: Path | None = typer.Option(
        None,
        "--save-dir",
        help="Optional adapter-side artefact directory (videos, traces).",
    ),
    n_episodes: int | None = typer.Option(
        None,
        "--n-episodes",
        help=(
            "Override `BenchmarkScene.n_episodes` (lower for quick smoke "
            "runs). The published-protocol value lives in the YAML; this "
            "flag is for fast iteration, not for paper-comparison numbers."
        ),
    ),
    view: bool | None = typer.Option(
        None,
        "--view/--no-view",
        help=(
            "Open a passive mujoco.viewer window and stream the rollout in "
            "real time (parity with `openral sim run --view`). Default "
            "(unset): headless — benchmark eval artefacts and CI/deploy "
            "runs are unaffected. Pass --view to require a window (errors "
            "loud if unsupported), or --no-view to force offscreen. "
            "Incompatible with MUJOCO_GL=egl."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Resolve the scene + rSkill and print the planned (task x "
            "seed) matrix without running any rollouts. Useful in CI to "
            "validate config wiring."
        ),
    ),
    update_manifest: bool = typer.Option(
        True,
        "--update-manifest/--no-update-manifest",
        help=(
            "On success, write the avg_success_rate back into the rSkill "
            "manifest at `benchmarks.<scene_id>`. Surgical edit — "
            "preserves comments. Only fires for locally-resolvable rSkills."
        ),
    ),
    write_eval: bool = typer.Option(
        True,
        "--write-eval/--no-write-eval",
        help=(
            "Persist the RSkillEvalResult JSON to --out (default "
            "rskills/<dir>/eval/scene_<scene_id>.json, a tracked path). "
            "Pass --no-write-eval for a fully non-mutating smoke run: the "
            "rollout still executes and prints its score, but nothing is "
            "written to the rSkill package (implies --no-update-manifest)."
        ),
    ),
    dashboard: bool = typer.Option(
        False,
        "--dashboard",
        help=(
            "Boot `openral dashboard` as a child process, point OTel at it, "
            "and shut it down on exit (same semantics as `openral sim run "
            "--dashboard`)."
        ),
    ),
    dashboard_port: int = typer.Option(
        4318,
        "--dashboard-port",
        help="Port for the spawned dashboard when --dashboard is set.",
    ),
) -> None:
    r"""Run a single-scene benchmark and write a validated `RSkillEvalResult` JSON.

    Single-scene sibling of ``openral benchmark run --suite`` — accepts
    exactly one :class:`BenchmarkScene` YAML and emits the same eval JSON
    shape so ``openral benchmark report`` does not need to distinguish
    the two entrypoints.

    Example:
        >>> # openral benchmark scene \\
        >>> #   --config scenes/benchmark/pusht.yaml \\
        >>> #   --rskill diffusion-pusht
    """
    from openral_core import BenchmarkScene, load_scene_strict

    scene = load_scene_strict(str(config), BenchmarkScene)
    if n_episodes is not None:
        scene = scene.model_copy(update={"n_episodes": n_episodes})

    if dry_run:
        # Dry-run validates config wiring only — do not touch the Hub or
        # load weights. Print the raw --rskill argument as-typed.
        console.print(
            f"[cyan]scene[/cyan] {scene.scene.id} — robot={scene.robot_id} "
            f"task={scene.task.id} n_episodes={scene.n_episodes} "
            f"seed={scene.seed}"
        )
        console.print(f"[cyan]vla[/cyan]   rskill={rskill}")
        console.print(
            f"[cyan]plan[/cyan]  {scene.n_episodes} episodes (seeds "
            f"{scene.seed}..{scene.seed + scene.n_episodes - 1})"
        )
        return

    vla_spec = _parse_rskill_cli_arg(rskill)
    out_path = out if out is not None else _default_benchmark_scene_out_path(vla_spec, scene)

    from openral_observability.dashboard import attached_dashboard
    from openral_sim.benchmark import run_benchmark_scene

    with attached_dashboard(enabled=dashboard, port=dashboard_port):
        result, episodes = run_benchmark_scene(
            scene,
            vla_spec,
            device=device,
            save_dir=str(save_dir) if save_dir is not None else None,
            config_path=str(config),
            view=view,
        )

    avg = result.results.get("avg_success_rate", 0.0)
    avg_f = float(avg) if isinstance(avg, (int, float)) else 0.0
    if _persist_scene_eval(result, out_path, write_eval=write_eval):
        console.print(
            f"[green]wrote {out_path}[/green] — avg success "
            f"= {avg_f:.3f} over {len(episodes)} episodes"
        )
    else:
        console.print(
            f"[yellow]--no-write-eval:[/yellow] not persisting result — avg success "
            f"= {avg_f:.3f} over {len(episodes)} episodes (nothing written to the rSkill)"
        )

    if not write_eval:
        # Non-mutating smoke run: skip the manifest writeback too.
        return

    if update_manifest and not _scene_id_is_benchmark_suite(scene.scene.id):
        # The rskill.yaml `benchmarks:` block holds canonical SUITE headlines
        # (RSkillManifest.benchmarks is keyed by the BenchmarkName literal).
        # A single scene whose id is not itself a suite id (e.g. 'metaworld',
        # 'robocasa/PickPlaceCounterToCabinet') has no headline slot — writing
        # it would raise ROSConfigError. The per-scene result is already
        # captured in the eval JSON, so we skip the manifest write rather than
        # crash. (Scenes whose id IS a suite id — pusht, libero_spatial — still
        # update the headline below.)
        console.print(
            f"[yellow]skipped manifest update:[/yellow] scene id {scene.scene.id!r} "
            f"is not a canonical benchmark suite id; per-scene result written to "
            f"{out_path} only (rskill.yaml benchmarks: holds suite headlines)."
        )
    elif update_manifest:
        from openral_rskill.loader import resolve_rskill_local_dir
        from openral_sim.benchmark import update_rskill_benchmarks

        local_dir = resolve_rskill_local_dir(vla_spec.weights_uri)
        skill_dir = str(local_dir) if local_dir is not None else vla_spec.weights_uri
        try:
            manifest_path = update_rskill_benchmarks(
                skill_dir,
                scene.scene.id,
                float(avg) if isinstance(avg, (int, float)) else 0.0,
            )
            console.print(
                f"[green]updated {manifest_path}[/green] — "
                f"benchmarks.{scene.scene.id} = "
                f"{float(avg) if isinstance(avg, (int, float)) else avg:.3f}"
            )
        except FileNotFoundError as exc:
            console.print(
                f"[yellow]skipped manifest update:[/yellow] {exc} (eval JSON was still written)"
            )


def _scene_id_is_benchmark_suite(scene_id: str) -> bool:
    """True iff ``scene_id`` is a canonical ``BenchmarkName`` suite id.

    ``openral benchmark scene`` only writes ``rskill.yaml``'s ``benchmarks:``
    block (the suite-headline map keyed by the ``BenchmarkName`` literal) when
    the scene's id IS one of those suite ids — e.g. ``"pusht"``,
    ``"libero_spatial"``. Arbitrary single-scene ids such as ``"metaworld"``
    (suite is ``"metaworld_mt50"``) or ``"robocasa/PickPlaceCounterToCabinet"``
    have no headline slot, so the manifest write is skipped (the per-scene
    eval JSON still records the result).
    """
    from typing import get_args

    from openral_core import BenchmarkName

    return scene_id in set(get_args(BenchmarkName))


def _persist_scene_eval(result: RSkillEvalResult, out_path: Path, *, write_eval: bool) -> bool:
    """Persist a benchmark-scene ``RSkillEvalResult`` to ``out_path``.

    Returns ``True`` if the file was written, ``False`` when ``write_eval``
    is ``False`` (the ``--no-write-eval`` non-mutating smoke-run mode — the
    rollout still runs, but nothing touches the tracked rSkill package).
    """
    if not write_eval:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2))
    return True


def _default_benchmark_scene_out_path(vla_spec: VLASpec, scene: BenchmarkScene) -> Path:
    """Derive ``rskills/<vla>/eval/scene_<scene_id>.json`` from a VLASpec.

    Mirrors :func:`_default_benchmark_out_path` for the single-scene
    entrypoint. The ``scene_`` prefix distinguishes per-scene JSONs from
    multi-task suite JSONs so both can coexist under the same rSkill.
    """
    from openral_rskill.loader import resolve_rskill_local_dir
    from openral_sim.benchmark import default_output_path

    local_dir = resolve_rskill_local_dir(vla_spec.weights_uri)
    if local_dir is not None:
        return local_dir / "eval" / f"scene_{scene.scene.id}.json"
    return Path(default_output_path(vla_spec.weights_uri, f"scene_{scene.scene.id}"))


@benchmark_app.command("report")
def benchmark_report(
    rskills_dir: Path = typer.Option(
        Path("rskills"),
        "--rskills-dir",
        help="Directory containing rSkill packages (each with optional eval/*.json).",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit a JSON dump instead of the rich-table summary.",
    ),
) -> None:
    """Walk every ``<skill>/eval/*.json`` and print a benchmark roll-up.

    Validates each JSON against `openral_core.RSkillEvalResult`
    (the same schema the rSkill loader uses at install time) so a rotted
    file fails loudly instead of silently being skipped.

    Example:
        >>> # openral benchmark report
        >>> # openral benchmark report --json > /tmp/report.json
    """
    from openral_core import RSkillEvalResult
    from pydantic import ValidationError

    if not rskills_dir.is_dir():
        console.print(f"[red]rskills directory not found: {rskills_dir}[/red]")
        raise typer.Exit(code=1)

    rows: list[dict[str, object]] = []
    for skill_dir in sorted(p for p in rskills_dir.iterdir() if p.is_dir()):
        eval_dir = skill_dir / "eval"
        if not eval_dir.is_dir():
            continue
        for json_path in sorted(eval_dir.glob("*.json")):
            try:
                result = RSkillEvalResult.from_json(str(json_path))
            except (ValidationError, _json.JSONDecodeError) as exc:
                console.print(f"[red]invalid {json_path}:[/red] {exc}")
                raise typer.Exit(code=1) from exc
            rows.append(
                {
                    "rskill": skill_dir.name,
                    "benchmark": result.benchmark.name,
                    "robot": result.benchmark.robot,
                    "simulator": result.benchmark.simulator,
                    "reproduced_locally": result.source.reproduced_locally,
                    "model_variant": result.source.model_variant,
                    "status": result.source.status,
                    "results": result.results,
                    "path": (
                        str(json_path.relative_to(Path.cwd()))
                        if json_path.is_relative_to(Path.cwd())
                        else str(json_path)
                    ),
                }
            )

    if json_output:
        console.print_json(_json.dumps(rows, indent=2, default=str))
        return

    if not rows:
        console.print(f"[yellow]no rskills/<id>/eval/*.json files under {rskills_dir}[/yellow]")
        return

    rows.sort(key=lambda r: (str(r["benchmark"]), str(r["rskill"])))
    table = Table(title=f"rSkill benchmark report — {len(rows)} entries")
    table.add_column("Benchmark", style="cyan")
    table.add_column("rSkill", style="magenta")
    table.add_column("Variant", style="dim")
    table.add_column("Robot", style="dim")
    table.add_column("Repro local?", justify="center")
    table.add_column("Headline result", style="green")
    table.add_column("Status", style="dim")
    for row in rows:
        results = row["results"]
        headline = _summarize_results(results) if isinstance(results, dict) else "—"
        table.add_row(
            str(row["benchmark"]),
            str(row["rskill"]),
            str(row["model_variant"]),
            str(row["robot"]),
            "✓" if row["reproduced_locally"] else "✗",
            headline,
            str(row["status"] or ""),
        )
    console.print(table)


def _summarize_results(results: dict[str, object]) -> str:
    """Produce a one-line headline from a freeform ``results`` block.

    Picks ``*_avg`` keys first, then falls back to a single-numeric value
    or a status string. Returns ``"—"`` when nothing summarisable is found.
    """
    avg_keys = [k for k in results if k.endswith("_avg") or k == "avg"]
    if avg_keys:
        v = results[avg_keys[0]]
        if isinstance(v, (int, float)):
            return f"avg = {v:.3f}"
        if isinstance(v, dict) and "success_rate" in v:
            sr = v["success_rate"]
            if isinstance(sr, (int, float)):
                return f"avg success = {sr:.3f}"
    numeric_keys = [
        k for k, v in results.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if len(numeric_keys) == 1:
        v = results[numeric_keys[0]]
        return f"{numeric_keys[0]} = {v:.3f}" if isinstance(v, (int, float)) else "—"
    if "status" in results and isinstance(results["status"], str):
        return f"status: {results['status']}"
    return "—"


# ── sim sub-app ───────────────────────────────────────────────────────────────
#
# Mounts the ``openral sim`` Typer group exported by ``openral_sim.cli`` so
# users can invoke the sim eval runner as ``openral sim run …``.
#
# Lazy-import discipline: importing `openral_sim.cli` at module load is
# light (only the Typer option metadata + a couple of pydantic / structlog
# imports). The heavy sim dependencies (torch, mujoco, gymnasium, lerobot)
# load inside `openral_sim.runner` and the per-adapter modules under
# `openral_sim.policies/backends`, which `_run()` imports lazily.
# `tests/unit/test_cli_eval.py::test_bh_cli_import_is_light` guards this.
app.add_typer(sim_app, name="sim")

# ADR-0021 — `openral install <group>` post-install escape hatch for the
# Tier-0 curl-bash installer (`scripts/install.sh`). The base install puts
# `openral` on $PATH with the CLI's own thin runtime; sim physics, LIBERO,
# MetaWorld, RoboCasa, and the sudo+apt ROS 2 bootstrap layer in on demand.
app.add_typer(install_app, name="install")

# ADR-0019 PR5: `openral dataset push` (publish a LeRobotDataset v3 to the HF Hub).
# Importing `dataset` at module top is cheap; the `push` command itself lazy-
# imports huggingface_hub only when actually publishing so `openral --help` stays
# sub-second.
app.add_typer(dataset_app, name="dataset")

# ADR-0030: `openral collision lower|check` — offline URDF/SRDF → manifest
# self-collision model. The `lower_robot` import is deferred inside the commands
# (it pulls yourdfpy/trimesh) so `openral --help` stays fast.
app.add_typer(collision_app, name="collision")

# ADR-0058: `openral robot vendor-urdf <id>` — expand an upstream xacro to a
# flat, committed URDF so end users need no xacro tooling at runtime. The
# `vendor_urdf` import is deferred inside the command (it pulls robot_descriptions/
# xacrodoc/yourdfpy) so `openral --help` stays fast.
robot_app = typer.Typer(
    name="robot",
    help="Robot description assets — vendor a flat URDF from an upstream xacro (ADR-0058).",
    no_args_is_help=True,
)
app.add_typer(robot_app, name="robot")


@robot_app.command("vendor-urdf")
def robot_vendor_urdf(
    robot_id: str = typer.Argument(
        ...,
        help="OpenRAL robot id; names the output file (e.g. ur5e → ur5e.urdf).",
    ),
    upstream: str = typer.Option(
        ...,
        "--upstream",
        help=(
            "Upstream source: 'rd:<robot_descriptions module>' (xacro, expanded "
            "via xacrodoc) or 'file:<path>' to an already-flat URDF."
        ),
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output directory; '<robot_id>.urdf' is written here.",
    ),
    rename: list[str] | None = typer.Option(
        None,
        "--rename",
        help=(
            "Joint-name rename as 'PATTERN=>REPL' (regex re.sub). Repeatable — "
            "applied in order (so100/so101 take 6, gr1/h1 take 1). Defaults to "
            "the per-robot rule (openarm strips its 'openarm_' prefix)."
        ),
    ),
    raw_text: bool = typer.Option(
        False,
        "--raw-text/--no-raw-text",
        help=(
            "Copy an already-flat upstream URDF verbatim and apply --rename to "
            "the raw XML (no yourdfpy round-trip), preserving package:// mesh "
            "paths byte-for-byte (so100/so101/gr1/h1)."
        ),
    ),
) -> None:
    """Expand an upstream description to a flat, committed URDF (ADR-0058)."""
    from openral_cli.robot import vendor_urdf

    rename_pairs: list[tuple[str, str]] | None = None
    if rename:
        rename_pairs = []
        for spec in rename:
            if "=>" not in spec:
                raise typer.BadParameter("--rename must be 'PATTERN=>REPL'", param_hint="--rename")
            pat, _, repl = spec.partition("=>")
            rename_pairs.append((pat, repl))
    written = vendor_urdf(
        robot_id, upstream=upstream, out_dir=out, rename=rename_pairs, raw_text=raw_text
    )
    typer.echo(f"Wrote {written}")


# ADR-0018 F10: `openral prompt "do X"` publishes a one-shot PromptStamped
# onto /openral/prompt_in/cli; the prompt_router_node fans it out to
# /openral/prompt for the F4 reasoner. rclpy import is deferred inside
# the command body so `openral --help` stays sub-second.
app.command(
    name="prompt",
    help=(
        "Publish a one-shot operator prompt to the prompt-router (ADR-0018 F10). "
        "Requires a sourced ROS 2 install."
    ),
)(prompt_command)


# ── openral dashboard — live debugging UI over the OTel stream (issue #44) ──────


@app.command(
    "dashboard",
    help=(
        "Serve a live debugging dashboard (read-only) at the given port. "
        "The same port also acts as an OTLP/HTTP receiver, so any "
        "`openral sim run` / `openral deploy run` pointed at "
        "OTEL_EXPORTER_OTLP_ENDPOINT=http://<host>:<port> + "
        "OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf streams in live. "
        "Works without Jaeger/Tempo running."
    ),
)
def dashboard(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address. Loopback by default; no auth.",
    ),
    port: int = typer.Option(
        4318,
        "--port",
        help=(
            "HTTP port; serves UI, /api/state, /api/stream, and OTLP/HTTP "
            "receiver. Defaults to 4318 (the OTLP/HTTP standard port) "
            "rather than 8000 (issue #132) — `mkdocs serve` and most "
            "FastAPI demos already squat on 8000."
        ),
    ),
    log_level: str = typer.Option(
        "warning",
        "--log-level",
        help="uvicorn log level (debug | info | warning | error).",
    ),
    inprocess: str | None = typer.Option(
        None,
        "--inprocess",
        help=(
            "Optional shell-quoted command to spawn as a child process with "
            "OTEL_EXPORTER_OTLP_ENDPOINT pointed at this dashboard. Pass the "
            "whole command as one string (shlex-tokenised), e.g. "
            "`--inprocess 'openral sim run --config scenes/benchmark/pusht.yaml"
            " --rskill diffusion-pusht'`."
        ),
    ),
) -> None:
    """Serve the OpenRAL live dashboard."""
    import shlex

    from openral_observability.dashboard import run_dashboard

    inprocess_cmd = shlex.split(inprocess) if inprocess else None
    run_dashboard(
        host=host,
        port=port,
        inprocess_cmd=inprocess_cmd,
        log_level=log_level,
    )


# ── openral deploy {run, list} ───────────────────────────────────────────────────

deploy_app = typer.Typer(
    name="deploy",
    help=(
        "Hardware deploy — run an rSkill on a real robot (or digital twin) "
        "per a RobotEnvironment YAML (`openral deploy run`), or list available "
        "robot configs (`openral deploy list`)."
    ),
    no_args_is_help=True,
)
app.add_typer(deploy_app, name="deploy")

deploy_app.command(
    "sim",
    help=(
        "Boot the full ROS graph (dashboard + safety_kernel + reasoner + "
        "prompt_router + runtime + HAL) against a digital-twin HAL, driven "
        "by a DeployScene YAML + rSkill. Sibling of ``deploy run``."
    ),
)(deploy_sim_command)


@deploy_app.command("list")
def deploy_list() -> None:
    """List every robot config under `deployments/*.yaml`.

    Each entry is a paste-able `--config` path for `openral deploy run`.
    No hardware touch, no GPU.
    """
    from openral_rskill.loader import _find_repo_root_from

    repo_root = _find_repo_root_from(Path(__file__))
    if repo_root is None:
        console.print("[red]Could not locate repo root.[/red]")
        raise typer.Exit(code=1)
    robot_examples = repo_root / "deployments"
    if not robot_examples.is_dir():
        print("<none>")
        return
    configs = sorted(robot_examples.rglob("*.yaml"))
    if not configs:
        print("<none>")
        return
    for cfg in configs:
        print(cfg.relative_to(repo_root))


@deploy_app.command("run")
def deploy_run(
    config: Path = typer.Option(  # reason: typer Option idiom
        ...,
        "--config",
        "-c",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Path to a RobotEnvironment YAML; its robot_id + hal.transport drive the launch.",
    ),
    robot: str | None = typer.Option(
        None,
        "--robot",
        help="Override the robot_id resolved from --config.",
    ),
    hal: list[str] | None = typer.Option(
        None,
        "--hal",
        help="Override HAL node params, key=value (repeatable), e.g. --hal port=/dev/ttyUSB1.",
    ),
    dashboard: bool = typer.Option(
        True,
        "--dashboard/--no-dashboard",
        help="Spawn the live dashboard (default on).",
    ),
    dashboard_port: int = typer.Option(
        4318,
        "--dashboard-port",
        help="Dashboard OTLP port.",
    ),
) -> None:
    """Run an rSkill on REAL hardware via the production ROS graph (ADR-0032).

    Unlike `openral deploy sim`, this drives the **real** hardware HAL: it
    resolves the robot from `--config` (a RobotEnvironment) and shells the SAME
    `sim_e2e.launch.py` graph with `hal_mode:=real` — the HAL lifecycle node +
    C++ safety kernel + reasoner + world state (+ SLAM/Nav2 when the robot
    declares a lidar). The HAL's `connect()` fails loudly if no hardware is
    attached; a simulation-only robot raises ROSCapabilityMismatch (use
    `openral deploy sim`). The robot's `hal.transport` (serial `port` /
    `robot_ip` / `fci_ip`) is forwarded as HAL node params; `--hal` wins.
    """
    from openral_core import RobotEnvironment  # reason: defer schema import
    from openral_core.exceptions import ROSCapabilityMismatch  # reason: defer

    from openral_cli.deploy_sim import (  # reason: defer heavy CLI import
        _parse_hal_overrides,
        resolve_launch_invocation,
        run_launch_invocation,
    )

    try:
        env = RobotEnvironment.from_yaml(str(config))
    except (FileNotFoundError, ROSConfigError) as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # RobotEnvironment.hal.transport (serial port / robot_ip / fci_ip) + params
    # become HAL node param overrides; an explicit --hal key=value wins.
    overrides: dict[str, object] = {**env.hal.transport, **env.hal.params}
    overrides.update(_parse_hal_overrides(hal))

    try:
        invocation = resolve_launch_invocation(
            config=None,
            robot_override=robot or env.robot_id,
            dashboard_port=dashboard_port,
            reset_to_pose_service=None,
            hal_param_overrides=overrides,
            hal_mode="real",
            enable_dashboard=dashboard,
        )
    except (ROSConfigError, ROSCapabilityMismatch) as exc:
        console.print(f"[red]deploy run:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[cyan]deploy run[/cyan] {invocation.robot_id} → real HAL "
        f"(hal_mode=real); the HAL's connect() requires the robot to be attached."
    )
    returncode = run_launch_invocation(invocation)
    raise typer.Exit(code=returncode)


# ── openral replay — bag↔OTel correlator (ADR-0018 F7) ──────────────────────────


def _resolve_frame_trace_id(frame_spec: str, dataset_root: Path) -> str:
    """Resolve a ``<repo_id>/<episode>/<frame>`` spec to its stored trace_id.

    ``repo_id`` itself contains a slash (``org/name``); the episode and
    frame indices are the last two ``/``-separated fields, so we split
    from the right. Exits non-zero with a typed message on a malformed
    spec, a missing frame, or a frame that carries no trace.
    """
    from openral_dataset import read_frame_trace

    try:
        _repo_id, ep_str, frame_str = frame_spec.rsplit("/", 2)
        episode_idx = int(ep_str)
        frame_idx = int(frame_str)
    except ValueError:
        console.print(
            f"[red]openral replay:[/red] malformed --frame {frame_spec!r}; "
            "expected '<repo_id>/<episode>/<frame>' (e.g. 'openral/dataset-pick/0/12')"
        )
        raise typer.Exit(code=2) from None

    try:
        trace_id, _span_id = read_frame_trace(
            root=dataset_root, episode_idx=episode_idx, frame_idx=frame_idx
        )
    except ROSConfigError as exc:
        console.print(f"[red]openral replay:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    if not trace_id:
        console.print(
            f"[red]openral replay:[/red] frame {frame_spec} carries no trace_id "
            "(its producing tick had no active OTel span); nothing to pivot to"
        )
        raise typer.Exit(code=2)
    return trace_id


@app.command(
    "replay",
    help=(
        "Join a rosbag2/.mcap file with OTel spans from the live dashboard "
        "(ADR-0018 F7). Prints a chronological JSON timeline keyed by trace_id; "
        "writes to `--out` when given. `--dashboard` may be omitted for a "
        "bag-only timeline."
    ),
)
def replay(
    bag: Path = typer.Argument(  # reason: typer Argument idiom
        ...,
        exists=True,
        readable=True,
        help="Path to a rosbag2 directory or a bare .mcap file.",
    ),
    trace: str | None = typer.Option(
        None,
        "--trace",
        help="32-hex-char trace_id to filter on. Defaults to the busiest one in the bag.",
    ),
    frame: str | None = typer.Option(
        None,
        "--frame",
        help=(
            "Pivot from a written LeRobotDataset frame: '<repo_id>/<episode>/<frame>' "
            "(e.g. 'openral/dataset-pick/0/12'). Reads that frame's trace_id and uses "
            "it as the join key. Requires --dataset-root; mutually exclusive with --trace."
        ),
    ),
    dataset_root: Path | None = typer.Option(
        None,
        "--dataset-root",
        help="Root directory of the LeRobotDataset that --frame refers to.",
    ),
    dashboard: str | None = typer.Option(
        None,
        "--dashboard",
        help="Dashboard base URL (e.g. http://127.0.0.1:8000). Omit for bag-only.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Write the timeline JSON to this file; print to stdout when omitted.",
    ),
) -> None:
    """Read a bag, join it with spans by trace_id, emit a JSON timeline."""
    from openral_observability.replay.cli import run_replay, write_timeline

    # ISSUE-109 pivot — resolve --frame into a concrete trace_id off the
    # dataset before the join. Done here (not in run_replay) so the
    # openral_observability replay module stays free of the lerobot dep.
    if frame is not None:
        if trace is not None:
            console.print("[red]openral replay:[/red] --frame and --trace are mutually exclusive")
            raise typer.Exit(code=2)
        if dataset_root is None:
            console.print("[red]openral replay:[/red] --frame requires --dataset-root")
            raise typer.Exit(code=2)
        trace = _resolve_frame_trace_id(frame, dataset_root)

    result = run_replay(bag_path=bag, trace_id=trace, dashboard_url=dashboard)
    if out is not None:
        write_timeline(result, out)
        console.print(
            f"[green]openral replay:[/green] wrote {len(result.timeline)} entries to {out}"
        )
        if result.trace_id:
            console.print(f"trace_id: {result.trace_id}")
        return
    print(_json.dumps(result.to_json(), indent=2, sort_keys=False))


# ── openral record — wrap `ros2 bag record` with profile presets (ADR-0018 F7) ──


@app.command(
    "record",
    help=(
        "Spawn `ros2 bag record` for the ADR-0018 graph with a slim/full profile. "
        "Requires a sourced ROS 2 install. Use `--dry-run` to print the argv "
        "instead of executing."
    ),
)
def record(
    out: Path = typer.Option(  # reason: typer Option idiom
        ...,
        "--out",
        "-o",
        help="Output directory passed to `ros2 bag record -o`.",
    ),
    profile: str = typer.Option(
        "slim",
        "--profile",
        help="Recording profile: 'slim' (default) or 'full'.",
    ),
    storage: str = typer.Option(
        "mcap",
        "--storage",
        help="rosbag2 storage backend; mcap is the openral default.",
    ),
    extra_topic: list[str] = typer.Option(
        [],
        "--extra-topic",
        help="Additional topic to record verbatim. Repeatable.",
    ),
    extra_regex: list[str] = typer.Option(
        [],
        "--extra-regex",
        help="Additional regex to OR into --regex. Repeatable.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the composed argv instead of executing.",
    ),
) -> None:
    """Wrap `ros2 bag record` with ADR-0018 F7's slim/full topic presets."""
    from openral_observability.replay.cli import run_record

    if profile not in {"slim", "full"}:
        console.print(f"[red]openral record:[/red] unknown profile {profile!r}; expected slim|full")
        raise typer.Exit(code=2)
    try:
        argv, completed = run_record(
            profile=profile,  # type: ignore[arg-type] # reason: validated above against the literal set
            output_dir=out,
            storage=storage,
            extra_topics=extra_topic,
            extra_regex=extra_regex,
            dry_run=dry_run,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]openral record:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if dry_run:
        print(" ".join(argv))
        return
    assert completed is not None
    if completed.returncode != 0:
        raise typer.Exit(code=completed.returncode)


# ── openral profile session — LTTng opt-in profiling (ADR-0018 F9) ──────────────

profile_app = typer.Typer(
    name="profile",
    help="Microsecond-accurate profiling via ros2_tracing / LTTng (ADR-0018 F9).",
    no_args_is_help=True,
)
app.add_typer(profile_app, name="profile")


@profile_app.command(
    "session",
    help=(
        "Start, stop, or view an LTTng session for the realtime hot path. "
        "Requires lttng-tools on PATH. Set OPENRAL_ROS2_TRACING=1 on the "
        "agent process to emit tracepoints; the env var is the runtime gate."
    ),
)
def profile_session(
    action: str = typer.Argument(  # reason: typer Argument idiom
        ...,
        help="One of: start | stop | view.",
    ),
    output: Path = typer.Option(  # reason: typer Option idiom
        Path("./lttng-traces"),
        "--output",
        "-o",
        help="LTTng output directory. Used by start (write here) and view (read from here).",
    ),
    name: str = typer.Option(
        "openral",
        "--name",
        "-n",
        help="LTTng session name.",
    ),
) -> None:
    """Drive an LTTng session — start, stop, view."""
    from openral_observability.tracing_lttng import (
        LttngSessionError,
        start_session,
        stop_session,
        view_session,
    )

    try:
        if action == "start":
            session = start_session(name=name, output_dir=output)
            console.print(
                f"[green]openral profile session start:[/green] "
                f"{session.name} → {session.output_dir}"
            )
            console.print(
                "Run your workload with OPENRAL_ROS2_TRACING=1, then "
                "`openral profile session stop` to flush."
            )
        elif action == "stop":
            stop_session(name=name)
            console.print(f"[green]openral profile session stop:[/green] {name}")
        elif action == "view":
            view_session(output_dir=output)
        else:
            console.print(
                f"[red]openral profile session:[/red] unknown action {action!r}; "
                "expected start | stop | view"
            )
            raise typer.Exit(code=2)
    except LttngSessionError as exc:
        console.print(f"[red]openral profile session:[/red] {exc}")
        raise typer.Exit(code=1) from exc
