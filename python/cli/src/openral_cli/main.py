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
behavior serve      Serve an rSkill to the official BEHAVIOR evaluator.
calibrate camera    Calibrate a camera sensor using ros2 camera_calibration.
install             Install opt-in dependency groups (sim, ros, libero, …).
rskill search       Find installable rSkills on the OpenRAL HF Hub org.
rskill install      Download an rSkill from the HF Hub and register it locally.
rskill list         List all locally installed rSkills.
rskill check        Report which installed rSkills will run on the current host.
rskill new          Scaffold a new local rSkill from rskills/template/.
collision lower     Lower a robot's URDF/SRDF into its self-collision model.
collision check     Fail if a manifest drifts from its lowered collision model.
check               Cross-validate every robot/skill/scene manifest in one pass.

Run ``openral --help`` for full usage.
"""

from __future__ import annotations

import contextlib
import json as _json
import os
import platform
import re
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
from rich.box import MINIMAL, ROUNDED
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from openral_cli.check import check_command
from openral_cli.collision import collision_app
from openral_cli.dataset import dataset_app
from openral_cli.deploy_sim import deploy_sim_command
from openral_cli.install import install_app
from openral_cli.prompt import prompt_command

if TYPE_CHECKING:
    from openral_core import (
        BenchmarkScene,
        RobotDescription,
        RSkillEvalResult,
        SensorSpec,
        VLASpec,
    )
    from openral_core.schemas import RSkillManifest
    from openral_detect import CompatibilityReport, DetectionReport, RSkillCompatRow
    from openral_detect.report import GpuProbeResult

    from openral_cli._rskill_intel import RSkillFamily, RSkillPatch

app = typer.Typer(
    name="openral",
    help="OpenRAL — open-source robot agent harness for rSkill / VLA models",
    invoke_without_command=True,
)
console = Console()

# ── REPL ──────────────────────────────────────────────────────────────────────

# Solid-block OpenRAL logo mark (single white weight, no gradient), sized to the
# wordmark's 6 rows: horns flaring out and down into a rounded head, eyes below.
_LOGO_ART: Final[str] = "\n".join(
    [
        "█             █",
        "██▄         ▄██",
        "████▄▄   ▄▄████",
        "▀██████ ██████▀",
        "   ▀███████▀   ",
        " ▀   ▀███▀   ▀ ",
    ]
)
# OPENRAL block-letter wordmark.
_WORDMARK_ART: Final[str] = "\n".join(
    [
        " ██████╗ ██████╗ ███████╗███╗   ██╗██████╗  █████╗ ██╗",
        "██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗██╔══██╗██║",
        "██║   ██║██████╔╝█████╗  ██╔██╗ ██║██████╔╝███████║██║",
        "██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██╗██╔══██║██║",
        "╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║██║  ██║███████╗",
        " ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝",
    ]
)
_TAGLINE_TAIL: Final[str] = " — Open Robot Agentic Layer (harness) for embodied AI"
_CAPABILITIES: Final[str] = "fast policies · slow reasoning · rewards · perception · control"
# Community links rendered in the top-right cell.
_LINKS: Final[tuple[tuple[str, str], ...]] = (
    ("Discord", "discord.gg/3paXT2bVyB"),
    ("GitHub", "github.com/OpenRAL/openral"),
    ("Hugging Face", "huggingface.co/OpenRAL"),
    ("Website", "openral.com"),
)
# Quick-start commands rendered in the bottom-right cell.
_COMMANDS: Final[tuple[tuple[str, str], ...]] = (
    ("doctor", "diagnose your host setup"),
    ("rskill search", "find installable skills"),
    ("help", "list every command"),
    ("exit", "leave the repl · Ctrl-D"),
)

# Minimum terminal columns each (content-sized) layout occupies — measured from
# the rendered box so the richest layout that still fits the terminal is chosen and
# the box never overflows. Below the side-by-side floor the logo stacks above the
# wordmark; below the stacked width the terminal is simply too narrow to fit.
_WIDE_MIN: Final[int] = 127  # two-column (logo|wordmark · links/divider/commands)
_SIDE_BY_SIDE_MIN: Final[int] = 82  # single column, logo + wordmark share a line


def _kv_grid(rows: tuple[tuple[str, str], ...], key_style: str) -> Table:
    """A two-column ``key  value`` grid (styled key, dim value) with no borders."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=key_style, no_wrap=True)
    grid.add_column(style="dim")
    for key, value in rows:
        grid.add_row(key, value)
    return grid


def _logo_wordmark(*, stacked: bool) -> RenderableType:
    """Logo mark + OPENRAL wordmark, side by side (wide) or stacked (narrow)."""
    logo = Text(_LOGO_ART, style="bold white")
    wordmark = Text(_WORDMARK_ART, style="bold white")
    if stacked:
        return Group(logo, wordmark)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(vertical="middle")
    grid.add_column(vertical="middle")
    grid.add_row(logo, wordmark)
    return grid


def _identity(*, stacked: bool) -> RenderableType:
    """Logo + wordmark above the centred tagline and capability strip."""
    return Group(
        _logo_wordmark(stacked=stacked),
        Text(""),
        Text.assemble(("OpenRAL", "bold white"), (_TAGLINE_TAIL, "white"), justify="center"),
        Text(_CAPABILITIES, style="dim", justify="center"),
    )


def render_banner(version_str: str, *, width: int | None = None) -> RenderableType:
    """Build the REPL welcome box as a rich renderable (Claude-Code style).

    Returns a :class:`rich.panel.Panel` so the same renderable can be printed to
    the live console *and* exported to plain text in tests, independent of
    terminal/TTY/colour state. The white-bordered rounded box carries ``OPENRAL
    v<version>`` inline in the top border and adapts to ``width``:

    * **Wide** (``>= _WIDE_MIN``): two columns — the logo mark beside the OPENRAL
      wordmark with the tagline below on the left; the community links above a
      horizontal divider above the quick-start commands on the right, split by a
      vertical divider.
    * **Narrow**: a single stacked column keeping every section; the logo sits
      beside the wordmark while it fits and stacks above it once it does not.

    Args:
        version_str: The installed ``openral-cli`` version, rendered as ``vX.Y.Z``.
        width: Target terminal width in columns; defaults to a wide layout.

    Example:
        >>> from io import StringIO
        >>> from rich.console import Console
        >>> con = Console(file=StringIO(), width=120)
        >>> con.print(render_banner("0.1.0", width=120))
        >>> "OPENRAL v0.1.0" in con.file.getvalue()
        True
    """
    cols = width if width is not None else _WIDE_MIN
    links = _kv_grid(_LINKS, "bold white")
    commands = _kv_grid(_COMMANDS, "bold cyan")

    body: RenderableType
    if cols >= _WIDE_MIN:
        columns = Table(
            box=MINIMAL,
            show_header=False,
            show_edge=False,
            show_lines=False,
            pad_edge=False,
            padding=(0, 2),
            border_style="dim",
            expand=False,
        )
        columns.add_column(vertical="top")
        columns.add_column(vertical="top")
        columns.add_row(
            _identity(stacked=False),
            Group(links, Rule(style="dim"), commands),
        )
        body = columns
    else:
        body = Group(
            _identity(stacked=cols < _SIDE_BY_SIDE_MIN),
            Text(""),
            links,
            Rule(style="dim"),
            commands,
        )

    # Size the box to its content (expand=False) rather than stretching it to the
    # terminal: a compact box keeps slack on wide terminals and only breaks if the
    # window is later dragged narrower than the box itself (printed output cannot
    # reflow). The layout above is chosen so the content always fits ``cols``.
    return Panel(
        body,
        box=ROUNDED,
        border_style="white",
        title=f"OPENRAL v{version_str}",
        title_align="left",
        padding=(1, 2),
        expand=False,
    )


def _cli_version() -> str:
    """Best-effort ``openral-cli`` version string for the banner (never raises)."""
    with contextlib.suppress(PackageNotFoundError):
        return version("openral-cli")
    return "0.0.0"


def _print_banner() -> None:
    """Print the OpenRAL welcome box, sized to the live terminal width."""
    console.print(render_banner(_cli_version(), width=console.width))


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
    "behavior": semconv.RUN_MODE_SIM,
    "benchmark": semconv.RUN_MODE_BENCHMARK,
    "deploy": semconv.RUN_MODE_HARDWARE,
    "connect": semconv.RUN_MODE_HARDWARE,
}

# Hardware deployments at >=100 Hz over 24 h would emit millions of tick spans
# per day at ALWAYS_ON. The 2026-05-17 sampling-policy amendment calls for a
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
    per the 2026-05-17 sampling-policy amendment. ``OPENRAL_OTEL_SAMPLE_RATIO``
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
        # `absent`, not `missing`: a Tier-0 install (the curl-bash installer)
        # deliberately ships no ROS 2 — it takes no sudo and touches no apt.
        # Reporting this as fatal made `curl … | bash && openral doctor` exit 1
        # on a *correct* install, which breaks scripted/CI use of the documented
        # quick-start. ROS 2 is an opt-in escalation, so point at it instead.
        results.append(
            CheckResult("ROS 2 binary", "absent", "not found — run: openral install ros")
        )
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
    # `absent` when unavailable for the same reason as the ROS 2 binary above:
    # colcon arrives with the ROS 2 bootstrap, which Tier-0 does not run.
    path = shutil.which("colcon")
    if path:
        return CheckResult("colcon", "ok", path)
    return CheckResult("colcon", "absent", "not found — run: openral install ros")


def _check_gpu(result: GpuProbeResult, warnings: list[str]) -> list[CheckResult]:
    """Return one row per detected GPU / SoC accelerator.

    Args:
        result: Pre-probed :class:`~openral_detect.GpuProbeResult` (shared
            with :func:`_check_compute_spec` so the probe runs exactly once).
        warnings: Non-fatal probe warnings to surface when no GPU is found.
    """
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


def _check_compute_spec(result: GpuProbeResult) -> list[CheckResult]:
    """Build :class:`~openral_core.ComputeSpec` rows from the GPU probe.

    Shares the same :class:`~openral_detect.GpuProbeResult` already obtained
    by :func:`_check_gpu` so no second probe is issued.  The assembled
    ``ComputeSpec`` mirrors exactly what ``openral detect`` would write into
    ``RobotDescription.compute_edge`` / ``compute_local`` — doctor and detect
    stay in sync.

    Tier labelling:
    - Jetson SoC detected → rows prefixed ``ComputeSpec (edge)``
    - Discrete NVIDIA / Apple Silicon / CPU-only → ``ComputeSpec (local)``

    Rows emitted per tier:
    - **runtimes** — comma-separated list of supported runtimes.
    - **dtypes** — comma-separated list of supported quantization dtypes.
    - **cuMotion** — ok when ``supports_cumotion()`` is True, info otherwise.
    - **NVMM** — ok/absent for zero-copy NVMM availability.
    """
    import datetime

    from openral_detect import build_compute_spec
    from openral_detect.report import DetectionReport, GpuProbeResult

    def _spec_rows(gpu: GpuProbeResult, tier: str) -> list[CheckResult]:
        report = DetectionReport(
            detected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            gpu=gpu,
        )
        spec = build_compute_spec(gpu, report)
        prefix = f"ComputeSpec ({tier})"
        rows: list[CheckResult] = []

        runtimes_str = ", ".join(r.value for r in spec.gpu_supported_runtimes) or "none"
        rows.append(CheckResult(f"{prefix} / runtimes", "info", runtimes_str))

        dtypes_str = ", ".join(d.value for d in spec.gpu_supported_dtypes) or "none"
        rows.append(CheckResult(f"{prefix} / dtypes", "info", dtypes_str))

        cumotion = spec.supports_cumotion()
        rows.append(
            CheckResult(
                f"{prefix} / cuMotion",
                "ok" if cumotion else "info",
                "supported" if cumotion else "not supported (needs Ampere+, CUDA≥13, ≥8 GB VRAM)",
            )
        )

        rows.append(
            CheckResult(
                f"{prefix} / NVMM",
                "ok" if spec.nvmm_available else "absent",
                "zero-copy NVMM available" if spec.nvmm_available else "not available",
            )
        )
        return rows

    all_rows: list[CheckResult] = []
    if result.jetson is not None:
        gpu_edge = GpuProbeResult(jetson=result.jetson, backend="jtop")
        all_rows.extend(_spec_rows(gpu_edge, "edge"))
    if result.nvidia or result.apple_silicon or result.jetson is None:
        gpu_local = GpuProbeResult(
            nvidia=result.nvidia,
            apple_silicon=result.apple_silicon,
            backend=result.backend,
        )
        all_rows.extend(_spec_rows(gpu_local, "local"))
    return all_rows


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
    if path:
        return CheckResult("just", "ok", path)
    # …and on a Tier-0 install there is no checkout, so there are no recipes for
    # `just` to run: warning about it is noise. Only a host with a Justfile
    # nearby is actually missing something.
    if not any((Path.cwd() / name).is_file() for name in ("Justfile", "justfile")):
        return CheckResult("just", "info", "not installed (no Justfile here — nothing to run)")
    return CheckResult("just", "warn", "not found")


def _cosmos_autostart_enabled() -> bool:
    """Mirror the reasoner client's OPENRAL_COSMOS3_AUTOSTART parsing.

    Kept local (like the base-URL table above) so `openral doctor` never
    imports the optionally-installed reasoner package; accepts the same
    falsy spellings the client does.
    """
    value = os.environ.get("OPENRAL_COSMOS3_AUTOSTART", "").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _is_local_base_url(url: str) -> bool:
    """Return True when ``url``'s host resolves to a loopback name."""
    host = urlparse(url).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _probe_tcp(host: str, port: int, *, timeout_s: float = 0.2) -> bool:
    """Return True if a TCP connection to ``host:port`` succeeds quickly."""
    with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=timeout_s):
        return True
    return False


def _resolve_reasoner_endpoint(entry: object, override: str) -> str:
    """Resolve a curated model's effective endpoint for the doctor row.

    ``managed`` → the managed loopback (``:8901``, mirroring the Cosmos
    sidecar default so doctor never imports the reasoner package); a set
    ``default_endpoint`` verbatim; ``None`` → the Anthropic API default.
    An explicit ``override`` always wins.
    """
    from openral_core import REASONER_MANAGED_ENDPOINT  # local import: flat doctor graph

    if override:
        return override
    default = getattr(entry, "default_endpoint", None)
    if default == REASONER_MANAGED_ENDPOINT:
        return "http://127.0.0.1:8901/v1"
    if default:
        return str(default)
    return "https://api.anthropic.com"


def _reasoner_endpoint_probe_row(
    label: str, base_url: str, *, managed: bool, autostart: bool
) -> CheckResult:
    """One loopback probe row, generic over hosting (ADR-0088).

    A down endpoint is `info` only for a managed model with autostart on (the
    client spawns the server on the first tick); a down managed endpoint with
    autostart disabled, or any down BYO/local endpoint, is a real `warn`.
    """
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (8901 if managed else 8000)
    if _probe_tcp(host, port):
        return CheckResult(label, "ok", f"endpoint reachable at {host}:{port}")
    if managed and autostart:
        return CheckResult(
            label,
            "info",
            f"endpoint unreachable at {host}:{port} — auto-starts on the first "
            "reasoner tick (managed vLLM sidecar); pre-warm with "
            "`python tools/cosmos3_reasoner_sidecar.py`.",
        )
    if managed:
        return CheckResult(
            label,
            "warn",
            f"endpoint unreachable at {host}:{port} — OPENRAL_COSMOS3_AUTOSTART is "
            "disabled; start your server (`python tools/cosmos3_reasoner_sidecar.py` "
            "or your own vLLM / NIM).",
        )
    return CheckResult(
        label,
        "warn",
        f"endpoint unreachable at {host}:{port} — start your local reasoner server.",
    )


def _check_reasoner_model(model_key: str) -> list[CheckResult]:
    """Model-first reasoner rows (ADR-0088).

    Resolve the registry entry and check each real precondition (curated? auth?
    endpoint reachable / managed?).
    """
    # Local imports: keep the doctor import graph flat. The preset table is
    # the SAME object the factory uses (openral_core, no reasoner-package
    # import needed) — a hand-mirrored copy here drifted twice (2fe732a,
    # 131a489: doctor rejected valid named endpoints, then passed a dialect
    # clash the factory refuses).
    from openral_core import REASONER_ENDPOINT_PRESETS, REASONER_MODELS

    curated = sorted(REASONER_MODELS)
    entry = REASONER_MODELS.get(model_key)
    endpoint_override = os.environ.get("OPENRAL_REASONER_ENDPOINT", "").strip()
    api_key = os.environ.get("OPENRAL_REASONER_API_KEY", "").strip()
    key_status = "set" if api_key else "unset"

    # A named endpoint carries its own URL, dialect and auth posture, so it has
    # to resolve here exactly as it does in the factory — otherwise doctor
    # reports `ENDPOINT=ollama` as invalid config that runs fine.
    preset = REASONER_ENDPOINT_PRESETS.get(endpoint_override.lower())
    if preset is not None:
        endpoint_override = preset.url

    if entry is None:
        dialect = os.environ.get("OPENRAL_REASONER_DIALECT", "").strip().lower()
        if preset is not None:
            dialect = dialect or preset.dialect
        if not endpoint_override or dialect not in {"anthropic", "openai"}:
            return [
                CheckResult(
                    "Reasoner LLM",
                    "fail",
                    f"OPENRAL_REASONER_MODEL={model_key!r} is not a curated model "
                    f"({', '.join(curated)}); set OPENRAL_REASONER_ENDPOINT to a named "
                    f"endpoint ({', '.join(sorted(REASONER_ENDPOINT_PRESETS))}) or to a "
                    "URL plus OPENRAL_REASONER_DIALECT (anthropic|openai) to use an "
                    "uncurated endpoint.",
                )
            ]
        hatch_rows = [
            CheckResult(
                "Reasoner LLM",
                "warn",
                f"model={model_key} (uncurated) dialect={dialect} "
                f"endpoint={endpoint_override} api_key={key_status} — untested for "
                "robotics tool calling.",
            )
        ]
        if preset is not None and preset.auth_required and not api_key:
            hatch_rows.append(
                CheckResult(
                    "Reasoner API_KEY",
                    "missing",
                    "OPENRAL_REASONER_API_KEY unset — required for "
                    f"OPENRAL_REASONER_ENDPOINT={endpoint_override}.",
                )
            )
        if _is_local_base_url(endpoint_override):
            hatch_rows.append(
                _reasoner_endpoint_probe_row(
                    "Reasoner endpoint", endpoint_override, managed=False, autostart=False
                )
            )
        return hatch_rows

    if preset is not None and preset.dialect != entry.dialect:
        # The factory refuses this outright — a named endpoint cannot re-dialect
        # a curated model. Reporting `ok` here would tell the operator their
        # config is fine right up until the reasoner refuses to configure.
        return [
            CheckResult(
                "Reasoner LLM",
                "fail",
                f"OPENRAL_REASONER_ENDPOINT speaks the {preset.dialect!r} dialect but model "
                f"{model_key!r} speaks {entry.dialect!r}; a named endpoint cannot "
                "re-dialect a curated model. Use a URL for a proxy that translates.",
            )
        ]

    endpoint = _resolve_reasoner_endpoint(entry, endpoint_override)
    # A named endpoint states its own auth posture; a bare URL means the operator
    # owns the endpoint (local vLLM, proxy-owned auth), so auth is not forced there.
    auth_required = (
        preset.auth_required
        if preset is not None
        else bool(entry.auth_required) and not (endpoint_override and entry.dialect == "openai")
    )
    summary = (
        f"model={model_key} dialect={entry.dialect} hosting={entry.hosting} "
        f"endpoint={endpoint} api_key={key_status}"
    )
    rows: list[CheckResult] = []
    incomplete: list[CheckResult] = []
    if auth_required and not api_key:
        incomplete.append(
            CheckResult(
                "Reasoner API_KEY",
                "missing",
                f"OPENRAL_REASONER_API_KEY unset — required for model {model_key!r}.",
            )
        )
    rows.append(CheckResult("Reasoner LLM", "warn" if incomplete else "ok", summary))
    rows.extend(incomplete)

    if _is_local_base_url(endpoint):
        managed = entry.hosting == "managed_local"
        autostart = _cosmos_autostart_enabled() if managed else False
        label = "Cosmos 3" if managed else "Reasoner endpoint"
        rows.append(
            _reasoner_endpoint_probe_row(label, endpoint, managed=managed, autostart=autostart)
        )
    return rows


def _check_reasoner_llm() -> list[CheckResult]:
    """Reasoner LLM doctor rows, model-first (ADR-0088).

    Reads ``OPENRAL_REASONER_MODEL`` and resolves the curated registry entry.
    The API key value is never printed — only ``set`` / ``unset``.
    """
    from openral_core import REASONER_MODELS  # local import: keep doctor import graph flat

    model_key = os.environ.get("OPENRAL_REASONER_MODEL", "").strip()
    curated = ", ".join(sorted(REASONER_MODELS))
    if not model_key:
        return [
            CheckResult(
                "Reasoner LLM",
                "absent",
                f"OPENRAL_REASONER_MODEL unset — set a curated model ({curated}); "
                "see packages/openral_reasoner_ros/README.md.",
            )
        ]
    return _check_reasoner_model(model_key)


def _gather_checks() -> list[CheckResult]:
    """Run all checks and return the combined result list.

    The GPU probe is issued exactly once and shared between
    :func:`_check_gpu` (hardware rows) and :func:`_check_compute_spec`
    (derived ``ComputeSpec`` rows) so ``openral doctor`` and
    ``openral detect`` build the same ``ComputeSpec`` from the same data.

    Returns:
        List of `CheckResult` in display order.
    """
    from openral_detect.probes import probe_gpus

    checks: list[CheckResult] = []
    checks.append(_check_python())
    checks.append(_check_platform())
    checks.append(_check_openral_core())
    checks.extend(_check_ros2())
    checks.append(_check_colcon())
    gpu_warnings: list[str] = []
    gpu_result = probe_gpus(warnings=gpu_warnings)
    checks.extend(_check_gpu(gpu_result, gpu_warnings))
    checks.extend(_check_compute_spec(gpu_result))
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
    robot_type: str | None = typer.Option(
        None,
        "--robot",
        "--as",
        help="Force the canonical base manifest (slug e.g. 'so100' or dir name "
        "'so100_follower'), overriding USB/DDS inference. A bare Feetech "
        "plug-in defaults to the SO-101; use this to select the SO-100 (the "
        "two are indistinguishable over USB).",
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
        "Choices: usb, can, dds, gpu, cameras_v4l2, cameras_realsense, network.",
    ),
    no_write: bool = typer.Option(
        False, "--no-write", help="Print summary and skip writing robot.yaml"
    ),
    deployment: Path | None = typer.Option(
        None,
        "--deployment",
        help="Also scaffold a DeployScene YAML at this path (robot_id + "
        "`sensors:` bindings from the camera wizard; safety left to the "
        "robot manifest). Paste-able as `openral deploy run --config`.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Overwrite existing file without prompting"
    ),
) -> None:
    """Probe the host and emit a complete RobotDescription robot.yaml.

    Runs the auto-provisioning flow from ``openral_detect``:

    1. Probe USB / CAN / DDS / GPU / V4L2 / RealSense / network.
    2. Identify the rig (DDS topology, SocketCAN interface names, or USB
       VID/PID match — in that order).  If a
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

    # Probe-only inspection short-circuits the builder (keeps CI / --report
    # non-interactive). Building a manifest is always interactive.
    if no_write:
        try:
            canonical = assemble_robot_description(
                detection, force_robot_type=robot_type, enrich_cameras=True
            )
        except ROSConfigError as exc:
            console.print(f"[red]detect:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        console.print("\n[dim]--no-write set — printing yaml to stdout:[/dim]\n")
        console.print(
            _yaml.safe_dump(
                canonical.model_dump(mode="json"), sort_keys=False, default_flow_style=False
            )
        )
        if deployment is not None:
            console.print(
                "[yellow]--deployment ignored under --no-write — no files written.[/yellow]"
            )
        return

    try:
        canonical = assemble_robot_description(
            detection, force_robot_type=robot_type, enrich_cameras=False
        )
    except ROSConfigError as exc:
        console.print(f"[red]detect:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    default_name = str(canonical.name)
    robot_name = _prompt_robot_name(default_name)

    robot_sensors, scene_sensor_specs = _run_camera_binding_wizard(canonical, detection)
    description = canonical.model_copy(
        update={"name": robot_name, "sensors": robot_sensors}, deep=True
    )
    description = _maybe_customize_limits(description)

    if output == Path("robot.yaml"):
        output = Path("robots") / robot_name / "robot.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)

    from openral_detect.registry import canonical_robot_path

    # Mirrors `openral_detect.assemble._infer_bh_robot_type` — DDS, then CAN,
    # then USB — so the asset-relocation base matches the manifest that was
    # actually loaded.
    inferred = (
        robot_type
        or detection.ros2.inferred_robot_type
        or next((m.bh_robot_type for m in detection.can.matches if m.bh_robot_type), None)
        or next((m.bh_robot_type for m in detection.usb.matches if m.bh_robot_type), None)
    )
    canon_path = canonical_robot_path(inferred) if inferred else None
    canonical_dir = canon_path.parent if canon_path is not None else None
    description = _relocate_file_assets(
        description, canonical_dir=canonical_dir, output_path=output
    )

    yaml_text = _yaml.safe_dump(
        description.model_dump(mode="json"),
        sort_keys=False,
        default_flow_style=False,
    )

    if output.exists() and not yes:
        overwrite = typer.confirm(f"{output} already exists. Overwrite?", default=False)
        if not overwrite:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=0)

    output.write_text(yaml_text, encoding="utf-8")
    console.print(f"\n[green]Wrote[/green] {output} (RobotDescription, {description.name})")
    console.print(f"[dim]Next step:[/dim] openral rskill check --robot {output}")

    if deployment is not None:
        _write_deploy_scene_scaffold(
            deployment, description, scene_sensor_specs, detection=detection, assume_yes=yes
        )


def _relocate_file_assets(
    description: RobotDescription, *, canonical_dir: Path | None, output_path: Path
) -> RobotDescription:
    """Rewrite ``file:`` asset refs to repo-root-relative for a relocated manifest.

    ``file:<rel>`` resolves against the manifest dir then the repo root; a
    relocated manifest loses the manifest-dir hit, so we repoint each ref at
    ``file:<repo-root-relative path to the canonical file>`` (which the repo-root
    fallback resolves). Mesh paths inside the URDF keep resolving from the
    canonical file's own location because we point at it in place, not copy it.
    """
    from openral_core.assets import _REPO_ROOT  # reason: match the resolver's root exactly

    if canonical_dir is None or canonical_dir.resolve() == output_path.parent.resolve():
        return description

    root = _REPO_ROOT.resolve()

    def _rewrite(ref: str | None) -> str | None:
        if not ref or not ref.startswith("file:"):
            return ref
        abs_path = (canonical_dir / ref[len("file:") :]).resolve()
        try:
            return f"file:{abs_path.relative_to(root)}"
        except ValueError:
            return f"file:{abs_path}"  # canonical dir outside repo → absolute

    assets = description.assets
    new_urdf = (
        assets.urdf.model_copy(update={"ref": _rewrite(assets.urdf.ref)})
        if assets.urdf is not None
        else None
    )
    new_assets = assets.model_copy(
        update={"urdf": new_urdf, "mjcf": _rewrite(assets.mjcf), "srdf": _rewrite(assets.srdf)}
    )
    return description.model_copy(update={"assets": new_assets})


_ROBOT_NAME_RE = re.compile(r"[a-z0-9_]+")


def _prompt_robot_name(default: str) -> str:
    """Prompt for the custom rig name written into robots/<name>/robot.yaml.

    Constrained to the slug convention every in-tree ``robots/<name>/`` dir
    already uses (lowercase letters, digits, underscores) so the answer can't
    escape the ``robots/`` tree via ``/`` or break the ``robot_id`` used by
    ``deploy run`` / ``rskill check`` via spaces or other punctuation.
    Re-prompts on an invalid answer instead of writing it through.

    ``default`` itself isn't guaranteed slug-safe — an unrecognised rig's
    scaffolded name embeds the raw hostname (e.g. ``unknown_my-laptop``),
    which can carry a hyphen. Sanitize it before offering it so accepting the
    default with a bare Enter always succeeds instead of re-prompting.
    """
    if not _ROBOT_NAME_RE.fullmatch(default):
        default = re.sub(r"[^a-z0-9_]+", "_", default.lower()).strip("_") or "robot"
    while True:
        name = typer.prompt("Robot name (writes robots/<name>/robot.yaml)", default=default).strip()
        name = name or default
        if _ROBOT_NAME_RE.fullmatch(name):
            return name
        console.print(
            f"  [yellow]{name!r} invalid — use lowercase letters, digits, underscores "
            f"(e.g. my_so101_bench).[/yellow]"
        )


def _prompt_float(label: str, default: float) -> float:
    """Enter-to-keep-default float prompt; non-numeric input keeps the default."""
    raw = typer.prompt(f"    {label}", default=str(default), show_default=True).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        console.print(f"    [yellow]not a number — keeping {default}[/yellow]")
        return default


def _maybe_customize_limits(description: RobotDescription) -> RobotDescription:
    """Opt-in per-joint limit + safety-scalar override; declining inherits canonical.

    Every ``JointSpec`` field (``position_limits``/``velocity_limit``/
    ``effort_limit``) is Optional — a canonical manifest may leave any of them
    unset. Only prompt for a field when the canonical value is present; a
    ``None`` field is carried through unchanged (never coerced to a number).
    """
    if not typer.confirm("Customize joint limits & safety envelope?", default=False):
        return description

    new_joints = []
    for j in description.joints:
        update: dict[str, object] = {}
        if j.position_limits is not None:
            lo = _prompt_float(f"{j.name} position min", j.position_limits[0])
            hi = _prompt_float(f"{j.name} position max", j.position_limits[1])
            update["position_limits"] = (lo, hi)
        if j.velocity_limit is not None:
            update["velocity_limit"] = _prompt_float(f"{j.name} velocity_limit", j.velocity_limit)
        if j.effort_limit is not None:
            update["effort_limit"] = _prompt_float(f"{j.name} effort_limit", j.effort_limit)
        new_joints.append(j.model_copy(update=update) if update else j)

    s = description.safety
    new_safety = s.model_copy(
        update={
            "max_ee_speed_m_s": _prompt_float("safety max_ee_speed_m_s", s.max_ee_speed_m_s),
            "max_joint_speed_factor": _prompt_float(
                "safety max_joint_speed_factor", s.max_joint_speed_factor
            ),
            "max_force_n": _prompt_float("safety max_force_n", s.max_force_n),
            "max_torque_nm": _prompt_float("safety max_torque_nm", s.max_torque_nm),
        }
    )
    return description.model_copy(update={"joints": new_joints, "safety": new_safety}, deep=True)


def _iter_wizard_cameras(detection: DetectionReport) -> list[tuple[str, str, str]]:
    """(device_path, human_label, serial) per bindable camera.

    V4L2 cameras bind on ``device_path`` (``serial`` empty). RealSense/Orbbec
    depth cameras have no ``/dev/video`` path here, so ``device_path`` is
    empty and the binding keys on ``serial`` instead.
    """
    cams: list[tuple[str, str, str]] = [(c.device_path, c.name, "") for c in detection.cameras.v4l2]
    cams += [
        ("", f"{d.name} (realsense {d.serial})", d.serial) for d in detection.cameras.realsense
    ]
    cams += [("", f"{d.name} (orbbec {d.serial})", d.serial) for d in detection.cameras.orbbec]
    return cams


def _run_camera_binding_wizard(  # noqa: PLR0915  # reason: one linear per-camera bind loop (thumbnail → prompt → dedup → reuse/new/workcell branch); splitting it hurts readability
    canonical: RobotDescription, detection: DetectionReport
) -> tuple[list[SensorSpec], list[SensorSpec]]:
    """Bind each detected camera to a sensor.

    Returns ``(robot_sensors, workcell_sensors)``: a canonical name reuses that
    manifest ``SensorSpec`` verbatim (+ real device binding) → robot manifest;
    a ``w:<name>`` answer → workcell camera → DeployScene; a new name → new
    robot sensor. Enter skips the device (dropping unbound canonical sensors).
    """
    import tempfile

    from openral_core import SensorDeployBinding, SensorSpec

    canonical_rgb = {s.name: s for s in canonical.sensors if s.modality == "rgb"}
    robot_sensors: list[SensorSpec] = []
    workcell_sensors: list[SensorSpec] = []
    # Names actually bound so far, keyed by target manifest — a canonical-name
    # reuse or a brand-new sensor both claim a robot-sensor name; a `w:<name>`
    # answer claims a workcell-sensor name. Skipping a camera (Enter) never
    # claims anything. Seeded empty: canonical names only become claimed once
    # an operator actually binds a camera to them.
    robot_claimed: set[str] = set()
    workcell_claimed: set[str] = set()
    thumb_dir = Path(tempfile.mkdtemp(prefix="openral_detect_cams_"))

    for device_path, label, serial in _iter_wizard_cameras(detection):
        thumb = _grab_camera_thumbnail(device_path, thumb_dir) if device_path else None
        console.print(
            f"\n[bold]{device_path or label}[/bold] — {label}"
            + (f"  [dim](thumbnail: {thumb})[/dim]" if thumb else "  [dim](no thumbnail)[/dim]")
        )
        options = ", ".join(canonical_rgb) if canonical_rgb else "<none in manifest>"

        # Re-prompt the same camera when the resolved target name is already
        # claimed, so two cameras can never land on the same sensor name
        # (RobotDescription/DeployScene have no uniqueness validator).
        answer = ""
        while True:
            answer = typer.prompt(
                f"  Which sensor is this? [{options}] or w:<name> for a workcell "
                f"camera (Enter = skip)",
                default="",
                show_default=False,
            ).strip()
            if not answer:
                break
            if answer.startswith("w:") and answer.removeprefix("w:").strip():
                name = answer.removeprefix("w:").strip()
                if name in workcell_claimed:
                    console.print(
                        f"  [yellow]{name!r} already bound to another camera — choose "
                        f"a different sensor.[/yellow]"
                    )
                    continue
            elif answer in robot_claimed:
                console.print(
                    f"  [yellow]{answer!r} already bound to another camera — choose "
                    f"a different sensor.[/yellow]"
                )
                continue
            break

        if not answer:
            continue
        params: dict[str, object] = {"fps": 30}
        if device_path:
            params["device"] = device_path
        elif serial:
            params["serial"] = serial
        binding = SensorDeployBinding(backend_params=params if (device_path or serial) else {})
        if answer in canonical_rgb:
            ref = canonical_rgb[answer]
            console.print(
                f"  [yellow]reusing canonical intrinsics for {answer!r} — best to "
                f"supply your rig's calibrated fx/fy/cx/cy.[/yellow]"
            )
            robot_sensors.append(ref.model_copy(update={"deploy_binding": binding}, deep=True))
            robot_claimed.add(answer)
            console.print(
                f"  [green]bound[/green] {answer} → {device_path or label} (robot sensor)"
            )
        elif answer.startswith("w:") and answer.removeprefix("w:").strip():
            name = answer.removeprefix("w:").strip()
            workcell_sensors.append(
                SensorSpec(
                    name=name,
                    modality="rgb",
                    frame_id=name,
                    rate_hz=30.0,
                    deploy_binding=binding,
                )
            )
            workcell_claimed.add(name)
            console.print(f"  [green]workcell[/green] {name} → {device_path or label}")
        else:
            parent = typer.prompt("    New robot sensor — parent_frame", default="base").strip()
            console.print(
                f"  [yellow]new sensor {answer!r} uses generic intrinsics — supply "
                f"calibrated fx/fy/cx/cy before production.[/yellow]"
            )
            robot_sensors.append(
                SensorSpec(
                    name=answer,
                    modality="rgb",
                    frame_id=f"{answer}_optical_frame",
                    parent_frame=parent or "base",
                    rate_hz=30.0,
                    encoding="rgb8",
                    deploy_binding=binding,
                )
            )
            robot_claimed.add(answer)
            console.print(
                f"  [green]bound[/green] {answer} → {device_path or label} (new robot sensor)"
            )

    return robot_sensors, workcell_sensors


# Print-once guard for the "opencv missing" hint. A set (mutated, never
# rebound) avoids a `global` statement while staying a single-process latch.
_CV2_WARNED: set[str] = set()


def _grab_camera_thumbnail(device_path: str, out_dir: Path) -> Path | None:
    """Grab one JPEG frame from ``device_path`` so the operator can SEE the camera.

    Best-effort: returns ``None`` when opencv is missing or the device won't
    deliver a frame (in use, no permission). Never raises — a thumbnail is a
    convenience, not a requirement.
    """
    try:
        import cv2  # reason: optional dep, ships with the CLI
    except ImportError:
        if not _CV2_WARNED:
            console.print("[yellow]opencv not installed — camera thumbnails disabled.[/yellow]")
            _CV2_WARNED.add("warned")
        return None
    cap = cv2.VideoCapture(device_path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (device_path.strip("/").replace("/", "_") + ".jpg")
    if not cv2.imwrite(str(out), frame):
        return None
    return out


def _write_deploy_scene_scaffold(
    path: Path,
    description: RobotDescription,
    sensor_specs: list[SensorSpec],
    *,
    detection: DetectionReport,
    assume_yes: bool,
) -> None:
    """Scaffold a ``DeployScene`` YAML next to the detected robot.

    Carries ``robot_id`` + the wizard's workspace/workcell camera bindings —
    ``sensor_specs`` here are always workcell cameras; robot-mounted cameras
    live in the robot manifest itself and never reach this scaffold. ``safety``
    is left unset so the robot manifest's own envelope applies. Validated
    through :class:`DeployScene` before writing so a malformed scaffold fails
    here, not on ``deploy run``.
    """
    import yaml as _yaml
    from openral_core import DeployScene, HalParameters, SceneSpec

    robot_id = str(description.name)
    # Seed the scene's HAL binding from the robot manifest's own
    # hal.parameters.defaults (e.g. serial port) + lerobot calibration
    # placeholders, so the scaffolded scene is a self-contained `deploy run`
    # target (no `--hal` needed once the operator fills the real port + commits
    # the calibration). Only serial (`port`) HALs get the calibration stub.
    hal_defaults: dict[str, object] = dict(description.hal.parameters.defaults)
    if "port" in hal_defaults:
        hal_defaults.setdefault("id", f"{robot_id}")
        hal_defaults.setdefault("calibration_dir", "calibration")
        hal_defaults.setdefault("calibrate_on_connect", False)
    # Override the stale manifest default port with the one actually probed on
    # this host, so the scaffolded scene targets real hardware out of the box.
    detected_port = next((m.device.port for m in detection.usb.matches if m.device.port), None)
    if detected_port and "port" in hal_defaults:
        hal_defaults["port"] = detected_port
    scene = DeployScene(
        scene=SceneSpec(id=f"{robot_id}_workcell"),
        robot_id=robot_id,
        hal=HalParameters(defaults=hal_defaults) if hal_defaults else None,
        sensors=list(sensor_specs),
    )
    if path.exists() and not assume_yes:
        overwrite = typer.confirm(f"{path} already exists. Overwrite?", default=False)
        if not overwrite:
            console.print("[yellow]Deployment scaffold aborted.[/yellow]")
            return
    banner = (
        "# DeployScene scaffolded by `openral detect --deployment` — review before\n"
        "# `openral deploy run --config <this file>`.\n"
        "# - safety: unset → the robot manifest's envelope applies as-is.\n"
        "# - hal: host-specific HAL binding — set the real serial `port` for this\n"
        "#   host and commit the lerobot calibration to `<scene dir>/calibration/`\n"
        "#   (a relative `calibration_dir` resolves against this file's directory).\n"
        "# - sensors: workspace (workcell) camera bindings; robot-mounted cameras\n"
        "#   live in the robot manifest.\n"
        "# - No rSkill is pinned here: the reasoner selects it at runtime.\n"
    )
    path.write_text(
        banner
        + _yaml.safe_dump(
            scene.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]Wrote[/green] {path} (DeployScene, {robot_id})")
    console.print(f"[dim]Next step:[/dim] openral deploy run --config {path}")

    # Serial (`port`) HALs are the lerobot Feetech followers (SO-100/SO-101/…);
    # they cannot connect without a lerobot calibration (per-servo IDs + homing
    # offsets). Nudge the operator to supply one when the calibration dir is
    # still empty, rather than letting `deploy run` fail on first connect.
    if "port" in hal_defaults:
        cal_dir = path.parent / str(hal_defaults.get("calibration_dir", "calibration"))
        cal_file = f"{hal_defaults.get('id', robot_id)}.json"
        has_cal = cal_dir.is_dir() and any(cal_dir.glob("*.json"))
        if not has_cal:
            console.print(
                "[yellow]Calibration required:[/yellow] this arm's serial HAL needs a "
                "lerobot calibration (per-servo IDs + homing offsets) before "
                "`deploy run` can connect.\n"
                f"  Link an existing calibration into [bold]{cal_dir}/{cal_file}[/bold], "
                "or generate one with the lerobot calibrate flow:\n"
                "  [cyan]https://huggingface.co/docs/lerobot/v0.6.0/en/so101#calibrate[/cyan]"
            )


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
    if detection.can.interfaces:
        up = [i for i in detection.can.interfaces if i.is_up]
        table.add_row(
            "can",
            f"{len(detection.can.interfaces)} interface(s), {len(up)} up, "
            f"{len(detection.can.matches)} matched",
        )
        for match in detection.can.matches:
            # State is what separates "wired up" from "actually answering",
            # so it is worth a row of its own per matched bus.
            for iface in match.interfaces:
                healthy = iface.state in ("ERROR-ACTIVE", "")
                colour = "green" if healthy else "yellow"
                rate = f"{iface.bitrate // 1000}k"
                if iface.fd_enabled and iface.data_bitrate:
                    rate += f"/{iface.data_bitrate // 1000}k FD"
                table.add_row(
                    f"  {iface.name}",
                    f"{match.bh_robot_type} · {iface.adapter} · {rate} · "
                    f"[{colour}]{iface.state or 'unknown'}[/{colour}]",
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
    robot: str = typer.Option(..., help="Robot type (so100, so101, g1, ur5e, …)"),
    port: str = typer.Option("", "--port", help="USB/serial port override, e.g. /dev/ttyUSB0"),
) -> None:
    """Open a HAL connection to a robot, read one joint state, and disconnect.

    Exits 0 on success; exits 1 with an error message on failure.

    Supported robots: so100, so101 (both drive the shared ``SO100FollowerHAL``
    Feetech serial bus — the SO-101 is the same controller as the SO-100).

    Example:
        >>> # openral connect --robot so100
        >>> # openral connect --robot so101 --port /dev/ttyUSB1
    """
    # so100 and so101 share the Feetech SO100FollowerHAL (identical USB
    # controller + driver); the label only changes the console banner.
    so_follower_labels = {"so100": "SO-100", "so101": "SO-101"}
    if robot in so_follower_labels:
        _connect_so_follower(so_follower_labels[robot], port or "/dev/ttyUSB0")
    else:
        console.print(f"[red]Unknown robot '{robot}'. Supported: so100, so101[/red]")
        raise typer.Exit(code=1)


def _connect_so_follower(label: str, port: str) -> None:
    """Connect to an SO-100/SO-101 follower arm, read state, and disconnect."""
    try:
        from openral_hal.so100_follower import SO100FollowerHAL
    except ImportError:
        console.print("[red]openral-hal is not installed. Run: uv sync --all-packages[/red]")
        raise typer.Exit(code=1)  # noqa: B904

    hal = SO100FollowerHAL(port=port)
    console.print(f"Connecting to {label} on [bold]{port}[/bold] …")
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

#: Canonical HF Hub org for first-party rSkills. Used to suggest a
#: repair when ``rskill install`` is handed an org-less id, and as the ``author``
#: filter for ``rskill search``.
_DEFAULT_RSKILL_ORG: Final[str] = "OpenRAL"


@rskill_app.command("install")
def rskill_install(
    hub_id: str = typer.Argument(
        ...,
        metavar="HUB_ID",
        help="HF Hub repository, e.g. OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16",
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
        >>> # openral rskill install OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16
        >>> # openral rskill install smolvla-libero --revision abc1234
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
    """Search the OpenRAL HF Hub org for installable rSkills.

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
            "or HF Hub repo id (e.g. OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16). "
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
        >>> # openral rskill check smolvla-libero --robot /tmp/robot.yaml --json
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
            "Benchmark suite to evaluate — a bare ``list[BenchmarkScene]`` YAML. "
            "Either a built-in id (resolved to "
            "`benchmarks/<id>.yaml`) or a direct YAML path."
        ),
    ),
    rskill: str = typer.Option(
        ...,
        "--rskill",
        help=(
            "rSkill reference — a bare name ('smolvla-libero'), a "
            "path ('rskills/smolvla-libero'), or an HF Hub repo id "
            "('OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16'). "
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
    task: str | None = typer.Option(
        None,
        "--task",
        help=(
            "Run only this single task id from the suite (e.g. "
            "'libero_spatial/3' or 'maniskill3/PushCube-v1'). Omit to run "
            "every task the rSkill supports (the suite is auto-filtered to "
            "the rSkill's evaluated_tasks)."
        ),
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
    video: bool = typer.Option(
        True,
        "--video/--no-video",
        help=(
            "Record per-step world frames and write one MP4 per episode "
            "(named <task>[_seed<n>]_<rskill>_<success|fail>.mp4 plus a "
            "videos.json manifest). On by default; frames are written and "
            "freed per episode. Disable for allocation-light CI runs."
        ),
    ),
    video_dir: Path | None = typer.Option(
        None,
        "--video-dir",
        help=(
            "Destination directory for --video MP4s. Defaults to "
            "<eval JSON dir>/videos/<suite_id> next to the RSkillEvalResult."
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
    carries its own scene + task + robot; the ``BenchmarkSpec`` wrapper
    class was removed so the suite is a bare list of scenes whose id is
    the YAML filename stem.

    Example:
        >>> # openral benchmark run --suite libero_spatial \\
        >>> #     --rskill smolvla-libero
    """
    scenes, suite_id = _resolve_benchmark_suite(suite, benchmarks_dir)
    vla_spec = _parse_rskill_cli_arg(rskill)

    # --task selects a single explicit task from the suite. The rSkill's
    # evaluated_tasks auto-filter (in run_benchmark) still applies, so an
    # explicitly-picked task the rSkill was not trained for is rejected — same
    # contract as `openral benchmark scene`.
    if task is not None:
        matched = [s for s in scenes if s.task.id == task]
        if not matched:
            available = [s.task.id for s in scenes]
            console.print(
                f"[red]--task {task!r} is not in suite {suite_id!r}.[/red] "
                f"Available tasks: {available}"
            )
            raise typer.Exit(1)
        scenes = matched

    # Apply --n-episodes override to every scene before dry-run or real run.
    if n_episodes is not None:
        scenes = [s.model_copy(update={"n_episodes": n_episodes}) for s in scenes]

    if dry_run:
        _print_benchmark_run_plan(scenes, suite_id=suite_id, vla_spec=vla_spec)
        return

    out_path = out if out is not None else _default_benchmark_out_path(vla_spec, suite_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    effective_video_dir: Path | None = None
    if video:
        effective_video_dir = (
            video_dir if video_dir is not None else (out_path.parent / "videos" / suite_id)
        )

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
            video_dir=str(effective_video_dir) if effective_video_dir is not None else None,
        )
    if effective_video_dir is not None:
        console.print(f"[cyan]videos[/cyan] {effective_video_dir}")

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


def _print_benchmark_run_plan(
    scenes: list[BenchmarkScene],
    *,
    suite_id: str,
    vla_spec: VLASpec,
) -> None:
    """Print `openral benchmark run --dry-run`'s plan, or exit if nothing would run.

    Applies the same ``evaluated_tasks`` filter :func:`run_benchmark` applies,
    so the printed plan is the plan that would actually execute. Without it a
    suite the rSkill covers for zero tasks dry-ran clean and then raised
    ``ROSCapabilityMismatch`` on the real invocation, and a partially covered
    suite over-reported its episode count.

    Raises:
        typer.Exit: The rSkill's ``evaluated_tasks`` match no task in the suite.
    """
    from openral_sim.benchmark import _manifest_for_filter, filter_scenes_for_skill

    # _manifest_for_filter returns None for mock / hf:// skills, which
    # filter_scenes_for_skill treats as permissive — same as the real run.
    kept, skipped = filter_scenes_for_skill(scenes, _manifest_for_filter(vla_spec))
    if not kept:
        console.print(
            f"[red]✗ task gate:[/red] rSkill {vla_spec.weights_uri!r} covers none of "
            f"the {len(scenes)} task(s) in suite {suite_id!r} "
            f"(e.g. {[s.task.id for s in scenes][:3]}). Nothing would run."
        )
        raise typer.Exit(1)
    if skipped:
        console.print(
            f"[yellow]note[/yellow]  {len(skipped)} of {len(scenes)} suite tasks are "
            f"outside this rSkill's evaluated_tasks and would be skipped: "
            f"{[s.task.id for s in skipped][:5]}"
        )
    scenes = kept
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


def _resolve_benchmark_suite(
    suite: str,
    benchmarks_dir: Path,
) -> tuple[list[BenchmarkScene], str]:
    """Map a ``--suite`` argument to a validated ``(scenes, suite_id)`` tuple.

    A benchmark suite is a bare ``list[BenchmarkScene]`` YAML;
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
    (``OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16``). The adapter id is read from
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
    if manifest.model_family is None:
        # Only `kind='vla'` skills carry a model_family; a detector/reward skill
        # has none and cannot drive a VLASpec.
        raise typer.BadParameter(
            f"rSkill {raw!r} has no model_family (kind={manifest.kind!r}); "
            f"--rskill expects a VLA skill."
        )
    return VLASpec(
        id=manifest.model_family,
        weights_uri=uri,
        extra=dict(manifest.policy_extras),
    )


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
            "('OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16'). "
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
    save_video: Path | None = typer.Option(
        None,
        "--save-video",
        help=(
            "Write a clean single-view world MP4 per episode to this "
            "directory, named <task>_<rskill>_<success|fail>.mp4, plus a "
            "videos.json manifest — for website hero clips (overlays are "
            "rendered by the page, not burned into pixels). The task slug "
            "keeps benchmark scenes sharing one backend from overwriting each "
            "other. Enables per-step frame capture. Pair with --n-episodes 1 "
            "for a single demo clip."
        ),
    ),
    video_size: int = typer.Option(
        1024,
        "--video-size",
        help=(
            "Square edge (px) for --save-video output. Frames are "
            "center-cropped to a square and resized to this size. Source "
            "sharpness is bounded by the scene's native render resolution."
        ),
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
            "Resolve the scene + rSkill, apply the evaluated_tasks gate, and "
            "print the planned (task x seed) matrix without running any "
            "rollouts or fetching weights. Useful in CI to validate config "
            "wiring; exits non-zero if the pairing would be rejected."
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
        # Resolve the rSkill here rather than echoing it as-typed: a dry run
        # that never parses --rskill lets a broken manifest (or a non-VLA
        # kind) through the exact check it is run to perform. Manifest-only —
        # no weights are fetched. Built-in mock policies have no manifest
        # (same carve-out as openral_sim.benchmark._manifest_for_filter), so
        # they keep the as-typed echo.
        from openral_core.exceptions import ROSCapabilityMismatch  # reason: defer
        from openral_sim.sim_runner import _MOCK_PLACEHOLDER_URI, _MOCK_POLICY_IDS

        vla_line = f"rskill={rskill}"
        if rskill not in _MOCK_POLICY_IDS and rskill != _MOCK_PLACEHOLDER_URI:
            spec = _parse_rskill_cli_arg(rskill)
            vla_line = f"rskill={rskill} id={spec.id}"

            # Same task gate `run_benchmark_scene` applies, so a task-mismatched
            # pairing fails here instead of looking planned and then raising.
            from openral_rskill.loader import load_rskill_manifest
            from openral_sim.benchmark import check_benchmark_task_compatibility

            try:
                check_benchmark_task_compatibility(
                    load_rskill_manifest(spec.weights_uri),
                    task_id=scene.task.id,
                    scene_id=scene.scene.id,
                )
            except ROSCapabilityMismatch as exc:
                console.print(f"[red]✗ task gate:[/red] {exc}")
                raise typer.Exit(code=1) from exc

        console.print(
            f"[cyan]scene[/cyan] {scene.scene.id} — robot={scene.robot_id} "
            f"task={scene.task.id} n_episodes={scene.n_episodes} "
            f"seed={scene.seed}"
        )
        console.print(f"[cyan]vla[/cyan]   {vla_line}")
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
            record_video=save_video is not None,
        )

    if save_video is not None:
        from openral_sim._website_video import write_world_videos

        write_world_videos(
            episodes,
            save_video,
            scene=scene.task.id,
            rskill=Path(rskill).name,
            section=Path(config).parent.name,
            size=video_size,
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

# `openral behavior serve` — expose an rSkill through the official BEHAVIOR
# Challenge WebSocket policy protocol. OmniGibson remains in its own conda
# environment and owns task loading, metrics, and video recording.
behavior_app = typer.Typer(
    name="behavior",
    help="BEHAVIOR Challenge integration — serve OpenRAL rSkills to OmniGibson.",
    no_args_is_help=True,
)
app.add_typer(behavior_app, name="behavior")


@behavior_app.command("serve")
def behavior_serve(
    rskill: str = typer.Option(
        ...,
        "--rskill",
        help="rSkill reference: local name/path or Hugging Face repo id.",
    ),
    task: str = typer.Option(
        ...,
        "--task",
        help="BEHAVIOR task name, e.g. turning_on_radio.",
    ),
    instruction: str | None = typer.Option(
        None,
        "--instruction",
        help="Policy language instruction. Defaults to the task name with underscores replaced.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="WebSocket bind address.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        min=1,
        max=65535,
        help="WebSocket port used by omnigibson.eval.eval.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Policy device override, e.g. auto, cuda:0, or cpu.",
    ),
    state_dim: int = typer.Option(
        61,
        "--state-dim",
        min=1,
        help="Expected R1Pro proprioception width.",
    ),
    action_dim: int = typer.Option(
        23,
        "--action-dim",
        min=1,
        help="Expected R1Pro action width.",
    ),
) -> None:
    """Serve one OpenRAL rSkill to BEHAVIOR's official evaluator."""
    from openral_cli.behavior import _serve_behavior_policy

    vla_spec = _parse_rskill_cli_arg(rskill).model_copy(update={"device": device})
    _serve_behavior_policy(
        vla_spec,
        task=task,
        instruction=instruction or task.replace("_", " "),
        host=host,
        port=port,
        state_dim=state_dim,
        action_dim=action_dim,
    )


# `openral install <group>` — post-install escape hatch for the
# Tier-0 curl-bash installer (`scripts/install.sh`). The base install puts
# `openral` on $PATH with the CLI's own thin runtime; sim physics, LIBERO,
# MetaWorld, RoboCasa, and the sudo+apt ROS 2 bootstrap layer in on demand.
app.add_typer(install_app, name="install")

# `openral dataset push` — publish a LeRobotDataset v3 to the HF Hub.
# Importing `dataset` at module top is cheap; the `push` command itself lazy-
# imports huggingface_hub only when actually publishing so `openral --help` stays
# sub-second.
app.add_typer(dataset_app, name="dataset")

# `openral collision lower|check` — offline URDF/SRDF → manifest
# self-collision model. The `lower_robot` import is deferred inside the commands
# (it pulls yourdfpy/trimesh) so `openral --help` stays fast.
app.add_typer(collision_app, name="collision")

# `openral check` — static, host-independent validation of the declarative
# robot/skill/scene set (manifests parse, asset refs resolve, scene robot_ids and
# rSkill embodiment tags resolve). Complements `openral rskill check`. Manifest
# JSON-Schema emission lives in `tools/schema_export.py` (CI-gated), not here.
app.command("check")(check_command)

# `openral robot vendor-urdf <id>` — expand an upstream xacro to a
# flat, committed URDF so end users need no xacro tooling at runtime. The
# `vendor_urdf` import is deferred inside the command (it pulls robot_descriptions/
# xacrodoc/yourdfpy) so `openral --help` stays fast.
robot_app = typer.Typer(
    name="robot",
    help="Robot description assets — vendor a flat URDF from an upstream xacro.",
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
    """Expand an upstream description to a flat, committed URDF."""
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


# `openral prompt "do X"` publishes a one-shot PromptStamped
# onto /openral/prompt_in/cli; the prompt_router_node fans it out to
# /openral/prompt for the reasoner. rclpy import is deferred inside
# the command body so `openral --help` stays sub-second.
app.command(
    name="prompt",
    help=(
        "Publish a one-shot operator prompt to the prompt-router. Requires a sourced ROS 2 install."
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
        "Hardware deploy — run the ROS graph against a real robot from a "
        "DeployScene YAML (`openral deploy run`) or list deploy scenes."
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
    """List every deploy scene under `scenes/deploy/*.yaml`.

    Each entry is a paste-able `--config` path for `openral deploy run` or `deploy sim`.
    No hardware touch, no GPU.
    """
    from openral_rskill.loader import _find_repo_root_from

    repo_root = _find_repo_root_from(Path(__file__))
    if repo_root is None:
        console.print("[red]Could not locate repo root.[/red]")
        raise typer.Exit(code=1)
    deploy_scenes = repo_root / "scenes" / "deploy"
    if not deploy_scenes.is_dir():
        print("<none>")
        return
    configs = sorted(deploy_scenes.rglob("*.yaml"))
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
        help="Path to a DeployScene YAML; its robot_id selects the real robot workcell.",
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
    enable_reward_monitor: bool | None = typer.Option(
        None,
        "--enable-reward-monitor/--no-enable-reward-monitor",
        help=(
            "Bring up the Robometer reward monitor parallel to the "
            "VLA (same leg `deploy sim` exposes): it scores the robot's first RGB "
            "camera topic and serves /openral/perception/query_task_progress. The "
            "manifest is auto-paired from the VLA palette's reward_rskill_name; "
            "override with --reward-monitor-manifest. Unset = the "
            "scene's runtime.enable_reward_monitor, else off."
        ),
    ),
    reward_monitor_manifest: str | None = typer.Option(
        None,
        "--reward-monitor-manifest",
        help=(
            "Path to a kind:reward rSkill manifest. Empty auto-pairs "
            "from the VLA palette, falling back to rskills/robometer-4b. Ignored "
            "unless --enable-reward-monitor."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved real-mode launch argv and exit without shelling out.",
    ),
) -> None:
    """Run an rSkill on REAL hardware via the production ROS graph.

    Unlike `openral deploy sim`, this drives the **real** hardware HAL: it
    resolves the robot from `--config` (a DeployScene) and shells the SAME
    `sim_e2e.launch.py` graph with `hal_mode:=real` — the HAL lifecycle node +
    C++ safety kernel + reasoner + world state (+ SLAM/Nav2 when the robot
    declares a lidar). The HAL's `connect()` fails loudly if no hardware is
    attached; a simulation-only robot raises ROSCapabilityMismatch (use
    `openral deploy sim`). Robot HAL defaults come from robot.yaml; `--hal` wins.
    """
    from openral_core import DeployScene  # reason: defer schema import
    from openral_core.exceptions import ROSCapabilityMismatch  # reason: defer
    from pydantic import ValidationError  # reason: defer CLI import

    from openral_cli.deploy_sim import (  # reason: defer heavy CLI import
        _parse_hal_overrides,
        resolve_launch_invocation,
        run_launch_invocation,
    )

    try:
        deploy_scene = DeployScene.from_yaml(str(config))
    except (FileNotFoundError, ROSConfigError, ValidationError) as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    overrides = _parse_hal_overrides(hal)

    # A committed, config-relative `calibration_dir` (deploy owns its calibration
    # instead of the ambient HF cache) is resolved against THIS config's
    # directory so it works regardless of the CWD `deploy run` is invoked from.
    cal_dir = overrides.get("calibration_dir")
    if isinstance(cal_dir, str) and cal_dir and not Path(cal_dir).is_absolute():
        overrides["calibration_dir"] = str((config.parent / cal_dir).resolve())

    try:
        invocation = resolve_launch_invocation(
            config=config,
            robot_override=robot or deploy_scene.robot_id,
            dashboard_port=dashboard_port,
            reset_to_pose_service=None,
            deploy_config=config,
            hal_param_overrides=overrides,
            hal_mode="real",
            enable_dashboard=dashboard,
            enable_reward_monitor=enable_reward_monitor,
            reward_monitor_manifest=reward_monitor_manifest,
        )
    except (ROSConfigError, ROSCapabilityMismatch) as exc:
        console.print(f"[red]deploy run:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[cyan]deploy run[/cyan] {invocation.robot_id} → real HAL "
        f"(hal_mode=real); the HAL's connect() requires the robot to be attached."
    )
    if dry_run:
        printed = [
            arg.replace("HAL_PARAMS_FILE_PLACEHOLDER", "<hal-params-tmp>")
            for arg in invocation.argv_template
        ]
        console.print(f"  hal_params: {invocation.hal_params}")
        console.print(f"  argv: {shlex.join(printed)}")
        return

    returncode = run_launch_invocation(invocation)
    raise typer.Exit(code=returncode)


@deploy_app.command("validate")
def deploy_validate(
    config: Path = typer.Option(  # reason: typer Option idiom
        ...,
        "--config",
        "-c",
        exists=True,
        readable=True,
        dir_okay=False,
        help="DeployScene YAML to check for real-run readiness.",
    ),
    robot: str | None = typer.Option(
        None,
        "--robot",
        help="Override the robot_id resolved from --config.",
    ),
    hal: list[str] | None = typer.Option(
        None,
        "--hal",
        help="HAL overrides applied before validation (same precedence as deploy run).",
    ),
) -> None:
    """Pre-run readiness check for `openral deploy run` — no hardware, no ROS launch.

    Validates the DeployScene + robot manifest resolve, then checks the
    runtime-required inputs a real run needs are present *before* the launch —
    the exact gaps that otherwise fail late at HAL configure / sensor leg:

    * **HAL transport** — a serial `port` is declared, and its device exists now.
    * **Calibration** — a serial HAL with `calibrate_on_connect=false` has an
      `id` + `calibration_dir`, and the `<calibration_dir>/<id>.json` file exists
      (missing → "has no calibration registered" at every send_action).
    * **Camera bindings** — each scene sensor has a `deploy_binding` (else it is
      never published and a camera VLA gets an empty observation), and any
      `/dev/*` device path exists now.

    Reports ERROR (missing committed data — exits non-zero) vs WARN (device just
    not attached right now). HAL param precedence matches `deploy run`
    (`--hal` > scene `hal` > `robot.yaml`).
    """
    from openral_core import DeployScene  # reason: defer schema import
    from openral_core.exceptions import ROSCapabilityMismatch  # reason: defer
    from pydantic import ValidationError  # reason: defer CLI import

    from openral_cli.deploy_sim import (  # reason: defer heavy CLI import
        _parse_hal_overrides,
        resolve_launch_invocation,
    )

    try:
        deploy_scene = DeployScene.from_yaml(str(config))
    except (FileNotFoundError, ROSConfigError, ValidationError) as exc:
        console.print(f"[red]✗ config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    overrides = _parse_hal_overrides(hal)
    cal_dir_override = overrides.get("calibration_dir")
    if (
        isinstance(cal_dir_override, str)
        and cal_dir_override
        and not Path(cal_dir_override).is_absolute()
    ):
        overrides["calibration_dir"] = str((config.parent / cal_dir_override).resolve())

    # Reuse the deploy-run resolver: raises on sim-only robot, name mismatch,
    # unknown HAL, missing manifest — and produces the merged hal_params
    # (registry → scene hal → --hal) we then inspect for readiness.
    try:
        invocation = resolve_launch_invocation(
            config=config,
            robot_override=robot or deploy_scene.robot_id,
            dashboard_port=4318,
            reset_to_pose_service=None,
            deploy_config=config,
            hal_param_overrides=overrides,
            hal_mode="real",
            enable_dashboard=False,
        )
    except (ROSConfigError, ROSCapabilityMismatch) as exc:
        console.print(f"[red]✗ resolve:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    errors: list[str] = []
    warns: list[str] = []
    hp = invocation.hal_params

    port = hp.get("port")
    if isinstance(port, str) and port and not Path(port).exists():
        warns.append(f"serial port {port!r} does not exist now (arm not attached?)")

    if isinstance(port, str) and port and not bool(hp.get("calibrate_on_connect", False)):
        cal_id = hp.get("id")
        cal_dir = hp.get("calibration_dir")
        if not cal_id or not cal_dir:
            errors.append(
                "serial HAL with calibrate_on_connect=false but no id/calibration_dir "
                "→ every send_action/read_state raises 'has no calibration registered'. "
                "Add a scene `hal:` binding with id + calibration_dir."
            )
        else:
            cal_file = Path(str(cal_dir)) / f"{cal_id}.json"
            if not cal_file.exists():
                errors.append(f"calibration file {cal_file} does not exist (id={cal_id!r}).")

    if not deploy_scene.sensors:
        warns.append("scene declares no sensors → a camera VLA will get an empty observation")
    for sensor in deploy_scene.sensors:
        binding = sensor.deploy_binding
        if binding is None:
            warns.append(
                f"sensor {sensor.name!r} has no deploy_binding → not published, VLA won't see it"
            )
            continue
        dev = binding.backend_params.get("device")
        if isinstance(dev, str) and dev.startswith("/dev/") and not Path(dev).exists():
            warns.append(f"sensor {sensor.name!r} device {dev!r} does not exist now")

    console.print(f"[cyan]deploy validate[/cyan] {invocation.robot_id} ← {config}")
    for warn in warns:
        console.print(f"  [yellow]⚠ {warn}[/yellow]")
    for err in errors:
        console.print(f"  [red]✗ {err}[/red]")
    if errors:
        console.print(
            f"[red]{len(errors)} error(s), {len(warns)} warning(s) — "
            "not ready for `deploy run`.[/red]"
        )
        raise typer.Exit(code=1)
    console.print(f"[green]✓ ready for `deploy run` ({len(warns)} warning(s)).[/green]")


# ── openral replay — bag↔OTel correlator ──────────────────────────


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
        "Join a rosbag2/.mcap file with OTel spans from the live dashboard. "
        "Prints a chronological JSON timeline keyed by trace_id; "
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


# ── openral record — wrap `ros2 bag record` with profile presets ──


@app.command(
    "record",
    help=(
        "Spawn `ros2 bag record` for the OpenRAL ROS graph with a slim/full profile. "
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
    """Wrap `ros2 bag record` with slim/full topic presets."""
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


# ── openral profile session — LTTng opt-in profiling ──────────────

profile_app = typer.Typer(
    name="profile",
    help="Microsecond-accurate profiling via ros2_tracing / LTTng.",
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
