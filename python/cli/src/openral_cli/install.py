"""``openral install`` — install opt-in dependency groups into the managed venv.

This subcommand is the post-install escape hatch for the Tier-0 curl-bash
installer (``scripts/install.sh``, ADR-0021). The base install gives the user
``openral`` on their ``$PATH`` with the CLI's own thin runtime; heavy / opt-in
extras (sim physics, LIBERO / MetaWorld / RoboCasa task suites, ROS 2 system
deps) ship separately and are layered in on demand.

Group taxonomy mirrors the workspace root ``pyproject.toml``
``[dependency-groups]`` table — keep the two in sync. ``ros`` is special-cased
because it re-exec's ``scripts/bootstrap_ubuntu.sh`` (sudo + apt) rather than
calling ``uv pip install``; everything else is a pure-Python group resolved by
the workspace lockfile.

The libero ↔ robocasa exclusion declared in the root ``[tool.uv].conflicts``
table is enforced here as a typed ``ROSConfigError`` so users see the failure
at the CLI instead of as a solver-conflict from ``uv pip install``.

Examples:
    Install the lightweight sim group (CPU-only physics)::

        openral install sim

    Install LIBERO (mutually exclusive with robocasa — see ADR-0011)::

        openral install libero

    Re-run the apt + ROS 2 + udev system bootstrap (needs sudo)::

        openral install ros
"""

from __future__ import annotations

import json as _json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import typer
from openral_core.exceptions import ROSConfigError
from rich.console import Console
from rich.table import Table

# ── Group → dependency-list mirror of root pyproject.toml ────────────────────
#
# These lists are duplicated from the workspace root ``pyproject.toml``
# ``[dependency-groups]`` table on purpose: the installer must work *before*
# the workspace is cloned (the curl-bash one-liner runs `uv tool install
# openral-cli` against PyPI, which does not see the root pyproject). When a
# group is added or a pin is bumped in the root file, mirror the change here
# in the same commit — the ``tests/unit/test_install_command.py`` real-
# components check loads the workspace pyproject when present and asserts the
# two are in lockstep.
#
# Each entry is exactly what would appear in ``[dependency-groups].<group>``;
# the installer hands them straight to ``uv pip install`` so PEP 508 markers
# are honoured.
_GROUPS: Final[dict[str, list[str]]] = {
    "sim": [
        "gym-aloha>=0.1.3",
        "accelerate>=1.13.0",
        "bitsandbytes>=0.45",
        "gym-pusht>=0.1.6",
        "gymnasium[mujoco]>=1.3.0",
        "mujoco>=3.8.0",
        "num2words>=0.5.14",
        "pymunk<7",
        "robot-descriptions>=1.12.0",
        "transformers==5.3.0",
    ],
    "libero": [
        "lerobot[libero]",
        "transformers==5.3.0",
        "num2words>=0.5.14",
        "bitsandbytes>=0.45",
    ],
    "metaworld": [
        "lerobot",
        "gymnasium[mujoco]>=1.3.0",
        "mujoco>=3.8.0",
        "transformers==5.3.0",
        "accelerate>=1.13.0",
        "num2words>=0.5.14",
    ],
    "maniskill3": [
        "mani-skill>=3.0.0b9",
        "gymnasium>=1.0.0",
        "lerobot",
        "transformers==5.3.0",
        "accelerate>=1.13.0",
        "num2words>=0.5.14",
    ],
    "simpler-env": [
        "mani-skill>=3.0.0b9",
        "gymnasium>=1.0.0",
    ],
    "robocasa": [
        "robosuite>=1.5.2",
        "lerobot",
        "transformers==5.3.0",
        "accelerate>=1.13.0",
        "bitsandbytes>=0.45",
        "num2words>=0.5.14",
    ],
    "rldx": [
        "pyzmq>=25",
        "msgpack>=1",
    ],
}

# Mutually exclusive groups — mirrors ``[tool.uv].conflicts`` in the workspace
# root pyproject.toml (ADR-0011). Each entry is a frozenset of group names
# that cannot coexist in a single resolved environment.
_CONFLICTS: Final[tuple[frozenset[str], ...]] = (frozenset({"libero", "robocasa"}),)


install_app = typer.Typer(
    name="install",
    help=(
        "Install opt-in dependency groups (sim, libero, metaworld, robocasa, "
        "rldx, …) or re-run the ROS 2 system bootstrap. See ADR-0021."
    ),
    no_args_is_help=True,
)
console = Console()


def _detect_target_python() -> str:
    """Return the absolute path of the Python interpreter ``uv pip`` should target.

    Resolution order:
        1. ``OPENRAL_INSTALL_PYTHON`` env var — explicit override.
        2. ``sys.executable`` — the interpreter currently running the CLI.

    The Tier-0 installer (``scripts/install.sh``) calls
    ``uv tool install openral-cli``; that places the CLI in a uv-managed
    tool venv whose ``sys.executable`` is exactly the interpreter we want
    to install into. No PATH-sniffing heuristics needed.

    Returns:
        Absolute path to the target ``python3`` binary.
    """
    override = os.environ.get("OPENRAL_INSTALL_PYTHON")
    if override:
        return override
    return sys.executable


def _ensure_uv() -> str:
    """Return the path to a ``uv`` binary, raising ``ROSConfigError`` if absent.

    ``uv`` is the only supported installer for this command — CLAUDE.md §4
    forbids invoking ``pip`` directly inside the workspace.

    Raises:
        ROSConfigError: when ``uv`` is not on ``$PATH``.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise ROSConfigError(
            "`uv` not found on $PATH. Install it with "
            "`curl -LsSf https://astral.sh/uv/install.sh | sh` "
            "(the scripts/install.sh one-liner does this for you)."
        )
    return uv


def _check_conflicts(group: str, already_installed: frozenset[str]) -> None:
    """Raise ``ROSConfigError`` if ``group`` conflicts with anything already in the venv.

    Args:
        group: Name of the group about to be installed.
        already_installed: Names of groups previously installed into the
            current target venv. Detected via importlib.metadata best-effort
            in :func:`_detect_installed_groups`.

    Raises:
        ROSConfigError: when installing ``group`` would violate an entry in
            :data:`_CONFLICTS`.
    """
    for conflict_set in _CONFLICTS:
        if group in conflict_set:
            collision = (conflict_set - {group}) & already_installed
            if collision:
                others = ", ".join(sorted(collision))
                raise ROSConfigError(
                    f"Cannot install `{group}` — mutually exclusive with already-"
                    f"installed group(s): {others}. See ADR-0011. "
                    f"Use a separate venv (`uv venv .venv-{group}`) for the other "
                    f"group, or `openral install --force {group}` to override."
                )


def _detect_installed_groups(python: str) -> frozenset[str]:
    """Best-effort: probe the target venv for sentinel packages of each group.

    Uses a one-shot subprocess call to ``importlib.metadata.distributions``
    in the target interpreter rather than ``pkg_resources`` (deprecated) or
    a hand-rolled site-packages walk. Returns the set of group names whose
    sentinel package is present.

    Sentinels are chosen to be the cheapest unambiguous proof a group was
    installed — typically the package that drags the rest of the group in.

    Args:
        python: Absolute path to the target python interpreter.

    Returns:
        Frozen set of group names detected as installed.
    """
    sentinels = {
        "sim": "gym-aloha",
        "libero": "lerobot",
        "metaworld": "lerobot",
        "maniskill3": "mani-skill",
        "simpler-env": "mani-skill",
        "robocasa": "robosuite",
        "rldx": "pyzmq",
    }
    probe = (
        "import importlib.metadata as m, json, sys; "
        "names = {d.metadata['Name'].lower() for d in m.distributions() "
        "if d.metadata and d.metadata['Name']}; "
        f"print(json.dumps([k for k,v in {sentinels!r}.items() if v.lower() in names]))"
    )
    try:
        out = subprocess.run(
            [python, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return frozenset()

    try:
        return frozenset(_json.loads(out.stdout.strip() or "[]"))
    except _json.JSONDecodeError:
        return frozenset()


def _run_uv_pip_install(group: str, python: str) -> int:
    """Invoke ``uv pip install --python <python> <packages…>`` and stream output.

    Args:
        group: Dependency-group name; must be a key in :data:`_GROUPS`.
        python: Absolute path to the target interpreter.

    Returns:
        The subprocess exit code (0 on success).
    """
    uv = _ensure_uv()
    pkgs = _GROUPS[group]
    cmd = [uv, "pip", "install", "--python", python, *pkgs]
    console.print(f"[cyan]$ {' '.join(cmd)}[/cyan]")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _install_group(group: str, *, force: bool) -> None:
    """Install one dependency group into the active managed venv.

    Args:
        group: Group name. Must be a key in :data:`_GROUPS`.
        force: When ``True``, bypass the libero ↔ robocasa conflict check.

    Raises:
        ROSConfigError: when the group is unknown, when ``uv`` is missing, or
            when a conflict is detected and ``force=False``.
    """
    if group not in _GROUPS:
        known = ", ".join(sorted(_GROUPS))
        raise ROSConfigError(f"Unknown group `{group}`. Known: {known}.")

    python = _detect_target_python()
    if not force:
        installed = _detect_installed_groups(python)
        _check_conflicts(group, installed)

    rc = _run_uv_pip_install(group, python)
    if rc != 0:
        raise ROSConfigError(
            f"`uv pip install` for group `{group}` exited {rc}. "
            f"See the output above for the resolver error."
        )
    console.print(f"[green]✓ installed group `{group}` into {python}[/green]")


# ── Typer subcommands — one per supported group ──────────────────────────────


@install_app.command("sim")
def install_sim(
    force: bool = typer.Option(False, "--force", help="Bypass conflict checks."),
) -> None:
    """Install the ``sim`` group (gym-aloha, gym-pusht, MuJoCo, bitsandbytes)."""
    _install_group("sim", force=force)


@install_app.command("libero")
def install_libero(
    force: bool = typer.Option(False, "--force", help="Bypass conflict checks."),
) -> None:
    """Install LIBERO (mutually exclusive with ``robocasa`` — see ADR-0011)."""
    _install_group("libero", force=force)


@install_app.command("metaworld")
def install_metaworld(
    force: bool = typer.Option(False, "--force", help="Bypass conflict checks."),
) -> None:
    """Install the MetaWorld MT50 task suite (Sawyer scenes)."""
    _install_group("metaworld", force=force)


@install_app.command("maniskill3")
def install_maniskill3(
    force: bool = typer.Option(False, "--force", help="Bypass conflict checks."),
) -> None:
    """Install ManiSkill3 (SAPIEN GPU physics — ADR-0014)."""
    _install_group("maniskill3", force=force)


@install_app.command("simpler-env")
def install_simpler_env(
    force: bool = typer.Option(False, "--force", help="Bypass conflict checks."),
) -> None:
    """Install the SimplerEnv real-to-sim correlator backend (ADR-0014)."""
    _install_group("simpler-env", force=force)


@install_app.command("robocasa")
def install_robocasa(
    force: bool = typer.Option(False, "--force", help="Bypass conflict checks."),
) -> None:
    """Install RoboCasa (mutually exclusive with ``libero`` — see ADR-0011/0015)."""
    _install_group("robocasa", force=force)


@install_app.command("rldx")
def install_rldx(
    force: bool = typer.Option(False, "--force", help="Bypass conflict checks."),
) -> None:
    """Install the RLDX-1 sidecar client (pyzmq + msgpack — ADR-0010)."""
    _install_group("rldx", force=force)


@install_app.command("ros")
def install_ros() -> None:
    """Re-run the ROS 2 + system-package bootstrap (sudo + apt; Linux only).

    Delegates to ``scripts/bootstrap_ubuntu.sh`` or ``scripts/bootstrap_macos.sh``
    from a cloned workspace. Resolution order for the script path:

        1. ``OPENRAL_REPO_ROOT/scripts/bootstrap_<os>.sh`` if the env var is set.
        2. ``<cwd>/scripts/bootstrap_<os>.sh`` if a workspace is present.
        3. Print a clone hint and exit non-zero.

    The Tier-0 curl-bash installer cannot ship apt packages without sudo,
    so this subcommand is the only supported escalation path.

    Raises:
        ROSConfigError: when the bootstrap script cannot be located.
    """
    if sys.platform == "darwin":
        script_name = "bootstrap_macos.sh"
    elif sys.platform.startswith("linux"):
        script_name = "bootstrap_ubuntu.sh"
    else:
        raise ROSConfigError(
            f"`openral install ros` is only supported on Linux and macOS (detected {sys.platform})."
        )

    candidates: list[Path] = []
    env_root = os.environ.get("OPENRAL_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "scripts" / script_name)
    candidates.append(Path.cwd() / "scripts" / script_name)

    for path in candidates:
        if path.is_file():
            console.print(f"[yellow]about to run {path} (will prompt for sudo)[/yellow]")
            proc = subprocess.run(["bash", str(path)], check=False)
            if proc.returncode != 0:
                raise ROSConfigError(f"{script_name} exited {proc.returncode}; see output above.")
            console.print("[green]✓ system bootstrap complete[/green]")
            return

    raise ROSConfigError(
        f"Could not locate {script_name}. Clone the openral repo and set "
        f"OPENRAL_REPO_ROOT, or run from a workspace checkout:\n"
        f"  git clone https://github.com/OpenRAL/openral.git\n"
        f"  cd openral && openral install ros"
    )


@install_app.command("list")
def install_list() -> None:
    """List every group the installer knows about, with package counts."""
    table = Table(title="openral install — known groups")
    table.add_column("group", style="cyan")
    table.add_column("packages", justify="right")
    table.add_column("conflicts-with", style="yellow")
    for group, pkgs in sorted(_GROUPS.items()):
        conflicts = sorted(other for cs in _CONFLICTS if group in cs for other in (cs - {group}))
        table.add_row(group, str(len(pkgs)), ", ".join(conflicts) or "—")
    table.add_row("ros", "(sudo + apt)", "—")
    console.print(table)


__all__ = ["install_app"]
