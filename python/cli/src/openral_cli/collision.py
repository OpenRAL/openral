"""``openral collision lower|check`` — offline URDF/SRDF → manifest collision model.

Lowers a robot's URDF (geometry) and SRDF (allowed-collision matrix, when present;
random-pose sampling otherwise) into ``robot.yaml``'s ``collision_geometry`` +
``allowed_collision_pairs`` — the blocks the C++ safety kernel consumes via
``collision_params_from_description`` (ADR-0030). Because those manifests carry
hand-written safety commentary, the writer splices **only** the two collision
blocks, leaving every other line (and its comments) byte-for-byte intact.

``lower`` prints a unified diff by default and mutates only with ``--write``; a
regenerated ACM never changes silently (a safety input — CLAUDE.md §3). ``check``
fails (exit 1) when any manifest drifts from its lowered model.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

if TYPE_CHECKING:
    from openral_safety.urdf_lowering import LoweredCollisionModel

__all__ = [
    "collision_app",
    "inject_joint_fk",
    "render_blocks",
    "splice_collision_blocks",
]

_console = Console()


def _replace_block(text: str, key: str, new_block: str) -> str:
    """Replace the ``key:`` top-level block with ``new_block``, preserving neighbours.

    The block is the ``key:`` line plus all following indented or blank lines; it
    ends at the first column-0 non-blank line — whether that's the next top-level
    key OR a comment that introduces the next section. Trailing blank lines are
    returned to the following segment so the blank separator (and any column-0
    comment that documents the *next* block) survives the splice. When the key is
    absent (a manifest onboarded onto self-collision for the first time) the block
    is appended at the end of the file.
    """
    lines = text.splitlines(keepends=True)
    key_idx = next((i for i, ln in enumerate(lines) if ln.startswith(f"{key}:")), None)
    if key_idx is None:
        # Absent block (a manifest being onboarded onto self-collision for the
        # first time) → append at end of file.
        if not new_block.endswith("\n"):
            new_block += "\n"
        sep = "" if text.endswith("\n") or not text else "\n"
        return text + sep + "\n" + new_block
    # Absorb a contiguous run of comment lines immediately above the key (the
    # block's own header — a prior "# GENERATED" line or the hand comment that
    # documents this block) so repeated lowers replace it instead of stacking a
    # second header. Stops at the first blank / non-comment line, so a separator
    # blank and the preceding block stay put.
    start = key_idx
    while start - 1 >= 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    end = key_idx + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip() == "" or ln[0] in (" ", "\t"):  # blank or indented → in block
            end += 1
            continue
        break  # column-0 non-blank (next key or section comment) → block ends
    # Keep trailing blank lines as separators in the following segment.
    while end - 1 > key_idx and lines[end - 1].strip() == "":
        end -= 1
    if not new_block.endswith("\n"):
        new_block += "\n"
    return "".join(lines[:start]) + new_block + "".join(lines[end:])


def splice_collision_blocks(
    text: str, *, geometry_block: str | None = None, acm_block: str | None = None
) -> str:
    """Return ``text`` with the two collision blocks replaced (each optional).

    Only ``collision_geometry`` / ``allowed_collision_pairs`` are touched; every
    other key and comment is preserved verbatim.
    """
    if geometry_block is not None:
        text = _replace_block(text, "collision_geometry", geometry_block)
    if acm_block is not None:
        text = _replace_block(text, "allowed_collision_pairs", acm_block)
    return text


_JOINT_NAME_RE = re.compile(r'^(\s*)-\s*name:\s*["\']?([^"\'\s]+)')


# FK via matrix inverse leaves sub-nm noise; snap anything below this to zero
# (no real link offset is below a nanometre) for stable, reviewable output.
_FK_ZERO_SNAP_M = 1e-9


def _fmt(v: float) -> str:
    """Format an FK scalar cleanly (8 sig figs; snap float noise / -0.0 to 0.0)."""
    if abs(v) < _FK_ZERO_SNAP_M:
        v = 0.0
    return f"{v + 0.0:.8g}"


_Vec3 = tuple[float, float, float]


def inject_joint_fk(text: str, joint_fk: dict[str, tuple[_Vec3, _Vec3, _Vec3]]) -> str:
    """Inject ``origin_xyz`` / ``origin_rpy`` / ``axis_xyz`` into named joint blocks.

    For each joint in ``joint_fk`` (keyed by manifest joint name), find its
    ``- name: "<name>"`` list item under ``joints:``, drop any existing
    origin/axis lines in that item, and insert the lowered values right after the
    name line. The kernel needs these to place the link capsules (ADR-0030). Joints
    not in ``joint_fk`` are untouched. Idempotent: re-running drops and re-inserts
    the same lines. Every other line and comment is preserved.
    """
    if not joint_fk:
        return text
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        m = _JOINT_NAME_RE.match(ln)
        jname = m.group(2) if m else None
        if m is None or jname is None or jname not in joint_fk:
            out.append(ln)
            i += 1
            continue
        dash_indent = len(m.group(1))  # spaces before '-'
        field_indent = " " * (dash_indent + 2)
        xyz, rpy, axis = joint_fk[jname]
        out.append(ln)  # keep the name line
        out.append(f"{field_indent}origin_xyz: [{', '.join(_fmt(v) for v in xyz)}]\n")
        out.append(f"{field_indent}origin_rpy: [{', '.join(_fmt(v) for v in rpy)}]\n")
        out.append(f"{field_indent}axis_xyz: [{', '.join(_fmt(v) for v in axis)}]\n")
        i += 1
        # Copy the rest of this joint's block, dropping any pre-existing FK lines.
        while i < n:
            l2 = lines[i]
            indent = len(l2) - len(l2.lstrip())
            if l2.strip() != "" and indent <= dash_indent:
                break  # sibling list item or dedent → block ended
            if l2.lstrip().startswith(("origin_xyz:", "origin_rpy:", "axis_xyz:")):
                i += 1
                continue
            out.append(l2)
            i += 1
    return "".join(out)


def render_blocks(model: LoweredCollisionModel) -> tuple[str, str]:
    """Render a :class:`LoweredCollisionModel` to ``(geometry_block, acm_block)`` YAML.

    Both blocks open with a generated-provenance comment so a reader knows the tool
    owns them; floats are rounded to 4 dp for a stable, reviewable diff.
    """
    geo_lines = [
        "# GENERATED by `openral collision lower` — do not hand-edit.\n",
        "collision_geometry:\n",
    ]
    for g in model.collision_geometry:
        geo_lines.append(f'  - link_name: "{g.link_name}"\n')
        if g.shape.shape == "sphere":
            geo_lines.append(
                f'    shape: {{ shape: "sphere", radius_m: {g.shape.radius_m:.4f} }}\n'
            )
        else:
            geo_lines.append(
                f'    shape: {{ shape: "capsule", radius_m: {g.shape.radius_m:.4f}, '
                f"length_m: {g.shape.length_m:.4f} }}\n"
            )
        geo_lines.append(
            f"    origin_xyz_rpy: [{', '.join(f'{v:.4f}' for v in g.origin_xyz_rpy)}]\n"
        )

    acm_lines = [
        f"# GENERATED by `openral collision lower` (source: {model.acm_source}) — "
        "do not hand-edit.\n",
        "allowed_collision_pairs:\n",
    ]
    for a, b in model.allowed_collision_pairs:
        acm_lines.append(f"  - [{a}, {b}]\n")
    return "".join(geo_lines), "".join(acm_lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

collision_app = typer.Typer(
    name="collision",
    help="Lower a robot's URDF/SRDF into its self-collision model (ADR-0030).",
    no_args_is_help=True,
)


def _lowered_text(robot_path: Path, *, acm_only: bool, geometry_only: bool) -> tuple[str, str]:
    """Return ``(current_manifest_text, spliced_manifest_text)`` for a manifest.

    Shared by ``lower`` and ``check`` (and the regression tests). Loads the robot,
    lowers its collision model, renders the affected block(s), and splices them
    into the on-disk text — touching only the requested block(s).
    """
    from openral_core import RobotDescription
    from openral_safety.urdf_lowering import lower_robot

    robot = RobotDescription.from_yaml(str(robot_path))
    model = lower_robot(robot, acm_only=acm_only, geometry_only=geometry_only)
    geo_block, acm_block = render_blocks(model)
    current = robot_path.read_text(encoding="utf-8")
    # MJCF-sourced robots keep their hand-authored geometry (the tool reuses it,
    # doesn't regenerate it), so never rewrite the geometry block for them.
    write_geometry = not acm_only and model.acm_source != "mjcf"
    spliced = splice_collision_blocks(
        current,
        geometry_block=geo_block if write_geometry else None,
        acm_block=None if geometry_only else acm_block,
    )
    # When onboarding (not --acm-only), the kernel also needs each link's parent-joint
    # FK; inject it into the joints block.
    if not acm_only:
        spliced = inject_joint_fk(spliced, model.joint_fk)
    return current, spliced


@collision_app.command("lower")
def lower(
    robot: Path = typer.Option(..., "--robot", help="Path to a robot.yaml manifest."),
    write: bool = typer.Option(False, "--write", help="Apply the change (default: dry diff)."),
    acm_only: bool = typer.Option(
        False, "--acm-only", help="Only regenerate allowed_collision_pairs."
    ),
    geometry_only: bool = typer.Option(
        False, "--geometry-only", help="Only regenerate collision_geometry."
    ),
) -> None:
    """Lower URDF/SRDF → collision model. Prints a diff; mutates only with ``--write``.

    A regenerated allowed-collision matrix is a safety input — review the diff with
    the safety WG before merging (CLAUDE.md §3).
    """
    if acm_only and geometry_only:
        _console.print("[red]--acm-only and --geometry-only are mutually exclusive.[/red]")
        raise typer.Exit(code=2)
    if not robot.exists():
        _console.print(f"[red]Robot description not found:[/red] {robot}")
        raise typer.Exit(code=2)
    current, spliced = _lowered_text(robot, acm_only=acm_only, geometry_only=geometry_only)
    if current == spliced:
        _console.print("[green]No change — manifest already matches the lowered model.[/green]")
        return
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        spliced.splitlines(keepends=True),
        fromfile=f"{robot} (current)",
        tofile=f"{robot} (lowered)",
    )
    # markup=False / highlight=False: the diff body contains "[a, b]" ACM rows that
    # rich would otherwise parse as console-markup tags and drop.
    _console.print("".join(diff), markup=False, highlight=False)
    if write:
        robot.write_text(spliced, encoding="utf-8")
        _console.print(
            f"[green]Wrote[/green] {robot} — review the ACM diff with the safety WG (CLAUDE.md §3)."
        )
    else:
        _console.print("[yellow]Dry run.[/yellow] Re-run with [bold]--write[/bold] to apply.")


@collision_app.command("check")
def check(
    robot: Path | None = typer.Option(
        None, "--robot", help="A single robot.yaml; omit with --all."
    ),
    all_robots: bool = typer.Option(
        False, "--all", help="Check every robots/*/robot.yaml with collision geometry."
    ),
    acm_only: bool = typer.Option(False, "--acm-only", help="Only check allowed_collision_pairs."),
    geometry_only: bool = typer.Option(
        False, "--geometry-only", help="Only check collision_geometry."
    ),
) -> None:
    """Fail (exit 1) if any manifest drifts from its lowered collision model."""
    if acm_only and geometry_only:
        _console.print("[red]--acm-only and --geometry-only are mutually exclusive.[/red]")
        raise typer.Exit(code=2)
    if all_robots:
        targets = [
            p
            for p in sorted(Path("robots").glob("*/robot.yaml"))
            if "allowed_collision_pairs" in p.read_text(encoding="utf-8")
        ]
    elif robot is not None:
        targets = [robot]
    else:
        _console.print("[red]Pass --robot <path> or --all.[/red]")
        raise typer.Exit(code=2)

    drift: list[Path] = []
    for t in targets:
        try:
            current, spliced = _lowered_text(t, acm_only=acm_only, geometry_only=geometry_only)
        except (ValueError, FileNotFoundError) as exc:
            _console.print(f"[yellow]skip[/yellow] {t}: {exc}")
            continue
        if current != spliced:
            drift.append(t)
            _console.print(f"[red]drift[/red] {t} differs from its lowered model")
    if drift:
        _console.print(
            f"[red]{len(drift)} manifest(s) drifted — run "
            "`openral collision lower --robot <path> --write`.[/red]"
        )
        raise typer.Exit(code=1)
    _console.print("[green]All checked manifests match their lowered collision model.[/green]")
