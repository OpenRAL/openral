"""``openral collision lower|check`` — offline URDF/SRDF → manifest collision model.

Lowers a robot's URDF (geometry) and SRDF (allowed-collision matrix, when present;
random-pose sampling otherwise) into ``robot.yaml``'s ``collision_geometry`` +
``allowed_collision_pairs`` — the blocks the C++ safety kernel consumes via
``collision_params_from_description``. Because those manifests carry
hand-written safety commentary, the writer splices **only** the two collision
blocks, leaving every other line (and its comments) byte-for-byte intact.

``lower`` prints a unified diff by default and mutates only with ``--write``; a
regenerated ACM never changes silently (a safety input — CLAUDE.md §3). ``check``
fails (exit 1) when any manifest drifts from its lowered model.

**A re-lower may not silently loosen a hand-tightened collision model.**
``urdf_lowering.lower_link_geometry`` emits a PCA bounding capsule for any mesh
collision — correct and conservative for onboarding, but strictly looser than a
hand-fitted oriented box. ``panda_mobile`` carries boxes (#103); re-lowering it
from ``rd:panda_description`` today would replace all seven with capsules of
**1.9-3.7x the volume** and **1.4-1.5x the circumradius**, undoing that work in a
block the writer stamps ``# GENERATED``. So ``lower`` measures the new geometry
against what the manifest already carries (:func:`geometry_loosening`) and
**refuses to write** a looser one. There is deliberately no override flag
(CLAUDE.md §3, "never add a flag that disables safety"): abandoning tighter
geometry means deleting it from the manifest first, which is a reviewable diff
rather than a switch.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from openral_core.exceptions import ROSError
from rich.console import Console

if TYPE_CHECKING:
    from openral_core import CollisionShape, LinkCollisionGeometry, RobotDescription
    from openral_safety.urdf_lowering import LoweredCollisionModel

__all__ = [
    "GeometryLoosening",
    "collision_app",
    "collision_primitive_envelope",
    "geometry_loosening",
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


# ── Loosening guard: a re-lower must not quietly widen a hand-fitted primitive ─


@dataclass(frozen=True)
class GeometryLoosening:
    """One link whose newly lowered primitive claims more space than the shipped one.

    A CLI report record, not a schema: it is never persisted, published or parsed,
    so a frozen dataclass is the right weight (CLAUDE.md §2 — Pydantic is for
    schemas and external interfaces).

    Attributes:
        link_name: the manifest ``collision_geometry`` entry's link.
        shipped: ``"<kind> vol=<m³> circ=<m>"`` for the committed primitive, or
            ``"box"``/``"capsule"``/``"sphere"`` rendering of it.
        lowered: the same for what the tool would write. Empty when the tool
            emits nothing for this link, i.e. the link would lose its geometry
            entirely.
        volume_ratio: lowered ÷ shipped enclosed volume. ``inf`` when a link
            would be dropped.
        circumradius_ratio: lowered ÷ shipped circumradius; ``inf`` likewise.
    """

    link_name: str
    shipped: str
    lowered: str
    volume_ratio: float
    circumradius_ratio: float


def collision_primitive_envelope(shape: CollisionShape) -> tuple[float, float]:
    """``(volume_m³, circumradius_m)`` — how much space one collision primitive claims.

    The two numbers a "did this get looser?" comparison needs, and the only two
    that are exact in closed form for every member of the ``CollisionShape``
    union. **Volume** is total claimed space — the term that drives how often the
    kernel flags a link against occupancy it is not really touching.
    **Circumradius** (the primitive's own bounding-sphere radius, about its own
    centre) is maximum extent — the term volume alone can hide, since a long thin
    primitive can reach further on less volume. Both are independent of where
    ``origin_xyz_rpy`` places the primitive and of how it is rotated, so the
    comparison is about the primitive and never about its placement.

    Args:
        shape: a ``BoxShape``, ``CapsuleShape`` or ``SphereShape``.

    Returns:
        ``(volume_m3, circumradius_m)``.

    Raises:
        ROSConfigError: the shape is none of the three known primitives — a new
            one must be measured here before it can be compared, never silently
            scored zero.

    Example:
        >>> from openral_core import BoxShape
        >>> v, c = collision_primitive_envelope(BoxShape(half_extents_m=(0.5, 0.5, 0.5)))
        >>> round(v, 3), round(c, 3)
        (1.0, 0.866)
    """
    from openral_core import BoxShape, CapsuleShape, SphereShape
    from openral_core.exceptions import ROSConfigError

    if isinstance(shape, BoxShape):
        hx, hy, hz = (float(v) for v in shape.half_extents_m)
        return 8.0 * hx * hy * hz, math.sqrt(hx * hx + hy * hy + hz * hz)
    if isinstance(shape, SphereShape):
        r = float(shape.radius_m)
        return 4.0 / 3.0 * math.pi * r**3, r
    if isinstance(shape, CapsuleShape):
        r, length = float(shape.radius_m), float(shape.length_m)
        return math.pi * r * r * length + 4.0 / 3.0 * math.pi * r**3, length / 2.0 + r
    raise ROSConfigError(
        f"cannot measure collision primitive of kind {shape.shape!r}: add it to "
        "openral_cli.collision.collision_primitive_envelope before comparing it."
    )


# `render_blocks` writes every primitive scalar with `:.4f`. A committed manifest
# therefore carries QUANTIZED values while a fresh lowering carries full-precision
# ones, so comparing the two directly makes every tool-generated manifest look
# 1.000x looser than itself. The comparison is between what is written and what
# WOULD be written, so both sides are quantized the same way first.
_BLOCK_FLOAT_DP = 4


def _writer_quantized(shape: CollisionShape) -> CollisionShape:
    """One primitive rounded to the precision ``render_blocks`` writes it at."""
    from openral_core import BoxShape, SphereShape

    if isinstance(shape, BoxShape):
        return shape.model_copy(
            update={
                "half_extents_m": tuple(
                    round(float(v), _BLOCK_FLOAT_DP) for v in shape.half_extents_m
                )
            }
        )
    if isinstance(shape, SphereShape):
        return shape.model_copy(update={"radius_m": round(float(shape.radius_m), _BLOCK_FLOAT_DP)})
    return shape.model_copy(
        update={
            "radius_m": round(float(shape.radius_m), _BLOCK_FLOAT_DP),
            "length_m": round(float(shape.length_m), _BLOCK_FLOAT_DP),
        }
    )


def _render_primitive(shape: CollisionShape) -> str:
    """One primitive as a short human string, with its measured envelope."""
    from openral_core import BoxShape, SphereShape

    volume, circ = collision_primitive_envelope(shape)
    if isinstance(shape, BoxShape):
        body = "box h=[" + ", ".join(f"{float(v):.4f}" for v in shape.half_extents_m) + "]"
    elif isinstance(shape, SphereShape):
        body = f"sphere r={float(shape.radius_m):.4f}"
    else:
        body = f"capsule r={float(shape.radius_m):.4f} L={float(shape.length_m):.4f}"
    return f"{body} (vol {volume * 1e6:.1f} cm³, circ {circ * 1e3:.1f} mm)"


def geometry_loosening(
    shipped: list[LinkCollisionGeometry],
    lowered: list[LinkCollisionGeometry],
) -> list[GeometryLoosening]:
    """Links where re-lowering would claim more space than the committed manifest.

    The guard behind ``lower``'s refusal. A link is reported when the lowered
    primitive's volume **or** circumradius exceeds the shipped one's — either is
    enough, because they fail in different directions (a capsule that swallows a
    box has more volume; one fitted to a long link reaches further). Equality is
    not a regression, so a manifest already equal to the tool's output — every
    ``# GENERATED`` manifest in ``robots/`` — reports nothing.

    A shipped link the tool emits *no* primitive for is reported with infinite
    ratios: losing a link's geometry outright removes it from the kernel's
    collision model, which is the worst version of the same failure.

    A link the tool adds and the manifest lacks is **not** reported: new coverage
    is not a loosening.

    Args:
        shipped: the manifest's committed ``collision_geometry``.
        lowered: what ``openral collision lower`` would write in its place.

    Returns:
        One :class:`GeometryLoosening` per affected link, worst volume ratio
        first. Empty when nothing would get looser.

    Example:
        >>> geometry_loosening([], [])
        []
    """
    by_link = {g.link_name: g for g in lowered}
    out: list[GeometryLoosening] = []
    for entry in shipped:
        old_volume, old_circ = collision_primitive_envelope(_writer_quantized(entry.shape))
        new = by_link.get(entry.link_name)
        if new is None:
            out.append(
                GeometryLoosening(
                    link_name=entry.link_name,
                    shipped=_render_primitive(entry.shape),
                    lowered="",
                    volume_ratio=math.inf,
                    circumradius_ratio=math.inf,
                )
            )
            continue
        new_volume, new_circ = collision_primitive_envelope(_writer_quantized(new.shape))
        if new_volume <= old_volume and new_circ <= old_circ:
            continue
        out.append(
            GeometryLoosening(
                link_name=entry.link_name,
                shipped=_render_primitive(entry.shape),
                lowered=_render_primitive(new.shape),
                volume_ratio=new_volume / old_volume if old_volume > 0.0 else math.inf,
                circumradius_ratio=new_circ / old_circ if old_circ > 0.0 else math.inf,
            )
        )
    out.sort(key=lambda r: (-r.volume_ratio, r.link_name))
    return out


def _report_loosening(robot_path: Path, findings: list[GeometryLoosening]) -> None:
    """Print the loosening table. Says what was measured, not merely that it failed."""
    _console.print(
        f"[red]Refusing to loosen {robot_path}[/red] — the lowered geometry claims more "
        f"space than the committed manifest on {len(findings)} link(s):"
    )
    for f in findings:
        _console.print(f"  [bold]{f.link_name}[/bold]", markup=True)
        _console.print(f"      shipped: {f.shipped}", markup=False, highlight=False)
        _console.print(
            f"      lowered: {f.lowered or '(no primitive — the link would lose its geometry)'}",
            markup=False,
            highlight=False,
        )
        _console.print(
            f"      volume x{f.volume_ratio:.2f}, circumradius x{f.circumradius_ratio:.2f}",
            markup=False,
            highlight=False,
        )
    _console.print(
        "[yellow]`lower_link_geometry` PCA-fits a bounding capsule to any mesh collision; "
        "that is conservative for onboarding but looser than hand-fitted boxes. To adopt "
        "the lowered geometry anyway, delete the affected `collision_geometry` entries "
        "from the manifest first — there is no override flag (CLAUDE.md §3).[/yellow]"
    )


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
    name line. The kernel needs these to place the link capsules. Joints
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
        elif g.shape.shape == "box":
            hx, hy, hz = g.shape.half_extents_m
            geo_lines.append(
                f'    shape: {{ shape: "box", half_extents_m: [{hx:.4f}, {hy:.4f}, {hz:.4f}] }}\n'
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
    help="Lower a robot's URDF/SRDF into its self-collision model.",
    no_args_is_help=True,
)


def _lower(
    robot_path: Path, *, acm_only: bool, geometry_only: bool
) -> tuple[RobotDescription, LoweredCollisionModel]:
    """Load a manifest and lower its collision model via the provenance dispatcher.

    Route via the provenance-correct dispatcher: SRDF+URDF → SRDF
    ACM, URDF-with-usable-meshes → sampling, MJCF-native → MJCF. The naive
    ``urdf if assets.urdf else mjcf`` wrongly sent openarm (unusable URDF meshes)
    to the URDF path. The byte-identical regression test routes through this same
    dispatcher.
    """
    from openral_core import RobotDescription
    from openral_safety.urdf_lowering import lower_robot_auto

    robot = RobotDescription.from_yaml(str(robot_path))
    model = lower_robot_auto(
        robot, acm_only=acm_only, geometry_only=geometry_only, manifest_dir=robot_path.parent
    )
    return robot, model


def _lowered_text(
    robot_path: Path, *, acm_only: bool, geometry_only: bool
) -> tuple[str, str, list[GeometryLoosening]]:
    """``(current_manifest_text, spliced_manifest_text, loosening)`` for a manifest.

    Shared by ``lower`` and ``check`` (and the regression tests). Loads the robot,
    lowers its collision model, renders the affected block(s), and splices them
    into the on-disk text — touching only the requested block(s).

    The third element is the geometry-loosening report
    (:func:`geometry_loosening`), computed here because this is the one place
    that holds the committed manifest and the tool's replacement for it at the
    same time. It is empty whenever the geometry block would not be rewritten at
    all (``--acm-only``, or an MJCF-sourced robot whose hand geometry the tool
    reuses rather than regenerates), so a caller cannot mistake "not compared"
    for "compared and clean".
    """
    robot, model = _lower(robot_path, acm_only=acm_only, geometry_only=geometry_only)
    geo_block, acm_block = render_blocks(model)
    current = robot_path.read_text(encoding="utf-8")
    # MJCF-sourced robots keep their hand-authored geometry (the tool reuses it,
    # doesn't regenerate it), so never rewrite the geometry block for them.
    write_geometry = not acm_only and model.acm_source != "mjcf"
    loosening = (
        geometry_loosening(list(robot.collision_geometry or []), list(model.collision_geometry))
        if write_geometry
        else []
    )
    spliced = splice_collision_blocks(
        current,
        geometry_block=geo_block if write_geometry else None,
        acm_block=None if geometry_only else acm_block,
    )
    # When onboarding (not --acm-only), the kernel also needs each link's parent-joint
    # FK; inject it into the joints block.
    if not acm_only:
        spliced = inject_joint_fk(spliced, model.joint_fk)
    return current, spliced, loosening


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
    emit_cumotion: Path | None = typer.Option(
        None,
        "--emit-cumotion",
        help="Also write a cuRobo robot-config (collision spheres + ACM) to this path.",
    ),
) -> None:
    """Lower URDF/SRDF → collision model. Prints a diff; mutates only with ``--write``.

    A regenerated allowed-collision matrix is a safety input — review the diff with
    the safety WG before merging (CLAUDE.md §3). ``--emit-cumotion <path>`` also
    derives a cuRobo robot-config from the *same* lowered geometry so cuMotion's
    plan-time collision matches the kernel's; it writes only with
    ``--write`` (dry run prints the config).
    """
    if acm_only and geometry_only:
        _console.print("[red]--acm-only and --geometry-only are mutually exclusive.[/red]")
        raise typer.Exit(code=2)
    if not robot.exists():
        _console.print(f"[red]Robot description not found:[/red] {robot}")
        raise typer.Exit(code=2)
    if emit_cumotion is not None:
        from openral_safety.cumotion_config import render_cumotion_config

        desc, model = _lower(robot, acm_only=False, geometry_only=False)
        config_text = render_cumotion_config(desc, model)
        if write:
            emit_cumotion.write_text(config_text, encoding="utf-8")
            _console.print(f"[green]Wrote[/green] cuRobo config → {emit_cumotion}")
        else:
            _console.print(config_text, markup=False, highlight=False)
            _console.print(
                f"[yellow]Dry run.[/yellow] Re-run with [bold]--write[/bold] to write "
                f"{emit_cumotion}."
            )
    current, spliced, loosening = _lowered_text(
        robot, acm_only=acm_only, geometry_only=geometry_only
    )
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
    if loosening:
        # Reported on a dry run too: a reader comparing the diff by eye cannot see
        # that a capsule swallows the box it replaces, which is the whole trap.
        _report_loosening(robot, loosening)
    if write:
        if loosening:
            _console.print(f"[red]Not written.[/red] {robot} is unchanged.")
            raise typer.Exit(code=3)
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
            current, spliced, loosening = _lowered_text(
                t, acm_only=acm_only, geometry_only=geometry_only
            )
        except (ValueError, FileNotFoundError, ROSError) as exc:
            # ROSError covers ROSConfigError, now raised by lower_robot when a
            # manifest declares no URDF/MJCF asset or an asset ref won't resolve.
            _console.print(f"[yellow]skip[/yellow] {t}: {exc}")
            continue
        if current != spliced:
            drift.append(t)
            _console.print(f"[red]drift[/red] {t} differs from its lowered model")
            if loosening:
                # The remedy line below tells the reader to re-lower. On these
                # manifests that would make the model LOOSER, and `lower --write`
                # will refuse — say so here rather than sending them into it.
                _console.print(
                    f"       …but the lowered geometry is looser on "
                    f"{len(loosening)} link(s) "
                    f"({', '.join(f.link_name for f in loosening)}); "
                    "`lower --write` will refuse. This manifest's geometry is "
                    "hand-tightened on purpose."
                )
    if drift:
        _console.print(
            f"[red]{len(drift)} manifest(s) drifted — run "
            "`openral collision lower --robot <path> --write`.[/red]"
        )
        raise typer.Exit(code=1)
    _console.print("[green]All checked manifests match their lowered collision model.[/green]")
