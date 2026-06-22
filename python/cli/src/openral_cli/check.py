"""``openral check`` — static validation of the declarative robot/skill/scene set.

Loads every ``robots/*/robot.yaml``, ``rskills/*/rskill.yaml``, and
``scenes/{deploy,sim,benchmark}/*.yaml`` and cross-validates them in one pass:
every manifest parses, every ``file:`` / ``ros2://`` asset ref resolves, every
scene ``robot_id`` resolves to a real robot directory, every rSkill's embodiment
tags reach at least one in-repo robot, and every sensor ``parent_frame`` is a
declared tf2 frame. No schema change — pure reuse of the existing Pydantic
contracts (``RobotDescription.from_yaml`` / ``resolve_asset`` / the scene tiers).

A static, host-independent lint (no hardware detection), complementing the
host-specific ``openral rskill check`` compatibility report. JSON-Schema emission
for the manifests lives in ``tools/schema_export.py`` (CI-gated via
``quality.yml``), not here.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Literal

import typer
import yaml
from openral_core.assets import AssetRefError, resolve_asset
from openral_core.exceptions import ROSError
from openral_core.schemas import (
    BenchmarkScene,
    DeployScene,
    RobotDescription,
    RSkillManifest,
    SimScene,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.console import Console
from rich.table import Table

__all__ = [
    "CheckFinding",
    "GraphCheckReport",
    "check_command",
    "check_description_graph",
]

_console = Console()

# ── Graph validation ────────────────────────────────────────────────────────────

CheckRule = Literal[
    "robot_parse",
    "rskill_parse",
    "scene_parse",
    "asset_ref",
    "scene_robot_id",
    "embodiment_reach",
    "frames",
]
Severity = Literal["error", "warning"]

# Asset-ref prefixes that download a package or need a sim-only dep; skipped by
# default so the check runs offline in CI. ``file:`` / ``ros2://`` resolve
# locally and are always checked.
_REMOTE_ASSET_PREFIXES = ("rd:", "gym_aloha:", "openarm:", "menagerie:")

_SCENE_TIERS: dict[str, type[DeployScene]] = {
    "deploy": DeployScene,
    "sim": SimScene,
    "benchmark": BenchmarkScene,
}

# Exceptions a ``from_yaml`` / ``model_validate`` may raise for a bad manifest.
_LOAD_ERRORS = (ValidationError, ROSError, OSError, ValueError, yaml.YAMLError)


class CheckFinding(BaseModel):
    """One problem surfaced by :func:`check_description_graph`."""

    model_config = ConfigDict(extra="forbid")

    rule: CheckRule
    severity: Severity
    target: str
    message: str


class GraphCheckReport(BaseModel):
    """Typed result of ``openral check graph`` (also the ``--json`` payload)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    generated_at: str
    n_robots: int = 0
    n_rskills: int = 0
    n_scenes: int = 0
    findings: list[CheckFinding] = Field(default_factory=list)

    @property
    def errors(self) -> list[CheckFinding]:
        """Findings that fail the check."""
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[CheckFinding]:
        """Findings that are advisory (do not fail unless ``--strict``)."""
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when there are no error-severity findings."""
        return not self.errors


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _error(rule: CheckRule, target: str, message: str) -> CheckFinding:
    return CheckFinding(rule=rule, severity="error", target=target, message=message)


def _warning(rule: CheckRule, target: str, message: str) -> CheckFinding:
    return CheckFinding(rule=rule, severity="warning", target=target, message=message)


def _exc(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _asset_refs(robot: RobotDescription) -> list[tuple[Literal["urdf", "mjcf", "srdf"], str]]:
    """Collect the (kind, ref) asset references declared on a robot."""
    refs: list[tuple[Literal["urdf", "mjcf", "srdf"], str]] = []
    assets = robot.assets
    if assets.urdf is not None:
        refs.append(("urdf", assets.urdf.ref))
    if assets.mjcf is not None:
        refs.append(("mjcf", assets.mjcf))
    if assets.srdf is not None:
        refs.append(("srdf", assets.srdf))
    return refs


_URDF_LINK_RE = re.compile(r'<link\s+name="([^"]+)"')


def _urdf_link_names(urdf_path: Path) -> set[str]:
    """Link names declared in a URDF — the tf frames robot_state_publisher emits."""
    try:
        text = urdf_path.read_text(encoding="utf-8")
    except OSError:
        return set()
    return set(_URDF_LINK_RE.findall(text))


def _declared_frames(robot: RobotDescription, manifest_dir: Path) -> set[str]:
    """tf2 frames a robot's manifest + local URDF declare.

    ``robot_state_publisher`` emits one tf frame per URDF link, so a local
    ``file:`` URDF is the authoritative frame source. The manifest's
    base/odom/map frames, joint links, and sensor frames augment it (and are the
    only source for sim-only robots whose URDF is a remote ``rd:`` ref we don't
    resolve offline).
    """
    frames = {robot.base_frame, robot.odom_frame, robot.map_frame}
    for joint in robot.joints:
        frames.add(joint.parent_link)
        frames.add(joint.child_link)
    for sensor in robot.sensors:
        frames.add(sensor.frame_id)
    urdf = robot.assets.urdf
    if urdf is not None:
        if urdf.root_frame is not None:
            frames.add(urdf.root_frame)
        if urdf.ref.startswith("file:"):
            try:
                urdf_path = resolve_asset(urdf.ref, "urdf", manifest_dir=manifest_dir)
            except AssetRefError:
                urdf_path = None
            if urdf_path is not None:
                frames |= _urdf_link_names(urdf_path)
    return frames


def _check_robots(
    repo_root: Path, *, resolve_remote_assets: bool
) -> tuple[dict[str, RobotDescription], set[str], list[CheckFinding]]:
    """Parse every robot manifest, resolve its asset refs, and check sensor frames."""
    findings: list[CheckFinding] = []
    robots: dict[str, RobotDescription] = {}
    robot_tags: set[str] = set()
    robots_dir = repo_root / "robots"
    for path in sorted(robots_dir.glob("*/robot.yaml")) if robots_dir.is_dir() else []:
        robot_id = path.parent.name
        try:
            robot = RobotDescription.from_yaml(str(path))
        except _LOAD_ERRORS as exc:
            findings.append(_error("robot_parse", f"robots/{robot_id}", _exc(exc)))
            continue
        robots[robot_id] = robot
        robot_tags |= set(robot.capabilities.embodiment_tags)
        findings.extend(_check_assets(robot_id, robot, path.parent, resolve_remote_assets))
        findings.extend(_check_frames(robot_id, robot, path.parent))
    return robots, robot_tags, findings


def _check_assets(
    robot_id: str, robot: RobotDescription, manifest_dir: Path, resolve_remote_assets: bool
) -> list[CheckFinding]:
    findings: list[CheckFinding] = []
    for kind, ref in _asset_refs(robot):
        if ref.startswith(_REMOTE_ASSET_PREFIXES) and not resolve_remote_assets:
            continue
        try:
            resolve_asset(ref, kind, manifest_dir=manifest_dir)
        except AssetRefError as exc:
            findings.append(_error("asset_ref", f"robots/{robot_id} [{kind}]", str(exc)))
    return findings


def _check_frames(robot_id: str, robot: RobotDescription, manifest_dir: Path) -> list[CheckFinding]:
    declared = _declared_frames(robot, manifest_dir)
    return [
        _warning(
            "frames",
            f"robots/{robot_id} [sensor:{sensor.name}]",
            f"parent_frame {sensor.parent_frame!r} is not a declared frame "
            "(base/odom/map, a joint link, a sensor frame, or the URDF root); "
            "confirm it is published by robot_state_publisher",
        )
        for sensor in robot.sensors
        if sensor.parent_frame is not None and sensor.parent_frame not in declared
    ]


def _check_rskills(repo_root: Path, robot_tags: set[str]) -> tuple[int, list[CheckFinding]]:
    """Parse every rSkill manifest and check its embodiment tags reach a robot."""
    findings: list[CheckFinding] = []
    manifests: list[tuple[str, RSkillManifest]] = []
    rskills_dir = repo_root / "rskills"
    for path in sorted(rskills_dir.glob("*/rskill.yaml")) if rskills_dir.is_dir() else []:
        rskill_id = path.parent.name
        try:
            manifest = RSkillManifest.from_yaml(str(path))
        except _LOAD_ERRORS as exc:
            findings.append(_error("rskill_parse", f"rskills/{rskill_id}", _exc(exc)))
            continue
        manifests.append((rskill_id, manifest))

    for rskill_id, manifest in manifests:
        tags = set(manifest.embodiment_tags)
        # Perception kinds (detector / vlm) legitimately ship no tags (match-any);
        # only flag a non-empty tag set that intersects no robot in the repo.
        if tags and not (tags & robot_tags):
            findings.append(
                _warning(
                    "embodiment_reach",
                    f"rskills/{rskill_id}",
                    f"embodiment_tags {sorted(tags)} intersect no in-repo robot "
                    "(intended for an external embodiment, or a typo)",
                )
            )
    return len(manifests), findings


def _check_scenes(
    repo_root: Path, robots: dict[str, RobotDescription]
) -> tuple[int, list[CheckFinding]]:
    """Parse every scene per tier and check its ``robot_id`` resolves to a robot dir."""
    findings: list[CheckFinding] = []
    n_scenes = 0
    for tier, model in _SCENE_TIERS.items():
        tier_dir = repo_root / "scenes" / tier
        for path in sorted(tier_dir.glob("*.yaml")) if tier_dir.is_dir() else []:
            n_scenes += 1
            target = f"scenes/{tier}/{path.name}"
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                findings.append(_error("scene_parse", target, _exc(exc)))
                continue
            if not isinstance(data, dict):
                continue  # an eval suite (bare list), not a scene — skip
            try:
                scene = model.model_validate(data)
            except _LOAD_ERRORS as exc:
                findings.append(_error("scene_parse", target, _exc(exc)))
                continue
            if scene.robot_id is not None and scene.robot_id not in robots:
                findings.append(
                    _error(
                        "scene_robot_id",
                        target,
                        f"robot_id={scene.robot_id!r} has no robots/{scene.robot_id}/robot.yaml",
                    )
                )
    return n_scenes, findings


def check_description_graph(
    repo_root: Path, *, resolve_remote_assets: bool = False
) -> GraphCheckReport:
    """Validate every robot / rSkill / scene manifest under ``repo_root``.

    Args:
        repo_root: Directory holding ``robots/``, ``rskills/``, and ``scenes/``.
        resolve_remote_assets: When ``True``, also resolve ``rd:`` /
            ``gym_aloha:`` / ``openarm:`` / ``menagerie:`` refs (downloads /
            sim-only imports). Default ``False`` keeps the check offline.

    Returns:
        A :class:`GraphCheckReport`; ``report.ok`` is ``False`` when any
        error-severity finding was raised.

    Example:
        >>> # report = check_description_graph(Path("."))
        >>> # report.ok
    """
    robots, robot_tags, findings = _check_robots(
        repo_root, resolve_remote_assets=resolve_remote_assets
    )
    n_rskills, rskill_findings = _check_rskills(repo_root, robot_tags)
    n_scenes, scene_findings = _check_scenes(repo_root, robots)
    findings.extend(rskill_findings)
    findings.extend(scene_findings)
    return GraphCheckReport(
        generated_at=_now(),
        n_robots=len(robots),
        n_rskills=n_rskills,
        n_scenes=n_scenes,
        findings=findings,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────────

_SEVERITY_STYLE = {"error": "red", "warning": "yellow"}


def _render_graph_report(report: GraphCheckReport) -> None:
    _console.print(
        f"Checked [bold]{report.n_robots}[/bold] robots, "
        f"[bold]{report.n_rskills}[/bold] rSkills, "
        f"[bold]{report.n_scenes}[/bold] scenes."
    )
    if not report.findings:
        _console.print("[green]All manifests valid and cross-references resolve.[/green]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("severity")
    table.add_column("rule")
    table.add_column("target")
    table.add_column("message")
    for finding in report.findings:
        style = _SEVERITY_STYLE[finding.severity]
        table.add_row(
            f"[{style}]{finding.severity}[/{style}]",
            finding.rule,
            finding.target,
            finding.message,
        )
    _console.print(table)
    _console.print(
        f"[red]{len(report.errors)} error(s)[/red], "
        f"[yellow]{len(report.warnings)} warning(s)[/yellow]."
    )


def check_command(
    repo_root: Path = typer.Option(
        Path("."), "--repo-root", help="Dir holding robots/, rskills/, scenes/."
    ),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as failures too."),
    resolve_remote_assets: bool = typer.Option(
        False,
        "--resolve-remote-assets",
        help="Also resolve rd:/gym_aloha:/openarm: refs (downloads / sim deps).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the GraphCheckReport as JSON."),
) -> None:
    """Cross-validate every robot, rSkill, and scene manifest in one pass.

    Exits 1 on any error finding (a manifest that fails to parse, an unresolvable
    asset ref, or a scene whose ``robot_id`` names no robot). Warnings (an
    unreachable embodiment tag, an undeclared sensor frame) only fail under
    ``--strict``.
    """
    report = check_description_graph(
        repo_root.resolve(), resolve_remote_assets=resolve_remote_assets
    )
    if json_out:
        _console.print_json(report.model_dump_json())
    else:
        _render_graph_report(report)
    if not report.ok or (strict and report.warnings):
        raise typer.Exit(code=1)
