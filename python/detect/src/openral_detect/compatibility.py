"""rSkill compatibility report — `check_installed_rskills`.

Walks installed rSkills and reports which can run on the assembled
:class:`RobotDescription`.  Reuses the production
:meth:`rSkill.check_compatibility` so the report uses the same semantics
as the runtime loader — no parallel logic.

When the local registry is empty (a fresh checkout that has not run
``ral skill install``), pass ``rskills_dir=Path("rskills")`` to walk the
in-tree manifests for CI / development.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Literal

from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_core.schemas import RobotDescription, RSkillManifest
from openral_rskill.loader import InstalledRSkillEntry, rSkill
from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "CompatibilityReport",
    "FailureKind",
    "RSkillCompatRow",
    "SectionVerdict",
    "check_installed_rskills",
    "check_single_rskill",
]


FailureKind = Literal[
    "embodiment_tag",
    "capability_flag",
    "runtime",
    "quantization",
    "sensor_modality",
    "sensor_key",
    "resolution",
    "manifest_load",
]


SectionLabel = Literal[
    "embodiment",
    "capability_flags",
    "gpu_runtime",
    "gpu_dtype",
    "sensors",
    "actuators",
]


class SectionVerdict(BaseModel):
    """Per-section verdict for ``openral rskill check <rskill_id>``.

    A single row of the per-section breakdown emitted when a caller asks
    for compatibility of one specific rSkill against the host's detected
    :class:`RobotDescription`.  Six sections cover the dimensions a user
    will reasonably ask about: embodiment, capability flags, GPU runtime,
    GPU dtype, sensors, actuators.

    Sections are populated in deterministic order so JSON consumers can
    rely on positional indexing.  ``compatible=True`` with
    ``failure_kind=None`` means the section was either satisfied or
    skipped because the robot has not declared that capability surface
    (e.g. empty ``gpu_supported_runtimes`` ⇒ "unknown — skip").
    """

    model_config = ConfigDict(extra="forbid")

    label: SectionLabel
    compatible: bool
    reason: str | None = None
    failure_kind: FailureKind | None = None
    informational: bool = False


class RSkillCompatRow(BaseModel):
    """One installed-skill x robot pairing in the report."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str
    version: str = ""
    role: str = "s1"
    manifest_path: str = ""
    embodiment_tags: list[str] = Field(default_factory=list)
    compatible: bool
    reason: str | None = None
    failure_kind: FailureKind | None = None
    sections: list[SectionVerdict] = Field(default_factory=list)


class CompatibilityReport(BaseModel):
    """``openral rskill check`` output."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    generated_at: str
    robot_name: str
    robot_embodiment_tags: list[str] = Field(default_factory=list)
    rows: list[RSkillCompatRow] = Field(default_factory=list)

    @property
    def compatible(self) -> list[RSkillCompatRow]:
        """Rows that passed every check."""
        return [r for r in self.rows if r.compatible]

    @property
    def incompatible(self) -> list[RSkillCompatRow]:
        """Rows that failed at least one check."""
        return [r for r in self.rows if not r.compatible]


def check_installed_rskills(
    robot: RobotDescription,
    *,
    registry_path: Path | None = None,
    rskills_dir: Path | None = None,
) -> CompatibilityReport:
    """Run :meth:`rSkill.check_compatibility` against every installed skill.

    Args:
        robot: Assembled :class:`RobotDescription` (typically the output
            of :func:`assemble_robot_description`).
        registry_path: Optional override for the local rSkill registry
            file.  Default: the user's per-host registry resolved by
            :meth:`rSkill.list_installed`.
        rskills_dir: Optional path to walk for in-tree ``rskill.yaml``
            files (CI / development convenience).  Each yaml found is
            evaluated alongside the entries from the local registry.

    Returns:
        A :class:`CompatibilityReport` ready for table rendering.
    """
    rows: list[RSkillCompatRow] = []
    seen_paths: set[str] = set()

    for entry in rSkill.list_installed(registry_path=registry_path):
        rows.append(_evaluate_entry(entry, robot, seen_paths))

    if rskills_dir is not None:
        for manifest_path in sorted(Path(rskills_dir).glob("*/rskill.yaml")):
            if str(manifest_path) in seen_paths:
                continue
            entry = _entry_for_in_tree(manifest_path)
            rows.append(_evaluate_entry(entry, robot, seen_paths))

    generated_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    return CompatibilityReport(
        generated_at=generated_at,
        robot_name=robot.name,
        robot_embodiment_tags=list(robot.capabilities.embodiment_tags),
        rows=rows,
    )


def _entry_for_in_tree(manifest_path: Path) -> InstalledRSkillEntry:
    """Build a synthetic :class:`InstalledRSkillEntry` for an in-tree manifest.

    Used by ``--rskills-dir`` so we can run compatibility against a fresh
    checkout that has not run ``ral skill install``.
    """
    return InstalledRSkillEntry(
        repo_id=f"in-tree:{manifest_path.parent.name}",
        version="0.0.0",
        revision=None,
        local_dir=str(manifest_path.parent),
        manifest_path=str(manifest_path),
        license="unknown",
        role="s1",
        embodiment_tags=[],
        installed_at="1970-01-01T00:00:00+00:00",
    )


def _evaluate_entry(
    entry: InstalledRSkillEntry,
    robot: RobotDescription,
    seen_paths: set[str],
) -> RSkillCompatRow:
    import yaml  # noqa: PLC0415  # reason: deferred to keep top-level imports light

    seen_paths.add(entry.manifest_path)
    try:
        manifest = RSkillManifest.from_yaml(entry.manifest_path)
    except (FileNotFoundError, ValidationError, OSError, yaml.YAMLError) as exc:
        return RSkillCompatRow(
            repo_id=entry.repo_id,
            version=entry.version,
            role=entry.role,
            manifest_path=entry.manifest_path,
            embodiment_tags=entry.embodiment_tags,
            compatible=False,
            reason=f"manifest load failed: {exc!r}",
            failure_kind="manifest_load",
        )
    try:
        rSkill.check_compatibility(manifest, robot)
    except ROSCapabilityMismatch as exc:
        return RSkillCompatRow(
            repo_id=entry.repo_id,
            version=manifest.version,
            role=manifest.role,
            manifest_path=entry.manifest_path,
            embodiment_tags=manifest.embodiment_tags,
            compatible=False,
            reason=str(exc),
            failure_kind=_classify(str(exc)),
        )
    return RSkillCompatRow(
        repo_id=entry.repo_id,
        version=manifest.version,
        role=manifest.role,
        manifest_path=entry.manifest_path,
        embodiment_tags=manifest.embodiment_tags,
        compatible=True,
        reason=None,
        failure_kind=None,
    )


_CLASSIFICATION_KEYWORDS: tuple[tuple[str, FailureKind], ...] = (
    ("embodiment tag", "embodiment_tag"),
    ("runtime", "runtime"),
    ("quantization", "quantization"),
    ("feature key", "sensor_key"),
    ("modality", "sensor_modality"),
    ("resolution", "resolution"),
    ("min_width", "resolution"),
    ("min_height", "resolution"),
)


def _classify(message: str) -> FailureKind:
    """Coarse classification of a ``ROSCapabilityMismatch`` message.

    Matches keywords from :meth:`rSkill.check_capabilities` /
    :meth:`rSkill.check_sensors` so the table renderer can color rows by
    failure family.  Returns the most specific match; falls back to
    ``"capability_flag"``.
    """
    lower = message.lower()
    for keyword, kind in _CLASSIFICATION_KEYWORDS:
        if keyword in lower:
            return kind
    return "capability_flag"


def _section_pass(
    label: SectionLabel,
    reason: str,
    *,
    informational: bool = False,
) -> SectionVerdict:
    return SectionVerdict(
        label=label,
        compatible=True,
        reason=reason,
        failure_kind=None,
        informational=informational,
    )


def _section_fail(
    label: SectionLabel,
    reason: str,
    failure_kind: FailureKind,
) -> SectionVerdict:
    return SectionVerdict(
        label=label,
        compatible=False,
        reason=reason,
        failure_kind=failure_kind,
        informational=False,
    )


def _evaluate_sections(
    manifest: RSkillManifest,
    robot: RobotDescription,
) -> list[SectionVerdict]:
    """Run each rSkill ↔ robot check independently and collect verdicts.

    Calls the **production** per-section static methods on
    :class:`rSkill` (no parallel logic). Six sections in fixed order:
    embodiment, capability_flags, gpu_runtime, gpu_dtype, sensors, and
    an informational actuators line.
    """
    sections: list[SectionVerdict] = []
    caps = robot.capabilities

    try:
        rSkill.check_embodiment_tags(manifest, caps)
        reason = (
            f"{sorted(set(manifest.embodiment_tags) & set(caps.embodiment_tags))} ∈ skill tags"
            if manifest.embodiment_tags
            else "skill declares no embodiment tags"
        )
        sections.append(_section_pass("embodiment", reason))
    except ROSCapabilityMismatch as exc:
        sections.append(_section_fail("embodiment", str(exc), "embodiment_tag"))

    try:
        rSkill.check_capability_flags(manifest, caps)
        n = len(manifest.capabilities_required)
        reason = f"{n} flag{'s' if n != 1 else ''} satisfied" if n else "no flags required"
        sections.append(_section_pass("capability_flags", reason))
    except ROSCapabilityMismatch as exc:
        sections.append(_section_fail("capability_flags", str(exc), "capability_flag"))

    try:
        rSkill.check_runtime(manifest, caps)
        if not caps.gpu_supported_runtimes:
            reason = f"host runtimes unknown — accepted {manifest.runtime.value}"
        else:
            reason = f"{manifest.runtime.value} ∈ {[r.value for r in caps.gpu_supported_runtimes]}"
        sections.append(_section_pass("gpu_runtime", reason))
    except ROSCapabilityMismatch as exc:
        sections.append(_section_fail("gpu_runtime", str(exc), "runtime"))

    try:
        rSkill.check_quantization_dtype(manifest, caps)
        if not caps.gpu_supported_dtypes:
            reason = f"host dtypes unknown — accepted {manifest.quantization.dtype.value}"
        else:
            reason = (
                f"{manifest.quantization.dtype.value} ∈ "
                f"{[d.value for d in caps.gpu_supported_dtypes]}"
            )
        sections.append(_section_pass("gpu_dtype", reason))
    except ROSCapabilityMismatch as exc:
        sections.append(_section_fail("gpu_dtype", str(exc), "quantization"))

    try:
        rSkill.check_sensors(manifest, robot.sensors)
        n_req = len(manifest.sensors_required)
        reason = (
            f"{n_req}/{n_req} requirement{'s' if n_req != 1 else ''} matched"
            if n_req
            else "no sensor requirements"
        )
        sections.append(_section_pass("sensors", reason))
    except ROSCapabilityMismatch as exc:
        sections.append(_section_fail("sensors", str(exc), _classify(str(exc))))

    # Informational only — the manifest does not declare per-joint
    # requirements today (see plan §4 / CLAUDE.md §6.1 for the ADR
    # gating any new joint-matching schema).
    joint_count = len(robot.joints)
    actuator_reason = (
        f"{joint_count} joint{'s' if joint_count != 1 else ''} on robot"
        if joint_count
        else "robot declares no joints"
    )
    sections.append(_section_pass("actuators", actuator_reason, informational=True))

    return sections


def check_single_rskill(rskill_id: str, robot: RobotDescription) -> CompatibilityReport:
    """Resolve ``rskill_id`` and report its compatibility with ``robot``.

    Resolution follows :func:`openral_rskill.loader.load_rskill_manifest`:
    bare in-tree name, ``<org>/rskill-<name>`` HF Hub alias, or arbitrary
    HF Hub repo id.  No scheme stripping is needed — ``weights_uri`` stores
    bare references directly.

    Returns a :class:`CompatibilityReport` with exactly one
    :class:`RSkillCompatRow`.  The row's ``sections`` list holds the
    per-section verdicts (six entries; see :class:`SectionVerdict`).
    Aggregate ``compatible`` is true when **every non-informational**
    section passes; the first failing section becomes the row-level
    ``reason`` / ``failure_kind``.

    Args:
        rskill_id: The rSkill identifier as printed by ``openral rskill list``.
        robot: The host's assembled :class:`RobotDescription` (typically
            from :func:`openral detect`).

    Returns:
        A one-row :class:`CompatibilityReport` ready for table or JSON
        rendering.

    Example:
        >>> # from openral_detect import check_single_rskill
        >>> # report = check_single_rskill("smolvla-libero", robot)
        >>> # report.rows[0].compatible
    """
    from openral_rskill.loader import load_rskill_manifest  # noqa: PLC0415

    generated_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

    try:
        manifest = load_rskill_manifest(rskill_id)
    except (ROSConfigError, ValidationError, FileNotFoundError, OSError) as exc:
        row = RSkillCompatRow(
            repo_id=rskill_id,
            version="",
            role="s1",
            manifest_path="",
            embodiment_tags=[],
            compatible=False,
            reason=f"manifest load failed: {exc!r}",
            failure_kind="manifest_load",
            sections=[],
        )
        return CompatibilityReport(
            generated_at=generated_at,
            robot_name=robot.name,
            robot_embodiment_tags=list(robot.capabilities.embodiment_tags),
            rows=[row],
        )

    sections = _evaluate_sections(manifest, robot)
    blocking = [s for s in sections if not s.informational and not s.compatible]
    first_fail = blocking[0] if blocking else None

    row = RSkillCompatRow(
        repo_id=manifest.name,
        version=manifest.version,
        role=manifest.role,
        manifest_path="",
        embodiment_tags=list(manifest.embodiment_tags),
        compatible=not blocking,
        reason=first_fail.reason if first_fail else None,
        failure_kind=first_fail.failure_kind if first_fail else None,
        sections=sections,
    )
    return CompatibilityReport(
        generated_at=generated_at,
        robot_name=robot.name,
        robot_embodiment_tags=list(robot.capabilities.embodiment_tags),
        rows=[row],
    )
