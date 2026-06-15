"""rSkill loader — HF Hub download, manifest validation, license guard, local registry.

This module provides :class:`rSkill`: the packaged, capability-tagged
distribution format for a robot skill (CLAUDE.md §6.4 / RFC §1.4, §8.7).

.. warning::
   Cryptographic signature verification of skills (sigstore) is **not yet
   implemented** — see ADR-0006.  ``from_pretrained`` / ``from_yaml`` validate
   the manifest schema and license posture but do **not** verify provenance of
   the downloaded weights.  Set ``OPENRAL_REQUIRE_SIGNED_SKILLS=1`` to fail
   closed (refuse to load) until verification lands.

Separation of concerns
-----------------------
- ``rSkill``      — on-disk / Hub-side *package* (manifest + weights + engines).
- ``Skill``       — in-process lifecycle *node* (ABC in ``openral_rskill.base``).
- ``RSkillInfo``   — runtime state snapshot of a live ``Skill``.
- ``RSkillManifest`` — Pydantic schema (in ``openral_core.schemas``).

Usage
-----
::

    # Install from HF Hub (downloads manifest + weights, registers locally):
    skill_pkg = rSkill.from_pretrained("openral/rskill-pick-cube-so100")

    # Load from a local rskill.yaml without network access:
    skill_pkg = rSkill.from_yaml("/path/to/rskill.yaml")

    # List all installed rSkills:
    entries = rSkill.list_installed()

    # Remove from registry (does NOT delete cached weights):
    rSkill.uninstall("openral/rskill-pick-cube-so100")
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError, ROSError
from openral_core.schemas import (
    RobotCapabilities,
    RobotDescription,
    RSkillEvalResult,
    RSkillLicensePosture,
    RSkillManifest,
    SensorRequirement,
    SensorSpec,
)
from pydantic import BaseModel, Field, ValidationError

log = structlog.get_logger(__name__)

# ── Registry / cache paths ─────────────────────────────────────────────────────

_DATA_HOME = Path(os.environ.get("OPENRAL_DATA_HOME", Path.home() / ".local" / "share" / "openral"))
_CACHE_HOME = Path(
    os.environ.get("OPENRAL_CACHE_HOME", Path.home() / ".cache" / "openral" / "rskills")
)

DEFAULT_REGISTRY_PATH: Path = _DATA_HOME / "rskills.json"
"""Default JSON registry file written by :meth:`rSkill.from_pretrained`."""

# In-process LRU for resolved rSkill manifests (avoids re-fetching the same
# rskill.yaml from the HF Hub every time a config references it).
_RSKILL_MANIFEST_CACHE: dict[str, RSkillManifest] = {}

# ── Environment variable that gates non-commercial weights ────────────────────

_ALLOW_NONCOMMERCIAL_ENV = "OPENRAL_ALLOW_NONCOMMERCIAL"
# Fail-closed switch for operators who require verified provenance.  Skill
# signature verification (sigstore) is not yet implemented (ADR-0006); when this
# is set to "1", the loader refuses to load any skill rather than silently
# trusting unverified weights.  CLAUDE.md §1.1 (safety beats helpfulness).
_REQUIRE_SIGNED_ENV = "OPENRAL_REQUIRE_SIGNED_SKILLS"
"""Set to ``"1"`` to acknowledge non-commercial research use of restricted weights."""

# Perception rSkills (``kind: "detector"`` / ``"vlm"``, ADR-0037/0047) are
# embodiment-agnostic: they consume camera frames and emit detections / scene
# text with no action contract, so a robot's embodiment is not a meaningful
# match axis. The rSkill↔robot embodiment gate exempts these kinds entirely —
# they run on any robot/scene and need not enumerate embodiment tags.
_EMBODIMENT_AGNOSTIC_KINDS: frozenset[str] = frozenset({"detector", "vlm", "reward"})


# ── Registry entry schema ──────────────────────────────────────────────────────


class InstalledRSkillEntry(BaseModel):
    """One row in the local rSkill registry (``~/.local/share/openral/rskills.json``).

    Attributes:
        repo_id: HF Hub repository identifier, e.g. ``"openral/rskill-pick-cube-so100"``.
        version: SemVer version string from the manifest.
        revision: HF Hub commit SHA (None when installed from a local path).
        local_dir: Absolute path to the snapshot directory in the HF Hub cache.
        manifest_path: Absolute path to the ``rskill.yaml`` file on disk.
        license: License posture value string from :class:`RSkillLicensePosture`.
        role: Skill slot — ``"s0"``, ``"s1"``, or ``"s2"``.
        embodiment_tags: Embodiment tags declared in the manifest.
        installed_at: ISO 8601 timestamp of installation.

    Example:
        >>> entry = InstalledRSkillEntry(
        ...     repo_id="example/skill",
        ...     version="0.1.0",
        ...     revision=None,
        ...     local_dir="/tmp/skills/example",
        ...     manifest_path="/tmp/skills/example/rskill.yaml",
        ...     license="apache-2.0",
        ...     role="s1",
        ...     embodiment_tags=["so100_follower"],
        ...     installed_at="2026-01-01T00:00:00+00:00",
        ... )
        >>> entry.repo_id
        'example/skill'
    """

    repo_id: str
    version: str = "0.1.0"
    revision: str | None = None
    local_dir: str
    manifest_path: str
    license: str = RSkillLicensePosture.UNKNOWN.value
    role: str = "s1"
    embodiment_tags: list[str] = Field(default_factory=list)
    installed_at: str


# ── rSkill ─────────────────────────────────────────────────────────────────────


class rSkill:  # noqa: N801  # reason: rSkill is the official package-format name (CLAUDE.md §6.4)
    """Packaged, capability-tagged robot skill from the HF Hub.

    .. warning::
       Signature verification (sigstore) is not yet implemented (ADR-0006); the
       loader validates the manifest and license but does not verify the
       provenance of downloaded weights.  See module docstring.

    An ``rSkill`` represents the *distribution artefact*: the ``rskill.yaml``
    manifest plus the associated weight files stored in the HF Hub cache (or
    locally).  It is distinct from the runtime :class:`~openral_rskill.Skill`
    ABC, which is the in-process lifecycle node.

    Typical workflow::

        pkg = rSkill.from_pretrained("openral/rskill-pick-cube-so100")
        # pkg.manifest  → RSkillManifest (license, embodiment_tags, latency_budget …)
        # pkg.local_dir → Path to weights on disk

    Attributes:
        manifest: Parsed and validated :class:`~openral_core.schemas.RSkillManifest`.
        local_dir: Filesystem path to the directory containing all skill files.
    """

    def __init__(self, manifest: RSkillManifest, local_dir: Path) -> None:
        """Store the manifest and local directory path.

        Args:
            manifest: Parsed and validated rSkill manifest.
            local_dir: Filesystem path to the directory containing all skill files.
        """
        self.manifest = manifest
        self.local_dir = local_dir

    # ── Constructors ───────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        cache_dir: Path | None = None,
        force_download: bool = False,
        commercial_use: bool = True,
        registry_path: Path | None = None,
    ) -> rSkill:
        """Download an rSkill from the HF Hub, validate it, and register it locally.

        Steps performed in order:

        1. Download ``rskill.yaml`` via :func:`huggingface_hub.hf_hub_download`.
        2. Parse and validate against :class:`~openral_core.schemas.RSkillManifest`.
        3. Run the license guard (hard-block for NVIDIA non-commercial without env).
        4. Run the provenance guard (warn that signatures are unverified; fail
           closed if ``OPENRAL_REQUIRE_SIGNED_SKILLS=1``).
        5. Download the full snapshot via :func:`huggingface_hub.snapshot_download`.
        6. Register the entry in the local JSON registry.

        .. warning::
           No cryptographic signature verification is performed (ADR-0006).  The
           weights are trusted on the basis of HF Hub transport security only.
           Pin ``revision`` to a commit SHA for reproducibility, and treat any
           ``*.pt`` weights as untrusted code (see :class:`PyTorchRuntime`).

        Args:
            repo_id: HF Hub repository, e.g. ``"openral/rskill-pick-cube-so100"``.
            revision: Git commit SHA or branch to pin.  ``None`` uses the latest
                commit on the default branch (not reproducible — pin in production).
            cache_dir: Override the default HF Hub cache directory.
            force_download: Re-download even if cached files exist.
            commercial_use: Set to ``False`` if the calling deployment is
                non-commercial research.  Relaxes the NVIDIA non-commercial guard.
            registry_path: Override the default registry JSON path.

        Returns:
            A validated :class:`rSkill` instance ready for use.

        Raises:
            ROSConfigError: If the manifest is invalid, the license blocks the
                deployment, or ``huggingface_hub`` is not installed.
            ImportError: Propagated if ``huggingface_hub`` is not importable (caught
                and re-raised as :class:`ROSConfigError`).

        Example:
            >>> # rSkill.from_pretrained("openral/rskill-pick-cube-so100")
            >>> # rSkill.from_pretrained("openral/rskill-pick-cube-so100", revision="abc123")
        """
        try:
            from huggingface_hub import hf_hub_download, snapshot_download  # noqa: PLC0415
        except ImportError as exc:
            raise ROSConfigError(
                "rSkill.from_pretrained requires 'huggingface_hub'. "
                "Install it: uv add huggingface_hub --package openral-rskill"
            ) from exc

        _cache = str(cache_dir or _CACHE_HOME)

        # ── 1. Fetch manifest ──────────────────────────────────────────────────
        log.info("rskill.fetch_manifest", repo_id=repo_id, revision=revision)
        try:
            manifest_path_str = hf_hub_download(
                repo_id=repo_id,
                filename="rskill.yaml",
                revision=revision,
                cache_dir=_cache,
                force_download=force_download,
            )
        except Exception as exc:
            raise ROSConfigError(f"Failed to download rskill.yaml for '{repo_id}': {exc}") from exc

        # ── 2. Parse manifest ──────────────────────────────────────────────────
        manifest = RSkillManifest.from_yaml(manifest_path_str)

        # ── 3. License guard ───────────────────────────────────────────────────
        cls._check_license(manifest, commercial_use=commercial_use)

        # ── 4. Provenance guard (signatures not yet verified — ADR-0006) ────────
        cls._check_provenance(manifest, source=repo_id)

        # ── 5. Download weights ────────────────────────────────────────────────
        log.info("rskill.download_snapshot", repo_id=repo_id, revision=revision)
        try:
            local_dir_str = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=_cache,
                force_download=force_download,
            )
        except Exception as exc:
            raise ROSConfigError(f"Failed to download snapshot for '{repo_id}': {exc}") from exc

        # ── 6. Register ────────────────────────────────────────────────────────
        entry = InstalledRSkillEntry(
            repo_id=repo_id,
            version=manifest.version,
            revision=revision,
            local_dir=local_dir_str,
            manifest_path=manifest_path_str,
            license=manifest.license.value,
            role=manifest.role,
            embodiment_tags=manifest.embodiment_tags,
            installed_at=datetime.now(timezone.utc).isoformat(),
        )
        cls._register(entry, registry_path or DEFAULT_REGISTRY_PATH)

        log.info(
            "rskill.installed",
            repo_id=repo_id,
            version=manifest.version,
            license=manifest.license.value,
        )
        return cls(manifest=manifest, local_dir=Path(local_dir_str))

    @classmethod
    def from_yaml(cls, path: str | Path, *, local_dir: Path | None = None) -> rSkill:
        """Load an rSkill from a local ``rskill.yaml`` without network access.

        Does NOT register in the local registry.  Useful for offline development
        and testing.

        Args:
            path: Filesystem path to ``rskill.yaml``.
            local_dir: Directory containing the skill files (defaults to the
                directory that contains ``rskill.yaml``).

        Returns:
            A validated :class:`rSkill` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            pydantic.ValidationError: If the YAML fails schema validation.
            ROSConfigError: If the license blocks the deployment, or
                ``OPENRAL_REQUIRE_SIGNED_SKILLS=1`` and verification is
                unavailable (see :meth:`_check_provenance`).

        Example:
            >>> # rSkill.from_yaml("/path/to/my-skill/rskill.yaml")
        """
        manifest = RSkillManifest.from_yaml(str(path))
        resolved = Path(path).resolve()
        cls._check_license(manifest, commercial_use=True)
        cls._check_provenance(manifest, source=str(resolved))
        skill_dir = local_dir or resolved.parent
        cls._validate_eval_jsons(skill_dir)
        return cls(manifest=manifest, local_dir=skill_dir)

    # ── Registry helpers ───────────────────────────────────────────────────────

    @staticmethod
    def list_installed(registry_path: Path | None = None) -> list[InstalledRSkillEntry]:
        """Return all entries in the local skill registry.

        Args:
            registry_path: Override the default registry JSON path.

        Returns:
            List of :class:`InstalledRSkillEntry`, newest-first.

        Raises:
            ROSConfigError: If the registry file is corrupt (invalid JSON or schema).

        Example:
            >>> import tempfile, pathlib
            >>> empty = pathlib.Path(tempfile.mkdtemp()) / "registry.json"
            >>> result = rSkill.list_installed(registry_path=empty)
            >>> isinstance(result, list) and len(result) == 0
            True
        """
        reg = registry_path or DEFAULT_REGISTRY_PATH
        if not reg.exists():
            return []
        try:
            raw: list[dict[str, Any]] = json.loads(reg.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ROSConfigError(f"Corrupt skill registry at {reg}: {exc}") from exc
        entries = [InstalledRSkillEntry.model_validate(r) for r in raw]
        # newest-first
        return sorted(entries, key=lambda e: e.installed_at, reverse=True)

    @staticmethod
    def uninstall(repo_id: str, registry_path: Path | None = None) -> bool:
        """Remove a skill from the local registry (does NOT delete cached weights).

        Args:
            repo_id: The HF Hub repository identifier to remove.
            registry_path: Override the default registry JSON path.

        Returns:
            ``True`` if the entry was found and removed, ``False`` if not found.

        Example:
            >>> rSkill.uninstall("openral/rskill-pick-cube-so100")
            False
        """
        reg = registry_path or DEFAULT_REGISTRY_PATH
        if not reg.exists():
            return False
        try:
            raw: list[dict[str, Any]] = json.loads(reg.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        before = len(raw)
        raw = [r for r in raw if r.get("repo_id") != repo_id]
        if len(raw) == before:
            return False
        reg.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        log.info("rskill.uninstalled", repo_id=repo_id)
        return True

    @staticmethod
    def check_embodiment_tags(
        manifest: RSkillManifest,
        robot_capabilities: RobotCapabilities,
    ) -> None:
        """Verify the manifest's embodiment tags intersect the robot's.

        Exempt for perception rSkills (``kind`` in
        :data:`_EMBODIMENT_AGNOSTIC_KINDS` — ``detector`` / ``vlm``): they are
        camera-in → detections/text-out producers with no action contract, so
        embodiment is not a meaningful axis; they run on any robot regardless of
        tags. Also skipped when a non-perception manifest declares no embodiment
        tags. Used by :meth:`check_capabilities` and by the per-section presenter
        in :func:`openral_detect.check_single_rskill`.

        Raises:
            ROSCapabilityMismatch: If the tag sets are disjoint.
        """
        if manifest.kind in _EMBODIMENT_AGNOSTIC_KINDS:
            return
        if not manifest.embodiment_tags:
            return
        robot_tags = set(robot_capabilities.embodiment_tags)
        skill_tags = set(manifest.embodiment_tags)
        if not skill_tags.intersection(robot_tags):
            raise ROSCapabilityMismatch(
                f"rSkill '{manifest.name}' requires embodiment tag(s) "
                f"{sorted(skill_tags)}, but robot only has {sorted(robot_tags)}."
            )

    @staticmethod
    def check_capability_flags(
        manifest: RSkillManifest,
        robot_capabilities: RobotCapabilities,
    ) -> None:
        """Verify every ``manifest.capabilities_required`` flag.

        Raises:
            ROSCapabilityMismatch: On the first unsatisfied flag or
                unknown field name.
        """
        for flag, required_value in manifest.capabilities_required.items():
            robot_value = getattr(robot_capabilities, flag, None)
            if robot_value is None:
                raise ROSCapabilityMismatch(
                    f"rSkill '{manifest.name}' requires capability flag '{flag}', "
                    "which is not a known field of RobotCapabilities."
                )
            if isinstance(required_value, bool):
                if not robot_value:
                    raise ROSCapabilityMismatch(
                        f"rSkill '{manifest.name}' requires '{flag}=True', "
                        f"but robot reports '{flag}={robot_value}'."
                    )
            elif isinstance(required_value, (int, float)) and float(robot_value) < float(
                required_value
            ):
                raise ROSCapabilityMismatch(
                    f"rSkill '{manifest.name}' requires '{flag}>={required_value}', "
                    f"but robot reports '{flag}={robot_value}'."
                )

    @staticmethod
    def check_runtime(
        manifest: RSkillManifest,
        robot_capabilities: RobotCapabilities,
    ) -> None:
        """Verify the manifest's runtime is in the robot's supported set.

        Skipped when ``robot_capabilities.gpu_supported_runtimes`` is empty
        (unknown — treat as "host has not been probed").

        Raises:
            ROSCapabilityMismatch: If the runtime is not supported.
        """
        if (
            robot_capabilities.gpu_supported_runtimes
            and manifest.runtime not in robot_capabilities.gpu_supported_runtimes
        ):
            raise ROSCapabilityMismatch(
                f"rSkill '{manifest.name}' requires runtime "
                f"'{manifest.runtime.value}', but robot only supports "
                f"{[r.value for r in robot_capabilities.gpu_supported_runtimes]}."
            )

    @staticmethod
    def check_quantization_dtype(
        manifest: RSkillManifest,
        robot_capabilities: RobotCapabilities,
    ) -> None:
        """Verify the manifest's quantization dtype is in the robot's set.

        Skipped when ``robot_capabilities.gpu_supported_dtypes`` is empty.

        Raises:
            ROSCapabilityMismatch: If the dtype is not supported.
        """
        if (
            robot_capabilities.gpu_supported_dtypes
            and manifest.quantization.dtype not in robot_capabilities.gpu_supported_dtypes
        ):
            raise ROSCapabilityMismatch(
                f"rSkill '{manifest.name}' requires quantization dtype "
                f"'{manifest.quantization.dtype.value}', but robot only supports "
                f"{[d.value for d in robot_capabilities.gpu_supported_dtypes]}."
            )

    @staticmethod
    def check_capabilities(
        manifest: RSkillManifest,
        robot_capabilities: RobotCapabilities,
    ) -> None:
        """Verify that the robot satisfies the rSkill's capability requirements.

        Composition of :meth:`check_embodiment_tags`,
        :meth:`check_capability_flags`, :meth:`check_runtime`, and
        :meth:`check_quantization_dtype` — raises on the first mismatch.
        Callers that want a per-section verdict (e.g. ``openral rskill check
        <rskill_id>``) should call the four narrower methods directly.

        Args:
            manifest: The rSkill manifest to check.
            robot_capabilities: The target robot's declared capabilities.

        Raises:
            ROSCapabilityMismatch: If any required capability is not satisfied
                or if embodiment tags do not intersect.

        Example:
            >>> from openral_core.schemas import RobotCapabilities
            >>> caps = RobotCapabilities(embodiment_tags=["so100_follower"])
            >>> # rSkill.check_capabilities(manifest, caps)  # raises if mismatch
        """
        rSkill.check_embodiment_tags(manifest, robot_capabilities)
        rSkill.check_capability_flags(manifest, robot_capabilities)
        rSkill.check_runtime(manifest, robot_capabilities)
        rSkill.check_quantization_dtype(manifest, robot_capabilities)

    @staticmethod
    def check_sensors(
        manifest: RSkillManifest,
        robot_sensors: list[SensorSpec],
    ) -> None:
        """Verify the robot exposes every sensor the rSkill requires.

        Resolves each :class:`~openral_core.SensorRequirement` against
        ``robot_sensors`` per the rules documented on
        :class:`~openral_core.SensorRequirement`:

        * If a requirement carries a ``vla_feature_key``, exactly one robot
          sensor must expose that key, with matching modality, and meeting
          any specified ``min_width`` / ``min_height`` minimum.
        * Otherwise, the robot must expose at least ``count`` sensors of
          the requested modality, each meeting any resolution minimum.

        Args:
            manifest: The rSkill manifest whose ``sensors_required`` to check.
            robot_sensors: The robot's declared :class:`SensorSpec` list
                (typically ``RobotDescription.sensors``).

        Raises:
            ROSCapabilityMismatch: If any requirement is not satisfied.

        Example:
            >>> from openral_core import SensorSpec, SensorModality
            >>> robot = [
            ...     SensorSpec(
            ...         name="cam", modality=SensorModality.RGB, frame_id="world", rate_hz=20.0
            ...     )
            ... ]
            >>> # rSkill.check_sensors(manifest, robot)  # raises if mismatch
        """
        for req in manifest.sensors_required:
            cls = rSkill  # for the raised-error name; statically rSkill here
            if req.vla_feature_key is not None:
                cls._match_keyed_sensor(manifest.name, req, robot_sensors)
            else:
                cls._match_modality_count(manifest.name, req, robot_sensors)

    @staticmethod
    def check_compatibility(
        manifest: RSkillManifest,
        robot: RobotDescription,
    ) -> None:
        """Run every rSkill ↔ robot compatibility check in one call.

        This is the umbrella entry point — combines
        :meth:`check_capabilities` (embodiment tags + boolean / numeric
        capability flags) and :meth:`check_sensors` (sensor requirements
        against ``robot.sensors``). Use this from runtime code that has the
        full :class:`RobotDescription`; the two narrower methods remain for
        cases where only one half is available.

        Args:
            manifest: The rSkill manifest to check.
            robot: The target robot's full description.

        Raises:
            ROSCapabilityMismatch: If embodiment tags mismatch, a capability
                flag is unsatisfied, or any sensor requirement fails.

        Example:
            >>> # rSkill.check_compatibility(manifest, robot)
        """
        rSkill.check_capabilities(manifest, robot.capabilities)
        rSkill.check_sensors(manifest, robot.sensors)

    # ── Sensor-match helpers ──────────────────────────────────────────────────

    @staticmethod
    def _match_keyed_sensor(
        skill_name: str,
        req: SensorRequirement,
        robot_sensors: list[SensorSpec],
    ) -> None:
        """Check a requirement that pins a specific ``vla_feature_key``."""
        matches = [s for s in robot_sensors if s.vla_feature_key == req.vla_feature_key]
        if not matches:
            available = sorted(s.vla_feature_key for s in robot_sensors if s.vla_feature_key)
            raise ROSCapabilityMismatch(
                f"rSkill '{skill_name}' requires sensor with vla_feature_key="
                f"{req.vla_feature_key!r}; robot exposes {available or '<none>'}."
            )
        sensor = matches[0]
        # SensorSpec uses ``use_enum_values=True`` so .modality is a plain string.
        sensor_modality = (
            sensor.modality.value if hasattr(sensor.modality, "value") else sensor.modality
        )
        req_modality = req.modality.value if hasattr(req.modality, "value") else req.modality
        if sensor_modality != req_modality:
            raise ROSCapabilityMismatch(
                f"rSkill '{skill_name}' sensor {req.vla_feature_key!r} requires "
                f"modality={req_modality!r} but robot sensor {sensor.name!r} is "
                f"{sensor_modality!r}."
            )
        rSkill._check_resolution(skill_name, req, sensor)

    @staticmethod
    def _match_modality_count(
        skill_name: str,
        req: SensorRequirement,
        robot_sensors: list[SensorSpec],
    ) -> None:
        """Check a requirement of the form 'N sensors of modality X'."""
        req_modality = req.modality.value if hasattr(req.modality, "value") else req.modality
        matches: list[SensorSpec] = []
        for s in robot_sensors:
            sensor_modality = s.modality.value if hasattr(s.modality, "value") else s.modality
            if sensor_modality != req_modality:
                continue
            try:
                rSkill._check_resolution(skill_name, req, s)
            except ROSCapabilityMismatch:
                continue
            matches.append(s)
        if len(matches) < req.count:
            raise ROSCapabilityMismatch(
                f"rSkill '{skill_name}' requires {req.count} sensor(s) of "
                f"modality={req_modality!r}"
                + (
                    f" with min resolution {req.min_width}x{req.min_height}"
                    if req.min_width or req.min_height
                    else ""
                )
                + f"; robot provides {len(matches)} matching sensor(s)."
            )

    @staticmethod
    def _check_resolution(
        skill_name: str,
        req: SensorRequirement,
        sensor: SensorSpec,
    ) -> None:
        """Raise if ``sensor`` does not meet ``req``'s width / height minima."""
        if req.min_width is None and req.min_height is None:
            return
        intr = sensor.intrinsics
        if intr is None:
            raise ROSCapabilityMismatch(
                f"rSkill '{skill_name}' requires resolution >= "
                f"{req.min_width or 0}x{req.min_height or 0} but sensor "
                f"{sensor.name!r} has no intrinsics declared."
            )
        if req.min_width is not None and intr.width < req.min_width:
            raise ROSCapabilityMismatch(
                f"rSkill '{skill_name}' requires width >= {req.min_width} but "
                f"sensor {sensor.name!r} has width={intr.width}."
            )
        if req.min_height is not None and intr.height < req.min_height:
            raise ROSCapabilityMismatch(
                f"rSkill '{skill_name}' requires height >= {req.min_height} but "
                f"sensor {sensor.name!r} has height={intr.height}."
            )

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _check_license(manifest: RSkillManifest, *, commercial_use: bool) -> None:
        """Enforce license posture guards (CLAUDE.md §7.4, §12).

        Hard-blocks **any** non-commercial weight posture — NVIDIA OneWay
        Noncommercial (GR00T N1-N1.6), RLWRLD Model License (RLDX-1), and
        permissive-research (π0.5) — in a commercial deployment unless the
        ``OPENRAL_ALLOW_NONCOMMERCIAL`` environment variable is set to ``"1"``.
        The block is driven by :attr:`RSkillManifest.is_commercial_use_allowed`
        so a new restricted posture is gated by construction rather than needing
        a hand-maintained branch here (the previous version only caught
        NVIDIA_NON_COMMERCIAL and silently let RLWRLD / research weights through).

        PROPRIETARY and UNKNOWN licenses are surfaced as structured warnings
        rather than hard-blocked: they need out-of-band vendor review or an
        undeclared-license decision that the research env var cannot stand in
        for — so it never unlocks them.

        Args:
            manifest: The rSkill manifest whose license to check.
            commercial_use: Whether the calling deployment is commercial.

        Raises:
            ROSConfigError: If the license blocks commercial deployment.
        """
        lic = manifest.license

        if lic == RSkillLicensePosture.PROPRIETARY:
            log.warning(
                "rskill.proprietary_license",
                repo=manifest.name,
                note="Proprietary weights (e.g. Helix, Gemini Robotics). "
                "Review and comply with vendor terms before any deployment.",
            )
            return

        if lic == RSkillLicensePosture.UNKNOWN:
            log.warning(
                "rskill.unknown_license",
                repo=manifest.name,
                note="License is not declared in rskill.yaml. Verify before deployment.",
            )
            return

        if not manifest.is_commercial_use_allowed:
            allow_env = os.environ.get(_ALLOW_NONCOMMERCIAL_ENV, "0")
            if commercial_use and allow_env != "1":
                raise ROSConfigError(
                    f"rSkill '{manifest.name}' uses non-commercial weights "
                    f"(license posture '{lic.value}'). "
                    "Commercial deployment is blocked per CLAUDE.md §7.4, §12. "
                    f"For research use only, set: export {_ALLOW_NONCOMMERCIAL_ENV}=1"
                )
            log.warning(
                "rskill.non_commercial_override",
                repo=manifest.name,
                license=lic.value,
                env=_ALLOW_NONCOMMERCIAL_ENV,
                note="Non-commercial weights loaded; ensure research-only deployment.",
            )

    @staticmethod
    def _check_provenance(manifest: RSkillManifest, *, source: str) -> None:
        """Surface the absence of signature verification, and optionally fail closed.

        CLAUDE.md §3 describes sigstore-signed skills whose loaders "verify before
        activation", but that control is **not yet implemented** (ADR-0006): there
        is no signature field on the manifest and no verification step here.  This
        guard makes that gap explicit per CLAUDE.md §1.2 (truth over plausibility)
        and §1.4 (explicit beats implicit):

        - Always emits a structured ``rskill.unverified_provenance`` warning so the
          missing control is observable in logs/traces, not silently assumed.
        - When ``OPENRAL_REQUIRE_SIGNED_SKILLS=1`` it raises, letting
          security-conscious deployments fail closed rather than load weights of
          unverified provenance.

        Args:
            manifest: The rSkill manifest being loaded.
            source: HF repo id or local path the skill is loaded from (for logs).

        Raises:
            ROSConfigError: If ``OPENRAL_REQUIRE_SIGNED_SKILLS`` is set to ``"1"``.
        """
        if os.environ.get(_REQUIRE_SIGNED_ENV, "0") == "1":
            raise ROSConfigError(
                f"rSkill '{manifest.name}' cannot be verified: signature verification "
                "is not yet implemented (ADR-0006), and "
                f"{_REQUIRE_SIGNED_ENV}=1 requires verified provenance. "
                f"Refusing to load '{source}'. Unset {_REQUIRE_SIGNED_ENV} to load "
                "unverified skills at your own risk."
            )
        log.warning(
            "rskill.unverified_provenance",
            repo=manifest.name,
            source=source,
            note="Skill signatures are NOT verified (ADR-0006); weights are trusted on "
            "HF Hub transport security only. Pin a revision SHA and treat *.pt weights "
            "as untrusted code.",
        )

    @staticmethod
    def _validate_eval_jsons(skill_dir: Path) -> None:
        """Validate every ``<skill_dir>/eval/*.json`` against :class:`RSkillEvalResult`.

        CLAUDE.md §6.4 lists ``eval/`` as required packaging for an rSkill;
        this guard turns a malformed benchmark JSON into a typed
        :class:`ROSConfigError` at load time instead of a downstream surprise
        when ``openral benchmark report`` walks the same files.

        Args:
            skill_dir: Directory containing the rSkill (parent of ``eval/``).

        Raises:
            ROSConfigError: If any JSON fails :class:`RSkillEvalResult` validation.
        """
        eval_dir = skill_dir / "eval"
        if not eval_dir.is_dir():
            return
        for json_path in sorted(eval_dir.glob("*.json")):
            try:
                RSkillEvalResult.from_json(str(json_path))
            except ValidationError as exc:
                raise ROSConfigError(f"invalid skill eval JSON at {json_path}: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ROSConfigError(f"malformed JSON at {json_path}: {exc}") from exc

    @staticmethod
    def _register(entry: InstalledRSkillEntry, registry_path: Path) -> None:
        """Append or update an entry in the local JSON registry.

        An existing entry with the same ``repo_id`` is replaced in-place.

        Args:
            entry: The skill entry to write.
            registry_path: Absolute path to the registry JSON file.
        """
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        raw: list[dict[str, Any]] = []
        if registry_path.exists():
            try:
                raw = json.loads(registry_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("rskill.registry_corrupt", path=str(registry_path))
                raw = []
        # Replace existing entry with same repo_id.
        raw = [r for r in raw if r.get("repo_id") != entry.repo_id]
        raw.append(entry.model_dump())
        registry_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    # ── Repr ───────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a concise developer representation."""
        return (
            f"rSkill(name={self.manifest.name!r}, "
            f"version={self.manifest.version!r}, "
            f"license={self.manifest.license.value!r})"
        )


# ── URI helpers used by VLA adapters ──────────────────────────────────────────


def resolve_rskill_local_dir(uri: str) -> Path | None:
    """Return the on-disk directory of an in-tree rSkill, or ``None``.

    The argument is a bare rSkill reference (``smolvla-libero``,
    ``rskills/smolvla-libero``, ``OpenRAL/rskill-smolvla-libero``, …).
    Walks the same candidate forms :func:`_candidate_local_paths` produces
    and returns the first one that is a directory containing an
    ``rskill.yaml``. Resolves to an absolute path when found.

    Used by ``openral benchmark run`` to write the ``RSkillEvalResult``
    JSON into ``<skill_dir>/eval/<benchmark_id>.json`` regardless of the
    user's cwd or which URI form they typed. Hub-only references that
    have no in-tree shim resolve to ``None`` and the caller is expected
    to fall back to a cwd-relative default.
    """
    for candidate in _candidate_local_paths(uri):
        if candidate.is_dir() and (candidate / "rskill.yaml").is_file():
            return candidate.resolve()
    return None


def _candidate_local_paths(uri: str) -> list[Path]:
    """Enumerate candidate on-disk locations for a bare rSkill reference.

    The argument may be a path-like string (``rskills/smolvla-libero``,
    ``rskills/smolvla-libero/rskill.yaml``) or an HF Hub repo id
    (``OpenRAL/rskill-smolvla-libero``). For each form we generate the
    file/dir paths to probe, both as-given (cwd-relative) and re-rooted at
    the OpenRAL repo so configs work regardless of the caller's cwd.
    """
    forms: list[Path] = []

    def add(p: Path) -> None:
        forms.append(p)
        forms.append(p / "rskill.yaml")

    candidate = Path(uri)
    add(candidate)

    # Re-anchor at the repo root so ``rskills/<name>`` works from any cwd.
    repo_root = _find_repo_root_from(Path(__file__))
    if repo_root is not None:
        add(repo_root / candidate)
        # Bare-name form: ``smolvla-libero`` → ``<repo>/rskills/smolvla-libero``.
        if "/" not in uri:
            add(repo_root / "rskills" / candidate)
        # HF Hub form ``<org>/rskill-<name>`` → in-tree ``rskills/<name>``.
        if "/" in uri:
            tail = uri.rsplit("/", 1)[1]
            stripped = tail.removeprefix("rskill-").removeprefix("rskill_")
            if stripped:
                add(repo_root / "rskills" / stripped)

    return forms


def discover_intree_rskills() -> list[tuple[str, RSkillManifest]]:
    """Walk the in-tree ``rskills/`` directory and return ``(name, manifest)`` pairs.

    Skipped on stderr-reported errors so a malformed local manifest does
    not break the listing. Sorted alphabetically by directory name. No
    HF Hub network call.

    Returns:
        List of ``(rskill_name, manifest)`` tuples, one per
        ``rskills/<name>/rskill.yaml`` on disk.
    """
    repo_root = _find_repo_root_from(Path(__file__))
    if repo_root is None:
        return []
    rskills_dir = repo_root / "rskills"
    if not rskills_dir.is_dir():
        return []

    out: list[tuple[str, RSkillManifest]] = []
    for child in sorted(rskills_dir.iterdir()):
        if not child.is_dir() or not (child / "rskill.yaml").is_file():
            continue
        try:
            manifest = load_rskill_manifest(child.name)
        except (ROSError, ValueError) as exc:
            print(f"  {child.name}: {exc}", file=sys.stderr)
            continue
        out.append((child.name, manifest))
    return out


def _find_repo_root_from(start: Path) -> Path | None:
    """Locate the OpenRAL repo root by walking up from ``start``.

    Returns the first parent directory that contains both ``pyproject.toml``
    and a ``rskills/`` directory, or ``None`` if no such ancestor exists.
    """
    here = start.resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "pyproject.toml").is_file() and (ancestor / "rskills").is_dir():
            return ancestor
    return None


def _validate_skill_ref(raw: str) -> str:
    """Validate and return a bare rSkill reference unchanged.

    Accepts any non-empty string that does not carry an explicit URI scheme:
    a bare name (``"smolvla-libero"``), a path (``"rskills/smolvla-libero"``),
    or an HF repo id (``"OpenRAL/rskill-smolvla-libero"``). Rejects inputs
    that carry a known scheme (``hf://``, ``local://``, ``file://``, etc.)
    so callers are never silently handed a non-rSkill URI.

    Args:
        raw: The user-supplied reference. Must be non-empty and carry no
            explicit URI scheme.

    Returns:
        The trimmed reference string unchanged.

    Raises:
        ROSConfigError: If ``raw`` is empty / whitespace-only or carries
            a known URI scheme.

    Example:
        >>> _validate_skill_ref("smolvla-libero")
        'smolvla-libero'
        >>> _validate_skill_ref("rskills/smolvla-libero")
        'rskills/smolvla-libero'
    """
    cleaned = raw.strip()
    if not cleaned:
        raise ROSConfigError("rskill reference must be a non-empty string")
    for bad in ("hf://", "local://", "file://", "http://", "https://"):
        if cleaned.startswith(bad):
            raise ROSConfigError(
                f"rskill reference {raw!r} carries a {bad!r} scheme; pass a "
                "bare rSkill name or path (rskills/<name>) instead. "
                "The sim layer rejects raw URI schemes by design — weights "
                "must come from an rSkill manifest."
            )
    return cleaned


def load_rskill_manifest(uri: str) -> RSkillManifest:
    """Resolve a bare rSkill reference to a parsed :class:`RSkillManifest`.

    Accepts a bare name, path, or HF repo id. Resolution order:

    1. **Local file or directory** — ``rskills/smolvla-libero``,
       ``rskills/smolvla-libero/rskill.yaml``, or any absolute path. Tried
       both relative to the current working directory and relative to the
       in-tree OpenRAL repo root.
    2. **HF Hub repo id mapped to local skill** — ``<org>/rskill-<name>`` is
       mapped to ``<repo_root>/rskills/<name>/rskill.yaml`` if that file
       exists.
    3. **HF Hub repo id** — anything else is treated as a Hub repo and the
       manifest is downloaded via :func:`huggingface_hub.hf_hub_download`.

    Results are memoised in-process so repeated lookups do not re-hit disk
    or the Hub.

    Args:
        uri: The reference, e.g. ``"rskills/smolvla-libero"`` or
            ``"OpenRAL/rskill-smolvla-libero"``.

    Returns:
        The parsed and validated :class:`~openral_core.RSkillManifest`.

    Raises:
        ROSConfigError: If the reference cannot be resolved.
    """
    if uri in _RSKILL_MANIFEST_CACHE:
        return _RSKILL_MANIFEST_CACHE[uri]

    yaml_path: Path | None = None
    for candidate in _candidate_local_paths(uri):
        if candidate.is_file() and candidate.suffix in (".yaml", ".yml"):
            yaml_path = candidate
            break
        if candidate.is_dir() and (candidate / "rskill.yaml").is_file():
            yaml_path = candidate / "rskill.yaml"
            break

    if yaml_path is not None:
        manifest = RSkillManifest.from_yaml(str(yaml_path))
        _RSKILL_MANIFEST_CACHE[uri] = manifest
        return manifest

    # Fall through to HF Hub.
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:
        raise ROSConfigError(
            f"rSkill {uri!r} is not a local path and 'huggingface_hub' is not "
            "installed; cannot resolve. Install with: "
            "uv add huggingface_hub --package openral-rskill"
        ) from exc

    try:
        manifest_path = hf_hub_download(
            repo_id=uri,
            filename="rskill.yaml",
            cache_dir=str(_CACHE_HOME),
        )
    except Exception as exc:
        raise ROSConfigError(
            f"failed to resolve rSkill reference {uri!r}: not a local path "
            f"and Hub lookup failed: {exc}"
        ) from exc

    manifest = RSkillManifest.from_yaml(manifest_path)
    _RSKILL_MANIFEST_CACHE[uri] = manifest
    return manifest


def resolve_rskill_to_hf(uri: str) -> str:
    """Resolve a bare rSkill reference to a HF Hub repo id or local path.

    Looks up the manifest via :func:`load_rskill_manifest` and returns
    either the bare HF Hub repo id (when ``weights_uri`` is ``hf://...``)
    or an absolute local filesystem path (when ``weights_uri`` is
    ``local://...``). Both forms are accepted by ``from_pretrained``
    helpers like ``lerobot.policies.pi05.PI05Policy.from_pretrained``,
    so callers can pass the result through unchanged.

    The ``local://`` form is the bridge for one-shot artifacts that
    haven't been (or can't be) pushed to the Hub -- e.g. the locally
    converted ``robocasa/robocasa365_checkpoints/pi05_pretrain_human300``
    orbax->lerobot output under
    ``outputs/run_artifacts/r365_pi05_ckpt_lerobot/``. Paths are
    resolved relative to the current working directory at load time;
    absolute paths in the manifest are honoured verbatim.

    Args:
        uri: A bare rSkill reference — same shapes as :func:`load_rskill_manifest`.

    Returns:
        Bare HF Hub repo id (e.g. ``"lerobot/smolvla_libero"``) or an
        absolute filesystem path string.

    Raises:
        ROSConfigError: If the manifest cannot be resolved, the
            ``weights_uri`` scheme is neither ``hf://`` nor ``local://``,
            or a ``local://`` target doesn't exist on disk.
    """
    target, _ = resolve_rskill_to_hf_with_revision(uri)
    return target


def resolve_rskill_to_hf_with_revision(uri: str) -> tuple[str, str | None]:
    """Resolve a bare rSkill reference to ``(repo_id_or_path, revision)``.

    Same resolution as :func:`resolve_rskill_to_hf`, but splits the optional
    ``@<branch-or-sha>`` revision pin off an ``hf://`` ``weights_uri`` and
    returns it separately. HF ``from_pretrained`` / ``snapshot_download`` treat
    their repo-id argument as a bare id and ignore an appended ``@<sha>``, so the
    pin must be passed via the ``revision`` kwarg or it is silently dropped
    (security audit 2026-06, finding H4). ``local://`` targets have no revision.

    Args:
        uri: A bare rSkill reference (name, path, or HF repo id).

    Returns:
        ``(repo_id, revision)`` for ``hf://`` weights (``revision`` is ``None``
        when unpinned), or ``(absolute_path, None)`` for ``local://`` weights.

    Raises:
        ROSConfigError: If the manifest cannot be resolved, has no
            ``weights_uri``, the scheme is unsupported, or a ``local://`` target
            doesn't exist.
    """
    manifest = load_rskill_manifest(uri)
    weights = manifest.weights_uri
    if weights is None:
        raise ROSConfigError(
            f"rSkill {manifest.name!r} has no weights_uri to resolve; only "
            "weights-bearing skills (vla/policy kinds) can be resolved to a HF "
            "Hub repo id or local path."
        )
    if weights.startswith("hf://"):
        repo_id, _, revision = weights[len("hf://") :].partition("@")
        return repo_id, (revision or None)
    if weights.startswith("local://"):
        raw = weights[len("local://") :]
        path = Path(raw)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise ROSConfigError(
                f"rSkill {manifest.name!r} weights_uri={weights!r} resolves to "
                f"{path!s} but that path does not exist"
            )
        return str(path), None
    raise ROSConfigError(
        f"rSkill {manifest.name!r} weights_uri is {weights!r}; "
        "only hf:// and local:// schemes are supported in a skill manifest's weights_uri"
    )
