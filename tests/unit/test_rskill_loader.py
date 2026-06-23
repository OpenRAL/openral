"""Unit tests for rSkill loader — no network, no GPU required.

All HF Hub I/O is mocked.  The local JSON registry is written to a
pytest tmp_path so tests are fully isolated.

Coverage
--------
- ``rSkill.from_yaml``                     — local manifest load, happy path
- ``rSkill.from_pretrained``               — mocked HF Hub download, happy path
- ``rSkill._check_license``               — NVIDIA non-commercial block + env override
- ``rSkill._check_license``               — Apache-2.0 always passes
- ``rSkill._check_license``               — PROPRIETARY logs warning (no raise)
- ``rSkill._check_license``               — PERMISSIVE_RESEARCH logs info (no raise)
- ``rSkill._check_license``               — UNKNOWN logs warning (no raise)
- ``rSkill.list_installed``               — empty registry → empty list
- ``rSkill.list_installed``               — populated registry → correct entries
- ``rSkill.uninstall``                    — removes matching entry; returns True
- ``rSkill.uninstall``                    — no-op when repo_id absent; returns False
- ``rSkill.check_capabilities``           — tag mismatch → ROSCapabilityMismatch
- ``rSkill.check_capabilities``           — bool flag fail → ROSCapabilityMismatch
- ``rSkill.check_capabilities``           — numeric flag fail → ROSCapabilityMismatch
- ``rSkill.check_capabilities``           — all satisfied → no raise
- ``InstalledRSkillEntry``                 — schema round-trip via JSON
- ``rSkill.__repr__``                     — contains name, version, license
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_core.schemas import (
    ActuatorRequirement,
    ControlMode,
    ControlModeSemantics,
    EmbodimentKind,
    IntrinsicsPinhole,
    JointSpec,
    JointType,
    QuantizationConfig,
    QuantizationDtype,
    RobotCapabilities,
    RobotDescription,
    RSkillAction,
    RSkillLatencyBudget,
    RSkillLicensePosture,
    RSkillManifest,
    RSkillProcessors,
    RSkillRuntime,
    SafetyEnvelope,
    SensorModality,
    SensorRequirement,
    SensorSpec,
)
from openral_rskill.loader import (
    InstalledRSkillEntry,
    _validate_skill_ref,
    resolve_rskill_to_hf_with_revision,
    rSkill,
)

_DEFAULT_ACTUATORS: list[ActuatorRequirement] = [
    ActuatorRequirement(
        kind=ControlMode.JOINT_POSITION,
        control_mode_semantics=ControlModeSemantics(mode="absolute"),
    ),
]


def _default_processors(repo: str = "test/skill") -> RSkillProcessors:
    """Modern lerobot families require an explicit processors block.

    Test fixtures construct manifests against synthetic ``hf://test/...``
    URIs; we just need shape-valid per-file URIs (the loader doesn't
    download in these tests).
    """
    return RSkillProcessors(
        preprocessor_uri=f"hf://{repo}/policy_preprocessor.json",
        postprocessor_uri=f"hf://{repo}/policy_postprocessor.json",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

_APACHE_YAML = textwrap.dedent("""\
    name: test/rskill-alpha
    version: "0.2.0"
    license: apache-2.0
    role: s1
    kind: vla
    model_family: smolvla
    embodiment_tags: [so100_follower]
    runtime: pytorch
    weights_uri: "hf://test/rskill-alpha"
    chunk_size: 16
    latency_budget:
      per_chunk_ms: 500.0
    actuators_required:
      - kind: joint_position
        control_mode_semantics: {mode: absolute}
    processors:
      preprocessor_uri: "hf://test/rskill-alpha/policy_preprocessor.json"
      postprocessor_uri: "hf://test/rskill-alpha/policy_postprocessor.json"
    description: "Loader-test rSkill fixture (apache-2.0)."
    actions:
      - generalist
""")

# NVIDIA non-commercial wraps a Franka-targeted GR00T checkpoint here
# (the original test used unitree_g1, but V1 closed embodiment_tags to
# the in-tree set; franka_panda exercises the same NVIDIA license guard).
_NVIDIA_YAML = textwrap.dedent("""\
    name: test/rskill-groot
    version: "1.0.0"
    license: nvidia_non_commercial
    role: s1
    kind: vla
    model_family: pi05
    embodiment_tags: [franka_panda]
    runtime: tensorrt
    weights_uri: "hf://nvidia/gr00t-n1"
    chunk_size: 32
    latency_budget:
      per_chunk_ms: 200.0
    actuators_required:
      - kind: joint_position
        control_mode_semantics: {mode: absolute}
    processors:
      preprocessor_uri: "hf://nvidia/gr00t-n1/policy_preprocessor.json"
      postprocessor_uri: "hf://nvidia/gr00t-n1/policy_postprocessor.json"
    description: "Loader-test rSkill fixture (NVIDIA non-commercial license guard)."
    actions:
      - generalist
""")

_PROPRIETARY_YAML = textwrap.dedent("""\
    name: test/rskill-helix
    version: "0.1.0"
    license: proprietary
    role: s1
    kind: vla
    model_family: pi05
    embodiment_tags: [franka_panda]
    runtime: pytorch
    weights_uri: "hf://aaeon/helix"
    chunk_size: 32
    latency_budget:
      per_chunk_ms: 300.0
    actuators_required:
      - kind: joint_position
        control_mode_semantics: {mode: absolute}
    processors:
      preprocessor_uri: "hf://aaeon/helix/policy_preprocessor.json"
      postprocessor_uri: "hf://aaeon/helix/policy_postprocessor.json"
    description: "Loader-test rSkill fixture (proprietary license posture)."
    actions:
      - generalist
""")


def _write_yaml(tmp_path: Path, content: str, name: str = "rskill.yaml") -> Path:
    """Write content to a yaml file in tmp_path and return the path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_entry(repo_id: str = "test/rskill-alpha") -> InstalledRSkillEntry:
    return InstalledRSkillEntry(
        repo_id=repo_id,
        version="0.2.0",
        revision=None,
        local_dir="/tmp/skills/test-skill-alpha",
        manifest_path="/tmp/skills/test-skill-alpha/rskill.yaml",
        license="apache-2.0",
        role="s1",
        kind="vla",
        embodiment_tags=["so100_follower"],
        installed_at="2026-01-01T00:00:00+00:00",
    )


# ── from_yaml ─────────────────────────────────────────────────────────────────


class TestFromYaml:
    def test_happy_path_apache(self, tmp_path: Path) -> None:
        """from_yaml loads and validates an Apache-licensed manifest."""
        p = _write_yaml(tmp_path, _APACHE_YAML)
        pkg = rSkill.from_yaml(p)
        assert pkg.manifest.name == "test/rskill-alpha"
        assert pkg.manifest.version == "0.2.0"
        assert pkg.manifest.license is RSkillLicensePosture.APACHE_2_0
        assert pkg.local_dir == tmp_path

    def test_local_dir_defaults_to_yaml_parent(self, tmp_path: Path) -> None:
        """local_dir defaults to the directory that contains rskill.yaml."""
        p = _write_yaml(tmp_path, _APACHE_YAML)
        pkg = rSkill.from_yaml(p)
        assert pkg.local_dir == tmp_path

    def test_local_dir_override(self, tmp_path: Path) -> None:
        """local_dir can be overridden via the local_dir kwarg."""
        p = _write_yaml(tmp_path, _APACHE_YAML)
        override = tmp_path / "weights"
        pkg = rSkill.from_yaml(p, local_dir=override)
        assert pkg.local_dir == override

    def test_nvidia_blocks_commercial_in_from_yaml(self, tmp_path: Path) -> None:
        """from_yaml must still raise for NVIDIA non-commercial in a commercial context."""
        p = _write_yaml(tmp_path, _NVIDIA_YAML)
        with pytest.raises(ROSConfigError, match="non-commercial"):
            rSkill.from_yaml(p)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """from_yaml must raise FileNotFoundError when the path does not exist."""
        with pytest.raises(FileNotFoundError):
            rSkill.from_yaml(tmp_path / "nonexistent.yaml")


# ── provenance guard (signatures unverified — ADR-0006) ───────────────────────


class TestProvenanceGuard:
    """Signature verification is not yet implemented; the loader must warn and
    must fail closed when the operator demands verified provenance."""

    def test_loads_unverified_by_default(self, tmp_path: Path) -> None:
        """Without the env, an unverified skill still loads (back-compat)."""
        p = _write_yaml(tmp_path, _APACHE_YAML)
        pkg = rSkill.from_yaml(p)  # must not raise
        assert pkg.manifest.name == "test/rskill-alpha"

    def test_require_signed_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OPENRAL_REQUIRE_SIGNED_SKILLS=1 refuses to load unverified skills."""
        monkeypatch.setenv("OPENRAL_REQUIRE_SIGNED_SKILLS", "1")
        p = _write_yaml(tmp_path, _APACHE_YAML)
        with pytest.raises(ROSConfigError, match="signature verification is not yet"):
            rSkill.from_yaml(p)

    def test_require_signed_off_value_still_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the exact string '1' fails closed; other values do not."""
        monkeypatch.setenv("OPENRAL_REQUIRE_SIGNED_SKILLS", "0")
        p = _write_yaml(tmp_path, _APACHE_YAML)
        pkg = rSkill.from_yaml(p)  # must not raise
        assert pkg.manifest.name == "test/rskill-alpha"


# ── resolve_rskill_to_hf_with_revision (H4) ───────────────────────────────────


class TestResolveToHfWithRevision:
    """A pinned ``@<sha>`` must be split into the revision so loaders can pass it
    to from_pretrained/snapshot_download instead of gluing it onto the repo id
    where HF drops it (security audit 2026-06, H4)."""

    def test_unpinned_hf_returns_none_revision(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, _APACHE_YAML)
        repo_id, revision = resolve_rskill_to_hf_with_revision(str(p))
        assert repo_id == "test/rskill-alpha"
        assert revision is None

    def test_pinned_hf_splits_revision(self, tmp_path: Path) -> None:
        pinned = _APACHE_YAML.replace(
            'weights_uri: "hf://test/rskill-alpha"',
            'weights_uri: "hf://test/rskill-alpha@d34db33fc0ffee"',
        )
        p = _write_yaml(tmp_path, pinned)
        repo_id, revision = resolve_rskill_to_hf_with_revision(str(p))
        assert repo_id == "test/rskill-alpha"
        assert revision == "d34db33fc0ffee"


# ── from_pretrained (mocked HF Hub) ───────────────────────────────────────────


class TestFromPretrained:
    def _mock_hf(
        self, tmp_path: Path, yaml_content: str = _APACHE_YAML
    ) -> tuple[MagicMock, MagicMock]:
        """Return (mock_hf_download, mock_snapshot) pre-configured for tmp_path."""
        manifest_file = _write_yaml(tmp_path, yaml_content)
        mock_download = MagicMock(return_value=str(manifest_file))
        mock_snapshot = MagicMock(return_value=str(tmp_path))
        return mock_download, mock_snapshot

    def test_happy_path(self, tmp_path: Path) -> None:
        """from_pretrained returns rSkill with correct manifest on success."""
        reg = tmp_path / "rskills.json"
        dl, snap = self._mock_hf(tmp_path)
        with (
            patch("openral_rskill.loader.hf_hub_download", dl, create=True),
            patch("openral_rskill.loader.snapshot_download", snap, create=True),
            patch("huggingface_hub.hf_hub_download", dl),
            patch("huggingface_hub.snapshot_download", snap),
        ):
            pkg = rSkill.from_pretrained(
                "test/rskill-alpha",
                registry_path=reg,
            )
        assert pkg.manifest.name == "test/rskill-alpha"
        assert pkg.local_dir == tmp_path

    def test_registers_in_json(self, tmp_path: Path) -> None:
        """from_pretrained writes an entry to the JSON registry."""
        reg = tmp_path / "reg" / "rskills.json"
        dl, snap = self._mock_hf(tmp_path)
        with (
            patch("huggingface_hub.hf_hub_download", dl),
            patch("huggingface_hub.snapshot_download", snap),
        ):
            rSkill.from_pretrained("test/rskill-alpha", registry_path=reg)
        entries = rSkill.list_installed(registry_path=reg)
        assert len(entries) == 1
        assert entries[0].repo_id == "test/rskill-alpha"

    def test_missing_huggingface_hub_raises_ros_config_error(self, tmp_path: Path) -> None:
        """from_pretrained must raise ROSConfigError when huggingface_hub is absent."""
        import sys

        saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ROSConfigError, match="huggingface_hub"):
                rSkill.from_pretrained("test/rskill-alpha")
        finally:
            if saved is not None:
                sys.modules["huggingface_hub"] = saved
            else:
                del sys.modules["huggingface_hub"]

    def test_download_error_raises_ros_config_error(self, tmp_path: Path) -> None:
        """Network errors during hf_hub_download surface as ROSConfigError."""
        dl = MagicMock(side_effect=RuntimeError("connection refused"))
        with (
            patch("huggingface_hub.hf_hub_download", dl),
            pytest.raises(ROSConfigError, match=r"Failed to download rskill\.yaml"),
        ):
            rSkill.from_pretrained("test/rskill-alpha")

    def test_revision_is_passed_through(self, tmp_path: Path) -> None:
        """from_pretrained forwards revision= to both HF Hub calls."""
        reg = tmp_path / "rskills.json"
        dl, snap = self._mock_hf(tmp_path)
        with (
            patch("huggingface_hub.hf_hub_download", dl),
            patch("huggingface_hub.snapshot_download", snap),
        ):
            rSkill.from_pretrained("test/rskill-alpha", revision="deadbeef", registry_path=reg)
        dl.assert_called_once()
        _, dl_kwargs = dl.call_args
        assert dl_kwargs.get("revision") == "deadbeef"


# ── License guard ──────────────────────────────────────────────────────────────


class TestCheckLicense:
    def _manifest_with_license(self, lic: RSkillLicensePosture) -> RSkillManifest:
        return RSkillManifest(
            name="test/skill",
            version="0.1.0",
            license=lic,
            role="s1",
            kind="vla",
            model_family="smolvla",
            embodiment_tags=["so100_follower"],
            weights_uri="hf://test/skill",
            chunk_size=16,
            latency_budget=RSkillLatencyBudget(per_chunk_ms=100.0),
            actuators_required=list(_DEFAULT_ACTUATORS),
            processors=_default_processors(),
            description="Loader-test rSkill fixture (license posture).",
            actions=[RSkillAction.GENERALIST],
        )

    def test_apache_always_passes(self) -> None:
        """Apache-2.0 must never raise regardless of commercial_use flag."""
        m = self._manifest_with_license(RSkillLicensePosture.APACHE_2_0)
        rSkill._check_license(m, commercial_use=True)
        rSkill._check_license(m, commercial_use=False)

    def test_nvidia_blocks_commercial_use(self) -> None:
        """NVIDIA_NON_COMMERCIAL must raise when commercial_use=True and env not set."""
        m = self._manifest_with_license(RSkillLicensePosture.NVIDIA_NON_COMMERCIAL)
        env_backup = os.environ.pop("OPENRAL_ALLOW_NONCOMMERCIAL", None)
        try:
            with pytest.raises(ROSConfigError, match="non-commercial"):
                rSkill._check_license(m, commercial_use=True)
        finally:
            if env_backup is not None:
                os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = env_backup

    def test_nvidia_passes_when_env_set(self) -> None:
        """NVIDIA_NON_COMMERCIAL must NOT raise when OPENRAL_ALLOW_NONCOMMERCIAL=1."""
        m = self._manifest_with_license(RSkillLicensePosture.NVIDIA_NON_COMMERCIAL)
        os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = "1"
        try:
            rSkill._check_license(m, commercial_use=True)  # must not raise
        finally:
            del os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"]

    def test_nvidia_passes_when_non_commercial_use(self) -> None:
        """NVIDIA_NON_COMMERCIAL must NOT raise when commercial_use=False."""
        m = self._manifest_with_license(RSkillLicensePosture.NVIDIA_NON_COMMERCIAL)
        env_backup = os.environ.pop("OPENRAL_ALLOW_NONCOMMERCIAL", None)
        try:
            rSkill._check_license(m, commercial_use=False)  # no raise
        finally:
            if env_backup is not None:
                os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = env_backup

    def test_proprietary_does_not_raise(self) -> None:
        """PROPRIETARY must warn but not raise."""
        m = self._manifest_with_license(RSkillLicensePosture.PROPRIETARY)
        rSkill._check_license(m, commercial_use=True)  # must not raise

    def test_rlwrld_blocks_commercial_use(self) -> None:
        """RLWRLD_NON_COMMERCIAL must raise when commercial_use=True and env not set."""
        m = self._manifest_with_license(RSkillLicensePosture.RLWRLD_NON_COMMERCIAL)
        env_backup = os.environ.pop("OPENRAL_ALLOW_NONCOMMERCIAL", None)
        try:
            with pytest.raises(ROSConfigError, match="non-commercial"):
                rSkill._check_license(m, commercial_use=True)
        finally:
            if env_backup is not None:
                os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = env_backup

    def test_rlwrld_passes_when_env_set(self) -> None:
        """RLWRLD_NON_COMMERCIAL must NOT raise when OPENRAL_ALLOW_NONCOMMERCIAL=1."""
        m = self._manifest_with_license(RSkillLicensePosture.RLWRLD_NON_COMMERCIAL)
        os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = "1"
        try:
            rSkill._check_license(m, commercial_use=True)  # must not raise
        finally:
            del os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"]

    def test_rlwrld_passes_when_non_commercial_use(self) -> None:
        """RLWRLD_NON_COMMERCIAL must NOT raise when commercial_use=False."""
        m = self._manifest_with_license(RSkillLicensePosture.RLWRLD_NON_COMMERCIAL)
        env_backup = os.environ.pop("OPENRAL_ALLOW_NONCOMMERCIAL", None)
        try:
            rSkill._check_license(m, commercial_use=False)  # no raise
        finally:
            if env_backup is not None:
                os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = env_backup

    def test_permissive_research_blocks_commercial_use(self) -> None:
        """PERMISSIVE_RESEARCH weights (e.g. π0.5) are non-commercial — must raise.

        Regression guard for the gate that previously only hard-blocked
        NVIDIA_NON_COMMERCIAL and let every other non-commercial posture
        through with an info log.
        """
        m = self._manifest_with_license(RSkillLicensePosture.PERMISSIVE_RESEARCH)
        env_backup = os.environ.pop("OPENRAL_ALLOW_NONCOMMERCIAL", None)
        try:
            with pytest.raises(ROSConfigError, match="non-commercial"):
                rSkill._check_license(m, commercial_use=True)
        finally:
            if env_backup is not None:
                os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = env_backup

    def test_permissive_research_passes_when_env_set(self) -> None:
        """PERMISSIVE_RESEARCH must NOT raise when OPENRAL_ALLOW_NONCOMMERCIAL=1."""
        m = self._manifest_with_license(RSkillLicensePosture.PERMISSIVE_RESEARCH)
        os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = "1"
        try:
            rSkill._check_license(m, commercial_use=True)  # must not raise
        finally:
            del os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"]

    def test_nvidia_open_model_always_passes(self) -> None:
        """NVIDIA_OPEN_MODEL (GR00T N1.7+) permits commercial use — never raises."""
        m = self._manifest_with_license(RSkillLicensePosture.NVIDIA_OPEN_MODEL)
        env_backup = os.environ.pop("OPENRAL_ALLOW_NONCOMMERCIAL", None)
        try:
            rSkill._check_license(m, commercial_use=True)  # must not raise
            rSkill._check_license(m, commercial_use=False)
        finally:
            if env_backup is not None:
                os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = env_backup

    def test_unknown_does_not_raise(self) -> None:
        """UNKNOWN license must log warning but not raise (still installable)."""
        m = self._manifest_with_license(RSkillLicensePosture.UNKNOWN)
        rSkill._check_license(m, commercial_use=True)


# ── Registry (list / uninstall) ────────────────────────────────────────────────


class TestRegistry:
    def test_list_installed_empty_returns_empty(self, tmp_path: Path) -> None:
        """list_installed returns [] when registry does not exist."""
        reg = tmp_path / "rskills.json"
        assert rSkill.list_installed(registry_path=reg) == []

    def test_list_installed_returns_entries(self, tmp_path: Path) -> None:
        """list_installed returns entries written by _register."""
        reg = tmp_path / "rskills.json"
        e1 = _make_entry("test/rskill-a")
        e2 = _make_entry("test/rskill-b")
        e2 = e2.model_copy(update={"installed_at": "2026-06-01T00:00:00+00:00"})
        rSkill._register(e1, reg)
        rSkill._register(e2, reg)
        entries = rSkill.list_installed(registry_path=reg)
        assert len(entries) == 2
        repo_ids = {e.repo_id for e in entries}
        assert "test/rskill-a" in repo_ids
        assert "test/rskill-b" in repo_ids

    def test_register_replaces_existing_entry(self, tmp_path: Path) -> None:
        """Registering a skill twice updates the existing entry (no duplicates)."""
        reg = tmp_path / "rskills.json"
        e1 = _make_entry("test/rskill-a")
        e2 = e1.model_copy(update={"version": "0.3.0"})
        rSkill._register(e1, reg)
        rSkill._register(e2, reg)
        entries = rSkill.list_installed(registry_path=reg)
        assert len(entries) == 1
        assert entries[0].version == "0.3.0"

    def test_uninstall_removes_entry(self, tmp_path: Path) -> None:
        """uninstall returns True and removes the entry from the registry."""
        reg = tmp_path / "rskills.json"
        rSkill._register(_make_entry("test/rskill-a"), reg)
        removed = rSkill.uninstall("test/rskill-a", registry_path=reg)
        assert removed is True
        assert rSkill.list_installed(registry_path=reg) == []

    def test_uninstall_returns_false_when_not_found(self, tmp_path: Path) -> None:
        """uninstall returns False when repo_id is not in the registry."""
        reg = tmp_path / "rskills.json"
        rSkill._register(_make_entry("test/rskill-a"), reg)
        removed = rSkill.uninstall("test/rskill-b", registry_path=reg)
        assert removed is False
        assert len(rSkill.list_installed(registry_path=reg)) == 1

    def test_uninstall_no_registry_returns_false(self, tmp_path: Path) -> None:
        """uninstall returns False gracefully when registry file doesn't exist."""
        reg = tmp_path / "rskills.json"
        assert rSkill.uninstall("test/rskill-a", registry_path=reg) is False

    def test_list_sorted_newest_first(self, tmp_path: Path) -> None:
        """list_installed returns entries newest-first by installed_at."""
        reg = tmp_path / "rskills.json"
        old = _make_entry("test/old").model_copy(
            update={"installed_at": "2026-01-01T00:00:00+00:00"}
        )
        new = _make_entry("test/new").model_copy(
            update={"installed_at": "2026-06-01T00:00:00+00:00"}
        )
        rSkill._register(old, reg)
        rSkill._register(new, reg)
        entries = rSkill.list_installed(registry_path=reg)
        assert entries[0].repo_id == "test/new"

    def test_corrupt_registry_raises_ros_config_error(self, tmp_path: Path) -> None:
        """list_installed raises ROSConfigError when the JSON is invalid."""
        reg = tmp_path / "rskills.json"
        reg.write_text("NOT_VALID_JSON{{{", encoding="utf-8")
        with pytest.raises(ROSConfigError, match="Corrupt skill registry"):
            rSkill.list_installed(registry_path=reg)


# ── check_capabilities ─────────────────────────────────────────────────────────


class TestCheckCapabilities:
    def _manifest(
        self,
        embodiment_tags: list[str] | None = None,
        capabilities_required: dict[str, bool | float | int | str] | None = None,
    ) -> RSkillManifest:
        return RSkillManifest(
            name="test/skill",
            version="0.1.0",
            license=RSkillLicensePosture.APACHE_2_0,
            role="s1",
            kind="vla",
            model_family="smolvla",
            embodiment_tags=embodiment_tags or ["so100_follower"],
            capabilities_required=capabilities_required or {},
            weights_uri="hf://test/skill",
            chunk_size=16,
            latency_budget=RSkillLatencyBudget(per_chunk_ms=100.0),
            actuators_required=list(_DEFAULT_ACTUATORS),
            processors=_default_processors(),
            description="Loader-test rSkill fixture (capability checks).",
            actions=[RSkillAction.GENERALIST],
        )

    def test_tag_mismatch_raises(self) -> None:
        """Mismatched embodiment tags must raise ROSCapabilityMismatch."""
        m = self._manifest(embodiment_tags=["franka_panda"])
        caps = RobotCapabilities(embodiment_tags=["so100_follower"])
        with pytest.raises(ROSCapabilityMismatch, match="embodiment tag"):
            rSkill.check_capabilities(m, caps)

    def test_tag_match_passes(self) -> None:
        """Matching embodiment tags must not raise."""
        m = self._manifest(embodiment_tags=["so100_follower"])
        caps = RobotCapabilities(embodiment_tags=["so100_follower"])
        rSkill.check_capabilities(m, caps)  # no raise

    def test_perception_kind_exempt_from_embodiment_match(self) -> None:
        """Detector / vlm rSkills are embodiment-agnostic: the gate passes on any
        robot via the explicit ``["any"]`` wildcard (ADR-0071).

        Real in-tree perception manifests (CLAUDE.md §1.11) ship
        ``embodiment_tags: ["any"]`` and must clear ``check_embodiment_tags``
        against a robot whose embodiment they never enumerate.
        """
        repo = Path(__file__).resolve().parents[2]
        caps = RobotCapabilities(embodiment_tags=["some_unrelated_robot"])
        for name in ("rtdetr-coco-r18", "qwen35-4b-nf4"):
            m = RSkillManifest.from_yaml(str(repo / "rskills" / name / "rskill.yaml"))
            assert m.kind in {"detector", "vlm"}
            assert m.embodiment_tags == ["any"]
            rSkill.check_embodiment_tags(m, caps)  # "any" wildcard — must not raise

    def test_bool_flag_fail_raises(self) -> None:
        """Required bool capability not met raises ROSCapabilityMismatch."""
        m = self._manifest(capabilities_required={"has_lidar": True})
        # V1 requires non-empty embodiment_tags; provide a matching robot tag
        # so the embodiment check passes and the capability check is what fires.
        caps = RobotCapabilities(embodiment_tags=["so100_follower"], has_lidar=False)
        with pytest.raises(ROSCapabilityMismatch, match="has_lidar"):
            rSkill.check_capabilities(m, caps)

    def test_bool_flag_satisfied_passes(self) -> None:
        """Required bool capability satisfied must not raise."""
        m = self._manifest(capabilities_required={"has_vision": True})
        caps = RobotCapabilities(embodiment_tags=["so100_follower"], has_vision=True)
        rSkill.check_capabilities(m, caps)

    def test_numeric_flag_below_threshold_raises(self) -> None:
        """Numeric capability below required threshold raises ROSCapabilityMismatch."""
        m = self._manifest(capabilities_required={"can_lift_kg": 5.0})
        caps = RobotCapabilities(embodiment_tags=["so100_follower"], can_lift_kg=2.0)
        with pytest.raises(ROSCapabilityMismatch, match="can_lift_kg"):
            rSkill.check_capabilities(m, caps)

    def test_numeric_flag_at_threshold_passes(self) -> None:
        """Numeric capability exactly at threshold must pass."""
        m = self._manifest(capabilities_required={"can_lift_kg": 5.0})
        caps = RobotCapabilities(embodiment_tags=["so100_follower"], can_lift_kg=5.0)
        rSkill.check_capabilities(m, caps)

    def test_unknown_flag_raises(self) -> None:
        """Requiring a flag that doesn't exist on RobotCapabilities raises."""
        m = self._manifest(capabilities_required={"has_telekinesis": True})
        caps = RobotCapabilities(embodiment_tags=["so100_follower"])
        with pytest.raises(ROSCapabilityMismatch, match="has_telekinesis"):
            rSkill.check_capabilities(m, caps)

    def test_empty_requirements_always_passes(self) -> None:
        """No capability requirements + matching embodiment tag must always pass."""
        m = self._manifest()
        rSkill.check_capabilities(m, RobotCapabilities(embodiment_tags=["so100_follower"]))

    # ── runtime + quantization (new in commit 3) ─────────────────────────────

    def _manifest_with_runtime(
        self,
        runtime: RSkillRuntime,
        quant_dtype: QuantizationDtype = QuantizationDtype.FP32,
    ) -> RSkillManifest:
        return RSkillManifest(
            name="test/rskill-runtime",
            version="0.1.0",
            license=RSkillLicensePosture.APACHE_2_0,
            role="s1",
            kind="vla",
            model_family="smolvla",
            embodiment_tags=["so100_follower"],
            weights_uri="hf://test/skill",
            runtime=runtime,
            quantization=QuantizationConfig(dtype=quant_dtype),
            chunk_size=16,
            latency_budget=RSkillLatencyBudget(per_chunk_ms=100.0),
            actuators_required=list(_DEFAULT_ACTUATORS),
            processors=_default_processors(),
            description="Loader-test rSkill fixture (runtime / quantization).",
            actions=[RSkillAction.GENERALIST],
        )

    @staticmethod
    def _caps_with_gpu_support(
        *,
        runtimes: list[RSkillRuntime] | None = None,
        dtypes: list[QuantizationDtype] | None = None,
    ) -> RobotCapabilities:
        caps = RobotCapabilities(embodiment_tags=["so100_follower"])
        object.__setattr__(caps, "gpu_supported_runtimes", list(runtimes or []))
        object.__setattr__(caps, "gpu_supported_dtypes", list(dtypes or []))
        return caps

    def test_runtime_unsupported_raises(self) -> None:
        """Skill requires TensorRT but the host only has PyTorch / ONNX."""
        m = self._manifest_with_runtime(RSkillRuntime.TENSORRT)
        caps = self._caps_with_gpu_support(runtimes=[RSkillRuntime.PYTORCH, RSkillRuntime.ONNX])
        with pytest.raises(ROSCapabilityMismatch, match="runtime"):
            rSkill.check_capabilities(m, caps)

    def test_runtime_supported_passes(self) -> None:
        m = self._manifest_with_runtime(RSkillRuntime.TENSORRT)
        caps = self._caps_with_gpu_support(runtimes=[RSkillRuntime.PYTORCH, RSkillRuntime.TENSORRT])
        rSkill.check_capabilities(m, caps)

    def test_runtime_unknown_skips_check(self) -> None:
        """Missing legacy runtime fields = unknown; should not enforce."""
        m = self._manifest_with_runtime(RSkillRuntime.TENSORRT)
        caps = RobotCapabilities(embodiment_tags=["so100_follower"])
        rSkill.check_capabilities(m, caps)  # no raise

    def test_quantization_dtype_unsupported_raises(self) -> None:
        """Skill needs FP4 but the host (Ada) only has up to FP8."""
        m = self._manifest_with_runtime(RSkillRuntime.TENSORRT, QuantizationDtype.FP4_NVFP4)
        caps = self._caps_with_gpu_support(
            runtimes=[RSkillRuntime.TENSORRT],
            dtypes=[
                QuantizationDtype.FP32,
                QuantizationDtype.FP16,
                QuantizationDtype.INT8,
            ],
        )
        with pytest.raises(ROSCapabilityMismatch, match="quantization"):
            rSkill.check_capabilities(m, caps)

    def test_quantization_dtype_supported_passes(self) -> None:
        """Skill needs FP4; Blackwell host supports it."""
        m = self._manifest_with_runtime(RSkillRuntime.TENSORRT, QuantizationDtype.FP4_NVFP4)
        caps = self._caps_with_gpu_support(
            runtimes=[RSkillRuntime.TENSORRT], dtypes=[QuantizationDtype.FP4_NVFP4]
        )
        rSkill.check_capabilities(m, caps)

    def test_quantization_dtype_unknown_skips_check(self) -> None:
        m = self._manifest_with_runtime(RSkillRuntime.TENSORRT, QuantizationDtype.FP4_NVFP4)
        # runtime declared, dtype unknown: only the runtime check applies.
        caps = self._caps_with_gpu_support(runtimes=[RSkillRuntime.TENSORRT])
        rSkill.check_capabilities(m, caps)


# ── check_sensors ──────────────────────────────────────────────────────────────


def _camera(
    name: str,
    *,
    vla_feature_key: str | None = None,
    width: int = 256,
    height: int = 256,
    modality: SensorModality = SensorModality.RGB,
) -> SensorSpec:
    return SensorSpec(
        name=name,
        modality=modality,
        frame_id="world",
        rate_hz=20.0,
        vla_feature_key=vla_feature_key,
        intrinsics=IntrinsicsPinhole(
            width=width,
            height=height,
            fx=float(width) * 0.75,
            fy=float(height) * 0.75,
            cx=float(width) / 2.0,
            cy=float(height) / 2.0,
        ),
    )


def _manifest_with_sensors(reqs: list[SensorRequirement]) -> RSkillManifest:
    return RSkillManifest(
        name="test/skill",
        version="0.1.0",
        license=RSkillLicensePosture.APACHE_2_0,
        role="s1",
        kind="vla",
        model_family="smolvla",
        embodiment_tags=["so100_follower"],
        sensors_required=reqs,
        weights_uri="hf://test/skill",
        chunk_size=16,
        latency_budget=RSkillLatencyBudget(per_chunk_ms=100.0),
        actuators_required=list(_DEFAULT_ACTUATORS),
        processors=_default_processors(),
        description="Loader-test rSkill fixture (sensor checks).",
        actions=[RSkillAction.GENERALIST],
    )


class TestCheckSensors:
    def test_no_requirements_always_passes(self) -> None:
        rSkill.check_sensors(_manifest_with_sensors([]), [])
        rSkill.check_sensors(_manifest_with_sensors([]), [_camera("cam")])

    def test_keyed_match_passes(self) -> None:
        m = _manifest_with_sensors(
            [
                SensorRequirement(
                    modality=SensorModality.RGB,
                    vla_feature_key="observation.images.camera1",
                    min_width=224,
                    min_height=224,
                )
            ]
        )
        sensors = [_camera("cam", vla_feature_key="observation.images.camera1")]
        rSkill.check_sensors(m, sensors)  # no raise

    def test_keyed_missing_raises(self) -> None:
        m = _manifest_with_sensors(
            [
                SensorRequirement(
                    modality=SensorModality.RGB,
                    vla_feature_key="observation.images.camera1",
                )
            ]
        )
        with pytest.raises(ROSCapabilityMismatch, match="vla_feature_key"):
            rSkill.check_sensors(m, [_camera("other", vla_feature_key="other.key")])

    def test_keyed_modality_mismatch_raises(self) -> None:
        m = _manifest_with_sensors(
            [
                SensorRequirement(
                    modality=SensorModality.RGB,
                    vla_feature_key="observation.images.camera1",
                )
            ]
        )
        sensors = [
            _camera(
                "depth_cam",
                vla_feature_key="observation.images.camera1",
                modality=SensorModality.DEPTH,
            )
        ]
        with pytest.raises(ROSCapabilityMismatch, match="modality"):
            rSkill.check_sensors(m, sensors)

    def test_keyed_resolution_too_low_raises(self) -> None:
        m = _manifest_with_sensors(
            [
                SensorRequirement(
                    modality=SensorModality.RGB,
                    vla_feature_key="observation.images.camera1",
                    min_width=224,
                    min_height=224,
                )
            ]
        )
        sensors = [
            _camera("cam", vla_feature_key="observation.images.camera1", width=128, height=128)
        ]
        with pytest.raises(ROSCapabilityMismatch, match="width"):
            rSkill.check_sensors(m, sensors)

    def test_modality_count_satisfied(self) -> None:
        m = _manifest_with_sensors([SensorRequirement(modality=SensorModality.RGB, count=2)])
        sensors = [_camera("cam1"), _camera("cam2")]
        rSkill.check_sensors(m, sensors)  # no raise

    def test_modality_count_unsatisfied_raises(self) -> None:
        m = _manifest_with_sensors([SensorRequirement(modality=SensorModality.RGB, count=2)])
        with pytest.raises(ROSCapabilityMismatch, match="modality"):
            rSkill.check_sensors(m, [_camera("only_one")])

    def test_modality_count_resolution_filtered_out(self) -> None:
        m = _manifest_with_sensors(
            [SensorRequirement(modality=SensorModality.RGB, count=2, min_width=224, min_height=224)]
        )
        sensors = [_camera("hi", width=256, height=256), _camera("lo", width=64, height=64)]
        with pytest.raises(ROSCapabilityMismatch, match="modality"):
            rSkill.check_sensors(m, sensors)


# ── check_compatibility (umbrella) ────────────────────────────────────────────


class TestCheckCompatibility:
    def _robot(self, *, embodiment_tags: list[str], sensors: list[SensorSpec]) -> RobotDescription:
        return RobotDescription(
            name="test_robot",
            embodiment_kind=EmbodimentKind.MANIPULATOR,
            joints=[
                JointSpec(
                    name="j",
                    joint_type=JointType.REVOLUTE,
                    parent_link="base",
                    child_link="link_1",
                )
            ],
            sensors=sensors,
            capabilities=RobotCapabilities(embodiment_tags=embodiment_tags),
            safety=SafetyEnvelope(),
        )

    def test_passes_when_both_tags_and_sensors_match(self) -> None:
        m = _manifest_with_sensors(
            [
                SensorRequirement(
                    modality=SensorModality.RGB,
                    vla_feature_key="observation.images.camera1",
                )
            ]
        )
        robot = self._robot(
            embodiment_tags=["so100_follower"],
            sensors=[_camera("cam", vla_feature_key="observation.images.camera1")],
        )
        rSkill.check_compatibility(m, robot)  # no raise

    def test_tag_mismatch_short_circuits_before_sensor_check(self) -> None:
        m = _manifest_with_sensors([])  # empty sensor reqs
        m = m.model_copy(update={"embodiment_tags": ["franka_panda"]})
        robot = self._robot(embodiment_tags=["so100_follower"], sensors=[])
        with pytest.raises(ROSCapabilityMismatch, match="embodiment tag"):
            rSkill.check_compatibility(m, robot)

    def test_sensor_mismatch_raises_when_tags_match(self) -> None:
        m = _manifest_with_sensors(
            [
                SensorRequirement(
                    modality=SensorModality.RGB,
                    vla_feature_key="observation.images.camera1",
                )
            ]
        )
        robot = self._robot(
            embodiment_tags=["so100_follower"],
            sensors=[],  # no cameras
        )
        with pytest.raises(ROSCapabilityMismatch, match="vla_feature_key"):
            rSkill.check_compatibility(m, robot)


# ── InstalledRSkillEntry schema round-trip ──────────────────────────────────────


class TestInstalledSkillEntry:
    def test_json_round_trip(self) -> None:
        """InstalledRSkillEntry must serialise to JSON and deserialise back cleanly."""
        entry = _make_entry()
        serialised = json.dumps(entry.model_dump())
        restored = InstalledRSkillEntry.model_validate_json(serialised)
        assert restored == entry

    def test_defaults_are_sensible(self) -> None:
        """Required fields only; defaults must be safe."""
        e = InstalledRSkillEntry(
            repo_id="x/y",
            local_dir="/tmp/xy",
            manifest_path="/tmp/xy/rskill.yaml",
            installed_at="2026-01-01T00:00:00+00:00",
        )
        assert e.version == "0.1.0"
        assert e.role == "s1"
        assert e.embodiment_tags == []
        assert e.revision is None


# ── rSkill.__repr__ ────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_contains_name_version_license(self, tmp_path: Path) -> None:
        """__repr__ must include name, version, and license posture."""
        p = _write_yaml(tmp_path, _APACHE_YAML)
        pkg = rSkill.from_yaml(p)
        r = repr(pkg)
        assert "test/rskill-alpha" in r
        assert "0.2.0" in r
        assert "apache-2.0" in r


# ── _validate_skill_ref ───────────────────────────────────────────────────────


class TestValidateSkillRef:
    """Exercises bare rSkill reference validation."""

    def test_bare_local_name_passes_through(self) -> None:
        assert _validate_skill_ref("smolvla-libero") == "smolvla-libero"

    def test_bare_path_passes_through(self) -> None:
        assert _validate_skill_ref("rskills/smolvla-libero") == "rskills/smolvla-libero"

    def test_hf_repo_id_passes_through(self) -> None:
        ref = "OpenRAL/rskill-smolvla-libero"
        assert _validate_skill_ref(ref) == ref

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert _validate_skill_ref("  smolvla-libero  ") == "smolvla-libero"

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ROSConfigError):
            _validate_skill_ref("")

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(ROSConfigError):
            _validate_skill_ref("   ")

    @pytest.mark.parametrize(
        "bad",
        [
            "hf://openral/smolvla-libero",
            "local:///tmp/skill",
            "file:///tmp/skill",
            "http://example.com/skill",
            "https://example.com/skill",
        ],
    )
    def test_explicit_schemes_rejected(self, bad: str) -> None:
        """Explicit URI schemes are rejected — only bare refs are accepted."""
        with pytest.raises(ROSConfigError):
            _validate_skill_ref(bad)
