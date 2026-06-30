"""Tests for :func:`check_installed_rskills`.

Real fixtures only (CLAUDE.md §1.11):
- Real in-tree ``rskills/<name>/rskill.yaml`` manifests.
- Real ``robots/so100_follower/robot.yaml``.
- Real ``rSkill.check_compatibility`` semantics — the compatibility
  report does not re-implement matching, it wraps the production check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.schemas import QuantizationDtype, RobotDescription
from openral_detect import (
    CompatibilityReport,
    DetectionReport,
    GpuProbeResult,
    NvidiaGpuInfo,
    UsbDeviceRecord,
    UsbMatchRecord,
    UsbProbeResult,
    assemble_robot_description,
    check_installed_rskills,
)
from openral_detect.compatibility import _classify

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "rskills"


@pytest.fixture
def empty_registry(tmp_path: Path) -> Path:
    """Empty registry file — keeps the unit test hermetic."""
    p = tmp_path / "rskills.json"
    p.write_text("[]")
    return p


@pytest.fixture
def so100_robot() -> RobotDescription:
    """SO-100 base + RTX 4090 GPU caps so we can exercise runtime matching."""
    detection = DetectionReport(
        detected_at="2026-05-10T00:00:00Z",
        host_os="Linux",
        python_version="3.12.3",
        usb=UsbProbeResult(
            devices=[UsbDeviceRecord(port="/dev/ttyUSB0", vid=0x1A86, pid=0x7523, description="")],
            matches=[
                UsbMatchRecord(
                    device=UsbDeviceRecord(
                        port="/dev/ttyUSB0", vid=0x1A86, pid=0x7523, description=""
                    ),
                    chip="CH340",
                    driver_hint="Feetech",
                    embodiment_tag="so100_follower",
                    bh_robot_type="so100",
                )
            ],
        ),
        gpu=GpuProbeResult(
            nvidia=[
                NvidiaGpuInfo(
                    index=0,
                    name="NVIDIA GeForce RTX 4090",
                    vram_total_mib=24576,
                    vram_free_mib=24000,
                    pci_bus_id="0000:01:00.0",
                    driver_version="550.78",
                    cuda_compute_capability=(8, 9),
                    tops_estimate=1321.0,
                    supported_dtypes=[
                        QuantizationDtype.FP32,
                        QuantizationDtype.FP16,
                        QuantizationDtype.BF16,
                        QuantizationDtype.INT8,
                        # RTX 4090 (sm_89) runs bitsandbytes NF4/INT4 — required
                        # by the in-tree nf4 SO-100/101 skills (e.g.
                        # molmoact2-so101-nf4). Omitting it made a legitimately
                        # compatible nf4 skill read as incompatible.
                        QuantizationDtype.INT4,
                    ],
                )
            ],
            backend="nvml",
        ),
    )
    return assemble_robot_description(detection)


class TestEmptyRegistry:
    def test_empty_registry_yields_empty_rows(
        self, empty_registry: Path, so100_robot: RobotDescription
    ) -> None:
        report = check_installed_rskills(so100_robot, registry_path=empty_registry)
        assert report.rows == []
        assert report.compatible == []
        assert report.incompatible == []
        assert report.robot_name == "so100_follower"


class TestInTreeSkillsAgainstSo100:
    def test_skills_dir_walk_yields_real_rows(
        self, empty_registry: Path, so100_robot: RobotDescription
    ) -> None:
        report = check_installed_rskills(
            so100_robot, registry_path=empty_registry, rskills_dir=SKILLS_DIR
        )
        assert len(report.rows) >= 4
        # SO-100-tagged skills must not be rejected on EMBODIMENT grounds
        # (their embodiment intersects so100_follower). A skill may still fail
        # on an orthogonal sensor/capability check — e.g. the `kind: detector`
        # RT-DETR rSkills are embodiment-agnostic (``embodiment_tags: ["any"]``,
        # ADR-0072) so they clear the embodiment gate but still require a 640x480
        # RGB camera the bare SO-100 manifest doesn't declare
        # (failure_kind="sensor_modality") — which is the sensor case, not the
        # embodiment case under test here.
        so100_rows = [r for r in report.rows if "so100_follower" in r.embodiment_tags]
        assert so100_rows, "expected at least one so100_follower-tagged manifest"
        assert all(r.failure_kind != "embodiment_tag" for r in so100_rows)
        # Skills with **no shared tag** with SO-100 must fail on embodiment_tag.
        # Excludes the ``["any"]`` wildcard skills (they clear the gate) and skills
        # that share `lerobot` (they may fail later on a sensor/capability check).
        so100_set = {"so100_follower", "lerobot"}
        embodiment_fail_rows = [
            r
            for r in report.rows
            if r.embodiment_tags
            and "any" not in r.embodiment_tags
            and not set(r.embodiment_tags) & so100_set
        ]
        assert embodiment_fail_rows, "expected at least one disjoint-embodiment manifest"
        for r in embodiment_fail_rows:
            assert r.compatible is False
            assert r.failure_kind == "embodiment_tag"
            assert "embodiment tag" in (r.reason or "").lower()

    def test_round_trip_through_json(
        self, empty_registry: Path, so100_robot: RobotDescription
    ) -> None:
        report = check_installed_rskills(
            so100_robot, registry_path=empty_registry, rskills_dir=SKILLS_DIR
        )
        rebuilt = CompatibilityReport.model_validate_json(report.model_dump_json())
        assert rebuilt.rows == report.rows


class TestFailureKindClassifier:
    @pytest.mark.parametrize(
        "message, expected",
        [
            ("rSkill 'x' requires embodiment tag(s) ['aloha']", "embodiment_tag"),
            ("rSkill 'x' requires runtime 'tensorrt', but robot ...", "runtime"),
            ("rSkill 'x' requires quantization dtype 'fp4_nvfp4'", "quantization"),
            ("rSkill 'x' requires VLA feature key 'observation.images'", "sensor_key"),
            ("rSkill 'x' requires modality DEPTH", "sensor_modality"),
            ("rSkill 'x' resolution too low (320x240 < 640x480)", "resolution"),
            ("rSkill 'x' requires 'has_lidar=True'", "capability_flag"),
        ],
    )
    def test_classifier_returns_expected_kind(self, message: str, expected: str) -> None:
        assert _classify(message) == expected


class TestManifestLoadFailureCarriesThrough:
    def test_invalid_manifest_yields_manifest_load_failure_kind(
        self, tmp_path: Path, so100_robot: RobotDescription
    ) -> None:
        # Synthesize an in-tree-style folder with a malformed yaml.
        rskills_dir = tmp_path / "skills"
        bad = rskills_dir / "broken-skill"
        bad.mkdir(parents=True)
        (bad / "rskill.yaml").write_text("not: a valid: rskill\nroot: [")
        empty_registry = tmp_path / "empty-registry.json"
        report = check_installed_rskills(
            so100_robot, registry_path=empty_registry, rskills_dir=rskills_dir
        )
        assert len(report.rows) == 1
        row = report.rows[0]
        assert row.compatible is False
        assert row.failure_kind == "manifest_load"
        assert "manifest load failed" in (row.reason or "")
