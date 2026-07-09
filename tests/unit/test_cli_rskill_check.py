"""Tests for ``openral rskill check <rskill_id>`` — single-skill compatibility.

Real fixtures only (CLAUDE.md §1.11):
- Real in-tree ``rskills/<name>/rskill.yaml`` manifests.
- Real ``RobotDescription`` assembled from a hand-built ``DetectionReport``.
- Real production ``rSkill.check_*`` methods via
  ``openral_detect.check_single_rskill``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from openral_cli.main import app
from openral_core.schemas import QuantizationDtype, RobotDescription
from openral_detect import (
    DetectionReport,
    GpuProbeResult,
    NvidiaGpuInfo,
    UsbDeviceRecord,
    UsbMatchRecord,
    UsbProbeResult,
    assemble_robot_description,
)
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def so100_robot_yaml(tmp_path: Path) -> Path:
    """Write an SO-100 + RTX 4090 RobotDescription yaml to tmp_path/robot.yaml."""
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
                    ],
                )
            ],
            backend="nvml",
        ),
    )
    description: RobotDescription = assemble_robot_description(detection)
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(description.model_dump(mode="json")))
    return path


@pytest.fixture
def cpu_only_robot_yaml(tmp_path: Path) -> Path:
    """Write a CPU-only robot (no GPU) yaml — exercises GPU runtime/dtype skip."""
    detection = DetectionReport(
        detected_at="2026-05-10T00:00:00Z",
        host_os="Linux",
        python_version="3.12.3",
        usb=UsbProbeResult(devices=[], matches=[]),
        gpu=GpuProbeResult(nvidia=[], backend="none"),
    )
    description = assemble_robot_description(detection)
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(description.model_dump(mode="json")))
    return path


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSingleSkillResolution:
    """`openral rskill check <id>` resolves the id like `openral rskill list`."""

    def test_intree_bare_name_resolves(self, so100_robot_yaml: Path) -> None:
        """A bare in-tree name (e.g. ``smolvla-libero``) resolves and runs."""
        result = runner.invoke(
            app,
            ["rskill", "check", "smolvla-libero", "--robot", str(so100_robot_yaml), "--json"],
            catch_exceptions=False,
        )
        # SO-100 ≠ franka_panda → embodiment fails, exit 1.
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert len(payload["rows"]) == 1
        row = payload["rows"][0]
        assert row["repo_id"] == "OpenRAL/rskill-smolvla-libero"
        section_labels = [s["label"] for s in row["sections"]]
        assert section_labels == [
            "embodiment",
            "capability_flags",
            "gpu_runtime",
            "gpu_dtype",
            "sensors",
            "actuators",
        ]
        emb = next(s for s in row["sections"] if s["label"] == "embodiment")
        assert emb["compatible"] is False
        assert emb["failure_kind"] == "embodiment_tag"

    def test_hf_hub_skill_prefixed_alias_maps_to_intree(self, so100_robot_yaml: Path) -> None:
        """``<org>/rskill-<name>`` resolves to the same in-tree manifest."""
        result = runner.invoke(
            app,
            [
                "rskill",
                "check",
                "OpenRAL/rskill-smolvla-libero",
                "--robot",
                str(so100_robot_yaml),
                "--json",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 1  # franka_panda manifest vs so100 host
        payload = json.loads(result.output)
        assert payload["rows"][0]["repo_id"] == "OpenRAL/rskill-smolvla-libero"

    def test_rskill_uri_scheme_is_stripped(self, so100_robot_yaml: Path) -> None:
        """``<id>`` is normalised before resolution."""
        result = runner.invoke(
            app,
            [
                "rskill",
                "check",
                "smolvla-libero",
                "--robot",
                str(so100_robot_yaml),
                "--json",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["rows"][0]["repo_id"] == "OpenRAL/rskill-smolvla-libero"


class TestCompatibleHost:
    """Compatible host yields exit 0 with every blocking section green."""

    def test_so100_skill_against_so100_host_exits_zero(self, so100_robot_yaml: Path) -> None:
        # Pick an in-tree so100_follower skill the bare host can actually run.
        # ``so100_robot_yaml`` is a camera-less host (USB arm + GPU, no rgb
        # sensor), so a deployable candidate must declare no ``sensors_required``
        # — the so100-tagged VLAs (pi05/molmoact2) and the RT-DETR ``detector``
        # rSkills all require an rgb camera this host lacks and would (correctly)
        # report incompatible. Iterate sorted for a deterministic choice that
        # doesn't depend on filesystem ``iterdir`` order.
        so100_rskills = sorted(
            child.name
            for child in (REPO_ROOT / "rskills").iterdir()
            if child.is_dir() and (child / "rskill.yaml").is_file()
        )
        from openral_core.schemas import RSkillManifest

        candidate = None
        for name in so100_rskills:
            manifest = RSkillManifest.from_yaml(str(REPO_ROOT / "rskills" / name / "rskill.yaml"))
            if "so100_follower" in manifest.embodiment_tags and not manifest.sensors_required:
                candidate = name
                break
        if candidate is None:
            pytest.skip("no in-tree sensorless rSkill tags so100_follower")

        result = runner.invoke(
            app,
            ["rskill", "check", candidate, "--robot", str(so100_robot_yaml), "--json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        row = payload["rows"][0]
        assert row["compatible"] is True
        blocking = [s for s in row["sections"] if not s["informational"] and not s["compatible"]]
        assert blocking == []


class TestBadRskillId:
    """Unresolvable id → manifest_load failure, exit 1."""

    def test_unknown_id_exits_nonzero_with_manifest_load(self, so100_robot_yaml: Path) -> None:
        result = runner.invoke(
            app,
            [
                "rskill",
                "check",
                "definitely/not-a-real-skill",
                "--robot",
                str(so100_robot_yaml),
                "--json",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        payload = json.loads(result.output)
        row = payload["rows"][0]
        assert row["compatible"] is False
        assert row["failure_kind"] == "manifest_load"
        assert row["sections"] == []


class TestActuatorsInformational:
    """Actuator row never blocks the verdict."""

    def test_actuators_row_is_informational(self, so100_robot_yaml: Path) -> None:
        result = runner.invoke(
            app,
            ["rskill", "check", "smolvla-libero", "--robot", str(so100_robot_yaml), "--json"],
            catch_exceptions=False,
        )
        payload = json.loads(result.output)
        actuators = next(s for s in payload["rows"][0]["sections"] if s["label"] == "actuators")
        assert actuators["informational"] is True
        assert actuators["compatible"] is True


class TestGpuSectionsSkipOnUnknownHost:
    """When the host has no GPU info, the dtype section passes with 'unknown'.

    A CPU-only host's ``derived_runtimes()`` still returns
    ``[pytorch, onnx, gguf]`` (real CPU runtimes), so the runtime section
    is a normal positive match — not a skip.  ``derived_dtypes()``,
    however, returns an empty list when no accelerator is detected, which
    routes ``check_quantization_dtype`` into the "unknown — skip" branch.
    """

    def test_cpu_only_host_passes_gpu_sections(self, cpu_only_robot_yaml: Path) -> None:
        result = runner.invoke(
            app,
            [
                "rskill",
                "check",
                "smolvla-libero",
                "--robot",
                str(cpu_only_robot_yaml),
                "--json",
            ],
            catch_exceptions=False,
        )
        payload = json.loads(result.output)
        sections = payload["rows"][0]["sections"]
        runtime = next(s for s in sections if s["label"] == "gpu_runtime")
        dtype = next(s for s in sections if s["label"] == "gpu_dtype")
        assert runtime["compatible"] is True
        assert dtype["compatible"] is True
        assert "unknown" in (dtype["reason"] or "").lower()


class TestLegacyWalkAllStillWorks:
    """``openral rskill check`` (no positional) preserves legacy walk-all behavior."""

    def test_no_arg_falls_through_to_walk_all(self, so100_robot_yaml: Path, tmp_path: Path) -> None:
        from unittest.mock import patch

        empty_reg = tmp_path / "rskills.json"
        empty_reg.write_text("[]")
        # Point --rskills-dir at a non-existent path so the default ("rskills/")
        # doesn't accidentally pick up the in-tree rskills/ from the repo cwd.
        missing_rskills = tmp_path / "no-such-rskills"
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", empty_reg):
            result = runner.invoke(
                app,
                [
                    "rskill",
                    "check",
                    "--robot",
                    str(so100_robot_yaml),
                    "--rskills-dir",
                    str(missing_rskills),
                    "--json",
                ],
                catch_exceptions=False,
            )
        # Empty registry + missing rskills dir → empty rows, exit 0.
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["rows"] == []


class TestRobotYamlMissing:
    """Missing robot.yaml → friendly error + exit 1, even in single-skill mode."""

    def test_missing_robot_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "rskill",
                "check",
                "smolvla-libero",
                "--robot",
                str(tmp_path / "nope.yaml"),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert "openral detect" in result.output
