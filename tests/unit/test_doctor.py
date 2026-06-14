"""Unit tests for openral doctor check functions."""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError
from typing import Any
from unittest.mock import patch

import pytest
from openral_cli.main import (
    CheckResult,
    _check_colcon,
    _check_gpu,
    _check_just,
    _check_openral_core,
    _check_platform,
    _check_python,
    _check_reasoner_llm,
    _check_ros2,
    _check_usb,
    _gather_checks,
    app,
)
from typer.testing import CliRunner

runner = CliRunner()


# ── _check_python ─────────────────────────────────────────────────────────────


def test_check_python_ok() -> None:
    assert _check_python().status == "ok"
    assert _check_python().check == "Python"


def test_check_python_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 9, 0))
    assert _check_python().status == "fail"


# ── _check_platform ───────────────────────────────────────────────────────────


def test_check_platform_info() -> None:
    result = _check_platform()
    assert result.status == "info"
    assert result.details != ""


# ── _check_openral_core ───────────────────────────────────────────────────


def test_check_openral_core_ok() -> None:
    with patch("openral_cli.main.version", return_value="0.1.0"):
        result = _check_openral_core()
    assert result.status == "ok"
    assert result.details == "0.1.0"


def test_check_openral_core_missing() -> None:
    with patch("openral_cli.main.version", side_effect=PackageNotFoundError("x")):
        result = _check_openral_core()
    assert result.status == "fail"


# ── _check_ros2 ───────────────────────────────────────────────────────────────


def test_check_ros2_binary_missing() -> None:
    with patch("openral_cli.main.shutil.which", return_value=None):
        results = _check_ros2()
    assert len(results) == 1
    assert results[0].status == "missing"


def test_check_ros2_binary_found_distro_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROS_DISTRO", "jazzy")
    monkeypatch.delenv("RMW_IMPLEMENTATION", raising=False)
    with patch("openral_cli.main.shutil.which", return_value="/usr/bin/ros2"):
        results = _check_ros2()
    checks = {r.check: r for r in results}
    assert checks["ROS 2 binary"].status == "ok"
    assert checks["ROS 2 distro"].status == "ok"
    assert checks["ROS 2 distro"].details == "jazzy"
    assert checks["RMW"].status == "info"


def test_check_ros2_distro_not_sourced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROS_DISTRO", raising=False)
    monkeypatch.delenv("RMW_IMPLEMENTATION", raising=False)
    with (
        patch("openral_cli.main.shutil.which", return_value="/usr/bin/ros2"),
        patch("openral_cli.main.glob", return_value=["/opt/ros/jazzy/setup.bash"]),
    ):
        results = _check_ros2()
    checks = {r.check: r for r in results}
    assert checks["ROS 2 distro"].status == "info"
    assert "jazzy" in checks["ROS 2 distro"].details


def test_check_ros2_no_opt_ros(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROS_DISTRO", raising=False)
    with (
        patch("openral_cli.main.shutil.which", return_value="/usr/bin/ros2"),
        patch("openral_cli.main.glob", return_value=[]),
    ):
        results = _check_ros2()
    checks = {r.check: r for r in results}
    assert checks["ROS 2 distro"].status == "missing"


# ── _check_colcon ─────────────────────────────────────────────────────────────


def test_check_colcon_present() -> None:
    with patch("openral_cli.main.shutil.which", return_value="/usr/bin/colcon"):
        assert _check_colcon().status == "ok"


def test_check_colcon_missing() -> None:
    with patch("openral_cli.main.shutil.which", return_value=None):
        assert _check_colcon().status == "missing"


# ── _check_gpu ────────────────────────────────────────────────────────────────


def _gpu_probe_stub(*, nvidia=None, jetson=None, apple_silicon=None, backend="none"):
    from openral_detect.report import GpuProbeResult

    return GpuProbeResult(
        nvidia=nvidia or [],
        jetson=jetson,
        apple_silicon=apple_silicon,
        backend=backend,
    )


def test_check_gpu_nvidia_ok() -> None:
    from openral_detect.report import NvidiaGpuInfo

    nv = NvidiaGpuInfo(
        index=0,
        name="NVIDIA GeForce RTX 4090",
        vram_total_mib=24576,
        vram_free_mib=24000,
        pci_bus_id="0000:01:00.0",
        driver_version="550.78",
        cuda_compute_capability=(8, 9),
    )
    with patch(
        "openral_detect.probes.probe_gpus",
        return_value=_gpu_probe_stub(nvidia=[nv], backend="nvml"),
    ):
        results = _check_gpu()
    assert len(results) == 1
    assert results[0].status == "ok"
    assert "RTX 4090" in results[0].details
    assert "24576 MiB" in results[0].details


def test_check_gpu_multi() -> None:
    from openral_detect.report import NvidiaGpuInfo

    cards = [
        NvidiaGpuInfo(
            index=0,
            name="Tesla T4",
            vram_total_mib=15360,
            vram_free_mib=15000,
            pci_bus_id="0000:01:00.0",
            driver_version="550",
            cuda_compute_capability=(7, 5),
        ),
        NvidiaGpuInfo(
            index=1,
            name="RTX 3090",
            vram_total_mib=24576,
            vram_free_mib=24000,
            pci_bus_id="0000:02:00.0",
            driver_version="550",
            cuda_compute_capability=(8, 6),
        ),
    ]
    with patch(
        "openral_detect.probes.probe_gpus",
        return_value=_gpu_probe_stub(nvidia=cards, backend="nvml"),
    ):
        results = _check_gpu()
    assert len(results) == 2
    assert results[0].check == "GPU 0"
    assert results[1].check == "GPU 1"


def test_check_gpu_nvsmi_query_fails() -> None:
    # Probe failed → empty result with a warning; doctor surfaces "absent".
    with patch(
        "openral_detect.probes.probe_gpus",
        return_value=_gpu_probe_stub(),
    ):
        results = _check_gpu()
    assert results[0].status == "absent"


def test_check_gpu_absent_non_mac() -> None:
    with patch(
        "openral_detect.probes.probe_gpus",
        return_value=_gpu_probe_stub(),
    ):
        results = _check_gpu()
    assert results[0].status == "absent"


def test_check_gpu_apple_silicon() -> None:
    from openral_detect.report import AppleSiliconInfo

    with patch(
        "openral_detect.probes.probe_gpus",
        return_value=_gpu_probe_stub(
            apple_silicon=AppleSiliconInfo(chip="Apple M3 Max", gpu_cores=40),
            backend="system_profiler",
        ),
    ):
        results = _check_gpu()
    assert results[0].status == "info"
    assert "Apple Silicon" in results[0].details


# ── _check_usb ────────────────────────────────────────────────────────────────


def test_check_usb_linux_devices_found() -> None:
    with (
        patch("openral_cli.main.platform.system", return_value="Linux"),
        patch(
            "openral_cli.main.glob",
            side_effect=lambda p: ["/dev/ttyUSB0"] if "ttyUSB" in p else [],
        ),
    ):
        results = _check_usb()
    assert results[0].status == "ok"
    assert "/dev/ttyUSB0" in results[0].details


def test_check_usb_linux_none() -> None:
    with (
        patch("openral_cli.main.platform.system", return_value="Linux"),
        patch("openral_cli.main.glob", return_value=[]),
    ):
        results = _check_usb()
    assert results[0].status == "info"
    assert "none" in results[0].details


def test_check_usb_windows() -> None:
    with patch("openral_cli.main.platform.system", return_value="Windows"):
        results = _check_usb()
    assert results[0].status == "info"


# ── _check_just ───────────────────────────────────────────────────────────────


def test_check_just_present() -> None:
    with patch("openral_cli.main.shutil.which", return_value="/usr/local/bin/just"):
        assert _check_just().status == "ok"


def test_check_just_missing() -> None:
    # `just` absence is non-fatal (`warn`, not `missing`) so `openral doctor`
    # still exits 0 on hosts that only need to run skills.
    with patch("openral_cli.main.shutil.which", return_value=None):
        assert _check_just().status == "warn"


# ── _check_reasoner_llm ───────────────────────────────────────────────────────


_REASONER_ENV = (
    "OPENRAL_REASONER_LLM_PROVIDER",
    "OPENRAL_REASONER_LLM_MODEL",
    "OPENRAL_REASONER_LLM_API_KEY",
    "OPENRAL_REASONER_LLM_BASE_URL",
)


@pytest.fixture
def _clear_reasoner_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _REASONER_ENV:
        monkeypatch.delenv(key, raising=False)


def test_check_reasoner_llm_absent(_clear_reasoner_env: None) -> None:
    rows = _check_reasoner_llm()
    assert len(rows) == 1
    assert rows[0].check == "Reasoner LLM"
    assert rows[0].status == "absent"
    assert "OPENRAL_REASONER_LLM_PROVIDER" in rows[0].details
    assert "packages/openral_reasoner_ros/README.md" in rows[0].details


def test_check_reasoner_llm_unknown_provider(
    monkeypatch: pytest.MonkeyPatch, _clear_reasoner_env: None
) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "groq-cloud")
    rows = _check_reasoner_llm()
    assert len(rows) == 1
    assert rows[0].status == "fail"
    assert "groq-cloud" in rows[0].details
    assert "anthropic" in rows[0].details


def test_check_reasoner_llm_anthropic_ok_redacts_key(
    monkeypatch: pytest.MonkeyPatch, _clear_reasoner_env: None
) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-ant-secret-12345")
    rows = _check_reasoner_llm()
    summary = next(r for r in rows if r.check == "Reasoner LLM")
    assert summary.status == "ok"
    assert "provider=anthropic" in summary.details
    assert "model=claude-haiku-4-5" in summary.details
    assert "api_key=set" in summary.details
    # API key value must never appear in the rendered output.
    assert "sk-ant-secret-12345" not in summary.details


def test_check_reasoner_llm_anthropic_missing_model(
    monkeypatch: pytest.MonkeyPatch, _clear_reasoner_env: None
) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-ant-secret")
    rows = _check_reasoner_llm()
    summary = next(r for r in rows if r.check == "Reasoner LLM")
    assert summary.status == "warn"
    follow_up = next(r for r in rows if r.check == "Reasoner MODEL")
    assert follow_up.status == "missing"
    assert "OPENRAL_REASONER_LLM_MODEL" in follow_up.details


def test_check_reasoner_llm_openrouter_missing_key(
    monkeypatch: pytest.MonkeyPatch, _clear_reasoner_env: None
) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "deepseek/deepseek-chat-v3:free")
    rows = _check_reasoner_llm()
    summary = next(r for r in rows if r.check == "Reasoner LLM")
    assert summary.status == "warn"
    key_row = next(r for r in rows if r.check == "Reasoner API_KEY")
    assert key_row.status == "missing"
    assert "openrouter" in key_row.details
    # The default OpenRouter URL is surfaced in the summary even though
    # BASE_URL isn't set explicitly.
    assert "openrouter.ai/api/v1" in summary.details


def test_check_reasoner_llm_local_endpoint_unreachable(
    monkeypatch: pytest.MonkeyPatch, _clear_reasoner_env: None
) -> None:
    # Port 1 is guaranteed-closed (privileged, never bound by a dev tool).
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_BASE_URL", "http://127.0.0.1:1/v1")
    rows = _check_reasoner_llm()
    ollama = next(r for r in rows if r.check == "Ollama")
    assert ollama.status == "warn"
    assert "bootstrap-ollama" in ollama.details


def test_check_reasoner_llm_cloud_endpoint_no_ollama_probe(
    monkeypatch: pytest.MonkeyPatch, _clear_reasoner_env: None
) -> None:
    """Cloud base URLs must not emit an Ollama row (no probe out)."""
    monkeypatch.setenv("OPENRAL_REASONER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_MODEL", "deepseek/deepseek-chat-v3:free")
    monkeypatch.setenv("OPENRAL_REASONER_LLM_API_KEY", "sk-or-secret")
    rows = _check_reasoner_llm()
    assert not any(r.check == "Ollama" for r in rows)


# ── _gather_checks ────────────────────────────────────────────────────────────


def test_gather_checks_returns_list() -> None:
    checks = _gather_checks()
    assert len(checks) >= 7
    assert all(isinstance(c, CheckResult) for c in checks)
    names = [c.check for c in checks]
    assert "Python" in names
    assert "Platform" in names
    assert "Reasoner LLM" in names


# ── CLI integration (typer CliRunner) ─────────────────────────────────────────


def test_doctor_table_output() -> None:
    result = runner.invoke(app, ["doctor"])
    assert "openral doctor" in result.output
    assert "Python" in result.output


def test_doctor_json_output() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    data: list[Any] = json.loads(result.output)
    assert isinstance(data, list)
    assert all({"check", "status", "details"} <= set(row.keys()) for row in data)
    names = [row["check"] for row in data]
    assert "Python" in names


def test_doctor_exits_1_on_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 9, 0))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1


def test_doctor_exits_0_on_all_ok() -> None:
    """Patch all checks to return 'ok'/'info' statuses."""
    happy: list[CheckResult] = [
        CheckResult("Python", "ok", "3.12.0"),
        CheckResult("Platform", "info", "Linux 6.0"),
    ]
    with patch("openral_cli.main._gather_checks", return_value=happy):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
