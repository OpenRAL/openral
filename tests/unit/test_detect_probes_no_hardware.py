"""Tests for the no-hardware / no-optional-dep contract of every probe.

Per CLAUDE.md §1.11 we use real probes (no MagicMock); the only
allowed boundary fakes are missing-optional-dependency injections at
module import boundaries (CLAUDE.md §5.4) — implemented here by
monkey-patching ``sys.modules`` so the in-function ``import`` raises
``ImportError`` exactly as it would on a stripped host.

The probes must:
1. Return an empty / sensible default record (never raise).
2. Append a typed warning string to the supplied list.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from openral_detect import detect_hardware
from openral_detect.probes import (
    probe_dds,
    probe_gpus,
    probe_network,
    probe_realsense_devices,
    probe_usb,
    probe_v4l2_cameras,
)


@pytest.fixture
def warnings_sink() -> list[str]:
    return []


class TestUsbProbe:
    def test_probe_never_raises_and_matches_map_to_seen_devices(
        self, warnings_sink: list[str]
    ) -> None:
        # Host-agnostic contract: the probe never raises, returns typed
        # records, and every VID/PID match refers to a device it actually
        # enumerated. (Empty on USB-less CI hosts, populated on a dev
        # laptop with a robot arm plugged in — both are valid.)
        result = probe_usb(warnings=warnings_sink)
        assert isinstance(result.devices, list)
        ports = {d.port for d in result.devices}
        assert all(m.device.port in ports for m in result.matches)


class TestDdsProbe:
    def test_returns_empty_when_ros2_not_sourced(self, warnings_sink: list[str]) -> None:
        # `ros2` may not be on $PATH in CI — the probe must not raise.
        result = probe_dds(timeout_s=0.5, warnings=warnings_sink)
        assert isinstance(result.topics, list)
        assert result.inferred_robot_type is None or isinstance(result.inferred_robot_type, str)

    def test_zero_timeout_path_short_circuits(self) -> None:
        # The umbrella honors timeout_s=0 by skipping; here we just call the
        # probe directly with a tiny timeout to confirm it returns quickly.
        result = probe_dds(timeout_s=0.01, warnings=[])
        assert result.topics == []


class TestGpuProbe:
    def test_no_accelerator_returns_backend_none_or_known(self, warnings_sink: list[str]) -> None:
        result = probe_gpus(warnings=warnings_sink)
        assert result.backend in {
            "none",
            "nvml",
            "nvidia-smi",
            "lspci",
            "jtop",
            "tegra-release",
            "system_profiler",
        }
        # At minimum the probe must not raise — empty lists are valid.
        assert isinstance(result.nvidia, list)

    def test_pynvml_import_error_falls_back_silently(
        self,
        monkeypatch: pytest.MonkeyPatch,
        warnings_sink: list[str],
    ) -> None:
        # Inject ImportError for pynvml so the nvml backend is forced to
        # fall through to the nvidia-smi parser (or to nothing on a clean host).
        # The probe must not raise regardless of whether nvidia-smi is on PATH.
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", None)
        result = probe_gpus(warnings=warnings_sink)
        assert isinstance(result.nvidia, list)
        assert result.backend in {"none", "nvidia-smi", "lspci", "tegra-release", "system_profiler"}


class TestUnifiedMemoryHosts:
    """A GPU with no discrete VRAM pool must still be detected.

    On a unified-memory NVIDIA SoC (GB10 / DGX Spark, Thor)
    ``nvmlDeviceGetMemoryInfo`` returns NVML_ERROR_NOT_SUPPORTED and
    ``nvidia-smi`` prints ``[N/A]`` for memory.total. Before the fallback below,
    that single unsupported call aborted the whole enumeration and the host
    reported "GPU absent" — taking gpu_supported_dtypes and the cuMotion gate
    down with it. Measured on a DGX Spark (GB10, cc 12.1, CUDA 13.0).
    """

    def test_system_memory_matches_proc_meminfo(self) -> None:
        # Fully real: the fallback figure must agree with the kernel's own
        # numbers, and must report *available* memory rather than the
        # page-cache-starved SC_AVPHYS_PAGES (which read ~1 GiB of 122 GiB).
        from openral_detect.probes.gpu import _system_memory_mib

        meminfo = Path("/proc/meminfo")
        if not meminfo.is_file():
            pytest.skip("no /proc/meminfo on this platform")
        result = _system_memory_mib()
        assert result is not None
        total_mib, avail_mib = result
        fields = {
            line.split(":")[0]: int(line.split()[1])
            for line in meminfo.read_text().splitlines()
            if ":" in line and len(line.split()) > 1
        }
        # Kernel MemTotal excludes firmware-reserved regions, so allow 10%.
        assert total_mib == pytest.approx(fields["MemTotal"] // 1024, rel=0.1)
        assert avail_mib == fields["MemAvailable"] // 1024
        assert 0 < avail_mib <= total_mib

    def test_cuda_toolkit_prefers_cuda_home_over_path(self) -> None:
        # A distro nvcc at /usr/bin (Ubuntu ships 12.0) must not shadow a newer
        # /usr/local/cuda-N install — that is what closed the cuMotion CUDA>=13
        # gate on a Spark that genuinely has CUDA 13.0.
        from openral_detect.probes.gpu import _CUDA_HOME_NVCC, _probe_cuda_toolkit_version

        if not _CUDA_HOME_NVCC.is_file():
            pytest.skip("no /usr/local/cuda/bin/nvcc on this host")
        raw = subprocess.check_output([str(_CUDA_HOME_NVCC), "--version"], text=True)
        expected = next(ln for ln in raw.splitlines() if "release" in ln)
        expected_version = expected.split("release")[1].split(",")[0].strip()
        assert _probe_cuda_toolkit_version() == expected_version

    def test_gpu_survives_an_unsupported_memory_query(
        self,
        monkeypatch: pytest.MonkeyPatch,
        warnings_sink: list[str],
    ) -> None:
        # The NVML driver boundary is stubbed the same way the existing
        # pynvml-ImportError test above stubs it — there is no way to make a
        # real discrete GPU answer NOT_SUPPORTED, and the whole point is the
        # driver's response.
        import sys
        import types

        class _NotSupportedError(Exception):
            pass

        class _Pci:
            # Mirrors NVML's own field name, which is camelCase.
            busId = "0000000F:01:00.0"  # noqa: N815

        stub = types.SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetCount=lambda: 1,
            nvmlSystemGetDriverVersion=lambda: "580.126.09",
            nvmlDeviceGetHandleByIndex=lambda i: object(),
            nvmlDeviceGetName=lambda h: "NVIDIA GB10",
            nvmlDeviceGetCudaComputeCapability=lambda h: (12, 1),
            nvmlDeviceGetPciInfo=lambda h: _Pci(),
        )

        def _no_memory(handle: object) -> None:
            raise _NotSupportedError("NVMLError_NotSupported(3)")

        stub.nvmlDeviceGetMemoryInfo = _no_memory
        monkeypatch.setitem(sys.modules, "pynvml", stub)

        result = probe_gpus(warnings=warnings_sink)
        assert result.backend == "nvml", "unified-memory GPU was dropped from enumeration"
        gpu = next(g for g in result.nvidia if g.name == "NVIDIA GB10")
        assert gpu.cuda_compute_capability == (12, 1)
        # VRAM fell back to real system RAM, so downstream gates see a real number.
        assert gpu.vram_total_mib > 0
        assert gpu.supported_dtypes, "no dtypes derived — the cc gate never ran"
        assert any("unified memory" in w for w in warnings_sink), (
            "the shared-with-OS caveat must be surfaced, not silently swallowed"
        )


class TestCameraProbes:
    def test_v4l2_returns_list(self, warnings_sink: list[str]) -> None:
        result = probe_v4l2_cameras(warnings=warnings_sink)
        assert isinstance(result, list)

    def test_realsense_returns_empty_when_sdk_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        warnings_sink: list[str],
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pyrealsense2", None)
        result = probe_realsense_devices(warnings=warnings_sink)
        assert result == []
        assert any("pyrealsense2" in w for w in warnings_sink)


class TestNetworkProbe:
    def test_always_returns_hostname(self, warnings_sink: list[str]) -> None:
        result = probe_network(warnings=warnings_sink)
        assert result.hostname  # never empty

    def test_psutil_import_error_yields_minimal_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        warnings_sink: list[str],
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "psutil", None)
        result = probe_network(warnings=warnings_sink)
        assert result.hostname
        assert result.interfaces == []
        assert any("psutil" in w for w in warnings_sink)


class TestUmbrella:
    def test_detect_hardware_runs_clean_with_default_args(self) -> None:
        report = detect_hardware(dds_timeout_s=0.0)
        assert report.python_version
        assert report.host_os
        # warnings is a list and may legitimately be non-empty (no GPU, no USB,
        # no v4l2-ctl), but every entry must be a string.
        assert all(isinstance(w, str) for w in report.warnings)

    def test_detect_hardware_include_filter(self) -> None:
        report = detect_hardware(include={"network"}, dds_timeout_s=0.0)
        assert report.network.hostname
        # Other probes were skipped — their results should be empty defaults.
        assert report.usb.devices == []
        assert report.gpu.backend == "none"

    def test_unknown_probe_name_appends_warning(self) -> None:
        report = detect_hardware(include={"network", "totally_invalid"}, dds_timeout_s=0.0)
        assert any("totally_invalid" in w for w in report.warnings)


class TestReportRoundTrip:
    def test_detect_hardware_output_round_trips(self) -> None:
        report = detect_hardware(include={"network"}, dds_timeout_s=0.0)
        rebuilt = report.model_validate_json(report.model_dump_json())
        assert rebuilt == report


class TestGpuStaticTables:
    """Sanity checks on the static lookup tables in ``probes.gpu``."""

    def test_known_skus_have_nonzero_tops(self) -> None:
        from openral_detect.probes.gpu import _tops_for_nvidia_name

        assert _tops_for_nvidia_name("NVIDIA GeForce RTX 4090") > 0
        assert _tops_for_nvidia_name("NVIDIA H100 80GB HBM3") > 0

    def test_unknown_sku_returns_zero(self) -> None:
        from openral_detect.probes.gpu import _tops_for_nvidia_name

        assert _tops_for_nvidia_name("Acme TurboGPU 9000") == 0.0

    @pytest.mark.parametrize(
        "cc, expected_dtype",
        [
            ((10, 0), "fp4_nvfp4"),  # Blackwell
            ((9, 0), "int4"),
            ((8, 9), "bf16"),
            ((8, 7), "bf16"),  # Orin
            ((7, 5), "int8"),  # Turing
            ((6, 0), "fp16"),  # Pascal
        ],
    )
    def test_dtype_table_includes_expected(self, cc: tuple[int, int], expected_dtype: str) -> None:
        from openral_detect.probes.gpu import _dtypes_for

        dtypes: list[Any] = _dtypes_for(cc)
        assert any(d.value == expected_dtype for d in dtypes)
