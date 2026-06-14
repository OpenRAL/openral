"""GPU / accelerator probe — NVIDIA + Jetson + Apple Silicon.

Backend priority:
- NVIDIA: ``pynvml`` → ``nvidia-smi`` parse → ``lspci`` last resort.
- Jetson: ``jtop`` → ``/etc/nv_tegra_release`` + ``/proc/device-tree/model``.
- Apple: ``system_profiler SPDisplaysDataType -json`` (macOS arm64).

Every backend that fails (missing optional dep, command absent, parse
error) appends a typed warning and produces an empty result; the probe
never raises.

The output is rich enough that the assembler can populate
``RobotCapabilities.gpu_supported_runtimes`` /
``RobotCapabilities.gpu_supported_dtypes`` without re-probing.
"""

from __future__ import annotations

import contextlib
import json
import platform
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from openral_core.schemas import QuantizationDtype

from openral_detect.report import (
    AppleSiliconInfo,
    GpuProbeResult,
    JetsonInfo,
    NvidiaGpuInfo,
)

__all__ = [
    "DTYPES_BY_COMPUTE_CAPABILITY",
    "JETSON_BOARD_TOPS",
    "NVIDIA_TOPS_BY_NAME_KEYWORD",
    "probe_gpus",
]

# ── Static lookup tables ──────────────────────────────────────────────────────

#: Coarse TOPS estimate keyed by a substring that must appear in the GPU
#: ``name`` field.  First-match-wins; entries are ordered most-specific
#: first.  Numbers are vendor-published peak INT8 dense TOPS for the SKU.
NVIDIA_TOPS_BY_NAME_KEYWORD: tuple[tuple[str, float], ...] = (
    ("RTX 5090", 3352.0),
    ("RTX 5080", 1801.0),
    ("RTX 4090", 1321.0),
    ("RTX 4080", 780.0),
    ("RTX 4070 Ti", 706.0),
    ("RTX 4070", 466.0),
    ("RTX 4060", 242.0),
    ("RTX 3090", 285.0),
    ("RTX 3080", 238.0),
    ("RTX 3070", 163.0),
    ("RTX 3060", 102.0),
    ("A100", 624.0),
    ("H100", 1979.0),
    ("L40", 724.0),
    ("L4", 242.0),
    ("T4", 130.0),
)

#: TOPS per Jetson board.  Source: NVIDIA Jetson product briefs (peak INT8).
JETSON_BOARD_TOPS: dict[str, float] = {
    "Jetson AGX Orin": 275.0,
    "Jetson Orin NX": 100.0,
    "Jetson Orin Nano": 40.0,
    "Jetson AGX Xavier": 32.0,
    "Jetson Xavier NX": 21.0,
    "Jetson Nano": 0.5,
}

#: Quantization dtypes a CUDA accelerator can reasonably execute, keyed by
#: the major.minor compute capability.  Conservative — only adds a dtype
#: when it has hardware backing (Tensor Core or quantized math).
DTYPES_BY_COMPUTE_CAPABILITY: tuple[tuple[tuple[int, int], tuple[QuantizationDtype, ...]], ...] = (
    # Blackwell (RTX 50 / B200): adds NVFP4.
    (
        (10, 0),
        (
            QuantizationDtype.FP32,
            QuantizationDtype.FP16,
            QuantizationDtype.BF16,
            QuantizationDtype.INT8,
            QuantizationDtype.INT4,
            QuantizationDtype.FP4_NVFP4,
        ),
    ),
    # Hopper (H100/H200): adds FP8.
    (
        (9, 0),
        (
            QuantizationDtype.FP32,
            QuantizationDtype.FP16,
            QuantizationDtype.BF16,
            QuantizationDtype.INT8,
            QuantizationDtype.INT4,
        ),
    ),
    # Ada Lovelace (RTX 40, L40): adds FP8.
    (
        (8, 9),
        (
            QuantizationDtype.FP32,
            QuantizationDtype.FP16,
            QuantizationDtype.BF16,
            QuantizationDtype.INT8,
            QuantizationDtype.INT4,
        ),
    ),
    # Ampere (A100, RTX 30, Orin): BF16 + INT8 + INT4.
    (
        (8, 0),
        (
            QuantizationDtype.FP32,
            QuantizationDtype.FP16,
            QuantizationDtype.BF16,
            QuantizationDtype.INT8,
            QuantizationDtype.INT4,
        ),
    ),
    # Turing (T4, RTX 20): INT8 + INT4 (no BF16 tensor cores).
    (
        (7, 5),
        (
            QuantizationDtype.FP32,
            QuantizationDtype.FP16,
            QuantizationDtype.INT8,
            QuantizationDtype.INT4,
        ),
    ),
    # Volta (V100): FP16 + INT8.
    (
        (7, 0),
        (
            QuantizationDtype.FP32,
            QuantizationDtype.FP16,
            QuantizationDtype.INT8,
        ),
    ),
    # Pascal (Jetson Xavier era predecessors): FP16 only.
    (
        (6, 0),
        (
            QuantizationDtype.FP32,
            QuantizationDtype.FP16,
        ),
    ),
)


def _dtypes_for(cc: tuple[int, int]) -> list[QuantizationDtype]:
    """Map a compute capability to supported quantization dtypes."""
    for major_minor, dtypes in DTYPES_BY_COMPUTE_CAPABILITY:
        if cc >= major_minor:
            return list(dtypes)
    return [QuantizationDtype.FP32]


def _tops_for_nvidia_name(name: str) -> float:
    for keyword, tops in NVIDIA_TOPS_BY_NAME_KEYWORD:
        if keyword in name:
            return tops
    return 0.0


def _tops_for_jetson_board(board: str) -> float:
    for k, v in JETSON_BOARD_TOPS.items():
        if k in board:
            return v
    return 0.0


#: Canonical mapping from device-tree model keyword → CUDA compute capability.
#: Order matters: more-specific keys first (e.g. ``"Orin NX"`` would be a
#: stricter match than ``"Orin"``, but the substring check finds the latter
#: anyway, so we list only one entry per SoC family). ``"Nano"`` matches
#: Maxwell-era Nano only — Orin Nano contains ``"Orin"`` and is caught by
#: the first entry. ADR-0013 §3.1 mandates the explicit table; the legacy
#: ``(8, 7) if "Orin" in board else (7, 2)`` heuristic silently classified
#: Xavier and Maxwell-Nano boards as Volta (CC 7.2), masking a
#: best-effort gap.
_JETSON_CC_BY_BOARD_KEYWORD: tuple[tuple[str, tuple[int, int]], ...] = (
    ("Orin", (8, 7)),  # Orin AGX / Orin NX / Orin Nano — Ampere CC 8.7
    ("Xavier", (7, 2)),  # Xavier AGX / Xavier NX — Volta CC 7.2
    ("Nano", (5, 3)),  # Legacy Maxwell Nano — best-effort per ADR-0013
)


def _cc_for_jetson_board(board: str) -> tuple[int, int] | None:
    """Return the CUDA compute capability for a Jetson board name, or ``None``.

    Replaces the legacy ``(8, 7) if "Orin" in board else (7, 2)``
    heuristic with the explicit :data:`_JETSON_CC_BY_BOARD_KEYWORD`
    table mandated by ADR-0013 §3.1.
    """
    for keyword, cc in _JETSON_CC_BY_BOARD_KEYWORD:
        if keyword in board:
            return cc
    return None


#: Canonical L4T install paths for ``libnvbufsurface.so``. Matches the
#: search list that
#: ``python/runner/src/openral_runner/backends/gstreamer/nvbufsurface.py``
#: walks at module load. Tegra (JetPack r35+) installs the library under
#: ``/usr/lib/aarch64-linux-gnu/tegra/``; stripped L4T images may have it
#: under the parent directory or omit it entirely.
_NVBUFSURFACE_SEARCH_PATHS: tuple[Path, ...] = (
    Path("/usr/lib/aarch64-linux-gnu/tegra"),
    Path("/usr/lib/aarch64-linux-gnu"),
)


def _probe_nvmm_available(*, search_paths: Sequence[Path] | None = None) -> bool:
    """Return ``True`` when ``libnvbufsurface.so`` is installed on this host.

    Populates :attr:`RobotCapabilities.nvmm_available` so
    ``rSkill.check_capabilities`` can refuse skills that require the
    NVMM zero-copy sensor-ingest path on a host that cannot provide it.
    The library ships with the L4T multimedia stack on JetPack r35+;
    its absence on a stripped L4T image (or on any non-Tegra host) means
    the NVMM path is unavailable even when the host is otherwise a
    Tegra. ADR-0013 PR 2/3.

    Args:
        search_paths: Override search roots for tests. Production omits
            this and the canonical L4T install locations
            (:data:`_NVBUFSURFACE_SEARCH_PATHS`) are walked.

    Returns:
        ``True`` iff ``libnvbufsurface.so`` exists in any of the
        configured search paths.
    """
    paths: Sequence[Path] = search_paths if search_paths is not None else _NVBUFSURFACE_SEARCH_PATHS
    return any((root / "libnvbufsurface.so").exists() for root in paths)


# ── NVIDIA discrete via pynvml ────────────────────────────────────────────────


def _probe_nvidia_pynvml(warnings: list[str]) -> list[NvidiaGpuInfo]:
    try:
        import pynvml  # noqa: PLC0415  # reason: optional extra
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        warnings.append(f"gpu.nvml: nvmlInit failed: {exc!r}")
        return []
    out: list[NvidiaGpuInfo] = []
    try:
        n = pynvml.nvmlDeviceGetCount()
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8", errors="replace")
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            cc = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            pci = pynvml.nvmlDeviceGetPciInfo(h)
            bus_id = pci.busId
            if isinstance(bus_id, bytes):
                bus_id = bus_id.decode("utf-8", errors="replace")
            cc_tuple = (int(cc[0]), int(cc[1]))
            out.append(
                NvidiaGpuInfo(
                    index=i,
                    name=name,
                    vram_total_mib=int(mem.total // (1024 * 1024)),
                    vram_free_mib=int(mem.free // (1024 * 1024)),
                    pci_bus_id=bus_id,
                    driver_version=driver,
                    cuda_compute_capability=cc_tuple,
                    cuda_toolkit_version=_probe_cuda_toolkit_version(),
                    tensorrt_version=_probe_tensorrt_version(),
                    supported_dtypes=_dtypes_for(cc_tuple),
                    tops_estimate=_tops_for_nvidia_name(name),
                )
            )
    except Exception as exc:
        warnings.append(f"gpu.nvml: enumeration failed: {exc!r}")
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
    return out


# ── NVIDIA discrete via nvidia-smi (fallback) ────────────────────────────────


def _probe_nvidia_smi(warnings: list[str]) -> list[NvidiaGpuInfo]:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return []
    try:
        raw = subprocess.check_output(
            [
                nvsmi,
                "--query-gpu=index,name,memory.total,memory.free,driver_version,"
                "compute_cap,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        warnings.append(f"gpu.nvidia-smi: query failed: {exc!r}")
        return []
    out: list[NvidiaGpuInfo] = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:  # noqa: PLR2004  # reason: 7 columns requested above
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            vram_total = int(parts[2])
            vram_free = int(parts[3])
            driver = parts[4]
            cc_str = parts[5]
            bus_id = parts[6]
            cc_major, cc_minor = (int(x) for x in cc_str.split("."))
        except (ValueError, IndexError):
            warnings.append(f"gpu.nvidia-smi: malformed row {line!r}")
            continue
        cc = (cc_major, cc_minor)
        out.append(
            NvidiaGpuInfo(
                index=idx,
                name=name,
                vram_total_mib=vram_total,
                vram_free_mib=vram_free,
                pci_bus_id=bus_id,
                driver_version=driver,
                cuda_compute_capability=cc,
                cuda_toolkit_version=_probe_cuda_toolkit_version(),
                tensorrt_version=_probe_tensorrt_version(),
                supported_dtypes=_dtypes_for(cc),
                tops_estimate=_tops_for_nvidia_name(name),
            )
        )
    return out


# ── Jetson via jtop / proc ────────────────────────────────────────────────────


_DEFAULT_MODEL_PATH: Path = Path("/proc/device-tree/model")
_DEFAULT_RELEASE_PATH: Path = Path("/etc/nv_tegra_release")


def _probe_jetson(
    warnings: list[str],
    *,
    model_path: Path | None = None,
    release_path: Path | None = None,
) -> JetsonInfo | None:
    """Probe for a Tegra (Jetson / Spark) host.

    Args:
        warnings: Mutated list; structured warnings are appended.
        model_path: Override for ``/proc/device-tree/model``. Tests pass a
            recorded fixture; production omits this and reads the real path.
        release_path: Override for ``/etc/nv_tegra_release``.

    Returns:
        ``JetsonInfo`` on a recognised board; ``None`` when no Tegra is
        detected or the board name is unknown (with a warning appended).
    """
    model_path = model_path or _DEFAULT_MODEL_PATH
    release_path = release_path or _DEFAULT_RELEASE_PATH
    use_defaults = model_path == _DEFAULT_MODEL_PATH and release_path == _DEFAULT_RELEASE_PATH

    # First try jtop — production path only. Tests pass overridden paths to
    # exercise the deterministic /proc + /etc fallback.
    if use_defaults:
        try:
            from jtop import jtop  # noqa: PLC0415  # reason: optional extra
        except ImportError:
            jtop = None  # type: ignore[assignment]  # reason: optional path
        if jtop is not None:
            try:
                with jtop() as j:
                    if j.ok():
                        board = str(j.board.get("Type", j.board.get("Model", "Jetson")))
                        soc = str(j.board.get("Module", ""))
                        jp = str(j.board.get("Jetpack", ""))
                        ram_gb = float(j.memory.get("RAM", {}).get("tot", 0.0)) / 1024.0
                        power_mode = str(j.nvpmodel) if hasattr(j, "nvpmodel") else ""
                        cc = _cc_for_jetson_board(board)
                        if cc is None:
                            warnings.append(
                                f"gpu.jetson: unknown board {board!r} from jtop; "
                                "compute capability not recorded"
                            )
                            return None
                        return JetsonInfo(
                            board=board,
                            soc=soc,
                            jetpack_version=jp,
                            tops=_tops_for_jetson_board(board),
                            ram_gb=ram_gb,
                            cuda_compute_capability=cc,
                            cuda_toolkit_version=_probe_cuda_toolkit_version(),
                            tensorrt_version=_probe_tensorrt_version(),
                            supported_dtypes=_dtypes_for(cc),
                            power_mode=power_mode,
                        )
            except Exception as exc:
                warnings.append(f"gpu.jtop: probe failed: {exc!r}")

    # Fallback — parse /proc + /etc/nv_tegra_release (or their test overrides).
    if not model_path.exists() and not release_path.exists():
        return None
    board = ""
    if model_path.exists():
        try:
            board = model_path.read_text(errors="replace").strip("\x00 \n")
        except OSError as exc:
            warnings.append(f"gpu.jetson: cannot read {model_path}: {exc!r}")
    jetpack = ""
    if release_path.exists():
        with contextlib.suppress(OSError, IndexError):
            jetpack = release_path.read_text(errors="replace").splitlines()[0].strip()
    if not board:
        return None
    cc = _cc_for_jetson_board(board)
    if cc is None:
        warnings.append(f"gpu.jetson: unknown board {board!r}; compute capability not recorded")
        return None
    return JetsonInfo(
        board=board,
        jetpack_version=jetpack,
        tops=_tops_for_jetson_board(board),
        cuda_compute_capability=cc,
        cuda_toolkit_version=_probe_cuda_toolkit_version(),
        tensorrt_version=_probe_tensorrt_version(),
        supported_dtypes=_dtypes_for(cc),
    )


# ── Apple Silicon via system_profiler ────────────────────────────────────────


def _probe_apple_silicon(warnings: list[str]) -> AppleSiliconInfo | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(raw)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        warnings.append(f"gpu.apple: system_profiler failed: {exc!r}")
        return None
    displays = data.get("SPDisplaysDataType") or []
    if not displays:
        return None
    dev = displays[0]
    chip = str(dev.get("sppci_model", platform.processor() or "Apple Silicon"))
    try:
        gpu_cores = int(dev.get("sppci_cores") or 0)
    except (TypeError, ValueError):
        gpu_cores = 0
    return AppleSiliconInfo(
        chip=chip,
        gpu_cores=gpu_cores,
        unified_mem_gb=0.0,
        supported_dtypes=[
            QuantizationDtype.FP32,
            QuantizationDtype.FP16,
            QuantizationDtype.INT8,
        ],
    )


# ── Helpers — CUDA toolkit / TensorRT version ────────────────────────────────


def _probe_cuda_toolkit_version() -> str | None:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    try:
        raw = subprocess.check_output(
            [nvcc, "--version"], text=True, timeout=2, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    for line in raw.splitlines():
        if "release" in line:
            # "Cuda compilation tools, release 12.4, V12.4.99"
            parts = line.split("release")
            if len(parts) > 1:
                return parts[1].split(",")[0].strip()
    return None


def _probe_tensorrt_version() -> str | None:
    try:
        import tensorrt  # noqa: PLC0415  # reason: optional extra
    except ImportError:
        return None
    return str(getattr(tensorrt, "__version__", "")) or None


# ── Umbrella ─────────────────────────────────────────────────────────────────


def probe_gpus(*, warnings: list[str] | None = None) -> GpuProbeResult:
    """Discover every GPU / SoC accelerator on the host.

    Args:
        warnings: Optional list to append non-fatal probe issues to.

    Returns:
        :class:`GpuProbeResult` with NVIDIA discrete cards, an optional
        Jetson record, an optional Apple Silicon record, and the backend
        that produced the NVIDIA list.
    """
    sink: list[str] = warnings if warnings is not None else []

    nvidia = _probe_nvidia_pynvml(sink)
    backend = "nvml" if nvidia else "none"
    if not nvidia:
        nvidia = _probe_nvidia_smi(sink)
        if nvidia:
            backend = "nvidia-smi"

    jetson = _probe_jetson(sink)
    if jetson is not None and backend == "none":
        backend = "jtop" if shutil.which("jtop") else "tegra-release"

    apple = _probe_apple_silicon(sink)
    if apple is not None and backend == "none":
        backend = "system_profiler"

    return GpuProbeResult(
        nvidia=nvidia,
        jetson=jetson,
        apple_silicon=apple,
        backend=backend,
    )
