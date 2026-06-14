"""Unit tests for the NVMM ctypes binding + shared CUDA context.

No mocks. Runtime tests that need ``libnvbufsurface.so`` skip cleanly
when the library is absent (i.e. on every non-Jetson host); the
remaining tests cover the Pydantic validation, the loader's failure
mode, and the cuda_context state probe. Real NVMM round-trips are
covered by ``tests/hil/test_gstreamer_spark_csi.py`` on Spark.
"""

from __future__ import annotations

import pytest
from openral_runner.backends.gstreamer import nvbufsurface
from openral_runner.backends.gstreamer.cuda_context import (
    cuda_context_state,
)
from openral_runner.backends.gstreamer.nvbufsurface import (
    NvBufSurfaceColorFormat,
    NvBufSurfaceHandle,
    NvBufSurfaceLibraryError,
)

# ── NvBufSurfaceHandle Pydantic validation ───────────────────────────────────


def test_nvbufsurface_handle_accepts_valid_fields() -> None:
    """A well-formed handle constructs cleanly and is frozen."""
    handle = NvBufSurfaceHandle(
        gpu_ptr=0x7F00_1000_0000,
        width=1280,
        height=720,
        pitch=1280,
        color_format=NvBufSurfaceColorFormat.NV12,
        size=1280 * 720 * 3 // 2,
    )
    assert handle.gpu_ptr == 0x7F00_1000_0000
    assert handle.batch_size == 1
    # Frozen — mutation must fail.
    with pytest.raises(ValueError):
        handle.gpu_ptr = 0  # type: ignore[misc]


def test_nvbufsurface_handle_rejects_null_gpu_ptr() -> None:
    """gpu_ptr == 0 is rejected (a NULL pointer is never a valid GPU surface)."""
    with pytest.raises(ValueError):
        NvBufSurfaceHandle(
            gpu_ptr=0,
            width=1,
            height=1,
            pitch=1,
            color_format=0,
            size=1,
        )


def test_nvbufsurface_handle_rejects_zero_dimensions() -> None:
    """Width / height / pitch / size must be strictly positive."""
    for bad_field in ("width", "height", "pitch", "size"):
        kwargs: dict[str, int] = {
            "gpu_ptr": 0x1000,
            "width": 1,
            "height": 1,
            "pitch": 1,
            "color_format": NvBufSurfaceColorFormat.NV12,
            "size": 1,
        }
        kwargs[bad_field] = 0
        with pytest.raises(ValueError):
            NvBufSurfaceHandle(**kwargs)  # type: ignore[arg-type]


# ── Loader ────────────────────────────────────────────────────────────────────


def test_load_skips_when_library_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the .so cannot be located, ``load`` raises NvBufSurfaceLibraryError.

    Uses monkeypatching to simulate a host without libnvbufsurface; the
    loader's normal probe (``ctypes.util.find_library`` + fallback paths)
    is bypassed.
    """
    monkeypatch.setattr(nvbufsurface.ctypes.util, "find_library", lambda _name: None)
    monkeypatch.setattr(nvbufsurface, "_FALLBACK_LIBRARY_PATHS", ())
    nvbufsurface._reset_loader_for_tests()
    with pytest.raises(NvBufSurfaceLibraryError, match="not found on this host"):
        nvbufsurface.load()


def test_load_memoises_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second call to ``load`` after a failed probe re-raises without re-probing."""
    monkeypatch.setattr(nvbufsurface.ctypes.util, "find_library", lambda _name: None)
    monkeypatch.setattr(nvbufsurface, "_FALLBACK_LIBRARY_PATHS", ())
    nvbufsurface._reset_loader_for_tests()
    with pytest.raises(NvBufSurfaceLibraryError):
        nvbufsurface.load()
    # Memoised failure — error message reflects the second-call path.
    with pytest.raises(NvBufSurfaceLibraryError, match="probed earlier"):
        nvbufsurface.load()


def test_load_returns_cdll_on_jetson() -> None:
    """On a Jetson / Spark host the loader returns a usable CDLL handle.

    Skipped here on every CI runner (we don't ship a Jetson). The live
    runtime check lives in ``tests/hil/test_gstreamer_spark_csi.py``
    behind a ``[self-hosted, lab-spark]`` runner label.
    """
    if not _is_libnvbufsurface_present():
        pytest.skip("libnvbufsurface.so absent — deferred to Spark HIL")
    handle = nvbufsurface.load()
    assert handle is not None


# ── cuda_context state probe ──────────────────────────────────────────────────


def test_cuda_context_state_pre_init_returns_uninitialised() -> None:
    """Before any get_shared_cuda_context() call, state is uninitialised."""
    state = cuda_context_state()
    assert isinstance(state, dict)
    assert "initialised" in state
    assert "device_index" in state
    assert "device_name" in state
    # device_name is None pre-init; initialised may be False (default) or True
    # if a prior test in the same process already populated it.
    if not state["initialised"]:
        assert state["device_name"] is None


def _is_libnvbufsurface_present() -> bool:
    """Helper for runtime-gated test skips; True only on Jetson / Spark."""
    import ctypes.util

    if ctypes.util.find_library("nvbufsurface") is not None:
        return True
    from pathlib import Path

    return any(Path(path).exists() for path in nvbufsurface._FALLBACK_LIBRARY_PATHS)
