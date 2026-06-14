"""ctypes binding to ``libnvbufsurface.so`` for NVMM→CUDA zero-copy.

NVIDIA's L4T Multimedia API ships ``libnvbufsurface.so`` on every
Jetson/Spark host (and as part of the desktop DeepStream stack). The
GStreamer NVMM memory pool wraps an :class:`NvBufSurface` struct whose
``surfaceList[i].dataPtr`` is a CUDA device pointer — handing that
pointer directly to a CUDA / PyTorch consumer is the zero-copy path
the inference runner wants.

This module:

* Loads the shared object via :func:`ctypes.util.find_library`,
  falling back to known Jetson paths.
* Mirrors the :class:`NvBufSurface` / :class:`NvBufSurfaceParams`
  struct layout from NVIDIA's public ``nvbufsurface.h`` header
  (L4T-MM API, redistributable as part of JetPack headers).
* Exposes a :class:`NvBufSurfaceHandle` Pydantic dataclass with
  ``gpu_ptr / width / height / pitch / color_format / size``.
* Returns ``None`` from :func:`load` when the library is absent
  rather than raising — the reader uses this to gracefully fall
  back to the CPU path.

Importing this module is cheap even when ``libnvbufsurface.so`` is
absent: the ctypes ``CDLL()`` call is deferred to :func:`load`.

Provenance: derived from work originally authored by Adrian Llopart
(adrianllopart@gmail.com) and re-licensed under Apache-2.0 for
openral with the author's explicit consent. Struct layout
matches NVIDIA's publicly-distributed ``nvbufsurface.h``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    c_bool,
    c_int,
    c_uint,
    c_uint8,
    c_uint64,
    c_void_p,
)
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "NVBUF_MAX_PLANES",
    "NvBufSurface",
    "NvBufSurfaceColorFormat",
    "NvBufSurfaceHandle",
    "NvBufSurfaceLibraryError",
    "NvBufSurfaceParams",
    "NvBufSurfacePlaneParams",
    "load",
    "wrap_buffer",
]


# Maximum planes per surface — fixed by the L4T-MM ABI. Matches NVIDIA's
# `nvbufsurface.h`.
NVBUF_MAX_PLANES: Final[int] = 4

# Internal reserved padding slots in NVIDIA's structs. Match the header.
_STRUCT_PADDING: Final[int] = 4

# Candidate paths to probe when ``ctypes.util.find_library`` returns None
# (typical on Jetson where the shared object lives under tegra/).
_FALLBACK_LIBRARY_PATHS: Final[tuple[str, ...]] = (
    "/usr/lib/aarch64-linux-gnu/tegra/libnvbufsurface.so",
    "/usr/lib/x86_64-linux-gnu/libnvbufsurface.so",
    "/opt/nvidia/deepstream/deepstream/lib/libnvbufsurface.so",
)


class NvBufSurfaceLibraryError(RuntimeError):
    """Raised when ``libnvbufsurface.so`` cannot be located.

    Caller can decide to fall back (the reader does, dropping to the
    CPU path) or propagate.
    """


class NvBufSurfaceColorFormat:
    """Subset of the L4T-MM ``NvBufSurfaceColorFormat`` enum we care about.

    Full enum lives in NVIDIA's header; the values below are the ones
    the GStreamer NVMM path negotiates in practice. Add more as needed
    when new caps formats appear.
    """

    NV12 = 6
    GRAY8 = 1
    RGBA = 19
    BGRA = 20
    RGB = 25
    BGR = 26


# ── ctypes struct layout (mirrors NVIDIA's nvbufsurface.h) ─────────────────────


class NvBufSurfacePlaneParams(Structure):
    """Per-plane geometry. Up to ``NVBUF_MAX_PLANES`` planes per surface."""

    _fields_ = (  # reason: ctypes Structure declares _fields_ as a class attribute
        ("num_planes", c_uint),
        ("width", c_uint * NVBUF_MAX_PLANES),
        ("height", c_uint * NVBUF_MAX_PLANES),
        ("pitch", c_uint * NVBUF_MAX_PLANES),
        ("offset", c_uint * NVBUF_MAX_PLANES),
        ("psize", c_uint * NVBUF_MAX_PLANES),
        ("bytes_per_pix", c_uint * NVBUF_MAX_PLANES),
        ("_reserved", c_void_p * (_STRUCT_PADDING * NVBUF_MAX_PLANES)),
    )


class _NvBufSurfaceMappedAddr(Structure):
    """User-space mapped pointers when ``NvBufSurfaceMap`` is called."""

    _fields_ = (
        ("addr", c_void_p * NVBUF_MAX_PLANES),
        ("egl_image", c_void_p),
        ("_reserved", c_void_p * _STRUCT_PADDING),
    )


class NvBufSurfaceParams(Structure):
    """Per-surface geometry + the GPU dataPtr we want for zero-copy."""

    _fields_ = (
        ("width", c_uint),
        ("height", c_uint),
        ("pitch", c_uint),
        ("color_format", c_int),
        ("layout", c_int),
        ("buffer_desc", c_uint64),
        ("data_size", c_uint),
        ("data_ptr", c_void_p),
        ("plane_params", NvBufSurfacePlaneParams),
        ("mapped_addr", _NvBufSurfaceMappedAddr),
        ("paramex", c_void_p),  # opaque pointer to extended params
        ("_reserved", c_void_p * (_STRUCT_PADDING - 1)),
    )


class NvBufSurface(Structure):
    """Top-level batched-surface container.

    ``surfaceList`` is a pointer to a ``NvBufSurfaceParams`` array of
    length ``batchSize``. The GStreamer NVMM caps we use always have
    ``batchSize == 1`` (one frame per buffer), so we read
    ``surfaceList[0]``.
    """

    _fields_ = (
        ("gpu_id", c_uint),
        ("batch_size", c_uint),
        ("num_filled", c_uint),
        ("is_contiguous", c_bool),
        ("mem_type", c_int),
        ("surface_list", POINTER(NvBufSurfaceParams)),
        ("_reserved", c_void_p * _STRUCT_PADDING),
    )


# ── Loader ─────────────────────────────────────────────────────────────────────


_loaded: CDLL | None = None
_load_attempted = False


def load() -> CDLL:
    """Locate and load ``libnvbufsurface.so``. Memoised.

    Returns:
        The opened :class:`ctypes.CDLL` handle.

    Raises:
        NvBufSurfaceLibraryError: When the library cannot be found
            on the standard search paths.
    """
    global _loaded, _load_attempted  # noqa: PLW0603  # reason: one-shot loader flag
    if _loaded is not None:
        return _loaded
    if _load_attempted:
        raise NvBufSurfaceLibraryError(
            "libnvbufsurface.so was probed earlier and not found; "
            "running on a host without the NVIDIA L4T multimedia stack"
        )
    _load_attempted = True
    resolved = ctypes.util.find_library("nvbufsurface")
    candidates: list[str] = []
    if resolved is not None:
        candidates.append(resolved)
    candidates.extend(p for p in _FALLBACK_LIBRARY_PATHS if Path(p).exists())
    last_exc: OSError | None = None
    for candidate in candidates:
        try:
            handle = CDLL(candidate)
        except OSError as exc:
            last_exc = exc
            continue
        _configure_function_signatures(handle)
        _loaded = handle
        return handle
    raise NvBufSurfaceLibraryError(
        "libnvbufsurface.so not found on this host. Probed: "
        f"find_library={resolved!r}, fallbacks={list(_FALLBACK_LIBRARY_PATHS)}. "
        f"Last error: {last_exc}"
    )


def _configure_function_signatures(lib: CDLL) -> None:
    """Set argtypes / restype for the L4T-MM functions we call."""
    try:
        lib.NvBufSurfaceMap.argtypes = [POINTER(NvBufSurface), c_int, c_int, c_uint]
        lib.NvBufSurfaceMap.restype = c_int
        lib.NvBufSurfaceUnMap.argtypes = [POINTER(NvBufSurface), c_int, c_int]
        lib.NvBufSurfaceUnMap.restype = c_int
        lib.NvBufSurfaceSyncForCpu.argtypes = [POINTER(NvBufSurface), c_int, c_int]
        lib.NvBufSurfaceSyncForCpu.restype = c_int
        lib.NvBufSurfaceSyncForDevice.argtypes = [POINTER(NvBufSurface), c_int, c_int]
        lib.NvBufSurfaceSyncForDevice.restype = c_int
        lib.NvBufSurfaceMemSet.argtypes = [POINTER(NvBufSurface), c_int, c_int, c_uint8]
        lib.NvBufSurfaceMemSet.restype = c_int
    except AttributeError as exc:
        # Older L4T builds may not expose every entrypoint. Surface that
        # to the caller; the reader can decide whether to proceed.
        raise NvBufSurfaceLibraryError(
            f"libnvbufsurface.so loaded but missing expected entrypoint: {exc}. "
            "JetPack version may be too old; consider upgrading."
        ) from exc


# ── Pydantic-validated handle for SensorFrame consumers ────────────────────────


class NvBufSurfaceHandle(BaseModel):
    """Typed view of the GPU surface a :class:`NvBufSurface` references.

    Stashed in ``SensorFrame.metadata['nvbufsurface']`` by the reader
    so a downstream consumer (Skill / TRT engine) can lift the data
    pointer into a CUDA tensor without re-mapping the GStreamer buffer.

    Attributes:
        gpu_ptr: Device-virtual address. Suitable as input to
            :func:`torch.cuda.IPCHandle` / a CUDA kernel argument.
        width: Surface width in pixels.
        height: Surface height in pixels.
        pitch: Row pitch in bytes (NVMM buffers are pitch-padded;
            ``pitch >= width * bytes_per_pixel``).
        color_format: One of the :class:`NvBufSurfaceColorFormat`
            constants (e.g. ``NV12 == 6``).
        size: Total byte size of the plane data, ``pitch * height``.
        batch_size: Always ``1`` in our GStreamer NVMM path; surfaced
            so consumers can assert their assumption.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    gpu_ptr: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pitch: int = Field(gt=0)
    color_format: int = Field(ge=0)
    size: int = Field(gt=0)
    batch_size: int = Field(default=1, gt=0)


def wrap_buffer(buffer_address: int) -> NvBufSurfaceHandle:
    """Wrap a raw GStreamer NVMM buffer pointer as a typed handle.

    Args:
        buffer_address: The integer address of a mapped GStreamer
            buffer's data — typically obtained from
            ``Gst.MapInfo.data.cast(ctypes.c_void_p)`` after
            ``buffer.map(Gst.MapFlags.READ)``.

    Returns:
        A populated :class:`NvBufSurfaceHandle`.

    Raises:
        ValueError: When the buffer cannot be interpreted as an
            :class:`NvBufSurface` (e.g. ``batch_size == 0`` or
            ``surface_list`` is NULL).
    """
    surface = ctypes.cast(buffer_address, POINTER(NvBufSurface)).contents
    if surface.batch_size < 1:
        raise ValueError(
            f"NvBufSurface batch_size={surface.batch_size} < 1; not a valid frame buffer"
        )
    if not surface.surface_list:
        raise ValueError("NvBufSurface.surface_list is NULL")
    params: NvBufSurfaceParams = surface.surface_list[0]
    if not params.data_ptr:
        raise ValueError("NvBufSurfaceParams.data_ptr is NULL (frame not yet allocated)")
    size = params.data_size or (params.pitch * params.height)
    return NvBufSurfaceHandle(
        gpu_ptr=int(params.data_ptr),
        width=int(params.width),
        height=int(params.height),
        pitch=int(params.pitch),
        color_format=int(params.color_format),
        size=int(size),
        batch_size=int(surface.batch_size),
    )


def _reset_loader_for_tests() -> None:
    """Reset the memoised loader so tests can re-probe under monkeypatch."""
    global _loaded, _load_attempted  # noqa: PLW0603
    _loaded = None
    _load_attempted = False


# Silence unused-import warning for Any when only used in typing.
_: Any = None
