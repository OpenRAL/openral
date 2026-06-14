"""GStreamer-backed :class:`SensorReader` for the inference runner.

This subpackage is split into modules that can be imported without
pulling in PyGObject:

* :mod:`pipeline` — pipeline-string builder + platform detect. Pure
  Python, no ``gi`` import. Safe to import on hosts without GStreamer.
* :mod:`reader` (commit #2) — the actual :class:`SensorReader` impl.
  Imports ``gi.repository.Gst`` at module load and therefore requires
  the ``gstreamer`` optional-extra (``pip install openral-runner[gstreamer]``).
* :mod:`nvbufsurface` (commit #3) — ctypes wrapper around
  ``libnvbufsurface.so`` for NVMM→CUDA zero-copy on Jetson / Spark.
* :mod:`cuda_context` (commit #3) — shared PyCUDA context singleton.
* :mod:`ros_tee` (commit #4) — optional ``rclpy.Image`` publisher fed
  from a second appsink.
* :mod:`perception_tee` (ADR-0018 F6) — optional ``PromptStamped``
  publisher fed from a third appsink. Runs per-frame event detectors
  (motion, scene change, …) and publishes on
  ``/openral/perception/<kind>``.

See ``docs/adr/0010-inference-runner.md`` (Amendment 2026-05-12,
"Backend evaluation: lean GStreamer vs Holoscan vs DeepStream") and
``docs/adr/0011-nvmm-handoff.md`` for the design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openral_runner.backends.gstreamer.pipeline import (
    LEAKY_BRANCH_QUEUE,
    TEE_NAME,
    PipelineSpec,
    Platform,
    Source,
    build_pipeline_string,
    detect_platform,
    ensure_appsink_name,
    inspect_element_present,
    leaky_branch,
    nvmm_convert_element,
)

if TYPE_CHECKING:
    # Re-export for type checkers without forcing a runtime gi import.
    from openral_runner.backends.gstreamer.reader import GStreamerSensorReader
    from openral_runner.backends.gstreamer.tee_manager import BranchHandle, TeeManager

__all__ = [
    "LEAKY_BRANCH_QUEUE",
    "TEE_NAME",
    "BranchHandle",
    "GStreamerSensorReader",
    "PipelineSpec",
    "Platform",
    "Source",
    "TeeManager",
    "build_pipeline_string",
    "detect_platform",
    "ensure_appsink_name",
    "inspect_element_present",
    "leaky_branch",
    "nvmm_convert_element",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401  # reason: PEP 562 attribute hook
    """Lazy-import the gi-dependent symbols so the package is import-safe without gi."""
    # Lazy import keeps the package import-safe on hosts without PyGObject:
    # the reader / tee_manager modules import ``gi`` at module load.
    if name == "GStreamerSensorReader":
        from openral_runner.backends.gstreamer.reader import (  # noqa: PLC0415
            GStreamerSensorReader,
        )

        return GStreamerSensorReader
    if name in ("TeeManager", "BranchHandle"):
        from openral_runner.backends.gstreamer.tee_manager import (  # noqa: PLC0415
            BranchHandle,
            TeeManager,
        )

        return {"TeeManager": TeeManager, "BranchHandle": BranchHandle}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
