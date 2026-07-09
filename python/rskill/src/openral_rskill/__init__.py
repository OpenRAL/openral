"""OpenRAL rSkill — lifecycle base class, runtime protocol, and rSkill loader.

Public surface
--------------
- ``rSkillBase``: Abstract base class with the ROS 2 lifecycle state machine.
- ``Runtime``: Structural protocol for inference backends.
- ``NullRuntime``: No-op runtime for tests and development.
- ``QUANT_PRESETS``: Named ``QuantizationConfig`` presets.
- ``auto_select_quant``: Device-aware quantization preset selector.
- ``EngineCache``: Filesystem cache for compiled engine files.
- ``DEFAULT_CACHE_DIR``: Default cache directory path.
- ``rSkill``: HF Hub rSkill loader (manifest + weights + license guard).
- ``InstalledRSkillEntry``: Local registry entry schema.
- ``resolve_runtime_backend``: Name -> ``Runtime`` class, built-in or via the
  ``openral.runtime_backends`` entry-point group (the OpenRAL Pro extraction seam).
- ``maybe_attach_pro_hooks``: Generic OpenRAL Pro policy-attach-hook lookup
  via the ``openral.policy_attach_hooks`` entry-point group.

Heavy-dependency adapters (``SmolVLAAdapter``, ``SO100SmolVLASkill``) and
backends (``PyTorchRuntime``, ``ONNXRuntime``) are **not** imported here.
Import them explicitly when their dependencies are installed:

    from openral_rskill.smolvla import SmolVLAAdapter, SO100SmolVLASkill
    from openral_rskill.runtime_pytorch import PyTorchRuntime
    from openral_rskill.runtime_onnx import ONNXRuntime

A TensorRT ``Runtime`` is not shipped in this package (the
TensorRT/NVMM fast path is a private OpenRAL Pro plugin); resolve it via
``resolve_runtime_backend("tensorrt")`` when ``openral-pro-trt`` is installed.
"""

from openral_rskill.backend_registry import maybe_attach_pro_hooks, resolve_runtime_backend
from openral_rskill.base import rSkillBase
from openral_rskill.engine_cache import DEFAULT_CACHE_DIR, EngineCache
from openral_rskill.loader import (
    DEFAULT_REGISTRY_PATH,
    InstalledRSkillEntry,
    discover_intree_rskills,
    rSkill,
)
from openral_rskill.quantization import QUANT_PRESETS, auto_select_quant
from openral_rskill.runtime import NullRuntime, Runtime

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_REGISTRY_PATH",
    "QUANT_PRESETS",
    "EngineCache",
    "InstalledRSkillEntry",
    "NullRuntime",
    "Runtime",
    "auto_select_quant",
    "discover_intree_rskills",
    "maybe_attach_pro_hooks",
    "rSkill",
    "rSkillBase",
    "resolve_runtime_backend",
]
__version__ = "0.1.0"
