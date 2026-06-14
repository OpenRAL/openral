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

Heavy-dependency adapters (``SmolVLAAdapter``, ``SO100SmolVLASkill``) and
backends (``PyTorchRuntime``, ``ONNXRuntime``, ``TensorRTRuntime``) are **not** imported here.
Import them explicitly when their dependencies are installed:

    from openral_rskill.smolvla import SmolVLAAdapter, SO100SmolVLASkill
    from openral_rskill.runtime_pytorch import PyTorchRuntime
    from openral_rskill.runtime_onnx import ONNXRuntime
    from openral_rskill.runtime_tensorrt import TensorRTRuntime
"""

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
    "rSkill",
    "rSkillBase",
]
__version__ = "0.1.0"
