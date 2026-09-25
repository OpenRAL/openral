"""OpenRAL rSkill — lifecycle base class, runtime protocol, and rSkill loader.

Public surface
--------------
- ``rSkillBase``: Abstract base class with the ROS 2 lifecycle state machine.
- ``Runtime``: Structural protocol for inference backends.
- ``NullRuntime``: No-op runtime for tests and development.
- ``rSkill``: HF Hub rSkill loader (manifest + weights + license guard).
- ``InstalledRSkillEntry``: Local registry entry schema.
- ``search_hub_rskills``: Free-text + facet search over an HF Hub org's
  rSkills (``HubRSkillHit``, ``HubRSkillSearchResult``).
- ``resolve_runtime_backend``: Name -> ``Runtime`` class, built-in or via the
  ``openral.runtime_backends`` entry-point group (the OpenRAL Pro extraction seam).
- ``maybe_attach_pro_hooks``: Generic OpenRAL Pro policy-attach-hook lookup
  via the ``openral.policy_attach_hooks`` entry-point group.
- ``hf_download_cached_first``: Cache-first HF Hub file resolution shared by
  the VLA adapters and ``openral_sim``'s quantization helpers.
- ``local_snapshot_dir``: The resolved checkpoint as a local directory — the
  directory itself for an installed rSkill snapshot, a Hub snapshot otherwise.
- ``gpu_allocated_mb``: Current CUDA allocator usage in MB, the single GPU-memory
  probe shared by the phase-timer heartbeats and the rskill runner node's
  eviction accounting.
- ``find_repo_root_from``: Locate the OpenRAL repo root by walking up from
  a start path.
- ``validate_skill_ref``: Validate a bare rSkill reference string.

Heavy-dependency backends (``PyTorchRuntime``, ``ONNXRuntime``) are **not**
imported here. Import them explicitly when their dependencies are installed:

    from openral_rskill.runtime_pytorch import PyTorchRuntime
    from openral_rskill.runtime_onnx import ONNXRuntime

A TensorRT ``Runtime`` is not shipped in this package (the
TensorRT/NVMM fast path is a private OpenRAL Pro plugin); resolve it via
``resolve_runtime_backend("tensorrt")`` when ``openral-pro-trt`` is installed.
"""

from importlib.metadata import version as _pkg_version

from openral_rskill._diagnostics import gpu_allocated_mb
from openral_rskill._vla_core import hf_download_cached_first, local_snapshot_dir
from openral_rskill.backend_registry import maybe_attach_pro_hooks, resolve_runtime_backend
from openral_rskill.base import rSkillBase
from openral_rskill.hub_search import HubRSkillHit, HubRSkillSearchResult, search_hub_rskills
from openral_rskill.loader import (
    DEFAULT_REGISTRY_PATH,
    InstalledRSkillEntry,
    discover_intree_rskills,
    find_repo_root_from,
    intree_embodiment_tags,
    known_benchmark_ids,
    rSkill,
    validate_skill_ref,
)
from openral_rskill.runtime import NullRuntime, Runtime

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "HubRSkillHit",
    "HubRSkillSearchResult",
    "InstalledRSkillEntry",
    "NullRuntime",
    "Runtime",
    "discover_intree_rskills",
    "find_repo_root_from",
    "gpu_allocated_mb",
    "hf_download_cached_first",
    "intree_embodiment_tags",
    "known_benchmark_ids",
    "local_snapshot_dir",
    "maybe_attach_pro_hooks",
    "rSkill",
    "rSkillBase",
    "resolve_runtime_backend",
    "search_hub_rskills",
    "validate_skill_ref",
]
__version__ = _pkg_version("openral-rskill")
