"""Unit tests for the NVIDIA GR00T-N1.7 rSkill contract (in-process backend).

GR00T-N1.7 now loads **in-process** via lerobot 0.6.0's native ``GrootPolicy``
(``openral_sim.policies.gr00t``), NF4-quantized like pi05 — there is no longer a
Python-3.10 ZMQ sidecar on the default path. The live in-process backend (load,
8 GiB NF4 fit, real action chunk) is exercised by the sim-tier
``tests/sim/test_franka_groot_libero.py`` on a GPU host.

This unit test pins only the GPU-free contract: the shipped manifest validates
and carries the right family/license posture. (The legacy sidecar behaviour
tests were removed together with ``tools/gr00t_sidecar.py`` — GR00T-N1.7 no
longer runs out-of-process; the in-process backend is gated by its 5/5
LIBERO-spatial rollout instead of a numerical parity harness.)
"""

from __future__ import annotations

from pathlib import Path

from openral_rskill.loader import load_rskill_manifest

_REPO_ROOT = Path(__file__).parent.parent.parent
_GR00T_RSKILL = _REPO_ROOT / "rskills" / "gr00t-n17-libero"


def test_gr00t_rskill_manifest_loads() -> None:
    """The shipped gr00t-n17-libero manifest passes the real validator.

    Pins the GR00T family contract: family string, the commercial Open Model
    License posture (the key distinction from N1/N1.5/N1.6), the OpenRAL-hosted
    repackaged weights repo, and the NVIDIA upstream provenance.
    """
    manifest = load_rskill_manifest(str(_GR00T_RSKILL))
    assert manifest.model_family == "gr00t"
    assert manifest.license == "nvidia_open_model"
    assert manifest.is_commercial_use_allowed is True
    # Weights are the root-level OpenRAL repackage of GR00T-N1.7-LIBERO's
    # libero_spatial/ inference checkpoint; nvidia stays as upstream provenance.
    assert str(manifest.weights_uri) == "hf://OpenRAL/rskill-gr00t_n17-franka_panda-libero_spatial-bf16"
    assert str(manifest.source_repo).startswith("hf://nvidia/")
