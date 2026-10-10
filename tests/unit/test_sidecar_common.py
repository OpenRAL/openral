"""Unit tests for the shared sidecar port-derivation helper.

``openral_sim._sidecar_common.sidecar_port_for_key`` replaces five
byte-for-byte copies of a SHA-256(-or-SHA-1)-digest-into-port-range helper
that used to live in ``backends/isaac_sim.py``, ``backends/robotwin.py``,
``backends/rlbench.py``, ``policies/lingbot_vla2.py``, and
``policies/rldx.py``. Each caller's port for a fixed identity key must stay
byte-identical after consolidation — a changed port would silently break a
sidecar an operator already has running. The expected values below were
computed from each module's original (pre-consolidation) implementation
before it was deleted.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from openral_sim._sidecar_common import sidecar_port_for_key


def test_isaac_sim_port_matches_original_implementation() -> None:
    from openral_sim.backends.isaac_sim import _scene_default_port

    assert _scene_default_port("test_task", "franka_panda", "default") == 26675


def test_robotwin_port_matches_original_implementation() -> None:
    from openral_sim.backends.robotwin import _scene_default_port

    assert _scene_default_port("test_task", "aloha") == 38498


def test_rlbench_port_matches_original_implementation() -> None:
    from openral_sim.backends.rlbench import _scene_default_port

    assert _scene_default_port("reach_target", 0) == 21361


def test_lingbot_vla2_port_matches_original_implementation() -> None:
    from openral_sim.policies.lingbot_vla2 import _policy_default_port

    assert _policy_default_port("lingbot/model-v2", "franka", "v2") == 34593
    assert _policy_default_port("lingbot/model-v1", "franka", "v1") == 37582


def test_rldx_port_matches_original_implementation() -> None:
    from openral_sim.policies.rldx import _derive_sidecar_port

    assert (
        _derive_sidecar_port(
            family="rldx",
            model="rlwrld/rldx-1",
            embodiment_tag="franka",
            quantization="nf4",
            layout="default",
        )
        == 22613
    )


def test_shared_helper_is_deterministic_and_in_range() -> None:
    a = sidecar_port_for_key("some-identity", port_min=20_000, port_max=40_000)
    b = sidecar_port_for_key("some-identity", port_min=20_000, port_max=40_000)
    assert a == b
    assert 20_000 <= a < 40_000


def test_shared_helper_algorithm_changes_the_port() -> None:
    """``rldx`` relies on the ``algorithm`` override to keep its SHA-1 ports."""
    sha256_port = sidecar_port_for_key("x", algorithm="sha256")
    sha1_port = sidecar_port_for_key("x", algorithm="sha1")
    assert sha256_port != sha1_port


# ``default_expandable_segments``: the one allocator default, never on Jetson,
# where torch's expandable path crashes on the iGPU's NVML fabric query.


# The real probe, pointed at a captured AGX Orin L4T release file or at a path
# that does not exist (``openral_core.gpu.is_tegra_host`` reads the constant at
# call time for exactly this).
_ORIN_RELEASE = Path(__file__).parent / "fixtures" / "jetson" / "orin_agx" / "nv_tegra_release"


@pytest.fixture
def _not_jetson(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import openral_core.gpu

    monkeypatch.setattr(openral_core.gpu, "TEGRA_RELEASE_PATH", tmp_path / "nv_tegra_release")


@pytest.fixture
def _jetson(monkeypatch: pytest.MonkeyPatch) -> None:
    import openral_core.gpu

    assert _ORIN_RELEASE.is_file(), _ORIN_RELEASE
    monkeypatch.setattr(openral_core.gpu, "TEGRA_RELEASE_PATH", _ORIN_RELEASE)


@pytest.mark.usefixtures("_not_jetson")
def test_default_expandable_segments_sets_this_process(monkeypatch: pytest.MonkeyPatch) -> None:
    from openral_sim._sidecar_common import default_expandable_segments, installed_alloc_conf_var

    var = installed_alloc_conf_var()
    monkeypatch.delenv(var, raising=False)
    default_expandable_segments()
    assert os.environ[var] == "expandable_segments:True"


@pytest.mark.usefixtures("_jetson")
def test_default_expandable_segments_leaves_jetson_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from openral_sim._sidecar_common import default_expandable_segments, installed_alloc_conf_var

    var = installed_alloc_conf_var()
    monkeypatch.delenv(var, raising=False)
    child: dict[str, str] = {}
    default_expandable_segments()
    default_expandable_segments(child, var="PYTORCH_ALLOC_CONF")
    assert var not in os.environ
    assert child == {}


@pytest.mark.usefixtures("_not_jetson")
def test_default_expandable_segments_targets_a_child_env_and_var() -> None:
    from openral_sim._sidecar_common import default_expandable_segments

    child: dict[str, str] = {}
    default_expandable_segments(child, var="PYTORCH_CUDA_ALLOC_CONF")
    assert child == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


@pytest.mark.usefixtures("_not_jetson")
def test_default_expandable_segments_keeps_an_operator_value() -> None:
    from openral_sim._sidecar_common import default_expandable_segments

    child = {"PYTORCH_ALLOC_CONF": "max_split_size_mb:128"}
    default_expandable_segments(child, var="PYTORCH_ALLOC_CONF")
    assert child == {"PYTORCH_ALLOC_CONF": "max_split_size_mb:128"}
