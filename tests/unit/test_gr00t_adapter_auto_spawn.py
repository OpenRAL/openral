"""Unit tests for the NVIDIA GR00T auto-managed sidecar adapter (ADR-0046).

Validates the openral-side of the GR00T integration without depending on the
upstream Isaac-GR00T package or starting a real GPU model. Per CLAUDE.md §1.11
there are no mocks — every test exercises **real** code paths:

* the real :class:`RSkillManifest` loaded from ``rskills/gr00t-n17-libero``,
* the real ``@POLICIES.register("gr00t")`` factory,
* the real ``_Gr00tFamilySidecarAdapter`` reused with ``family="gr00t"`` (RLDX-1 is a
  GR00T-N1.5 finetune sharing the LIBERO wire contract).

What we deliberately do NOT exercise here (covered by the PR2 sim-tier eval on a
Python-3.10 GPU host): the live ZMQ round-trip against a booted GR00T server and
actual NF4 inference. These tests pin the adapter contract regardless.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import SimEnvironment, VLASpec
from openral_core.exceptions import ROSConfigError
from openral_rskill.loader import load_rskill_manifest

# Skip on hosts without the `gr00t` opt-in group (pyzmq + msgpack).
pytest.importorskip("zmq", reason="GR00T adapter tests require the `gr00t` group (pyzmq + msgpack)")
pytest.importorskip("msgpack", reason="GR00T adapter tests require the `gr00t` group")

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
    assert str(manifest.weights_uri) == "hf://OpenRAL/rskill-gr00t-n17-libero"
    assert str(manifest.source_repo).startswith("hf://nvidia/")


@pytest.fixture
def gr00t_env_cfg() -> SimEnvironment:
    """Compose a SimEnvironment around the gr00t-n17-libero rSkill manifest.

    Drives the same code path as ``openral sim run --config X --rskill Y`` on
    the LIBERO Franka scene; no test doubles.
    """
    from tests.sim.conftest import compose_sim_env

    return compose_sim_env(
        _REPO_ROOT / "scenes" / "sim" / "libero_spatial.yaml",
        "rskills/gr00t-n17-libero",
        n_episodes=1,
        max_steps=1,
    )


def test_build_gr00t_routes_libero_layout_and_fails_without_sidecar(
    gr00t_env_cfg: SimEnvironment,
) -> None:
    """The factory builds a libero-layout adapter; no server → ROSConfigError.

    auto_spawn is disabled so the unit test never forks tools/gr00t_sidecar.py
    (which would clone Isaac-GR00T + build a Python 3.10 venv — minutes of CI
    time and a network dependency).
    """
    from openral_sim.policies import gr00t as gr00t_mod  # noqa: F401  — register side effect
    from openral_sim.registry import POLICIES

    free_port = _allocate_free_port()
    gr00t_env_cfg.vla = VLASpec(
        id="gr00t",
        weights_uri=gr00t_env_cfg.vla.weights_uri,
        extra={"port": free_port, "auto_spawn": False, "timeout_ms": 100},
    )
    with pytest.raises(ROSConfigError, match="did not answer ping"):
        POLICIES.get("gr00t")(gr00t_env_cfg)


def test_gr00t_auto_spawn_disabled_via_env(gr00t_env_cfg: SimEnvironment) -> None:
    """``OPENRAL_GR00T_AUTO_SPAWN=0`` overrides ``vla.extra.auto_spawn=true``."""
    from openral_sim.registry import POLICIES

    free_port = _allocate_free_port()
    gr00t_env_cfg.vla = VLASpec(
        id="gr00t",
        weights_uri=gr00t_env_cfg.vla.weights_uri,
        extra={"port": free_port, "auto_spawn": True, "timeout_ms": 100},
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENRAL_GR00T_AUTO_SPAWN", "0")
        with pytest.raises(ROSConfigError, match="auto_spawn is disabled"):
            POLICIES.get("gr00t")(gr00t_env_cfg)


def test_locate_gr00t_sidecar_script_finds_real_tool() -> None:
    """The family-parameterized locator finds ``tools/gr00t_sidecar.py``.

    Exercises the real upwards-walk in ``_Gr00tFamilySidecarAdapter.
    _locate_sidecar_script`` with ``family="gr00t"``; no mocks, no env override.
    """
    from openral_sim.policies.rldx import _Gr00tFamilySidecarAdapter

    adapter = _Gr00tFamilySidecarAdapter.__new__(_Gr00tFamilySidecarAdapter)
    adapter.family = "gr00t"
    script = adapter._locate_sidecar_script()
    assert script.is_file()
    assert script.name == "gr00t_sidecar.py"
    assert script.resolve().is_relative_to(_REPO_ROOT.resolve())


def _allocate_free_port() -> int:
    """Ask the OS for an unused loopback port, close immediately."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
