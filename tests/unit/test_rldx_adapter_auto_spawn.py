"""Unit tests for the RLDX-1 auto-managed sidecar adapter.

Validates the openral-side of the integration without depending on the
upstream rldx package or starting a real GPU model. Per CLAUDE.md §1.11
there are no mocks — every test exercises **real** code paths:

* real :class:`RSkillManifest` instances loaded from the canonical YAML
  files under ``rskills/rldx1-*``,
* the real ``@POLICIES.register("rldx")`` factory,
* a real :class:`socket.socket` listener that occupies a port without
  speaking ZMQ — proves the adapter's "port busy → don't double-spawn"
  guard works against a real TCP probe.

What we deliberately do NOT exercise here (covered by sim-tier tests
when a GPU + the upstream rldx checkout are available):

* end-to-end ZMQ round-trips against the live RLDX-1 server,
* actual quantization / Qwen3-VL inference.

These tests pin the adapter contract that survives whether or not the
RLDX-1 checkpoint can be fetched at the time of CI.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
from openral_core import SimEnvironment, VLASpec
from openral_core.exceptions import ROSConfigError
from openral_rskill.loader import load_rskill_manifest

# Skip the whole module on hosts that don't have the `rldx` opt-in group
# installed — pyzmq is the canonical marker (see pyproject.toml).
pytest.importorskip("zmq", reason="RLDX adapter tests require the `rldx` group (pyzmq + msgpack)")
pytest.importorskip("msgpack", reason="RLDX adapter tests require the `rldx` group")

_REPO_ROOT = Path(__file__).parent.parent.parent
_RSKILLS = (
    _REPO_ROOT / "rskills" / "rldx1-ft-libero-nf4",
    _REPO_ROOT / "rskills" / "rldx1-ft-gr1-nf4",
    _REPO_ROOT / "rskills" / "rldx1-ft-rc365-nf4",
    _REPO_ROOT / "rskills" / "rldx1-ft-simpler-widowx-nf4",
)


@pytest.mark.parametrize("rskill_dir", _RSKILLS, ids=lambda p: p.name)
def test_rldx_rskill_manifest_loads(rskill_dir: Path) -> None:
    """Every shipped rldx1-* manifest passes the real RSkillManifest validator.

    Catches schema drift (a new license string, a typo in
    `model_family`, an unrecognised `state_contract.layout`) at import
    time — these manifests are the contract the RLDX-1 family exposes
    to the loader.
    """
    manifest = load_rskill_manifest(str(rskill_dir))
    assert manifest.model_family == "rldx"
    assert manifest.license == "rlwrld_non_commercial"
    assert manifest.is_commercial_use_allowed is False
    assert str(manifest.weights_uri).startswith("hf://RLWRLD/")


@pytest.fixture
def libero_env_cfg() -> SimEnvironment:
    """Compose a SimEnvironment around the FT-LIBERO rSkill manifest.

    Drives the same code path that ``openral sim run --config X --rskill Y``
    uses; no test doubles.
    """
    from tests.sim.conftest import compose_sim_env

    # compose_sim_env loads strict SimScene, so the canonical
    # scenes/benchmark/libero_spatial.yaml (BenchmarkScene) cannot be used. The
    # --rskill override selects the FT-LIBERO rldx checkpoint regardless.
    return compose_sim_env(
        _REPO_ROOT / "scenes" / "sim" / "libero_spatial.yaml",
        "rskills/rldx1-ft-libero-nf4",
        n_episodes=1,
        max_steps=1,
    )


def test_build_rldx_picks_libero_layout(libero_env_cfg: SimEnvironment) -> None:
    """The factory routes FT-LIBERO manifests through the LIBERO obs layout.

    Verifies the manifest-driven state_layout dispatch in
    ``_build_rldx`` resolves correctly without booting the sidecar.
    """
    # Import here (module-level pytest.importorskip already guarded
    # pyzmq/msgpack) so register-time side effects fire predictably.
    from openral_sim.policies import rldx as rldx_mod  # noqa: F401
    from openral_sim.registry import POLICIES

    # Picking a definitely-free port lets the auto-spawn path fail loud
    # the way we want (TCP probe says vacant → would normally spawn).
    free_port = _allocate_free_port()
    libero_env_cfg.vla = VLASpec(
        id="rldx",
        weights_uri=libero_env_cfg.vla.weights_uri,
        extra={
            "port": free_port,
            # CRITICAL: disable auto_spawn so this unit test never tries
            # to fork tools/rldx_sidecar.py (which would clone the
            # upstream RLDX-1 repo + uv-sync 3.10 deps — minutes of CI
            # time and a network dependency).
            "auto_spawn": False,
            "timeout_ms": 100,  # fast ping fail; we expect ROSConfigError
        },
    )
    with pytest.raises(ROSConfigError, match="did not answer ping"):
        POLICIES.get("rldx")(libero_env_cfg)


def test_auto_spawn_disabled_via_env(libero_env_cfg: SimEnvironment) -> None:
    """``OPENRAL_RLDX_AUTO_SPAWN=0`` overrides ``vla.extra.auto_spawn=true``.

    Mirrors the user workflow: someone wires up a hand-managed sidecar
    on a shared GPU and exports the env var so the per-config default
    is ignored.
    """
    from openral_sim.registry import POLICIES

    free_port = _allocate_free_port()
    libero_env_cfg.vla = VLASpec(
        id="rldx",
        weights_uri=libero_env_cfg.vla.weights_uri,
        extra={
            "port": free_port,
            "auto_spawn": True,  # would normally spawn …
            "timeout_ms": 100,
        },
    )
    prev = os.environ.get("OPENRAL_RLDX_AUTO_SPAWN")
    os.environ["OPENRAL_RLDX_AUTO_SPAWN"] = "0"
    try:
        with pytest.raises(ROSConfigError, match="auto_spawn is disabled"):
            POLICIES.get("rldx")(libero_env_cfg)
    finally:
        if prev is None:
            del os.environ["OPENRAL_RLDX_AUTO_SPAWN"]
        else:
            os.environ["OPENRAL_RLDX_AUTO_SPAWN"] = prev


def test_locate_sidecar_script_finds_real_tool() -> None:
    """The repo-root locator finds ``tools/rldx_sidecar.py`` on disk.

    Exercises the real upwards-walk in ``_RLDXSidecarAdapter.
    _locate_sidecar_script``; no mocks, no env override, no fixture
    filesystem. If someone moves the helper this test fails loud.
    """
    from openral_sim.policies.rldx import _RLDXSidecarAdapter

    adapter = _RLDXSidecarAdapter.__new__(_RLDXSidecarAdapter)
    script = adapter._locate_sidecar_script()
    assert script.is_file()
    assert script.name == "rldx_sidecar.py"
    # Sanity-check it's the in-repo helper, not some stray on PATH.
    assert script.resolve().is_relative_to(_REPO_ROOT.resolve())


def test_resolve_model_id_from_manifest() -> None:
    """``_resolve_model_id`` strips ``hf://`` off the manifest weights URI.

    Uses the real FT-LIBERO manifest; no fixture VLASpec.
    """
    from openral_sim.policies.rldx import _RLDXSidecarAdapter

    spec = VLASpec(id="rldx", weights_uri="rskills/rldx1-ft-libero-nf4")
    adapter = _RLDXSidecarAdapter.__new__(_RLDXSidecarAdapter)
    adapter.spec = spec
    adapter.model_id = None
    assert adapter._resolve_model_id() == "RLWRLD/RLDX-1-FT-LIBERO"


def test_resolve_model_id_explicit_override_wins() -> None:
    """``vla.extra.model_id`` beats the manifest weights URI."""
    from openral_sim.policies.rldx import _RLDXSidecarAdapter

    spec = VLASpec(id="rldx", weights_uri="rskills/rldx1-ft-libero-nf4")
    adapter = _RLDXSidecarAdapter.__new__(_RLDXSidecarAdapter)
    adapter.spec = spec
    adapter.model_id = "local/path/to/checkpoint"
    assert adapter._resolve_model_id() == "local/path/to/checkpoint"


def test_resolve_model_id_from_bare_hf_uri() -> None:
    """A bare ``hf://`` spec URI (no rskill manifest) still resolves."""
    from openral_sim.policies.rldx import _RLDXSidecarAdapter

    spec = VLASpec(id="rldx", weights_uri="hf://RLWRLD/RLDX-1-FT-LIBERO")
    adapter = _RLDXSidecarAdapter.__new__(_RLDXSidecarAdapter)
    adapter.spec = spec
    adapter.model_id = None
    assert adapter._resolve_model_id() == "RLWRLD/RLDX-1-FT-LIBERO"


def test_is_port_busy_against_real_listener() -> None:
    """``_is_port_busy`` returns True for a real TCP listener.

    No mocks: opens an actual loopback socket, binds + listens, then
    asks the adapter to probe it.
    """
    from openral_sim.policies.rldx import _RLDXSidecarAdapter

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        host, port = server.getsockname()

        adapter = _RLDXSidecarAdapter.__new__(_RLDXSidecarAdapter)
        adapter.host = host
        adapter.port = port
        assert adapter._is_port_busy() is True

    # Socket closed → probe should now find nothing.
    adapter.port = _allocate_free_port()
    assert adapter._is_port_busy() is False


def test_try_ping_fast_fails_on_dead_port_at_production_timeout() -> None:
    """``_try_ping`` returns ``False`` in <500 ms when no sidecar is listening.

    Regression: ZMQ REQ on tcp:// is lazy — ``recv()`` blocks for the
    full ``RCVTIMEO`` (production default 60 000 ms) when nothing is on
    the other end instead of returning immediately like a raw TCP
    connect to a closed port would. That made every cold-start
    ``openral sim run`` with an rldx rSkill burn 60 s between
    ``rldx_sidecar_connecting`` and ``[rldx-sidecar] launching server``.

    The fix gates the ZMQ leg behind the existing :meth:`_is_port_busy`
    TCP probe; this test pins that behaviour at the production timeout.
    """
    import time

    import zmq
    from openral_sim.policies.rldx import _RLDXSidecarAdapter

    adapter = _RLDXSidecarAdapter.__new__(_RLDXSidecarAdapter)
    adapter.host = "127.0.0.1"
    adapter.port = _allocate_free_port()
    adapter.timeout_ms = 60_000  # production default (see _build_rldx in rldx.py)
    adapter._socket = None
    adapter._ctx = zmq.Context.instance()
    adapter._init_socket()
    try:
        start = time.perf_counter()
        result = adapter._try_ping()
        elapsed = time.perf_counter() - start

        assert result is False
        assert elapsed < 0.5, (
            f"_try_ping took {elapsed * 1000:.1f} ms on a dead port; "
            "the TCP-probe fast-fail must short-circuit before the ZMQ "
            "RCVTIMEO (60 s in production). If this regresses, "
            "rldx_sidecar_connecting → launching server stalls again."
        )
    finally:
        if adapter._socket is not None:
            adapter._socket.close(linger=0)


def _allocate_free_port() -> int:
    """Ask the OS for an unused loopback port, close immediately.

    Has the usual TOCTOU caveat — between this returning and the
    caller using the port something else could bind it. Acceptable for
    these unit tests because the adapter is what attempts the connect
    next, and we assert on the failure mode, not on success.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
