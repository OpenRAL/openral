"""Unit tests for sidecar identity verification, layout dispatch, and ports.

Covers the three fixes for the "always loads RLDX's environment" bug, where a
second sim eval silently bound to whatever sidecar was already holding the
shared default port (or, for GR00T, was force-fed the LIBERO obs contract):

1. **Manifest-driven layout dispatch** (:func:`_resolve_state_layout`) shared
   by the ``rldx`` *and* ``gr00t`` factories — the GR00T adapter no longer
   hardcodes ``state_layout="libero"``.
2. **Per-identity default port** (:func:`_derive_sidecar_port` /
   :func:`_resolve_sidecar_port`) so two different checkpoints never collide
   on one port.
3. **Identity-checked reuse** (``_verify_existing_identity``) — the adapter
   refuses to adopt a pre-existing sidecar serving a different checkpoint.

Per CLAUDE.md §1.11 there are no mocks: layout dispatch runs against the real
``RSkillManifest`` files under ``rskills/``; the reuse tests stand up a **real**
ZMQ REP socket that answers ``ping`` (a legitimate network-boundary double),
and write a real on-disk identity record via the shared sidecar helpers.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from openral_core import SimEnvironment, VLASpec
from openral_core.exceptions import ROSConfigError
from openral_rskill.loader import load_rskill_manifest

pytest.importorskip("zmq", reason="sidecar tests require the `rldx` group (pyzmq + msgpack)")
pytest.importorskip("msgpack", reason="sidecar tests require the `rldx` group")

_REPO_ROOT = Path(__file__).parent.parent.parent


# ─── Layout dispatch (shared by both factories) ──────────────────────────


@pytest.mark.parametrize(
    ("rskill", "expected_layout"),
    [
        ("rldx1-ft-libero-nf4", "libero"),
        ("rldx1-ft-gr1-nf4", "gr1"),
        ("rldx1-ft-rc365-nf4", "rc365"),
        ("rldx1-ft-simpler-widowx-nf4", "simpler_widowx"),
        ("gr00t-n17-libero", "libero"),
    ],
)
def test_resolve_state_layout_from_real_manifests(rskill: str, expected_layout: str) -> None:
    """The shared layout helper reads ``state_contract.layout`` off each manifest.

    This is the dispatch both ``_build_rldx`` and ``_build_gr00t`` now call;
    a regression here is exactly the bug where every checkpoint collapses to
    the LIBERO obs contract.
    """
    from openral_sim.policies.rldx import _resolve_state_layout

    manifest = load_rskill_manifest(str(_REPO_ROOT / "rskills" / rskill))
    assert _resolve_state_layout(manifest) == expected_layout


def test_resolve_state_layout_defaults_libero_without_manifest() -> None:
    """A missing manifest / state_contract falls back to the LIBERO contract."""
    from openral_sim.policies.rldx import _resolve_state_layout

    assert _resolve_state_layout(None) == "libero"


# ─── Per-identity default port ───────────────────────────────────────────


def test_derive_sidecar_port_is_deterministic_and_in_range() -> None:
    from openral_sim.policies.rldx import (
        _SIDECAR_PORT_MAX,
        _SIDECAR_PORT_MIN,
        _derive_sidecar_port,
    )

    kw = dict(family="rldx", model="RLWRLD/RLDX-1-FT-LIBERO", embodiment_tag="GENERAL_EMBODIMENT")
    a = _derive_sidecar_port(quantization="nf4", layout="libero", **kw)
    b = _derive_sidecar_port(quantization="nf4", layout="libero", **kw)
    assert a == b
    assert _SIDECAR_PORT_MIN <= a < _SIDECAR_PORT_MAX


def test_derive_sidecar_port_separates_distinct_identities() -> None:
    """Different family / model / layout / quant → different default port.

    This is what stops a RoboCasa or GR00T run from defaulting onto the port
    an RLDX-LIBERO run is already using.
    """
    from openral_sim.policies.rldx import _derive_sidecar_port

    base = dict(
        family="rldx",
        model="RLWRLD/RLDX-1-FT-LIBERO",
        embodiment_tag="GENERAL_EMBODIMENT",
        quantization="nf4",
        layout="libero",
    )
    ports = {
        _derive_sidecar_port(**base),
        _derive_sidecar_port(**{**base, "family": "gr00t"}),
        _derive_sidecar_port(**{**base, "model": "RLWRLD/RLDX-1-FT-RC365"}),
        _derive_sidecar_port(**{**base, "layout": "rc365"}),
        _derive_sidecar_port(**{**base, "quantization": "int8"}),
    }
    assert len(ports) == 5  # all distinct


def test_resolve_sidecar_port_precedence() -> None:
    """env pin > vla.extra pin > per-identity default."""
    from openral_sim.policies.rldx import _derive_sidecar_port, _resolve_sidecar_port

    ident = dict(
        family="rldx",
        model="m",
        embodiment_tag="e",
        quantization="nf4",
        layout="libero",
    )
    assert _resolve_sidecar_port(port_env="5599", extra_port=7, **ident) == 5599
    assert _resolve_sidecar_port(port_env=None, extra_port=7, **ident) == 7
    assert _resolve_sidecar_port(port_env=None, extra_port=None, **ident) == _derive_sidecar_port(
        **ident
    )


# ─── Identity-checked reuse (real ZMQ ping responder) ────────────────────


@contextmanager
def _zmq_ping_responder(port: int) -> Iterator[None]:
    """Bind a real ZMQ REP socket that answers any request with an empty blob.

    The adapter only needs ``ping`` to return *something* unpackable to treat
    the sidecar as alive; this stands in for the upstream PolicyServer's ping
    at the network boundary (a permitted double, CLAUDE.md §1.11).
    """
    import msgpack
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.bind(f"tcp://127.0.0.1:{port}")
    stop = threading.Event()

    def _serve() -> None:
        while not stop.is_set():
            try:
                sock.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            sock.send(msgpack.packb([{}, {}], use_bin_type=True))

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2.0)
        sock.close(linger=0)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def rc365_env_cfg() -> SimEnvironment:
    """A SimEnvironment whose rSkill declares the non-default ``rc365`` layout."""
    from tests.sim.conftest import compose_sim_env

    return compose_sim_env(
        _REPO_ROOT / "scenes" / "sim" / "franka_libero_pnp.yaml",
        "rskills/rldx1-ft-rc365-nf4",
        n_episodes=1,
        max_steps=1,
    )


def test_factory_reuses_matching_sidecar_with_correct_layout(
    rc365_env_cfg: SimEnvironment,
) -> None:
    """A live, identity-matching sidecar is reused — and the adapter carries
    the manifest's ``rc365`` layout, not a hardcoded LIBERO contract.
    """
    from openral_sim._sidecar_common import sidecar_identity_path, write_sidecar_identity
    from openral_sim.registry import POLICIES

    port = _free_port()
    rc365_env_cfg.vla = VLASpec(
        id="rldx",
        weights_uri=rc365_env_cfg.vla.weights_uri,
        extra={"port": port, "timeout_ms": 500},
    )
    write_sidecar_identity(
        port=port,
        family="rldx",
        model="RLWRLD/RLDX-1-FT-RC365",
        embodiment_tag="GENERAL_EMBODIMENT",
        quantization="nf4",
    )
    try:
        with _zmq_ping_responder(port):
            adapter = POLICIES.get("rldx")(rc365_env_cfg)
        assert adapter.state_layout == "rc365"
        assert adapter._child is None  # reused, never spawned
    finally:
        sidecar_identity_path(port).unlink(missing_ok=True)


def test_factory_refuses_mismatched_sidecar(rc365_env_cfg: SimEnvironment) -> None:
    """A sidecar serving a *different* checkpoint is rejected, not adopted.

    This is the core "always loads RLDX's environment" guard: the port is
    busy and answers ping, but the recorded identity is a different model,
    so the factory fails closed instead of running the wrong policy.
    """
    from openral_sim._sidecar_common import sidecar_identity_path, write_sidecar_identity
    from openral_sim.registry import POLICIES

    port = _free_port()
    rc365_env_cfg.vla = VLASpec(
        id="rldx",
        weights_uri=rc365_env_cfg.vla.weights_uri,
        extra={"port": port, "timeout_ms": 500},
    )
    # A stale LIBERO sidecar is holding the port.
    write_sidecar_identity(
        port=port,
        family="rldx",
        model="RLWRLD/RLDX-1-FT-LIBERO",
        embodiment_tag="GENERAL_EMBODIMENT",
        quantization="nf4",
    )
    try:
        with _zmq_ping_responder(port), pytest.raises(ROSConfigError, match="already serving"):
            POLICIES.get("rldx")(rc365_env_cfg)
    finally:
        sidecar_identity_path(port).unlink(missing_ok=True)


def test_factory_trusts_unidentified_sidecar(rc365_env_cfg: SimEnvironment) -> None:
    """No identity record on disk → reuse is allowed (operator-managed boot).

    Preserves the "boot the sidecar yourself" workflow: a live sidecar with
    no record is unverifiable, not a mismatch.
    """
    from openral_sim._sidecar_common import sidecar_identity_path
    from openral_sim.registry import POLICIES

    port = _free_port()
    sidecar_identity_path(port).unlink(missing_ok=True)  # ensure no record
    rc365_env_cfg.vla = VLASpec(
        id="rldx",
        weights_uri=rc365_env_cfg.vla.weights_uri,
        extra={"port": port, "timeout_ms": 500},
    )
    with _zmq_ping_responder(port):
        adapter = POLICIES.get("rldx")(rc365_env_cfg)
    assert adapter.state_layout == "rc365"
