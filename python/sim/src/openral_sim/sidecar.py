"""Shared out-of-process sidecar transport — ZMQ REQ/REP + msgpack ndarray codec.

Some policies/scenes can't run in the openral py3.12 venv (Isaac Sim ships
per-interpreter wheels; RLDX-1 pins py3.10 + an incompatible torch stack), so
they run in their own venv as a long-lived process and are driven over a ZMQ
``REQ`` ↔ ``REP`` socket framed by msgpack. This module is the **canonical
openral-side transport** for that pattern: a numpy-aware msgpack codec plus a
:class:`SidecarClient` that owns the socket, optionally auto-spawns the child
process, and answers typed errors.

New sidecar integrations (the Isaac Sim scene backend,
:mod:`openral_sim.backends.isaac_sim`) consume this directly. The RLDX-1 policy
adapter (:mod:`openral_sim.policies.rldx`) predates it and keeps its own copy —
its wire codec is locked to the upstream server's ``__ndarray_class__`` sentinel
and its real path (a Qwen3-VL sidecar) cannot be exercised in CI — so migrating
it is deferred; this module is the shape it should move toward.

Exception contract (CLAUDE.md §5):
    * connect-time failures (no sidecar, spawn never binds) → ``ROSConfigError``;
    * a sidecar-side endpoint fault or a malformed reply at call time →
      ``ROSRuntimeError`` (a runtime fault, not a misconfiguration).
"""

from __future__ import annotations

import atexit
import contextlib
import io
import os
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

from openral_sim._sidecar_common import read_sidecar_identity

_log = structlog.get_logger(__name__)

_NDARRAY_SENTINEL = "__ndarray__"

# Sidecars spawn detached (own session) so a terminal Ctrl-C during the slow
# boot can't kill them. The cost is that a HARD-killed ``openral`` (SIGTERM
# from ``timeout`` / an orchestrator, or a crash before the backend's
# ``close()`` runs) leaks the sidecar — it keeps holding GPU memory and its
# port. We reap any sidecar WE spawned when the owning process exits: ``atexit``
# covers clean exits, ``sys.exit``, Ctrl-C (KeyboardInterrupt unwinds to exit)
# and uncaught exceptions; a chained, main-thread-only SIGTERM handler covers
# ``timeout``-style kills. SIGINT is left alone so we don't fight rclpy's
# Ctrl-C shutdown in ``openral deploy sim``. Reaping only ever touches children
# THIS process spawned (an adopted, hand-launched sidecar is never in here).
_spawned_children: list[subprocess.Popen[bytes]] = []
# Non-empty once the process-exit reaper has been installed (a set, not a bool,
# so the install guard needs no `global` statement).
_exit_reaper_installed: set[str] = set()


def _reap_spawned_children() -> None:
    """SIGTERM the process group of every still-running sidecar we spawned."""
    for child in list(_spawned_children):
        if child.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
    _spawned_children.clear()


def _install_exit_reaper() -> None:
    """Install the atexit + SIGTERM reaper once per process (idempotent)."""
    if _exit_reaper_installed:
        return
    _exit_reaper_installed.add("installed")
    atexit.register(_reap_spawned_children)
    # signal.signal only works on the main thread; in a worker thread (e.g. a
    # ROS executor callback) fall back to atexit-only rather than raising.
    if threading.current_thread() is not threading.main_thread():
        return
    with contextlib.suppress(ValueError):
        _previous_sigterm = signal.getsignal(signal.SIGTERM)

        def _on_sigterm(signum: int, frame: Any) -> None:
            _reap_spawned_children()
            if callable(_previous_sigterm):
                _previous_sigterm(signum, frame)
            else:
                # Restore default disposition and re-raise so exit status is right.
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        signal.signal(signal.SIGTERM, _on_sigterm)


def _register_spawned_child(child: subprocess.Popen[bytes]) -> None:
    _spawned_children.append(child)
    _install_exit_reaper()


def _deregister_spawned_child(child: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ValueError):
        _spawned_children.remove(child)


def encode_ndarray(obj: Any) -> Any:
    """Msgpack ``default`` hook: serialize ndarrays via ``np.save``.

    Keeps the wire msgpack-only (no msgpack-numpy dependency). The sidecar
    process must use the same sentinel shape on its send path.
    """
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {_NDARRAY_SENTINEL: True, "npy": buf.getvalue()}
    return obj


def decode_ndarray(obj: dict[str, Any]) -> Any:
    """Msgpack ``object_hook``: reverse :func:`encode_ndarray`.

    A sentinel dict missing the ``npy`` payload (corrupt / drifted frame) is
    returned unchanged rather than raising a bare ``KeyError`` mid-unpack.
    """
    if _NDARRAY_SENTINEL in obj and "npy" in obj:
        return np.load(io.BytesIO(obj["npy"]), allow_pickle=False)
    return obj


def require_key(reply: dict[str, Any], key: str, *, name: str) -> Any:
    """Return ``reply[key]`` or raise a typed ``ROSRuntimeError`` if absent.

    Guards against a well-formed-but-incomplete sidecar reply (protocol drift)
    surfacing as a bare ``KeyError`` instead of the typed-exception hierarchy.
    """
    if key not in reply:
        raise ROSRuntimeError(f"{name} sidecar reply missing required key {key!r}")
    return reply[key]


@dataclass
class SidecarClient:
    """Owns a ZMQ REQ socket + an optional child sidecar process.

    One client per sidecar. Pings an existing sidecar at ``host:port`` first;
    when none answers and ``auto_spawn`` is set, ``Popen``-s ``launch_argv`` in
    its own session and polls until it binds.

    Attributes:
        name: Short label used in log events and error messages (e.g. "isaac").
        host / port: ZMQ endpoint.
        timeout_ms: REQ send/recv timeout (one slow GPU step must not read as a
            dead sidecar — keep generous).
        boot_timeout_s: How long :meth:`connect` waits for an auto-spawned child
            to answer its first ping.
        launch_argv: Full command to spawn the sidecar (interpreter + script +
            args). Only used when ``auto_spawn`` and no sidecar is already up.
        auto_spawn: Spawn the child when the initial ping fails.
    """

    name: str
    host: str
    port: int
    timeout_ms: int
    boot_timeout_s: float
    launch_argv: list[str]
    auto_spawn: bool
    # When set, an EXISTING sidecar found on host:port must report a matching
    # identity in its ``ping`` reply (every key here equals the ping value) — else
    # it serves a different scene and adopting it would yield silently-wrong data
    # (e.g. two Isaac scenes that collided on one port). Keys absent from the ping
    # reply are skipped (back-compat with sidecars that don't report identity).
    expected_identity: dict[str, Any] | None = None
    _ctx: Any = None
    _socket: Any = None
    _child: subprocess.Popen[bytes] | None = field(default=None)

    def connect(self) -> None:
        """Ping an existing sidecar, else spawn one and wait for it to bind."""
        import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # reason: opt-in sidecar group

        self._ctx = zmq.Context.instance()
        self._init_socket()
        endpoint = f"tcp://{self.host}:{self.port}"
        existing = self._ping_reply()
        if existing is not None:
            mismatched = self._identity_mismatch(existing)
            if not mismatched:
                _log.info(
                    "sidecar_connected", sidecar=self.name, endpoint=endpoint, mode="existing"
                )
                return
            if not self.auto_spawn:
                # Can't restart a sidecar we were told not to spawn — surface
                # the explicit "stop it / use a distinct port" error.
                self._assert_identity(existing, endpoint)
            # auto_spawn: a reused sidecar is serving a DIFFERENT scene (a stale
            # one left bound to this port by a prior run). Reuse only ever helps
            # within one scene, so reap the stale sidecar and spawn a fresh one
            # rather than failing. Falls through to the _spawn() below.
            _log.info(
                "sidecar_restart_scene_mismatch",
                sidecar=self.name,
                endpoint=endpoint,
                mismatch={k: got for k, (got, _want) in mismatched.items()},
            )
            self._reap_stale_sidecar(endpoint)
        if not self.auto_spawn:
            raise ROSConfigError(
                f"{self.name} sidecar at {endpoint} did not answer ping and auto_spawn "
                f"is disabled. Boot it manually:\n  {' '.join(self.launch_argv)}"
            )
        self._spawn()
        if not self._wait_for_boot():
            # Distinguish a crash-at-boot (child exited non-zero) from a genuine
            # timeout (child still running but never answered ping). Capture the
            # exit code BEFORE _terminate_child() clears self._child.
            child = self._child
            rc = child.poll() if child is not None else None
            self._terminate_child()
            raise self._boot_failure_error(endpoint, rc)
        _log.info("sidecar_connected", sidecar=self.name, endpoint=endpoint, mode="auto-spawned")

    def _boot_failure_error(self, endpoint: str, returncode: int | None) -> ROSConfigError:
        """Build the right boot-failure error: crash-at-boot vs ping timeout.

        ``returncode`` is the child's exit code (``None`` if it was still
        running == genuine timeout, a non-zero int if it crashed during boot).
        A crash is NOT a slow bootstrap, so reporting "did not answer ping
        within {timeout}s" is misleading — surface the exit code and the
        common causes instead (e.g. evaluating a pretrain base like RLDX-1-PT
        directly: its processor has no modality config for the requested
        embodiment, so ``get_modality_configs()[<tag>]`` KeyErrors at boot).
        """
        if returncode is not None and returncode != 0:
            return ROSConfigError(
                f"{self.name} sidecar process exited with code {returncode} during boot on "
                f"{endpoint} — it crashed, it did not time out. Inspect the sidecar stdout "
                "above. Common causes: the checkpoint's processor has no modality config for "
                "the requested embodiment (e.g. running a pretrain base such as RLDX-1-PT "
                "directly instead of a task finetune), missing/incompatible weights, or CUDA OOM."
            )
        return ROSConfigError(
            f"{self.name} sidecar spawned but did not answer ping within "
            f"{self.boot_timeout_s:.0f}s on {endpoint}. Inspect the sidecar stdout "
            "above, or raise the boot timeout if the first-run bootstrap is slow."
        )

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """One REQ/REP round trip; raise on a sidecar-side or transport fault."""
        import msgpack  # type: ignore[import-not-found,import-untyped,unused-ignore]  # reason: opt-in sidecar group

        msg = {"endpoint": endpoint, "data": data or {}}
        self._socket.send(msgpack.packb(msg, default=encode_ndarray, use_bin_type=True))
        raw = self._socket.recv()
        reply = msgpack.unpackb(raw, object_hook=decode_ndarray, raw=False)
        # A sidecar-side endpoint exception (GPU fault, IK error, …) is a RUNTIME
        # fault, not a config error — type it so callers can tell it apart from
        # the connect-time ROSConfigError paths (CLAUDE.md §5).
        if isinstance(reply, dict) and "error" in reply:
            raise ROSRuntimeError(f"{self.name} sidecar error on {endpoint!r}: {reply['error']!r}")
        if not isinstance(reply, dict):
            raise ROSRuntimeError(
                f"{self.name} sidecar returned a non-dict reply for {endpoint!r} "
                f"(got {type(reply).__name__})."
            )
        return reply

    def require(self, reply: dict[str, Any], key: str) -> Any:
        """:func:`require_key` bound to this client's ``name`` for error text."""
        return require_key(reply, key, name=self.name)

    def close(self) -> None:
        """Idempotent teardown: socket first (so child exit can't race a recv)."""
        if self._socket is not None:
            with contextlib.suppress(Exception):
                self._socket.close()
            self._socket = None
        self._terminate_child()

    # ── internals ────────────────────────────────────────────────────────────

    def _init_socket(self) -> None:
        """Create (or recreate) the REQ socket — recreate clears the EFSM lock.

        A ZMQ REQ socket whose ``recv`` timed out is stuck in EFSM (strict
        send→recv pairing); every later ``send`` then raises until reopened.
        :meth:`_try_ping` recreates on failure so boot polling does not wedge.
        """
        import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # reason: opt-in sidecar group

        if self._socket is not None:
            with contextlib.suppress(Exception):
                self._socket.close(linger=0)
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def _try_ping(self) -> bool:
        """One ping gated behind a cheap TCP probe (REQ is lazy on tcp://)."""
        return self._ping_reply() is not None

    def _ping_reply(self) -> dict[str, Any] | None:
        """Ping behind a cheap TCP probe; return the reply, or None if none answered.

        The reply carries the sidecar's identity, which :meth:`_assert_identity`
        checks before an existing sidecar is adopted.
        """
        if not self._is_port_busy():
            return None
        try:
            return self.call("ping")
        except Exception:
            self._init_socket()
            return None

    def _identity_mismatch(self, reply: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
        """Return ``{key: (got, expected)}`` for each contradicted identity key.

        Empty dict means the existing sidecar matches (safe to adopt). Keys the
        sidecar does not report are skipped (back-compat with sidecars that
        predate identity reporting).
        """
        if not self.expected_identity:
            return {}
        return {
            k: (reply[k], v)
            for k, v in self.expected_identity.items()
            if k in reply and reply[k] != v
        }

    def _assert_identity(self, reply: dict[str, Any], endpoint: str) -> None:
        """Reject an existing sidecar whose ping identity contradicts the request.

        Adopting a sidecar serving a different scene would yield silently-wrong
        data on the same port. Used on the ``auto_spawn=False`` path (where we
        must not restart it); the ``auto_spawn`` path reaps + respawns instead.
        """
        mismatched = self._identity_mismatch(reply)
        if mismatched:
            detail = ", ".join(
                f"{k}: got {got!r}, expected {want!r}" for k, (got, want) in mismatched.items()
            )
            raise ROSConfigError(
                f"{self.name} sidecar at {endpoint} is already serving a different "
                f"scene ({detail}). A stale sidecar is bound to this port — stop it "
                f"(it runs in its own session) or run this scene on a distinct port "
                f"via backend_options['port']."
            )

    def _reap_stale_sidecar(self, endpoint: str) -> None:
        """SIGTERM→SIGKILL a stale sidecar bound to our port, then wait it out.

        The sidecar runs in its own session (PID == PGID), so we signal the
        whole process group — that takes down any grandchildren (e.g. the Isaac
        Kit subprocess) too. The PID is read from the on-disk identity record
        :func:`write_sidecar_identity` left for this port. Raises
        :class:`ROSConfigError` if the port can't be freed (so the caller never
        silently adopts the wrong-scene sidecar via the post-reap ping).
        """
        identity = read_sidecar_identity(self.port)
        pid_raw = identity.get("pid") if identity else None
        pid = int(pid_raw) if pid_raw is not None else None
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if pid is not None:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(os.getpgid(pid), sig)
            deadline = time.monotonic() + (10.0 if sig == signal.SIGTERM else 5.0)
            while time.monotonic() < deadline:
                if not self._is_port_busy():
                    _log.info("sidecar_reaped_stale", sidecar=self.name, pid=pid, port=self.port)
                    return
                time.sleep(0.5)
        raise ROSConfigError(
            f"{self.name} sidecar on {endpoint} serves a different scene and could not "
            f"be reaped automatically (pid={pid}). Stop it manually or run this scene "
            f"on a distinct port via backend_options['port']."
        )

    def _spawn(self) -> None:
        """Fork the launcher in its own session so Ctrl-C to openral spares boot."""
        if self._is_port_busy():
            _log.info("sidecar_spawn_skipped_port_busy", sidecar=self.name, port=self.port)
            return
        _log.info("sidecar_spawning", sidecar=self.name, argv=self.launch_argv, port=self.port)
        # Strip PYTHONPATH (+ VIRTUAL_ENV) from the child env: the sidecar runs a
        # DIFFERENT-interpreter venv (the Isaac py3.11 / RLDX py3.10 one) and is
        # self-contained, but our parent may carry a PYTHONPATH pointing at this
        # py3.12 venv's site-packages (e.g. `openral deploy sim` injects it). That
        # would shadow the sidecar venv's own numpy with an ABI-incompatible build
        # → "No module named 'numpy._core._multiarray_umath'" at sidecar boot.
        child_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
        self._child = subprocess.Popen(
            self.launch_argv,
            stdout=None,
            stderr=None,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=child_env,
        )
        # Reap this detached child if the owning process exits without calling
        # close() (crash / SIGTERM) — otherwise it leaks GPU memory + the port.
        _register_spawned_child(self._child)

    def _wait_for_boot(self) -> bool:
        """Poll ``ping`` until success, child death, or timeout."""
        deadline = time.monotonic() + self.boot_timeout_s
        while time.monotonic() < deadline:
            if self._child is not None and self._child.poll() is not None:
                _log.error(
                    "sidecar_died_during_boot", sidecar=self.name, returncode=self._child.returncode
                )
                return False
            if self._try_ping():
                return True
            time.sleep(2.0)
        return False

    def _is_port_busy(self) -> bool:
        """Cheap TCP probe — True if something is listening on the port."""
        with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            s.connect((self.host, self.port))
            return True
        return False

    def _terminate_child(self) -> None:
        """Best-effort SIGTERM → SIGKILL of any child we own."""
        child = self._child
        if child is None or child.poll() is not None:
            if child is not None:
                _deregister_spawned_child(child)
            self._child = None
            return
        _log.info("sidecar_terminating", sidecar=self.name, pid=child.pid)
        with contextlib.suppress(Exception):
            child.terminate()
            try:
                child.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5.0)
        _deregister_spawned_child(child)
        self._child = None
