"""Unit tests for the rldx/gr00t-family sidecar boot-failure classification.

The GR00T-family adapter (:class:`_Gr00tFamilySidecarAdapter`, driving both
``rldx`` and ``gr00t``) forks its own boot helper and has its own connect loop —
separate from :class:`openral_sim.sidecar.SidecarClient`. When the spawned child
*crashes* during boot (child exits non-zero — e.g. CUDA OOM when the bf16-resident
3B weights do not co-fit alongside the Isaac renderer on an 8 GB host, observed as
``rldx_sidecar_died_during_boot returncode=-9``), it must NOT be reported as "did
not answer ping within {timeout}s" — that reads like a slow/hung bootstrap and
sends the operator down the wrong path. :meth:`_boot_failure_error` distinguishes
the two off the captured child exit code, mirroring the SidecarClient fix.
"""

from __future__ import annotations

from openral_core.exceptions import ROSConfigError
from openral_sim.policies.rldx import _Gr00tFamilySidecarAdapter


def _adapter(family: str = "rldx") -> _Gr00tFamilySidecarAdapter:
    # Bypass __init__/__post_init__ (which imports zmq, ensures backend deps,
    # and opens a socket) — _boot_failure_error only reads .family and
    # .boot_timeout_s.
    adapter = _Gr00tFamilySidecarAdapter.__new__(_Gr00tFamilySidecarAdapter)
    adapter.family = family
    adapter.boot_timeout_s = 900.0
    return adapter


def test_crash_at_boot_reports_exit_code_not_timeout() -> None:
    err = _adapter()._boot_failure_error("tcp://127.0.0.1:37229", returncode=1)
    assert isinstance(err, ROSConfigError)
    msg = str(err)
    assert "exited with code 1" in msg
    assert "crashed, it did not time out" in msg
    # Names the concrete 8 GB co-residency OOM cause so the operator can act.
    assert "co-fit" in msg
    assert "did not answer ping within" not in msg


def test_oom_killer_signal_is_a_crash_not_a_timeout() -> None:
    # rc=-9 is the OOM-killer SIGKILL — the canonical 8 GB Isaac+rldx failure.
    err = _adapter("gr00t")._boot_failure_error("tcp://127.0.0.1:37229", returncode=-9)
    msg = str(err)
    assert "exited with code -9" in msg
    assert "gr00t sidecar process" in msg
    assert "did not answer ping within" not in msg


def test_genuine_timeout_keeps_slow_path_message() -> None:
    # returncode None == child still running == real timeout (slow bootstrap).
    err = _adapter()._boot_failure_error("tcp://127.0.0.1:37229", returncode=None)
    msg = str(err)
    assert "did not answer ping within" in msg
    assert "900 s" in msg
    assert "git clone" in msg
    assert "exited with code" not in msg


def test_clean_exit_zero_falls_back_to_timeout_message() -> None:
    # rc==0 (clean exit, no ping) is not a crash signature — keep the generic msg.
    err = _adapter()._boot_failure_error("tcp://127.0.0.1:37229", returncode=0)
    assert "did not answer ping within" in str(err)
