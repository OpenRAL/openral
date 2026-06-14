"""Unit tests for SidecarClient boot-failure error classification.

A sidecar that *crashes* during boot (child exits non-zero — e.g. the GR00T/RLDX
processor KeyErrors because a pretrain base like RLDX-1-PT has no modality config
for the requested embodiment) must NOT be reported as "did not answer ping within
{timeout}s" — that reads like a slow/hung bootstrap and sends the operator down
the wrong path. :meth:`SidecarClient._boot_failure_error` distinguishes the two
off the captured child exit code.
"""

from __future__ import annotations

from openral_core.exceptions import ROSConfigError
from openral_sim.sidecar import SidecarClient


def _client() -> SidecarClient:
    return SidecarClient(
        name="rldx",
        host="127.0.0.1",
        port=37229,
        timeout_ms=1000,
        boot_timeout_s=900.0,
        launch_argv=["python", "tools/rldx_sidecar.py"],
        auto_spawn=True,
    )


def test_crash_at_boot_reports_exit_code_not_timeout() -> None:
    err = _client()._boot_failure_error("tcp://127.0.0.1:37229", returncode=1)
    assert isinstance(err, ROSConfigError)
    msg = str(err)
    assert "exited with code 1" in msg
    assert "crashed, it did not time out" in msg
    # Names the concrete RLDX-1-PT pretrain-base cause so the operator can act.
    assert "pretrain base" in msg
    assert "did not answer ping within" not in msg


def test_genuine_timeout_keeps_ping_message() -> None:
    # returncode None == child still running == real timeout.
    err = _client()._boot_failure_error("tcp://127.0.0.1:37229", returncode=None)
    msg = str(err)
    assert "did not answer ping within" in msg
    assert "900s" in msg
    assert "exited with code" not in msg


def test_clean_exit_zero_falls_back_to_timeout_message() -> None:
    # rc==0 (clean exit, no ping) is not a crash signature — keep the generic msg.
    err = _client()._boot_failure_error("tcp://127.0.0.1:37229", returncode=0)
    assert "did not answer ping within" in str(err)
