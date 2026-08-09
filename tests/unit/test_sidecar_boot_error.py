"""Unit tests for SidecarClient boot-failure error classification.

A sidecar that *crashes* during boot (child exits non-zero — e.g. the GR00T/RLDX
processor KeyErrors because a pretrain base like RLDX-1-PT has no modality config
for the requested embodiment) must NOT be reported as "did not answer ping within
{timeout}s" — that reads like a slow/hung bootstrap and sends the operator down
the wrong path. :meth:`SidecarClient._boot_failure_error` distinguishes the two
off the captured child exit code.

The live-but-silent case splits the same way (issue #89): a child that never
bound the port may be slow *or* stalled, and one that bound it but never
answered is neither. Advising "raise the boot timeout" is only right for the
first, so the message must not assert the slow reading — Isaac's Kit reached
``app ready`` in 12 s and then burned the remaining 1188 s on a failed
extension load.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager

from openral_core.exceptions import ROSConfigError
from openral_sim.sidecar import SidecarClient

_ENDPOINT = "tcp://127.0.0.1:37229"


def _client(port: int = 37229) -> SidecarClient:
    return SidecarClient(
        name="rldx",
        host="127.0.0.1",
        port=port,
        timeout_ms=1000,
        boot_timeout_s=900.0,
        launch_argv=["python", "tools/rldx_sidecar.py"],
        auto_spawn=True,
    )


@contextmanager
def _listening_port() -> Iterator[int]:
    """Bind a real ephemeral TCP port that accepts but never replies."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        yield int(srv.getsockname()[1])


def test_crash_at_boot_reports_exit_code_not_timeout() -> None:
    err = _client()._boot_failure_error(_ENDPOINT, returncode=1)
    assert isinstance(err, ROSConfigError)
    msg = str(err)
    assert "exited with code 1" in msg
    assert "crashed, it did not time out" in msg
    # Names the concrete RLDX-1-PT pretrain-base cause so the operator can act.
    assert "pretrain base" in msg
    assert "never bound" not in msg


def test_live_child_that_never_bound_offers_both_readings() -> None:
    # returncode None == child still running, port free == it never got that far.
    err = _client()._boot_failure_error(_ENDPOINT, returncode=None)
    msg = str(err)
    assert "never bound" in msg
    assert "900s" in msg
    assert "exited with code" not in msg
    # Both readings, neither asserted: slow bootstrap -> raise the timeout;
    # stalled after its own startup -> raising it only lengthens the wait.
    assert "raise the boot timeout" in msg
    assert "it stalled" in msg


def test_bound_but_mute_port_rules_out_a_slow_bootstrap() -> None:
    with _listening_port() as port:
        err = _client(port)._boot_failure_error(f"tcp://127.0.0.1:{port}", returncode=None)
    msg = str(err)
    assert "never answered ping" in msg
    assert "not a slow bootstrap" in msg
    assert "raising the boot timeout will not help" in msg


def test_clean_exit_zero_falls_back_to_timeout_message() -> None:
    # rc==0 (clean exit, no ping) is not a crash signature — keep the generic msg.
    assert "never bound" in str(_client()._boot_failure_error(_ENDPOINT, returncode=0))
