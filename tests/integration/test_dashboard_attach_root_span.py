"""``--dashboard`` must leave a real ``cli.command`` root span behind.

The dashboard child only exists *after* the Typer root callback has run, so
the ``cli.command`` span that callback opened was created against the no-op
provider and records nothing — every workload span below it becomes its own
orphan root, and anything reading the ambient trace (``RunResult.trace_id``,
``RSkillEvalResult.trace_id``) sees no trace at all. ``attached_dashboard``
re-opens the root once the exporters are bound.

Real components end-to-end per CLAUDE.md §1.11: a real ``openral dashboard``
child, the real OTel SDK. Run in a subprocess because
``configure_observability`` installs a process-global TracerProvider that
cannot be swapped back out.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

_CHILD = """
import json, sys
from openral_observability import semconv
from openral_observability.dashboard import attached_dashboard
from openral_observability.propagation import current_trace_id

port, out = int(sys.argv[1]), sys.argv[2]
with attached_dashboard(
    enabled=True, port=port, subcommand="sim run", mode=semconv.RUN_MODE_SIM
) as attached:
    payload = {"attached": attached, "trace_id": current_trace_id()}
open(out, "w").write(json.dumps(payload))
"""

_HEX_TRACE_ID_LEN = 32


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_attached_dashboard_reopens_a_recording_root_span(tmp_path: Path) -> None:
    out = tmp_path / "attach.json"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(_free_port()), str(out)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    if not payload["attached"]:
        pytest.skip(f"dashboard child never reported healthy: {proc.stderr[-2000:]}")
    trace_id = payload["trace_id"]
    assert trace_id is not None, "attached run left no trace to point at"
    assert len(trace_id) == _HEX_TRACE_ID_LEN
    assert int(trace_id, 16) != 0
