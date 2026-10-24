"""With the OTLP bridge on, every package's structlog events reach the floor, not root's.

#418 r4 (Spark, 2026-10-17): with an OTLP endpoint set, ``install_structlog_bridge``
routed structlog through ``logging.getLogger(<module name>)``. Only the ``openral``
namespace had a level and a handler, so a module logger such as
``openral_world_state.object_tracker.tracker`` fell through to the root logger: its INFO
and DEBUG events (every tracker fit refusal) were neither printed nor exported, even
with ``OPENRAL_LOG_LEVEL=DEBUG``, and its warnings reached stderr only through
``logging.lastResort``. Each case runs in a fresh interpreter (the bridge is global).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

_PROBE = textwrap.dedent(
    """
    import json
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
    import structlog
    from openral_observability.logging import install_structlog_bridge

    exporter = InMemoryLogExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    install_structlog_bridge(provider)
    log = structlog.get_logger("openral_world_state.object_tracker.tracker")
    log.debug("probe_debug", why="too_few_points")
    log.info("probe_info", why="map_split")
    log.warning("probe_warning")
    exported = [str(r.log_record.body) for r in exporter.get_finished_logs()]
    print("EXPORTED " + json.dumps(exported), flush=True)
    """
)


def _run(level: str) -> tuple[list[str], str]:
    env = {**os.environ, "OPENRAL_LOG_LEVEL": level}
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    line = next(x for x in out.stdout.splitlines() if x.startswith("EXPORTED "))
    return json.loads(line.removeprefix("EXPORTED ")), out.stdout + out.stderr


@pytest.mark.parametrize(
    ("level", "seen", "unseen"),
    [
        ("INFO", ("probe_info", "probe_warning"), ("probe_debug",)),
        ("DEBUG", ("probe_debug", "probe_info", "probe_warning"), ()),
    ],
)
def test_a_package_logger_is_exported_and_printed_at_the_floor(
    level: str, seen: tuple[str, ...], unseen: tuple[str, ...]
) -> None:
    exported, console = _run(level)
    for event in seen:
        assert any(event in body for body in exported), (event, exported)
        assert event in console, (event, console)  # the deploy log shows it too
    for event in unseen:
        assert not any(event in body for body in exported)
        assert event not in console
