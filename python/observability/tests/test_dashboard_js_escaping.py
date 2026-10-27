"""Telemetry strings must not reach ``innerHTML`` in ``dashboard.js`` unescaped.

Joint names, camera ids/modality, log bodies and event kinds come from any OTLP
sender (``POST /v1/traces`` / ``/v1/logs``). The dashboard renders several of them
through HTML template strings, so each one goes through the page's ``esc`` helper.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_JS = (
    Path(__file__).resolve().parents[1] / "src/openral_observability/dashboard/static/dashboard.js"
).read_text(encoding="utf-8")

# Interpolations that carried raw telemetry into innerHTML before the fix.
_RAW_TELEMETRY = (
    ">${name}<",
    '"${name}"',
    ">${body}<",
    ">${meta}<",
    ">${sev}<",
    "${ev.kind}",
    "${cam.modality",
)


@pytest.mark.parametrize("raw", _RAW_TELEMETRY)
def test_no_raw_telemetry_interpolation(raw: str) -> None:
    assert raw not in _JS


def test_esc_neutralises_markup() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip(reason="node is not installed")
    helper = "\n".join(
        line for line in _JS.splitlines() if re.match(r"\s*const (_ESC|esc) = ", line)
    )
    assert helper.count("const ") == 2, "esc helper not found in dashboard.js"
    payload = "<img src=x onerror=alert(1)>\"'&"
    script = f"{helper}\nprocess.stdout.write(esc({json.dumps(payload)}) + '|' + esc(null));"
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout
    assert out == "&lt;img src=x onerror=alert(1)&gt;&quot;&#39;&amp;|"
