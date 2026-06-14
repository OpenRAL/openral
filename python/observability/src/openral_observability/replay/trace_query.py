"""HTTP client over the ADR-0017 dashboard's F7 trace-query endpoints.

The dashboard receiver exposes:

* ``GET /api/traces`` — list of indexed ``trace_id`` records.
* ``GET /api/spans/{trace_id}`` — full span list for one trace.

This client wraps both with ``urllib.request`` (no extra deps; the
dashboard's ``httpx`` is dev-only). It is intentionally tiny — the
correlator does the joining, this only fetches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__all__ = ["DashboardTraceClient", "TraceQueryError"]

_HTTP_NOT_FOUND = 404


class TraceQueryError(RuntimeError):
    """Raised when the dashboard does not return a usable JSON body."""


@dataclass(frozen=True)
class DashboardTraceClient:
    """Minimal read-only HTTP client for the dashboard trace endpoints.

    Args:
        base_url: Dashboard root (e.g. ``http://127.0.0.1:8000``). The
            client appends ``/api/traces`` / ``/api/spans/<trace_id>``
            verbatim.
        timeout_s: Per-request socket timeout. The dashboard is local
            so 5 s is generous.
    """

    base_url: str = "http://127.0.0.1:8000"
    timeout_s: float = 5.0

    def list_traces(self) -> list[dict[str, Any]]:
        """Return ``GET /api/traces`` decoded, sorted server-side most-recent first."""
        body = self._get_json("/api/traces")
        traces = body.get("traces") if isinstance(body, dict) else None
        if not isinstance(traces, list):
            msg = "dashboard /api/traces: response missing 'traces' list"
            raise TraceQueryError(msg)
        return [t for t in traces if isinstance(t, dict)]

    def get_spans(self, trace_id: str) -> list[dict[str, Any]]:
        """Return every span for ``trace_id``; empty list when not indexed."""
        try:
            body = self._get_json(f"/api/spans/{trace_id}")
        except HTTPError as exc:
            if exc.code == _HTTP_NOT_FOUND:
                return []
            raise
        spans = body.get("spans") if isinstance(body, dict) else None
        if not isinstance(spans, list):
            msg = f"dashboard /api/spans/{trace_id}: response missing 'spans'"
            raise TraceQueryError(msg)
        return [s for s in spans if isinstance(s, dict)]

    def _get_json(self, path: str) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        req = Request(url, headers={"Accept": "application/json"})
        # Dashboard is local loopback by default; users may override
        # ``base_url`` for proxied hosts.
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                payload = resp.read()
        except URLError as exc:
            msg = f"dashboard unreachable at {url}: {exc.reason}"
            raise TraceQueryError(msg) from exc
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            msg = f"dashboard {path}: non-JSON response"
            raise TraceQueryError(msg) from exc
        if not isinstance(decoded, dict):
            msg = f"dashboard {path}: expected JSON object, got {type(decoded).__name__}"
            raise TraceQueryError(msg)
        return decoded
