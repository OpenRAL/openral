"""Live debugging dashboard for OpenRAL — embedded OTLP/HTTP receiver + SSE UI.

The dashboard is a single ASGI app that:

* Accepts OTLP/HTTP protobuf exports on ``/v1/traces``, ``/v1/metrics``,
  ``/v1/logs`` (the OTel-spec endpoints used when
  ``OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf``).
* Aggregates the latest signals into a thread-safe
  :class:`~openral_observability.dashboard.store.TelemetryStore`.
* Streams deltas to the browser over Server-Sent Events at
  ``/api/stream`` and serves a single-page UI at ``/``.

Run it with :func:`run_dashboard` or mount the ASGI app returned by
:func:`create_app` into your own server.

The dashboard is read-only: it never writes back to the agent. It is
designed so that pointing any OpenRAL workload at it via
``OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:<port>`` +
``OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`` surfaces the run live —
no Jaeger or Tempo required.

Example:
    >>> from openral_observability.dashboard import create_app, TelemetryStore
    >>> store = TelemetryStore()
    >>> app = create_app(store)  # FastAPI ASGI app
"""

from __future__ import annotations

from openral_observability.dashboard.app import create_app
from openral_observability.dashboard.attach import attached_dashboard, spawn_dashboard
from openral_observability.dashboard.server import run_dashboard
from openral_observability.dashboard.store import TelemetryEvent, TelemetryStore

__all__ = [
    "TelemetryEvent",
    "TelemetryStore",
    "attached_dashboard",
    "create_app",
    "run_dashboard",
    "spawn_dashboard",
]
