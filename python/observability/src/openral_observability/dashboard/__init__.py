"""The robot dashboard (ADR-0129) — embedded OTLP/HTTP receiver, robot API, robot UI.

The dashboard is a single ASGI app that:

* Accepts OTLP/HTTP protobuf exports on ``/v1/traces``, ``/v1/metrics``,
  ``/v1/logs`` (the OTel-spec endpoints used when
  ``OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf``), including the robot
  manifest the runtime emits as a log record.
* Aggregates the latest signals into a thread-safe ``TelemetryStore``.
* Serves the robot UI at ``/``, the manifest, state and an SSE stream under
  ``/api/``, and runs the manifest's actions (``POST /api/actions/{id}``):
  E-STOP, e-stop reset, prompts and rSkills, under the role and listener
  rules of ``remote_auth``. The safety kernel still disposes all motion.

Run it with ``run_dashboard`` or mount the ASGI app returned by
``create_app`` into your own server. Pointing any OpenRAL workload at it via
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
