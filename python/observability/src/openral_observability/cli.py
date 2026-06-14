"""``cli.command`` root-span helper.

Per design §4.7, every ``openral`` CLI invocation opens a single ``cli.command``
span as the root of the trace tree. Every downstream span (``sim.run``,
``rskill.tick``, ``hal.send_action``, ``safety.check``, …) becomes a
child of it, and :attr:`RunResult.trace_id` ends up being the trace id of
``cli.command`` — making the trace trivially queryable from the printed
output.

Lives in its own module (rather than next to ``rskill_span`` /
``inference_span`` / ``safety_span``) because the CLI is one of the few
places the helpers know enough domain to set ``openral.run.*`` attributes,
and keeping the import surface flat avoids dragging the rest of
``tracing`` into the ``openral`` startup path.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

from opentelemetry import trace
from opentelemetry.trace import Span

from openral_observability import semconv

__all__ = ["cli_command_span"]

_TRACER_NAME = "openral"
_GIT_SHA_LEN: Final[int] = 12


def _git_sha() -> str | None:
    """Best-effort short git SHA via env var.

    The runner host is not guaranteed to have ``git`` on PATH (CI containers
    sometimes strip it), so this only reads ``OPENRAL_GIT_SHA`` /
    ``GIT_SHA`` / ``GITHUB_SHA``. Returns ``None`` when unset rather than
    shelling out — observability must never block the actuation path.
    """
    for env in ("OPENRAL_GIT_SHA", "GIT_SHA", "GITHUB_SHA"):
        value = os.environ.get(env)
        if value:
            return value[:_GIT_SHA_LEN] if len(value) > _GIT_SHA_LEN else value
    return None


@contextmanager
def cli_command_span(
    subcommand: str,
    *,
    mode: str | None = None,
    run_id: str | None = None,
    **attrs: Any,
) -> Iterator[Span]:
    """Open the root ``cli.command`` span for one ``openral`` invocation.

    Args:
        subcommand: ``openral`` subcommand name (``"sim run"``, ``"benchmark run"``,
            ``"skill install"``, …). Recorded as ``cli.subcommand``.
        mode: Optional ``openral.run.mode`` — one of
            :data:`semconv.RUN_MODE_SIM` / :data:`semconv.RUN_MODE_HARDWARE` /
            :data:`semconv.RUN_MODE_BENCHMARK`.
        run_id: Optional caller-supplied id (e.g. a ``RunResult.run_name``).
            Falls back to a new UUID4 hex when ``None`` — every invocation
            gets a stable identifier even in no-op mode.
        **attrs: Extra attributes recorded verbatim on the span.

    Yields:
        The active root span.

    Example:
        >>> from openral_observability import cli_command_span, semconv
        >>> with cli_command_span("sim run", mode=semconv.RUN_MODE_SIM):
        ...     pass
    """
    tagged: dict[str, Any] = {
        "cli.subcommand": subcommand,
        semconv.RUN_ID: run_id if run_id is not None else uuid.uuid4().hex,
    }
    if mode is not None:
        tagged[semconv.RUN_MODE] = mode
    git_sha = _git_sha()
    if git_sha is not None:
        tagged[semconv.RUN_GIT_SHA] = git_sha
    tagged.update(attrs)
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(semconv.SPAN_CLI_COMMAND, attributes=tagged) as span:
        yield span
