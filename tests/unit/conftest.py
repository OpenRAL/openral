"""Shared fixtures and helpers for ``tests/unit/``.

Consolidates fixtures/helpers that were duplicated verbatim across multiple
``tests/unit/test_*.py`` modules (repo cleanup, CLAUDE.md §1.3/§1.13 — don't
duplicate). Kept import-light: anything pulling in rclpy, onnx, or a VLA
stack is imported lazily inside the function that needs it, so modules that
don't exercise those paths pay nothing at collection time.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import structlog
from opentelemetry import metrics, trace
from opentelemetry.metrics import _internal as metrics_internal
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# ── OTel in-memory telemetry (mirrors python/observability/tests/conftest.py) ──


@pytest.fixture
def memory_exporter() -> Iterator[InMemorySpanExporter]:
    """Replace the global TracerProvider with one that records to memory."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # opentelemetry-api guards against re-setting the global provider once it
    # has been set; bypass that by writing through the private holder.
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    trace.set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        exporter.clear()


@pytest.fixture
def exporter(memory_exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    """Alias of ``memory_exporter`` — matches the reasoner tests' fixture name."""
    return memory_exporter


@pytest.fixture
def memory_metric_reader() -> Iterator[InMemoryMetricReader]:
    """Replace the global MeterProvider with one whose reader keeps data in memory."""
    from openral_observability.metrics import _reset_instrument_cache

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    # Same private-holder dance as TracerProvider — the API enforces set-once
    # semantics that get in the way of per-test isolation.
    metrics_internal._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    metrics_internal._METER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    metrics.set_meter_provider(provider)
    # The metrics module caches instruments per id(meter); swapping the
    # provider invalidates those keys naturally, but drop the cache
    # explicitly so two tests sharing fixture order don't see stale instruments.
    _reset_instrument_cache()
    try:
        yield reader
    finally:
        provider.shutdown()
        _reset_instrument_cache()


# ── rclpy context (module-scoped: one init/shutdown pair per test module) ─────


@pytest.fixture(scope="module")
def ros_init() -> Iterator[None]:
    """Module-scoped rclpy init/shutdown for tests gated behind ``importorskip("rclpy")``."""
    import rclpy

    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope="module")
def _rclpy_ctx(ros_init: None) -> None:
    """Alias of ``ros_init`` — matches test_hal_lifecycle_manifest.py's fixture name."""
    return ros_init


# ── structlog capture ───────────────────────────────────────────────────────


class _CaptureProcessor:
    """Real ``structlog`` processor that buffers events for assertion.

    Implements the processor contract — call returns the event_dict or raises
    ``structlog.DropEvent`` to stop the pipeline (dropped so test logs don't
    pollute pytest output).
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, logger: Any, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        del logger, method
        name = str(event_dict.pop("event", ""))
        self.events.append((name, dict(event_dict)))
        raise structlog.DropEvent

    def named(self, event: str) -> list[dict[str, object]]:
        """Every captured payload logged under ``event``."""
        return [payload for name, payload in self.events if name == event]


@pytest.fixture
def cap() -> Iterator[_CaptureProcessor]:
    """Install a fresh capture processor; restore structlog defaults after.

    ``structlog.reset_defaults`` undoes the global ``configure`` so the test
    pollutes neither pytest's own structlog setup nor other tests in the same
    session.
    """
    proc = _CaptureProcessor()
    structlog.reset_defaults()
    structlog.configure(processors=[proc])
    try:
        yield proc
    finally:
        structlog.reset_defaults()


# ── Collision-shape stand-in (the CollisionShape union has no 4th member) ─────


@dataclass(frozen=True)
class _CylinderShape:
    """A fourth primitive kind — what a future ``CollisionShape`` member looks like.

    ``openral_core.CollisionShape`` is closed over sphere/capsule/box, so no
    fixture in ``robots/`` can produce this; it exists only to reach an
    unknown-shape branch. Mirrors the stand-in in
    ``packages/openral_slam_bringup/test/test_depth_height_filter.py``.

    It deliberately carries ``radius_m``: that is exactly the shape that used
    to slip through, because an old bare ``else`` read ``shape.radius_m`` off
    anything that was not a ``BoxShape``.
    """

    shape: str = "cylinder"
    radius_m: float = 0.1
    length_m: float = 0.4


# ── Deterministic RT-DETR-like ONNX fixture (onnx is an optional dependency) ──


def _write_rtdetr_like_onnx(path: Path) -> None:
    """Write a deterministic 4-class RT-DETR-like ONNX to *path*.

    - q0 logits ``[-5, -5, 3.0, -5]``  -> car (idx 2),   sigmoid(3)  ~= 0.953
    - q1 logits ``[2.0, -5, -5, -5]``  -> person (idx 0), sigmoid(2) ~= 0.881
    - q2 logits ``[-5, -5, -5, -5]``   -> max ~= 0.007 (below 0.5 threshold)
    """
    import numpy as np
    import onnx
    import onnx.helper as h
    import onnx.numpy_helper as nph

    logits_data = np.array(
        [[[-5.0, -5.0, 3.0, -5.0], [2.0, -5.0, -5.0, -5.0], [-5.0, -5.0, -5.0, -5.0]]],
        dtype=np.float32,
    )
    boxes_data = np.array(
        [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]],
        dtype=np.float32,
    )

    logits_tensor = nph.from_array(logits_data, name="logits_const")
    boxes_tensor = nph.from_array(boxes_data, name="boxes_const")

    images_input = h.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 640, 640])
    logits_out = h.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 3, 4])
    boxes_out = h.make_tensor_value_info("boxes", onnx.TensorProto.FLOAT, [1, 3, 4])
    passthrough_out = h.make_tensor_value_info(
        "images_passthrough", onnx.TensorProto.FLOAT, [1, 3, 640, 640]
    )

    id_node = h.make_node("Identity", inputs=["images"], outputs=["images_passthrough"])
    logits_node = h.make_node("Constant", inputs=[], outputs=["logits"], value=logits_tensor)
    boxes_node = h.make_node("Constant", inputs=[], outputs=["boxes"], value=boxes_tensor)

    graph = h.make_graph(
        nodes=[id_node, logits_node, boxes_node],
        name="rtdetr_test",
        inputs=[images_input],
        outputs=[logits_out, boxes_out, passthrough_out],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


# ── openai SDK network-boundary double (CLAUDE.md §1.11) ──────────────────────


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, *, arguments: str) -> None:
    """Patch ``openai.OpenAI`` so ``create()`` returns one ``emit_prompt`` tool call.

    Mirrors the real SDK response graph
    (``response.choices[0].message.tool_calls[0].function.{name,arguments}``).
    """

    def _create(**_kwargs: object) -> SimpleNamespace:
        function = SimpleNamespace(name="emit_prompt", arguments=arguments)
        tool_call = SimpleNamespace(function=function)
        message = SimpleNamespace(tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class _FakeOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

    import openai  # reason: network-boundary double per §1.11

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)


# ── One-prompt ContextRenderer (so the empty-palette short-circuit doesn't fire) ──


def _renderer_with_prompt() -> Any:
    """A ``ContextRenderer`` carrying one prompt."""
    from openral_reasoner import ContextRenderer, PromptRecord

    r = ContextRenderer()
    r.append_prompt(PromptRecord(text="x", metadata_json="", stamp_ns=0))
    return r


# ── CI-script TARGETS=() parser ────────────────────────────────────────────────


def _script_targets(script: Path) -> set[str]:
    """The repo-relative paths inside *script*'s ``TARGETS=( … )`` block."""
    body = script.read_text(encoding="utf-8")
    # Anchor the closing paren to its own line so a parenthesis inside an
    # array comment can never truncate the parse.
    block = re.search(r"TARGETS=\((.*?)^\)$", body, flags=re.DOTALL | re.MULTILINE)
    assert block is not None, f"no TARGETS=() block in {script}"
    return {
        stripped
        for line in block.group(1).splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    }


# ── PEP 735 ``dependency-groups`` include-group expansion ─────────────────────


@pytest.fixture
def expand_dependency_group() -> Callable[..., list[str]]:
    """Factory: recursively expand a PEP 735 ``dependency-groups`` include-group reference.

    Returns ``expand(groups, name) -> list[str]``, flattening
    ``{"include-group": ...}`` entries into the package strings they
    reference (used by the optional-dependency-group regression tests to
    check a lazily-imported extra like ``pyzmq``/``msgpack`` is really
    declared, not just assumed present).
    """

    def _expand(
        groups: dict[str, list[Any]], name: str, _seen: set[str] | None = None
    ) -> list[str]:
        if _seen is None:
            _seen = set()
        if name in _seen:
            return []
        _seen.add(name)
        result: list[str] = []
        for entry in groups.get(name, []):
            if isinstance(entry, dict) and "include-group" in entry:
                result.extend(_expand(groups, entry["include-group"], _seen))
            elif isinstance(entry, str):
                result.append(entry)
        return result

    return _expand
