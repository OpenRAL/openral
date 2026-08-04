"""Latency metrics are emitted by the span helpers, not by their callers.

`openral.inference.duration` and the two `openral.hal.*.duration`
histograms used to be recorded only inside
`openral_runner.InferenceRunnerBase` / `DeployRunner`. The ROS deploy graph
never instantiates those — `rskill_runner_node` runs its own tick loop and
opens the spans directly — so a live `openral deploy run` produced the
spans and **no latency histograms at all**. Verified on an SO-101 before
the fix: the Metrics panel carried `openral.system.*` and the OTel SDK's
own counters, and not one OpenRAL latency instrument.

Emitting each histogram from the same seam as its span makes them
impossible to diverge: any path that produces the span necessarily
produces the metric.
"""

from __future__ import annotations

from openral_observability import inference_span, semconv
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


def _histogram_counts(reader: InMemoryMetricReader, name: str) -> int:
    data = reader.get_metrics_data()
    total = 0
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == name:
                    for pt in m.data.data_points:
                        total += pt.count
    return total


def test_inference_span_emits_the_duration_histogram(
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """The deploy path opens this span directly — the metric must follow it."""
    for i in range(3):
        with inference_span(chunk_index=i, kind="foreground"):
            pass

    assert _histogram_counts(memory_metric_reader, semconv.METRIC_INFERENCE_DURATION) == 3


def test_inference_metric_is_recorded_even_when_the_body_raises(
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """A failed inference is still an inference that cost time."""
    try:
        with inference_span(chunk_index=0):
            raise RuntimeError("CUDA OOM")
    except RuntimeError:
        pass

    assert _histogram_counts(memory_metric_reader, semconv.METRIC_INFERENCE_DURATION) == 1


def test_inference_metric_labels_stay_closed_set(
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """Only `kind` labels the series — device/engine ride the span.

    Cardinality discipline (design §9): a per-device or per-rskill label
    would fragment the histogram.
    """
    with inference_span(chunk_index=0, kind="prefetch", device="cuda:0", engine="pytorch"):
        pass

    labels: set[str] = set()
    for rm in memory_metric_reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == semconv.METRIC_INFERENCE_DURATION:
                    for pt in m.data.data_points:
                        labels |= set(dict(pt.attributes))
    assert labels == {semconv.LABEL_KIND}
