// SPDX-License-Identifier: Apache-2.0
// ADR-0020 PR-F — OTel tracing surface for the C++ safety kernel.
//
// Closes the gap called out at
// python/observability/.../tracing.py:107-111 ("the C++ safety kernel
// will emit a sibling safety.check span via opentelemetry-cpp parented
// to the same rskill.tick via the W3C traceparent carried on
// ActionChunk.msg"). The dashboard ingests these spans on its
// /v1/traces OTLP/HTTP route (default http://localhost:4318) and routes
// them into the Safety card + Identity row (kernel attribute) per
// python/observability/.../dashboard/store.py:591-603 and the
// _IDENTITY_KEYS set at store.py:732-748.

#pragma once

#include <memory>
#include <string>

#include <opentelemetry/context/context.h>
#include <opentelemetry/nostd/shared_ptr.h>
#include <opentelemetry/trace/tracer.h>

namespace openral_safety_kernel::otel {

/// Default OTLP/HTTP endpoint when no env var overrides it. Matches the
/// dashboard's bind port (python/observability/.../dashboard/server.py:29).
inline constexpr const char* kDefaultOtlpHttpEndpoint =
    "http://localhost:4318";

/// Service name resource attribute attached to every span.
inline constexpr const char* kServiceName = "openral_safety_kernel";

/// Span name. Matches the closed-set constant
/// ``openral_observability.semconv.SPAN_SAFETY_CHECK``.
inline constexpr const char* kSafetyCheckSpanName = "safety.check";

/// Closed-set value for ``safety.kernel``, mirroring
/// ``openral_observability.semconv.SAFETY_KERNEL_CPP``. Surfaces in the
/// dashboard's Identity row via the _IDENTITY_KEYS latch.
inline constexpr const char* kSafetyKernelValue = "cpp";

/// Event name fired on a violation span so the dashboard's
/// ``_COUNTED_EVENTS`` set ticks. Mirrors
/// ``openral_observability.semconv.EVENT_SAFETY_VIOLATION``.
inline constexpr const char* kSafetyViolationEventName =
    "openral.event.safety_violation";

/// Install a global ``TracerProvider`` with an OTLP/HTTP exporter +
/// ``BatchSpanProcessor``. Returns ``true`` if the provider was
/// installed, ``false`` if the call was a no-op (already initialised in
/// this process). Reads ``OTEL_EXPORTER_OTLP_ENDPOINT`` for the
/// collector URL; falls back to :data:`kDefaultOtlpHttpEndpoint`.
///
/// Safe to call from any thread; the provider is process-global so
/// re-initialising would only confuse the BatchSpanProcessor's
/// background flush. CLAUDE.md §1.4 (explicit beats implicit): if
/// you want a different endpoint you set the env var before the
/// lifecycle node configures.
bool initialize_tracing(const std::string& service_name = kServiceName);

/// Flush + shut down the global ``TracerProvider`` installed by
/// :func:`initialize_tracing`. Idempotent. Called from
/// ``SafetyKernelLifecycleNode::on_cleanup`` so the BatchSpanProcessor
/// drains its queue before the node releases its resources.
void shutdown_tracing();

/// Return the kernel's ``Tracer`` handle. Always safe to call — when
/// :func:`initialize_tracing` was not run (e.g. in unit tests that
/// don't want network I/O), the returned tracer is the no-op provider's
/// tracer and ``StartSpan`` is essentially free.
opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer> tracer();

/// Parse a W3C ``traceparent`` value (``00-<trace>-<span>-<flags>``)
/// into a :class:`Context` whose active span becomes the parent of any
/// span started inside ``opentelemetry::trace::Scope`` over the
/// returned context. Empty / malformed input returns an empty context
/// — callers that want a span to be a root in that case should pass
/// the returned context unchanged to ``StartSpanOptions::parent``.
///
/// Mirrors ``openral_observability.propagation.extract_traceparent``
/// (python/observability/.../propagation.py) — same propagator, same
/// header names — so spans created here become children of the
/// producer-side ``rskill.tick`` from the runner.
opentelemetry::context::Context extract_parent_context(
    const std::string& traceparent);

}  // namespace openral_safety_kernel::otel
