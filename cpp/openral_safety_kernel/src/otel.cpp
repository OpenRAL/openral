// SPDX-License-Identifier: Apache-2.0
// ADR-0020 PR-F — OTel tracing for the C++ safety kernel.

#include "openral_safety_kernel/otel.hpp"

#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include <opentelemetry/context/propagation/global_propagator.h>
#include <opentelemetry/context/propagation/text_map_propagator.h>
#include <opentelemetry/exporters/otlp/otlp_http_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_http_exporter_options.h>
#include <opentelemetry/sdk/resource/resource.h>
#include <opentelemetry/sdk/trace/batch_span_processor_factory.h>
#include <opentelemetry/sdk/trace/batch_span_processor_options.h>
#include <opentelemetry/sdk/trace/tracer_provider_factory.h>
#include <opentelemetry/trace/propagation/http_trace_context.h>
#include <opentelemetry/trace/provider.h>

namespace osk_otel = openral_safety_kernel::otel;
namespace otel_api = opentelemetry;
namespace otel_sdk_trace = opentelemetry::sdk::trace;
namespace otel_sdk_resource = opentelemetry::sdk::resource;
namespace otel_otlp = opentelemetry::exporter::otlp;

namespace {

constexpr const char* kEnvOtlpEndpoint = "OTEL_EXPORTER_OTLP_ENDPOINT";
constexpr const char* kTracerName = "openral_safety_kernel";

std::mutex g_init_mutex;
bool g_initialized = false;

// W3C trace-context carrier backed by a flat map; the
// HttpTraceContext propagator reads ``traceparent`` / ``tracestate``
// keys from this kind of carrier (identical to the Python side which
// goes through opentelemetry's stock TraceContextTextMapPropagator).
class MapCarrier : public otel_api::context::propagation::TextMapCarrier {
public:
  otel_api::nostd::string_view Get(otel_api::nostd::string_view key) const noexcept override {
    const auto it = headers_.find(std::string(key.data(), key.size()));
    if (it == headers_.end()) {
      return {};
    }
    return it->second;
  }

  void Set(otel_api::nostd::string_view key, otel_api::nostd::string_view value) noexcept override {
    headers_[std::string(key.data(), key.size())] = std::string(value.data(), value.size());
  }

  std::map<std::string, std::string> headers_;
};

std::string resolve_endpoint() {
  if (const char* env = std::getenv(kEnvOtlpEndpoint)) {
    if (env[0] != '\0') {
      return std::string(env);
    }
  }
  return std::string(osk_otel::kDefaultOtlpHttpEndpoint);
}

}  // namespace

namespace openral_safety_kernel::otel {

bool initialize_tracing(const std::string& service_name) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_initialized) {
    return false;
  }

  std::string base = resolve_endpoint();
  // OTel HTTP exporter wants the full /v1/traces URL — match what
  // openral_observability._sdk.py:160-164 builds.
  while (!base.empty() && base.back() == '/') {
    base.pop_back();
  }
  const std::string traces_url = base + "/v1/traces";

  otel_otlp::OtlpHttpExporterOptions exporter_opts;
  exporter_opts.url = traces_url;
  // Default OtlpHttpExporterOptions::content_type is binary protobuf,
  // which is what the dashboard's FastAPI ingress expects.

  auto exporter = otel_otlp::OtlpHttpExporterFactory::Create(exporter_opts);

  // BatchSpanProcessor ferries spans off the hot path on a background
  // thread — the validator stays alloc-free (test_no_alloc.cpp still
  // passes). Defaults are fine for the safety chunk rate (≪1 kHz).
  otel_sdk_trace::BatchSpanProcessorOptions bsp_opts;
  auto processor = otel_sdk_trace::BatchSpanProcessorFactory::Create(std::move(exporter), bsp_opts);

  // Resource attributes: service.name latches the dashboard's
  // _service_name field (store.py:680).
  otel_sdk_resource::ResourceAttributes attrs = {
      {"service.name", service_name},
  };
  auto resource = otel_sdk_resource::Resource::Create(attrs);

  auto provider = otel_sdk_trace::TracerProviderFactory::Create(std::move(processor), resource);
  // ``provider`` is unique_ptr; the API's SetTracerProvider wants a
  // nostd::shared_ptr<TracerProvider>.
  otel_api::nostd::shared_ptr<otel_api::trace::TracerProvider> api_provider(provider.release());
  otel_api::trace::Provider::SetTracerProvider(api_provider);

  // Install the W3C TraceContext propagator so traceparent strings
  // injected by the Python runner round-trip through this process.
  otel_api::context::propagation::GlobalTextMapPropagator::SetGlobalPropagator(
      otel_api::nostd::shared_ptr<otel_api::context::propagation::TextMapPropagator>(
          new otel_api::trace::propagation::HttpTraceContext()));

  g_initialized = true;
  return true;
}

void shutdown_tracing() {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (!g_initialized) {
    return;
  }
  auto provider = otel_api::trace::Provider::GetTracerProvider();
  // Down-cast to the SDK provider so we can call ForceFlush + Shutdown.
  auto* sdk_provider = dynamic_cast<otel_sdk_trace::TracerProvider*>(provider.get());
  if (sdk_provider != nullptr) {
    sdk_provider->ForceFlush(std::chrono::microseconds(2'000'000));
    // opentelemetry-cpp 1.16.1 has no-arg Shutdown(); the deadline is
    // already absorbed by the preceding ForceFlush call.
    sdk_provider->Shutdown();
  }
  // Replace with a no-op provider so subsequent tracer() calls don't
  // touch the now-shut-down BatchSpanProcessor.
  otel_api::nostd::shared_ptr<otel_api::trace::TracerProvider> noop;
  otel_api::trace::Provider::SetTracerProvider(noop);
  g_initialized = false;
}

otel_api::nostd::shared_ptr<otel_api::trace::Tracer> tracer() {
  auto provider = otel_api::trace::Provider::GetTracerProvider();
  return provider->GetTracer(kTracerName);
}

otel_api::context::Context extract_parent_context(const std::string& traceparent) {
  if (traceparent.empty()) {
    return otel_api::context::Context{};
  }
  MapCarrier carrier;
  carrier.headers_["traceparent"] = traceparent;
  auto propagator = otel_api::context::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
  // 1.16.1's TextMapPropagator::Extract takes a non-const lvalue ref for
  // the parent context; bind to a named empty Context.
  otel_api::context::Context empty_ctx;
  return propagator->Extract(carrier, empty_ctx);
}

}  // namespace openral_safety_kernel::otel
