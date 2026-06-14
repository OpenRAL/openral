cmake_minimum_required(VERSION 3.16)
project(opentelemetry_cpp_vendor)

# ROS 2 vendor package for opentelemetry-cpp.
#
# Ubuntu 24.04 (Noble) does not ship opentelemetry-cpp in the default
# apt repos, and we need at least the trace API + OTLP HTTP exporter
# to close ADR-0020 §9 (PR-F). Following the conventional ROS 2 pattern
# (cf. nlohmann_json_vendor, fastcdr_vendor), this package fetches and
# builds opentelemetry-cpp at colcon build time, then re-exports its
# CMake config to dependent packages.
#
# Dependents do NOT call ExternalProject_Add — they call
# ``find_package(opentelemetry-cpp CONFIG REQUIRED)`` after this package
# has built and installed its artefacts to the colcon prefix.

find_package(ament_cmake REQUIRED)

include(ExternalProject)

# Pinned release; bump deliberately. Match the latest stable as of the
# ADR-0020 PR-F landing (v1.16.x line — first OTLP HTTP series the
# CMake config exports without protobuf-compile contortions on GCC 13).
set(OTEL_CPP_VERSION "1.16.1")
set(OTEL_CPP_GIT_TAG "v${OTEL_CPP_VERSION}")

# Build only what we use. The safety kernel emits TRACES over OTLP/HTTP;
# we explicitly disable gRPC (extra abseil + protoc dep), metrics, zPages,
# OTLP-File, and tests/examples to keep the build under 90 s on a
# 4-core host. Dependents that need metrics/logs over OTLP would extend
# this package; for the safety kernel they go through the ROS topic
# /openral/failure/safety + /diagnostics instead.
#
# Note on CMAKE_POLICY_VERSION_MINIMUM=3.10:
# opentelemetry-cpp 1.16.1's top-level CMakeLists declares
# ``cmake_minimum_required(VERSION 3.8)`` which CMake >= 3.27 now warns is
# below the 3.10 deprecation floor. We don't control upstream's pin, so
# bump the effective policy floor at our ExternalProject boundary —
# silences the deprecation noise without disabling other warnings.
# WITH_DEPRECATED_SDK_FACTORY=OFF: upstream 1.16.x defaults it ON and
# emits a stderr warning unless explicitly disabled; we don't use the
# deprecated factory path, so opt out and drop the warning.
# WITH_LOGS_PREVIEW and WITH_JAEGER were removed upstream (logs signal is
# stable; Jaeger exporter deprecated in 1.13.x). Passing them now triggers
# "Manually-specified variables were not used by the project" warnings.
set(OTEL_CPP_CMAKE_ARGS
  -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_POLICY_VERSION_MINIMUM=3.10
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DBUILD_SHARED_LIBS=ON
  -DBUILD_TESTING=OFF
  -DWITH_EXAMPLES=OFF
  -DWITH_FUNC_TESTS=OFF
  -DWITH_BENCHMARK=OFF
  -DWITH_OTLP_HTTP=ON
  -DWITH_OTLP_GRPC=OFF
  -DWITH_OTLP_FILE=OFF
  -DWITH_ZIPKIN=OFF
  -DWITH_ELASTICSEARCH=OFF
  -DWITH_PROMETHEUS=OFF
  -DWITH_DEPRECATED_SDK_FACTORY=OFF
  -DWITH_ABSEIL=OFF
  -DCMAKE_CXX_STANDARD=17
)

ExternalProject_Add(opentelemetry_cpp_external
  GIT_REPOSITORY https://github.com/open-telemetry/opentelemetry-cpp.git
  GIT_TAG ${OTEL_CPP_GIT_TAG}
  GIT_SHALLOW ON
  GIT_SUBMODULES_RECURSE ON
  CMAKE_ARGS ${OTEL_CPP_CMAKE_ARGS}
  # Install into the colcon prefix so find_package locates the config.
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  # Route each step's output to a log file under the build tree instead
  # of leaking through colcon as stderr. The git clone step in particular
  # writes "Cloning into ..." / "Submodule 'x' registered" lines to
  # stderr by design, which colcon flags as "1 package had stderr
  # output" even though nothing went wrong. Logs are still available at
  # ``build/opentelemetry_cpp_vendor/opentelemetry_cpp_external-prefix/
  # src/opentelemetry_cpp_external-stamp/*.log`` for debugging.
  LOG_DOWNLOAD ON
  LOG_UPDATE ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  BUILD_BYPRODUCTS
    "${CMAKE_INSTALL_PREFIX}/lib/libopentelemetry_trace.so"
    "${CMAKE_INSTALL_PREFIX}/lib/libopentelemetry_common.so"
    "${CMAKE_INSTALL_PREFIX}/lib/libopentelemetry_resources.so"
    "${CMAKE_INSTALL_PREFIX}/lib/libopentelemetry_exporter_otlp_http.so"
)

# Re-export the find_package result. Dependents add a build-time
# dependency on this package via package.xml; ament_cmake then prepends
# our INSTALL_PREFIX to CMAKE_PREFIX_PATH so the line below in their
# CMakeLists resolves to the artefacts we just built.
ament_package()
