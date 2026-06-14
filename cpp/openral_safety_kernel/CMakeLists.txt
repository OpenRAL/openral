cmake_minimum_required(VERSION 3.16)
project(openral_safety_kernel CXX)

# ADR-0020 / CLAUDE.md §5.2 — C++17 minimum; prefer C++20 where the
# distro's compiler supports it. We keep 17 here so the package builds
# on ROS 2 Humble (Ubuntu 22.04, GCC 11) and Jazzy (Ubuntu 24.04, GCC 13)
# without conditional flags. The validator itself is C++17-clean.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  # Release-with-debinfo is the right default for a real-time path: -O2
  # but with line numbers so /diagnostics and the LTTng CTF dumps point at
  # readable source. Override with `colcon build --cmake-args
  # -DCMAKE_BUILD_TYPE=Debug` when iterating on the validator.
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Strict warnings on the hot path. -Werror is intentionally NOT set
# globally so a distro-quirk warning does not block a downstream build;
# the project's own CI flips it on with `-DOPENRAL_SAFETY_KERNEL_WERROR=ON`.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
  )
endif()
if(OPENRAL_SAFETY_KERNEL_WERROR)
  add_compile_options(-Werror)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(openral_msgs REQUIRED)
# nlohmann_json: distro package on Ubuntu 22.04+; the dependency is
# header-only so we tolerate it being optional behind a feature flag
# until ADR-0020 PR-E ships FailureTrigger emission.
find_package(nlohmann_json QUIET)

# ADR-0020 PR-F — OTel trace emission. opentelemetry-cpp comes from the
# in-tree vendor package (cpp/opentelemetry_cpp_vendor) which fetches
# and builds the upstream library at colcon-build time; CMake's
# CONFIG-mode find_package finds the installed cmake/ files under the
# colcon workspace prefix. The components match the build flags set
# in the vendor: trace API + OTLP HTTP exporter + SDK only.
# opentelemetry-cpp 1.16.1's exported targets reference protobuf::libprotobuf
# (OTLP HTTP serializes spans via protobuf) but its config does not
# find_dependency(Protobuf), so we import it here before the CONFIG load.
find_package(Protobuf REQUIRED)
find_package(opentelemetry-cpp CONFIG REQUIRED)

# ── Library ──────────────────────────────────────────────────────────────────

add_library(${PROJECT_NAME}
  src/envelope.cpp
  src/validator.cpp
  src/collision.cpp
  src/lifecycle_kernel.cpp
  src/otel.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OPENTELEMETRY_CPP_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    # opentelemetry-cpp CONFIG package exports namespaced targets; we
    # pull the trace SDK + OTLP HTTP exporter so the kernel can build
    # a TracerProvider and ship safety.check spans to port 4318.
    opentelemetry-cpp::api
    opentelemetry-cpp::sdk
    opentelemetry-cpp::trace
    opentelemetry-cpp::resources
    opentelemetry-cpp::otlp_http_exporter
)
ament_target_dependencies(${PROJECT_NAME}
  PUBLIC
    rclcpp
    rclcpp_lifecycle
    lifecycle_msgs
    std_msgs
    std_srvs
    diagnostic_msgs
    openral_msgs
)

# ── Executable ──────────────────────────────────────────────────────────────

add_executable(safety_kernel_node src/safety_kernel_node.cpp)
target_link_libraries(safety_kernel_node PRIVATE ${PROJECT_NAME})
ament_target_dependencies(safety_kernel_node PUBLIC rclcpp rclcpp_lifecycle)

# ── Install ──────────────────────────────────────────────────────────────────

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(
  TARGETS safety_kernel_node
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(
  rclcpp rclcpp_lifecycle lifecycle_msgs std_msgs std_srvs
  diagnostic_msgs openral_msgs opentelemetry-cpp
)

# ── Tests ────────────────────────────────────────────────────────────────────

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # CLAUDE.md §1.11 forbids mocks; gtest fixtures consume real
  # RobotDescription / SafetyEnvelope structs constructed in-test.
  #
  # `ament_lint_auto` / `ament_lint_common` are intentionally NOT wired
  # in here. Their default linters (ament_flake8 single-quote rule,
  # ament_cpplint legal/copyright + Google include-order, ament_uncrustify
  # Allman braces, ament_pep257 D213) directly conflict with the
  # workspace's own style discipline — ruff (double quotes), Google-style
  # docstrings (summary on line 1), clang-format LLVM style (K&R braces)
  # — codified in CLAUDE.md §5.1/§5.2 and applied uniformly across every
  # other ROS 2 package in `packages/`. Wiring ament_lint_common only
  # here produced ~259 style failures in `just ros2-test` that the rest
  # of the repo doesn't enforce; ruff + clang-tidy + mypy remain the
  # sole style authorities, run via `just lint`.

  ament_add_gtest(test_validator test/test_validator.cpp)
  target_link_libraries(test_validator ${PROJECT_NAME})

  ament_add_gtest(test_collision test/test_collision.cpp)
  target_link_libraries(test_collision ${PROJECT_NAME})

  ament_add_gtest(test_no_alloc test/test_no_alloc.cpp)
  target_link_libraries(test_no_alloc ${PROJECT_NAME})

  ament_add_gtest(test_lifecycle_kernel test/test_lifecycle_kernel.cpp)
  target_link_libraries(test_lifecycle_kernel ${PROJECT_NAME})
endif()

ament_package()
