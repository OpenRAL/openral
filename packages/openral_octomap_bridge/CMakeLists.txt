cmake_minimum_required(VERSION 3.16)
project(openral_octomap_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # No -Werror here: octomap / tf2 third-party headers trip -Wconversion etc.,
  # and this node is not on the real-time safety path (the kernel is).
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(openral_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# ── Core library (octree → grid; ROS-graph-free, unit-testable) ───────────────

add_library(octree_to_grid src/octree_to_grid.cpp src/payload_clearing.cpp src/self_filter.cpp)
target_include_directories(octree_to_grid
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OCTOMAP_INCLUDE_DIRS}
)
target_link_libraries(octree_to_grid PUBLIC ${OCTOMAP_LIBRARIES})
ament_target_dependencies(octree_to_grid PUBLIC openral_msgs tf2)

# ── Bridge node ───────────────────────────────────────────────────────────────

add_executable(octomap_voxel_bridge src/octomap_voxel_bridge_node.cpp)
target_link_libraries(octomap_voxel_bridge PRIVATE octree_to_grid)
ament_target_dependencies(octomap_voxel_bridge
  PUBLIC rclcpp octomap_msgs tf2 tf2_ros tf2_geometry_msgs geometry_msgs openral_msgs)

# ── Install ──────────────────────────────────────────────────────────────────

# Robot (and held-payload) self-filter ahead of octomap_server on a real camera.
add_executable(robot_self_filter src/robot_self_filter_node.cpp)
target_link_libraries(robot_self_filter PRIVATE octree_to_grid)
ament_target_dependencies(robot_self_filter
  PUBLIC rclcpp tf2 tf2_ros tf2_geometry_msgs geometry_msgs sensor_msgs openral_msgs)
install(TARGETS robot_self_filter DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION include)
install(
  TARGETS octree_to_grid
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(TARGETS octomap_voxel_bridge DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(octree_to_grid)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(rclcpp octomap_msgs tf2 openral_msgs)

# ── Tests ────────────────────────────────────────────────────────────────────

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_octree_to_grid test/test_octree_to_grid.cpp)
  target_link_libraries(test_octree_to_grid octree_to_grid)
  ament_add_gtest(test_occupancy_persistence test/test_occupancy_persistence.cpp)
  target_link_libraries(test_occupancy_persistence octree_to_grid)
  ament_add_gtest(test_payload_clearing test/test_payload_clearing.cpp)
  target_link_libraries(test_payload_clearing octree_to_grid)
  # The real node, in-process: its class is the private src/ header.
  # The FK equivalence test links the kernel library: a TEST-only dependency,
  # so the filter's runtime stays in Layer 2 (see self_filter.hpp).
  # The kernel's exported targets pull opentelemetry-cpp, whose config
  # references protobuf::libprotobuf without finding it (same workaround as the
  # kernel's own CMakeLists).
  find_package(Protobuf REQUIRED)
  find_package(openral_safety_kernel REQUIRED)
  # The manifest test reads robots/panda_mobile/robot.yaml straight from the
  # source tree (yaml-cpp, test-only), so it runs on real tight geometry.
  find_package(yaml-cpp REQUIRED)
  ament_add_gtest(test_self_filter test/test_self_filter.cpp)
  target_link_libraries(test_self_filter octree_to_grid yaml-cpp::yaml-cpp)
  target_compile_definitions(test_self_filter
    PRIVATE OPENRAL_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../..")
  ament_target_dependencies(test_self_filter openral_safety_kernel openral_msgs tf2)
  ament_add_gtest(test_robot_self_filter_node test/test_robot_self_filter_node.cpp TIMEOUT 60)
  target_include_directories(test_robot_self_filter_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_robot_self_filter_node octree_to_grid)
  ament_target_dependencies(test_robot_self_filter_node
    rclcpp tf2 tf2_ros tf2_geometry_msgs geometry_msgs sensor_msgs openral_msgs)
  ament_add_gtest(test_bridge_staleness test/test_bridge_staleness.cpp TIMEOUT 60)
  target_include_directories(test_bridge_staleness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_bridge_staleness octree_to_grid)
  ament_target_dependencies(test_bridge_staleness
    rclcpp octomap_msgs tf2 tf2_ros tf2_geometry_msgs geometry_msgs openral_msgs)
  ament_add_gtest(test_bridge_rate_budget test/test_bridge_rate_budget.cpp TIMEOUT 60)
  target_include_directories(test_bridge_rate_budget PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_bridge_rate_budget octree_to_grid)
  ament_target_dependencies(test_bridge_rate_budget
    rclcpp octomap_msgs tf2 tf2_ros tf2_geometry_msgs geometry_msgs openral_msgs)
endif()

ament_package()
