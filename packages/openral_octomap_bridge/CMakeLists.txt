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

# ── Core library (octree → grid; ROS-graph-free, unit-testable) ───────────────

add_library(octree_to_grid src/octree_to_grid.cpp)
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
endif()

ament_package()
