cmake_minimum_required(VERSION 3.8)
project(openral_hal_openarm)

# Bringup-only package: the OpenArm HAL node itself is the generic
# `openral_hal_node`. This installs the vendor ros2_control bringup that
# `robots/openarm/robot.yaml` names through `hal.real_bringup`.
find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch patches
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES deps.repos
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
