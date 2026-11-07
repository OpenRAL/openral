cmake_minimum_required(VERSION 3.8)
project(openral_hal_unitree)

# Bringup-only package: the Unitree hg HAL node itself is the generic
# `openral_hal_node` (openral_hal.unitree_real:UnitreeHgRealHAL). This installs the
# NVIDIA ros2_control bringup + per-model / per-policy controller configs that
# `robots/g1/robot.yaml` names through `hal.real_bringup`.
find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
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
