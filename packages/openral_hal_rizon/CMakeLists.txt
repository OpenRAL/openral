cmake_minimum_required(VERSION 3.8)
project(openral_hal_rizon)

# Bringup-only package: the HAL node is the generic `openral_hal_node`. Installs the
# ros2_control bringup `robots/rizon_dual/robot.yaml` names through `hal.real_bringup`.
find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config urdf DESTINATION share/${PROJECT_NAME})
install(FILES deps.repos DESTINATION share/${PROJECT_NAME})

ament_package()
