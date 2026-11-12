cmake_minimum_required(VERSION 3.8)
project(openral_hal_unitree)

# Bringup-only package: the Unitree hg HAL node itself is the generic
# `openral_hal_node` (openral_hal.unitree_real:UnitreeHgRealHAL). This installs the
# NVIDIA ros2_control bringup + per-model / per-policy controller configs that
# `robots/g1/robot.yaml` names through `hal.real_bringup`.
find_package(ament_cmake REQUIRED)

# GuardedSafetyController (issue #548) subclasses NVIDIA's SafetyController, which is
# provisioned out-of-tree (deps.repos + the Isaac ROS apt repository). Without it the
# plugin is not built, and a bringup that names it fails to load the controller — it
# never falls back to the unguarded upstream class.
find_package(isaac_ros_deploy_ros2_control QUIET)
if(isaac_ros_deploy_ros2_control_FOUND)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  find_package(controller_interface REQUIRED)
  find_package(pluginlib REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_lifecycle REQUIRED)

  add_library(openral_hal_unitree_guarded_safety_controller SHARED
    src/guarded_safety_controller.cpp
  )
  target_compile_options(openral_hal_unitree_guarded_safety_controller PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
  target_include_directories(openral_hal_unitree_guarded_safety_controller PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(openral_hal_unitree_guarded_safety_controller PUBLIC
    isaac_ros_deploy_ros2_control::isaac_ros_deploy_ros2_control_controllers
  )
  ament_target_dependencies(openral_hal_unitree_guarded_safety_controller PUBLIC
    controller_interface pluginlib rclcpp rclcpp_lifecycle)
  pluginlib_export_plugin_description_file(
    controller_interface openral_hal_unitree_controllers.xml)
  install(TARGETS openral_hal_unitree_guarded_safety_controller
    LIBRARY DESTINATION lib)
  install(DIRECTORY include/ DESTINATION include)
else()
  message(STATUS "isaac_ros_deploy_ros2_control not found: "
    "GuardedSafetyController is not built (real G1/H2 bringup unavailable)")
endif()

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
  if(isaac_ros_deploy_ros2_control_FOUND)
    find_package(ament_cmake_gtest REQUIRED)
    find_package(hardware_interface REQUIRED)
    ament_add_gtest(test_guarded_safety_controller test/test_guarded_safety_controller.cpp)
    target_link_libraries(test_guarded_safety_controller
      openral_hal_unitree_guarded_safety_controller)
    ament_target_dependencies(test_guarded_safety_controller hardware_interface pluginlib)
  endif()
endif()

ament_package()
