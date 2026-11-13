cmake_minimum_required(VERSION 3.8)
project(openral_hal_unitree)

# Bringup-only package: the Unitree hg HAL node itself is the generic
# `openral_hal_node` (openral_hal.unitree_real:UnitreeHgRealHAL). This installs the
# NVIDIA ros2_control bringup + per-model / per-policy controller configs that
# `robots/g1/robot.yaml` names through `hal.real_bringup`.
find_package(ament_cmake REQUIRED)

# GuardedSafetyController (issues #548, #565) and GuardedInferenceController (#565) subclass
# NVIDIA's SafetyController / InferenceController, which are provisioned out-of-tree
# (deps.repos + the Isaac ROS apt repository). Without them the plugins are not built, and a
# bringup that names them fails to load the controllers — it never falls back to the
# unguarded upstream classes. A robot host passes
# -DOPENRAL_REQUIRE_GUARDED_SAFETY_CONTROLLER=ON so a missing (or not-yet-built, e.g. colcon
# ordered it later) NVIDIA package fails the build instead of the bringup.
option(OPENRAL_REQUIRE_GUARDED_SAFETY_CONTROLLER
  "Fail the build when isaac_ros_deploy_ros2_control is missing" OFF)
if(OPENRAL_REQUIRE_GUARDED_SAFETY_CONTROLLER)
  find_package(isaac_ros_deploy_ros2_control REQUIRED)
else()
  find_package(isaac_ros_deploy_ros2_control QUIET)
endif()
if(isaac_ros_deploy_ros2_control_FOUND)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  find_package(controller_interface REQUIRED)
  find_package(pluginlib REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_lifecycle REQUIRED)
  # isaac_ros_deploy v4.6-0 does not export its controllers target's link dependencies.
  find_package(controller_manager_msgs REQUIRED)
  find_package(hardware_interface REQUIRED)
  find_package(isaac_ros_deploy_converters REQUIRED)
  find_package(isaac_ros_deploy_interfaces REQUIRED)
  find_package(realtime_tools REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(std_msgs REQUIRED)

  add_library(openral_hal_unitree_guarded_controllers SHARED
    src/guarded_inference_controller.cpp
    src/guarded_safety_controller.cpp
  )
  target_compile_options(openral_hal_unitree_guarded_controllers PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
  target_include_directories(openral_hal_unitree_guarded_controllers PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(openral_hal_unitree_guarded_controllers PUBLIC
    isaac_ros_deploy_ros2_control::isaac_ros_deploy_ros2_control_controllers
  )
  ament_target_dependencies(openral_hal_unitree_guarded_controllers PUBLIC
    controller_interface pluginlib rclcpp rclcpp_lifecycle)
  pluginlib_export_plugin_description_file(
    controller_interface openral_hal_unitree_controllers.xml)
  install(TARGETS openral_hal_unitree_guarded_controllers
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
      openral_hal_unitree_guarded_controllers)
    ament_target_dependencies(test_guarded_safety_controller hardware_interface pluginlib)
    find_package(geometry_msgs REQUIRED)
    ament_add_gtest_executable(test_guarded_inference_controller
      test/test_guarded_inference_controller.cpp)
    target_link_libraries(test_guarded_inference_controller
      openral_hal_unitree_guarded_controllers)
    ament_target_dependencies(test_guarded_inference_controller
      geometry_msgs hardware_interface pluginlib)
    # One process per policy test: a second in-process Triton server after the first is
    # torn down segfaults (Triton 2.62, seen on q5090), so each test loading the policy
    # gets its own.
    foreach(case
        "units:TopicInput*:GuardedInferenceControllerPlugin.*"
        "upstream:InferenceWatchdogTest.Upstream*"
        "watchdog:InferenceWatchdogTest.NeverReceived*"
        "rearm:InferenceWatchdogTest.AMessageFromBeforeActivation*"
        "refusals:InferenceWatchdogTest.RefusesToConfigure*")
      string(REPLACE ":" ";" parts "${case}")
      list(POP_FRONT parts name)
      list(JOIN parts ":" filter)
      ament_add_gtest_test(test_guarded_inference_controller
        TEST_NAME test_guarded_inference_controller_${name}
        TIMEOUT 300
        ENV "GTEST_FILTER=${filter}")
    endforeach()
  endif()
endif()

ament_package()
