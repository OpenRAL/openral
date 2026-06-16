cmake_minimum_required(VERSION 3.8)
project(openral_perception_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Install the node module AS a program (not a setuptools console_script):
# colcon regenerates console_scripts with a hardcoded ``#!/usr/bin/python3``
# (the system interpreter, which lacks the openral_runner workspace package),
# whereas install(PROGRAMS ...) copies the file verbatim, preserving its
# ``#!/usr/bin/env python3`` shebang that PATH-resolves to the .venv
# interpreter the rest of the graph runs under. Same pattern as every other
# node-bearing package (e.g. openral_hal_panda_mobile).
install(
  PROGRAMS
    openral_perception_ros/ros_image_detector_node.py
    openral_perception_ros/scene_vlm_node.py
    openral_perception_ros/reward_monitor_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
