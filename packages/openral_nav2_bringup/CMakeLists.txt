cmake_minimum_required(VERSION 3.8)
project(openral_nav2_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# The node modules install AS programs (not setuptools console_scripts) so they
# keep their own `#!/usr/bin/env python3` — same pattern as every other
# node-bearing package (cf. openral_perception_ros).
#
# Both sources MUST be committed mode 100755. `just ros2-build` passes
# --symlink-install, so this rule SYMLINKS them into lib/${PROJECT_NAME}
# rather than copying — and PROGRAMS' "+x on install" only applies to a copy.
# A 0644 source therefore yields a 0644 libexec entry, `shutil.which()` inside
# launch_ros returns None, and launch abandons the WHOLE launch description
# with "executable '…' not found on the libexec directory". Enforced by
# test/test_nav2_launch.py and tests/unit/test_ros_node_exec_bits.py.
install(PROGRAMS
  ${PROJECT_NAME}/payload_scan_filter_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_nav2_launch test/test_nav2_launch.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_payload_scan_filter test/test_payload_scan_filter.py
    TIMEOUT 60
  )
endif()

ament_package()
