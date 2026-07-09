cmake_minimum_required(VERSION 3.8)
project(openral_safety_watchdog)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    openral_safety_watchdog/deadman_watchdog_node.py
    openral_safety_watchdog/hardware_estop_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_deadman_watchdog test/test_deadman_watchdog.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_deadman_watchdog_node_sigint_shape
    test/test_deadman_watchdog_node_sigint_shape.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_hardware_estop_node_sigint_shape
    test/test_hardware_estop_node_sigint_shape.py
    TIMEOUT 30
  )
endif()

ament_package()
