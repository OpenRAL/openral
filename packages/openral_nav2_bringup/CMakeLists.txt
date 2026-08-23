cmake_minimum_required(VERSION 3.8)
project(openral_nav2_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  ${PROJECT_NAME}/payload_footprint_node.py
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
  ament_add_pytest_test(test_payload_footprint test/test_payload_footprint.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_payload_scan_filter test/test_payload_scan_filter.py
    TIMEOUT 60
  )
endif()

ament_package()
