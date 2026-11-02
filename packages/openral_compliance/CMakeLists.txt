cmake_minimum_required(VERSION 3.8)
project(openral_compliance)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    openral_compliance/recorder_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_recorder_node test/test_recorder_node.py
    TIMEOUT 60
  )
endif()

ament_package()
