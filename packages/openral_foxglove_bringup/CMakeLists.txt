cmake_minimum_required(VERSION 3.8)
project(openral_foxglove_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_foxglove_launch test/test_foxglove_launch.py
    TIMEOUT 30
  )
endif()

ament_package()
