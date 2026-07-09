cmake_minimum_required(VERSION 3.8)
project(openral_human_estop)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    openral_human_estop/forwarder_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_forwarder_node test/test_forwarder_node.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_forwarder_node_sigint_shape
    test/test_forwarder_node_sigint_shape.py
    TIMEOUT 30
  )
endif()

ament_package()
