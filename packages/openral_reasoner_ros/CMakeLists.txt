cmake_minimum_required(VERSION 3.8)
project(openral_reasoner_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    openral_reasoner_ros/reasoner_node.py
    openral_reasoner_ros/critic_producer_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_reasoner_node test/test_reasoner_node.py
    TIMEOUT 60
  )
endif()

ament_package()
