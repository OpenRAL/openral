cmake_minimum_required(VERSION 3.8)
project(openral_safety)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    scripts/supervisor_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # The real integration test under test/test_supervisor_node.py runs
  # via ``uv run pytest`` (it depends on the openral_observability /
  # openral_msgs / structlog stack on the workspace venv); the colcon
  # test environment cannot import those. Keep the file in-tree as the
  # canonical contract test, but don't wire it into ``colcon test``.

  # The AST-only structural guard (no ROS runtime needed) is safe to wire
  # into colcon test — it only uses stdlib ``ast`` + pytest.
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_supervisor_node_sigint_shape
    test/test_supervisor_node_sigint_shape.py
    TIMEOUT 30
  )
endif()

ament_package()
