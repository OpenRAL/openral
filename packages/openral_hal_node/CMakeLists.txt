cmake_minimum_required(VERSION 3.8)
project(openral_hal_node)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    openral_hal_node/lifecycle_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # One lifecycle case per robot manifest with a sim twin (MuJoCo per case).
  ament_add_pytest_test(test_lifecycle_node test/test_lifecycle_node.py
    TIMEOUT 600
  )
  ament_add_pytest_test(test_lifecycle_estop_safety test/test_lifecycle_estop_safety.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_real_hal_spans test/test_real_hal_spans.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_mobile_base_helpers test/test_mobile_base_helpers.py
    TIMEOUT 60
  )
  # test/test_panda_mobile_sensor_bridge_regression.py is not registered: it
  # builds a RoboCasa kitchen (300 s cold) — run it directly under pytest.
endif()

ament_package()
