cmake_minimum_required(VERSION 3.8)
project(openral_foxglove_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Bucket-2 converter node, installed as a runnable executable so the launch
# file's executable="bucket2_markers" resolves (PROGRAMS pattern, cf. openral_prompt_router).
install(PROGRAMS
  scripts/bucket2_markers
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_foxglove_launch test/test_foxglove_launch.py
    TIMEOUT 30
  )
  # Pure-function unit tests for the Bucket-2 converter.
  ament_add_pytest_test(test_bucket2_markers test/test_bucket2_markers.py
    TIMEOUT 30
  )
  # Hermetic guards for the MCAP recorder launch.
  ament_add_pytest_test(test_record_launch test/test_record_launch.py
    TIMEOUT 30
  )
  # Layout generator — the shipped layout may only reference exposed topics,
  # and no generated layout may contain a write-capable panel.
  ament_add_pytest_test(test_layout test/test_layout.py
    TIMEOUT 30
  )
endif()

ament_package()
