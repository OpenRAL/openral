cmake_minimum_required(VERSION 3.8)
project(openral_foxglove_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# ADR-0059 Phase 3 — the Bucket-2 converter node (custom msgs -> standard
# viz types). Installed as a runnable executable so the launch file's
# executable="bucket2_markers" resolves (matches the repo's PROGRAMS pattern,
# cf. openral_prompt_router).
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
  # ADR-0059 Phase 3 — pure-function unit tests for the Bucket-2 converter.
  ament_add_pytest_test(test_bucket2_markers test/test_bucket2_markers.py
    TIMEOUT 30
  )
  # ADR-0059 Phase 4 — hermetic guards for the MCAP recorder launch.
  ament_add_pytest_test(test_record_launch test/test_record_launch.py
    TIMEOUT 30
  )
endif()

ament_package()
