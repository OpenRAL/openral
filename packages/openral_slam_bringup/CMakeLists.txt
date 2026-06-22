cmake_minimum_required(VERSION 3.8)
project(openral_slam_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  openral_slam_bringup/depth_height_filter_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_slam_toolbox_launch test/test_slam_toolbox_launch.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_cuvslam_launch test/test_cuvslam_launch.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_nvblox_launch test/test_nvblox_launch.py
    TIMEOUT 30
  )
  ament_add_pytest_test(test_depth_height_filter test/test_depth_height_filter.py
    TIMEOUT 30
  )
endif()

ament_package()
