cmake_minimum_required(VERSION 3.16)
project(openral_world_state)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11_vendor REQUIRED)
find_package(pybind11 REQUIRED)
find_package(PCL REQUIRED COMPONENTS registration kdtree)

pybind11_add_module(_openral_pcl_registration src/pcl_registration.cpp)
target_compile_features(_openral_pcl_registration PRIVATE cxx_std_17)
target_include_directories(_openral_pcl_registration SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(_openral_pcl_registration PRIVATE ${PCL_LIBRARIES})
ament_get_python_install_dir(PYTHON_INSTALL_DIR)
install(TARGETS _openral_pcl_registration LIBRARY DESTINATION ${PYTHON_INSTALL_DIR})

ament_python_install_package(openral_world_state_ros)

install(PROGRAMS
  scripts/world_state_node
  scripts/object_tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
