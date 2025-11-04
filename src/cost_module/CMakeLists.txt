cmake_minimum_required(VERSION 3.8)
project(cost_module)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(interfaces REQUIRED)
find_package(ament_index_cpp REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Include directories
include_directories(include)
#include_directories(${OpenCV_INCLUDE_DIRS})


# # Create the depth_publisher executable
# add_executable(depth_publisher 
#   # src/depth_publisher.cpp
# )

# # Add dependencies for example executable
# ament_target_dependencies(depth_publisher
#   rclcpp
#   sensor_msgs
#   ament_index_cpp
# )

# Install C++ executables
install(TARGETS
  # surface_normal_estimator
  # depth_publisher
  DESTINATION lib/${PROJECT_NAME}
)

# # Install Python scripts
# install(PROGRAMS
#   #scripts/image_saver.py #example script, uncomment if needed
#   DESTINATION lib/${PROJECT_NAME}
# )

##### Install DIRECTORIES #####

# Install launch files (if you have them)
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
  OPTIONAL
)

# Install config files (if you have them)
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
  OPTIONAL
)

# Install model files
install(DIRECTORY
  models/
  DESTINATION share/${PROJECT_NAME}/models
  OPTIONAL
)

ament_package()
