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
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(interfaces REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)


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

# Create the costmap_sne executable
add_executable(costmap_sne 
  src/costmap_sne.cpp
)

# Add dependencies for costmap_sne executable
ament_target_dependencies(costmap_sne
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  nav2_msgs
  tf2
  tf2_geometry_msgs
)

# Install C++ executables
install(TARGETS
  costmap_sne
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
