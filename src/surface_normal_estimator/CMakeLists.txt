cmake_minimum_required(VERSION 3.5)
project(surface_normal_estimator)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(interfaces REQUIRED)
find_package(ament_index_cpp REQUIRED)
#find_package(OpenCV REQUIRED)

# Include directories
include_directories(include)
#include_directories(${OpenCV_INCLUDE_DIRS})

# Create the C++ executable
add_executable(surface_normal_estimator 
  src/surface_normal_estimator.cpp
)

# Add dependencies for surface_normal_estimator executable
ament_target_dependencies(surface_normal_estimator
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
)

# Create the depth_publisher executable
add_executable(depth_publisher 
  src/depth_publisher.cpp
)

# Add dependencies for depth_publisher executable
ament_target_dependencies(depth_publisher
  rclcpp
  sensor_msgs
  ament_index_cpp
)

# Link OpenCV libraries
target_link_libraries(surface_normal_estimator
  ${OpenCV_LIBRARIES}
)

# Install C++ executables
install(TARGETS
  surface_normal_estimator
  depth_publisher
  DESTINATION lib/${PROJECT_NAME}
)

# Install DIODE dataset separately
install(DIRECTORY
  DIODE_dataset
  DESTINATION share/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

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

# Mark the package as complete
ament_package()