cmake_minimum_required(VERSION 3.8)
project(sync_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(interfaces REQUIRED)

# After all find_package calls, add:
include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
add_executable(publish_raw_data src/raw_data_publisher/raw_data_publisher.cpp)
ament_target_dependencies(publish_raw_data
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  geometry_msgs
  OpenCV
)

# This node provides a service to synchronise data and publish synchronised data topic
add_executable(data_synchroniser src/synchronised_data_publisher/sync_service_node.cpp)
ament_target_dependencies(data_synchroniser
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  message_filters
  OpenCV
  geometry_msgs
  interfaces
)

# This node is a client to trigger the synchronisation service request
add_executable(data_synchroniser_client src/synchronised_data_publisher/sync_client_node.cpp)
ament_target_dependencies(data_synchroniser_client
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  message_filters
  OpenCV
  geometry_msgs
  interfaces
)

install(TARGETS
  publish_raw_data
  data_synchroniser
  data_synchroniser_client
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

# Install Depth_data directory (source sits under src/) into the package share so it is available at runtime
#install(DIRECTORY src/Depth_data
#  DESTINATION share/${PROJECT_NAME}
#)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/
  DESTINATION share/${PROJECT_NAME}/data
  USE_SOURCE_PERMISSIONS
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/frame_data/
  DESTINATION share/${PROJECT_NAME}/frame_data
  USE_SOURCE_PERMISSIONS
)

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

ament_package()
