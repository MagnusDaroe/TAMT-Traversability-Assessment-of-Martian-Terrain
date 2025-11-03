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

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(publisher_depth src/publisher_depth.cpp)
ament_target_dependencies(publisher_depth
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
)

add_executable(publisher_rgb src/publisher_rgb.cpp)
ament_target_dependencies(publisher_rgb
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
)

add_executable(publisher_sync_data src/sync_data.cpp)
ament_target_dependencies(publisher_sync_data
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  message_filters
)

add_executable(publisher_sync_data_new src/new_sync_node.cpp)
ament_target_dependencies(publisher_sync_data_new
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  message_filters
)

# add_executable(test_node src/test_node.cpp)
# ament_target_dependencies(test_node
#   rclcpp
#   sensor_msgs
#   cv_bridge
#   OpenCV
# )

install(TARGETS
  publisher_depth
  publisher_rgb 
  publisher_sync_data
  publisher_sync_data_new
  # test_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Depth_data directory (source sits under src/) into the package share so it is available at runtime
#install(DIRECTORY src/Depth_data
#  DESTINATION share/${PROJECT_NAME}
#)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/depth
  ${CMAKE_CURRENT_SOURCE_DIR}/data/rgb
  DESTINATION share/${PROJECT_NAME}
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
