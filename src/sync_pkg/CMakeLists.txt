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

include_directories(include)

# Generate service files
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/TriggerSync.srv"
  DEPENDENCIES sensor_msgs geometry_msgs
)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(publisher_depth src/input_publisher_nodes/publisher_depth.cpp)
ament_target_dependencies(publisher_depth
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
)

add_executable(publisher_rgb src/input_publisher_nodes/publisher_rgb.cpp)
ament_target_dependencies(publisher_rgb
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
)

add_executable(publisher_campose src/input_publisher_nodes/publisher_cam_pose.cpp)
ament_target_dependencies(publisher_campose
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  geometry_msgs
)

add_executable(publisher_all_data src/input_publisher_nodes/publisher_all_data.cpp)
ament_target_dependencies(publisher_all_data
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  geometry_msgs
  OpenCV
)

add_executable(publisher_sync_data src/sync_publisher_nodes/sync_node.cpp)
ament_target_dependencies(publisher_sync_data
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  message_filters
)

add_executable(test_publisher_sync_data src/sync_publisher_nodes/test_sync_node.cpp)
ament_target_dependencies(test_publisher_sync_data
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  message_filters
  OpenCV
  geometry_msgs
)

add_executable(sanity_test src/sanity_tester/sanity_test.cpp)
ament_target_dependencies(sanity_test
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  message_filters
  OpenCV
  geometry_msgs
)

add_executable(sync_service_node src/sync_service_nodes/sync_service_node.cpp)
ament_target_dependencies(sync_service_node
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  message_filters
  OpenCV
  geometry_msgs
)

add_executable(sync_client_node src/sync_service_nodes/sync_client_node.cpp)
ament_target_dependencies(sync_client_node
  rclcpp
  sensor_msgs
  ament_index_cpp
  cv_bridge
  message_filters
  OpenCV
  geometry_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(sync_service_node "${cpp_typesupport_target}")
target_link_libraries(sync_client_node "${cpp_typesupport_target}")

install(TARGETS
  publisher_depth
  publisher_rgb 
  publisher_campose
  publisher_sync_data
  publisher_all_data
  sanity_test
  test_publisher_sync_data
  sync_service_node
  sync_client_node
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
