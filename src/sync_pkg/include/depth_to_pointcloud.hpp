#ifndef DEPTH_TO_POINTCLOUD_HPP
#define DEPTH_TO_POINTCLOUD_HPP

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <rclcpp/rclcpp.hpp>

struct CameraIntrinsics {
  double fx;  // Focal length x
  double fy;  // Focal length y
  double cx;  // Principal point x
  double cy;  // Principal point y
};

/**
 * @brief Convert depth image to PointCloud2
 * 
 * @param depth_msg Input depth image message
 * @param intrinsics Camera intrinsic parameters
 * @param logger ROS logger for error messages
 * @return sensor_msgs::msg::PointCloud2::SharedPtr Point cloud message
 */
sensor_msgs::msg::PointCloud2::SharedPtr depthToPointCloud(
  const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
  const CameraIntrinsics& intrinsics,
  const rclcpp::Logger& logger)
{
  auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  
  // Copy header from depth image
  cloud_msg->header = depth_msg->header;
  
  // Set up point cloud metadata
  cloud_msg->height = depth_msg->height;
  cloud_msg->width = depth_msg->width;
  cloud_msg->is_dense = false;
  cloud_msg->is_bigendian = false;
  
  // Manually define point fields with FLOAT32 (type 7)
  cloud_msg->fields.resize(3);
  
  // X field
  cloud_msg->fields[0].name = "x";
  cloud_msg->fields[0].offset = 0;
  cloud_msg->fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg->fields[0].count = 1;
  
  // Y field
  cloud_msg->fields[1].name = "y";
  cloud_msg->fields[1].offset = 4;
  cloud_msg->fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg->fields[1].count = 1;
  
  // Z field
  cloud_msg->fields[2].name = "z";
  cloud_msg->fields[2].offset = 8;
  cloud_msg->fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg->fields[2].count = 1;
  
  // Set point step and row step
  cloud_msg->point_step = 12; // 3 fields * 4 bytes (FLOAT32)
  cloud_msg->row_step = cloud_msg->point_step * cloud_msg->width;
  
  // Resize data buffer
  cloud_msg->data.resize(cloud_msg->row_step * cloud_msg->height);
  
  // Create iterators for x, y, z
  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
  
  // Determine depth encoding and scale factor
  float depth_scale = 1.0f;
  bool is_float = false;
  
  if (depth_msg->encoding == "32FC1") {
    is_float = true;
    depth_scale = 1.0f;  // Depth in meters
  } else if (depth_msg->encoding == "16UC1") {
    is_float = false;
    depth_scale = 0.001f;  // Depth in millimeters, convert to meters
  } else {
    RCLCPP_ERROR(logger, "Unsupported depth encoding: %s", depth_msg->encoding.c_str());
    return nullptr;
  }
  
  // Convert depth to point cloud
  const uint8_t* depth_data = depth_msg->data.data();
  
  for (uint32_t v = 0; v < depth_msg->height; ++v) {
    for (uint32_t u = 0; u < depth_msg->width; ++u) {
      float depth;
      
      if (is_float) {
        depth = reinterpret_cast<const float*>(depth_data)[v * depth_msg->width + u];
      } else {
        depth = static_cast<float>(
          reinterpret_cast<const uint16_t*>(depth_data)[v * depth_msg->width + u]
        ) * depth_scale;
      }
      
      // Skip invalid depth values
      if (depth <= 0.0f || std::isnan(depth) || std::isinf(depth)) {
        *iter_x = std::numeric_limits<float>::quiet_NaN();
        *iter_y = std::numeric_limits<float>::quiet_NaN();
        *iter_z = std::numeric_limits<float>::quiet_NaN();
      } else {
        // Back-project to 3D using pinhole camera model
        *iter_z = depth;
        *iter_x = (u - intrinsics.cx) * depth / intrinsics.fx;
        *iter_y = (v - intrinsics.cy) * depth / intrinsics.fy;
      }
      
      ++iter_x;
      ++iter_y;
      ++iter_z;
    }
  }
  
  return cloud_msg;
}

#endif // DEPTH_TO_POINTCLOUD_HPP