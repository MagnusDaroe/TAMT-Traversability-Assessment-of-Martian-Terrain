#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "depth_to_pointcloud.hpp"
#include "sync_pkg/srv/trigger_sync.hpp"

class SyncServicePublisherNode : public rclcpp::Node
{
public:
  SyncServicePublisherNode()
  : Node("sync_service_publisher_node")
  {
    // --- Declare and retrieve camera intrinsics ---
    this->declare_parameter<double>("fx", 525.0);
    this->declare_parameter<double>("fy", 525.0);
    this->declare_parameter<double>("cx", 319.5);
    this->declare_parameter<double>("cy", 239.5);

    intrinsics_.fx = this->get_parameter("fx").as_double();
    intrinsics_.fy = this->get_parameter("fy").as_double();
    intrinsics_.cx = this->get_parameter("cx").as_double();
    intrinsics_.cy = this->get_parameter("cy").as_double();

    // --- Message filter subscribers ---
    rgb_sub_.subscribe(this, "/left_image");
    depth_sub_.subscribe(this, "/depth");
    pose_sub_.subscribe(this, "/camera_pose");

    // --- Synchronizer (ApproximateTime) ---
    sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(
      10, rgb_sub_, depth_sub_, pose_sub_);
    sync_->registerCallback(
      std::bind(&SyncServicePublisherNode::sync_callback, this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3));

    // --- Publishers for synchronized data ---
    rgb_publisher_   = this->create_publisher<sensor_msgs::msg::Image>("/sync_rgb", 10);
    depth_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/sync_depth", 10);
    pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/sync_pointcloud", 10);
    cam_pose_publisher_   = this->create_publisher<geometry_msgs::msg::PoseStamped>("/sync_cam_2_glob_pose", 10);

    // --- Service for triggering publish --- with the name "trigger_sync"
    trigger_service_ = this->create_service<sync_pkg::srv::TriggerSync>(
      "trigger_sync",
      std::bind(&SyncServicePublisherNode::handle_trigger, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(),
                "SyncServicePublisherNode started. Waiting for trigger requests...");
  }

private:
  using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image,
    geometry_msgs::msg::PoseStamped>;

  // --- Synchronization callback: store latest synchronized messages ---
  void sync_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr & cam_pose_msg)
  {
    latest_rgb_msg_   = rgb_msg;
    latest_depth_msg_ = depth_msg;
    latest_cam_pose_msg_ = cam_pose_msg;
    
    RCLCPP_INFO(this->get_logger(), 
                "Sync callback: New synchronized data received at time %.3f",
                rgb_msg->header.stamp.sec + rgb_msg->header.stamp.nanosec * 1e-9);
  }

  // --- Service callback: publishes synchronized data on demand ---
  void handle_trigger(
    const std::shared_ptr<sync_pkg::srv::TriggerSync::Request> /*request*/,
    std::shared_ptr<sync_pkg::srv::TriggerSync::Response> response)
  {
    if (!latest_rgb_msg_ || !latest_depth_msg_ || !latest_cam_pose_msg_) {
      response->success = false;
      response->message = "No synchronized data available yet.";
      RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    // Publish the synchronized messages
    rgb_publisher_->publish(*latest_rgb_msg_);
    depth_publisher_->publish(*latest_depth_msg_);
    cam_pose_publisher_->publish(*latest_cam_pose_msg_);

    // Convert depth to point cloud
    auto pointcloud_msg = depthToPointCloud(latest_depth_msg_, intrinsics_, this->get_logger());
    if (pointcloud_msg) {
      pointcloud_publisher_->publish(*pointcloud_msg);
      response->success = true;
      
      // Log the timestamp of published data for debugging
      double timestamp = latest_rgb_msg_->header.stamp.sec + 
                        latest_rgb_msg_->header.stamp.nanosec * 1e-9;
      response->message = "Published synchronized data with timestamp: " + 
                         std::to_string(timestamp);
      RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
      
      // Clear the cached messages so next trigger waits for new synchronized data
      latest_rgb_msg_.reset();
      latest_depth_msg_.reset();
      latest_cam_pose_msg_.reset();
    } else {
      response->success = false;
      response->message = "Failed to generate point cloud.";
      RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
    }
  }

  // --- Subscribers and Synchronizer ---
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  message_filters::Subscriber<geometry_msgs::msg::PoseStamped> pose_sub_;
  std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

  // --- Publishers ---
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr cam_pose_publisher_;

  // --- Latest synchronized messages ---
  sensor_msgs::msg::Image::ConstSharedPtr latest_rgb_msg_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_depth_msg_;
  geometry_msgs::msg::PoseStamped::ConstSharedPtr latest_cam_pose_msg_;

  // --- Service ---
  rclcpp::Service<sync_pkg::srv::TriggerSync>::SharedPtr trigger_service_;

  // --- Camera intrinsics ---
  CameraIntrinsics intrinsics_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SyncServicePublisherNode>());
  rclcpp::shutdown();
  return 0;
}
