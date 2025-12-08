/*
command: ros2 run sync_pkg sync_service_node --ros-args -p fx:=525.0 -p fy:=525.0 -p cx:=319.5 -p cy:=239.5
call service: ros2 service call trigger_sync sync_pkg/srv/TriggerSync
*/
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <geometry_msgs/msg/pose_stamped.hpp>

// Include your custom service (adjust package name as needed)
#include "sync_pkg/srv/trigger_sync.hpp"

#include "depth_to_pointcloud.hpp"

class SyncServiceNode : public rclcpp::Node
{
public:
  SyncServiceNode()
  : Node("sync_service_node")
  {
    // Subscribers using message_filters
    rgb_subscriber_.subscribe(this, "/left_image");
    depth_subscriber_.subscribe(this, "/depth");
    cam_pose_subscriber_.subscribe(this, "/rover_pose");

    // Camera intrinsic parameters
    this->declare_parameter<double>("fx", 525.0);
    this->declare_parameter<double>("fy", 525.0);
    this->declare_parameter<double>("cx", 319.5);
    this->declare_parameter<double>("cy", 239.5);

    // Get camera intrinsic parameters
    intrinsics_.fx = this->get_parameter("fx").as_double();
    intrinsics_.fy = this->get_parameter("fy").as_double();
    intrinsics_.cx = this->get_parameter("cx").as_double();
    intrinsics_.cy = this->get_parameter("cy").as_double();

    // Synchronizer with ApproximateTime policy and queue size = 50
    sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(
      ApproxSyncPolicy(50), rgb_subscriber_, depth_subscriber_, cam_pose_subscriber_);
    sync_->registerCallback(std::bind(&SyncServiceNode::sync_callback, this,
                                      std::placeholders::_1, 
                                      std::placeholders::_2, 
                                      std::placeholders::_3));

    // Publishers for synchronized data
    rgb_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/tamt/sync/rgb", 10);
    depth_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/tamt/sync/depth", 10);
    pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/tamt/sync/pointcloud", 10);
    cam_pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/tamt/sync/rover_pose", 10);

    // Create service
    service_ = this->create_service<sync_pkg::srv::TriggerSync>(
      "trigger_sync",
      std::bind(&SyncServiceNode::handle_service_request, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(),
                "SyncServiceNode started. Service 'trigger_sync' is ready.");
  }

private:
  using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image,
    geometry_msgs::msg::PoseStamped>;

  void sync_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr & cam_pose_msg)
  {
    // Store the latest synchronized messages
    latest_rgb_msg_ = rgb_msg;
    latest_depth_msg_ = depth_msg;
    latest_cam_pose_msg_ = cam_pose_msg;

    // Log timestamps to verify synchronization
    double rgb_time = rgb_msg->header.stamp.sec + rgb_msg->header.stamp.nanosec * 1e-9;
    double depth_time = depth_msg->header.stamp.sec + depth_msg->header.stamp.nanosec * 1e-9;
    double pose_time = cam_pose_msg->header.stamp.sec + cam_pose_msg->header.stamp.nanosec * 1e-9;
    double max_diff = std::max({std::abs(rgb_time - depth_time), 
                                std::abs(rgb_time - pose_time), 
                                std::abs(depth_time - pose_time)});

    RCLCPP_INFO(this->get_logger(), 
                "Sync callback | RGB: %.3f, Depth: %.3f, Pose: %.3f | Max diff: %.3f sec",
                rgb_time, depth_time, pose_time, max_diff);
  }

  void handle_service_request(
    const std::shared_ptr<sync_pkg::srv::TriggerSync::Request> request,
    std::shared_ptr<sync_pkg::srv::TriggerSync::Response> response)
  {
    (void)request; // Request is empty, so we suppress unused parameter warning

    // Check if synchronized data is available
    if (!latest_rgb_msg_ || !latest_depth_msg_ || !latest_cam_pose_msg_) {
      response->success = false;
      response->message = "No synchronized messages available. Waiting for data...";
      RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    try {
      // Publish RGB and Depth
      rgb_publisher_->publish(*latest_rgb_msg_);
      depth_publisher_->publish(*latest_depth_msg_);
      cam_pose_publisher_->publish(*latest_cam_pose_msg_);

      // Convert depth image to point cloud and publish
      auto pointcloud_msg = depthToPointCloud(latest_depth_msg_, intrinsics_, this->get_logger());
      
      if (pointcloud_msg) {
        pointcloud_publisher_->publish(*pointcloud_msg);
// Log timestamps to verify synchronization  -- new section
        const auto& rgb_msg = latest_rgb_msg_;
        const auto& depth_msg = latest_depth_msg_;
        const auto& cam_pose_msg = latest_cam_pose_msg_;
        const auto& pointcloud = pointcloud_msg;

        double rgb_time = rgb_msg->header.stamp.sec + rgb_msg->header.stamp.nanosec * 1e-9;
        double depth_time = depth_msg->header.stamp.sec + depth_msg->header.stamp.nanosec * 1e-9;
        double pose_time = cam_pose_msg->header.stamp.sec + cam_pose_msg->header.stamp.nanosec * 1e-9;
        double pointcloud_time = pointcloud->header.stamp.sec + pointcloud->header.stamp.nanosec * 1e-9;
        
        response->success = true;
        response->message = "Successfully published synchronized data (RGB, Depth, PointCloud, CameraPose).";
        RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
        RCLCPP_INFO(this->get_logger(), 
                "RGB: %.3f, Depth: %.3f, Pose: %.3f | PointCloud: %.3f sec",
                rgb_time, depth_time, pose_time, pointcloud_time);
      } else {
        response->success = false;
        response->message = "Failed to convert depth to point cloud.";
        RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
      }
    } catch (const std::exception& e) {
      response->success = false;
      response->message = std::string("Error publishing synchronized data: ") + e.what();
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
    }

    latest_rgb_msg_.reset();
    latest_depth_msg_.reset();
    latest_cam_pose_msg_.reset();
  }

  // Subscribers and synchronizer
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_subscriber_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_subscriber_;
  message_filters::Subscriber<geometry_msgs::msg::PoseStamped> cam_pose_subscriber_;
  std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr cam_pose_publisher_;

  // Service
  rclcpp::Service<sync_pkg::srv::TriggerSync>::SharedPtr service_;

  // Storage for latest synchronized pair
  sensor_msgs::msg::Image::ConstSharedPtr latest_rgb_msg_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_depth_msg_;
  geometry_msgs::msg::PoseStamped::ConstSharedPtr latest_cam_pose_msg_;

  // Camera intrinsics
  CameraIntrinsics intrinsics_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SyncServiceNode>());
  rclcpp::shutdown();
  return 0;
}