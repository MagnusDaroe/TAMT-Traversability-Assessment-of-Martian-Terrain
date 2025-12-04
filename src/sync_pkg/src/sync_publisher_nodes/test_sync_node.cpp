/*
command: ros2 run sync_pkg test_publisher_sync_data --ros-args -p publish_interval_sec:=2.0
(make sure that the publishing interval is longer than the subscription delay to see the effect)
*/
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include "depth_to_pointcloud.hpp"

class ApproxSyncNode : public rclcpp::Node
{
public:
  ApproxSyncNode()
  : Node("test_sync_node")
  {
    // Declare configurable parameter (default = 1.0 second)
    this->declare_parameter<double>("publish_interval_sec", 1.0);
    publish_interval_sec_ = this->get_parameter("publish_interval_sec").as_double();

    // Subscribers using message_filters
    rgb_subscriber_.subscribe(this, "/left_image");
    depth_subscriber_.subscribe(this, "/depth");
    // Subscriber for camera pose
    cam_pose_subscriber_.subscribe(this, "/camera_pose");

    //Camera intrinsic parameters
    this->declare_parameter<double>("fx", 525.0);
    this->declare_parameter<double>("fy", 525.0);
    this->declare_parameter<double>("cx", 319.5);
    this->declare_parameter<double>("cy", 239.5);

    // Get camera intrinsic parameters
    intrinsics_.fx = this->get_parameter("fx").as_double();
    intrinsics_.fy = this->get_parameter("fy").as_double();
    intrinsics_.cx = this->get_parameter("cx").as_double();
    intrinsics_.cy = this->get_parameter("cy").as_double();

    // Synchronizer with ApproximateTime policy and queue size = 10
    sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(10, rgb_subscriber_, depth_subscriber_, cam_pose_subscriber_);
    sync_->registerCallback(std::bind(&ApproxSyncNode::sync_callback, this,
                                      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    // Create a timer to configure the periodic publishing
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(publish_interval_sec_), 
      std::bind(&ApproxSyncNode::timer_callback, this));

    // Publishers for synchronized data
    rgb_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/sync_rgb", 10);
    depth_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/sync_depth", 10);
    // Added publisher for PointCloud2 -> topic: /sync_pointcloud
    pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/sync_pointcloud", 10);
    // Publisher for camera pose -> topic: /sync_cam_2_glob_pose
    cam_pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/sync_cam_2_glob_pose", 10);


    last_publish_time_ = this->now();

    RCLCPP_INFO(this->get_logger(),
                "ApproxSyncNode started (publish interval = %.2f sec)",
                publish_interval_sec_);
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
    //Subscription timing debugger:
    //rclcpp::Time t_rgb(rgb_msg->header.stamp);
    //rclcpp::Time t_depth(depth_msg->header.stamp);
    //double diff_ms = std::abs((t_rgb - t_depth).seconds() * 1000.0); // milliseconds
    //RCLCPP_INFO(this->get_logger(),
    //            "Synchronized pair received | RGB time: %.3f, Depth time: %.3f, Diff: %.3f ms",
    //            t_rgb.seconds(), t_depth.seconds(), diff_ms);

    // Store the latest synchronized messages
    latest_rgb_msg_ = rgb_msg;
    latest_depth_msg_ = depth_msg;
    latest_cam_pose_msg_ = cam_pose_msg;
  }

  /*void timer_callback()
  {
    // This function will be called periodically based on the timer
    // Here we can implement logic to publish the latest synchronized messages
    if (latest_rgb_msg_ && latest_depth_msg_) {
      rgb_publisher_->publish(*latest_rgb_msg_);
      depth_publisher_->publish(*latest_depth_msg_);

    //Publisher Timing debugger:
    {
        rclcpp::Time now = this->now();
        double delta_ms = (now - last_publish_time_).seconds() * 1000.0;
        RCLCPP_INFO(this->get_logger(),
                                "Time since previous published pair: %.3f ms", delta_ms);
        last_publish_time_ = now;
    }

    RCLCPP_INFO(this->get_logger(),
                "Published synchronized pair from timer callback.");
    } else {
      RCLCPP_WARN(this->get_logger(),
                  "No synchronized messages available to publish.");
    }
  }*/

  void timer_callback()
{
  static bool first_call = true;
  rclcpp::Time now = this->now();

  // Skip delta computation on the very first call
  if (!first_call) {
    double elapsed_sec = (now - last_publish_time_).seconds();
    double jitter_ms = (elapsed_sec - publish_interval_sec_) * 1000.0;

    RCLCPP_INFO(this->get_logger(),
                "Timer expected period: %.3f sec | Actual elapsed: %.3f sec | Jitter: %.3f ms",
                publish_interval_sec_, elapsed_sec, jitter_ms);
  } else {
    first_call = false;
  }

  last_publish_time_ = now;

  // Publish the synchronized pair (if available)
  if (latest_rgb_msg_ && latest_depth_msg_ && latest_cam_pose_msg_) {
    rgb_publisher_->publish(*latest_rgb_msg_);
    depth_publisher_->publish(*latest_depth_msg_);
    cam_pose_publisher_->publish(*latest_cam_pose_msg_);

    // Convert depth image to point cloud and publish
    auto pointcloud_msg = depthToPointCloud(latest_depth_msg_, intrinsics_, this->get_logger());

    if (pointcloud_msg) {
        // Debug: Count valid points
        int valid_points = 0;
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*pointcloud_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*pointcloud_msg, "z");
        
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_z) {
          if (!std::isnan(*iter_x) && !std::isnan(*iter_z) && *iter_z > 0.0f) {
            valid_points++;
          }
        }
        
        pointcloud_publisher_->publish(*pointcloud_msg);
        RCLCPP_INFO(this->get_logger(),
                    "Published synchronized pair (RGB, Depth, PointCloud[%d valid points], CameraPose) | Frame: %s",
                    valid_points, pointcloud_msg->header.frame_id.c_str());
      } else {
        RCLCPP_WARN(this->get_logger(),
                    "Failed to convert depth to point cloud.");
      }
  } else {
    RCLCPP_WARN(this->get_logger(),
                "No synchronized messages available to publish.");
  }
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

  // Storage for latest synchronized pair
  sensor_msgs::msg::Image::ConstSharedPtr latest_rgb_msg_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_depth_msg_;
  geometry_msgs::msg::PoseStamped::ConstSharedPtr latest_cam_pose_msg_;

  // Camera intrinsics
  CameraIntrinsics intrinsics_;
  
  // Parameters and timing
  double publish_interval_sec_;
  rclcpp::Time last_publish_time_;

  // Timer declaration
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ApproxSyncNode>());
  rclcpp::shutdown();
  return 0;
}
