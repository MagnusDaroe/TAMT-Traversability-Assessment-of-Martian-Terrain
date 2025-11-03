#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

class ApproxSyncNode : public rclcpp::Node
{
public:
  ApproxSyncNode()
  : Node("sync_node")
  {
    // Declare configurable parameter (default = 1.0 second)
    this->declare_parameter<double>("publish_interval_sec", 1.0);
    publish_interval_sec_ = this->get_parameter("publish_interval_sec").as_double();

    // Subscribers using message_filters
    rgb_subscriber_.subscribe(this, "/left_image");
    depth_subscriber_.subscribe(this, "/depth");

    // Synchronizer with ApproximateTime policy and queue size = 10
    sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(10, rgb_subscriber_, depth_subscriber_);
    sync_->registerCallback(std::bind(&ApproxSyncNode::sync_callback, this,
                                      std::placeholders::_1, std::placeholders::_2));

    // Publishers for synchronized data
    rgb_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/sync_rgb", 10);
    depth_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/sync_depth", 10);

    last_publish_time_ = this->now();

    RCLCPP_INFO(this->get_logger(),
                "ApproxSyncNode started (publish interval = %.2f sec)",
                publish_interval_sec_);
  }

private:
  using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image>;

  void sync_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg)
  { 
    rclcpp::Time t_rgb(rgb_msg->header.stamp);
    rclcpp::Time t_depth(depth_msg->header.stamp);
    double diff_ms = std::abs((t_rgb - t_depth).seconds() * 1000.0); // milliseconds
    RCLCPP_INFO(this->get_logger(),
                "Synchronized pair received | RGB time: %.3f, Depth time: %.3f, Diff: %.3f ms",
                t_rgb.seconds(), t_depth.seconds(), diff_ms);
    // Compute elapsed time since last published pair
    rclcpp::Time now = this->now();
    double elapsed = (now - last_publish_time_).seconds();

    if (elapsed >= publish_interval_sec_) {
      // Publish the first pair after the interval
      rgb_publisher_->publish(*rgb_msg);
      depth_publisher_->publish(*depth_msg);
      last_publish_time_ = now;

      RCLCPP_INFO(this->get_logger(),
                  "Published synchronized pair | elapsed = %.3f sec | diff = %.3f ms",
                  elapsed, diff_ms);
    } else {
      // Skip this pair
      RCLCPP_DEBUG(this->get_logger(),
                   "Skipping synchronized pair (elapsed %.3f < %.3f sec)",
                   elapsed, publish_interval_sec_);
    }
  }

  // Subscribers and synchronizer
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_subscriber_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_subscriber_;
  std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_publisher_;

  // Parameters and timing
  double publish_interval_sec_;
  rclcpp::Time last_publish_time_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ApproxSyncNode>());
  rclcpp::shutdown();
  return 0;
}
