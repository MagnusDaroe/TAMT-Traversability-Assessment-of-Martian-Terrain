/*
Publisher for camera pose messages
Topic: /camera_pose

Example values:
Header:
  stamp: {sec: 1625247600, nanosec: 123456789}
  frame_id: "camera_frame"
Pose:
  position: {x: 1.70932006835937, y: -4.63688278198242, z: 0.800000011920929}
  orientation: {x: -0.174203562915725, y: 0.681054306263119, z: 0.700247409626463, w: -0.12438535242285}

The above message is being published regularly at a configurable frequency.
*/

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <chrono>

class CameraPosePublisher : public rclcpp::Node
{
public:
  CameraPosePublisher()
  : Node("camera_pose_publisher")
  {
    // Declare configurable parameters
    this->declare_parameter<double>("publish_frequency_hz", 1.0);
    this->declare_parameter<std::string>("frame_id", "camera_frame");
    
    // Position parameters
    this->declare_parameter<double>("pos_x", 1.70932006835937);
    this->declare_parameter<double>("pos_y", -4.63688278198242);
    this->declare_parameter<double>("pos_z", 0.800000011920929);
    
    // Orientation parameters (quaternion)
    this->declare_parameter<double>("ori_x", -0.174203562915725);
    this->declare_parameter<double>("ori_y", 0.681054306263119);
    this->declare_parameter<double>("ori_z", 0.700247409626463);
    this->declare_parameter<double>("ori_w", -0.12438535242285);

    // Get parameters
    double frequency = this->get_parameter("publish_frequency_hz").as_double();
    frame_id_ = this->get_parameter("frame_id").as_string();
    
    pose_.position.x = this->get_parameter("pos_x").as_double();
    pose_.position.y = this->get_parameter("pos_y").as_double();
    pose_.position.z = this->get_parameter("pos_z").as_double();
    
    pose_.orientation.x = this->get_parameter("ori_x").as_double();
    pose_.orientation.y = this->get_parameter("ori_y").as_double();
    pose_.orientation.z = this->get_parameter("ori_z").as_double();
    pose_.orientation.w = this->get_parameter("ori_w").as_double();

    // Create publisher
    publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/camera_pose", 10);

    // Create timer based on frequency
    auto period = std::chrono::duration<double>(1.0 / frequency);
    timer_ = this->create_wall_timer(
      period,
      std::bind(&CameraPosePublisher::timer_callback, this));

    RCLCPP_INFO(this->get_logger(),
                "CameraPosePublisher started (frequency = %.2f Hz, frame = %s)",
                frequency, frame_id_.c_str());
  }

private:
  void timer_callback()
  {
    auto msg = geometry_msgs::msg::PoseStamped();
    
    // Set header
    msg.header.stamp = this->now();
    msg.header.frame_id = frame_id_;
    
    // Set pose
    msg.pose = pose_;

    // log iteration
    RCLCPP_INFO(this->get_logger(),
                 "Publishing camera pose at time: %d.%d",
                 msg.header.stamp.sec, msg.header.stamp.nanosec);
    
    // Publish
    publisher_->publish(msg);
  }

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  geometry_msgs::msg::Pose pose_;
  std::string frame_id_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraPosePublisher>());
  rclcpp::shutdown();
  return 0;
}