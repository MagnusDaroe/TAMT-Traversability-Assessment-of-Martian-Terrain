#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <memory>
#include <string>

using namespace std::chrono_literals;

class SurfaceNormalEstimation : public rclcpp::Node
{
public:
    SurfaceNormalEstimation()
    : Node("surface_normal_estimation")
    {
        // ROS 2 QoS settings
        rclcpp::QoS qos(10);
        qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        qos.durability(rclcpp::DurabilityPolicy::Volatile);
        
        RCLCPP_INFO(get_logger(), "SurfaceNormalEstimation node ready");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        
    }
    
    void processingLoop()
    {
        
        
    }
    
    
    
    
    // Member variables
    
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SurfaceNormalEstimation>());
    rclcpp::shutdown();
    return 0;
}