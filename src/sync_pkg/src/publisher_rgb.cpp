#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <cstring>

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class RgbPublisher : public rclcpp::Node
{
public:
    RgbPublisher()
    : Node("RgbPublisher")
    {
        // ROS 2 QoS settings to match ZED2i
        rclcpp::QoS qos(rclcpp::KeepLast(10));
        qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
        qos.durability(rclcpp::DurabilityPolicy::TransientLocal);
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/left_image", qos);
        
        // Find a .png file in the package's RGB_data using ament index
        namespace fs = std::filesystem;
        {
            std::string pkg_share = ament_index_cpp::get_package_share_directory("sync_pkg");
            fs::path rgb_dir = fs::path(pkg_share) / "rgb";
            if (!fs::exists(rgb_dir) || !fs::is_directory(rgb_dir)) {
            RCLCPP_ERROR(this->get_logger(), "rgb directory not found: %s", rgb_dir.string().c_str());
            } else {
            fs::path png_path;
            for (auto &entry : fs::directory_iterator(rgb_dir)) {
                if (!entry.is_regular_file()) continue;
                if (entry.path().extension() == ".png" || entry.path().extension() == ".PNG") {
                png_path = entry.path();
                break;
                }
            }
            if (png_path.empty()) {
                RCLCPP_ERROR(this->get_logger(), "No .png file found in: %s", rgb_dir.string().c_str());
            } else {
                // Load image using OpenCV
                cv::Mat img = cv::imread(png_path.string(), cv::IMREAD_UNCHANGED);
                if (img.empty()) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to read image: %s", png_path.string().c_str());
                } else {
                    cv::Mat converted;
                    std::string encoding;
                    if (img.channels() == 4) {
                        // OpenCV loads as BGRA -> convert to RGBA
                        cv::cvtColor(img, converted, cv::COLOR_BGRA2RGBA);
                        encoding = "rgba8";
                    } else if (img.channels() == 3) {
                        // BGR -> RGB
                        cv::cvtColor(img, converted, cv::COLOR_BGR2RGB);
                        encoding = "rgb8";
                    } else if (img.channels() == 1) {
                        cv::cvtColor(img, converted, cv::COLOR_GRAY2RGB);
                        encoding = "rgb8";
                    } else {
                        RCLCPP_ERROR(this->get_logger(), "Unsupported channel count: %d", img.channels());
                        converted = img;
                        encoding = "rgb8";
                    }

                    // Fill sensor_msgs::msg::Image
                    image_msg_.height = static_cast<uint32_t>(converted.rows);
                    image_msg_.width = static_cast<uint32_t>(converted.cols);
                    image_msg_.encoding = encoding;
                    image_msg_.is_bigendian = 0;
                    image_msg_.step = static_cast<uint32_t>(converted.cols * converted.elemSize());
                    image_msg_.data.assign(converted.data, converted.data + converted.total() * converted.elemSize());
                    image_msg_.header.frame_id = "left_camera";
                    RCLCPP_INFO(this->get_logger(), "Loaded image %s (%ux%u, %s)", png_path.string().c_str(),
                                            image_msg_.width, image_msg_.height, image_msg_.encoding.c_str());
                }
            }
        }
        }
                 
        timer_ = this->create_wall_timer(
            2s, std::bind(&RgbPublisher::timer_callback, this));  // ~0.5 Hz (~2 s period) Change time as needed
         
    }

private:
    void timer_callback()
    {
        if (image_msg_.data.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "No image to publish");
            return;
        }
        image_msg_.header.stamp = this->now();
        publisher_->publish(image_msg_);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Published /left_image (%ux%u, %s)",
                                                image_msg_.width, image_msg_.height, image_msg_.encoding.c_str());
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    sensor_msgs::msg::Image image_msg_;
};

// Replace main to start RgbPublisher
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RgbPublisher>());
    rclcpp::shutdown();
    return 0;
}