#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <fstream>
#include <vector>
#include <cstring>

class DepthPublisher : public rclcpp::Node
{
public:
    DepthPublisher()
    : Node("depth_publisher"), published_(false)
    {
        // ROS 2 QoS settings to match ZED2i //changed
        rclcpp::QoS qos(rclcpp::KeepLast(10));
        qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
        qos.durability(rclcpp::DurabilityPolicy::TransientLocal);
        
        // Publisher for depth images
        depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/sync_depth", qos);
        
        // Load depth data from .npy file
        std::string package_share_dir = ament_index_cpp::get_package_share_directory("surface_normal_estimator");
        std::string depth_file = package_share_dir + "/DIODE_dataset/depth/depth_1.npy";
        
        if (!loadNpyFile(depth_file)) {
            RCLCPP_ERROR(get_logger(), "Failed to load depth file: %s", depth_file.c_str());
            return;
        }
        
        RCLCPP_INFO(get_logger(), "Loaded depth image: %dx%d", width_, height_);
        
        // Timer to publish once after a short delay (to ensure subscriber is ready)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),  // 500ms delay
            std::bind(&DepthPublisher::publishDepth, this));
        
        RCLCPP_INFO(get_logger(), "Depth publisher node ready, will publish once to /sync_depth");
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<float> depth_data_;
    int height_;
    int width_;
    bool published_;

    bool loadNpyFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            RCLCPP_ERROR(get_logger(), "Cannot open file: %s", filename.c_str());
            return false;
        }

        // Read NPY header
        char magic[6];
        file.read(magic, 6);
        if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
            RCLCPP_ERROR(get_logger(), "Invalid NPY file format");
            return false;
        }

        uint8_t major_version, minor_version;
        file.read(reinterpret_cast<char*>(&major_version), 1);
        file.read(reinterpret_cast<char*>(&minor_version), 1);

        uint16_t header_len;
        file.read(reinterpret_cast<char*>(&header_len), 2);

        std::vector<char> header(header_len);
        file.read(header.data(), header_len);
        std::string header_str(header.begin(), header.end());

        // Parse shape from header (e.g., "{'descr': '<f4', 'fortran_order': False, 'shape': (768, 1024), }")
        size_t shape_pos = header_str.find("'shape': (");
        if (shape_pos == std::string::npos) {
            RCLCPP_ERROR(get_logger(), "Cannot find shape in NPY header");
            return false;
        }

        size_t shape_start = shape_pos + 10;
        size_t shape_end = header_str.find(")", shape_start);
        std::string shape_str = header_str.substr(shape_start, shape_end - shape_start);
        
        size_t comma_pos = shape_str.find(",");
        height_ = std::stoi(shape_str.substr(0, comma_pos));
        width_ = std::stoi(shape_str.substr(comma_pos + 1));

        // Read depth data
        size_t data_size = height_ * width_;
        depth_data_.resize(data_size);
        file.read(reinterpret_cast<char*>(depth_data_.data()), data_size * sizeof(float));

        file.close();
        return true;
    }

    void publishDepth()
    {
        if (published_) {
            return;  // Already published, do nothing
        }

        auto msg = sensor_msgs::msg::Image();
        
        // Set header
        msg.header.stamp = this->now();
        msg.header.frame_id = "zed2i_left_camera_optical_frame";  // ZED2i frame convention
        
        // Set image properties
        msg.height = height_;
        msg.width = width_;
        msg.encoding = "32FC1";  // 32-bit float, 1 channel (depth in meters)
        msg.is_bigendian = false;
        msg.step = width_ * sizeof(float);
        
        // Copy depth data (divide by 1000 to convert mm to meters, matching ZED2i)
        msg.data.resize(height_ * width_ * sizeof(float));
        float* data_ptr = reinterpret_cast<float*>(msg.data.data());
        for (size_t i = 0; i < depth_data_.size(); ++i) {
            data_ptr[i] = depth_data_[i] / 1000.0f;  // Convert mm to meters
        }
        
        depth_pub_->publish(msg);
        published_ = true;
        
        RCLCPP_INFO(get_logger(), "Published depth image once: %dx%d", width_, height_);
        
        // Cancel the timer after publishing once
        timer_->cancel();
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DepthPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}