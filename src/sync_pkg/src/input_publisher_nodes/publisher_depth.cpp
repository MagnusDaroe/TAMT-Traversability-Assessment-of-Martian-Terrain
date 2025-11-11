#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <fstream>
#include <vector>
#include <cstring>
#include <filesystem>
#include <algorithm>

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
            "/depth", qos);
        
        // Load depth data from .npy file
        std::string package_share_dir;
        try {
            // Get the installed package share directory for this package.
            package_share_dir = ament_index_cpp::get_package_share_directory("sync_pkg");
        } catch (const std::exception & e) {
            // If that fails (package not found in overlay), fall back to current working dir.
            RCLCPP_WARN(get_logger(), "Could not locate package share directory: %s. Falling back to current working directory.\n",
                         e.what());
            package_share_dir = ".";
        }



            std::string depth_dir = package_share_dir + "/depth";
            std::vector<std::string> npy_files;

            try {
                namespace fs = std::filesystem;
                if (fs::exists(depth_dir) && fs::is_directory(depth_dir)) {
                for (const auto &entry : fs::directory_iterator(depth_dir)) {
                    if (!entry.is_regular_file()) continue;
                    auto ext = entry.path().extension().string();
                    if (ext == ".npy") {
                    npy_files.push_back(entry.path().string());
                    }
                }
                } else {
                RCLCPP_WARN(get_logger(), "Depth_data directory not found at %s, falling back to single file", depth_dir.c_str());
                }
            } catch (const std::exception &e) {
                RCLCPP_WARN(get_logger(), "Error scanning Depth_data directory: %s. Falling back to single file.", e.what());
            }

            if (npy_files.empty()) {
                // No files found in directory, fall back to previous single-file behavior
                std::string depth_file = package_share_dir + "/Depth_data/depth_1.npy";
                if (!loadNpyFile(depth_file)) {
                RCLCPP_ERROR(get_logger(), "Failed to load depth file: %s", depth_file.c_str());
                return;
                }
                RCLCPP_INFO(get_logger(), "Found 0 files in %s, loaded fallback file.", depth_dir.c_str());
            } else {
                // Sort files for deterministic order and load the first one
                std::sort(npy_files.begin(), npy_files.end());
                RCLCPP_INFO(get_logger(), "Found %zu .npy files in %s, loading %s", npy_files.size(), depth_dir.c_str(), npy_files[0].c_str());
                if (!loadNpyFile(npy_files[0])) {
                RCLCPP_ERROR(get_logger(), "Failed to load depth file: %s", npy_files[0].c_str());
                return;
                }
            }
        
        // Log loaded file and dimensions
        std::string loaded_file;
        if (!npy_files.empty()) {
            loaded_file = npy_files[0];
        } else {
            loaded_file = package_share_dir + "/Depth_data/depth_1.npy";
        }
        RCLCPP_INFO(get_logger(), "Loaded depth image: %dx%d from %s", width_, height_, loaded_file.c_str());
        
        // Timer to publish once after a short delay (to ensure subscriber is ready)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),  // 500ms delay
            std::bind(&DepthPublisher::publishDepth, this));
        
        RCLCPP_INFO(get_logger(), "Depth publisher node ready, will publish once to /depth");
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
        
        RCLCPP_INFO(get_logger(), "Published depth image: %dx%d", width_, height_);
        
        // Keep publishing continuously at a configurable frequency.
        // Declare parameter once (ignore if already declared) and read it.
        try {
            this->declare_parameter<double>("depth_publish_hz", 3.0);
        } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
            // already declared, ignore
        }
        double hz = this->get_parameter_or<double>("depth_publish_hz", 3.0);
        if (hz <= 0.0) {
            RCLCPP_WARN(get_logger(), "Invalid depth_publish_hz (%f), using 3.0 Hz", hz);
            hz = 3.0;
        }

        // Recreate timer with the requested period (cancel previous timer first)
        if (timer_) {
            timer_->cancel();
        }
        auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / hz));
        timer_ = this->create_wall_timer(
            period_ns,
            std::bind(&DepthPublisher::publishDepth, this)
        );

        // Allow further publishes and log the configured rate
        published_ = false;
        RCLCPP_INFO(get_logger(), "Publishing depth at %.2f Hz", hz);

        // Also print the header timestamp to the console (seconds.nanoseconds)
        {
            auto stamp = msg.header.stamp;  // builtin_interfaces::msg::Time
            // Convert seconds to local broken-down time and format HH:MM:SS
            std::time_t t = static_cast<std::time_t>(stamp.sec);
            std::tm tm{};
    #if defined(_WIN32) || defined(_WIN64)
            localtime_s(&tm, &t);
    #else
            localtime_r(&t, &tm);
    #endif
            char time_buf[9]; // "HH:MM:SS"
            std::snprintf(time_buf, sizeof(time_buf), "%02d:%02d:%02d",
                          tm.tm_hour, tm.tm_min, tm.tm_sec);
            RCLCPP_INFO(get_logger(), "Published depth image timestamp: %s", time_buf);
        }
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