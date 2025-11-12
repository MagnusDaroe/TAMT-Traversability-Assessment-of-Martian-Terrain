/*
Unified Publisher Node for synchronized RGB, Depth, and Camera Pose data
Topics: /left_image, /depth, /camera_pose

Data Structure:
/sync_pkg/data/
  ├── data_1/          # Frame 1
  │   ├── rgb_img_00001.png
  │   ├── depth_00001.npy
  │   └── campose_00001.csv
  ├── data_2/          # Frame 2
  │   ├── rgb_img_00002.png
  │   ├── depth_00002.npy
  │   └── campose_00002.csv
  └── data_3/          # Frame 3
      ├── rgb_img_00003.png
      ├── depth_00003.npy
      └── campose_00003.csv

Each data_X folder contains exactly ONE frame (3 files: RGB image, depth data, camera pose)

command:
  ros2 run sync_pkg publisher_all_data --ros-args -p publish_frequency_hz:=0.5 -p loop_playback:=true
*/

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;

struct DatasetFrame {
    fs::path rgb_path;
    fs::path depth_path;
    fs::path campose_path;
    std::string dataset_name;
};

class UnifiedDataPublisher : public rclcpp::Node
{
public:
    UnifiedDataPublisher()
    : Node("unified_data_publisher"), current_frame_idx_(0)
    {
        // Declare parameters
        this->declare_parameter<double>("publish_frequency_hz", 1.0);
        this->declare_parameter<std::string>("frame_id", "camera_frame");
        this->declare_parameter<bool>("loop_playback", true);
        
        double frequency = this->get_parameter("publish_frequency_hz").as_double();
        frame_id_ = this->get_parameter("frame_id").as_string();
        loop_playback_ = this->get_parameter("loop_playback").as_bool();

        // Setup QoS for image topics
        rclcpp::QoS qos(rclcpp::KeepLast(10));
        qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
        qos.durability(rclcpp::DurabilityPolicy::Volatile);  // Changed from TransientLocal to Volatile for streaming data
        
        // Create publishers
        rgb_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/left_image", qos);
        depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/depth", qos);
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/camera_pose", 10);
        
        // Load all dataset frames
        if (!loadDatasets()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load datasets. Exiting.");
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Loaded %zu frames", frames_.size());
        
        // Create timer for publishing
        auto period = std::chrono::duration<double>(1.0 / frequency);
        timer_ = this->create_wall_timer(
            period,
            std::bind(&UnifiedDataPublisher::publishFrame, this));
        
        RCLCPP_INFO(this->get_logger(), 
                    "Unified publisher started (frequency = %.2f Hz, frame = %s, loop = %s)",
                    frequency, frame_id_.c_str(), loop_playback_ ? "true" : "false");
    }

private:
    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Data
    std::vector<DatasetFrame> frames_;
    size_t current_frame_idx_;
    std::string frame_id_;
    bool loop_playback_;

    bool loadDatasets()
    {
        std::string package_share_dir;
        try {
            package_share_dir = ament_index_cpp::get_package_share_directory("sync_pkg");
        } catch (const std::exception &e) {
            RCLCPP_ERROR(get_logger(), "Could not locate package share directory: %s", e.what());
            return false;
        }
        
        fs::path data_dir = fs::path(package_share_dir) / "data";
        
        if (!fs::exists(data_dir) || !fs::is_directory(data_dir)) {
            RCLCPP_ERROR(get_logger(), "Data directory not found: %s", data_dir.string().c_str());
            return false;
        }
        
        // Collect all data_* subdirectories
        std::vector<fs::path> dataset_dirs;
        for (const auto &entry : fs::directory_iterator(data_dir)) {
            if (entry.is_directory()) {
                std::string dir_name = entry.path().filename().string();
                if (dir_name.find("data_") == 0) {
                    dataset_dirs.push_back(entry.path());
                }
            }
        }
        
        // Sort numerically by extracting the number after "data_"
        std::sort(dataset_dirs.begin(), dataset_dirs.end(), 
            [](const fs::path &a, const fs::path &b) {
                std::string name_a = a.filename().string();
                std::string name_b = b.filename().string();
                
                // Extract numeric part after "data_"
                size_t pos_a = name_a.find("data_");
                size_t pos_b = name_b.find("data_");
                
                if (pos_a != std::string::npos && pos_b != std::string::npos) {
                    int num_a = std::stoi(name_a.substr(pos_a + 5));
                    int num_b = std::stoi(name_b.substr(pos_b + 5));
                    return num_a < num_b;
                }
                
                // Fallback to lexicographical if parsing fails
                return name_a < name_b;
            });
        
        RCLCPP_INFO(get_logger(), "Found %zu dataset directories", dataset_dirs.size());
        
        // Load frame from each dataset (each data_X has exactly 1 frame)
        for (const auto &dataset_path : dataset_dirs) {
            loadDatasetFrame(dataset_path);
        }
        
        if (frames_.empty()) {
            RCLCPP_ERROR(get_logger(), "No valid frames found in any dataset");
            return false;
        }
        
        return true;
    }

    void loadDatasetFrame(const fs::path &dataset_path)
    {
        std::string dataset_name = dataset_path.filename().string();
        
        // Find the 3 files in this dataset directory
        fs::path rgb_file, depth_file, campose_file;
        
        for (const auto &entry : fs::directory_iterator(dataset_path)) {
            if (!entry.is_regular_file()) continue;
            
            std::string filename = entry.path().filename().string();
            std::string ext = entry.path().extension().string();
            
            if (filename.find("rgb_img_") == 0 && (ext == ".png" || ext == ".PNG")) {
                rgb_file = entry.path();
            }
            else if (filename.find("depth_") == 0 && ext == ".npy") {
                depth_file = entry.path();
            }
            else if (filename.find("campose_") == 0 && ext == ".csv") {
                campose_file = entry.path();
            }
        }
        
        // Validate that all 3 files exist
        if (rgb_file.empty() || depth_file.empty() || campose_file.empty()) {
            RCLCPP_WARN(get_logger(), "Incomplete dataset in %s (RGB: %s, Depth: %s, Campose: %s)",
                        dataset_name.c_str(),
                        rgb_file.empty() ? "missing" : "found",
                        depth_file.empty() ? "missing" : "found",
                        campose_file.empty() ? "missing" : "found");
            return;
        }
        
        // Create frame entry
        DatasetFrame frame;
        frame.rgb_path = rgb_file;
        frame.depth_path = depth_file;
        frame.campose_path = campose_file;
        frame.dataset_name = dataset_name;
        frames_.push_back(frame);
        
        RCLCPP_INFO(get_logger(), "Loaded frame from %s", dataset_name.c_str());
    }

    bool loadNpyFile(const fs::path &filepath, std::vector<float> &data, int &height, int &width)
    {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            RCLCPP_ERROR(get_logger(), "Cannot open NPY file: %s", filepath.string().c_str());
            return false;
        }

        // Read NPY header
        char magic[6];
        file.read(magic, 6);
        if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
            RCLCPP_ERROR(get_logger(), "Invalid NPY file format: %s", filepath.string().c_str());
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

        // Parse shape
        size_t shape_pos = header_str.find("'shape': (");
        if (shape_pos == std::string::npos) {
            RCLCPP_ERROR(get_logger(), "Cannot find shape in NPY header");
            return false;
        }

        size_t shape_start = shape_pos + 10;
        size_t shape_end = header_str.find(")", shape_start);
        std::string shape_str = header_str.substr(shape_start, shape_end - shape_start);
        
        size_t comma_pos = shape_str.find(",");
        height = std::stoi(shape_str.substr(0, comma_pos));
        width = std::stoi(shape_str.substr(comma_pos + 1));

        // Read depth data
        size_t data_size = height * width;
        data.resize(data_size);
        file.read(reinterpret_cast<char*>(data.data()), data_size * sizeof(float));

        file.close();
        return true;
    }

    bool loadCamPose(const fs::path &filepath, geometry_msgs::msg::Pose &pose)
    {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            RCLCPP_ERROR(get_logger(), "Cannot open campose file: %s", filepath.string().c_str());
            return false;
        }

        std::string line;
        // Skip header line
        if (!std::getline(file, line)) {
            RCLCPP_ERROR(get_logger(), "Empty campose file: %s", filepath.string().c_str());
            return false;
        }

        // Read data line
        if (!std::getline(file, line)) {
            RCLCPP_ERROR(get_logger(), "No data in campose file: %s", filepath.string().c_str());
            return false;
        }

        // Parse CSV: frame_index,image,tx,ty,tz,qx,qy,qz,qw
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 9) {
            RCLCPP_ERROR(get_logger(), "Invalid campose format: %s", filepath.string().c_str());
            return false;
        }

        try {
            // tx, ty, tz are at indices 2, 3, 4
            pose.position.x = std::stod(tokens[2]);
            pose.position.y = std::stod(tokens[3]);
            pose.position.z = std::stod(tokens[4]);
            
            // qx, qy, qz, qw are at indices 5, 6, 7, 8
            pose.orientation.x = std::stod(tokens[5]);
            pose.orientation.y = std::stod(tokens[6]);
            pose.orientation.z = std::stod(tokens[7]);
            pose.orientation.w = std::stod(tokens[8]);
        } catch (const std::exception &e) {
            RCLCPP_ERROR(get_logger(), "Error parsing campose values: %s", e.what());
            return false;
        }

        file.close();
        return true;
    }

    void publishFrame()
    {
        if (frames_.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 5000, 
                                 "No frames to publish");
            return;
        }

        const auto &frame = frames_[current_frame_idx_];
        auto timestamp = this->now();

        RCLCPP_INFO(get_logger(), "Publishing frame %zu/%zu from %s", 
                    current_frame_idx_ + 1, frames_.size(), 
                    frame.dataset_name.c_str());

        // Publish RGB image
        publishRgbImage(frame.rgb_path, timestamp);
        
        // Publish depth image
        publishDepthImage(frame.depth_path, timestamp);
        
        // Publish camera pose
        publishCameraPose(frame.campose_path, timestamp);

        // Advance to next frame
        current_frame_idx_++;
        if (current_frame_idx_ >= frames_.size()) {
            if (loop_playback_) {
                current_frame_idx_ = 0;
                RCLCPP_INFO(get_logger(), "Looping back to first frame");
            } else {
                RCLCPP_INFO(get_logger(), "Reached end of dataset. Stopping.");
                timer_->cancel();
            }
        }
    }

    void publishRgbImage(const fs::path &rgb_path, const rclcpp::Time &timestamp)
    {
        cv::Mat img = cv::imread(rgb_path.string(), cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            RCLCPP_ERROR(get_logger(), "Failed to read RGB image: %s", rgb_path.string().c_str());
            return;
        }

        auto msg = sensor_msgs::msg::Image();
        msg.header.stamp = timestamp;
        msg.header.frame_id = "left_camera";
        
        cv::Mat converted;
        std::string encoding;
        
        if (img.channels() == 4) {
            cv::cvtColor(img, converted, cv::COLOR_BGRA2RGBA);
            encoding = "rgba8";
        } else if (img.channels() == 3) {
            cv::cvtColor(img, converted, cv::COLOR_BGR2RGB);
            encoding = "rgb8";
        } else if (img.channels() == 1) {
            cv::cvtColor(img, converted, cv::COLOR_GRAY2RGB);
            encoding = "rgb8";
        } else {
            converted = img;
            encoding = "rgb8";
        }

        msg.height = static_cast<uint32_t>(converted.rows);
        msg.width = static_cast<uint32_t>(converted.cols);
        msg.encoding = encoding;
        msg.is_bigendian = 0;
        msg.step = static_cast<uint32_t>(converted.cols * converted.elemSize());
        msg.data.assign(converted.data, converted.data + converted.total() * converted.elemSize());
        
        rgb_pub_->publish(msg);
    }

    void publishDepthImage(const fs::path &depth_path, const rclcpp::Time &timestamp)
    {
        std::vector<float> depth_data;
        int height, width;
        
        if (!loadNpyFile(depth_path, depth_data, height, width)) {
            RCLCPP_ERROR(get_logger(), "Failed to load depth file: %s", depth_path.string().c_str());
            return;
        }

        auto msg = sensor_msgs::msg::Image();
        msg.header.stamp = timestamp;
        msg.header.frame_id = "zed2i_left_camera_optical_frame";
        
        msg.height = height;
        msg.width = width;
        msg.encoding = "32FC1";
        msg.is_bigendian = false;
        msg.step = width * sizeof(float);
        
        // Unit in meters
        msg.data.resize(height * width * sizeof(float));
        float* data_ptr = reinterpret_cast<float*>(msg.data.data());
        for (size_t i = 0; i < depth_data.size(); ++i) {
            data_ptr[i] = depth_data[i];
        }
        
        depth_pub_->publish(msg);
    }

    void publishCameraPose(const fs::path &campose_path, const rclcpp::Time &timestamp)
    {
        geometry_msgs::msg::Pose pose;
        
        if (!loadCamPose(campose_path, pose)) {
            RCLCPP_ERROR(get_logger(), "Failed to load campose: %s", campose_path.string().c_str());
            return;
        }

        auto msg = geometry_msgs::msg::PoseStamped();
        msg.header.stamp = timestamp;
        msg.header.frame_id = frame_id_;
        msg.pose = pose;
        
        pose_pub_->publish(msg);
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<UnifiedDataPublisher>());
    rclcpp::shutdown();
    return 0;
}