/*
Unified Publisher Node for synchronized RGB, Depth, and Camera Pose data
Loads one frame at a time -> Publishes -> Clears -> Moves to next frame

Topics:
  /left_image
  /depth
  /camera_pose

Directory Structure:
/sync_pkg/frame_data/
  ├── images/
  │   ├── rgb_000000.png
  │   ├── rgb_000001.png
  │   └── ...
  ├── depth/
  │   ├── depth_000000.npy
  │   ├── depth_000001.npy
  │   └── ...
  └── cam_poses.csv

Command example:
  ros2 run sync_pkg publish_updated_raw_data --ros-args -p publish_frequency_hz:=0.5 -p loop_playback:=true
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

struct FrameData {
  std::string rgb_filename;
  double pos_x, pos_y, pos_z;
  double quat_x, quat_y, quat_z, quat_w;
};

class UnifiedDataPublisher : public rclcpp::Node
{
public:
  UnifiedDataPublisher()
  : Node("unified_data_publisher"), current_frame_idx_(0)
  {
    // Declare and get parameters
    this->declare_parameter<double>("publish_frequency_hz", 10.0);
    this->declare_parameter<bool>("loop_playback", true);
    this->declare_parameter<std::string>("frame_id", "camera_frame");

    publish_frequency_ = this->get_parameter("publish_frequency_hz").as_double();
    loop_playback_ = this->get_parameter("loop_playback").as_bool();
    frame_id_ = this->get_parameter("frame_id").as_string();

    // QoS setup
    rclcpp::QoS qos(10);
    qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
    qos.durability(rclcpp::DurabilityPolicy::Volatile);

    // Publishers
    rgb_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/rgb", qos);
    depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/depth", qos);
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/rover_pose", qos);

    // Locate data directory
    try {
      package_share_dir_ = ament_index_cpp::get_package_share_directory("sync_pkg");
    } catch (const std::exception &e) {
      RCLCPP_FATAL(this->get_logger(), "Package not found: %s", e.what());
      return;
    }

    data_dir_ = fs::path(package_share_dir_) / "frame_data";

    if (!fs::exists(data_dir_) || !fs::is_directory(data_dir_)) {
      RCLCPP_FATAL(this->get_logger(), "Data directory not found: %s", data_dir_.string().c_str());
      return;
    }

    // Set up paths for new structure
    images_dir_ = data_dir_ / "images";
    depth_dir_ = data_dir_ / "depth";
    camposes_file_ = data_dir_ / "rover_poses.csv";

    // Validate structure
    if (!fs::exists(images_dir_) || !fs::is_directory(images_dir_)) {
      RCLCPP_FATAL(this->get_logger(), "Images directory not found: %s", images_dir_.string().c_str());
      return;
    }
    if (!fs::exists(depth_dir_) || !fs::is_directory(depth_dir_)) {
      RCLCPP_FATAL(this->get_logger(), "Depth directory not found: %s", depth_dir_.string().c_str());
      return;
    }
    if (!fs::exists(camposes_file_)) {
      RCLCPP_FATAL(this->get_logger(), "camposes.csv not found: %s", camposes_file_.string().c_str());
      return;
    }

    // Parse camposes.csv and build frame list
    if (!parseCamposesCSV()) {
      RCLCPP_FATAL(this->get_logger(), "Failed to parse camposes.csv");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Loaded %zu frames from camposes.csv", frames_.size());

    // Setup timer for publishing
    auto period = std::chrono::duration<double>(1.0 / publish_frequency_);
    timer_ = this->create_wall_timer(period, std::bind(&UnifiedDataPublisher::publishNextFrame, this));
  }

private:
  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Parameters and state
  double publish_frequency_;
  bool loop_playback_;
  std::string frame_id_;
  std::string package_share_dir_;
  fs::path data_dir_;
  fs::path images_dir_;
  fs::path depth_dir_;
  fs::path camposes_file_;
  std::vector<FrameData> frames_;
  size_t current_frame_idx_;

  bool parseCamposesCSV()
  {
    std::ifstream file(camposes_file_);
    if (!file.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Cannot open cam_poses.csv");
      return false;
    }

    std::string line;
    
    while (std::getline(file, line)) {
      // Skip empty lines and comment lines
      if (line.empty() || line[0] == '#') continue;
      
      // Skip header line (contains "frame_index")
      if (line.find("frame_index") != std::string::npos) continue;

      std::stringstream ss(line);
      std::vector<std::string> tokens;
      std::string token;
      
      while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
      }

      // Expecting at least 9 columns: index, rgb_filename, x, y, z, qx, qy, qz, qw
      if (tokens.size() < 9) {
        RCLCPP_WARN(this->get_logger(), "Skipping malformed line: %s", line.c_str());
        continue;
      }

      FrameData frame;
      frame.rgb_filename = tokens[1];  // Second column is RGB filename
      frame.pos_x = std::stod(tokens[2]);
      frame.pos_y = std::stod(tokens[3]);
      frame.pos_z = std::stod(tokens[4]);
      frame.quat_x = std::stod(tokens[5]);
      frame.quat_y = std::stod(tokens[6]);
      frame.quat_z = std::stod(tokens[7]);
      frame.quat_w = std::stod(tokens[8]);

      frames_.push_back(frame);
    }

    file.close();
    return !frames_.empty();
  }

  std::string extractFrameNumber(const std::string &rgb_filename)
  {
    // Extract number from images/rgb_000000.png -> 000000
    // First get just the filename part
    size_t slash_pos = rgb_filename.find_last_of("/\\");
    std::string filename = (slash_pos != std::string::npos) 
                           ? rgb_filename.substr(slash_pos + 1) 
                           : rgb_filename;
    
    // Now extract the number: rgb_000000.png -> 000000
    size_t underscore_pos = filename.find('_');
    size_t dot_pos = filename.find('.');
    if (underscore_pos != std::string::npos && dot_pos != std::string::npos) {
      return filename.substr(underscore_pos + 1, dot_pos - underscore_pos - 1);
    }
    return "";
  }

  void publishNextFrame()
  {
    if (frames_.empty()) {
      RCLCPP_WARN(this->get_logger(), "No frames to publish.");
      return;
    }

    if (current_frame_idx_ >= frames_.size()) {
      if (loop_playback_) {
        current_frame_idx_ = 0;
        RCLCPP_INFO(this->get_logger(), "Looping back to first frame.");
      } else {
        RCLCPP_INFO(this->get_logger(), "End of dataset reached. Stopping timer.");
        timer_->cancel();
        return;
      }
    }

    const FrameData &frame = frames_[current_frame_idx_];
    RCLCPP_INFO(this->get_logger(), "Publishing frame %zu: %s", current_frame_idx_, frame.rgb_filename.c_str());

    // Build file paths
    // RGB filename is in format "images/rgb_000000.png", so just use it directly
    fs::path rgb_path = data_dir_ / frame.rgb_filename;
    
    // Extract frame number from RGB filename to build depth filename
    std::string frame_number = extractFrameNumber(frame.rgb_filename);
    std::string depth_filename = "depth_" + frame_number + ".npy";
    fs::path depth_path = depth_dir_ / depth_filename;

    auto timestamp = this->now();

    // Publish RGB
    if (fs::exists(rgb_path)) {
      publishRgb(rgb_path, timestamp);
    } else {
      RCLCPP_WARN(this->get_logger(), "RGB file not found: %s", rgb_path.string().c_str());
    }

    // Publish Depth
    if (fs::exists(depth_path)) {
      publishDepth(depth_path, timestamp);
    } else {
      RCLCPP_WARN(this->get_logger(), "Depth file not found: %s", depth_path.string().c_str());
    }

    // Publish Pose (from CSV data)
    publishPose(frame, timestamp);

    current_frame_idx_++;
  }

  void publishRgb(const fs::path &path, const rclcpp::Time &stamp)
  {
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to read RGB image: %s", path.string().c_str());
      return;
    }

    cv::Mat converted;
    std::string encoding;
    if (img.channels() == 3) {
      cv::cvtColor(img, converted, cv::COLOR_BGR2RGB);
      encoding = "rgb8";
    } else if (img.channels() == 4) {
      cv::cvtColor(img, converted, cv::COLOR_BGRA2RGBA);
      encoding = "rgba8";
    } else {
      converted = img;
      encoding = "mono8";
    }

    sensor_msgs::msg::Image msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "left_camera";
    msg.height = converted.rows;
    msg.width = converted.cols;
    msg.encoding = encoding;
    msg.is_bigendian = 0;
    msg.step = converted.cols * converted.elemSize();
    msg.data.assign(converted.data, converted.data + converted.total() * converted.elemSize());

    rgb_pub_->publish(msg);
  }

  bool loadNpyFile(const fs::path &filepath, std::vector<float> &data, int &height, int &width)
  {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    char magic[6];
    file.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) return false;

    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    std::vector<char> header(header_len);
    file.read(header.data(), header_len);
    std::string header_str(header.begin(), header.end());

    size_t shape_pos = header_str.find("'shape': (");
    size_t shape_end = header_str.find(")", shape_pos);
    std::string shape_str = header_str.substr(shape_pos + 10, shape_end - shape_pos - 10);
    size_t comma_pos = shape_str.find(",");
    height = std::stoi(shape_str.substr(0, comma_pos));
    width = std::stoi(shape_str.substr(comma_pos + 1));

    data.resize(height * width);
    file.read(reinterpret_cast<char*>(data.data()), height * width * sizeof(float));
    file.close();
    return true;
  }

  void publishDepth(const fs::path &path, const rclcpp::Time &stamp)
  {
    std::vector<float> depth;
    int h, w;
    if (!loadNpyFile(path, depth, h, w)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load depth file: %s", path.string().c_str());
      return;
    }

    sensor_msgs::msg::Image msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "camera_depth_frame";
    msg.height = h;
    msg.width = w;
    msg.encoding = "32FC1";
    msg.is_bigendian = false;
    msg.step = w * sizeof(float);
    msg.data.resize(depth.size() * sizeof(float));
    std::memcpy(msg.data.data(), depth.data(), depth.size() * sizeof(float));
    depth_pub_->publish(msg);
  }

  void publishPose(const FrameData &frame, const rclcpp::Time &stamp)
  {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = frame_id_;
    pose_msg.pose.position.x = frame.pos_x;
    pose_msg.pose.position.y = frame.pos_y;
    pose_msg.pose.position.z = frame.pos_z;
    pose_msg.pose.orientation.x = frame.quat_x;
    pose_msg.pose.orientation.y = frame.quat_y;
    pose_msg.pose.orientation.z = frame.quat_z;
    pose_msg.pose.orientation.w = frame.quat_w;

    pose_pub_->publish(pose_msg);
  }
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<UnifiedDataPublisher>());
  rclcpp::shutdown();
  return 0;
}