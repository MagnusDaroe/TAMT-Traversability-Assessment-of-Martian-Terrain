/*
Unified Publisher Node for synchronized RGB, Depth, and Camera Pose data
Loads one frame at a time -> Publishes -> Clears -> Moves to next frame

Topics:
  /left_image
  /depth
  /camera_pose

Directory Structure:
/sync_pkg/data/
  ├── data_1/
  │   ├── rgb_img_00001.png
  │   ├── depth_00001.npy
  │   └── campose_00001.csv
  ├── data_2/
  │   ├── ...
  └── data_N/

Command example:
  ros2 run sync_pkg publish_raw_data --ros-args -p publish_frequency_hz:=0.5 -p loop_playback:=true
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

class UnifiedDataPublisher : public rclcpp::Node
{
public:
  UnifiedDataPublisher()
  : Node("unified_data_publisher"), current_frame_idx_(0)
  {
    // Declare and get parameters
    this->declare_parameter<double>("publish_frequency_hz", 1.0);
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
    rgb_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/left_image", qos);
    depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/depth", qos);
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/camera_pose", qos);

    // Locate data directory
    try {
      package_share_dir_ = ament_index_cpp::get_package_share_directory("sync_pkg");
    } catch (const std::exception &e) {
      RCLCPP_FATAL(this->get_logger(), "Package not found: %s", e.what());
      return;
    }

    data_dir_ = fs::path(package_share_dir_) / "data";

    if (!fs::exists(data_dir_) || !fs::is_directory(data_dir_)) {
      RCLCPP_FATAL(this->get_logger(), "Data directory not found: %s", data_dir_.string().c_str());
      return;
    }

    // Collect dataset folders
    for (const auto &entry : fs::directory_iterator(data_dir_)) {
      if (entry.is_directory() && entry.path().filename().string().find("data_") == 0) {
        dataset_dirs_.push_back(entry.path());
      }
    }

    if (dataset_dirs_.empty()) {
      RCLCPP_FATAL(this->get_logger(), "No valid data_X directories found.");
      return;
    }

    // Sort numerically
    std::sort(dataset_dirs_.begin(), dataset_dirs_.end(), [](const fs::path &a, const fs::path &b) {
      auto numA = std::stoi(a.filename().string().substr(5));
      auto numB = std::stoi(b.filename().string().substr(5));
      return numA < numB;
    });

    RCLCPP_INFO(this->get_logger(), "Found %zu dataset folders", dataset_dirs_.size());

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
  std::vector<fs::path> dataset_dirs_;
  size_t current_frame_idx_;

  void publishNextFrame()
  {
    if (dataset_dirs_.empty()) {
      RCLCPP_WARN(this->get_logger(), "No datasets to publish.");
      return;
    }

    if (current_frame_idx_ >= dataset_dirs_.size()) {
      if (loop_playback_) {
        current_frame_idx_ = 0;
        RCLCPP_INFO(this->get_logger(), "Looping back to first frame.");
      } else {
        RCLCPP_INFO(this->get_logger(), "End of dataset reached. Stopping timer.");
        timer_->cancel();
        return;
      }
    }

    fs::path current_dir = dataset_dirs_[current_frame_idx_];
    RCLCPP_INFO(this->get_logger(), "Publishing frame from: %s", current_dir.string().c_str());

    fs::path rgb_path, depth_path, campose_path;
    for (const auto &entry : fs::directory_iterator(current_dir)) {
      std::string name = entry.path().filename().string();
      if (name.find("rgb_img_") == 0) rgb_path = entry.path();
      else if (name.find("depth_") == 0) depth_path = entry.path();
      else if (name.find("campose_") == 0) campose_path = entry.path();
    }

    auto timestamp = this->now();

    if (!rgb_path.empty()) publishRgb(rgb_path, timestamp);
    if (!depth_path.empty()) publishDepth(depth_path, timestamp);
    if (!campose_path.empty()) publishPose(campose_path, timestamp);

    // “Clear” temporary memory by just letting local vars go out of scope
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

  void publishPose(const fs::path &path, const rclcpp::Time &stamp)
  {
    std::ifstream file(path);
    if (!file.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Cannot open campose file: %s", path.string().c_str());
      return;
    }

    std::string line;
    std::getline(file, line); // skip header
    std::getline(file, line); // data line
    file.close();

    std::stringstream ss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(ss, token, ',')) tokens.push_back(token);

    if (tokens.size() < 9) return;

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = frame_id_;
    pose_msg.pose.position.x = std::stod(tokens[2]);
    pose_msg.pose.position.y = std::stod(tokens[3]);
    pose_msg.pose.position.z = std::stod(tokens[4]);
    pose_msg.pose.orientation.x = std::stod(tokens[5]);
    pose_msg.pose.orientation.y = std::stod(tokens[6]);
    pose_msg.pose.orientation.z = std::stod(tokens[7]);
    pose_msg.pose.orientation.w = std::stod(tokens[8]);

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
