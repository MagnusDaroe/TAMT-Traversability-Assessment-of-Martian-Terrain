#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <regex>

class SanityTestNode : public rclcpp::Node
{
public:
    SanityTestNode()
    : Node("sanity_test_simple"), frame_counter_(0)
    {
        rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/sync_rgb", rclcpp::SensorDataQoS(),
            std::bind(&SanityTestNode::rgbCallback, this, std::placeholders::_1));

        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/sync_depth", rclcpp::SensorDataQoS(),
            std::bind(&SanityTestNode::depthCallback, this, std::placeholders::_1));

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/sync_cam_2_glob_pose", 10,
            std::bind(&SanityTestNode::poseCallback, this, std::placeholders::_1));

        pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/sync_pointcloud", 10,
            std::bind(&SanityTestNode::pointcloudCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "SanityTestNode started, saving RGB, Depth, Pose, and PointCloud.");
    }

private:
    void trySaveFrame()
    {
        // wait until all data is ready
        if (!latest_rgb_ || !latest_depth_ || !latest_pose_ || !latest_pc_)
            return;

        namespace fs = std::filesystem;
        fs::path base_dir = fs::current_path() / "sanity_data";
        fs::create_directories(base_dir);

        // create frame-specific folder
        frame_counter_++;
        std::stringstream frame_folder_ss;
        frame_folder_ss << "sanity_data_" << frame_counter_;
        fs::path frame_dir = base_dir / frame_folder_ss.str();
        fs::create_directories(frame_dir);

        // filename index
        std::stringstream idx;
        idx << std::setw(5) << std::setfill('0') << frame_counter_;

        // ---- Save RGB ----
        try {
            cv::Mat img = cv_bridge::toCvCopy(latest_rgb_, "bgr8")->image;
            fs::path rgb_path = frame_dir / ("rgb_img_" + idx.str() + ".png");
            cv::imwrite(rgb_path.string(), img);
        } catch (...) {
            RCLCPP_WARN(this->get_logger(), "Failed to save RGB image");
        }

        // ---- Save Depth as CSV ----
        try {
            cv::Mat depth = cv_bridge::toCvCopy(latest_depth_, latest_depth_->encoding)->image;
            depth.convertTo(depth, CV_32F);
            fs::path depth_path = frame_dir / ("depth_" + idx.str() + ".csv");
            std::ofstream ofs(depth_path);
            for (int i = 0; i < depth.rows; ++i) {
                for (int j = 0; j < depth.cols; ++j) {
                    if (j) ofs << ",";
                    ofs << depth.at<float>(i,j);
                }
                ofs << "\n";
            }
        } catch (...) {
            RCLCPP_WARN(this->get_logger(), "Failed to save depth CSV");
        }

        // ---- Save Camera Pose as CSV ----
        try {
            fs::path pose_path = frame_dir / ("campose_" + idx.str() + ".csv");
            std::ofstream csv(pose_path);
            csv << "x,y,z,qx,qy,qz,qw\n";
            csv << latest_pose_->pose.position.x << ","
                << latest_pose_->pose.position.y << ","
                << latest_pose_->pose.position.z << ","
                << latest_pose_->pose.orientation.x << ","
                << latest_pose_->pose.orientation.y << ","
                << latest_pose_->pose.orientation.z << ","
                << latest_pose_->pose.orientation.w << "\n";
        } catch (...) {
            RCLCPP_WARN(this->get_logger(), "Failed to save camera pose CSV");
        }

        // ---- Save PointCloud as .xyz ----
        try {
            fs::path pc_path = frame_dir / ("pointcloud_" + idx.str() + ".xyz");
            std::ofstream pc_file(pc_path);
            sensor_msgs::PointCloud2ConstIterator<float> iter_x(*latest_pc_, "x");
            sensor_msgs::PointCloud2ConstIterator<float> iter_y(*latest_pc_, "y");
            sensor_msgs::PointCloud2ConstIterator<float> iter_z(*latest_pc_, "z");
            for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
                pc_file << *iter_x << " " << *iter_y << " " << *iter_z << "\n";
            }
        } catch (...) {
            RCLCPP_WARN(this->get_logger(), "Failed to save point cloud");
        }

        RCLCPP_INFO(this->get_logger(), "Saved frame %d", frame_counter_);

        // Clear data for next frame
        latest_rgb_.reset();
        latest_depth_.reset();
        latest_pose_.reset();
        latest_pc_.reset();
    }

    void rgbCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        latest_rgb_ = msg;
        trySaveFrame();
    }

    void depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        latest_depth_ = msg;
        trySaveFrame();
    }

    void poseCallback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg)
    {
        latest_pose_ = msg;
        trySaveFrame();
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
    {
        latest_pc_ = msg;
        trySaveFrame();
    }

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;

    // Latest messages
    sensor_msgs::msg::Image::ConstSharedPtr latest_rgb_;
    sensor_msgs::msg::Image::ConstSharedPtr latest_depth_;
    geometry_msgs::msg::PoseStamped::ConstSharedPtr latest_pose_;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_pc_;

    int frame_counter_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SanityTestNode>());
    rclcpp::shutdown();
    return 0;
}
