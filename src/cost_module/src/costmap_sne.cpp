#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
#include <nav2_msgs/msg/costmap.hpp>
#include <cmath>
#include <vector>
#include <memory>
#include <limits>

class CostmapSNE : public rclcpp::Node
{
public:
    CostmapSNE() : Node("costmap_sne")
    {
        // Declare parameters with default values (can be overridden by YAML file)
        this->declare_parameter("camera.fov_x", 90.0);
        this->declare_parameter("camera.fov_y", 60.0);
        this->declare_parameter("camera.mounting_angle", 0.0);
        this->declare_parameter("camera.height", 1.0);
        this->declare_parameter("rover.width", 1.0);
        this->declare_parameter("rover.length", 1.5);
        
        // Get parameters from YAML file (or use defaults)
        fov_x_ = this->get_parameter("camera.fov_x").as_double();
        fov_y_ = this->get_parameter("camera.fov_y").as_double();
        mounting_angle_ = this->get_parameter("camera.mounting_angle").as_double();
        camera_height_ = this->get_parameter("camera.height").as_double();
        rover_width_ = this->get_parameter("rover.width").as_double();
        rover_length_ = this->get_parameter("rover.length").as_double();
        
        RCLCPP_INFO(this->get_logger(), "Loaded camera parameters - FOV X: %.1f°, FOV Y: %.1f°, Mounting angle: %.1f°, Height: %.2fm",
                    fov_x_, fov_y_, mounting_angle_, camera_height_);
        RCLCPP_INFO(this->get_logger(), "Loaded rover parameters - Width: %.2fm, Length: %.2fm",
                    rover_width_, rover_length_);
        
        // Subscribe to synchronized pointcloud topic
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/sync_pointcloud",
            10,
            std::bind(&CostmapSNE::pointcloudCallback, this, std::placeholders::_1)
        );

        // Subscribe to synchronized pose topic
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/sync_cam_2_glob_pose",
            10,
            std::bind(&CostmapSNE::poseCallback, this, std::placeholders::_1)
        );

        // Subscribe to surface normals topic
        surface_normals_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/surface_normals",
            10,
            std::bind(&CostmapSNE::surfaceNormalsCallback, this, std::placeholders::_1)
        );

        // Create publisher for costmap
        costmap_pub_ = this->create_publisher<nav2_msgs::msg::Costmap>(
            "/costmap_sne",
            10
        );

        RCLCPP_INFO(this->get_logger(), "CostmapSNE node initialized");
    }

private:
    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Store the latest pointcloud
        sync_pointcloud_ = msg;
        RCLCPP_DEBUG(this->get_logger(), "Received pointcloud with %d points", msg->width * msg->height);
    }
    
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {        
        // Process camera to global pose
        RCLCPP_DEBUG(this->get_logger(), "Received camera to global pose at time %.2f", 
                     msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);

        // Store timestamp for use in costmap message
        latest_pose_timestamp_ = msg->header.stamp;

        // Convert PoseStamped to tf2::Transform
        tf2::Vector3 translation(
            msg->pose.position.x,
            msg->pose.position.y,
            msg->pose.position.z
        );
        
        tf2::Quaternion rotation(
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z,
            msg->pose.orientation.w
        );
        
        // Create the transformation from camera to global frame
        cam_to_global_transform_.setOrigin(translation);
        cam_to_global_transform_.setRotation(rotation);
        
        // Apply 208 degree rotation around x-axis
        tf2::Quaternion rotation_x;
        //rotation_x.setRPY(180.0 * M_PI / 180.0, 0, 0); // 180 degrees around x-axis
        rotation_x.setRPY(208.0 * M_PI / 180.0, 0, 0); // 208 degrees around x-axis //! Does the 28 degree calibration
        tf2::Transform rotation_transform;
        rotation_transform.setOrigin(tf2::Vector3(0, 0, 0));
        rotation_transform.setRotation(rotation_x);
        cam_to_global_transform_ = cam_to_global_transform_ * rotation_transform;

        //cam_to_global_transform_ = cam_to_global_transform_.inverse(); //! Should be deleted, when the correct transform is published
        
        
        RCLCPP_DEBUG(this->get_logger(), 
                     "Updated camera to global transform: Translation [%.2f, %.2f, %.2f]",
                     translation.x(), translation.y(), translation.z());
    }

    void surfaceNormalsCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Validate image format
        if (msg->encoding != "32FC3")
        {
            RCLCPP_ERROR(this->get_logger(), "Expected encoding 32FC3, got %s", msg->encoding.c_str());
            return;
        }
        
        // Check if we have received a sync_pointcloud_
        if (!sync_pointcloud_)
        {
            RCLCPP_WARN(this->get_logger(), "No pointcloud received yet, skipping normal processing");
            return;
        }

        // Normals in camera frame - convert from image data to float vector
        const float* normals_ptr = reinterpret_cast<const float*>(msg->data.data());
        std::vector<float> normals_camera(normals_ptr, normals_ptr + (msg->width * msg->height * 3));

        uint32_t width = msg->width;
        uint32_t height = msg->height;
        
        // Debug: Check if normals data contains valid non-zero values
        size_t total_normals = width * height;
        size_t non_zero_count = 0;
        size_t nan_count = 0;
        
        for (size_t i = 0; i < total_normals; ++i)
        {
            float nx = normals_camera[i * 3 + 0];
            float ny = normals_camera[i * 3 + 1];
            float nz = normals_camera[i * 3 + 2];
            
            if (std::isnan(nx) || std::isnan(ny) || std::isnan(nz))
            {
                nan_count++;
            }
            else if (nx != 0.0f || ny != 0.0f || nz != 0.0f)
            {
                non_zero_count++;
            }
        }
        
        RCLCPP_INFO(this->get_logger(), 
                    "Normals validation: Total=%zu, Non-zero=%zu (%.1f%%), NaN=%zu (%.1f%%), Zero=%zu (%.1f%%)",
                    total_normals, non_zero_count, (non_zero_count * 100.0 / total_normals),
                    nan_count, (nan_count * 100.0 / total_normals),
                    total_normals - non_zero_count - nan_count, 
                    ((total_normals - non_zero_count - nan_count) * 100.0 / total_normals));

        // Combine pointcloud with normals
        std::vector<float> points_with_normals = combinePointcloudWithNormals(normals_camera, width, height);
        // Print x,y,z,nx,ny,nz for pixels at (36-38, 283)
        if (width > 38 && height > 283)
        {
            RCLCPP_INFO(this->get_logger(), "Pixels at row 283, columns 36-38:");
            for (size_t u = 36; u <= 38; ++u)
            {
                size_t v = 283;
                size_t idx = v * static_cast<size_t>(width) + u;
                size_t base = idx * 6; // 6 values per point: x,y,z,nx,ny,nz

                if (base + 5 < points_with_normals.size())
                {
                    float px = points_with_normals[base + 0];
                    float py = points_with_normals[base + 1];
                    float pz = points_with_normals[base + 2];
                    float nx = points_with_normals[base + 3];
                    float ny = points_with_normals[base + 4];
                    float nz = points_with_normals[base + 5];

                    RCLCPP_INFO(this->get_logger(),
                            "  Pixel (u=%zu, v=%zu, idx=%zu): x=%.6f y=%.6f z=%.6f nx=%.6f ny=%.6f nz=%.6f",
                            u, v, idx, px, py, pz, nx, ny, nz);
                }
            }
        }
        // Check if combining failed (returns empty vector on error)
        if (points_with_normals.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to combine pointcloud with normals");
            return;
        }

        // Transform normals to global frame using pointcloud coordinates
        std::vector<float> points_with_normals_global = transformToGlobalFrame(points_with_normals, width, height);
        RCLCPP_INFO(this->get_logger(), "Transformed normals to global frame");
        // Compute polar angles from normals and combine with 3D coordinates
        // Output format: [x, y, z, theta] for each point
        std::vector<float> points_with_theta_global = computePolarAngles(points_with_normals_global, width, height);
        
        // Compute traversability cost for each point based on polar angle
        std::vector<float> traversability_costs = computeTraversabilityCost(points_with_theta_global, width, height);

        // Publish costmap
        publishCostmap(traversability_costs, width, height);

        RCLCPP_INFO(this->get_logger(), "Processed surface normals image: %dx%d", width, height);
    }
    
    // Combines pointcloud XYZ coordinates with their corresponding surface normals
    // Returns a vector with format: [x, y, z, nx, ny, nz] for each point
    std::vector<float> combinePointcloudWithNormals(const std::vector<float>& normals_camera,
                                                     uint32_t width, uint32_t height)
    {
        // Ensure pointcloud size matches normals size
        if (sync_pointcloud_->width != width || sync_pointcloud_->height != height)
        {
            RCLCPP_ERROR(this->get_logger(), 
                         "Size mismatch! Normals: %dx%d, Pointcloud: %dx%d",
                         width, height, 
                         sync_pointcloud_->width, sync_pointcloud_->height);
            return std::vector<float>();
        }
        
        size_t num_pixels = width * height;
        std::vector<float> points_with_normals(num_pixels * 6); // 6 values per point: x, y, z, nx, ny, nz
        
        // Parse pointcloud to get 3D coordinates for each pixel
        const uint8_t* pc_data = sync_pointcloud_->data.data();
        uint32_t point_step = sync_pointcloud_->point_step;
        
        // Iterate through each point
        for (size_t i = 0; i < num_pixels; ++i)
        {
            // Get 3D coordinates from pointcloud (assuming XYZ fields at offset 0, 4, 8)
            const float* point_ptr = reinterpret_cast<const float*>(pc_data + i * point_step);
            float x = point_ptr[0];
            float y = point_ptr[1];
            float z = point_ptr[2];
            
            // Get corresponding normal vector from normals_camera
            float nx = normals_camera[i * 3 + 0];
            float ny = normals_camera[i * 3 + 1];
            float nz = normals_camera[i * 3 + 2];
            
            // Store combined data: [x, y, z, nx, ny, nz]
            points_with_normals[i * 6 + 0] = x;
            points_with_normals[i * 6 + 1] = y;
            points_with_normals[i * 6 + 2] = z;
            points_with_normals[i * 6 + 3] = nx;
            points_with_normals[i * 6 + 4] = ny;
            points_with_normals[i * 6 + 5] = nz;
        }
        
        return points_with_normals;
    }


    std::vector<float> transformToGlobalFrame(const std::vector<float>& points_with_normals, 
                                              uint32_t width, uint32_t height)
    {
        size_t num_pixels = width * height;
        std::vector<float> points_with_normals_global(num_pixels * 6); // 6 values per point: x, y, z, nx, ny, nz
        
        // Print 3 points at row 283, columns 36-38 BEFORE transformation
        if (height > 283 && width > 38)
        {
            RCLCPP_INFO(this->get_logger(), "BEFORE transformation - Points at row 283, columns 36-38:");
            for (size_t u = 36; u <= 38; ++u)
            {
                size_t v = 283;
                size_t idx = v * width + u;
                float px = points_with_normals[idx * 6 + 0];
                float py = points_with_normals[idx * 6 + 1];
                float pz = points_with_normals[idx * 6 + 2];
                float nx = points_with_normals[idx * 6 + 3];
                float ny = points_with_normals[idx * 6 + 4];
                float nz = points_with_normals[idx * 6 + 5];
                RCLCPP_INFO(this->get_logger(), 
                           "  Point[%zu] (u=%zu, v=%zu): x=%.6f y=%.6f z=%.6f nx=%.6f ny=%.6f nz=%.6f",
                           idx, u, v, px, py, pz, nx, ny, nz);
            }
        }

        // Transform each point and its normal vector to the global frame
        for (size_t i = 0; i < num_pixels; ++i)
        {
            // Extract point coordinates in camera frame
            float x_cam = points_with_normals[i * 6 + 0];
            float y_cam = points_with_normals[i * 6 + 1];
            float z_cam = points_with_normals[i * 6 + 2];
            
            // Extract normal vector in camera frame
            float nx_cam = points_with_normals[i * 6 + 3];
            float ny_cam = points_with_normals[i * 6 + 4];
            float nz_cam = points_with_normals[i * 6 + 5];
            
            // Transform point to global frame
            tf2::Vector3 point_cam(x_cam, y_cam, z_cam);
            tf2::Vector3 point_global = cam_to_global_transform_ * point_cam;
            
            // Transform normal vector to global frame (rotation only, no translation)
            tf2::Vector3 normal_cam(nx_cam, ny_cam, nz_cam);
            tf2::Vector3 normal_global = cam_to_global_transform_.getBasis() * normal_cam;
            
            // Store transformed data: [x, y, z, nx, ny, nz] in global frame
            points_with_normals_global[i * 6 + 0] = point_global.x();
            points_with_normals_global[i * 6 + 1] = point_global.y();
            points_with_normals_global[i * 6 + 2] = point_global.z();
            points_with_normals_global[i * 6 + 3] = normal_global.x();
            points_with_normals_global[i * 6 + 4] = normal_global.y();
            points_with_normals_global[i * 6 + 5] = normal_global.z();
        }

        // Print 3 points at row 283, columns 36-38 AFTER transformation
        if (height > 283 && width > 38)
        {
            RCLCPP_INFO(this->get_logger(), "AFTER transformation - Points at row 283, columns 36-38 (GLOBAL frame):");
            for (size_t u = 36; u <= 38; ++u)
            {
                size_t v = 283;
                size_t idx = v * width + u;
                float px = points_with_normals_global[idx * 6 + 0];
                float py = points_with_normals_global[idx * 6 + 1];
                float pz = points_with_normals_global[idx * 6 + 2];
                float nx = points_with_normals_global[idx * 6 + 3];
                float ny = points_with_normals_global[idx * 6 + 4];
                float nz = points_with_normals_global[idx * 6 + 5];
                RCLCPP_INFO(this->get_logger(), 
                           "  Point[%zu] (u=%zu, v=%zu): x=%.6f y=%.6f z=%.6f nx=%.6f ny=%.6f nz=%.6f",
                           idx, u, v, px, py, pz, nx, ny, nz);
            }
        }
        
        return points_with_normals_global;
    }
    
    std::vector<float> computePolarAngles(const std::vector<float>& points_with_normals_global, 
                                                    uint32_t width, uint32_t height)
    {
        size_t num_pixels = width * height;
        std::vector<float> points_with_theta_global(num_pixels * 4); // 4 values per point: x, y, z, theta
        
        // Compute polar angle for each point's normal vector
        for (size_t i = 0; i < num_pixels; ++i)
        {
            // Extract point coordinates in global frame
            float x_global = points_with_normals_global[i * 6 + 0];
            float y_global = points_with_normals_global[i * 6 + 1];
            float z_global = points_with_normals_global[i * 6 + 2];
            
            // Extract normal vector components in global frame
            float nx_global = points_with_normals_global[i * 6 + 3];
            float ny_global = points_with_normals_global[i * 6 + 4];
            float nz_global = points_with_normals_global[i * 6 + 5];
            
            // Compute polar angle θ (theta) using arctan formula
            // θ = arctan(√(nx² + ny²) / nz)
            float xy_magnitude = std::sqrt(nx_global * nx_global + ny_global * ny_global);
            float theta = std::atan2(xy_magnitude, nz_global);
            
            // Store combined data: [x, y, z, theta] in global frame
            points_with_theta_global[i * 4 + 0] = x_global;
            points_with_theta_global[i * 4 + 1] = y_global;
            points_with_theta_global[i * 4 + 2] = z_global;
            points_with_theta_global[i * 4 + 3] = theta;
        }
        
        // Print theta for pixels at row 283, columns 36-38
        if (width > 38 && height > 283)
        {
            RCLCPP_INFO(this->get_logger(), "Theta values for pixels at row 283, columns 36-38:");
            for (size_t u = 36; u <= 38; ++u)
            {
                size_t v = 283;
                size_t idx = v * static_cast<size_t>(width) + u;
                float x = points_with_theta_global[idx * 4 + 0];
                float y = points_with_theta_global[idx * 4 + 1];
                float z = points_with_theta_global[idx * 4 + 2];
                float theta = points_with_theta_global[idx * 4 + 3];
                RCLCPP_INFO(this->get_logger(), "  Pixel[%zu] (u=%zu, v=%zu): x=%.3f y=%.3f z=%.3f theta=%.6f", 
                           idx, u, v, x, y, z, theta);
            }
        }

        return points_with_theta_global;
    }
    
    std::vector<float> computeTraversabilityCost(const std::vector<float>& points_with_theta_global,
                                                  uint32_t width, uint32_t height)
    {
        // Create output vector for points with traversability costs
        // Format: [x, y, z, cost] for each point
        size_t num_pixels = width * height;
        std::vector<float> points_with_costs(num_pixels * 4); // 4 values per point: x, y, z, cost
        
        // Cost function: C = 102.94 * theta²
        const float cost_coefficient = 102.94f;
        
        // Compute cost for each point
        for (size_t i = 0; i < num_pixels; ++i)
        {
            // Get point coordinates from the points_with_theta_global vector
            // Input format: [x, y, z, theta] per point
            float x_global = points_with_theta_global[i * 4 + 0];
            float y_global = points_with_theta_global[i * 4 + 1];
            float z_global = points_with_theta_global[i * 4 + 2];
            float theta = points_with_theta_global[i * 4 + 3];
            
            // Compute cost
            float cost;
            if (std::isnan(theta) || theta == 0.0f)
            {
                cost = 255.0f;
            }
            else
            {
                // Compute cost: C = 102.94 * theta² (theta in radians)
                cost = cost_coefficient * theta * theta;
            }
            
            // Store combined data: [x, y, z, cost] in global frame
            points_with_costs[i * 4 + 0] = x_global;
            points_with_costs[i * 4 + 1] = y_global;
            points_with_costs[i * 4 + 2] = z_global;
            points_with_costs[i * 4 + 3] = cost;
        }
        
        // Print cost for pixels at row 283, columns 36-38
        if (width > 38 && height > 283)
        {
            RCLCPP_INFO(this->get_logger(), "Cost values for pixels at row 283, columns 36-38:");
            for (size_t u = 36; u <= 38; ++u)
            {
                size_t v = 283;
                size_t idx = v * static_cast<size_t>(width) + u;
                float x = points_with_costs[idx * 4 + 0];
                float y = points_with_costs[idx * 4 + 1];
                float z = points_with_costs[idx * 4 + 2];
                float cost = points_with_costs[idx * 4 + 3];
                RCLCPP_INFO(this->get_logger(), "  Pixel[%zu] (u=%zu, v=%zu): x=%.3f y=%.3f z=%.3f cost=%.6f", 
                           idx, u, v, x, y, z, cost);
            }
        }

        //TODO make into correct type for costmap (now it is x,y,z,cost)
        //TODO Assign 255 when theta is 0 or nan and figure out what to do when x and y are nan using the nav_msgs/msg/OccupancyGrid type
        return points_with_costs;
    }

    void publishCostmap(const std::vector<float>& points_with_costs, uint32_t width, uint32_t height)
    {
        // Create Costmap message
        auto costmap_msg = nav2_msgs::msg::Costmap();
        
        // Set header
        costmap_msg.header.stamp = latest_pose_timestamp_;
        costmap_msg.header.frame_id = "map"; // or use your global frame
        
        // Set metadata
        costmap_msg.metadata.size_x = width;
        costmap_msg.metadata.size_y = height;
        costmap_msg.metadata.resolution = 0.05; // 5cm per cell - adjust as needed
        
        // Set origin (position of cell (0,0) in the map frame)
        // Use the first point's coordinates as the origin
        costmap_msg.metadata.origin.position.x = points_with_costs[0]; // x of first point
        costmap_msg.metadata.origin.position.y = points_with_costs[1]; // y of first point
        costmap_msg.metadata.origin.position.z = points_with_costs[2]; // z of first point
        
        // Set orientation from cam_to_global_transform_
        tf2::Quaternion origin_orientation = cam_to_global_transform_.getRotation();
        costmap_msg.metadata.origin.orientation.x = origin_orientation.x();
        costmap_msg.metadata.origin.orientation.y = origin_orientation.y();
        costmap_msg.metadata.origin.orientation.z = origin_orientation.z();
        costmap_msg.metadata.origin.orientation.w = origin_orientation.w();
        
        // Allocate data array
        costmap_msg.data.resize(width * height);
        
        // Convert cost values to costmap format (0-254 scale, 255 for lethal obstacle, UNKNOWN for invalid)
        for (size_t i = 0; i < width * height; ++i)
        {
            float cost = points_with_costs[i * 4 + 3]; // Extract cost from [x, y, z, cost]
            
            if (std::isnan(cost))
            {
                // Unknown/invalid cells - use NO_INFORMATION or FREE_SPACE
                costmap_msg.data[i] = 255; // NO_INFORMATION in nav2
            }
            else if (cost >= 255.0f)
            {
                // Maximum cost (lethal obstacle)
                costmap_msg.data[i] = 254; // LETHAL_OBSTACLE
            }
            else
            {
                // Scale cost to 0-254 range
                // Assuming max cost before clamping is around 102.94 * (π/2)² ≈ 254
                uint8_t costmap_value = static_cast<uint8_t>(std::min(254.0f, cost));
                costmap_msg.data[i] = costmap_value;
            }
        }
        
        // Publish the costmap
        costmap_pub_->publish(costmap_msg);
        RCLCPP_DEBUG(this->get_logger(), "Published costmap with %dx%d cells", width, height);
    }

    
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr surface_normals_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    
    // Publisher
    rclcpp::Publisher<nav2_msgs::msg::Costmap>::SharedPtr costmap_pub_;
    
    // Synchronized pointcloud data
    sensor_msgs::msg::PointCloud2::SharedPtr sync_pointcloud_;

    // Camera to global transformation (tf2::Transform)
    tf2::Transform cam_to_global_transform_;
    
    // Latest pose timestamp for costmap synchronization
    rclcpp::Time latest_pose_timestamp_;
    
    // Camera parameters
    double fov_x_;
    double fov_y_;
    double mounting_angle_;
    double camera_height_;
    
    // Rover parameters
    double rover_width_;
    double rover_length_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CostmapSNE>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


//TODO average cost over pixels to fit cell size of costmap
//TODO Figure out size of costmap and set metadata accordingly (depth x something with FOV)
//TODO Find coordinates of costmap origin (pixel 0,0 in global frame)
//TODO Create service client to call sync_node to get synchronized data