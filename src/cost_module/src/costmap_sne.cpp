#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cmath>
#include <vector>
#include <memory>
#include <limits>

class CostmapSNE : public rclcpp::Node
{
public:
    CostmapSNE() : Node("costmap_sne")
    {
        // Declare parameters (values will come from YAML file)
        this->declare_parameter("camera.fov_x");
        this->declare_parameter("camera.fov_y");
        this->declare_parameter("camera.mounting_angle");
        this->declare_parameter("camera.height");
        this->declare_parameter("rover.width");
        this->declare_parameter("rover.length");
        
        // Get parameters from YAML file
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
        
        // Subscribe to surface normals topic
        surface_normals_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/surface_normals",
            10,
            std::bind(&CostmapSNE::surfaceNormalsCallback, this, std::placeholders::_1)
        );
        
        // Subscribe to synchronized pointcloud topic
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/sync_pointcloud",
            10,
            std::bind(&CostmapSNE::pointcloudCallback, this, std::placeholders::_1)
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
    
    void surfaceNormalsCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Validate image format
        if (msg->encoding != "32FC3")
        {
            RCLCPP_ERROR(this->get_logger(), "Expected encoding 32FC3, got %s", msg->encoding.c_str());
            return;
        }
        
        // Check if we have received a pointcloud
        if (!sync_pointcloud_)
        {
            RCLCPP_WARN(this->get_logger(), "No pointcloud received yet, skipping normal processing");
            return;
        }
        
        uint32_t width = msg->width;
        uint32_t height = msg->height;
        
        // Transform normals to global frame using pointcloud coordinates
        std::vector<float> normals_global = transformToGlobalFrame(msg->data, width, height);
        
        // Compute polar angles from normals and combine with 3D coordinates
        // Output format: [x, y, z, theta] for each point
        std::vector<float> points_with_theta = computePointsWithPolarAngles(normals_global, width, height);
        
        // Compute traversability cost for each point based on polar angle
        std::vector<float> traversability_costs = computeTraversabilityCost(points_with_theta, width, height);
        
        RCLCPP_INFO(this->get_logger(), "Processed surface normals image: %dx%d", width, height);
    }
    
    std::vector<float> transformToGlobalFrame(const std::vector<uint8_t>& normals_data, 
                                               uint32_t width, uint32_t height)
    {
        // Convert byte data to float vector (3 channels: nx, ny, nz)
        size_t num_pixels = width * height;
        std::vector<float> normals_global(num_pixels * 3);
        
        // Reinterpret byte data as float (normals in camera frame)
        const float* normals_camera = reinterpret_cast<const float*>(normals_data.data());
        
        // Parse pointcloud to get 3D coordinates for each pixel
        // PointCloud2 data is stored as a flat array with point_step bytes per point
        const uint8_t* pc_data = sync_pointcloud_->data.data();
        uint32_t point_step = sync_pointcloud_->point_step;
        
        // Calculate rotation matrix based on camera mounting angle
        // Rotation around Y-axis (pitch) by mounting_angle
        double angle_rad = mounting_angle_ * M_PI / 180.0;
        double cos_angle = std::cos(angle_rad);
        double sin_angle = std::sin(angle_rad);
        
        // Process each pixel/normal
        for (size_t i = 0; i < num_pixels; ++i)
        {
            // Get 3D coordinates from pointcloud (assuming XYZ fields at offset 0, 4, 8)
            const float* point_ptr = reinterpret_cast<const float*>(pc_data + i * point_step);
            float x_cam = point_ptr[0];
            float y_cam = point_ptr[1];
            float z_cam = point_ptr[2];
            
            // Get normal vector in camera frame
            float nx_cam = normals_camera[i * 3 + 0];
            float ny_cam = normals_camera[i * 3 + 1];
            float nz_cam = normals_camera[i * 3 + 2];
            
            // Transform normal from camera frame to global frame
            // Apply rotation matrix (rotation around Y-axis by mounting_angle)
            // R_y(θ) = [cos(θ)  0  sin(θ)]
            //          [   0    1    0   ]
            //          [-sin(θ) 0  cos(θ)]
            float nx_global = cos_angle * nx_cam + sin_angle * nz_cam;
            float ny_global = ny_cam;
            float nz_global = -sin_angle * nx_cam + cos_angle * nz_cam;
            
            // Store transformed normal
            normals_global[i * 3 + 0] = nx_global;
            normals_global[i * 3 + 1] = ny_global;
            normals_global[i * 3 + 2] = nz_global;
        }
        
        return normals_global;
    }
    
    std::vector<float> computePointsWithPolarAngles(const std::vector<float>& normals, 
                                                    uint32_t width, uint32_t height)
    {
        // Create output vector for (x, y, z, theta) - 4 values per point
        size_t num_pixels = width * height;
        std::vector<float> points_with_theta(num_pixels * 4);
        
        // Parse pointcloud to get 3D coordinates for each pixel
        const uint8_t* pc_data = sync_pointcloud_->data.data();
        uint32_t point_step = sync_pointcloud_->point_step;
        
        // Compute polar angle for each normal vector
        for (size_t i = 0; i < num_pixels; ++i)
        {
            // Get 3D coordinates from pointcloud
            const float* point_ptr = reinterpret_cast<const float*>(pc_data + i * point_step);
            float x = point_ptr[0];
            float y = point_ptr[1];
            float z = point_ptr[2];
            
            // Get normal vector components (nx, ny, nz) - these are the transformed global normals
            float nx = normals[i * 3 + 0];
            float ny = normals[i * 3 + 1];
            float nz = normals[i * 3 + 2];
            
            float theta;
            
            // Check if all normal components are zero
            if (std::abs(nx) < 1e-9f && std::abs(ny) < 1e-9f && std::abs(nz) < 1e-9f)
            {
                theta = std::numeric_limits<float>::quiet_NaN();
            }
            else
            {
                // Compute polar angle (theta) in spherical coordinates using the normal vector
                // θ = arccos(nz / √(nx² + ny² + nz²))
                float magnitude = std::sqrt(nx*nx + ny*ny + nz*nz);
                theta = std::acos(nz / magnitude);
            }
            
            // Store x, y, z, theta for this point
            points_with_theta[i * 4 + 0] = x;
            points_with_theta[i * 4 + 1] = y;
            points_with_theta[i * 4 + 2] = z;
            points_with_theta[i * 4 + 3] = theta;
        }
        
        return points_with_theta;
    }
    
    std::vector<float> computeTraversabilityCost(const std::vector<float>& points_with_theta,
                                                  uint32_t width, uint32_t height)
    {
        // Create output vector for traversability costs
        size_t num_pixels = width * height;
        std::vector<float> costs(num_pixels);
        
        // Cost function: C = 103.35 * theta²
        const float cost_coefficient = 103.35f;
        
        // Compute cost for each point
        for (size_t i = 0; i < num_pixels; ++i)
        {
            // Get theta from the points_with_theta vector
            // Format: [x, y, z, theta] per point
            float theta = points_with_theta[i * 4 + 3];
            
            // Check if theta is NaN
            if (std::isnan(theta))
            {
                costs[i] = std::numeric_limits<float>::quiet_NaN();
            }
            else
            {
                // Compute cost: C = 103.35 * theta² (theta in radians)
                costs[i] = cost_coefficient * theta * theta;
            }
        }
        
        return costs;
    }

    
    
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr surface_normals_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    
    // Synchronized pointcloud data
    sensor_msgs::msg::PointCloud2::SharedPtr sync_pointcloud_;
    
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
