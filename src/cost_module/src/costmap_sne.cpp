#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
#include <nav2_msgs/msg/costmap.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tuple>
#include <algorithm>
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
        this->declare_parameter("camera.height", 1.0);
        this->declare_parameter("camera.tilt_angle", 20.0);
        this->declare_parameter("camera.fov_x", 90.0);
        this->declare_parameter("camera.fov_y", 60.0);
        this->declare_parameter("camera.max_distance", 5.0);
        
        this->declare_parameter("camera.transform.translation.x", 0.0);
        this->declare_parameter("camera.transform.translation.y", 0.0);
        this->declare_parameter("camera.transform.translation.z", 0.0);
        this->declare_parameter("camera.transform.rotation.x", 0.0);
        this->declare_parameter("camera.transform.rotation.y", 0.0);
        this->declare_parameter("camera.transform.rotation.z", 0.0);
        this->declare_parameter("camera.transform.rotation.w", 1.0);
        
        this->declare_parameter("rover.width", 1.0);
        this->declare_parameter("rover.length", 1.5);
        this->declare_parameter("costmap.resolution", 0.05);
        
        // Get parameters from YAML file (or use defaults)
        camera_height_ = this->get_parameter("camera.height").as_double();
        tilt_angle_ = this->get_parameter("camera.tilt_angle").as_double();
        fov_x_ = this->get_parameter("camera.fov_x").as_double();
        fov_y_ = this->get_parameter("camera.fov_y").as_double();
        max_distance_ = this->get_parameter("camera.max_distance").as_double();

        // Unpack static transform from rover frame to camera frame (when horizontal)
        tf2::Quaternion q_cam_to_rover_horizontal(
            this->get_parameter("camera.transform.rotation.x").as_double(), 
            this->get_parameter("camera.transform.rotation.y").as_double(),
            this->get_parameter("camera.transform.rotation.z").as_double(),
            this->get_parameter("camera.transform.rotation.w").as_double()
        );

        // Apply tilt rotation around the rover Y-axis
        tf2::Quaternion q_tilt;
        q_tilt.setRPY(0, tilt_angle_ * M_PI / 180.0, 0);  // Rotation around Y-axis

        // Combine: tilt is applied in rover frame, so multiply in this order
        tf2::Quaternion q_cam_to_rover = q_tilt * q_cam_to_rover_horizontal;

        // Set the complete transform
        cam_x_to_rover_transform_.setRotation(q_cam_to_rover);
        cam_x_to_rover_transform_.setOrigin(tf2::Vector3(
            this->get_parameter("camera.transform.translation.x").as_double(),
            this->get_parameter("camera.transform.translation.y").as_double(),
            this->get_parameter("camera.transform.translation.z").as_double()
        ));
        rover_width_ = this->get_parameter("rover.width").as_double();
        rover_length_ = this->get_parameter("rover.length").as_double();
        resolution_ = this->get_parameter("costmap.resolution").as_double();

        RCLCPP_INFO(this->get_logger(), "Loaded camera parameters - FOV X: %.1f°, FOV Y: %.1f°, Max distance: %.2fm",
                    fov_x_, fov_y_, max_distance_);
        RCLCPP_INFO(this->get_logger(), "Loaded rover parameters - Width: %.2fm, Length: %.2fm",
                    rover_width_, rover_length_);
        RCLCPP_INFO(this->get_logger(), "Loaded costmap parameters - Resolution: %.3fm",
                    resolution_);
        
        // Subscribe to synchronized pointcloud topic
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/sync_pointcloud",
            10,
            std::bind(&CostmapSNE::pointcloudCallback, this, std::placeholders::_1)
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

        // Create publisher for visualization in RViz2
        costmap_viz_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/costmap_sne_viz",
            10
        );
        
        costmap_metrics_ = getCostMapMetricsRoverFrame(getCostMapMetrics(fov_x_, fov_y_, camera_height_, tilt_angle_, max_distance_), cam_x_to_rover_transform_, camera_height_);
                
        RCLCPP_INFO(this->get_logger(), "CostmapSNE node initialized");
    }

private:

    struct costMapMetrics 
    {
    double origin[2] = {0.0, 0.0}; // origin[0]: x coordinate of closest left point, origin[1]: y coordinate of closest left point
    double size[2] = {0.0, 0.0}; // size[0]: Height of the map, size[1]: Width of the map
    };

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

        std::vector<float> pointcloud = pointcloudToVector();

        std::vector<float> pointcloud_rover = transformToRoverFrame(pointcloud, width, height, false);
        std::vector<float> normals_rover = transformToRoverFrame(normals_camera, width, height, true);
        

        // Combine pointcloud with normals
        std::vector<float> points_with_normals = combinePointcloudWithNormals(pointcloud_rover, normals_rover, width, height);
        
        // Compute polar angles from normals and combine with 3D coordinates
        // Output format: [x, y, z, theta] for each point
        std::vector<float> points_with_theta_rover = computePolarAngles(points_with_normals_rover, width, height);
        
        // Compute traversability cost for each point based on polar angle
        std::vector<float> traversability_costs = computeTraversabilityCost(points_with_theta_rover, width, height);


        // Create averaged cost grid
        auto [averaged_grid, width_cells, height_cells, origin_x, origin_y] = createAveragedCostGrid(traversability_costs, costmap_metrics_);
        RCLCPP_INFO(this->get_logger(), "Created averaged cost grid: %dx%d cells", width_cells, height_cells);

        // Publish costmap with the actual origin used for binning
        publishCostmap(averaged_grid, width_cells, height_cells, origin_x, origin_y);

        RCLCPP_INFO(this->get_logger(), "Computed and published topography costmap");
    }
    
    costMapMetrics getCostMapMetrics(double fov_horizontal, double fov_vertical, double camera_height, double camera_pitch, double max_ray_length) {
        // Handle when camera is below ground level, handle when max_ray_length is less than camera height
        if (camera_height < 0.0 || max_ray_length <= camera_height) {
            throw std::invalid_argument("Camera height must be non-negative and less than max ray length.");
        }
        else if (fov_horizontal <= 0.0 || fov_vertical <= 0.0) {
            throw std::invalid_argument("Field of view angles must be positive.");
        }
        else if (fov_horizontal >= 180.0 || fov_vertical >= 180.0) {
            throw std::invalid_argument("Field of view angles must be less than 180 degrees.");
        }
        else if (max_ray_length <= 0.0) {
            throw std::invalid_argument("Max ray length must be positive.");
        }
        else if (camera_pitch < 0 || camera_pitch > 90.0) {
            throw std::invalid_argument("Camera pitch must be between 0 and 90 degrees.");
        }

        // Initialize the costMapMetrics structure to hold the results
        costMapMetrics metrics;

        // Calculate the half angle of the horizontal field of view in radians
        double theta_horizontal = (fov_horizontal / 2.0) * M_PI / 180.0;

        // Ray angles:
        double top_ray_angle = (90.0 + (fov_vertical / 2.0) - camera_pitch);
        double bottom_ray_angle = (90.0 - (fov_vertical / 2.0) - camera_pitch);

        // Calculate min and max distances in the x direction. Min is the closest point the camera can see on the ground
        double min_dist_x = std::max(0.0, std::tan(bottom_ray_angle * M_PI / 180.0) * camera_height);
        
        // Check if the top ray goes beyond vertical
        double max_dist_x = 1.0;
        double ray_limit = std::sqrt(max_ray_length * max_ray_length - camera_height * camera_height);
        const double EPSILON = 1e-6; // Small tolerance for floating point comparison
        
        if (top_ray_angle >= 90.0 - EPSILON) {
            max_dist_x = ray_limit; // Ray goes beyond vertical - limit by max ray length
        } else { 
            // Ray hits ground - calculate both limits
            double ground_intersection = std::tan(top_ray_angle * M_PI / 180.0) * camera_height;  
            max_dist_x = std::min(ray_limit, ground_intersection);
        }
        
        // Calculate the y distance based on the max ray length and the horizontal field of view
        double y_ray_restricted = std::tan(theta_horizontal) * std::cos(theta_horizontal) * max_ray_length;
        double y_max_dist_x = std::tan(theta_horizontal) * std::sqrt(max_dist_x*max_dist_x + camera_height*camera_height);
        double y = std::min(y_ray_restricted, y_max_dist_x);
        
        // Calculate size of the map
        double height = max_dist_x - min_dist_x;
        double width = 2.0 * y;

        // Populate the metrics structure
        metrics.origin[0] = min_dist_x;
        metrics.origin[1] = y;

        metrics.size[0] = height;
        metrics.size[1] = width;

        return metrics;
    }
    
    costMapMetrics getCostMapMetricsRoverFrame(costMapMetrics metrics, tf2::Transform tf_camera_to_rover, double camera_height) {
        // Unpack metrics in camera frame
        tf2::Vector3 origin_camera_frame(metrics.origin[0], metrics.origin[1], camera_height);
        
        //Overwrite rotation in tf_camera_to_rover to be identity
        tf2::Matrix3x3 rotation_identity;
        rotation_identity.setIdentity();
        tf2::Transform tf_camera_to_rover_identity(rotation_identity, tf_camera_to_rover.getOrigin());

        // Apply transformation to rover frame
        tf2::Vector3 origin_rover_frame = tf_camera_to_rover_identity * origin_camera_frame;

        costMapMetrics rover_metrics;
        rover_metrics.origin[0] = origin_rover_frame.x();
        rover_metrics.origin[1] = origin_rover_frame.y();
        rover_metrics.size[0] = metrics.size[0]; // Height remains the same
        rover_metrics.size[1] = metrics.size[1]; // Width remains the same
        
        return rover_metrics;
    }


    std::vector<float> pointcloudToVector()
    {
        size_t num_points = sync_pointcloud_->width * sync_pointcloud_->height;
        std::vector<float> points(num_points * 3); // 3 values per point: x, y, z
        
        // Parse pointcloud to get 3D coordinates for each point
        const uint8_t* pc_data = sync_pointcloud_->data.data();
        uint32_t point_step = sync_pointcloud_->point_step;
        
        // Iterate through each point
        for (size_t i = 0; i < num_points; ++i)
        {
            // Get 3D coordinates from pointcloud (assuming XYZ fields at offset 0, 4, 8)
            const float* point_ptr = reinterpret_cast<const float*>(pc_data + i * point_step);
            float x = point_ptr[0];
            float y = point_ptr[1];
            float z = point_ptr[2];

            // Store coordinates
            points[i * 3 + 0] = x;
            points[i * 3 + 1] = y;
            points[i * 3 + 2] = z;
        }

        return points;
    }


    std::vector<float> transformToRoverFrame(const std::vector<float>& points, 
                                              uint32_t width, uint32_t height, bool normals)
    {
        size_t num_pixels = width * height;
        std::vector<float> points_transformed(num_pixels * 3); // 3 values per point: x, y, z

        // Transform each point to the rover frame
        for (size_t i = 0; i < num_pixels; ++i)
        {
            // Extract point coordinates in camera frame
            float x_cam = points[i * 3 + 0];
            float y_cam = points[i * 3 + 1];
            float z_cam = points[i * 3 + 2];
            
            tf2::Vector3 point_cam(x_cam, y_cam, z_cam);
            tf2::Vector3 point_rover;
            if (normals) {
                // Transform normal vector to rover frame (rotation only, no translation)
                point_rover = cam_x_to_rover_transform_.inverse().getBasis() * point_cam;
            }
            else {
                // Transform point to rover frame
                point_rover = cam_x_to_rover_transform_.inverse() * point_cam;
            }
                    
            // Store transformed data: [x, y, z, nx, ny, nz] in rover frame
            points_transformed[i * 3 + 0] = point_rover.x();
            points_transformed[i * 3 + 1] = point_rover.y();
            points_transformed[i * 3 + 2] = point_rover.z();
        }
        
        return points_transformed;
    }


    std::vector<float> combinePointcloudWithNormals(const std::vector<float>& pointcloud_rover, const std::vector<float>& normals_rover,
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
        
        int points_beyond_max_distance = 0;
        
        // Iterate through each point
        for (size_t i = 0; i < num_pixels; ++i)
        {
            float x = pointcloud_rover[i * 3 + 0];
            float y = pointcloud_rover[i * 3 + 1];
            float z = pointcloud_rover[i * 3 + 2];
            
            // Get corresponding normal vector from normals_rover
            float nx = normals_rover[i * 3 + 0];
            float ny = normals_rover[i * 3 + 1];
            float nz = normals_rover[i * 3 + 2];
            
            // Set normals to [0, 0, 0] if z (depth) exceeds max_distance
            if (z > max_distance_ || std::isnan(z) || std::isinf(z))
            {
                nx = 0.0f;
                ny = 0.0f;
                nz = 0.0f;
                points_beyond_max_distance++;
            }
            
            // Store combined data: [x, y, z, nx, ny, nz]
            points_with_normals[i * 6 + 0] = x;
            points_with_normals[i * 6 + 1] = y;
            points_with_normals[i * 6 + 2] = z;
            points_with_normals[i * 6 + 3] = nx;
            points_with_normals[i * 6 + 4] = ny;
            points_with_normals[i * 6 + 5] = nz;
        }

        RCLCPP_DEBUG(this->get_logger(), 
                     "Set normals to [0,0,0] for %d/%zu points beyond max_distance (%.2fm)",
                     points_beyond_max_distance, num_pixels, max_distance_);
        
        
        return points_with_normals;
    }
    
    std::vector<float> computePolarAngles(const std::vector<float>& points_with_normals_rover, 
                                                    uint32_t width, uint32_t height)
    {
        size_t num_pixels = width * height;
        std::vector<float> points_with_theta_rover(num_pixels * 4); // 4 values per point: x, y, z, theta
        
        // Compute polar angle for each point's normal vector
        for (size_t i = 0; i < num_pixels; ++i)
        {
            // Extract point coordinates in rover frame
            float x_rover = points_with_normals_rover[i * 6 + 0];
            float y_rover = points_with_normals_rover[i * 6 + 1];
            float z_rover = points_with_normals_rover[i * 6 + 2];
            
            // Extract normal vector components in rover frame
            float nx_rover = points_with_normals_rover[i * 6 + 3];
            float ny_rover = points_with_normals_rover[i * 6 + 4];
            float nz_rover = points_with_normals_rover[i * 6 + 5];
            
            // Compute polar angle θ (theta) using arctan formula
            // θ = arctan(√(nx² + ny²) / nz)
            float xy_magnitude = std::sqrt(nx_rover * nx_rover + ny_rover * ny_rover);
            float theta = std::atan2(xy_magnitude, nz_rover);
            
            // Store combined data: [x, y, z, theta] in rover frame
            points_with_theta_rover[i * 4 + 0] = x_rover;
            points_with_theta_rover[i * 4 + 1] = y_rover;
            points_with_theta_rover[i * 4 + 2] = z_rover;
            points_with_theta_rover[i * 4 + 3] = theta;
        }
        
        // Print theta for pixels at row 283, columns 36-38
        if (width > 38 && height > 283)
        {
            RCLCPP_INFO(this->get_logger(), "Theta values for pixels at row 283, columns 36-38:");
            for (size_t u = 36; u <= 38; ++u)
            {
                size_t v = 283;
                size_t idx = v * static_cast<size_t>(width) + u;
                float x = points_with_theta_rover[idx * 4 + 0];
                float y = points_with_theta_rover[idx * 4 + 1];
                float z = points_with_theta_rover[idx * 4 + 2];
                float theta = points_with_theta_rover[idx * 4 + 3];
                RCLCPP_INFO(this->get_logger(), "  Pixel[%zu] (u=%zu, v=%zu): x=%.3f y=%.3f z=%.3f theta=%.6f", 
                           idx, u, v, x, y, z, theta);
            }
        }

        return points_with_theta_rover;
    }
    
    std::vector<float> computeTraversabilityCost(const std::vector<float>& points_with_theta_rover,
                                                  uint32_t width, uint32_t height)
    {
        // Create output vector for points with traversability costs
        // Format: [x, y, z, cost] for each point
        size_t num_pixels = width * height;
        std::vector<float> points_with_costs(num_pixels * 4); // 4 values per point: x, y, z, cost
        
        // Cost function: C = 102.54 * theta²
        const float cost_coefficient = 102.54f;
        
        // Compute cost for each point
        for (size_t i = 0; i < num_pixels; ++i)
        {
            // Get point coordinates from the points_with_theta_rover vector
            // Input format: [x, y, z, theta] per point
            float x_rover = points_with_theta_rover[i * 4 + 0];
            float y_rover = points_with_theta_rover[i * 4 + 1];
            float z_rover = points_with_theta_rover[i * 4 + 2];
            float theta = points_with_theta_rover[i * 4 + 3];
            
            // Compute cost
            float cost;
            if (std::isnan(theta) || theta == 0.0f)
            {
                cost = 255.0f;
            }
            else
            {
                // Compute cost: C = 102.54 * theta² (theta in radians)
                cost = cost_coefficient * theta * theta;
            }
            
            // Store combined data: [x, y, z, cost] in rover frame
            points_with_costs[i * 4 + 0] = x_rover;
            points_with_costs[i * 4 + 1] = y_rover;
            points_with_costs[i * 4 + 2] = z_rover;
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

        return points_with_costs;
    }

    std::tuple<std::vector<float>, uint32_t, uint32_t, float, float> createAveragedCostGrid(const std::vector<float>& points_with_costs, const costMapMetrics& costmap_metrics_)
    {
        double origin_x = costmap_metrics_.origin[0];
        double origin_y = costmap_metrics_.origin[1];
        double costmap_height = costmap_metrics_.size[0];
        double costmap_width = costmap_metrics_.size[1];

        // Convert to number of cells (round up)
        uint32_t width_cells = static_cast<uint32_t>(std::ceil(costmap_width / resolution_));
        uint32_t height_cells = static_cast<uint32_t>(std::ceil(costmap_height / resolution_));

        size_t num_cells = static_cast<size_t>(width_cells) * static_cast<size_t>(height_cells);

        // Accumulators for sum and count per cell
        std::vector<double> sum(num_cells, 0.0);
        std::vector<uint32_t> count(num_cells, 0);

        // Bin points into cells
        int points_binned = 0;
        int points_out_of_bounds = 0;
        int points_invalid = 0;
        
        for (size_t i = 0; i + 3 < points_with_costs.size(); i += 4)
        {
            float x = points_with_costs[i + 0];
            float y = points_with_costs[i + 1];
            float cost = points_with_costs[i + 3];

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(cost))
            {
                points_invalid++;
                continue; // skip invalid
            }

            int ix = static_cast<int>(std::floor((x - origin_x) / resolution_));
            int iy = static_cast<int>(std::floor((y - origin_y) / resolution_));

            if (ix < 0 || iy < 0) {
                points_out_of_bounds++;
                continue;
            }
            if (static_cast<uint32_t>(ix) >= width_cells || static_cast<uint32_t>(iy) >= height_cells) {
                points_out_of_bounds++;
                continue;
            }

            size_t cell_idx = static_cast<size_t>(iy) * static_cast<size_t>(width_cells) + static_cast<size_t>(ix);
            sum[cell_idx] += static_cast<double>(cost);
            count[cell_idx] += 1;
            points_binned++;
        }
        
        RCLCPP_INFO(this->get_logger(), "Binning results: %d points binned, %d out of bounds, %d invalid (total points: %zu)", 
                    points_binned, points_out_of_bounds, points_invalid, points_with_costs.size() / 4);

        // Compute averages
        std::vector<float> avg(num_cells, 255.0f);  // Default to 255 for empty cells
        for (size_t ci = 0; ci < num_cells; ++ci)
        {
            if (count[ci] > 0)
            {
                avg[ci] = static_cast<float>(sum[ci] / static_cast<double>(count[ci]));
            }
            // otherwise leave as 255
        }

        return std::make_tuple(std::move(avg), width_cells, height_cells, origin_x, origin_y);
    }


    void publishCostmap(const std::vector<float>& averaged_grid, uint32_t width_cells, uint32_t height_cells, float origin_x, float origin_y)
    {
        // Create Costmap message
        auto costmap_msg = nav2_msgs::msg::Costmap();
        
        // Set header
        costmap_msg.header.stamp = latest_pose_timestamp_;
        costmap_msg.header.frame_id = "rover"; 
        
        // Set metadata
        costmap_msg.metadata.size_x = width_cells;
        costmap_msg.metadata.size_y = height_cells;
        costmap_msg.metadata.resolution = resolution_;
        
        // Set origin (position of cell (0,0) in the map frame)
        costmap_msg.metadata.origin.position.x = origin_x;
        costmap_msg.metadata.origin.position.y = origin_y;
        costmap_msg.metadata.origin.position.z = 0;
        
        // Keep initial orientation X foward, Y left, Z up
        costmap_msg.metadata.origin.orientation.x = 0;
        costmap_msg.metadata.origin.orientation.y = 0;
        costmap_msg.metadata.origin.orientation.z = 0;
        costmap_msg.metadata.origin.orientation.w = 1;
        
        // Allocate data array
        costmap_msg.data.resize(width_cells * height_cells);
        
        // Convert averaged costs directly to uint8_t (already in 0-255 range)
        for (size_t i = 0; i < width_cells * height_cells; ++i)
        {
            float cost = averaged_grid[i];
            
            // Costs are already in 0-255 range from createAveragedCostGrid
            // Just clamp and convert to uint8_t
            if (cost >= 255.0f)
            {
                costmap_msg.data[i] = 255;
            }
            else if (cost <= 0.0f)
            {
                costmap_msg.data[i] = 0;
            }
            else
            {
                costmap_msg.data[i] = static_cast<uint8_t>(cost);
            }
        }
        
        // Publish the costmap
        costmap_pub_->publish(costmap_msg);
        
        // Also publish as OccupancyGrid for RViz2 visualization
        publishCostmapViz(averaged_grid, width_cells, height_cells, origin_x, origin_y);
        
        RCLCPP_DEBUG(this->get_logger(), "Published costmap with %dx%d cells", width_cells, height_cells);
    }

    void publishCostmapViz(const std::vector<float>& averaged_grid, uint32_t width_cells, uint32_t height_cells, float origin_x, float origin_y)
    {
        // Create OccupancyGrid message for RViz2 visualization
        auto viz_msg = nav_msgs::msg::OccupancyGrid();
        
        // Set header
        viz_msg.header.stamp = latest_pose_timestamp_;
        viz_msg.header.frame_id = "rover";
        
        // Set metadata
        viz_msg.info.width = width_cells;
        viz_msg.info.height = height_cells;
        viz_msg.info.resolution = resolution_;
        viz_msg.info.map_load_time = latest_pose_timestamp_;
        
        // Origin position in rover frame 2D
        viz_msg.info.origin.position.x = origin_x;
        viz_msg.info.origin.position.y = origin_y;
        viz_msg.info.origin.position.z = 0.0;  // 2D costmap on ground plane
        
        // Keep initial orientation X foward, Y left, Z up
        viz_msg.info.origin.orientation.x = 0;
        viz_msg.info.origin.orientation.y = 0;
        viz_msg.info.origin.orientation.z = 0;
        viz_msg.info.origin.orientation.w = 1;
        
        // Allocate data array
        viz_msg.data.resize(width_cells * height_cells);
        
        // Convert costs to OccupancyGrid format
        // OccupancyGrid uses: -1 = unknown, 0 = free, 100 = occupied
        // Scale our 0-255 costs to 0-100 range
        for (size_t i = 0; i < width_cells * height_cells; ++i)
        {
            float cost = averaged_grid[i];
            
            if (cost >= 255.0f)
            {
                // Maximum cost or unknown
                viz_msg.data[i] = 100;
            }
            else if (cost <= 0.0f)
            {
                // Free space
                viz_msg.data[i] = 0;
            }
            else
            {
                // Scale 0-255 to 0-100
                int8_t occupancy = static_cast<int8_t>((cost / 255.0f) * 100.0f);
                viz_msg.data[i] = occupancy;
            }
        }
        
        // Publish the visualization costmap
        costmap_viz_pub_->publish(viz_msg);
    }

    
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr surface_normals_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;

    
    // Publishers
    rclcpp::Publisher<nav2_msgs::msg::Costmap>::SharedPtr costmap_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_viz_pub_;
    
    // Synchronized pointcloud data
    sensor_msgs::msg::PointCloud2::SharedPtr sync_pointcloud_;

    // Camera to rover transformation (tf2::Transform)
    tf2::Transform cam_x_to_rover_transform_;

    // Costmap metrics
    costMapMetrics costmap_metrics_;
    
    // Latest pose timestamp for costmap synchronization
    rclcpp::Time latest_pose_timestamp_;
    
    // Camera parameters
    double camera_height_;
    double tilt_angle_;
    double fov_x_;
    double fov_y_;
    double max_distance_;
    
    // Rover parameters
    double rover_width_;
    double rover_length_;
    
    // Costmap parameters
    double resolution_;
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