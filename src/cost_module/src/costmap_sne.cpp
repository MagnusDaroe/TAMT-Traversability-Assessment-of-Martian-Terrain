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
        this->declare_parameter("camera.fov_x", 90.0);
        this->declare_parameter("camera.fov_y", 60.0);
        this->declare_parameter("camera.max_distance", 5.0);
        this->declare_parameter("rover.width", 1.0);
        this->declare_parameter("rover.length", 1.5);
        this->declare_parameter("costmap.resolution", 0.05);
        
        // Get parameters from YAML file (or use defaults)
        fov_x_ = this->get_parameter("camera.fov_x").as_double();
        fov_y_ = this->get_parameter("camera.fov_y").as_double();
        max_distance_ = this->get_parameter("camera.max_distance").as_double();
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

        // Create publisher for visualization in RViz2
        costmap_viz_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/costmap_sne_viz",
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

        // Create averaged cost grid
        auto [averaged_grid, width_cells, height_cells, origin_x_cam, origin_y_cam] = createAveragedCostGrid(traversability_costs);
        RCLCPP_INFO(this->get_logger(), "Created averaged cost grid: %dx%d cells", width_cells, height_cells);

        // Publish costmap with the actual origin used for binning
        publishCostmap(averaged_grid, width_cells, height_cells, origin_x_cam, origin_y_cam);

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
        
        int points_beyond_max_distance = 0;
        
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
        
        // Cost function: C = 102.54 * theta²
        const float cost_coefficient = 102.54f;
        
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
                // Compute cost: C = 102.54 * theta² (theta in radians)
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

        return points_with_costs;
    }

    // Create an averaged cost grid from per-point costs.
    // Input: points_with_costs - vector with [x,y,z,cost] per point
    // Output: tuple(grid, width_cells, height_cells, origin_x, origin_y)
    // - grid: vector<float> of size width_cells*height_cells where each element is the
    //         average cost of points falling into that cell (NaN if no points)
    // - width_cells/height_cells: dimensions of the grid
    // - origin_x/origin_y: coordinates of the first cell (0,0) in camera frame
    std::tuple<std::vector<float>, uint32_t, uint32_t, float, float> createAveragedCostGrid(const std::vector<float>& points_with_costs)
    {
        // Compute FIXED map size in meters using camera FOV parameters (Option B)
        double fov_x_rad = fov_x_ * M_PI / 180.0;
        double fov_y_rad = fov_y_ * M_PI / 180.0;
        
        // Width: left to right extent of FOV at max_distance
        double costmap_width_m = 2.0 * max_distance_ * std::tan(fov_x_rad / 2.0);
        // Height: bottom to top extent of FOV at max_distance  
        double costmap_height_m = 2.0 * max_distance_ * std::tan(fov_y_rad / 2.0);
        
        RCLCPP_INFO(this->get_logger(), "Fixed costmap size: width=%.3fm, height=%.3fm (FOV: %.1f°x%.1f°, max_dist: %.2fm)", 
                    costmap_width_m, costmap_height_m, fov_x_, fov_y_, max_distance_);
        
        // Convert to number of cells (round up)
        uint32_t width_cells = static_cast<uint32_t>(std::ceil(costmap_width_m / resolution_));
        uint32_t height_cells = static_cast<uint32_t>(std::ceil(costmap_height_m / resolution_));

        if (width_cells == 0 || height_cells == 0) {
            RCLCPP_WARN(this->get_logger(), "Costmap has zero size: %ux%u", width_cells, height_cells);
            return std::make_tuple(std::vector<float>(), width_cells, height_cells, 0.0f, 0.0f);
        }

        size_t num_cells = static_cast<size_t>(width_cells) * static_cast<size_t>(height_cells);
        
        // Calculate origin: Find the actual min/max of points to set grid bounds
        // This ensures all valid points will fall within the grid
        float min_x = std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        
        for (size_t i = 0; i + 3 < points_with_costs.size(); i += 4)
        {
            float x = points_with_costs[i + 0];
            float y = points_with_costs[i + 1];
            if (std::isfinite(x) && std::isfinite(y))
            {
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
            }
        }
        
        if (!std::isfinite(min_x) || !std::isfinite(min_y))
        {
            RCLCPP_WARN(this->get_logger(), "No valid points found for costmap origin");
            min_x = 0.0f;
            min_y = 0.0f;
        }
        
        RCLCPP_INFO(this->get_logger(), "Grid origin (min of points) in global frame: [%.3f, %.3f]", min_x, min_y);

        // Accumulators for sum and count per cell
        std::vector<double> sum(num_cells, 0.0);
        std::vector<uint32_t> count(num_cells, 0);

        // Bin points into cells
        int points_binned = 0;
        int points_out_of_bounds = 0;
        int points_invalid = 0;
        float sample_x = 0, sample_y = 0;  // Sample a valid point for debugging
        bool found_sample = false;
        
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
            
            // Sample first valid point for debugging
            if (!found_sample) {
                sample_x = x;
                sample_y = y;
                found_sample = true;
            }

            int ix = static_cast<int>(std::floor((x - min_x) / resolution_));
            int iy = static_cast<int>(std::floor((y - min_y) / resolution_));

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
        if (found_sample) {
            RCLCPP_INFO(this->get_logger(), "Sample valid point in global frame: [%.3f, %.3f]", sample_x, sample_y);
            RCLCPP_INFO(this->get_logger(), "Grid bounds: X[%.3f to %.3f], Y[%.3f to %.3f]", 
                        min_x, min_x + costmap_width_m, min_y, min_y + costmap_height_m);
        }

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

        return std::make_tuple(std::move(avg), width_cells, height_cells, min_x, min_y);
    }



    void publishCostmap(const std::vector<float>& averaged_grid, uint32_t width_cells, uint32_t height_cells, float origin_x, float origin_y)
    {
        // Create Costmap message
        auto costmap_msg = nav2_msgs::msg::Costmap();
        
        // Set header
        costmap_msg.header.stamp = latest_pose_timestamp_;
        costmap_msg.header.frame_id = "map"; // or use your global frame
        
        // Set metadata
        costmap_msg.metadata.size_x = width_cells;
        costmap_msg.metadata.size_y = height_cells;
        costmap_msg.metadata.resolution = resolution_;
        
        // Apply 208 degree rotation around x-axis, then transform to global frame
        tf2::Quaternion rotation_x;
        rotation_x.setRPY(180.0 * M_PI / 180.0, 0, 0); // 208 degrees around x-axis
        tf2::Transform rotation_transform;
        rotation_transform.setOrigin(tf2::Vector3(0, 0, 0));
        rotation_transform.setRotation(rotation_x);
        
        // Calculate origin in camera frame based on FOV
        double fov_x_rad = fov_x_ * M_PI / 180.0;
        //double fov_y_rad = fov_y_ * M_PI / 180.0;
        float origin_x_camera = -cos((M_PI-fov_x_rad)/2)*max_distance_; 
        float origin_y_camera = 0.0f;
        float origin_z_camera = 0.0f; 
        
        // Transform origin: first apply 208° rotation, then cam_to_global_transform_
        tf2::Vector3 origin_cam(origin_x_camera, origin_y_camera, origin_z_camera);
        RCLCPP_INFO(this->get_logger(), "Origin in camera frame: [%.6f, %.6f, %.6f]", 
                    origin_cam.x(), origin_cam.y(), origin_cam.z());
        tf2::Vector3 origin_rotated = rotation_transform * origin_cam;
        RCLCPP_INFO(this->get_logger(), "Origin after 180° rotation: [%.6f, %.6f, %.6f]", 
                    origin_rotated.x(), origin_rotated.y(), origin_rotated.z());
        
        tf2::Vector3 origin_global = cam_to_global_transform_ * origin_rotated;
        RCLCPP_INFO(this->get_logger(), "Origin in global frame: [%.6f, %.6f, %.6f]", 
                    origin_global.x(), origin_global.y(), origin_global.z());
        
        // Set origin (position of cell (0,0) in the map frame)
        costmap_msg.metadata.origin.position.x = origin_global.x();
        costmap_msg.metadata.origin.position.y = origin_global.y();
        costmap_msg.metadata.origin.position.z = origin_global.z();
        
        // Set orientation: first apply 208° rotation, then cam_to_global_transform_
        tf2::Quaternion origin_orientation = cam_to_global_transform_.getRotation() * rotation_x;
        costmap_msg.metadata.origin.orientation.x = origin_orientation.x();
        costmap_msg.metadata.origin.orientation.y = origin_orientation.y();
        costmap_msg.metadata.origin.orientation.z = origin_orientation.z();
        costmap_msg.metadata.origin.orientation.w = origin_orientation.w();
        
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
        viz_msg.header.frame_id = "map";
        
        // Set metadata
        viz_msg.info.width = width_cells;
        viz_msg.info.height = height_cells;
        RCLCPP_INFO(this->get_logger(), "Costmap dimensions: width_cells=%u, height_cells=%u", width_cells, height_cells);
        
        viz_msg.info.resolution = resolution_;
        viz_msg.info.map_load_time = latest_pose_timestamp_;
        
        // Use the EXACT origin that was used for binning in createAveragedCostGrid
        // origin_x and origin_y are already in global frame
        viz_msg.info.origin.position.x = origin_x;
        viz_msg.info.origin.position.y = origin_y;
        viz_msg.info.origin.position.z = 0.0;  // 2D costmap on ground plane
        
        // Set orientation: first apply 208° rotation, then cam_to_global_transform_
        tf2::Quaternion origin_orientation = cam_to_global_transform_.getRotation();
        viz_msg.info.origin.orientation.x = origin_orientation.x();
        viz_msg.info.origin.orientation.y = origin_orientation.y();
        viz_msg.info.origin.orientation.z = origin_orientation.z();
        viz_msg.info.origin.orientation.w = origin_orientation.w();
        
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
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    
    // Publishers
    rclcpp::Publisher<nav2_msgs::msg::Costmap>::SharedPtr costmap_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_viz_pub_;
    
    // Synchronized pointcloud data
    sensor_msgs::msg::PointCloud2::SharedPtr sync_pointcloud_;

    // Camera to global transformation (tf2::Transform)
    tf2::Transform cam_to_global_transform_;
    
    // Latest pose timestamp for costmap synchronization
    rclcpp::Time latest_pose_timestamp_;
    
    // Camera parameters
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