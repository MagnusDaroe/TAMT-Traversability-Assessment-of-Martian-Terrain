#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
#include <nav2_msgs/msg/costmap.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>
#include <limits>
#include <cstddef>
#include <fstream> //<-- only used for outputting testing files

class Costmaps : public rclcpp::Node
{
public:
    Costmaps() : Node("costmaps")
    {
        // Declare parameters with default values (can be overridden by YAML file)
        this->declare_parameter("camera.height", 1.0);
        this->declare_parameter("camera.tilt_angle", 20.0);
        this->declare_parameter("camera.fov_x", 110.0);
        this->declare_parameter("camera.fov_y", 70.0);
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
        this->declare_parameter("costmap.internal_resolution", 0.01);
        this->declare_parameter("costmap.output_resolution", 0.05);
        
        //! Need params for dilation 
        this->declare_parameter("costmap.segmentation.dilation_enabled", false);
        this->declare_parameter("costmap.segmentation.dilation_kernel_size", 3);
        this->declare_parameter("costmap.segmentation.dilation_min_confidence", 0.7);
        this->declare_parameter("costmap.confidence_dampening", 1.0);

        dilation_enabled_ = this->get_parameter("costmap.segmentation.dilation_enabled").as_bool();
        dilation_kernel_size_ = this->get_parameter("costmap.segmentation.dilation_kernel_size").as_int();
        dilation_min_confidence_ = this->get_parameter("costmap.segmentation.dilation_min_confidence").as_double();
        confidence_dampening_ = this->get_parameter("costmap.confidence_dampening").as_double();

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
        internal_resolution_ = this->get_parameter("costmap.internal_resolution").as_double();
        output_resolution_ = this->get_parameter("costmap.output_resolution").as_double();

        RCLCPP_INFO(this->get_logger(), "Loaded camera parameters - FOV X: %.1f°, FOV Y: %.1f°, Max distance: %.2fm",
                    fov_x_, fov_y_, max_distance_);
        RCLCPP_INFO(this->get_logger(), "Loaded rover parameters - Width: %.2fm, Length: %.2fm",
                    rover_width_, rover_length_);
        RCLCPP_INFO(this->get_logger(), "Loaded costmap parameters - Internal resolution: %.3fm, Output resolution: %.3fm",
                internal_resolution_, output_resolution_);
                // Declare risk parameters for different segmentation classes
        this->declare_parameter("class_risks.soil", 0.2);
        this->declare_parameter("class_risks.bedrock", 0.1);
        this->declare_parameter("class_risks.sand", 0.3);
        this->declare_parameter("class_risks.rocks", 0.9);
        this->declare_parameter("class_risks.hole", 1.0);
        this->declare_parameter("class_risks.unknown", 1.0);
        
        // Load risk parameters - map class names to risk values
        class_risk_map_["soil"] = this->get_parameter("class_risks.soil").as_double();
        class_risk_map_["bedrock"] = this->get_parameter("class_risks.bedrock").as_double();
        class_risk_map_["sand"] = this->get_parameter("class_risks.sand").as_double();
        class_risk_map_["rocks"] = this->get_parameter("class_risks.rocks").as_double();
        class_risk_map_["hole"] = this->get_parameter("class_risks.hole").as_double();
        class_risk_map_["unknown"] = this->get_parameter("class_risks.unknown").as_double();
        
        // Map class IDs to risk values (assuming class IDs: 0=soil, 1=bedrock, 2=sand, 3=rocks, 4=hole)
        risk_params_[0] = class_risk_map_["soil"];
        risk_params_[1] = class_risk_map_["bedrock"];
        risk_params_[2] = class_risk_map_["sand"];
        risk_params_[3] = class_risk_map_["rocks"];
        risk_params_[4] = class_risk_map_["hole"];
        risk_params_[255] = class_risk_map_["unknown"];  // Unknown class
        hole_id_ = 4;

        // Subscribe to synchronized pointcloud topic
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/tamt/sync/pointcloud",
            10,
            std::bind(&Costmaps::pointcloudCallback, this, std::placeholders::_1)
        );

        // Subscribe to surface normals topic
        surface_normals_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/tamt/surface_normals/normals",
            10,
            std::bind(&Costmaps::surfaceNormalsCallback, this, std::placeholders::_1)
        );
 
        // Subscribe to encoded segmentation mask (16UC1 format)
        // High 8 bits: class ID, Low 8 bits: confidence (0-255)
        segmentation_mask_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/tamt/segmentation/masks_with_confidence",
            10,
            std::bind(&Costmaps::segmentationMaskCallback, this, std::placeholders::_1)
        );

        rover_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/tamt/sync/rover_pose",
            10,
            std::bind(&Costmaps::roverPoseCallback, this, std::placeholders::_1)
        );

        // - - - - - - - - - - Publishers - - - - - - - - - - 

         // Create publisher for costmap
        costmap_sne_pub_ = this->create_publisher<nav2_msgs::msg::Costmap>(
            "/tamt/costmap/surface_normals",
            10
        );

        // Create publisher for visualization in RViz2
        costmap_sne_viz_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/tamt/costmap/surface_normals_viz",
            10
        );

        // Create publisher for roughness costmap
        costmap_roughness_pub_ = this->create_publisher<nav2_msgs::msg::Costmap>(
            "/tamt/costmap/roughness",
            10
        );

        // Create publisher for roughness visualization in RViz2
        costmap_roughness_viz_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/tamt/costmap/roughness_viz",
            10
        );

        // Create publisher for per-pixel costs
        cost_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/tamt/costmap/surface_normals/cost_image",
            10
        );

        // Publisher for costmap (nav2_msgs::msg::Costmap)
        costmap_segmentation_pub_ = this->create_publisher<nav2_msgs::msg::Costmap>(
            "/tamt/costmap/segmentation",
            10
        );
        
        // Publisher for costmap visualization (nav_msgs::msg::OccupancyGrid)
        costmap_segmentation_viz_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/tamt/costmap/segmentation_viz",
            10
        );

        // Combined costmap publisher
        costmap_combined_pub_ = this->create_publisher<nav2_msgs::msg::Costmap>(
            "/tamt/costmap/combined",
            10
        );

        costmap_combined_viz_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/tamt/costmap/combined_viz",
            10
        );
        
        // Create timer that runs every 50ms
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&Costmaps::timerCallback, this)
        );

        costmap_metrics_ = getCostMapMetricsRoverFrame(getCostMapMetrics(fov_x_, fov_y_, camera_height_, tilt_angle_, max_distance_), cam_x_to_rover_transform_, camera_height_);
                
        RCLCPP_INFO(this->get_logger(), "Costmaps node initialized");


                
        // After setting up cam_x_to_rover_transform_ in constructor:
        tf2::Vector3 cam_origin = cam_x_to_rover_transform_.getOrigin();
        RCLCPP_INFO(this->get_logger(), 
                    "Camera origin in rover frame: x=%.3f, y=%.3f, z=%.3f", 
                    cam_origin.x(), cam_origin.y(), cam_origin.z());

        // Also print what the transform does to a forward ray
        tf2::Vector3 forward_ray_cam(0, 0, 1);  // Forward in camera
        tf2::Vector3 forward_ray_rover = cam_x_to_rover_transform_.getBasis() * forward_ray_cam;
        RCLCPP_INFO(this->get_logger(), 
                    "Forward camera ray (0,0,1) transforms to rover frame: x=%.3f, y=%.3f, z=%.3f", 
                    forward_ray_rover.x(), forward_ray_rover.y(), forward_ray_rover.z());

    }

private:

    struct costMapMetrics 
    {
    double origin[2] = {0.0, 0.0}; // origin[0]: x coordinate of closest left point, origin[1]: y coordinate of closest left point
    double size[2] = {0.0, 0.0}; // size[0]: Height of the map, size[1]: Width of the map
    };

    // - - - - - - - - - - - Callback Functions - - - - - - - - - - -

    void timerCallback()
    {
        if(new_sne_data_ && new_segmentation_mask_data_)
        {
            if (!sync_pointcloud_)
            {
                RCLCPP_WARN(this->get_logger(), "Pointcloud not yet received, skipping processing");
                return;
            }
            
            rclcpp::Time timestamp = this->now();
            
            std::vector<float> pointcloud = pointcloudToVector();
            std::vector<float> pointcloud_rover = transformToRoverFrame(pointcloud, pointcloud_width_, pointcloud_height_, false);
            
            // Process SNE data
            auto [sne_costmap, roughness_costmap, sne_width_cells, sne_height_cells, sne_origin_x, sne_origin_y] = 
                surfaceNormals(pointcloud_rover, normals_camera_, sne_width_, sne_height_, timestamp);
            new_sne_data_ = false;
            
            // Publish SNE costmap with CORRECT variable names
            publishSNECostmap(sne_costmap, sne_width_cells, sne_height_cells, sne_origin_x, sne_origin_y, timestamp);

            // Publish roughness costmap
            publishRoughnessCostmap(roughness_costmap, sne_width_cells, sne_height_cells, sne_origin_x, sne_origin_y, timestamp);

            // Process segmentation mask data and get costmap
            auto [seg_costmap, class_grid, confidence_grid, seg_width_cells, seg_height_cells, seg_origin_x, seg_origin_y] = 
                segmentationMask(pointcloud_rover, class_ids_, confidences_, 
                            segmentation_width_, segmentation_height_, timestamp);
            new_segmentation_mask_data_ = false;

            if (seg_costmap.empty())
            {
                RCLCPP_WARN(this->get_logger(), "Segmentation costmap is empty, skipping processing");
                return;
            }

            // Apply dilation if enabled
            std::vector<float> dilated_costmap = seg_costmap;
            if (dilation_enabled_ == true)
            {
                dilated_costmap = dilateToFillUnknown(
                    seg_costmap, class_grid, confidence_grid, 
                    seg_width_cells, seg_height_cells, 
                    dilation_kernel_size_, dilation_min_confidence_);
                
                RCLCPP_INFO(this->get_logger(), "Applied dilation to segmentation costmap");
            }

            // Fill holes with convex hull method
            dilated_costmap = fillHolesWithConvexHull(
                dilated_costmap,        // existing costmap
                pointcloud_rover,       // 3D points
                class_ids_,            // segmentation classes
                confidences_,          // segmentation confidence
                segmentation_width_,   // image width
                segmentation_height_,  // image height
                seg_width_cells,       // costmap width
                seg_height_cells,      // costmap height
                seg_origin_x,          // costmap origin x
                seg_origin_y);         // costmap origin y
        
            // Publish final segmentation costmap
            publishSegmentationCostmap(dilated_costmap, seg_width_cells, seg_height_cells, seg_origin_x, seg_origin_y, timestamp);

            // Combine Cost maps
            std::vector<float> combined_costmap = combineCostMaps(sne_costmap, dilated_costmap);
            
            // Downscale
            auto [downscaled_costmap, new_width, new_height] = downscaleCostGrid(combined_costmap, seg_width_cells, seg_height_cells, internal_resolution_, output_resolution_);
            publishCombinedCostmap(downscaled_costmap, new_width, new_height, seg_origin_x, seg_origin_y, timestamp);
        }
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Store the latest pointcloud
        sync_pointcloud_ = msg;
        pointcloud_width_ = msg->width;
        pointcloud_height_ = msg->height;
        RCLCPP_DEBUG(this->get_logger(), "Received pointcloud with %d points", msg->width * msg->height);
    }

    void segmentationMaskCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Validate image format - should be 16UC1 (16-bit unsigned, 1 channel)
        if (msg->encoding != "16UC1")
        {
            RCLCPP_ERROR(this->get_logger(), "Expected encoding 16UC1, got %s", msg->encoding.c_str());
            return;
        }
        
        segmentation_width_ = msg->width;
        segmentation_height_ = msg->height;
        
        RCLCPP_DEBUG(this->get_logger(), "Received segmentation mask: %dx%d, encoding: %s", 
                     segmentation_width_, segmentation_height_, msg->encoding.c_str());
        
        // Decode the 16UC1 mask into class IDs and confidence values
        decodeMask(msg, class_ids_, confidences_);

        new_segmentation_mask_data_ = true;
    }

    void roverPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        // Convert PoseStamped to tf2::Transform
        tf2::Quaternion q(
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z,
            msg->pose.orientation.w
        );
        tf2::Vector3 t(
            msg->pose.position.x,
            msg->pose.position.y,
            msg->pose.position.z
        );
        rover_to_global_transform_.setOrigin(t);
        rover_to_global_transform_.setRotation(q);
    }

    // - - - - - - - - - - - Segmentation Functions - - - - - - - - - - -

    std::tuple<std::vector<float>, std::vector<uint8_t>, std::vector<float>, uint32_t, uint32_t, float, float> segmentationMask(
        std::vector<float>& pointcloud_rover, 
        std::vector<uint8_t> class_ids_, 
        std::vector<float> confidences_, 
        uint32_t segmentation_width_, 
        uint32_t segmentation_height_,
        const rclcpp::Time& timestamp) 
        {
            // Combine pointcloud with segmentation data
            std::vector<float> points_with_segmentation = combinePointcloudWithSegmentation(
                pointcloud_rover, class_ids_, confidences_, segmentation_width_, segmentation_height_);
            
            if (points_with_segmentation.empty())
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to combine pointcloud with segmentation");
                return std::make_tuple(std::vector<float>(), std::vector<uint8_t>(), std::vector<float>(), 0, 0, 0.0f, 0.0f);
            }
            
            // Compute traversability cost based on segmentation
            std::vector<float> points_with_costs = computeSegmentationTraversabilityCost(
                points_with_segmentation, segmentation_width_, segmentation_height_);
            
            // Create averaged cost grid
            auto [averaged_grid, width_cells, height_cells, origin_x, origin_y] = 
                createAveragedGrid(points_with_costs, costmap_metrics_);
            
            // Create class and confidence grids
            auto [class_grid, confidence_grid] = createClassAndConfidenceGrids(
                points_with_segmentation, points_with_costs,
                width_cells, height_cells, origin_x, origin_y);
            
            // Return all grids for further processing
            return std::make_tuple(averaged_grid, class_grid, confidence_grid, 
                                width_cells, height_cells, origin_x, origin_y);
        }

    std::tuple<std::vector<uint8_t>, std::vector<float>> createClassAndConfidenceGrids(
    const std::vector<float>& points_with_segmentation,
    const std::vector<float>& points_with_costs,
    uint32_t width_cells, 
    uint32_t height_cells,
    float origin_x,
    float origin_y)
    {
        size_t num_cells = static_cast<size_t>(width_cells) * static_cast<size_t>(height_cells);
        
        // Accumulators for class and confidence
        std::vector<double> confidence_sum(num_cells, 0.0);
        std::vector<uint32_t> count(num_cells, 0);
        std::vector<std::map<uint8_t, uint32_t>> class_counts(num_cells);
        
        // Bin points into cells
        for (size_t i = 0; i + 3 < points_with_costs.size(); i += 4)
        {
            float x = points_with_costs[i + 0];
            float y = points_with_costs[i + 1];
            
            // Get class ID and confidence from points_with_segmentation
            // Format: [x, y, z, class_id, confidence]
            size_t seg_idx = i / 4 * 5;
            uint8_t class_id = static_cast<uint8_t>(points_with_segmentation[seg_idx + 3]);
            float confidence = points_with_segmentation[seg_idx + 4];

            if (!std::isfinite(x) || !std::isfinite(y))
                continue;

            int ix = static_cast<int>(std::floor((x - origin_x) / internal_resolution_));
            int iy = static_cast<int>(std::floor(-(y - origin_y) / internal_resolution_));

            if (ix < 0 || iy < 0 || 
                static_cast<uint32_t>(ix) >= height_cells || 
                static_cast<uint32_t>(iy) >= width_cells)
                continue;

            size_t cell_idx = static_cast<size_t>(ix) * static_cast<size_t>(width_cells) + static_cast<size_t>(iy);
            confidence_sum[cell_idx] += static_cast<double>(confidence);
            count[cell_idx] += 1;
            class_counts[cell_idx][class_id] += 1;
        }

        // Compute dominant class and average confidence per cell
        std::vector<uint8_t> class_map(num_cells, 255);  // Default to unknown
        std::vector<float> confidence_map(num_cells, 0.0f);
        
        for (size_t ci = 0; ci < num_cells; ++ci)
        {
            if (count[ci] > 0)
            {
                confidence_map[ci] = static_cast<float>(confidence_sum[ci] / static_cast<double>(count[ci]));
                
                // Find dominant class in this cell
                uint8_t dominant_class = 255;
                uint32_t max_count = 0;
                for (const auto& [cls, cnt] : class_counts[ci])
                {
                    if (cnt > max_count)
                    {
                        max_count = cnt;
                        dominant_class = cls;
                    }
                }
                class_map[ci] = dominant_class;
            }
        }

        return std::make_tuple(std::move(class_map), std::move(confidence_map));
    }

    std::vector<float> fillHolesWithConvexHull(
    const std::vector<float>& costmap,
    const std::vector<float>& pointcloud_rover,
    const std::vector<uint8_t>& class_ids,
    const std::vector<float>& confidences,
    uint32_t image_width, 
    uint32_t image_height,
    uint32_t costmap_width,
    uint32_t costmap_height,
    float origin_x,
    float origin_y)
    {
        // Step 1: Find hole mask in image space
        cv::Mat hole_mask_image(image_height, image_width, CV_8UC1, cv::Scalar(0));
        
        for (uint32_t v = 0; v < image_height; ++v)
        {
            for (uint32_t u = 0; u < image_width; ++u)
            {
                size_t idx = v * image_width + u;
                if (class_ids[idx] == hole_id_ && confidences[idx] > 0.3f)
                {
                    hole_mask_image.at<uint8_t>(v, u) = 255;
                }
            }
        }
        
        // Step 2: Find contours in image space
        std::vector<std::vector<cv::Point>> contours_image;
        cv::findContours(hole_mask_image, contours_image, cv::RETR_EXTERNAL, 
                        cv::CHAIN_APPROX_SIMPLE);
        
        RCLCPP_INFO(this->get_logger(), "Found %zu hole(s) in image space", 
                    contours_image.size());
        
        std::vector<float> filled_costmap = costmap;
        float hole_cost = std::min(254.0f, risk_params_[hole_id_] * 255.0f);
        
        // Step 3: Process each hole
        for (size_t hole_idx = 0; hole_idx < contours_image.size(); ++hole_idx)
        {
            const auto& contour_image = contours_image[hole_idx];
            double area = cv::contourArea(contour_image);
            
            if (area < 100.0)  // Skip small holes
            {
                continue;
            }
            
            // Step 4: Get ALL hole pixels for this contour (not just contour boundary)
            cv::Mat hole_region_mask = cv::Mat::zeros(image_height, image_width, CV_8UC1);
            std::vector<std::vector<cv::Point>> single_contour = {contour_image};
            cv::drawContours(hole_region_mask, single_contour, 0, cv::Scalar(255), cv::FILLED);
            
            // Step 5: Project ALL hole pixels to costmap coordinates
            std::vector<cv::Point> costmap_points;
            
            for (uint32_t v = 0; v < image_height; ++v)
            {
                for (uint32_t u = 0; u < image_width; ++u)
                {
                    // Only process pixels marked as this hole
                    if (hole_region_mask.at<uint8_t>(v, u) == 0)
                        continue;
                    
                    // Get 3D coordinates
                    size_t pc_idx = v * image_width + u;
                    float x = pointcloud_rover[pc_idx * 3 + 0];
                    float y = pointcloud_rover[pc_idx * 3 + 1];
                    float z = pointcloud_rover[pc_idx * 3 + 2];
                    
                    // Skip invalid points
                    if (!std::isfinite(x) || !std::isfinite(y) || z <= 0.0f)
                        continue;
                    
                    // Project to costmap coordinates
                    int ix = static_cast<int>(std::floor((x - origin_x) / internal_resolution_));
                    int iy = static_cast<int>(std::floor(-(y - origin_y) / internal_resolution_));
                    
                    // Check bounds
                    if (ix >= 0 && iy >= 0 && 
                        static_cast<uint32_t>(ix) < costmap_height && 
                        static_cast<uint32_t>(iy) < costmap_width)
                    {
                        costmap_points.push_back(cv::Point(iy, ix));  // Note: OpenCV Point(x, y) = (col, row)
                    }
                }
            }
            
            if (costmap_points.size() < 3)
            {
                RCLCPP_WARN(this->get_logger(), 
                        "Hole %zu: insufficient costmap points (%zu), skipping", 
                        hole_idx, costmap_points.size());
                continue;
            }
            
            RCLCPP_INFO(this->get_logger(), 
                    "Hole %zu: projected %zu pixels to costmap", 
                    hole_idx, costmap_points.size());
            
            // Step 6: Compute convex hull in COSTMAP space
            std::vector<cv::Point> hull_costmap;
            cv::convexHull(costmap_points, hull_costmap);
            
            RCLCPP_INFO(this->get_logger(), 
                    "Hole %zu: convex hull has %zu points in costmap space", 
                    hole_idx, hull_costmap.size());
            
            // Step 7: Create fill mask and validate
            cv::Mat fill_mask = cv::Mat::zeros(costmap_height, costmap_width, CV_8UC1);
            cv::fillConvexPoly(fill_mask, hull_costmap, cv::Scalar(255));
            
            // Step 8: Validate - count what's in the hull
            int total_cells = 0;
            int hole_cells = 0;
            int other_cells = 0;
            
            // Create a set of original projected points for fast lookup
            std::set<std::pair<int, int>> hole_point_set;
            for (const auto& pt : costmap_points)
            {
                hole_point_set.insert({pt.y, pt.x});  // Store as (row, col)
            }
            
            for (uint32_t y = 0; y < costmap_height; ++y)
            {
                for (uint32_t x = 0; x < costmap_width; ++x)
                {
                    if (fill_mask.at<uint8_t>(y, x) == 0)
                        continue;
                    
                    total_cells++;
                    
                    // Check if this cell was part of the original hole projection
                    if (hole_point_set.count({y, x}))
                    {
                        hole_cells++;
                    }
                    else if (costmap[y * costmap_width + x] < 254.0f)  // Has other classification
                    {
                        other_cells++;
                    }
                }
            }
            
            float hole_percentage = total_cells > 0 ? 
                static_cast<float>(hole_cells) / total_cells : 0.0f;
            float other_percentage = total_cells > 0 ?
                static_cast<float>(other_cells) / total_cells : 0.0f;
            
            RCLCPP_INFO(this->get_logger(), 
                    "Hull validation: %.1f%% hole, %.1f%% other (total %d cells)",
                    hole_percentage * 100.0f, other_percentage * 100.0f, total_cells);
            
            // Reject if too much non-hole terrain
            const float MAX_OTHER_PERCENTAGE = 0.30f;
            
            if (other_percentage > MAX_OTHER_PERCENTAGE)
            {
                RCLCPP_WARN(this->get_logger(), 
                        "Hole %zu: hull covers too much non-hole terrain (%.1f%%), skipping",
                        hole_idx, other_percentage * 100.0f);
                continue;
            }
            
            // Step 9: Fill the validated hull
            int filled_cells = 0;
            for (uint32_t y = 0; y < costmap_height; ++y)
            {
                for (uint32_t x = 0; x < costmap_width; ++x)
                {
                    if (fill_mask.at<uint8_t>(y, x) > 0)
                    {
                        size_t idx = y * costmap_width + x;
                        filled_costmap[idx] = hole_cost;
                        filled_cells++;
                    }
                }
            }
            
            RCLCPP_INFO(this->get_logger(), 
                    "Hole %zu: filled %d costmap cells with convex hull", 
                    hole_idx, filled_cells);
        }
        
        return filled_costmap;
    }


    std::vector<float> dilateToFillUnknown(
    const std::vector<float>& costmap,
    const std::vector<uint8_t>& class_map,
    const std::vector<float>& confidence_map,
    uint32_t width, uint32_t height,
    int kernel_size,
    float min_confidence)
    {
        if (kernel_size % 2 == 0)
        {
            kernel_size += 1;
            RCLCPP_WARN(this->get_logger(), "Kernel size must be odd, increased to %d", kernel_size);
        }
        
        int half_kernel = kernel_size / 2;
        std::vector<float> dilated_costmap = costmap;
        std::vector<uint8_t> dilated_class_map = class_map;
        
        // Track distance and confidence for each cell to determine best source
        std::vector<float> best_distance(costmap.size(), std::numeric_limits<float>::max());
        std::vector<float> best_confidence(costmap.size(), 0.0f);
        
        // Iterate through all known cells
        for (uint32_t y = 0; y < height; ++y)
        {
            for (uint32_t x = 0; x < width; ++x)
            {
                size_t center_idx = y * width + x;
                
                // Only dilate FROM cells that are known and confident
                if (class_map[center_idx] == 255 || 
                    confidence_map[center_idx] < min_confidence ||
                    costmap[center_idx] >= 255.0f)
                {
                    continue;
                }
                
                float center_cost = costmap[center_idx];
                uint8_t center_class = class_map[center_idx];
                float center_confidence = confidence_map[center_idx];
                
                // Propagate INTO neighborhood
                for (int ky = -half_kernel; ky <= half_kernel; ++ky)
                {
                    for (int kx = -half_kernel; kx <= half_kernel; ++kx)
                    {
                        int nx = static_cast<int>(x) + kx;
                        int ny = static_cast<int>(y) + ky;
                        
                        if (nx < 0 || ny < 0 || 
                            nx >= static_cast<int>(width) || 
                            ny >= static_cast<int>(height))
                        {
                            continue;
                        }
                        
                        size_t neighbor_idx = ny * width + nx;
                        
                        // Only fill unknown cells (cost = 255)
                        if (costmap[neighbor_idx] >= 255.0f)
                        {
                            // Calculate Euclidean distance from center to neighbor
                            float distance = std::sqrt(kx * kx + ky * ky);
                            
                            // Decide if this source is better than the current best
                            bool should_update = false;
                            
                            if (best_distance[neighbor_idx] == std::numeric_limits<float>::max())
                            {
                                // First time filling this cell
                                should_update = true;
                            }
                            else if (distance < best_distance[neighbor_idx] - 0.01f)  // Small epsilon for floating point
                            {
                                // Closer neighbor wins (priority 1: distance)
                                should_update = true;
                            }
                            else if (std::abs(distance - best_distance[neighbor_idx]) < 0.01f && 
                                    center_confidence > best_confidence[neighbor_idx])
                            {
                                // Same distance, but higher confidence wins (priority 2: confidence)
                                should_update = true;
                            }
                            
                            if (should_update)
                            {
                                dilated_costmap[neighbor_idx] = center_cost;
                                dilated_class_map[neighbor_idx] = center_class;
                                best_distance[neighbor_idx] = distance;
                                best_confidence[neighbor_idx] = center_confidence;
                            }
                        }
                    }
                }
            }
        }
        
        // Count how many cells were filled
        int filled_cells = 0;
        for (size_t i = 0; i < costmap.size(); ++i)
        {
            if (costmap[i] >= 255.0f && dilated_costmap[i] < 255.0f)
            {
                filled_cells++;
            }
        }
        
        RCLCPP_INFO(this->get_logger(), 
                    "Dilation filled %d unknown cells (kernel_size=%d)", 
                    filled_cells, kernel_size);
        
        return dilated_costmap;
    }

    void decodeMask(const sensor_msgs::msg::Image::SharedPtr encoded_mask,
                    std::vector<uint8_t>& class_ids,
                    std::vector<float>& confidences)
    {
        uint32_t width = encoded_mask->width;
        uint32_t height = encoded_mask->height;
        size_t num_pixels = width * height;
        
        class_ids.resize(num_pixels);
        confidences.resize(num_pixels);
        
        // Get pointer to data
        const uint16_t* encoded_ptr = reinterpret_cast<const uint16_t*>(encoded_mask->data.data());
        
        // Decode each pixel
        for (size_t i = 0; i < num_pixels; ++i)
        {
            uint16_t encoded_value = encoded_ptr[i];
            
            // Extract upper 8 bits for class_id
            class_ids[i] = static_cast<uint8_t>(encoded_value >> 8);
            
            // Extract lower 8 bits for confidence and normalize to 0.0-1.0
            uint8_t confidence_byte = static_cast<uint8_t>(encoded_value & 0xFF);
            confidences[i] = confidence_byte / 255.0f;
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Decoded segmentation mask: %zu pixels", num_pixels);
    }

    std::vector<float> computeSegmentationTraversabilityCost(std::vector<float> points_with_segmentation,
                                                  uint32_t width, uint32_t height)
    {
        size_t num_pixels = width * height;
        std::vector<float> points_with_costs(num_pixels * 4); // 4 values per point: x, y, z, cost
        
        // Compute cost for each point
        // Cost function: C = C_risk * certainty * 255, where C in range [0, 255]
        for (size_t i = 0; i < num_pixels; ++i)
        {
            // Extract point data
            float x = points_with_segmentation[i * 5 + 0];
            float y = points_with_segmentation[i * 5 + 1];
            float z = points_with_segmentation[i * 5 + 2];
            uint8_t class_id = static_cast<uint8_t>(points_with_segmentation[i * 5 + 3]);
            float confidence = points_with_segmentation[i * 5 + 4];

            float cost;
            
            // If confidence is 0 (invalid point), set cost to 255 (unknown/obstacle)
            if (confidence <= 0.0f || class_id == 255) 
            {
                cost = 255.0f;
            }
            else
            {
                // Get risk value for this class
                float risk = 1.0f;          // Default risk if class not found
                auto it = risk_params_.find(class_id);
                if (it != risk_params_.end())
                {
                    risk = it->second;
                }
                float risk_weighted = risk/(std::pow(confidence, confidence_dampening_));
                // Compute cost: C = risk * confidence * 255
                const float a = 10.0f; // steepness
                const float b = 0.5f;  // midpoint
                float sigmoid = 1.0f / (1.0f + std::exp(-a * ((risk_weighted) - b)));
                cost = sigmoid * 255.0f;
            
                // Clamp to valid range
                cost = std::clamp(cost, 0.0f, 255.0f);
            }
            
           
            // Store: [x, y, z, cost]
            points_with_costs[i * 4 + 0] = x;
            points_with_costs[i * 4 + 1] = y;
            points_with_costs[i * 4 + 2] = z;
            points_with_costs[i * 4 + 3] = cost;
        }
        return points_with_costs;
    }

    std::vector<float> combinePointcloudWithSegmentation(
    std::vector<float> pointcloud_rover,
    const std::vector<uint8_t>& class_ids,
    const std::vector<float>& confidences,
    uint32_t width, uint32_t height)
    {        
        size_t num_pixels = width * height;
        std::vector<float> points_with_segmentation(num_pixels * 5); // 5 values per point: x, y, z, class_id, confidence
        
        int points_beyond_max_distance = 0;
        
        // Iterate through each point
        for (size_t i = 0; i < num_pixels; ++i)
        {
            float x = pointcloud_rover[i * 3 + 0];
            float y = pointcloud_rover[i * 3 + 1];
            float z = pointcloud_rover[i * 3 + 2];
            
            // Get corresponding segmentation data
            float class_id = static_cast<float>(class_ids[i]);
            float confidence = confidences[i];
            
            // Set confidence to 0 if z (depth) exceeds max_distance or is invalid
            if (class_id != hole_id_ )
            {

            }
            else if (z > max_distance_ || std::isnan(z) || std::isinf(z) )
            {
                confidence = 0.0f;
                points_beyond_max_distance++;
            }
            
            // Store combined data: [x, y, z, class_id, confidence]
            points_with_segmentation[i * 5 + 0] = x;
            points_with_segmentation[i * 5 + 1] = y;
            points_with_segmentation[i * 5 + 2] = z;
            points_with_segmentation[i * 5 + 3] = class_id;
            points_with_segmentation[i * 5 + 4] = confidence;
        }
        
        return points_with_segmentation;
    }
   
    // - - - - - - - - - - - Surface Normal Functions - - - - - - - - - - -

    void surfaceNormalsCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Validate image format
        if (msg->encoding != "32FC3")
        {
            RCLCPP_ERROR(this->get_logger(), "Expected encoding 32FC3, got %s", msg->encoding.c_str());
            return;
        }

        // Normals in camera frame - convert from image data to float vector
        const float* normals_ptr = reinterpret_cast<const float*>(msg->data.data());
        normals_camera_ = std::vector<float>(normals_ptr, normals_ptr + (msg->width * msg->height * 3));

        sne_width_ = msg->width;
        sne_height_ = msg->height;

        new_sne_data_ = true;
    }

    std::tuple<std::vector<float>, std::vector<float>, uint32_t, uint32_t, float, float> surfaceNormals(const std::vector<float>& pointcloud_rover, 
        const std::vector<float>& normals_camera, uint32_t width, uint32_t height,
        const rclcpp::Time& timestamp)
    {
        std::vector<float> normals_global = transformToRoverFrame(normals_camera, width, height, true);

        // Flip normals with negative nz values
        for (size_t i = 0; i < normals_global.size(); i += 3)
        {
            if (normals_global[i + 2] < 0)  // Check if nz < 0
            {
                normals_global[i + 0] *= -1.0f;  // Flip nx
                normals_global[i + 1] *= -1.0f;  // Flip ny
                normals_global[i + 2] *= -1.0f;  // Flip nz
            }
        }

        // Combine pointcloud with normals
        std::vector<float> points_with_normals_rover = combinePointcloudWithNormals(pointcloud_rover, normals_global, width, height);
        
        // Compute polar angles from normals and combine with 3D coordinates
        // Output format: [x, y, z, theta] for each point
        std::vector<float> points_with_theta_rover = computePolarAngles(points_with_normals_rover, width, height);

        auto [averaged_theta_grid, width_cells, height_cells, origin_x, origin_y] = createAveragedGrid(points_with_theta_rover, costmap_metrics_);
        

        //!should be removed later
        // // Save averaged_theta_grid to CSV (one value per line)
        // {
        //     std::ofstream csv_file("averaged_theta_grid.csv");
        //     if (csv_file.is_open()) {
        //         for (const auto& value : averaged_theta_grid) {
        //             csv_file << value << "\n";
        //         }
        //         csv_file.close();
        //     }
        // }

        // Compute traversability cost for each point based on polar angle
        std::vector<float> slope_cost_grid = computeSNETraversabilityCost(averaged_theta_grid);

        // !this will probably not work anymore since traversability costs are now in cost grid format
        // publishCosts(traversability_costs);

        // compute the roughness cost
        std::vector<float> roughness_cost_grid = computeGradientTraversabilityCost(averaged_theta_grid, width_cells, height_cells);

        return std::make_tuple(slope_cost_grid, roughness_cost_grid, width_cells, height_cells, origin_x, origin_y);
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
        
        return points_with_theta_rover;
    }

    std::vector<float> computeSNETraversabilityCost(const std::vector<float>& theta_grid)
    {
        std::vector<float> grid_costs(theta_grid.size()); 
        
        const float quadratic_coefficient = -102.94f;
        const float linear_coefficient = 323.40f;
        
        for (size_t i = 0; i < theta_grid.size(); ++i)
        {
            float cost;
            if (std::isnan(theta_grid[i]) || theta_grid[i] == 0.0f || theta_grid[i] == 255.0f)
            {
                cost = 255.0f;
            }            
            else
            {
                cost = quadratic_coefficient * theta_grid[i] * theta_grid[i] + linear_coefficient * theta_grid[i];
            }
            grid_costs[i] = std::clamp(cost, 0.0f, 255.0f);
        }

        return grid_costs;
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

    // roughness cost function
    std::vector<float> computeGradientTraversabilityCost(const std::vector<float>& theta_grid,
                                                     int height, int width) 
    {
        // Dilated gradient map calculation
        std::vector<float> scaled_image = scaling(theta_grid, height, width);
        
        
        std::vector<float> dilated_image = dilate(scaled_image, height, width, 2, false);
        
        std::vector<float> dilated_gradient = gradient_magnitude(dilated_image, height, width);

        std::vector<float> gradient = dilate(dilated_gradient, height, width, 1, true);

        std::vector<float> scaled_gradient = sigmoid_scaling(gradient, 40, 0.06, 0, 255);

        return dilated_gradient;
    }

    // Function to dilate image (vectorized version with multiple iterations)
    // dilate_255: if true, dilates 255 values into non-255 neighbors
    //             if false, dilates non-255 values into 255 neighbors
    std::vector<float> dilate(const std::vector<float>& image, int height, int width, int iterations = 1, bool dilate_255 = false) {
        std::vector<float> dilated_image = image;  // Start with copy of original image
        
        RCLCPP_INFO(this->get_logger(), "Starting %d dilation iteration(s) on image of size %dx%d (dilate_255=%s)", 
                    iterations, width, height, dilate_255 ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "Image size: %zu", image.size());
        RCLCPP_INFO(this->get_logger(), "Width: %d, Height: %d", width, height);
        
        for (int iter = 0; iter < iterations; iter++) {
            // Create temporary output for this iteration - copy current state
            std::vector<float> temp_image = dilated_image;
            
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    int idx = x * height + y;
                    
                    if (dilate_255) {
                        // Set all edge pixels to 255 first (before any dilation logic)
                        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                            temp_image[idx] = 255.0f;
                        }
                        // DILATE 255 INTO NON-255: If any neighbor is 255, replace current pixel
                        else if (dilated_image[idx] != 255) {
                            // Check 8-connected neighbors
                            std::vector<int> neighbor_indices = {
                                (x - 1) * height + (y - 1), (x - 1) * height + y, (x - 1) * height + (y + 1),
                                x * height + (y - 1),                             x * height + (y + 1),
                                (x + 1) * height + (y - 1), (x + 1) * height + y, (x + 1) * height + (y + 1)
                            };
                            // If any neighbor is 255, set this pixel to 255
                            for (int neighbor_idx : neighbor_indices) {
                                if (dilated_image[neighbor_idx] == 255) {
                                    temp_image[idx] = 255.0f;
                                    break;
                                }
                            }
                        }
                    } else {
                        // Skip edge pixels for dilate_255 = false
                        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                            continue;
                        }
                        
                        // ORIGINAL BEHAVIOR: If current pixel is 255, compute mean of non-255 neighbors
                        if (dilated_image[idx] == 255) {
                            std::vector<float> neighbor_values = {
                                dilated_image[(x - 1) * height + (y - 1)], 
                                dilated_image[(x - 1) * height + y], 
                                dilated_image[(x - 1) * height + (y + 1)],
                                dilated_image[x * height + (y - 1)],                     
                                dilated_image[x * height + (y + 1)],
                                dilated_image[(x + 1) * height + (y - 1)], 
                                dilated_image[(x + 1) * height + y], 
                                dilated_image[(x + 1) * height + (y + 1)]
                            };
                            
                            float sum = 0.0f;
                            int count = 0;
                            for(float val : neighbor_values) {
                                if (val != 255.0f) {
                                    sum += val;
                                    count++;
                                }
                            }
                            
                            if (count > 0) {
                                temp_image[idx] = sum / count; // Assign mean of neighbors
                            } else {
                                temp_image[idx] = 255.0f; // No valid neighbors, keep as 255
                            }
                        }
                    }
                }
            }
            
            // Update dilated_image for next iteration
            dilated_image = temp_image;
            
            RCLCPP_INFO(this->get_logger(), "Completed dilation iteration %d/%d", iter + 1, iterations);
        }
        
        return dilated_image;
    }

    // Function to calculate mean within radius (vectorized version, column-major)
    std::vector<float> mean_within_radius(const std::vector<float>& image, int height, int width, int radius = 1) {
        // Pad the image
        int padded_height = height + 2 * radius;
        int padded_width = width + 2 * radius;
        std::vector<float> padded_image(padded_height * padded_width);
        
        // Edge padding - replicate border values
        for (int x = 0; x < padded_width; x++) {
            for (int y = 0; y < padded_height; y++) {
                int padded_idx = x * padded_height + y;  // Column-major indexing
                
                // Clamp coordinates to original image bounds
                int orig_x = std::min(std::max(x - radius, 0), width - 1);
                int orig_y = std::min(std::max(y - radius, 0), height - 1);
                int orig_idx = orig_x * height + orig_y;  // Column-major indexing
                
                padded_image[padded_idx] = image[orig_idx];
            }
        }
        
        std::vector<float> mean_image(padded_height * padded_width);
        
        // Calculate mean for each pixel in the padded region
        for (int x = radius; x < padded_width - radius; x++) {
            for (int y = radius; y < padded_height - radius; y++) {
                int padded_idx = x * padded_height + y;  // Column-major indexing
                
                // If current pixel is 255, keep it as 255
                if (padded_image[padded_idx] == 255.0f) {
                    mean_image[padded_idx] = 255.0f;
                    continue;
                }
                
                float sum = 0.0f;
                int count = 0;
                
                // Sum all pixels within radius, ignoring 255 values
                for (int dx = -radius; dx <= radius; dx++) {
                    for (int dy = -radius; dy <= radius; dy++) {
                        int neighbor_idx = (x + dx) * padded_height + (y + dy);
                        
                        // Only include non-255 values in the mean
                        if (padded_image[neighbor_idx] != 255.0f) {
                            sum += padded_image[neighbor_idx];
                            count++;
                        }
                    }
                }
                
                // Compute mean only if there are valid (non-255) neighbors
                if (count > 0) {
                    mean_image[padded_idx] = sum / count;
                } else {
                    mean_image[padded_idx] = 255.0f;  // If all neighbors are 255, keep as 255
                }
            }
        }
        
        // Remove padding and extract result
        std::vector<float> result(height * width);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int result_idx = x * height + y;  // Column-major indexing
                int padded_idx = (x + radius) * padded_height + (y + radius);
                result[result_idx] = mean_image[padded_idx];
            }
        }
        
        return result;
    }

    // Linear scaling function
    // TODO set old_min and old_max based on observed data range
    std::vector<float> scaling(const std::vector<float>& image, int height, int width, 
                            double old_min = 0, double old_max = 1.58, 
                            double new_min = 0, double new_max = 255) {
        
        // Create a COPY inside the function to work with
        std::vector<float> scaled_image = image;
        
        // ... all your existing code, but work on scaled_image ...
                                
        double input_min = 999999.0;
        double input_max = -999999.0;
        for (const auto& val : image) {  // Read from original
            if (val == 255.0) continue;
            if (val < input_min) input_min = val;
            if (val > input_max) input_max = val;
        }
        
        RCLCPP_INFO(this->get_logger(), "Input image min: %f, max: %f", input_min, input_max);
        
        int pixel_count = width * height;
        for (int i = 0; i < pixel_count; i++) {
            if (scaled_image[i] == 255.0f) {
                scaled_image[i] = 255.0f;
            } else {
                scaled_image[i] = (scaled_image[i] - old_min) / (old_max - old_min) * 
                                (new_max - new_min) + new_min;
            }
        }

        // log the min, max, and average values of the scaled image for debugging
        double min_val = 999999.0;
        double max_val = -999999.0;
        double sum_val = 0.0;
        int count_val = 0;
        
        for (int i = 0; i < pixel_count; i++) {
            if (image[i] != 255.0) {  // Ignore unknown/invalid values
                if (image[i] < min_val) {
                min_val = image[i];
                }
                if (image[i] > max_val) {
                max_val = image[i];
                }
                sum_val += image[i];
                count_val++;
            }
        }
        
        double avg_val = (count_val > 0) ? sum_val / count_val : 0.0;
        RCLCPP_INFO(this->get_logger(), "Scaled image average: %f", avg_val);
        RCLCPP_INFO(this->get_logger(), "Scaled image min: %f, max: %f", min_val, max_val);
        RCLCPP_INFO(this->get_logger(), "HELLO IS ANYTHING WORKING");
        
        return scaled_image;
    }

    // Sigmoid scaling function (vectorized)
    std::vector<float> sigmoid_scaling(const std::vector<float>& image, 
                                    double midpoint, double steepness,
                                    double new_min = 0, double new_max = 255) {
        std::vector<float> scaled_image(image.size());
        
        for (size_t i = 0; i < image.size(); i++) {
            if (image[i] == 255) {
                scaled_image[i] = 255;
            } else {
                scaled_image[i] = new_min + (new_max - new_min) / (1 + exp(-steepness * (image[i] - midpoint)));
            }
        }
        
        return scaled_image;
    }

    // Function to calculate gradient magnitude (vectorized, column-major)
    std::vector<float> gradient_magnitude(const std::vector<float>& image, int height, int width) {
        std::vector<float> magnitude(height * width, 255.0f);  // Initialize to 255
        
        for (int x = 1; x < width - 1; x++) {
            for (int y = 1; y < height - 1; y++) {
                int idx = x * height + y;  // Column-major indexing
                
                // If current pixel is 255, keep it as 255
                if (image[idx] >= 254.5f) {
                    magnitude[idx] = 255.0f;
                    continue;
                }
                
                // Get neighbor indices
                int idx_right = (x + 1) * height + y;
                int idx_left = (x - 1) * height + y;
                int idx_down = x * height + (y + 1);
                int idx_up = x * height + (y - 1);
                
                // Skip if any neighbor is 255
                if (image[idx_right] >= 254.5f || image[idx_left] >= 254.5f ||
                    image[idx_down] >= 254.5f || image[idx_up] >= 254.5f) {
                    magnitude[idx] = 0.0f;  // Set to 0 for borders near obstacles
                    continue;
                }
                
                // Calculate gradients using central difference
                float dx = (image[idx_right] - image[idx_left]) / 2.0f;
                float dy = (image[idx_down] - image[idx_up]) / 2.0f;
                
                magnitude[idx] = std::sqrt(dx * dx + dy * dy);
            }
        }
        
        return magnitude;
    }

    // - - - - - - - - - - - Cost Map Functions - - - - - - - - - - -

    std::vector<float> combineCostMaps(
    const std::vector<float>& sne_costmap,  const std::vector<float>& seg_costmap)
    {
        // Ensure both costmaps have the same dimensions
        if (sne_costmap.size() != seg_costmap.size())
        {
            RCLCPP_ERROR(this->get_logger(), 
                         "Costmap size mismatch! SNE size: %zu, Segmentation size: %zu",
                         sne_costmap.size(), seg_costmap.size());
            return std::vector<float>();
        }

        size_t num_cells = sne_costmap.size();
        std::vector<float> combined_costmap(num_cells);

        // Combine costmaps by taking the maximum cost at each cell
        for (size_t i = 0; i < num_cells; ++i)
        {
           combined_costmap[i] = (sne_costmap[i]+ seg_costmap[i]) / 2.0f;
        }

        return combined_costmap;
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
    
    std::tuple<std::vector<float>, uint32_t, uint32_t, float, float> createAveragedGrid(const std::vector<float>& points_with_costs, const costMapMetrics& costmap_metrics_)
    {
        double origin_x = costmap_metrics_.origin[0];
        double origin_y = costmap_metrics_.origin[1];
        double costmap_height = costmap_metrics_.size[0];
        double costmap_width = costmap_metrics_.size[1];

        // Convert to number of cells (round up)
        uint32_t width_cells = static_cast<uint32_t>(std::ceil(costmap_width / internal_resolution_));
        uint32_t height_cells = static_cast<uint32_t>(std::ceil(costmap_height / internal_resolution_));

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

            int ix = static_cast<int>(std::floor((x - origin_x) / internal_resolution_));
            int iy = static_cast<int>(std::floor(-(y - origin_y) / internal_resolution_));

            if (ix < 0 || iy < 0) {
                points_out_of_bounds++;
                continue;
            }
            if (static_cast<uint32_t>(ix) >= height_cells || static_cast<uint32_t>(iy) >= width_cells) {
                points_out_of_bounds++;
                continue;
            }

            size_t cell_idx = static_cast<size_t>(ix) * static_cast<size_t>(width_cells) + static_cast<size_t>(iy);
            sum[cell_idx] += static_cast<double>(cost);
            count[cell_idx] += 1;
            points_binned++;
        }
        
        // RCLCPP_INFO(this->get_logger(), "Binning results: %d points binned, %d out of bounds, %d invalid (total points: %zu)", 
        //             points_binned, points_out_of_bounds, points_invalid, points_with_costs.size() / 4);

        // Compute averages
        std::vector<float> avg(num_cells, 255.0f);  // Default to 255 for empty cells
        for (size_t ci = 0; ci < num_cells; ++ci)
        {
            if (count[ci] > 0)
            {
                avg[ci] = static_cast<float>(sum[ci] / static_cast<double>(count[ci]));
                //RCLCPP_INFO(this->get_logger(), "Averaged cost cell: %.2f", avg[ci]);
            }
            // otherwise leave as 255
        }
        

        return std::make_tuple(std::move(avg), width_cells, height_cells, origin_x, origin_y);
    }

    std::tuple<std::vector<float>, uint32_t, uint32_t> downscaleCostGrid(
    const std::vector<float>& cost_grid, 
    uint32_t original_width, 
    uint32_t original_height,
    float original_resolution,  // Current resolution (e.g., 0.01 m/cell)
    float target_resolution)    // Desired resolution (e.g., 0.05 m/cell)
    {
        // Calculate downscale factor
        float scale_ratio = target_resolution / original_resolution;
        
        if (scale_ratio < 1.0f)
        {
            RCLCPP_WARN(this->get_logger(), 
                        "Target resolution %.3f is smaller than original %.3f. No downscaling needed.",
                        target_resolution, original_resolution);
            return std::make_tuple(cost_grid, original_width, original_height);
        }
        
        uint32_t downscale_factor = static_cast<uint32_t>(std::round(scale_ratio));
        
        // Calculate new dimensions
        uint32_t new_width = original_width / downscale_factor;
        uint32_t new_height = original_height / downscale_factor;
        
        if (new_width == 0 || new_height == 0)
        {
            RCLCPP_ERROR(this->get_logger(), 
                        "Downscale factor %d too large for grid size %dx%d",
                        downscale_factor, original_width, original_height);
            return std::make_tuple(std::vector<float>(), 0, 0);
        }
        
        std::vector<float> downscaled_grid(new_width * new_height, 255.0f); // Initialize with 255 (unknown)
            
        // Downsample by averaging blocks
        for (uint32_t y = 0; y < new_height; ++y)
        {
            for (uint32_t x = 0; x < new_width; ++x)
            {
                float sum = 0.0f;
                uint32_t count = 0;
                float max_cost = 0.0f;  // Track maximum cost in block
                
                // Average over the downscale_factor x downscale_factor block
                for (uint32_t dy = 0; dy < downscale_factor; ++dy)
                {
                    for (uint32_t dx = 0; dx < downscale_factor; ++dx)
                    {
                        uint32_t orig_x = x * downscale_factor + dx;
                        uint32_t orig_y = y * downscale_factor + dy;
                        
                        // Bounds check
                        if (orig_x >= original_width || orig_y >= original_height)
                            continue;
                        
                        size_t orig_idx = orig_y * original_width + orig_x;
                        float cost = cost_grid[orig_idx];
                        
                        if (cost < 255.0f) // Only consider known costs
                        {
                            sum += cost;
                            count++;
                            max_cost = std::max(max_cost, cost);
                        }
                    }
                }
                
                size_t new_idx = y * new_width + x;
                
                if (count > 0)
                {
                    downscaled_grid[new_idx] = sum / static_cast<float>(count);

                }
                // else remains 255.0f (unknown)
            }
        }
        
        return std::make_tuple(std::move(downscaled_grid), new_width, new_height);
    }

    // - - - - - - - - - - - General functions - - - - - - - - - - -

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
            points[i * 3 + 2] = -z; //! Invert z because stupid isaac frame
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
                //point_rover = rover_to_global_transform_.getBasis() * cam_x_to_rover_transform_.getBasis() * point_cam; //! FIX! When rover transform is fixed apply this
                point_rover = cam_x_to_rover_transform_.getBasis() * point_cam;
            }
            else {
                // Transform point to rover frame
                point_rover = cam_x_to_rover_transform_ * point_cam;
            }

            // Store transformed data: [x, y, z] in rover frame
            points_transformed[i * 3 + 0] = point_rover.x();
            points_transformed[i * 3 + 1] = point_rover.y();
            points_transformed[i * 3 + 2] = point_rover.z();
        }

        // Print transformed pixels at row 283, columns 36-38
        if (width > 38 && height > 283)
        {
            std::string frame_type = normals ? "Normals" : "Points";
            //RCLCPP_INFO(this->get_logger(), "%s transformation for pixels at row 283, columns 36-38:", frame_type.c_str());
            for (size_t u = 36; u <= 38; ++u)
            {
                size_t v = 283;
                size_t idx = v * static_cast<size_t>(width) + u;
                
                // Before transformation (camera frame)
                float x_cam = points[idx * 3 + 0];
                float y_cam = points[idx * 3 + 1];
                float z_cam = points[idx * 3 + 2];
                
                // After transformation (rover frame)
                float x_rov = points_transformed[idx * 3 + 0];
                float y_rov = points_transformed[idx * 3 + 1];
                float z_rov = points_transformed[idx * 3 + 2];
            }
        }
        
        return points_transformed;
    }

 
    // - - - - - - - - - - - Publishers Functions - - - - - - - - - - -

    void publishCosts(const std::vector<float>& traversability_costs)
    {
        // traversability_costs format: [x, y, z, cost] for each point
        // We need to reconstruct the image with costs in their original pixel positions
        
        size_t num_points = traversability_costs.size() / 4;
        
        // Assume standard camera resolution (you may want to pass width/height as parameters)
        uint32_t width = sync_pointcloud_->width;
        uint32_t height = sync_pointcloud_->height;
        
        // Create image message
        auto cost_image_msg = sensor_msgs::msg::Image();
        cost_image_msg.header.stamp = this->now();
        cost_image_msg.header.frame_id = "camera_frame";
        cost_image_msg.height = height;
        cost_image_msg.width = width;
        cost_image_msg.encoding = "32FC1";  // Single channel 32-bit float
        cost_image_msg.is_bigendian = false;
        cost_image_msg.step = width * sizeof(float);
        
        // Allocate data
        cost_image_msg.data.resize(width * height * sizeof(float));
        float* cost_data = reinterpret_cast<float*>(cost_image_msg.data.data());
        
        // Fill the image with cost values
        for (size_t i = 0; i < num_points; ++i)
        {
            float cost = traversability_costs[i * 4 + 3];  // Extract cost from [x, y, z, cost]
            cost_data[i] = cost;
        }
        
        // Publish the cost image
        cost_image_pub_->publish(cost_image_msg);
        
        RCLCPP_DEBUG(this->get_logger(), "Published cost image with %dx%d pixels", width, height);
    }   

    void publishSegmentationCostmap(const std::vector<float>& averaged_grid, 
                        uint32_t width_cells, uint32_t height_cells, 
                        float origin_x, float origin_y,
                        const rclcpp::Time& timestamp)
    {
        // Create Costmap message (nav2_msgs::msg::Costmap)
        auto costmap_msg = nav2_msgs::msg::Costmap();

        // Set header
        costmap_msg.header.stamp = timestamp;
        costmap_msg.header.frame_id = "map";  
        
        // Set metadata
        costmap_msg.metadata.size_x = width_cells;
        costmap_msg.metadata.size_y = height_cells;
        costmap_msg.metadata.resolution = internal_resolution_;

        // Set origin (position of cell (0,0) in the map frame)
        costmap_msg.metadata.origin.position.x = origin_x;
        costmap_msg.metadata.origin.position.y = origin_y;
        costmap_msg.metadata.origin.position.z = 0.0;

        // Keep initial orientation X foward, Y left, Z up
        costmap_msg.metadata.origin.orientation.x = 0;
        costmap_msg.metadata.origin.orientation.y = 0;
        costmap_msg.metadata.origin.orientation.z = -0.7071068;
        costmap_msg.metadata.origin.orientation.w = 0.7071068;
        
        // Convert float costs to uint8
        costmap_msg.data.resize(width_cells * height_cells);
        for (size_t i = 0; i < averaged_grid.size(); ++i)
        {
            costmap_msg.data[i] = static_cast<uint8_t>(std::clamp(averaged_grid[i], 0.0f, 255.0f));
        }
        
        costmap_segmentation_pub_->publish(costmap_msg);
        
        // Also publish visualization (OccupancyGrid)
        publishSegmentationCostmapViz(averaged_grid, width_cells, height_cells, origin_x, origin_y, timestamp);
        
        RCLCPP_DEBUG(this->get_logger(), "Published costmap: %dx%d in base_link frame", width_cells, height_cells);
    }

    void publishSegmentationCostmapViz(const std::vector<float>& averaged_grid, 
                           uint32_t width_cells, uint32_t height_cells, 
                           float origin_x, float origin_y,
                           const rclcpp::Time& timestamp)
    {
        // Create OccupancyGrid message for RViz2 visualization
        auto grid_msg = nav_msgs::msg::OccupancyGrid();

        // Set header
        grid_msg.header.stamp = timestamp;
        grid_msg.header.frame_id = "map";  //  rover frame
        
        // Set metadata
        grid_msg.info.resolution = internal_resolution_;
        grid_msg.info.width = width_cells;
        grid_msg.info.height = height_cells;

        // Set origin (position of cell (0,0) in the map frame)
        grid_msg.info.origin.position.x = origin_x;
        grid_msg.info.origin.position.y = origin_y;
        grid_msg.info.origin.position.z = 0.0;

        // Keep initial orientation X foward, Y left, Z up
        grid_msg.info.origin.orientation.x = 0;
        grid_msg.info.origin.orientation.y = 0;
        grid_msg.info.origin.orientation.z = -0.7071068;
        grid_msg.info.origin.orientation.w = 0.7071068;
        
        // Convert to OccupancyGrid format (0-100, -1 for unknown)
        grid_msg.data.resize(width_cells * height_cells);
        for (size_t i = 0; i < averaged_grid.size(); ++i)
        {
            if (averaged_grid[i] >= 255.0f)
            {
                grid_msg.data[i] = -1; // Unknown
            }
            else
            {
                // Scale from 0-255 to 0-100
                grid_msg.data[i] = static_cast<int8_t>(averaged_grid[i] * 100.0f / 254.0f);
            }
        }
        
        costmap_segmentation_viz_pub_->publish(grid_msg);
    }

    void publishSNECostmap(const std::vector<float>& averaged_grid, uint32_t width_cells, uint32_t height_cells, float origin_x, float origin_y,
                           const rclcpp::Time& timestamp)
    {
        // Create Costmap message
        auto costmap_msg = nav2_msgs::msg::Costmap();
        
        // Set header
        costmap_msg.header.stamp = timestamp;
        costmap_msg.header.frame_id = "map"; 
        
        // Set metadata
        costmap_msg.metadata.size_x = height_cells;
        costmap_msg.metadata.size_y = width_cells;
        costmap_msg.metadata.resolution = internal_resolution_;
        
        // Set origin (position of cell (0,0) in the map frame)
        costmap_msg.metadata.origin.position.x = origin_x;
        costmap_msg.metadata.origin.position.y = origin_y;
        costmap_msg.metadata.origin.position.z = 0;
        
        // Keep initial orientation X foward, Y left, Z up
        costmap_msg.metadata.origin.orientation.x = 0;
        costmap_msg.metadata.origin.orientation.y = 0;
        costmap_msg.metadata.origin.orientation.z = -0.7071068;
        costmap_msg.metadata.origin.orientation.w = 0.7071068;
        
        // Allocate data array
        costmap_msg.data.resize(width_cells * height_cells);
        
        // Convert averaged costs directly to uint8_t (already in 0-255 range)
        for (size_t i = 0; i < width_cells * height_cells; ++i)
        {
            float cost = averaged_grid[i];
            
            // Costs are already in 0-255 range from createAveragedGrid
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
        costmap_sne_pub_->publish(costmap_msg);
        
        // Also publish as OccupancyGrid for RViz2 visualization
        publishSNECostmapViz(averaged_grid, width_cells, height_cells, origin_x, origin_y, timestamp);
        
        RCLCPP_DEBUG(this->get_logger(), "Published costmap with %dx%d cells", width_cells, height_cells);
    }

    void publishSNECostmapViz(const std::vector<float>& averaged_grid, uint32_t width_cells, uint32_t height_cells, float origin_x, float origin_y,
                              const rclcpp::Time& timestamp)
    {
        // Create OccupancyGrid message for RViz2 visualization
        auto viz_msg = nav_msgs::msg::OccupancyGrid();
        
        // Set header
        viz_msg.header.stamp = timestamp;
        viz_msg.header.frame_id = "map";
        
        // Set metadata
        viz_msg.info.width = width_cells;
        viz_msg.info.height = height_cells;
        viz_msg.info.resolution = internal_resolution_;
        
        // Origin position in rover frame 2D
        viz_msg.info.origin.position.x = origin_x;
        viz_msg.info.origin.position.y = origin_y;
        viz_msg.info.origin.position.z = 0.0;  // 2D costmap on ground plane
        
        // -90 degrees around Z axis
        viz_msg.info.origin.orientation.x = 0;
        viz_msg.info.origin.orientation.y = 0;
        viz_msg.info.origin.orientation.z = -0.7071068;
        viz_msg.info.origin.orientation.w = 0.7071068; 
        
        // Allocate data array
        viz_msg.data.resize(width_cells * height_cells);
        
        // Convert costs to OccupancyGrid format
        // OccupancyGrid uses: -1 = unknown, 0 = free, 100 = occupied
        // Scale our 0-255 costs to 0-100 range
        for (size_t i = 0; i < width_cells * height_cells; ++i)
        {
            float cost = averaged_grid[i];
            
            if (averaged_grid[i] >= 255.0f)
            {
                viz_msg.data[i] = -1; // Unknown
            }
            else
            {
                // Scale from 0-255 to 0-100
                viz_msg.data[i] = static_cast<int8_t>(averaged_grid[i] * 100.0f / 254.0f);
            }
        }
        
        // Publish the visualization costmap
        costmap_sne_viz_pub_->publish(viz_msg);
    }

    void publishRoughnessCostmap(const std::vector<float>& averaged_grid, uint32_t width_cells, uint32_t height_cells, float origin_x, float origin_y,
                           const rclcpp::Time& timestamp)
    {
        // Create Costmap message
        auto costmap_msg = nav2_msgs::msg::Costmap();
        
        // Set header
        costmap_msg.header.stamp = timestamp;
        costmap_msg.header.frame_id = "map"; 
        
        // Set metadata
        costmap_msg.metadata.size_x = height_cells;
        costmap_msg.metadata.size_y = width_cells;
        costmap_msg.metadata.resolution = internal_resolution_;
        
        // Set origin (position of cell (0,0) in the map frame)
        costmap_msg.metadata.origin.position.x = origin_x;
        costmap_msg.metadata.origin.position.y = origin_y;
        costmap_msg.metadata.origin.position.z = 0;
        
        // Keep initial orientation X foward, Y left, Z up
        costmap_msg.metadata.origin.orientation.x = 0;
        costmap_msg.metadata.origin.orientation.y = 0;
        costmap_msg.metadata.origin.orientation.z = -0.7071068;
        costmap_msg.metadata.origin.orientation.w = 0.7071068;
        
        // Allocate data array
        costmap_msg.data.resize(width_cells * height_cells);
        
        // Convert averaged costs directly to uint8_t (already in 0-255 range)
        for (size_t i = 0; i < width_cells * height_cells; ++i)
        {
            float cost = averaged_grid[i];
            
            // Costs are already in 0-255 range from createAveragedGrid
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
        costmap_roughness_pub_->publish(costmap_msg);
        
        // Also publish as OccupancyGrid for RViz2 visualization
        publishRoughnessCostmapViz(averaged_grid, width_cells, height_cells, origin_x, origin_y, timestamp);
        
        RCLCPP_DEBUG(this->get_logger(), "Published costmap with %dx%d cells", width_cells, height_cells);
    }

    void publishRoughnessCostmapViz(const std::vector<float>& averaged_grid, uint32_t width_cells, uint32_t height_cells, float origin_x, float origin_y,
                              const rclcpp::Time& timestamp)
    {
        // Create OccupancyGrid message for RViz2 visualization
        auto viz_msg = nav_msgs::msg::OccupancyGrid();
        
        // Set header
        viz_msg.header.stamp = timestamp;
        viz_msg.header.frame_id = "map";
        
        // Set metadata
        viz_msg.info.width = width_cells;
        viz_msg.info.height = height_cells;
        viz_msg.info.resolution = internal_resolution_;
        
        // Origin position in rover frame 2D
        viz_msg.info.origin.position.x = origin_x;
        viz_msg.info.origin.position.y = origin_y;
        viz_msg.info.origin.position.z = 0.0;  // 2D costmap on ground plane
        
        // -90 degrees around Z axis
        viz_msg.info.origin.orientation.x = 0;
        viz_msg.info.origin.orientation.y = 0;
        viz_msg.info.origin.orientation.z = -0.7071068;
        viz_msg.info.origin.orientation.w = 0.7071068; 
        
        // Allocate data array
        viz_msg.data.resize(width_cells * height_cells);
        
        // Convert costs to OccupancyGrid format
        // OccupancyGrid uses: -1 = unknown, 0 = free, 100 = occupied
        // Scale our 0-255 costs to 0-100 range
        for (size_t i = 0; i < width_cells * height_cells; ++i)
        {
            float cost = averaged_grid[i];
            
            if (averaged_grid[i] >= 255.0f)
            {
                viz_msg.data[i] = -1; // Unknown
            }
            else
            {
                // Scale from 0-255 to 0-100
                viz_msg.data[i] = static_cast<int8_t>(averaged_grid[i] * 100.0f / 254.0f);
            }
        }
        
        // Publish the visualization costmap
        costmap_roughness_viz_pub_->publish(viz_msg);
    }
    

     void publishCombinedCostmap(const std::vector<float>& averaged_grid, uint32_t width_cells, uint32_t height_cells, float origin_x, float origin_y,
                           const rclcpp::Time& timestamp)
    {
        // Create Costmap message
        auto costmap_msg = nav2_msgs::msg::Costmap();
        
        // Set header
        costmap_msg.header.stamp = timestamp;
        costmap_msg.header.frame_id = "map"; 
        
        // Set metadata
        costmap_msg.metadata.size_x = height_cells;
        costmap_msg.metadata.size_y = width_cells;
        costmap_msg.metadata.resolution = output_resolution_;
        
        // Set origin (position of cell (0,0) in the map frame)
        costmap_msg.metadata.origin.position.x = origin_x;
        costmap_msg.metadata.origin.position.y = origin_y;
        costmap_msg.metadata.origin.position.z = 0;
        
        // Keep initial orientation X foward, Y left, Z up
        costmap_msg.metadata.origin.orientation.x = 0;
        costmap_msg.metadata.origin.orientation.y = 0;
        costmap_msg.metadata.origin.orientation.z = -0.7071068;
        costmap_msg.metadata.origin.orientation.w = 0.7071068;
        
        // Allocate data array
        costmap_msg.data.resize(width_cells * height_cells);
        
        // Convert averaged costs directly to uint8_t (already in 0-255 range)
        for (size_t i = 0; i < width_cells * height_cells; ++i)
        {
            float cost = averaged_grid[i];
            
            // Costs are already in 0-255 range from createAveragedGrid
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
        costmap_combined_pub_->publish(costmap_msg);
        
        // Also publish as OccupancyGrid for RViz2 visualization
        publishCombinedCostmapViz(averaged_grid, width_cells, height_cells, origin_x, origin_y, timestamp);
        
        RCLCPP_DEBUG(this->get_logger(), "Published costmap with %dx%d cells", width_cells, height_cells);
    }

    void publishCombinedCostmapViz(const std::vector<float>& averaged_grid, uint32_t width_cells, uint32_t height_cells, float origin_x, float origin_y,
                              const rclcpp::Time& timestamp)
    {
        // Create OccupancyGrid message for RViz2 visualization
        auto viz_msg = nav_msgs::msg::OccupancyGrid();
        
        // Set header
        viz_msg.header.stamp = timestamp;
        viz_msg.header.frame_id = "map";
        
        // Set metadata
        viz_msg.info.width = width_cells;
        viz_msg.info.height = height_cells;
        viz_msg.info.resolution = output_resolution_;
        
        // Origin position in rover frame 2D
        viz_msg.info.origin.position.x = origin_x;
        viz_msg.info.origin.position.y = origin_y;
        viz_msg.info.origin.position.z = 0.0;  // 2D costmap on ground plane
        
        // -90 degrees around Z axis
        viz_msg.info.origin.orientation.x = 0;
        viz_msg.info.origin.orientation.y = 0;
        viz_msg.info.origin.orientation.z = -0.7071068;
        viz_msg.info.origin.orientation.w = 0.7071068; 
        
        // Allocate data array
        viz_msg.data.resize(width_cells * height_cells);
        
        // Convert costs to OccupancyGrid format
        // OccupancyGrid uses: -1 = unknown, 0 = free, 100 = occupied
        // Scale our 0-255 costs to 0-100 range
        for (size_t i = 0; i < width_cells * height_cells; ++i)
        {
            float cost = averaged_grid[i];
            
            if (averaged_grid[i] >= 255.0f)
            {
                viz_msg.data[i] = -1; // Unknown
            }
            else
            {
                // Scale from 0-255 to 0-100
                viz_msg.data[i] = static_cast<int8_t>(averaged_grid[i] * 100.0f / 254.0f);
            }
        }
        
        // Publish the visualization costmap
        costmap_combined_viz_pub_->publish(viz_msg);
    }
    
    

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr surface_normals_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr segmentation_mask_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr rover_pose_sub_;
    
    // Publishers
    rclcpp::Publisher<nav2_msgs::msg::Costmap>::SharedPtr costmap_sne_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sne_viz_pub_;
    rclcpp::Publisher<nav2_msgs::msg::Costmap>::SharedPtr costmap_roughness_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_roughness_viz_pub_;
    rclcpp::Publisher<nav2_msgs::msg::Costmap>::SharedPtr costmap_segmentation_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_segmentation_viz_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cost_image_pub_;
    rclcpp::Publisher<nav2_msgs::msg::Costmap>::SharedPtr costmap_combined_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_combined_viz_pub_;
    
    // Synchronized pointcloud data
    sensor_msgs::msg::PointCloud2::SharedPtr sync_pointcloud_;
    uint32_t pointcloud_width_;
    uint32_t pointcloud_height_;

    // SNE data
    std::vector<float> normals_camera_;
    uint32_t sne_width_;
    uint32_t sne_height_;

    // Segmentation mask data
    uint32_t segmentation_width_;
    uint32_t segmentation_height_;
    std::vector<uint8_t> class_ids_;
    std::vector<float> confidences_;
    float confidence_dampening_; // exponent for confidence adjustment

    int dilation_kernel_size_;
    float dilation_min_confidence_;
    bool dilation_enabled_;


    // Camera to rover transformation
    tf2::Transform cam_x_to_rover_transform_;
    
    // Rover to global transformation
    tf2::Transform rover_to_global_transform_;

    // Costmap metrics
    costMapMetrics costmap_metrics_;
    
    // Camera parameters
    double camera_height_;
    double tilt_angle_;
    double fov_x_;
    double fov_y_;
    double max_distance_;
    
    // Rover parameters
    double rover_width_;
    double rover_length_;

    int hole_id_;
    
    // Costmap parameters
    double internal_resolution_;
    double output_resolution_;

    // Risk parameters for different segmentation classes
    std::map<std::string, double> class_risk_map_;  // Map class names to risks
    std::map<uint8_t, float> risk_params_;          // Map class IDs to risks

    // Flags to indicate new data availability
    bool new_sne_data_ = false;
    bool new_segmentation_mask_data_ = false;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Costmaps>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


//TODO Fix the cost of sne costmap to be in global frame