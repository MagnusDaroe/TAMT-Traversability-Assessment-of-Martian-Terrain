#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <memory>
#include <string>

using namespace std::chrono_literals;

class SurfaceNormalEstimation : public rclcpp::Node
{
public:
    SurfaceNormalEstimation()
    : Node("surface_normal_estimation")
    {
        // ROS 2 QoS settings
        rclcpp::QoS qos(rclcpp::KeepLast(10));
        qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
        qos.durability(rclcpp::DurabilityPolicy::TransientLocal);

        // Subscribe to sync_depth topic
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/sync_depth",
            qos,
            std::bind(&SurfaceNormalEstimation::imageCallback, this, std::placeholders::_1)
        );

        // Publisher for raw surface normals (for computation)
        normal_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/surface_normals", qos);
        
        // Publisher for visualization normals (for RViz)
        normal_viz_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/surface_normals_viz", qos);

        RCLCPP_INFO(get_logger(), "SurfaceNormalEstimation node ready, listening to /sync_depth");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr normal_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr normal_viz_pub_;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        RCLCPP_INFO(get_logger(), "Received depth image: %dx%d", msg->width, msg->height);
        
        // Extract depth data from the message
        int height = msg->height;
        int width = msg->width;

        // Assuming depth image is 32FC1 (float depth values)
        const float* depth_data = reinterpret_cast<const float*>(msg->data.data());

        // Declare and convert to 2D vector locally
        std::vector<std::vector<float>> depth(height, std::vector<float>(width));
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                depth[i][j] = depth_data[i * width + j];
            }
        }

        // Call SNE method
        auto normals = SNE(depth, height, width);

        // ===== PUBLISH RAW NORMALS (32FC3) for computation =====
        auto normal_msg = sensor_msgs::msg::Image();
        normal_msg.header = msg->header;
        normal_msg.header.stamp = this->now();
        normal_msg.height = height;
        normal_msg.width = width;
        normal_msg.encoding = "32FC3";  // 3-channel float (nx, ny, nz)
        normal_msg.is_bigendian = false;
        normal_msg.step = width * 3 * sizeof(float);
        
        // Flatten normals into 1D array
        normal_msg.data.resize(height * width * 3 * sizeof(float));
        float* data_ptr = reinterpret_cast<float*>(normal_msg.data.data());
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int idx = (i * width + j) * 3;
                data_ptr[idx + 0] = normals[0][i][j];  // nx
                data_ptr[idx + 1] = normals[1][i][j];  // ny
                data_ptr[idx + 2] = normals[2][i][j];  // nz
            }
        }
        
        normal_pub_->publish(normal_msg);

        // ===== PUBLISH VISUALIZATION NORMALS (RGB8) for RViz =====
        auto viz_msg = sensor_msgs::msg::Image();
        viz_msg.header = msg->header;
        viz_msg.header.stamp = this->now();
        viz_msg.height = height;
        viz_msg.width = width;
        viz_msg.encoding = "rgb8";
        viz_msg.is_bigendian = false;
        viz_msg.step = width * 3;
        
        // Convert normals from [-1, 1] to [0, 255]
        viz_msg.data.resize(height * width * 3);
        uint8_t* viz_ptr = viz_msg.data.data();
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int idx = (i * width + j) * 3;
                viz_ptr[idx + 0] = static_cast<uint8_t>(255 * (normals[0][i][j] + 1.0f) / 2.0f);  // R = nx
                viz_ptr[idx + 1] = static_cast<uint8_t>(255 * (normals[1][i][j] + 1.0f) / 2.0f);  // G = ny
                viz_ptr[idx + 2] = static_cast<uint8_t>(255 * (normals[2][i][j] + 1.0f) / 2.0f);  // B = nz
            }
        }
        
        normal_viz_pub_->publish(viz_msg);

        RCLCPP_INFO(get_logger(), "Published raw and visualization normals: %dx%d", width, height);
    }
    
    std::vector<std::vector<std::vector<float>>> SNE(const std::vector<std::vector<float>>& depth_image, int height, int width)
    {
        // Input: depth_image - 2D array (height x width)
        // Input: camParam - 3x3 camera intrinsic matrix
        // Output: 3-channel surface normal map (3 x height x width)

        RCLCPP_DEBUG(get_logger(), "Surface normal estimation started");

        // Declare camera parameters locally
        std::array<std::array<float, 3>, 3> camParam = {{
            {7.215377e+02f, 0.000000e+00f, 6.095593e+02f},
            {0.000000e+00f, 7.215377e+02f, 1.728540e+02f},
            {0.000000e+00f, 0.000000e+00f, 1.000000e+00f}
        }};

        int h = height;
        int w = width;

        // Create coordinate meshgrids
        std::vector<std::vector<float>> v_map(h, std::vector<float>(w));
        std::vector<std::vector<float>> u_map(h, std::vector<float>(w));
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                v_map[i][j] = static_cast<float>(i);
                u_map[i][j] = static_cast<float>(j);
            }
        }

        // Compute X, Y, Z coordinates
        std::vector<std::vector<float>> Z = depth_image; // h x w
        std::vector<std::vector<float>> Y(h, std::vector<float>(w));
        std::vector<std::vector<float>> X(h, std::vector<float>(w));
        std::vector<std::vector<float>> D(h, std::vector<float>(w));

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                Y[i][j] = Z[i][j] * (v_map[i][j] - camParam[1][2]) / camParam[1][1];  // Use fy, not fx
                X[i][j] = Z[i][j] * (u_map[i][j] - camParam[0][2]) / camParam[0][0];  // This one is correct
                if (std::isnan(Z[i][j])) Z[i][j] = 0;
                D[i][j] = (Z[i][j] != 0) ? 1.0f / Z[i][j] : 0.0f;
            }
        }

        // Define gradient kernels
        float Gx[3][3] = {{0,0,0},{-1,0,1},{0,0,0}};
        float Gy[3][3] = {{0,-1,0},{0,0,0},{0,1,0}};

        // Apply convolution for Gu and Gv
        auto conv2d = [](const std::vector<std::vector<float>>& input, float kernel[3][3], int h, int w) {
            std::vector<std::vector<float>> output(h, std::vector<float>(w, 0));
            for (int i = 1; i < h - 1; ++i) {
                for (int j = 1; j < w - 1; ++j) {
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            output[i][j] += input[i + ki - 1][j + kj - 1] * kernel[ki][kj];
                        }
                    }
                }
            }
            return output;
        };

        auto Gu = conv2d(D, Gx, h, w);
        auto Gv = conv2d(D, Gy, h, w);

        std::vector<std::vector<float>> nx_t(h, std::vector<float>(w));
        std::vector<std::vector<float>> ny_t(h, std::vector<float>(w));
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                nx_t[i][j] = Gu[i][j] * camParam[0][0];
                ny_t[i][j] = Gv[i][j] * camParam[1][1];
            }
        }

        // Compute phi, a, b
        std::vector<std::vector<float>> phi(h, std::vector<float>(w));
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                phi[i][j] = std::atan2(ny_t[i][j], nx_t[i][j]) + M_PI;
            }
        }

        // Define difference kernels
        float diffKernelArray[8][9] = {
            {-1, 0, 0, 0, 1, 0, 0, 0, 0},
            { 0,-1, 0, 0, 1, 0, 0, 0, 0},
            { 0, 0,-1, 0, 1, 0, 0, 0, 0},
            { 0, 0, 0,-1, 1, 0, 0, 0, 0},
            { 0, 0, 0, 0, 1,-1, 0, 0, 0},
            { 0, 0, 0, 0, 1, 0,-1, 0, 0},
            { 0, 0, 0, 0, 1, 0, 0,-1, 0},
            { 0, 0, 0, 0, 1, 0, 0, 0,-1}
        };

        std::vector<std::vector<float>> sum_nx(h, std::vector<float>(w, 0));
        std::vector<std::vector<float>> sum_ny(h, std::vector<float>(w, 0));
        std::vector<std::vector<float>> sum_nz(h, std::vector<float>(w, 0));

        for (int k = 0; k < 8; ++k) {
            float diffKernel[3][3];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    diffKernel[i][j] = diffKernelArray[k][i * 3 + j];
                }
            }
            
            auto X_d = conv2d(X, diffKernel, h, w);
            auto Y_d = conv2d(Y, diffKernel, h, w);
            auto Z_d = conv2d(Z, diffKernel, h, w);
            
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j) {
                    float nz_i = (Z_d[i][j] != 0) ? (nx_t[i][j] * X_d[i][j] + ny_t[i][j] * Y_d[i][j]) / Z_d[i][j] : 0;
                    float norm = std::sqrt(nx_t[i][j] * nx_t[i][j] + ny_t[i][j] * ny_t[i][j] + nz_i * nz_i);
                    
                    float nx_t_i = (norm != 0) ? nx_t[i][j] / norm : 0;
                    float ny_t_i = (norm != 0) ? ny_t[i][j] / norm : 0;
                    float nz_t_i = (norm != 0) ? nz_i / norm : 0;
                    
                    if (std::isnan(nx_t_i)) nx_t_i = 0;
                    if (std::isnan(ny_t_i)) ny_t_i = 0;
                    if (std::isnan(nz_t_i)) nz_t_i = 0;
                    
                    sum_nx[i][j] += nx_t_i;
                    sum_ny[i][j] += ny_t_i;
                    sum_nz[i][j] += nz_t_i;
                }
            }
        }

        // Compute final normals
        std::vector<std::vector<float>> nx(h, std::vector<float>(w));
        std::vector<std::vector<float>> ny(h, std::vector<float>(w));
        std::vector<std::vector<float>> nz(h, std::vector<float>(w));

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                float a = std::cos(phi[i][j]);
                float b = std::sin(phi[i][j]);
                float theta = -std::atan((sum_nx[i][j] * a + sum_ny[i][j] * b) / sum_nz[i][j]);
                
                nx[i][j] = std::sin(theta) * std::cos(phi[i][j]);
                ny[i][j] = std::sin(theta) * std::sin(phi[i][j]);
                nz[i][j] = std::cos(theta);
                
                if (std::isnan(nz[i][j])) {
                    nx[i][j] = 0;
                    ny[i][j] = 0;
                    nz[i][j] = -1;
                }
                
                float sign = (ny[i][j] > 0) ? -1.0f : 1.0f;
                nx[i][j] *= sign;
                ny[i][j] *= sign;
                nz[i][j] *= sign;
            }
        }

        // Return or store nx, ny, nz as 3-channel output
        RCLCPP_DEBUG(get_logger(), "Surface normal estimation completed");
        
        // Return 3-channel normal map as [nx, ny, nz]
        return {nx, ny, nz};
    }
    
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SurfaceNormalEstimation>());
    rclcpp::shutdown();
    return 0;
}