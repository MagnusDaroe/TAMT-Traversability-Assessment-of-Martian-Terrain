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
        rclcpp::QoS qos(10);
        qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        qos.durability(rclcpp::DurabilityPolicy::Volatile);
        
        RCLCPP_INFO(get_logger(), "SurfaceNormalEstimation node ready");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Extract depth data from the message
        int height = msg->height;
        int width = msg->width;

        // Assuming depth image is 32FC1 (float depth values)
        const float* depth_data = reinterpret_cast<const float*>(msg->data.data());

        // Convert to 2D vector
        depth.resize(height, std::vector<float>(width));
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                depth[i][j] = depth_data[i * width + j];
            }
        }

        // Set dimensions
        h = height;
        w = width;

        // Call SNE method
        auto normals = SNE(depth, height, width);

        RCLCPP_INFO(get_logger(), "Processed depth image: %dx%d", width, height);
    }
    
    std::vector<std::vector<std::vector<float>>> SNE(const std::vector<std::vector<float>>& depth_image, int height, int width)
    {
        // Input: depth_image - 2D array (height x width)
        // Input: camParam - 3x3 camera intrinsic matrix
        // Output: 3-channel surface normal map (3 x height x width)

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
                Y[i][j] = Z[i][j] * (v_map[i][j] - camParam[1][2]) / camParam[0][0];
                X[i][j] = Z[i][j] * (u_map[i][j] - camParam[0][2]) / camParam[0][0];
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
    
    
    
    
    // Member variables
    
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SurfaceNormalEstimation>());
    rclcpp::shutdown();
    return 0;
}