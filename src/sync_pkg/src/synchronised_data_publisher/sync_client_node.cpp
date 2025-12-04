#include "rclcpp/rclcpp.hpp"
#include "sync_pkg/srv/trigger_sync.hpp"

#include <chrono>
#include <memory>

using namespace std::chrono_literals;

class ClientNode : public rclcpp::Node
{
public:
  ClientNode()
  : Node("sync_client_node")
  {
    client_ = this->create_client<sync_pkg::srv::TriggerSync>("trigger_sync");

    // Wait for the service to be available
    while (!client_->wait_for_service(2s)) {
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for service. Exiting.");
        return;
      }
      RCLCPP_INFO(this->get_logger(), "Waiting for the trigger_sync service to be available...");
    }

    // Set up a timer to call the service periodically
    timer_ = this->create_wall_timer(
      2s, std::bind(&ClientNode::call_trigger_service, this));
    
    RCLCPP_INFO(this->get_logger(), "Client node started. Calling service every 2 seconds...");
  }

private:
  void call_trigger_service()
  {
    auto request = std::make_shared<sync_pkg::srv::TriggerSync::Request>();

    auto future = client_->async_send_request(request,
      std::bind(&ClientNode::handle_response, this, std::placeholders::_1));
  }

  void handle_response(rclcpp::Client<sync_pkg::srv::TriggerSync>::SharedFuture future)
  {
    auto response = future.get();
    if (response->success) {
      RCLCPP_INFO(this->get_logger(), "TriggerSync succeeded: %s", response->message.c_str());
    } else {
      RCLCPP_WARN(this->get_logger(), "TriggerSync failed: %s", response->message.c_str());
    }
  }

  rclcpp::Client<sync_pkg::srv::TriggerSync>::SharedPtr client_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ClientNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}