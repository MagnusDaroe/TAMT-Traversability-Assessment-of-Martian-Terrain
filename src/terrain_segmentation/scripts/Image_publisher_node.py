#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from pathlib import Path


class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')
        
        # Declare parameters
        self.declare_parameter('image_folder', '')
        self.declare_parameter('publish_rate', 2.0)  # seconds between images
        self.declare_parameter('loop', True)  # Loop back to start when finished
        
        # Get parameters
        image_folder = self.get_parameter('image_folder').value
        publish_rate = self.get_parameter('publish_rate').value
        self.loop = self.get_parameter('loop').value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load image paths
        self.image_paths = self.load_image_paths(image_folder)
        self.current_index = 0
        
        if len(self.image_paths) == 0:
            self.get_logger().error(f"No images found in folder: {image_folder}")
            raise ValueError("No images found")
        
        # Publisher
        self.image_pub = self.create_publisher(Image, 'tamt/camera/image_raw', 10)
        
        # Timer to publish images
        self.timer = self.create_timer(publish_rate, self.timer_callback)
        
        self.get_logger().info(f'Image Publisher Node initialized')
        self.get_logger().info(f'Found {len(self.image_paths)} images in {image_folder}')
        self.get_logger().info(f'Publishing at rate: 1 image every {publish_rate} seconds')
        self.get_logger().info(f'Loop mode: {self.loop}')
    
    def load_image_paths(self, folder_path):
        """Load all image file paths from the specified folder"""
        if not folder_path:
            self.get_logger().error("No image folder specified!")
            return []
        
        folder_path = os.path.expanduser(folder_path)
        
        if not os.path.exists(folder_path):
            self.get_logger().error(f"Folder does not exist: {folder_path}")
            return []
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Get all image files
        image_paths = []
        for file_path in sorted(Path(folder_path).glob('*')):
            if file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path))
        
        return image_paths
    
    def timer_callback(self):
        """Publish the next image in the sequence"""
        if self.current_index >= len(self.image_paths):
            if self.loop:
                self.current_index = 0
                self.get_logger().info("Looping back to first image")
            else:
                self.get_logger().info("All images published. Stopping.")
                self.timer.cancel()
                return
        
        # Load image
        image_path = self.image_paths[self.current_index]
        image = cv2.imread(image_path)
        
        if image is None:
            self.get_logger().error(f"Failed to load image: {image_path}")
            self.current_index += 1
            return
        
        # Convert to ROS Image message
        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = "camera_frame"
            self.image_pub.publish(image_msg)
            
            self.get_logger().info(
                f"Published image {self.current_index + 1}/{len(self.image_paths)}: "
                f"{os.path.basename(image_path)}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {str(e)}")
        
        self.current_index += 1


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ImagePublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Image Publisher Node...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()