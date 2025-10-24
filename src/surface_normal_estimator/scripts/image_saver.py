#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.saved = False
        
        # QoS settings to match the publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to surface normals visualization topic
        self.sub = self.create_subscription(
            Image, 
            '/surface_normals_viz', 
            self.callback, 
            10
        )
        
        # Set output directory
        self.output_dir = os.path.expanduser('~/TAMT/src/surface_normal_estimator/images')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.get_logger().info(f'Image saver node started, listening to /surface_normals_viz')
        self.get_logger().info(f'Output directory: {self.output_dir}')

    def callback(self, msg):
        if self.saved:
            return  # Already saved, ignore subsequent messages
        
        try:
            # Convert ROS Image message to OpenCV format
            # Since encoding is rgb8, we convert to bgr8 for OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            self.get_logger().info(f'Received image: {cv_image.shape}, dtype: {cv_image.dtype}')
            
            # Save as PNG
            png_path = os.path.join(self.output_dir, 'surface_normals_viz.png')
            cv2.imwrite(png_path, cv_image)
            self.get_logger().info(f'Saved visualization to {png_path}')
            
            self.saved = True
            self.get_logger().info('Image saved successfully!')
            
            # Shutdown after saving
            rclpy.shutdown()
            
        except Exception as e:
            self.get_logger().error(f'Error saving image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()