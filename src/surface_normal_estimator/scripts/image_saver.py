#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.saved_viz = False
        self.saved_raw = False
        
        # QoS settings to match the publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to surface normals visualization topic (rgb8)
        self.sub_viz = self.create_subscription(
            Image, 
            '/surface_normals_viz', 
            self.callback_viz, 
            qos
        )
        
        # Subscribe to raw surface normals topic (32FC3)
        self.sub_raw = self.create_subscription(
            Image,
            '/surface_normals',
            self.callback_raw,
            qos
        )
        
        # Set output directory
        self.output_dir = os.path.expanduser('~/TAMT/src/surface_normal_estimator/images')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.get_logger().info(f'Image saver node started')
        self.get_logger().info(f'Listening to /surface_normals_viz and /surface_normals')
        self.get_logger().info(f'Output directory: {self.output_dir}')

    def callback_viz(self, msg):
        """Callback for visualization normals (rgb8)"""
        if self.saved_viz:
            return  # Already saved, ignore subsequent messages
        
        try:
            # Convert ROS Image message to OpenCV format
            # Since encoding is rgb8, we convert to bgr8 for OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            self.get_logger().info(f'Received viz image: {cv_image.shape}, dtype: {cv_image.dtype}')
            
            # Save as PNG
            png_path = os.path.join(self.output_dir, 'surface_normals_viz.png')
            cv2.imwrite(png_path, cv_image)
            self.get_logger().info(f'Saved visualization to {png_path}')
            
            self.saved_viz = True
            self.check_shutdown()
            
        except Exception as e:
            self.get_logger().error(f'Error saving viz image: {str(e)}')

    def callback_raw(self, msg):
        """Callback for raw normals (32FC3)"""
        if self.saved_raw:
            return  # Already saved, ignore subsequent messages
        
        try:
            # Convert to numpy array (32FC3 - 3 channel float32)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            self.get_logger().info(f'Received raw image: {cv_image.shape}, dtype: {cv_image.dtype}')
            self.get_logger().info(f'Raw normals range: [{cv_image.min():.3f}, {cv_image.max():.3f}]')
            
            # Save raw normals as .npy (for computation)
            npy_path = os.path.join(self.output_dir, 'surface_normals_raw.npy')
            np.save(npy_path, cv_image)
            self.get_logger().info(f'Saved raw normals to {npy_path}')
            
            # Also convert to RGB8 for an alternative visualization
            normals_viz = ((cv_image + 1.0) / 2.0 * 255.0).astype(np.uint8)
            normals_bgr = cv2.cvtColor(normals_viz, cv2.COLOR_RGB2BGR)
            
            png_path = os.path.join(self.output_dir, 'surface_normals_raw_viz.png')
            cv2.imwrite(png_path, normals_bgr)
            self.get_logger().info(f'Saved raw visualization to {png_path}')
            
            self.saved_raw = True
            self.check_shutdown()
            
        except Exception as e:
            self.get_logger().error(f'Error saving raw image: {str(e)}')

    def check_shutdown(self):
        """Shutdown after both images are saved"""
        if self.saved_viz and self.saved_raw:
            self.get_logger().info('Both images saved successfully!')
            rclpy.shutdown()

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