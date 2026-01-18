#!/usr/bin/env python3
"""
Costmap Capture Node

This node:
1. Reads frame data (RGB, depth, poses) from frame_data directory
2. Publishes synchronized data to /tamt/sync/* topics
3. Captures all costmap outputs (combined, segmentation, roughness, surface_normals)
4. Saves each costmap as a numbered image in separate folders

Output Structure:
  ~/costmap_captures/
    combined/
      frame_001.png
      frame_002.png
      ...
    segmentation/
      frame_001.png
      frame_002.png
      ...
    roughness/
      frame_001.png
      frame_002.png
      ...
    surface_normals/
      frame_001.png
      frame_002.png
      ...
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from nav2_msgs.msg import Costmap
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge
import numpy as np
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import struct


class CostmapCaptureNode(Node):
    def __init__(self):
        super().__init__('costmap_capture_node')
        
        # Declare parameters
        self.declare_parameter('camera.intrinsics.fx', 336.1)
        self.declare_parameter('camera.intrinsics.fy', 385.6)
        self.declare_parameter('camera.intrinsics.cx', 480.0)
        self.declare_parameter('camera.intrinsics.cy', 270.0)
        self.declare_parameter('camera.resolution.width', 960)
        self.declare_parameter('camera.resolution.height', 540)
        
        # Get camera parameters
        self.fx = self.get_parameter('camera.intrinsics.fx').value
        self.fy = self.get_parameter('camera.intrinsics.fy').value
        self.cx = self.get_parameter('camera.intrinsics.cx').value
        self.cy = self.get_parameter('camera.intrinsics.cy').value
        self.img_width = self.get_parameter('camera.resolution.width').value
        self.img_height = self.get_parameter('camera.resolution.height').value
        
        # Publishers
        self.rgb_pub = self.create_publisher(Image, '/tamt/sync/rgb', 10)
        self.depth_pub = self.create_publisher(Image, '/tamt/sync/depth', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/tamt/sync/pointcloud', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/tamt/sync/rover_pose', 10)
        
        # Subscribers for all costmap types
        self.combined_sub = self.create_subscription(
            Costmap,
            '/tamt/costmap/combined',
            self.combined_callback,
            10
        )
        self.segmentation_sub = self.create_subscription(
            Costmap,
            '/tamt/costmap/segmentation',
            self.segmentation_callback,
            10
        )
        self.roughness_sub = self.create_subscription(
            Costmap,
            '/tamt/costmap/roughness',
            self.roughness_callback,
            10
        )
        self.surface_normals_sub = self.create_subscription(
            Costmap,
            '/tamt/costmap/surface_normals',
            self.surface_normals_callback,
            10
        )
        
        # Bridge for image conversion
        self.bridge = CvBridge()
        
        # State tracking
        self.current_frame_idx = 0
        self.frames = []
        self.waiting_for_response = False
        
        # Response tracking for current frame
        self.received_combined = False
        self.received_segmentation = False
        self.received_roughness = False
        self.received_surface_normals = False
        
        # Setup output directories
        self.output_base = Path.home() / 'costmap_captures'
        self.output_dirs = {
            'combined': self.output_base / 'combined',
            'segmentation': self.output_base / 'segmentation',
            'roughness': self.output_base / 'roughness',
            'surface_normals': self.output_base / 'surface_normals'
        }
        
        # Create output directories
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.get_logger().info(f'Output directory: {self.output_base}')
        
        # Load frame data
        self.load_frame_data()
        
        if len(self.frames) == 0:
            self.get_logger().error('No frames loaded! Shutting down.')
            rclpy.shutdown()
            return
        
        self.get_logger().info(f'Loaded {len(self.frames)} frames. Starting capture...')
        
        # Start publishing first frame
        self.publish_next_frame()
    
    def load_frame_data(self):
        """Load frame data from the sync_pkg directory structure"""
        try:
            package_share_dir = get_package_share_directory('sync_pkg')
        except Exception as e:
            self.get_logger().fatal(f'Package sync_pkg not found: {e}')
            return
        
        data_dir = Path(package_share_dir) / 'frame_data'
        
        if not data_dir.exists():
            self.get_logger().fatal(f'Data directory not found: {data_dir}')
            return
        
        poses_file = data_dir / 'rover_poses.csv'
        
        if not poses_file.exists():
            self.get_logger().fatal(f'rover_poses.csv not found: {poses_file}')
            return
        
        # Parse CSV
        with open(poses_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or 'frame_index' in line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 9:
                    continue
                
                frame_data = {
                    'rgb_filename': parts[1],
                    'pos': [float(parts[2]), float(parts[3]), float(parts[4])],
                    'quat': [float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])]
                }
                
                # Extract frame number for depth file
                rgb_file = parts[1]
                frame_num = rgb_file.split('_')[-1].split('.')[0]
                frame_data['depth_filename'] = f'depth_{frame_num}.npy'
                frame_data['rgb_path'] = data_dir / rgb_file
                frame_data['depth_path'] = data_dir / 'depth' / frame_data['depth_filename']
                
                self.frames.append(frame_data)
    
    def load_npy_depth(self, filepath):
        """Load depth data from .npy file"""
        try:
            depth = np.load(filepath)
            return depth
        except Exception as e:
            self.get_logger().error(f'Failed to load depth file {filepath}: {e}')
            return None
    
    def create_pointcloud(self, rgb, depth):
        """Generate pointcloud from RGB and depth images"""
        h, w = depth.shape
        
        # Create coordinate grids
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)
        
        # Filter out invalid depth values
        valid = (depth > 0) & np.isfinite(depth)
        
        # Only compute for valid pixels
        z_valid = depth[valid]
        u_valid = u[valid]
        v_valid = v[valid]
        
        # Back-project to 3D
        x = (u_valid - self.cx) * z_valid / self.fx
        y = (v_valid - self.cy) * z_valid / self.fy
        z = z_valid
        
        points_3d = np.stack([x, y, z], axis=-1)
        colors = rgb[valid]
        
        # Create PointCloud2 message
        return self.create_pointcloud2_msg(points_3d, colors, (h, w), valid)
    
    def create_pointcloud2_msg(self, points_3d, colors, original_shape, valid_mask):
        """Create PointCloud2 message"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_frame'
        
        h, w = original_shape
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        cloud_data = []
        valid_idx = 0
        
        for i in range(h * w):
            if valid_mask.flat[i]:
                x, y, z = points_3d[valid_idx]
                r, g, b = colors[valid_idx]
                valid_idx += 1
            else:
                x = y = z = float('nan')
                r = g = b = 0
            
            rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
            rgb_float = struct.unpack('f', struct.pack('I', rgb_int))[0]
            
            cloud_data.append(struct.pack('ffff', x, y, z, rgb_float))
        
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = h
        cloud_msg.width = w
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = False
        cloud_msg.data = b''.join(cloud_data)
        
        return cloud_msg
    
    def publish_next_frame(self):
        """Publish next frame of synchronized data"""
        if self.current_frame_idx >= len(self.frames):
            self.get_logger().info('All frames processed. Capture complete!')
            self.get_logger().info(f'Costmaps saved to: {self.output_base}')
            rclpy.shutdown()
            return
        
        if self.waiting_for_response:
            return
        
        frame = self.frames[self.current_frame_idx]
        self.get_logger().info(f'Processing frame {self.current_frame_idx + 1}/{len(self.frames)}')
        
        # Load RGB
        rgb_img = cv2.imread(str(frame['rgb_path']), cv2.IMREAD_UNCHANGED)
        if rgb_img is None:
            self.get_logger().error(f'Failed to load RGB: {frame["rgb_path"]}')
            self.current_frame_idx += 1
            self.publish_next_frame()
            return
        
        # Convert BGR to RGB
        if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 3:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Load Depth
        depth = self.load_npy_depth(frame['depth_path'])
        if depth is None:
            self.get_logger().error(f'Failed to load depth: {frame["depth_path"]}')
            self.current_frame_idx += 1
            self.publish_next_frame()
            return
        
        # Create timestamp
        timestamp = self.get_clock().now()
        
        # Publish RGB
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding='rgb8')
        rgb_msg.header.stamp = timestamp.to_msg()
        rgb_msg.header.frame_id = 'left_camera'
        self.rgb_pub.publish(rgb_msg)
        
        # Publish Depth
        depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding='32FC1')
        depth_msg.header.stamp = timestamp.to_msg()
        depth_msg.header.frame_id = 'camera_depth_frame'
        self.depth_pub.publish(depth_msg)
        
        # Create and publish pointcloud
        pc_msg = self.create_pointcloud(rgb_img, depth)
        self.pointcloud_pub.publish(pc_msg)
        
        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp.to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = frame['pos'][0]
        pose_msg.pose.position.y = frame['pos'][1]
        pose_msg.pose.position.z = frame['pos'][2]
        pose_msg.pose.orientation.x = frame['quat'][0]
        pose_msg.pose.orientation.y = frame['quat'][1]
        pose_msg.pose.orientation.z = frame['quat'][2]
        pose_msg.pose.orientation.w = frame['quat'][3]
        self.pose_pub.publish(pose_msg)
        
        # Reset response flags
        self.waiting_for_response = True
        self.received_combined = False
        self.received_segmentation = False
        self.received_roughness = False
        self.received_surface_normals = False
    
    def save_costmap_image(self, costmap_data, costmap_type):
        """Save costmap as image"""
        # Use grayscale directly
        img = costmap_data.copy()
        
        # Rotate 90 degrees counter-clockwise (to the left)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Generate filename
        filename = f'frame_{self.current_frame_idx + 1:03d}.png'
        filepath = self.output_dirs[costmap_type] / filename
        
        # Save image as grayscale
        cv2.imwrite(str(filepath), img)
        self.get_logger().debug(f'Saved {costmap_type}: {filename}')
    
    def combined_callback(self, msg):
        """Callback for combined costmap"""
        if not self.waiting_for_response or self.received_combined:
            return
        
        try:
            width = msg.metadata.size_x
            height = msg.metadata.size_y
            # Reshape as (width, height) then transpose
            costmap_data = np.array(msg.data, dtype=np.uint8).reshape((width, height))
            
            # Transpose to match image orientation
            costmap_data = costmap_data.T
            
            self.save_costmap_image(costmap_data, 'combined')
            self.received_combined = True
            self.check_all_received()
            
        except Exception as e:
            self.get_logger().error(f'Failed to process combined costmap: {e}')
    
    def segmentation_callback(self, msg):
        """Callback for segmentation costmap"""
        if not self.waiting_for_response or self.received_segmentation:
            return
        
        try:
            width = msg.metadata.size_x
            height = msg.metadata.size_y
            # Segmentation uses (height, width) - keep as is
            costmap_data = np.array(msg.data, dtype=np.uint8).reshape((height, width))
            
            # Transpose to match image orientation
            costmap_data = costmap_data.T
            
            self.save_costmap_image(costmap_data, 'segmentation')
            self.received_segmentation = True
            self.check_all_received()
            
        except Exception as e:
            self.get_logger().error(f'Failed to process segmentation costmap: {e}')
    
    def roughness_callback(self, msg):
        """Callback for roughness costmap"""
        if not self.waiting_for_response or self.received_roughness:
            return
        
        try:
            width = msg.metadata.size_x
            height = msg.metadata.size_y
            # Reshape as (width, height) then transpose
            costmap_data = np.array(msg.data, dtype=np.uint8).reshape((width, height))
            
            # Transpose to match image orientation
            costmap_data = costmap_data.T
            
            self.save_costmap_image(costmap_data, 'roughness')
            self.received_roughness = True
            self.check_all_received()
            
        except Exception as e:
            self.get_logger().error(f'Failed to process roughness costmap: {e}')
    
    def surface_normals_callback(self, msg):
        """Callback for surface normals costmap"""
        if not self.waiting_for_response or self.received_surface_normals:
            return
        
        try:
            width = msg.metadata.size_x
            height = msg.metadata.size_y
            # Reshape as (width, height) then transpose
            costmap_data = np.array(msg.data, dtype=np.uint8).reshape((width, height))
            
            # Transpose to match image orientation
            costmap_data = costmap_data.T
            
            self.save_costmap_image(costmap_data, 'surface_normals')
            self.received_surface_normals = True
            self.check_all_received()
            
        except Exception as e:
            self.get_logger().error(f'Failed to process surface normals costmap: {e}')
    
    def check_all_received(self):
        """Check if all costmaps received, then publish next frame"""
        if (self.received_combined and self.received_segmentation and 
            self.received_roughness and self.received_surface_normals):
            
            self.get_logger().info(f'✓ Frame {self.current_frame_idx + 1} captured')
            
            self.waiting_for_response = False
            self.current_frame_idx += 1
            
            # Publish next frame immediately
            self.publish_next_frame()


def main(args=None):
    rclpy.init(args=args)
    
    node = CostmapCaptureNode()
    
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