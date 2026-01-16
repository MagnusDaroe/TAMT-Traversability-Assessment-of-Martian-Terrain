#!/usr/bin/env python3
"""
Test node for cost module pipeline performance analysis

This node:
1. Reads frame data (RGB, depth, poses) from frame_data directory
2. Publishes synchronized data to /tamt/sync/* topics
3. Measures latency through the pipeline (segmentation, surface normals, costmap)
4. Calculates agreement heatmaps showing spatial consistency across rover movement
5. Displays and saves results at conclusion

Topics Published:
  /tamt/sync/pointcloud
  /tamt/sync/rgb
  /tamt/sync/rover_pose

Topics Subscribed:
  /tamt/segmentation/masks_with_confidence
  /tamt/surface_normals/normals
  /tamt/costmap/combined
  /tamt/costmap/segmentation
  /tamt/costmap/roughness
  /tamt/costmap/surface_normals
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
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import struct
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


class CostModuleTester(Node):
    def __init__(self):
        super().__init__('cost_module_tester')
        
        # Declare parameters matching the YAML structure
        self.declare_parameter('camera.intrinsics.fx', 336.1)
        self.declare_parameter('camera.intrinsics.fy', 385.6)
        self.declare_parameter('camera.intrinsics.cx', 480.0)
        self.declare_parameter('camera.intrinsics.cy', 270.0)
        self.declare_parameter('camera.resolution.width', 960)
        self.declare_parameter('camera.resolution.height', 540)
        self.declare_parameter('costmap.output_resolution', 0.02)
        
        # Get camera parameters
        self.fx = self.get_parameter('camera.intrinsics.fx').value
        self.fy = self.get_parameter('camera.intrinsics.fy').value
        self.cx = self.get_parameter('camera.intrinsics.cx').value
        self.cy = self.get_parameter('camera.intrinsics.cy').value
        self.img_width = self.get_parameter('camera.resolution.width').value
        self.img_height = self.get_parameter('camera.resolution.height').value
        self.costmap_resolution = self.get_parameter('costmap.output_resolution').value
        
        self.get_logger().info(f'Camera parameters loaded:')
        self.get_logger().info(f'  Intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')
        self.get_logger().info(f'  Resolution: {self.img_width}x{self.img_height}')
        self.get_logger().info(f'  Costmap resolution: {self.costmap_resolution}m/cell')
        
        # Publishers
        self.rgb_pub = self.create_publisher(Image, '/tamt/sync/rgb', 10)
        self.depth_pub = self.create_publisher(Image, '/tamt/sync/depth', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/tamt/sync/pointcloud', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/tamt/sync/rover_pose', 10)
        
        # Subscribers - using correct message types
        self.seg_sub = self.create_subscription(
            Image,  # 16UC1 encoded mask
            '/tamt/segmentation/masks_with_confidence',
            self.segmentation_callback,
            10
        )
        self.normals_sub = self.create_subscription(
            Image,  # 32FC3 normals
            '/tamt/surface_normals/normals',
            self.normals_callback,
            10
        )
        self.costmap_sub = self.create_subscription(
            Costmap,  # nav2_msgs::msg::Costmap
            '/tamt/costmap/combined',
            self.costmap_callback,
            10
        )
        
        # Individual costmap subscribers
        self.costmap_seg_sub = self.create_subscription(
            Costmap,
            '/tamt/costmap/segmentation',
            self.costmap_segmentation_callback,
            10
        )
        self.costmap_rough_sub = self.create_subscription(
            Costmap,
            '/tamt/costmap/roughness',
            self.costmap_roughness_callback,
            10
        )
        self.costmap_normals_sub = self.create_subscription(
            Costmap,
            '/tamt/costmap/surface_normals',
            self.costmap_surface_normals_callback,
            10
        )
        
        # Bridge for image conversion
        self.bridge = CvBridge()
        
        # State tracking
        self.current_frame_idx = 0
        self.frames = []
        self.waiting_for_response = False
        self.publish_time = None
        
        # Timing data
        self.segmentation_latencies = []
        self.normals_latencies = []
        self.costmap_latencies = []
        self.costmap_seg_latencies = []
        self.costmap_rough_latencies = []
        self.costmap_normals_latencies = []
        
        # Response tracking for current frame
        self.received_segmentation = False
        self.received_normals = False
        self.received_costmap = False
        self.received_costmap_seg = False
        self.received_costmap_rough = False
        self.received_costmap_normals = False
        
        # Agreement heatmap data - Combined costmap
        self.previous_costmap = None
        self.previous_pose = None
        
        # Track both differences AND cost value history - Combined
        self.agreement_grid = None  # Sum of absolute differences
        self.agreement_counts = None  # Count of comparisons
        
        # Track cost values for variance calculation - Combined
        self.cost_value_sum = None  # Sum of costs (for mean)
        self.cost_value_sq_sum = None  # Sum of squared costs (for variance)
        self.cost_observation_counts = None  # Count of observations
        self.cost_value_histogram = np.zeros(256, dtype=np.int64)  # Histogram for cost values 0-255
        
        # Individual costmap tracking - Segmentation
        self.previous_costmap_seg = None
        self.agreement_grid_seg = None
        self.agreement_counts_seg = None
        self.cost_value_sum_seg = None
        self.cost_value_sq_sum_seg = None
        self.cost_observation_counts_seg = None
        self.cost_value_histogram_seg = np.zeros(256, dtype=np.int64)
        
        # Individual costmap tracking - Roughness
        self.previous_costmap_rough = None
        self.agreement_grid_rough = None
        self.agreement_counts_rough = None
        self.cost_value_sum_rough = None
        self.cost_value_sq_sum_rough = None
        self.cost_observation_counts_rough = None
        self.cost_value_histogram_rough = np.zeros(256, dtype=np.int64)
        
        # Individual costmap tracking - Surface Normals
        self.previous_costmap_normals = None
        self.agreement_grid_normals = None
        self.agreement_counts_normals = None
        self.cost_value_sum_normals = None
        self.cost_value_sq_sum_normals = None
        self.cost_observation_counts_normals = None
        self.cost_value_histogram_normals = np.zeros(256, dtype=np.int64)
        
        # Load frame data
        self.load_frame_data()
        
        if len(self.frames) == 0:
            self.get_logger().error('No frames loaded! Shutting down.')
            rclpy.shutdown()
            return
        
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
        
        images_dir = data_dir / 'images'
        depth_dir = data_dir / 'depth'
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
                frame_data['depth_path'] = depth_dir / frame_data['depth_filename']
                
                self.frames.append(frame_data)
        
        self.get_logger().info(f'Loaded {len(self.frames)} frames from dataset')
    
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
        
        # Filter out invalid depth values BEFORE computation
        # Valid depth: positive, finite values
        valid = (depth > 0) & np.isfinite(depth)
        
        # Only compute for valid pixels
        z_valid = depth[valid]
        u_valid = u[valid]
        v_valid = v[valid]
        
        # Back-project to 3D (only for valid points)
        x = (u_valid - self.cx) * z_valid / self.fx
        y = (v_valid - self.cy) * z_valid / self.fy
        z = z_valid
        
        points_3d = np.stack([x, y, z], axis=-1)
        colors = rgb[valid]
        
        # Log statistics about valid points
        total_pixels = h * w
        valid_pixels = np.sum(valid)
        self.get_logger().debug(
            f'Pointcloud: {valid_pixels}/{total_pixels} valid points ({100*valid_pixels/total_pixels:.1f}%)'
        )
        
        # Create PointCloud2 message (organized)
        return self.create_pointcloud2_msg(points_3d, colors, (h, w), valid)
    
    def create_pointcloud2_msg(self, points_3d, colors, original_shape, valid_mask):
        """Create PointCloud2 message preserving image structure (organized pointcloud)"""
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
        
        # Create organized pointcloud (preserve image dimensions)
        cloud_data = []
        valid_idx = 0
        
        for i in range(h * w):
            if valid_mask.flat[i]:
                # Valid point - use actual 3D coordinates and color
                x, y, z = points_3d[valid_idx]
                r, g, b = colors[valid_idx]
                valid_idx += 1
            else:
                # Invalid point - use NaN
                x = y = z = float('nan')
                r = g = b = 0
            
            # Pack RGB as float32
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
            self.get_logger().info('All frames processed. Generating results...')
            self.generate_results()
            return
        
        if self.waiting_for_response:
            self.get_logger().warn('Still waiting for previous frame responses')
            return
        
        frame = self.frames[self.current_frame_idx]
        self.get_logger().info(f'Publishing frame {self.current_frame_idx + 1}/{len(self.frames)}')
        
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
        
        # Publish Depth as Image message
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
        
        # Start timing
        self.publish_time = self.get_clock().now()
        self.waiting_for_response = True
        self.received_segmentation = False
        self.received_normals = False
        self.received_costmap = False
        self.received_costmap_seg = False
        self.received_costmap_rough = False
        self.received_costmap_normals = False
        
        # Store current pose for agreement calculation
        self.current_pose = {
            'pos': np.array(frame['pos']),
            'quat': np.array(frame['quat'])
        }


    def segmentation_callback(self, msg):
        """Callback for segmentation results"""
        if not self.waiting_for_response or self.received_segmentation:
            return
        
        latency = (self.get_clock().now() - self.publish_time).nanoseconds / 1e9
        self.segmentation_latencies.append(latency)
        self.get_logger().info(f'Segmentation latency: {latency:.3f}s')
        self.received_segmentation = True
        self.check_all_received()
    
    def normals_callback(self, msg):
        """Callback for surface normals results"""
        if not self.waiting_for_response or self.received_normals:
            return
        
        latency = (self.get_clock().now() - self.publish_time).nanoseconds / 1e9
        self.normals_latencies.append(latency)
        self.get_logger().info(f'Surface normals latency: {latency:.3f}s')
        self.received_normals = True
        self.check_all_received()
    
    def costmap_callback(self, msg):
        """Callback for costmap results"""
        if not self.waiting_for_response or self.received_costmap:
            return
        
        latency = (self.get_clock().now() - self.publish_time).nanoseconds / 1e9
        self.costmap_latencies.append(latency)
        self.get_logger().info(f'Costmap latency: {latency:.3f}s')
        self.received_costmap = True
        
        # Convert Costmap message to numpy array
        try:
            # Extract costmap data from nav2_msgs::msg::Costmap
            width = msg.metadata.size_x
            height = msg.metadata.size_y
            resolution = msg.metadata.resolution
            origin_x = msg.metadata.origin.position.x
            origin_y = msg.metadata.origin.position.y
            
            # Convert data to numpy array
            current_costmap = np.array(msg.data, dtype=np.uint8).reshape((width, height))
            
            valid_costs = current_costmap[current_costmap < 255]
            for cost_val in range(256):
                self.cost_value_histogram[cost_val] += np.sum(valid_costs == cost_val)

            # Store metadata for agreement calculation
            costmap_info = {
                'data': current_costmap,
                'resolution': resolution,
                'origin_x': origin_x,
                'origin_y': origin_y,
                'width': width,
                'height': height
            }
            
            self.update_agreement_heatmap(costmap_info)
            
        except Exception as e:
            self.get_logger().error(f'Failed to convert costmap: {e}')
        
        self.check_all_received()
    
    def costmap_segmentation_callback(self, msg):
        """Callback for segmentation costmap"""
        if not self.waiting_for_response or self.received_costmap_seg:
            return
        
        latency = (self.get_clock().now() - self.publish_time).nanoseconds / 1e9
        self.costmap_seg_latencies.append(latency)
        self.get_logger().info(f'Segmentation costmap latency: {latency:.3f}s')
        self.received_costmap_seg = True
        
        try:
            width = msg.metadata.size_x
            height = msg.metadata.size_y
            resolution = msg.metadata.resolution
            origin_x = msg.metadata.origin.position.x
            origin_y = msg.metadata.origin.position.y
            
            current_costmap = np.array(msg.data, dtype=np.uint8).reshape((width, height))
            
            valid_costs = current_costmap[current_costmap < 255]
            for cost_val in range(256):
                self.cost_value_histogram_seg[cost_val] += np.sum(valid_costs == cost_val)
            
            costmap_info = {
                'data': current_costmap,
                'resolution': resolution,
                'origin_x': origin_x,
                'origin_y': origin_y,
                'width': width,
                'height': height
            }
            
            self.update_agreement_heatmap_individual(costmap_info, 'segmentation')
            
        except Exception as e:
            self.get_logger().error(f'Failed to convert segmentation costmap: {e}')
        
        self.check_all_received()
    
    def costmap_roughness_callback(self, msg):
        """Callback for roughness costmap"""
        if not self.waiting_for_response or self.received_costmap_rough:
            return
        
        latency = (self.get_clock().now() - self.publish_time).nanoseconds / 1e9
        self.costmap_rough_latencies.append(latency)
        self.get_logger().info(f'Roughness costmap latency: {latency:.3f}s')
        self.received_costmap_rough = True
        
        try:
            width = msg.metadata.size_x
            height = msg.metadata.size_y
            resolution = msg.metadata.resolution
            origin_x = msg.metadata.origin.position.x
            origin_y = msg.metadata.origin.position.y
            
            current_costmap = np.array(msg.data, dtype=np.uint8).reshape((width, height))
            
            valid_costs = current_costmap[current_costmap < 255]
            for cost_val in range(256):
                self.cost_value_histogram_rough[cost_val] += np.sum(valid_costs == cost_val)
            
            costmap_info = {
                'data': current_costmap,
                'resolution': resolution,
                'origin_x': origin_x,
                'origin_y': origin_y,
                'width': width,
                'height': height
            }
            
            self.update_agreement_heatmap_individual(costmap_info, 'roughness')
            
        except Exception as e:
            self.get_logger().error(f'Failed to convert roughness costmap: {e}')
        
        self.check_all_received()
    
    def costmap_surface_normals_callback(self, msg):
        """Callback for surface normals costmap"""
        if not self.waiting_for_response or self.received_costmap_normals:
            return
        
        latency = (self.get_clock().now() - self.publish_time).nanoseconds / 1e9
        self.costmap_normals_latencies.append(latency)
        self.get_logger().info(f'Surface normals costmap latency: {latency:.3f}s')
        self.received_costmap_normals = True
        
        try:
            width = msg.metadata.size_x
            height = msg.metadata.size_y
            resolution = msg.metadata.resolution
            origin_x = msg.metadata.origin.position.x
            origin_y = msg.metadata.origin.position.y
            
            current_costmap = np.array(msg.data, dtype=np.uint8).reshape((width, height))
            
            valid_costs = current_costmap[current_costmap < 255]
            for cost_val in range(256):
                self.cost_value_histogram_normals[cost_val] += np.sum(valid_costs == cost_val)
            
            costmap_info = {
                'data': current_costmap,
                'resolution': resolution,
                'origin_x': origin_x,
                'origin_y': origin_y,
                'width': width,
                'height': height
            }
            
            self.update_agreement_heatmap_individual(costmap_info, 'surface_normals')
            
        except Exception as e:
            self.get_logger().error(f'Failed to convert surface normals costmap: {e}')
        
        self.check_all_received()
    
    def check_all_received(self):
        """Check if all responses received, then publish next frame"""
        if (self.received_segmentation and self.received_normals and 
            self.received_costmap and self.received_costmap_seg and 
            self.received_costmap_rough and self.received_costmap_normals):
            total_latency = (self.get_clock().now() - self.publish_time).nanoseconds / 1e9
            self.get_logger().info(f'Total pipeline latency: {total_latency:.3f}s\n')
            
            self.waiting_for_response = False
            self.current_frame_idx += 1
            
            # Publish next frame
            self.publish_next_frame()
    
    def update_agreement_heatmap(self, costmap_info):
        """Calculate and accumulate both difference and variance statistics"""
        current_costmap = costmap_info['data']
        
        if self.previous_costmap is None:
            # First frame - initialize
            self.previous_costmap = costmap_info.copy()
            self.previous_pose = self.current_pose.copy()
            
            # Initialize grids
            shape = current_costmap.shape
            self.agreement_grid = np.zeros(shape, dtype=np.float64)
            self.agreement_counts = np.zeros(shape, dtype=np.int32)
            
            # NEW: Initialize variance tracking
            self.cost_value_sum = np.zeros(shape, dtype=np.float64)
            self.cost_value_sq_sum = np.zeros(shape, dtype=np.float64)
            self.cost_observation_counts = np.zeros(shape, dtype=np.int32)
            
            self.get_logger().info(f'Initialized agreement heatmap: {shape}')
            return
        
        # Calculate spatial difference (warp previous into current frame)
        diff = self.calculate_spatial_difference(
            self.previous_costmap,
            costmap_info,
            self.previous_pose,
            self.current_pose
        )
        
        # Update DIFFERENCE statistics
        valid_diff_mask = ~np.isnan(diff)
        self.agreement_grid[valid_diff_mask] += np.abs(diff[valid_diff_mask])
        self.agreement_counts[valid_diff_mask] += 1
        
        # NEW: Update VARIANCE statistics
        # Add current frame's cost values (these are in current frame, no warping needed)
        curr_map = costmap_info['data'].astype(np.float32)
        valid_cost_mask = (curr_map < 254.5) & np.isfinite(curr_map)
        
        self.cost_value_sum[valid_cost_mask] += curr_map[valid_cost_mask]
        self.cost_value_sq_sum[valid_cost_mask] += curr_map[valid_cost_mask] ** 2
        self.cost_observation_counts[valid_cost_mask] += 1
        
        # Log statistics
        num_valid_diff = np.sum(valid_diff_mask)
        num_valid_cost = np.sum(valid_cost_mask)
        
        if num_valid_diff > 0:
            mean_diff = np.mean(np.abs(diff[valid_diff_mask]))
            max_diff = np.max(np.abs(diff[valid_diff_mask]))
            self.get_logger().info(
                f'Frame-to-frame: {num_valid_diff} overlap cells, '
                f'mean diff: {mean_diff:.2f}, max diff: {max_diff:.2f}'
            )
        
        if num_valid_cost > 0:
            current_observations = self.cost_observation_counts[valid_cost_mask]
            multi_obs_mask = current_observations > 1
            if np.any(multi_obs_mask):
                # Calculate variance for cells with multiple observations
                n = self.cost_observation_counts[valid_cost_mask][multi_obs_mask]
                mean = self.cost_value_sum[valid_cost_mask][multi_obs_mask] / n
                mean_sq = self.cost_value_sq_sum[valid_cost_mask][multi_obs_mask] / n
                variance = mean_sq - mean**2
                std = np.sqrt(np.maximum(variance, 0))  # Avoid negative due to numerical errors
                
                self.get_logger().info(
                    f'Variance stats: {np.sum(multi_obs_mask)} cells with 2+ observations, '
                    f'mean std: {np.mean(std):.2f}, max std: {np.max(std):.2f}'
                )
        
        # Update previous
        self.previous_costmap = costmap_info.copy()
        self.previous_pose = self.current_pose.copy()
    
    def update_agreement_heatmap_individual(self, costmap_info, costmap_type):
        """Calculate and accumulate statistics for individual costmaps"""
        current_costmap = costmap_info['data']
        
        # Select the appropriate tracking variables based on costmap type
        if costmap_type == 'segmentation':
            prev_map = self.previous_costmap_seg
            agreement_grid = self.agreement_grid_seg
            agreement_counts = self.agreement_counts_seg
            cost_value_sum = self.cost_value_sum_seg
            cost_value_sq_sum = self.cost_value_sq_sum_seg
            cost_observation_counts = self.cost_observation_counts_seg
        elif costmap_type == 'roughness':
            prev_map = self.previous_costmap_rough
            agreement_grid = self.agreement_grid_rough
            agreement_counts = self.agreement_counts_rough
            cost_value_sum = self.cost_value_sum_rough
            cost_value_sq_sum = self.cost_value_sq_sum_rough
            cost_observation_counts = self.cost_observation_counts_rough
        elif costmap_type == 'surface_normals':
            prev_map = self.previous_costmap_normals
            agreement_grid = self.agreement_grid_normals
            agreement_counts = self.agreement_counts_normals
            cost_value_sum = self.cost_value_sum_normals
            cost_value_sq_sum = self.cost_value_sq_sum_normals
            cost_observation_counts = self.cost_observation_counts_normals
        else:
            self.get_logger().error(f'Unknown costmap type: {costmap_type}')
            return
        
        if prev_map is None:
            # First frame - initialize
            shape = current_costmap.shape
            
            if costmap_type == 'segmentation':
                self.previous_costmap_seg = costmap_info.copy()
                self.agreement_grid_seg = np.zeros(shape, dtype=np.float64)
                self.agreement_counts_seg = np.zeros(shape, dtype=np.int32)
                self.cost_value_sum_seg = np.zeros(shape, dtype=np.float64)
                self.cost_value_sq_sum_seg = np.zeros(shape, dtype=np.float64)
                self.cost_observation_counts_seg = np.zeros(shape, dtype=np.int32)
            elif costmap_type == 'roughness':
                self.previous_costmap_rough = costmap_info.copy()
                self.agreement_grid_rough = np.zeros(shape, dtype=np.float64)
                self.agreement_counts_rough = np.zeros(shape, dtype=np.int32)
                self.cost_value_sum_rough = np.zeros(shape, dtype=np.float64)
                self.cost_value_sq_sum_rough = np.zeros(shape, dtype=np.float64)
                self.cost_observation_counts_rough = np.zeros(shape, dtype=np.int32)
            elif costmap_type == 'surface_normals':
                self.previous_costmap_normals = costmap_info.copy()
                self.agreement_grid_normals = np.zeros(shape, dtype=np.float64)
                self.agreement_counts_normals = np.zeros(shape, dtype=np.int32)
                self.cost_value_sum_normals = np.zeros(shape, dtype=np.float64)
                self.cost_value_sq_sum_normals = np.zeros(shape, dtype=np.float64)
                self.cost_observation_counts_normals = np.zeros(shape, dtype=np.int32)
            
            self.get_logger().info(f'Initialized {costmap_type} agreement heatmap: {shape}')
            return
        
        # Calculate spatial difference
        diff = self.calculate_spatial_difference(
            prev_map,
            costmap_info,
            self.previous_pose,
            self.current_pose
        )
        
        # Update difference statistics
        valid_diff_mask = ~np.isnan(diff)
        agreement_grid[valid_diff_mask] += np.abs(diff[valid_diff_mask])
        agreement_counts[valid_diff_mask] += 1
        
        # Update variance statistics
        curr_map = costmap_info['data'].astype(np.float32)
        valid_cost_mask = (curr_map < 254.5) & np.isfinite(curr_map)
        
        cost_value_sum[valid_cost_mask] += curr_map[valid_cost_mask]
        cost_value_sq_sum[valid_cost_mask] += curr_map[valid_cost_mask] ** 2
        cost_observation_counts[valid_cost_mask] += 1
        
        # Update the class variables
        if costmap_type == 'segmentation':
            self.agreement_grid_seg = agreement_grid
            self.agreement_counts_seg = agreement_counts
            self.cost_value_sum_seg = cost_value_sum
            self.cost_value_sq_sum_seg = cost_value_sq_sum
            self.cost_observation_counts_seg = cost_observation_counts
            self.previous_costmap_seg = costmap_info.copy()
        elif costmap_type == 'roughness':
            self.agreement_grid_rough = agreement_grid
            self.agreement_counts_rough = agreement_counts
            self.cost_value_sum_rough = cost_value_sum
            self.cost_value_sq_sum_rough = cost_value_sq_sum
            self.cost_observation_counts_rough = cost_observation_counts
            self.previous_costmap_rough = costmap_info.copy()
        elif costmap_type == 'surface_normals':
            self.agreement_grid_normals = agreement_grid
            self.agreement_counts_normals = agreement_counts
            self.cost_value_sum_normals = cost_value_sum
            self.cost_value_sq_sum_normals = cost_value_sq_sum
            self.cost_observation_counts_normals = cost_observation_counts
            self.previous_costmap_normals = costmap_info.copy()
        
        # Log statistics
        num_valid_diff = np.sum(valid_diff_mask)
        if num_valid_diff > 0:
            mean_diff = np.mean(np.abs(diff[valid_diff_mask]))
            max_diff = np.max(np.abs(diff[valid_diff_mask]))
            self.get_logger().info(
                f'{costmap_type} overlap: {num_valid_diff} cells, '
                f'mean diff: {mean_diff:.2f}, max diff: {max_diff:.2f}'
            )
    
    def calculate_spatial_difference(
    self,
    prev_costmap_info,
    curr_costmap_info,
    prev_pose,
    curr_pose,
    ):
        """
        Compare previous costmap against current costmap
        by warping the previous costmap into the current frame.
        """

        prev_map = prev_costmap_info["data"].astype(np.float32)
        curr_map = curr_costmap_info["data"].astype(np.float32)

        h, w = curr_map.shape
        res = curr_costmap_info["resolution"]

        # Origins (map frame)
        prev_origin = np.array([
            prev_costmap_info["origin_x"],
            prev_costmap_info["origin_y"],
        ])
        curr_origin = np.array([
            curr_costmap_info["origin_x"],
            curr_costmap_info["origin_y"],
        ])

        # Poses
        prev_pos = np.array(prev_pose["pos"][:2])
        curr_pos = np.array(curr_pose["pos"][:2])

        prev_yaw = Rotation.from_quat(prev_pose["quat"]).as_euler("xyz")[2]
        curr_yaw = Rotation.from_quat(curr_pose["quat"]).as_euler("xyz")[2]

        # Rotation matrices
        R_prev = np.array([
            [np.cos(prev_yaw), -np.sin(prev_yaw)],
            [np.sin(prev_yaw),  np.cos(prev_yaw)],
        ])

        R_curr = np.array([
            [np.cos(curr_yaw), -np.sin(curr_yaw)],
            [np.sin(curr_yaw),  np.cos(curr_yaw)],
        ])

        # Relative transform: prev → curr
        R_rel = R_curr.T @ R_prev
        t_rel = R_curr.T @ (prev_pos - curr_pos)

        diff = np.full((h, w), np.nan, dtype=np.float32)

        for y in range(h):
            for x in range(w):
                curr_cost = curr_map[y, x]
                if curr_cost >= 254.5:
                    continue

                # Current cell → current rover frame
                p_curr = np.array([
                    curr_origin[0] + x * res,
                    curr_origin[1] + (h - 1 - y) * res,
                ])

                # Current rover → previous rover
                p_prev = R_rel @ p_curr + t_rel

                # Previous rover → previous costmap indices
                px = int((p_prev[0] - prev_origin[0]) / res)
                py = int((p_prev[1] - prev_origin[1]) / res)

                py = (prev_map.shape[0] - 1) - py  # flip Y for image coords

                if (
                    0 <= px < prev_map.shape[1]
                    and 0 <= py < prev_map.shape[0]
                ):
                    prev_cost = prev_map[py, px]
                    if prev_cost < 254.5:
                        diff[y, x] = curr_cost - prev_cost

        valid = ~np.isnan(diff)
        if np.any(valid):
            self.get_logger().info(
                f"Overlap cells: {np.sum(valid)}, "
                f"mean diff: {np.mean(np.abs(diff[valid])):.2f}, "
                f"max diff: {np.max(np.abs(diff[valid])):.2f}"
            )
        else:
            self.get_logger().warn("No overlapping cells found")

        return diff


    def log_individual_costmap_stats(self, name, agreement_grid, agreement_counts, 
                                     observation_counts, value_sum, value_sq_sum):
        """Log statistics for individual costmaps"""
        if agreement_counts is None:
            self.get_logger().info(f'\n{name} COSTMAP: No data collected')
            return
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info(f'SPATIAL CONSISTENCY ANALYSIS - {name} COSTMAP')
        self.get_logger().info('='*60)
        
        # Difference metrics
        diff_valid_mask = agreement_counts > 0
        if np.any(diff_valid_mask):
            avg_difference = np.zeros_like(agreement_grid)
            avg_difference[diff_valid_mask] = (agreement_grid[diff_valid_mask] / 
                                              agreement_counts[diff_valid_mask])
            
            self.get_logger().info('\nFrame-to-Frame Difference Metrics:')
            self.get_logger().info(f'  Cells analyzed: {np.sum(diff_valid_mask)}')
            self.get_logger().info(f'  Mean difference: {np.mean(avg_difference[diff_valid_mask]):.2f}')
            self.get_logger().info(f'  Median difference: {np.median(avg_difference[diff_valid_mask]):.2f}')
            self.get_logger().info(f'  Std of differences: {np.std(avg_difference[diff_valid_mask]):.2f}')
            self.get_logger().info(f'  Max difference: {np.max(avg_difference[diff_valid_mask]):.2f}')
        
        # Variance metrics
        var_valid_mask = observation_counts > 1
        if np.any(var_valid_mask):
            n = observation_counts[var_valid_mask]
            mean_cost = value_sum[var_valid_mask] / n
            mean_sq = value_sq_sum[var_valid_mask] / n
            variance = mean_sq - mean_cost**2
            variance = np.maximum(variance, 0)
            std_cost = np.sqrt(variance)
            
            self.get_logger().info('\nVariance-Based Metrics:')
            self.get_logger().info(f'  Cells with 2+ observations: {np.sum(var_valid_mask)}')
            self.get_logger().info(f'  Mean cost value: {np.mean(mean_cost):.2f}')
            self.get_logger().info(f'  Mean std dev: {np.mean(std_cost):.2f}')
            self.get_logger().info(f'  Median std dev: {np.median(std_cost):.2f}')
            self.get_logger().info(f'  Max std dev: {np.max(std_cost):.2f}')
            
            # Coefficient of variation
            mean_nonzero = mean_cost > 1e-6
            if np.any(mean_nonzero):
                cv = std_cost[mean_nonzero] / mean_cost[mean_nonzero]
                self.get_logger().info(f'  Mean CV: {np.mean(cv):.3f}')
                self.get_logger().info(f'  Median CV: {np.median(cv):.3f}')
                self.get_logger().info(f'  % cells with CV < 0.2: {100*np.sum(cv < 0.2)/len(cv):.1f}%')
                self.get_logger().info(f'  % cells with CV < 0.3: {100*np.sum(cv < 0.3)/len(cv):.1f}%')

   
    def generate_results(self):
        """Generate and display/save final results"""
        self.get_logger().info('='*60)
        self.get_logger().info('COST MODULE PIPELINE TEST RESULTS')
        self.get_logger().info('='*60)
        
        # Timing statistics
        if self.segmentation_latencies:
            self.get_logger().info(f'\nSegmentation:')
            self.get_logger().info(f'  Mean: {np.mean(self.segmentation_latencies):.3f}s')
            self.get_logger().info(f'  Std:  {np.std(self.segmentation_latencies):.3f}s')
            self.get_logger().info(f'  Min:  {np.min(self.segmentation_latencies):.3f}s')
            self.get_logger().info(f'  Max:  {np.max(self.segmentation_latencies):.3f}s')
        
        if self.normals_latencies:
            self.get_logger().info(f'\nSurface Normals:')
            self.get_logger().info(f'  Mean: {np.mean(self.normals_latencies):.3f}s')
            self.get_logger().info(f'  Std:  {np.std(self.normals_latencies):.3f}s')
            self.get_logger().info(f'  Min:  {np.min(self.normals_latencies):.3f}s')
            self.get_logger().info(f'  Max:  {np.max(self.normals_latencies):.3f}s')
        
        if self.costmap_latencies:
            self.get_logger().info(f'\nCostmap Generation (Combined):')
            self.get_logger().info(f'  Mean: {np.mean(self.costmap_latencies):.3f}s')
            self.get_logger().info(f'  Std:  {np.std(self.costmap_latencies):.3f}s')
            self.get_logger().info(f'  Min:  {np.min(self.costmap_latencies):.3f}s')
            self.get_logger().info(f'  Max:  {np.max(self.costmap_latencies):.3f}s')
        
        if self.costmap_seg_latencies:
            self.get_logger().info(f'\nSegmentation Costmap:')
            self.get_logger().info(f'  Mean: {np.mean(self.costmap_seg_latencies):.3f}s')
            self.get_logger().info(f'  Std:  {np.std(self.costmap_seg_latencies):.3f}s')
            self.get_logger().info(f'  Min:  {np.min(self.costmap_seg_latencies):.3f}s')
            self.get_logger().info(f'  Max:  {np.max(self.costmap_seg_latencies):.3f}s')
        
        if self.costmap_rough_latencies:
            self.get_logger().info(f'\nRoughness Costmap:')
            self.get_logger().info(f'  Mean: {np.mean(self.costmap_rough_latencies):.3f}s')
            self.get_logger().info(f'  Std:  {np.std(self.costmap_rough_latencies):.3f}s')
            self.get_logger().info(f'  Min:  {np.min(self.costmap_rough_latencies):.3f}s')
            self.get_logger().info(f'  Max:  {np.max(self.costmap_rough_latencies):.3f}s')
        
        if self.costmap_normals_latencies:
            self.get_logger().info(f'\nSurface Normals Costmap:')
            self.get_logger().info(f'  Mean: {np.mean(self.costmap_normals_latencies):.3f}s')
            self.get_logger().info(f'  Std:  {np.std(self.costmap_normals_latencies):.3f}s')
            self.get_logger().info(f'  Min:  {np.min(self.costmap_normals_latencies):.3f}s')
            self.get_logger().info(f'  Max:  {np.max(self.costmap_normals_latencies):.3f}s')
        
        # Calculate total pipeline latency
        if self.costmap_latencies:  # Costmap is last in pipeline
            self.get_logger().info(f'\nTotal Pipeline:')
            self.get_logger().info(f'  Mean: {np.mean(self.costmap_latencies):.3f}s')
            self.get_logger().info(f'  Std:  {np.std(self.costmap_latencies):.3f}s')
        
           # Generate both agreement heatmaps
        if self.agreement_counts is not None and np.sum(self.agreement_counts > 0) > 0:
            # Calculate DIFFERENCE metrics
            diff_valid_mask = self.agreement_counts > 0
            avg_difference = np.zeros_like(self.agreement_grid)
            avg_difference[diff_valid_mask] = (self.agreement_grid[diff_valid_mask] / 
                                            self.agreement_counts[diff_valid_mask])
            
            # Calculate VARIANCE metrics
            var_valid_mask = self.cost_observation_counts > 1
            
            mean_cost = np.zeros_like(self.cost_value_sum)
            std_cost = np.zeros_like(self.cost_value_sum)
            cv_cost = np.zeros_like(self.cost_value_sum)
            
            if np.any(var_valid_mask):
                n = self.cost_observation_counts[var_valid_mask]
                mean_cost[var_valid_mask] = self.cost_value_sum[var_valid_mask] / n
                mean_sq = self.cost_value_sq_sum[var_valid_mask] / n
                variance = mean_sq - mean_cost[var_valid_mask]**2
                variance = np.maximum(variance, 0)  # Numerical stability
                std_cost[var_valid_mask] = np.sqrt(variance)
                
                # Coefficient of variation (avoid division by zero)
                mean_nonzero = mean_cost[var_valid_mask] > 1e-6
                if np.any(mean_nonzero):
                    cv_mask = var_valid_mask.copy()
                    cv_mask[var_valid_mask] = mean_nonzero
                    cv_cost[cv_mask] = (std_cost[cv_mask] / mean_cost[cv_mask])
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Row 1: Difference-based metrics
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(avg_difference, cmap='hot', interpolation='nearest', origin='lower')
            ax1.set_title('Frame-to-Frame Difference\n(Lower = More Consistent)', 
                        fontsize=12, fontweight='bold')
            ax1.set_xlabel('X (cells)')
            ax1.set_ylabel('Y (cells)')
            plt.colorbar(im1, ax=ax1, label='Avg Absolute Difference')
            
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(self.agreement_counts, cmap='viridis', interpolation='nearest', origin='lower')
            ax2.set_title('Difference Sample Counts', fontsize=12, fontweight='bold')
            ax2.set_xlabel('X (cells)')
            ax2.set_ylabel('Y (cells)')
            plt.colorbar(im2, ax=ax2, label='Number of Comparisons')
            
            # Difference histogram
            ax3 = fig.add_subplot(gs[0, 2])
            if np.any(diff_valid_mask):
                ax3.hist(avg_difference[diff_valid_mask], bins=50, alpha=0.7, 
                        edgecolor='black', color='coral')
                ax3.set_title('Difference Distribution', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Average Difference (cost units)')
                ax3.set_ylabel('Frequency')
                ax3.axvline(np.mean(avg_difference[diff_valid_mask]), color='r', 
                        linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(avg_difference[diff_valid_mask]):.2f}')
                ax3.legend()
                ax3.grid(alpha=0.3)
            
            # Row 2: Variance-based metrics
            ax4 = fig.add_subplot(gs[1, 0])
            im4 = ax4.imshow(std_cost, cmap='hot', interpolation='nearest', origin='lower')
            ax4.set_title('Standard Deviation\n(Lower = More Consistent)', 
                        fontsize=12, fontweight='bold')
            ax4.set_xlabel('X (cells)')
            ax4.set_ylabel('Y (cells)')
            plt.colorbar(im4, ax=ax4, label='Std Dev (cost units)')
            
            ax5 = fig.add_subplot(gs[1, 1])
            # Mask out invalid CV values for better visualization
            cv_display = cv_cost.copy()
            cv_display[~var_valid_mask] = np.nan
            im5 = ax5.imshow(cv_display, cmap='hot', interpolation='nearest', 
                            origin='lower', vmin=0, vmax=0.5)
            ax5.set_title('Coefficient of Variation\n(Std/Mean, Lower = More Consistent)', 
                        fontsize=12, fontweight='bold')
            ax5.set_xlabel('X (cells)')
            ax5.set_ylabel('Y (cells)')
            plt.colorbar(im5, ax=ax5, label='CV (dimensionless)')
            
            ax6 = fig.add_subplot(gs[1, 2])
            im6 = ax6.imshow(self.cost_observation_counts, cmap='viridis', 
                            interpolation='nearest', origin='lower')
            ax6.set_title('Variance Sample Counts', fontsize=12, fontweight='bold')
            ax6.set_xlabel('X (cells)')
            ax6.set_ylabel('Y (cells)')
            plt.colorbar(im6, ax=ax6, label='Number of Observations')
            
            # Row 3: Detailed timing analysis with histograms for each stage
            ax7 = fig.add_subplot(gs[2, 0])
            if self.segmentation_latencies:
                seg_data = np.array(self.segmentation_latencies)
                ax7.hist(seg_data, bins=20, alpha=0.7, edgecolor='black', color='coral')
                ax7.axvline(np.mean(seg_data), color='r', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(seg_data):.3f}s')
                ax7.axvline(np.median(seg_data), color='b', linestyle=':', linewidth=2,
                           label=f'Median: {np.median(seg_data):.3f}s')
                ax7.set_title(f'Segmentation Latency\nσ={np.std(seg_data):.3f}s', 
                            fontsize=12, fontweight='bold')
                ax7.set_xlabel('Latency (s)')
                ax7.set_ylabel('Frequency')
                ax7.legend(fontsize=9)
                ax7.grid(alpha=0.3)
            
            ax8 = fig.add_subplot(gs[2, 1])
            if self.normals_latencies:
                norm_data = np.array(self.normals_latencies)
                ax8.hist(norm_data, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
                ax8.axvline(np.mean(norm_data), color='r', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(norm_data):.3f}s')
                ax8.axvline(np.median(norm_data), color='b', linestyle=':', linewidth=2,
                           label=f'Median: {np.median(norm_data):.3f}s')
                ax8.set_title(f'Surface Normals Latency\nσ={np.std(norm_data):.3f}s', 
                            fontsize=12, fontweight='bold')
                ax8.set_xlabel('Latency (s)')
                ax8.set_ylabel('Frequency')
                ax8.legend(fontsize=9)
                ax8.grid(alpha=0.3)
            
            # Costmap/Total latency histogram
            ax9 = fig.add_subplot(gs[2, 2])
            if self.costmap_latencies:
                cost_data = np.array(self.costmap_latencies)
                ax9.hist(cost_data, bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
                ax9.axvline(np.mean(cost_data), color='r', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(cost_data):.3f}s')
                ax9.axvline(np.median(cost_data), color='b', linestyle=':', linewidth=2,
                           label=f'Median: {np.median(cost_data):.3f}s')
                ax9.set_title(f'Total Pipeline Latency\nσ={np.std(cost_data):.3f}s', 
                            fontsize=12, fontweight='bold')
                ax9.set_xlabel('Latency (s)')
                ax9.set_ylabel('Frequency')
                ax9.legend(fontsize=9)
                ax9.grid(alpha=0.3)
            
            # Log comprehensive statistics
            self.get_logger().info('\n' + '='*60)
            self.get_logger().info('SPATIAL CONSISTENCY ANALYSIS - COMBINED COSTMAP')
            self.get_logger().info('='*60)
            
            if np.any(diff_valid_mask):
                self.get_logger().info('\nFrame-to-Frame Difference Metrics:')
                self.get_logger().info(f'  Cells analyzed: {np.sum(diff_valid_mask)}')
                self.get_logger().info(f'  Mean difference: {np.mean(avg_difference[diff_valid_mask]):.2f}')
                self.get_logger().info(f'  Median difference: {np.median(avg_difference[diff_valid_mask]):.2f}')
                self.get_logger().info(f'  Std of differences: {np.std(avg_difference[diff_valid_mask]):.2f}')
                self.get_logger().info(f'  Max difference: {np.max(avg_difference[diff_valid_mask]):.2f}')
            
            if np.any(var_valid_mask):
                self.get_logger().info('\nVariance-Based Metrics:')
                self.get_logger().info(f'  Cells with 2+ observations: {np.sum(var_valid_mask)}')
                self.get_logger().info(f'  Mean cost value: {np.mean(mean_cost[var_valid_mask]):.2f}')
                self.get_logger().info(f'  Mean std dev: {np.mean(std_cost[var_valid_mask]):.2f}')
                self.get_logger().info(f'  Median std dev: {np.median(std_cost[var_valid_mask]):.2f}')
                self.get_logger().info(f'  Max std dev: {np.max(std_cost[var_valid_mask]):.2f}')
                
                cv_valid = var_valid_mask & (mean_cost > 1e-6)
                if np.any(cv_valid):
                    self.get_logger().info(f'  Mean CV: {np.mean(cv_cost[cv_valid]):.3f}')
                    self.get_logger().info(f'  Median CV: {np.median(cv_cost[cv_valid]):.3f}')
                    self.get_logger().info(f'  % cells with CV < 0.2: {100*np.sum(cv_cost[cv_valid] < 0.2)/np.sum(cv_valid):.1f}%')
                    self.get_logger().info(f'  % cells with CV < 0.3: {100*np.sum(cv_cost[cv_valid] < 0.3)/np.sum(cv_valid):.1f}%')
            
            # Log individual costmap statistics
            self.log_individual_costmap_stats('SEGMENTATION', 
                                            self.agreement_grid_seg, 
                                            self.agreement_counts_seg,
                                            self.cost_observation_counts_seg,
                                            self.cost_value_sum_seg,
                                            self.cost_value_sq_sum_seg)
            
            self.log_individual_costmap_stats('ROUGHNESS', 
                                            self.agreement_grid_rough, 
                                            self.agreement_counts_rough,
                                            self.cost_observation_counts_rough,
                                            self.cost_value_sum_rough,
                                            self.cost_value_sq_sum_rough)
            
            self.log_individual_costmap_stats('SURFACE NORMALS', 
                                            self.agreement_grid_normals, 
                                            self.agreement_counts_normals,
                                            self.cost_observation_counts_normals,
                                            self.cost_value_sum_normals,
                                            self.cost_value_sq_sum_normals)
            
            # Create output directory
            output_dir = Path.home() / 'cost_module_results'
            output_dir.mkdir(exist_ok=True)
            
            # Save combined visualization
            output_path = output_dir / 'combined_overview.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.get_logger().info(f'\nCombined results saved to: {output_path}')

            # Save individual plots for combined costmap
            overlap_mask = diff_valid_mask & var_valid_mask
            self.save_individual_plots(
                output_dir, 'combined', avg_difference, diff_valid_mask, std_cost, cv_cost, 
                var_valid_mask, mean_cost, overlap_mask, self.cost_value_histogram, None,
                self.agreement_counts, self.cost_observation_counts
            )
            
            # Save individual plots for segmentation costmap
            if self.agreement_counts_seg is not None:
                self.save_costmap_plots(output_dir, 'segmentation',
                                       self.agreement_grid_seg,
                                       self.agreement_counts_seg,
                                       self.cost_value_sum_seg,
                                       self.cost_value_sq_sum_seg,
                                       self.cost_observation_counts_seg,
                                       self.cost_value_histogram_seg,
                                       self.costmap_seg_latencies)
            
            # Save individual plots for roughness costmap
            if self.agreement_counts_rough is not None:
                self.save_costmap_plots(output_dir, 'roughness',
                                       self.agreement_grid_rough,
                                       self.agreement_counts_rough,
                                       self.cost_value_sum_rough,
                                       self.cost_value_sq_sum_rough,
                                       self.cost_observation_counts_rough,
                                       self.cost_value_histogram_rough,
                                       self.costmap_rough_latencies)
            
            # Save individual plots for surface normals costmap
            if self.agreement_counts_normals is not None:
                self.save_costmap_plots(output_dir, 'surface_normals',
                                       self.agreement_grid_normals,
                                       self.agreement_counts_normals,
                                       self.cost_value_sum_normals,
                                       self.cost_value_sq_sum_normals,
                                       self.cost_observation_counts_normals,
                                       self.cost_value_histogram_normals,
                                       self.costmap_normals_latencies)
            
            # Save raw data
            results_data = {
                'segmentation_latencies': np.array(self.segmentation_latencies),
                'normals_latencies': np.array(self.normals_latencies),
                'costmap_latencies': np.array(self.costmap_latencies),
                'costmap_seg_latencies': np.array(self.costmap_seg_latencies),
                'costmap_rough_latencies': np.array(self.costmap_rough_latencies),
                'costmap_normals_latencies': np.array(self.costmap_normals_latencies),
                'avg_difference': avg_difference,
                'difference_counts': self.agreement_counts,
                'mean_cost': mean_cost,
                'std_cost': std_cost,
                'cv_cost': cv_cost,
                'observation_counts': self.cost_observation_counts,
                'cost_value_histogram': self.cost_value_histogram,
            }
            
            # Add individual costmap data if available
            if self.agreement_counts_seg is not None:
                diff_valid_mask_seg = self.agreement_counts_seg > 0
                avg_difference_seg = np.zeros_like(self.agreement_grid_seg)
                if np.any(diff_valid_mask_seg):
                    avg_difference_seg[diff_valid_mask_seg] = (
                        self.agreement_grid_seg[diff_valid_mask_seg] / 
                        self.agreement_counts_seg[diff_valid_mask_seg]
                    )
                results_data['avg_difference_seg'] = avg_difference_seg
                results_data['difference_counts_seg'] = self.agreement_counts_seg
                results_data['observation_counts_seg'] = self.cost_observation_counts_seg
                results_data['cost_value_histogram_seg'] = self.cost_value_histogram_seg
                
            if self.agreement_counts_rough is not None:
                diff_valid_mask_rough = self.agreement_counts_rough > 0
                avg_difference_rough = np.zeros_like(self.agreement_grid_rough)
                if np.any(diff_valid_mask_rough):
                    avg_difference_rough[diff_valid_mask_rough] = (
                        self.agreement_grid_rough[diff_valid_mask_rough] / 
                        self.agreement_counts_rough[diff_valid_mask_rough]
                    )
                results_data['avg_difference_rough'] = avg_difference_rough
                results_data['difference_counts_rough'] = self.agreement_counts_rough
                results_data['observation_counts_rough'] = self.cost_observation_counts_rough
                results_data['cost_value_histogram_rough'] = self.cost_value_histogram_rough
                
            if self.agreement_counts_normals is not None:
                diff_valid_mask_normals = self.agreement_counts_normals > 0
                avg_difference_normals = np.zeros_like(self.agreement_grid_normals)
                if np.any(diff_valid_mask_normals):
                    avg_difference_normals[diff_valid_mask_normals] = (
                        self.agreement_grid_normals[diff_valid_mask_normals] / 
                        self.agreement_counts_normals[diff_valid_mask_normals]
                    )
                results_data['avg_difference_normals'] = avg_difference_normals
                results_data['difference_counts_normals'] = self.agreement_counts_normals
                results_data['observation_counts_normals'] = self.cost_observation_counts_normals
                results_data['cost_value_histogram_normals'] = self.cost_value_histogram_normals
            
            data_path = output_dir / 'test_results_data.npz'
            np.savez(data_path, **results_data)
            self.get_logger().info(f'Raw data saved to: {data_path}')
            self.get_logger().info(f'\nAll results saved to: {output_dir}')
            
            plt.show()

        else:
            self.get_logger().warn('No agreement heatmap data to visualize (only 1 frame or no valid comparisons)')
        
        self.get_logger().info('='*60)
        self.get_logger().info('Test complete. Shutting down...')
        
        # Shutdown
        rclpy.shutdown()

    def save_costmap_plots(self, output_dir, costmap_name, agreement_grid, agreement_counts,
                          value_sum, value_sq_sum, observation_counts, histogram, latencies):
        """Generate and save all plots for a specific costmap"""
        
        # Calculate metrics
        diff_valid_mask = agreement_counts > 0
        avg_difference = np.zeros_like(agreement_grid)
        if np.any(diff_valid_mask):
            avg_difference[diff_valid_mask] = (agreement_grid[diff_valid_mask] / 
                                              agreement_counts[diff_valid_mask])
        
        var_valid_mask = observation_counts > 1
        mean_cost = np.zeros_like(value_sum)
        std_cost = np.zeros_like(value_sum)
        cv_cost = np.zeros_like(value_sum)
        
        if np.any(var_valid_mask):
            n = observation_counts[var_valid_mask]
            mean_cost[var_valid_mask] = value_sum[var_valid_mask] / n
            mean_sq = value_sq_sum[var_valid_mask] / n
            variance = mean_sq - mean_cost[var_valid_mask]**2
            variance = np.maximum(variance, 0)
            std_cost[var_valid_mask] = np.sqrt(variance)
            
            mean_nonzero = mean_cost[var_valid_mask] > 1e-6
            if np.any(mean_nonzero):
                cv_mask = var_valid_mask.copy()
                cv_mask[var_valid_mask] = mean_nonzero
                cv_cost[cv_mask] = (std_cost[cv_mask] / mean_cost[cv_mask])
        
        overlap_mask = diff_valid_mask & var_valid_mask
        
        # Save all individual plots
        self.save_individual_plots(output_dir, costmap_name, avg_difference, diff_valid_mask,
                                  std_cost, cv_cost, var_valid_mask, mean_cost, overlap_mask,
                                  histogram, latencies, agreement_counts, observation_counts)

    def save_individual_plots(self, output_dir, costmap_name, avg_difference, diff_valid_mask, 
                            std_cost, cv_cost, var_valid_mask, mean_cost, overlap_mask,
                            histogram, latencies=None, agreement_counts_data=None, 
                            observation_counts_data=None):
        """Save each visualization as a separate image file"""

        title_fontsize = 18
        label_fontsize = 16
        tick_fontsize = 14
        dpi = 800
        
        prefix = f"{costmap_name}_"
        self.get_logger().info(f'Saving {costmap_name} costmap plots to: {output_dir}')
        
        # 1. Frame-to-Frame Difference Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(avg_difference, cmap='hot', interpolation='nearest', origin='lower')
        ax.set_title('Frame-to-Frame Difference\n(Lower = More Consistent)', 
                    fontsize=title_fontsize, fontweight='bold')
        ax.set_xlabel('X (cells)', fontsize=label_fontsize)
        ax.set_ylabel('Y (cells)', fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        cbar = plt.colorbar(im, ax=ax, label='Avg Absolute Difference', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.set_label('Avg Absolute Difference', fontsize=label_fontsize)
        plt.tight_layout()
        plt.savefig(output_dir / f'{prefix}1_difference_heatmap.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # 2. Difference Sample Counts
        fig, ax = plt.subplots(figsize=(10, 8))
        if agreement_counts_data is None:
            agreement_counts_data = self.agreement_counts
        im = ax.imshow(agreement_counts_data, cmap='viridis', interpolation='nearest', origin='lower')
        ax.set_title('Difference Sample Counts', fontsize=title_fontsize, fontweight='bold')
        ax.set_xlabel('X (cells)', fontsize=label_fontsize)
        ax.set_ylabel('Y (cells)', fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        cbar = plt.colorbar(im, ax=ax, label='Number of Comparisons', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.set_label('Number of Comparisons', fontsize=label_fontsize)
        plt.tight_layout()
        plt.savefig(output_dir / f'{prefix}2_difference_sample_counts.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # 3. Difference Distribution Histogram
        if np.any(diff_valid_mask):
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.hist(avg_difference[diff_valid_mask], bins=50, alpha=0.7, 
                    edgecolor='black', color='coral')
            ax.set_title('Difference Distribution', fontsize=title_fontsize, fontweight='bold')
            ax.set_xlabel('Average Difference (cost units)', fontsize=label_fontsize)
            ax.set_ylabel('Frequency', fontsize=label_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            ax.axvline(np.mean(avg_difference[diff_valid_mask]), color='r', 
                    linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(avg_difference[diff_valid_mask]):.2f}')
            ax.legend(fontsize=label_fontsize)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f'{prefix}3_difference_histogram.png', dpi=dpi, bbox_inches='tight')
            plt.close()
        
        # 4. Standard Deviation Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(std_cost, cmap='hot', interpolation='nearest', origin='lower')
        ax.set_title('Standard Deviation\n(Lower = More Consistent)', 
                    fontsize=title_fontsize, fontweight='bold')
        ax.set_xlabel('X (cells)', fontsize=label_fontsize)
        ax.set_ylabel('Y (cells)', fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        cbar = plt.colorbar(im, ax=ax, label='Std Dev (cost units)', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.set_label('Std Dev (cost units)', fontsize=label_fontsize)
        plt.tight_layout()
        plt.savefig(output_dir / f'{prefix}4_std_dev_heatmap.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # 5. Coefficient of Variation
        fig, ax = plt.subplots(figsize=(10, 8))
        cv_display = cv_cost.copy()
        cv_display[~var_valid_mask] = np.nan
        im = ax.imshow(cv_display, cmap='hot', interpolation='nearest', 
                    origin='lower', vmin=0, vmax=0.5)
        ax.set_title('Coefficient of Variation\n(Std/Mean, Lower = More Consistent)', 
                    fontsize=title_fontsize, fontweight='bold')
        ax.set_xlabel('X (cells)', fontsize=label_fontsize)
        ax.set_ylabel('Y (cells)', fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        cbar = plt.colorbar(im, ax=ax, label='CV (dimensionless)', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.set_label('CV (dimensionless)', fontsize=label_fontsize)
        plt.tight_layout()
        plt.savefig(output_dir / f'{prefix}5_coefficient_of_variation.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # 6. Variance Sample Counts
        fig, ax = plt.subplots(figsize=(10, 8))
        if observation_counts_data is None:
            observation_counts_data = self.cost_observation_counts
        im = ax.imshow(observation_counts_data, cmap='viridis', 
                    interpolation='nearest', origin='lower')
        ax.set_title('Variance Sample Counts', fontsize=title_fontsize, fontweight='bold')
        ax.set_xlabel('X (cells)', fontsize=label_fontsize)
        ax.set_ylabel('Y (cells)', fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        cbar = plt.colorbar(im, ax=ax, label='Number of Observations', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.set_label('Number of Observations', fontsize=label_fontsize)
        plt.tight_layout()
        plt.savefig(output_dir / f'{prefix}6_variance_sample_counts.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        
        # 7-9. Latency Distribution (only for combined costmap, uses main pipeline latencies)
        # For individual costmaps, plot their specific costmap generation latency
        if latencies is not None and len(latencies) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            latency_data = np.array(latencies) * 1000  # Convert to ms
            ax.hist(latency_data, bins=20, alpha=0.7, edgecolor='black', color='lightgreen',
                   weights=np.ones(len(latency_data)) / len(latency_data))
            ax.axvline(np.mean(latency_data), color='r', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(latency_data):.1f}ms')
            ax.axvline(np.median(latency_data), color='b', linestyle=':', linewidth=2,
                      label=f'Median: {np.median(latency_data):.1f}ms')
            ax.set_title(f'{costmap_name.replace("_", " ").title()} Costmap Latency\\nσ={np.std(latency_data):.1f}ms', 
                        fontsize=title_fontsize, fontweight='bold')
            ax.set_xlabel('Latency (ms)', fontsize=label_fontsize)
            ax.set_ylabel('Frequency', fontsize=label_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            ax.legend(fontsize=label_fontsize)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f'{prefix}7_costmap_latency.png', dpi=dpi, bbox_inches='tight')
            plt.close()
        elif costmap_name == 'combined':
            # For combined, include segmentation, normals, and total pipeline
            if self.segmentation_latencies:
                fig, ax = plt.subplots(figsize=(10, 8))
                seg_data = np.array(self.segmentation_latencies) * 1000
                ax.hist(seg_data, bins=20, alpha=0.7, edgecolor='black', color='coral',
                       weights=np.ones(len(seg_data)) / len(seg_data))
                ax.axvline(np.mean(seg_data), color='r', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(seg_data):.1f}ms')
                ax.axvline(np.median(seg_data), color='b', linestyle=':', linewidth=2,
                          label=f'Median: {np.median(seg_data):.1f}ms')
                ax.set_title(f'Segmentation Latency Distribution\\nσ={np.std(seg_data):.1f}ms', 
                            fontsize=title_fontsize, fontweight='bold')
                ax.set_xlabel('Latency (ms)', fontsize=label_fontsize)
                ax.set_ylabel('Frequency', fontsize=label_fontsize)
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                ax.legend(fontsize=label_fontsize)
                ax.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / f'{prefix}7_segmentation_latency.png', dpi=dpi, bbox_inches='tight')
                plt.close()
            
            if self.normals_latencies:
                fig, ax = plt.subplots(figsize=(10, 8))
                norm_data = np.array(self.normals_latencies) * 1000
                ax.hist(norm_data, bins=20, alpha=0.7, edgecolor='black', color='skyblue',
                       weights=np.ones(len(norm_data)) / len(norm_data))
                ax.axvline(np.mean(norm_data), color='r', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(norm_data):.1f}ms')
                ax.axvline(np.median(norm_data), color='b', linestyle=':', linewidth=2,
                          label=f'Median: {np.median(norm_data):.1f}ms')
                ax.set_title(f'Surface Normals Latency Distribution\\nσ={np.std(norm_data):.1f}ms', 
                            fontsize=title_fontsize, fontweight='bold')
                ax.set_xlabel('Latency (ms)', fontsize=label_fontsize)
                ax.set_ylabel('Frequency', fontsize=label_fontsize)
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                ax.legend(fontsize=label_fontsize)
                ax.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / f'{prefix}8_normals_latency.png', dpi=dpi, bbox_inches='tight')
                plt.close()
            
            if self.costmap_latencies:
                fig, ax = plt.subplots(figsize=(10, 8))
                cost_data = np.array(self.costmap_latencies) * 1000
                ax.hist(cost_data, bins=20, alpha=0.7, edgecolor='black', color='lightgreen',
                       weights=np.ones(len(cost_data)) / len(cost_data))
                ax.axvline(np.mean(cost_data), color='r', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(cost_data):.1f}ms')
                ax.axvline(np.median(cost_data), color='b', linestyle=':', linewidth=2,
                          label=f'Median: {np.median(cost_data):.1f}ms')
                ax.set_title(f'Total Pipeline Latency Distribution\\nσ={np.std(cost_data):.1f}ms', 
                            fontsize=title_fontsize, fontweight='bold')
                ax.set_xlabel('Latency (ms)', fontsize=label_fontsize)
                ax.set_ylabel('Frequency', fontsize=label_fontsize)
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                ax.legend(fontsize=label_fontsize)
                ax.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / f'{prefix}9_total_pipeline_latency.png', dpi=dpi, bbox_inches='tight')
                plt.close()

        # 10. Cost Value Distribution Across All Frames
        if np.sum(histogram) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Normalize by total pixel count to get probability distribution
            total_pixels = np.sum(histogram[:-1])  # Exclude value 255 (unknown)
            normalized_hist = histogram[:-1] / total_pixels if total_pixels > 0 else histogram[:-1]
            
            # Create bar plot
            cost_values = np.arange(0, 255)
            ax.bar(cost_values, normalized_hist, color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.8)
            
            # Add statistics
            weighted_mean = np.sum(cost_values * normalized_hist)
            weighted_std = np.sqrt(np.sum(((cost_values - weighted_mean) ** 2) * normalized_hist))
            
            ax.axvline(weighted_mean, color='r', linestyle='--', linewidth=2.5,
                    label=f'Mean: {weighted_mean:.1f}')
            ax.axvline(weighted_mean + weighted_std, color='orange', linestyle=':', linewidth=2,
                    label=f'±1σ: {weighted_std:.1f}')
            ax.axvline(weighted_mean - weighted_std, color='orange', linestyle=':', linewidth=2)
            
            ax.set_title(f'{costmap_name.replace("_", " ").title()} - Cost Value Distribution', 
                        fontsize=title_fontsize, fontweight='bold')
            ax.set_xlabel('Cost Value (0-254)', fontsize=label_fontsize)
            ax.set_ylabel('Normalized Frequency', fontsize=label_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            ax.legend(fontsize=label_fontsize, loc='upper right')
            ax.grid(alpha=0.3, axis='y')
            ax.set_xlim([0, 254])
            
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{prefix}10_cost_value_distribution.png', dpi=dpi, bbox_inches='tight')
            plt.close()

        self.get_logger().info(f'{costmap_name} costmap plots saved to: {output_dir}')
        

def main(args=None):
    rclpy.init(args=args)
    
    node = CostModuleTester()
    
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