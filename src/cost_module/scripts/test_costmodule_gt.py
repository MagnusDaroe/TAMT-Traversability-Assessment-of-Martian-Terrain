#!/usr/bin/env python3
"""
Ground Truth Costmap Capture Node

This node:
1. Reads frame data (poses) from frame_data directory
2. Loads GROUND TRUTH segmentation masks (sem_ids_*.png) and surface normals (normals_*.npy)
3. Publishes GT data directly to cost module (bypassing YOLO and surface normal estimator)
4. Captures all costmap outputs (combined, segmentation, roughness, surface_normals)
5. Saves each costmap as a numbered grayscale image in separate folders

Output Structure:
  ~/costmap_captures_gt/
    combined/
      frame_001.png
      frame_002.png
      ...
    segmentation/
      frame_001.png
      ...
    roughness/
      frame_001.png
      ...
    surface_normals/
      frame_001.png
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


class GroundTruthCostmapCaptureNode(Node):
    def __init__(self):
        super().__init__('gt_costmap_capture_node')
        
        # Declare and get camera intrinsics parameters
        self.declare_parameter('camera.intrinsics.fx', 336.1)
        self.declare_parameter('camera.intrinsics.fy', 385.6)
        self.declare_parameter('camera.intrinsics.cx', 480.0)
        self.declare_parameter('camera.intrinsics.cy', 270.0)
        self.declare_parameter('camera.max_distance', 3.0)
        
        # Parameter to flip GT normals if needed (for coordinate system compatibility)
        self.declare_parameter('gt.flip_normals', False)
        
        # Get camera intrinsics for depth to pointcloud conversion
        self.fx = self.get_parameter('camera.intrinsics.fx').value
        self.fy = self.get_parameter('camera.intrinsics.fy').value
        self.cx = self.get_parameter('camera.intrinsics.cx').value
        self.cy = self.get_parameter('camera.intrinsics.cy').value
        self.max_distance = self.get_parameter('camera.max_distance').value
        self.flip_normals = self.get_parameter('gt.flip_normals').value
        
        # Publishers for ground truth data
        self.segmentation_pub = self.create_publisher(Image, '/tamt/segmentation/masks_with_confidence', 10)
        self.normals_pub = self.create_publisher(Image, '/tamt/surface_normals/normals', 10)
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
        self.output_base = Path.home() / 'costmap_captures_gt'
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
        self.get_logger().info(f'Camera intrinsics - fx: {self.fx:.1f}, fy: {self.fy:.1f}, cx: {self.cx:.1f}, cy: {self.cy:.1f}')
        
        # Load frame data
        self.load_frame_data()
        
        if len(self.frames) == 0:
            self.get_logger().error('No frames loaded! Shutting down.')
            rclpy.shutdown()
            return
        
        self.get_logger().info(f'Loaded {len(self.frames)} frames with ground truth data.')
        self.get_logger().info('Starting capture with GT segmentation and normals...')
        
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
                
                # Extract frame number
                rgb_file = parts[1]
                frame_num = rgb_file.split('_')[-1].split('.')[0]
                
                # Build paths for GT data
                frame_data = {
                    'frame_num': frame_num,
                    'pos': [float(parts[2]), float(parts[3]), float(parts[4])],
                    'quat': [float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])],
                    'yolo_label_path': data_dir / 'labels' / f'rgb_{frame_num}.txt',
                    'normals_path': data_dir / 'normals' / f'normals_{frame_num}.npy',
                    'depth_path': data_dir / 'depth' / f'depth_{frame_num}.npy',
                    'image_width': 960,
                    'image_height': 540
                }
                
                # Only add if all GT files exist
                if (frame_data['yolo_label_path'].exists() and 
                    frame_data['normals_path'].exists() and 
                    frame_data['depth_path'].exists()):
                    self.frames.append(frame_data)
    
    def publish_next_frame(self):
        """Publish next frame of ground truth data"""
        if self.current_frame_idx >= len(self.frames):
            self.get_logger().info('All frames processed. Capture complete!')
            self.get_logger().info(f'GT costmaps saved to: {self.output_base}')
            # Stop the timer and signal completion instead of calling shutdown directly
            self.timer.cancel()
            raise KeyboardInterrupt  # Clean shutdown
        
        if self.waiting_for_response:
            return
        
        frame = self.frames[self.current_frame_idx]
        self.get_logger().info(f'Processing frame {self.current_frame_idx + 1}/{len(self.frames)} (frame_{frame["frame_num"]})')
        
        # Load GT Segmentation from YOLO labels
        seg_img = self.yolo_to_segmentation_mask(
            frame['yolo_label_path'], 
            frame['image_width'], 
            frame['image_height']
        )
        
        if seg_img is None:
            self.get_logger().error(f'Failed to load segmentation: {frame["yolo_label_path"]}')
            self.current_frame_idx += 1
            self.publish_next_frame()
            return
        
        # Load GT Surface Normals (normals_*.npy)
        try:
            normals = np.load(frame['normals_path'])  # Shape should be (H, W, 3) or (H, W, 4)
            
            # If normals have 4 channels (RGBA), extract only RGB (first 3 channels)
            if normals.shape[2] == 4:
                normals = normals[:, :, :3]
            elif normals.shape[2] != 3:
                raise ValueError(f"Expected 3 or 4 channels, got {normals.shape[2]}")
                
        except Exception as e:
            self.get_logger().error(f'Failed to load normals: {frame["normals_path"]}: {e}')
            self.current_frame_idx += 1
            self.publish_next_frame()
            return
        
        # Load GT Depth (depth_*.npy)
        try:
            depth = np.load(frame['depth_path'])  # Shape should be (H, W)
            
            if depth.ndim != 2:
                raise ValueError(f"Expected 2D depth image, got shape {depth.shape}")
                
        except Exception as e:
            self.get_logger().error(f'Failed to load depth: {frame["depth_path"]}: {e}')
            self.current_frame_idx += 1
            self.publish_next_frame()
            return
        
        # Create timestamp
        timestamp = self.get_clock().now()
        
        # Publish GT Segmentation
        # Convert 8-bit class IDs to 16-bit encoded format: (class_id << 8) | confidence
        # For ground truth, set confidence = 1.0 (255 in uint8)
        seg_encoded = np.zeros((seg_img.shape[0], seg_img.shape[1]), dtype=np.uint16)
        seg_encoded = (seg_img.astype(np.uint16) << 8) | 255  # class_id in upper 8 bits, confidence=255 in lower 8 bits
        
        seg_msg = self.bridge.cv2_to_imgmsg(seg_encoded, encoding='16UC1')
        seg_msg.header.stamp = timestamp.to_msg()
        seg_msg.header.frame_id = 'camera_depth_frame'
        self.segmentation_pub.publish(seg_msg)
        
        # Log segmentation statistics
        unique_classes = np.unique(seg_img)
        self.get_logger().debug(f'Published GT segmentation: {seg_img.shape[0]}x{seg_img.shape[1]}, unique classes: {unique_classes}')
        
        # Publish GT Surface Normals as Image (3-channel float32)
        # The normals array should be (H, W, 3) with float values
        # Convert to 32FC3 encoding (same as surface_normal_estimator.cpp)
        # 
        # Optionally flip normals if coordinate system is different
        if self.flip_normals:
            normals = -normals  # Flip all components
            self.get_logger().info(f'Flipped GT normals (gt.flip_normals=true)')
        
        # Log detailed statistics about what we're publishing
        valid_mask = np.linalg.norm(normals, axis=2) > 0.1
        num_valid = np.sum(valid_mask)
        if num_valid > 0:
            valid_normals = normals[valid_mask]
            self.get_logger().info(
                f'GT normals stats: {num_valid}/{normals.shape[0]*normals.shape[1]} valid, '
                f'nz range=[{valid_normals[:,2].min():.3f}, {valid_normals[:,2].max():.3f}], '
                f'nz mean={valid_normals[:,2].mean():.3f}'
            )
        
        normals_msg = self.bridge.cv2_to_imgmsg(normals.astype(np.float32), encoding='32FC3')
        normals_msg.header.stamp = timestamp.to_msg()
        normals_msg.header.frame_id = 'camera_depth_frame'
        
        self.normals_pub.publish(normals_msg)
        
        # Log normals statistics
        valid_normals = normals[np.linalg.norm(normals, axis=2) > 0.1]
        if len(valid_normals) > 0:
            self.get_logger().debug(f'Published GT normals: {normals.shape[0]}x{normals.shape[1]}, '
                                  f'{len(valid_normals)} valid, range: [{normals.min():.2f}, {normals.max():.2f}]')
        else:
            self.get_logger().warn(f'Published GT normals: {normals.shape[0]}x{normals.shape[1]}, NO VALID NORMALS!')
        
        # Publish GT Pointcloud (converted from depth)
        pc_msg = self.depth_to_pointcloud(depth, timestamp, 'camera_depth_frame')
        self.pointcloud_pub.publish(pc_msg)
        
        self.get_logger().debug(f'Published GT pointcloud: {depth.shape[0]}x{depth.shape[1]}')
        
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
        """Save costmap as grayscale image"""
        img = costmap_data.copy()
        
        # Rotate 90 degrees counter-clockwise (to the left)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Generate filename
        filename = f'frame_{self.current_frame_idx + 1:03d}.png'
        filepath = self.output_dirs[costmap_type] / filename
        
        # Save image as grayscale
        cv2.imwrite(str(filepath), img)
        self.get_logger().debug(f'Saved {costmap_type}: {filename}')
    
    def yolo_to_segmentation_mask(self, yolo_label_path, width, height):
        """
        Convert YOLO polygon format to segmentation mask.
        
        YOLO format: class_id x1 y1 x2 y2 ... (normalized coordinates)
        
        Args:
            yolo_label_path: Path to YOLO label file
            width: Image width in pixels
            height: Image height in pixels
        
        Returns:
            np.ndarray: Segmentation mask (H, W) with class IDs
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if not yolo_label_path.exists():
            return mask
        
        with open(yolo_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:  # Need at least class + 3 points (6 coords)
                    continue
                
                class_id = int(parts[0])
                # Convert normalized coordinates to pixels
                coords = []
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        x = float(parts[i]) * width
                        y = float(parts[i + 1]) * height
                        coords.append([x, y])
                
                if len(coords) >= 3:
                    # Convert to numpy array and draw filled polygon
                    pts = np.array(coords, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], class_id)
        
        return mask
    
    def depth_to_pointcloud(self, depth, timestamp, frame_id):
        """
        Convert depth image to organized PointCloud2 message.
        Based on sync_pkg/include/depth_to_pointcloud.hpp
        
        Args:
            depth: numpy array of shape (H, W) with depth values in meters
            timestamp: rclpy.time.Time object
            frame_id: frame ID for the pointcloud
        
        Returns:
            sensor_msgs.msg.PointCloud2
        """
        height, width = depth.shape
        
        # Create PointCloud2 message
        msg = PointCloud2()
        msg.header.stamp = timestamp.to_msg()
        msg.header.frame_id = frame_id
        
        # Set up organized point cloud (maintains image structure)
        msg.height = height
        msg.width = width
        msg.is_dense = False
        msg.is_bigendian = False
        
        # Define point fields (x, y, z as float32)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        msg.point_step = 12  # 3 fields * 4 bytes
        msg.row_step = msg.point_step * width
        
        # Preallocate data buffer
        msg.data = bytearray(msg.row_step * height)
        
        # Convert depth to points using pinhole camera model
        # Note: Cost module will filter points based on ray_length > max_distance
        for v in range(height):
            for u in range(width):
                depth_value = depth[v, u]
                
                # Calculate byte offset for this point
                offset = (v * width + u) * msg.point_step
                
                # Skip invalid depth values
                if depth_value <= 0.0 or np.isnan(depth_value) or np.isinf(depth_value):
                    # Write NaN for invalid points
                    struct.pack_into('fff', msg.data, offset, 
                                   float('nan'), float('nan'), float('nan'))
                else:
                    # Back-project to 3D using pinhole camera model
                    z = depth_value
                    x = (u - self.cx) * depth_value / self.fx
                    y = (v - self.cy) * depth_value / self.fy
                    
                    struct.pack_into('fff', msg.data, offset, x, y, z)
        
        return msg
    
    def combined_callback(self, msg):
        """Callback for combined costmap"""
        if not self.waiting_for_response or self.received_combined:
            return
        
        try:
            width = msg.metadata.size_x
            height = msg.metadata.size_y
            # Reshape as (width, height) then transpose
            costmap_data = np.array(msg.data, dtype=np.uint8).reshape((width, height)).T
            
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
            costmap_data = np.array(msg.data, dtype=np.uint8).reshape((width, height)).T
            
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
            costmap_data = np.array(msg.data, dtype=np.uint8).reshape((width, height)).T
            
            self.save_costmap_image(costmap_data, 'surface_normals')
            self.received_surface_normals = True
            self.check_all_received()
            
        except Exception as e:
            self.get_logger().error(f'Failed to process surface normals costmap: {e}')
    
    def check_all_received(self):
        """Check if all costmaps received, then publish next frame"""
        if (self.received_combined and self.received_segmentation and 
            self.received_roughness and self.received_surface_normals):
            
            self.get_logger().info(f'✓ Frame {self.current_frame_idx + 1} captured (GT)')
            
            self.waiting_for_response = False
            self.current_frame_idx += 1
            
            # Publish next frame immediately
            self.publish_next_frame()


def main(args=None):
    rclpy.init(args=args)
    
    node = GroundTruthCostmapCaptureNode()
    
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
