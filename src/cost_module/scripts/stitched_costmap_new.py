#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import math
import threading
import time
import csv
from pathlib import Path
import struct

from nav2_msgs.msg import Costmap
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, Header
from sensor_msgs.msg import Image, PointCloud2, PointField
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


class FrameGatedStitchedCostmap(Node):
    """
    Combined frame-gated synchronizer + stitched global costmap.
    
    Workflow:
      1. Load poses from CSV
      2. Publish pose for current frame
      3. Wait for costmap response (with timestamp validation)
      4. Stitch costmap into global map
      5. Move to next frame
      6. Repeat until all frames processed
    """

    def __init__(self):
        super().__init__('frame_gated_stitched_costmap')

        # =====================================================
        # CAMERA PARAMETERS
        # =====================================================
        self.declare_parameter('camera.intrinsics.fx', 336.1)
        self.declare_parameter('camera.intrinsics.fy', 385.6)
        self.declare_parameter('camera.intrinsics.cx', 480.0)
        self.declare_parameter('camera.intrinsics.cy', 270.0)
        self.declare_parameter('camera.resolution.width', 960)
        self.declare_parameter('camera.resolution.height', 540)
        
        self.fx = self.get_parameter('camera.intrinsics.fx').value
        self.fy = self.get_parameter('camera.intrinsics.fy').value
        self.cx = self.get_parameter('camera.intrinsics.cx').value
        self.cy = self.get_parameter('camera.intrinsics.cy').value
        self.img_width = self.get_parameter('camera.resolution.width').value
        self.img_height = self.get_parameter('camera.resolution.height').value
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # =====================================================
        # DATASET CONFIGURATION
        # =====================================================
        self.dataset_dir = Path.home() / 'tamt' / 'src' / 'sync_pkg' / 'frame_data'
        self.poses_csv = self.dataset_dir / 'rover_poses.csv'

        # =====================================================
        # FIXED WORLD DEFINITION (MATCHES ISAAC SIM)
        # =====================================================
        self.WORLD_MIN_X = 0.0
        self.WORLD_MAX_X = 8.0
        self.WORLD_MIN_Y = -6.0
        self.WORLD_MAX_Y = 2.0

        # =====================================================
        # COSTMAP FUSION POLICY
        # =====================================================
        # Options:
        #   'max'       -> conservative (keep worst cost ever seen)
        #   'overwrite' -> always trust latest observation
        #   'average'   -> average overlapping observations
        self.fusion_policy = 'max'

        self.hit_count = None

        # =====================================================
        # FRAME TRAVERSAL POLICY
        # =====================================================
        # Options:
        #   'forward' -> 0 → N-1 (default)
        #   'reverse' -> start_frame → 0
        self.frame_traversal = 'forward'
        self.start_frame = 19


        # =====================================================
        # Resolution / grid (initialized lazily from first costmap)
        # =====================================================
        self.resolution = None
        self.width = None
        self.height = None
        self.grid = None
        self.observed = None
        self.map_initialized = False

        # =====================================================
        # Frame sequencing state
        # =====================================================
        self.frames = self.load_poses()
        self.total_frames = len(self.frames)

        if self.frame_traversal == 'reverse':
            self.current_frame = min(self.start_frame, self.total_frames - 1)
        else:
            self.current_frame = 0

        
        # Synchronization flags (pattern from document 3)
        self.waiting_for_costmap = False
        self.last_pose_stamp = None
        
        # Thread safety
        self.lock = threading.Lock()

        if self.total_frames == 0:
            self.get_logger().fatal('No frames loaded from CSV.')
            rclpy.shutdown()
            return

        # =====================================================
        # ROS TOPICS
        # =====================================================
        # Input: costmap from processing pipeline
        self.costmap_in_topic = '/tamt/costmap/combined'
        
        # Output: synchronized sensor data for pipeline
        self.rgb_out_topic = '/tamt/sync/rgb'
        self.depth_out_topic = '/tamt/sync/depth'
        self.pointcloud_out_topic = '/tamt/sync/pointcloud'
        self.pose_out_topic = '/tamt/sync/rover_pose'
        
        # Output: gated data for monitoring
        self.gated_pose_topic = '/tamt/gated/rover_pose'
        self.frame_id_topic = '/tamt/gated/frame_id'

        # =====================================================
        # ROS INTERFACES
        # =====================================================
        # Publishers for pipeline input (sensor data)
        self.rgb_pub = self.create_publisher(Image, self.rgb_out_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_out_topic, 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, self.pointcloud_out_topic, 10)
        self.pose_sync_pub = self.create_publisher(PoseStamped, self.pose_out_topic, 10)
        
        # Publishers for monitoring
        self.pose_gated_pub = self.create_publisher(PoseStamped, self.gated_pose_topic, 10)
        self.frame_pub = self.create_publisher(Int32, self.frame_id_topic, 10)

        # Subscriber for costmap output
        self.costmap_sub = self.create_subscription(
            Costmap,
            self.costmap_in_topic,
            self.costmap_callback,
            50
        )

        # =====================================================
        # Matplotlib (initialized after resolution known)
        # =====================================================
        plt.ion()
        self.fig = None
        self.ax = None
        self.im = None
        self.last_plot = time.time()

        # =====================================================
        # Start processing
        # =====================================================
        self.get_logger().info(
            f"Frame-gated stitched costmap initialized:\n"
            f"  - Frames: {self.total_frames}\n"
            f"  - Fusion policy: {self.fusion_policy}\n"
            f"  - World bounds: X[{self.WORLD_MIN_X}, {self.WORLD_MAX_X}], "
            f"Y[{self.WORLD_MIN_Y}, {self.WORLD_MAX_Y}]"
        )

        # Publish first frame
        self.publish_current_pose()

    # =====================================================
    # DATASET LOADING
    # =====================================================
    def load_poses(self):
        """Load rover poses and file paths from CSV file"""
        frames = []

        if not self.poses_csv.exists():
            self.get_logger().error(f'Poses CSV not found: {self.poses_csv}')
            return frames

        with open(self.poses_csv, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not row or row[0].startswith('#') or row[0] == 'frame_index':
                    continue

                # Extract RGB filename
                rgb_filename = row[1]
                
                # Extract frame number for depth file
                frame_num = rgb_filename.split('_')[-1].split('.')[0]
                depth_filename = f'depth_{frame_num}.npy'
                
                frames.append({
                    'x': float(row[2]),
                    'y': float(row[3]),
                    'z': float(row[4]),
                    'qx': float(row[5]),
                    'qy': float(row[6]),
                    'qz': float(row[7]),
                    'qw': float(row[8]),
                    'rgb_filename': rgb_filename,
                    'depth_filename': depth_filename,
                    'rgb_path': self.dataset_dir / rgb_filename,
                    'depth_path': self.dataset_dir / 'depth' / depth_filename,
                })

        self.get_logger().info(f'Loaded {len(frames)} poses from {self.poses_csv}')
        return frames

    # =====================================================
    # DATA LOADING HELPERS (from document 3)
    # =====================================================
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

    # =====================================================
    # FRAME SEQUENCING (with synchronization from doc 3)
    # =====================================================
    def publish_current_pose(self):
        """Load sensor data, publish to pipeline, and wait for costmap"""
        if (
            (self.frame_traversal == 'forward' and self.current_frame >= self.total_frames) or
            (self.frame_traversal == 'reverse' and self.current_frame < 0)
        ):

            self.get_logger().info('✓ All frames processed!')
            self.get_logger().info('Global costmap construction complete.')
            # Keep visualization open
            plt.ioff()
            plt.show()
            return

        # Get current frame data
        frame = self.frames[self.current_frame]
        
        self.get_logger().info(
            f'Processing frame {self.current_frame + 1}/{self.total_frames}'
        )
        
        # =====================================================
        # Load RGB image
        # =====================================================
        rgb_img = cv2.imread(str(frame['rgb_path']), cv2.IMREAD_UNCHANGED)
        if rgb_img is None:
            self.get_logger().error(f'Failed to load RGB: {frame["rgb_path"]}')
            self.get_logger().error(f'Skipping to next frame')
            with self.lock:
                self.current_frame += 1
            # Skip this frame and try next (not recursive - just tail call)
            if self.current_frame < self.total_frames:
                self.publish_current_pose()
            return
        
        # Convert BGR to RGB
        if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 3:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # =====================================================
        # Load depth data
        # =====================================================
        depth = self.load_npy_depth(frame['depth_path'])
        if depth is None:
            self.get_logger().error(f'Failed to load depth: {frame["depth_path"]}')
            self.get_logger().error(f'Skipping to next frame')
            with self.lock:
                self.current_frame += 1
            # Skip this frame and try next (not recursive - just tail call)
            if self.current_frame < self.total_frames:
                self.publish_current_pose()
            return
        
        # =====================================================
        # Create timestamp
        # =====================================================
        now = self.get_clock().now()
        timestamp = now.to_msg()
        
        # =====================================================
        # Publish RGB
        # =====================================================
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding='rgb8')
        rgb_msg.header.stamp = timestamp
        rgb_msg.header.frame_id = 'left_camera'
        self.rgb_pub.publish(rgb_msg)
        
        # =====================================================
        # Publish Depth
        # =====================================================
        depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding='32FC1')
        depth_msg.header.stamp = timestamp
        depth_msg.header.frame_id = 'camera_depth_frame'
        self.depth_pub.publish(depth_msg)
        
        # =====================================================
        # Create and publish pointcloud
        # =====================================================
        pc_msg = self.create_pointcloud(rgb_img, depth)
        self.pointcloud_pub.publish(pc_msg)
        
        # =====================================================
        # Create and publish pose
        # =====================================================
        pose = PoseStamped()
        pose.header.stamp = timestamp
        pose.header.frame_id = 'map'
        pose.pose.position.x = frame['x']
        pose.pose.position.y = frame['y']
        pose.pose.position.z = frame['z']
        pose.pose.orientation.x = frame['qx']
        pose.pose.orientation.y = frame['qy']
        pose.pose.orientation.z = frame['qz']
        pose.pose.orientation.w = frame['qw']

        # Publish to pipeline sync topic
        self.pose_sync_pub.publish(pose)
        
        # Publish to gated topic for monitoring
        self.pose_gated_pub.publish(pose)

        # Publish frame ID
        fid = Int32()
        fid.data = self.current_frame
        self.frame_pub.publish(fid)

        # Set synchronization state
        with self.lock:
            self.last_pose_stamp = now
            self.waiting_for_costmap = True

        self.get_logger().info(
            f'→ Published all sensor data for frame {self.current_frame}/{self.total_frames - 1}'
        )

    # =====================================================
    # COSTMAP CALLBACK (with timestamp gating from doc 2)
    # =====================================================
    def costmap_callback(self, msg: Costmap):
        """
        Receive costmap, validate timestamp, stitch into global map.
        Pattern from document 3: check waiting flag before processing.
        """
        should_publish_next = False
        
        with self.lock:
            # Check if we're waiting for a response (pattern from doc 3)
            if not self.waiting_for_costmap:
                self.get_logger().debug('Received costmap but not waiting - ignoring')
                return

            # Validate timestamp (strict gating from doc 2)
            costmap_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
            if costmap_stamp <= self.last_pose_stamp:
                self.get_logger().debug(
                    'Ignoring stale costmap (older than current pose)'
                )
                return
            
            self.get_logger().debug(f'Accepted costmap for frame {self.current_frame}')

            # Initialize global map from first costmap (releases lock temporarily)
            if not self.map_initialized:
                # Release lock before matplotlib initialization
                pass  # Will initialize outside lock below
                
        # Initialize outside lock if needed (matplotlib can be slow)
        if not self.map_initialized:
            self.initialize_global_map(msg)
            
        with self.lock:
            # Stitch costmap into global map
            self.stitch_costmap(msg)

            self.get_logger().info(
                f'✓ Stitched costmap (frame index: {self.current_frame})'
            )


            # Mark costmap as received (pattern from doc 3)
            self.waiting_for_costmap = False
            
            # Move to next frame based on traversal policy
            if self.frame_traversal == 'forward':
                self.current_frame += 1
            else:  # reverse
                self.current_frame -= 1

            should_publish_next = True


        # Update visualization OUTSIDE the lock to avoid blocking
        self.update_plot()
        
        # Publish next frame outside lock
        if should_publish_next:
            self.get_logger().debug(f'Moving to next frame: {self.current_frame}')
            self.publish_current_pose()

    # =====================================================
    # GLOBAL MAP INITIALIZATION
    # =====================================================
    def initialize_global_map(self, msg: Costmap):
        """Initialize global costmap grid from first incoming costmap"""
        self.resolution = msg.metadata.resolution

        self.width = int((self.WORLD_MAX_X - self.WORLD_MIN_X) / self.resolution)
        self.height = int((self.WORLD_MAX_Y - self.WORLD_MIN_Y) / self.resolution)

        # Initialize with unknown (255) everywhere
        self.grid = np.full((self.height, self.width), 255, dtype=np.uint8)
        self.observed = np.zeros((self.height, self.width), dtype=bool)

        # Only needed for averaging
        self.hit_count = np.zeros((self.height, self.width), dtype=np.uint32)

        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Global Costmap Builder')
        self.im = self.ax.imshow(
            np.zeros((self.height, self.width)),
            origin='lower',
            cmap='gray',
            vmin=0,
            vmax=255,
            interpolation='nearest',
            extent=[
                self.WORLD_MIN_X,
                self.WORLD_MAX_X,
                self.WORLD_MIN_Y,
                self.WORLD_MAX_Y
            ]
        )

        self.ax.set_title("Global Costmap (Frame-by-Frame Stitching)")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.fig.colorbar(self.im, ax=self.ax)
        
        # Draw initially to create the window
        plt.pause(0.001)  # Very short pause to create window without blocking

        self.map_initialized = True

        self.get_logger().info(
            f"Initialized global map: {self.width} x {self.height} cells "
            f"@ {self.resolution:.3f} m/cell"
        )

    # =====================================================
    # COSTMAP STITCHING (from document 1)
    # =====================================================
    def stitch_costmap(self, msg: Costmap):
        """Stitch local costmap into global map using current pose"""
        # Get current pose from frame data
        frame = self.frames[self.current_frame]
        tx, ty = frame['x'], frame['y']
        
        # Calculate yaw from quaternion
        quat_obj = type('obj', (object,), {
            'x': frame['qx'], 
            'y': frame['qy'], 
            'z': frame['qz'], 
            'w': frame['qw']
        })()
        yaw = yaw_from_quaternion(quat_obj)

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Local costmap metadata
        res = msg.metadata.resolution
        sx = msg.metadata.size_x
        sy = msg.metadata.size_y
        ox = msg.metadata.origin.position.x
        oy = msg.metadata.origin.position.y

        # nav2 Costmap is column-major → transpose
        local = np.array(msg.data, dtype=np.uint8).reshape(sx, sy).T
        #local = np.fliplr(local)   # fixes left/right
        local = np.flipud(local)   # fixes up/down

        # Stitch each cell
        for iy in range(sy):
            for ix in range(sx):
                cost = local[iy, ix]

                # Skip unknown cells
                if cost == 255:
                    continue

                # Local costmap → rover frame
                lx = ix * res + ox
                ly = iy * res + oy

                # Rover frame → world frame
                wx = tx + cos_y * lx - sin_y * ly
                wy = ty + sin_y * lx + cos_y * ly

                # World → global grid
                gx = int((wx - self.WORLD_MIN_X) / self.resolution)
                gy = int((wy - self.WORLD_MIN_Y) / self.resolution)

                # Update global grid
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    if not self.observed[gy, gx]:
                        # First observation
                        self.grid[gy, gx] = cost
                        self.observed[gy, gx] = True
                        self.hit_count[gy, gx] = 1
                    else:
                        if self.fusion_policy == 'max':
                            self.grid[gy, gx] = max(self.grid[gy, gx], cost)

                        elif self.fusion_policy == 'overwrite':
                            self.grid[gy, gx] = cost

                        elif self.fusion_policy == 'average':
                            # Incremental mean:
                            n = self.hit_count[gy, gx]
                            new_val = (self.grid[gy, gx] * n + cost) / (n + 1)
                            self.grid[gy, gx] = int(round(new_val))
                            self.hit_count[gy, gx] += 1


    # =====================================================
    # VISUALIZATION
    # =====================================================
    def update_plot(self):
        """Update matplotlib visualization (non-blocking)"""
        if not self.map_initialized:
            return
            
        now = time.time()
        if now - self.last_plot < 0.1:
            return

        # Prepare visualization: invert costs for display (0=black, 255=white)
        view = np.zeros_like(self.grid, dtype=float)
        view[self.observed] = 255 - self.grid[self.observed]
        view[~self.observed] = 0  # Unobserved = black

        self.im.set_data(view)
        self.ax.set_title(
            f"Global Costmap (Frame {self.current_frame - 1}/{self.total_frames - 1})"
        )
        
        # Use plt.pause for non-blocking update
        plt.pause(0.001)
        self.last_plot = now


def main():
    rclpy.init()
    node = FrameGatedStitchedCostmap()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()