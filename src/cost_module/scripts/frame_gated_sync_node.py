#!/usr/bin/env python3
"""
Unified Ground Truth Replay + Global Costmap Stitching Node

This node:
1. Iterates through frame_data (0..N)
2. Publishes GT segmentation, GT normals, GT depth-derived pointcloud, and pose
3. Waits for the combined costmap for that frame
4. Immediately stitches the costmap into a global world map
5. Visualizes the stitched global costmap live

No YOLO, no normal estimator, no replay node, no disk round-trip.

Pose ↔ costmap pairing is guaranteed by strict frame gating.
"""

import rclpy
from rclpy.node import Node

import numpy as np
import math
import threading
import time
import struct
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from nav2_msgs.msg import Costmap
from std_msgs.msg import Header
from ament_index_python.packages import get_package_share_directory


# ============================================================
# Helpers
# ============================================================

def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


# ============================================================
# Main Node
# ============================================================

class GTReplayAndStitch(Node):

    def __init__(self):
        super().__init__('gt_replay_and_stitch')

        # ----------------------------------------------------
        # World definition (Isaac Sim)
        # ----------------------------------------------------
        self.WORLD_MIN_X = -5.0
        self.WORLD_MAX_X =  5.0
        self.WORLD_MIN_Y = -5.0
        self.WORLD_MAX_Y =  5.0

        self.fusion_policy = 'overwrite'  # or 'max'

        # ----------------------------------------------------
        # Camera intrinsics
        # ----------------------------------------------------
        self.fx = 336.1
        self.fy = 385.6
        self.cx = 480.0
        self.cy = 270.0

        # ----------------------------------------------------
        # Publishers (GT inputs)
        # ----------------------------------------------------
        self.seg_pub = self.create_publisher(
            Image, '/tamt/segmentation/masks_with_confidence', 10)
        self.normals_pub = self.create_publisher(
            Image, '/tamt/surface_normals/normals', 10)
        self.pc_pub = self.create_publisher(
            PointCloud2, '/tamt/sync/pointcloud', 10)
        self.pose_pub = self.create_publisher(
            PoseStamped, '/tamt/sync/rover_pose', 10)

        # ----------------------------------------------------
        # Subscriber (combined costmap only)
        # ----------------------------------------------------
        self.create_subscription(
            Costmap,
            '/tamt/costmap/combined',
            self.costmap_callback,
            10
        )

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # ----------------------------------------------------
        # Global costmap state
        # ----------------------------------------------------
        self.initialized = False
        self.resolution = None
        self.grid = None
        self.observed = None
        self.width = None
        self.height = None

        # ----------------------------------------------------
        # Frame gating
        # ----------------------------------------------------
        self.frames = []
        self.current_frame_idx = 0
        self.waiting_for_costmap = False

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------
        plt.ion()
        self.fig = None
        self.ax = None
        self.im = None
        self.last_plot = time.time()

        # ----------------------------------------------------
        # Load frame data
        # ----------------------------------------------------
        self.load_frames()

        if not self.frames:
            self.get_logger().fatal("No frames loaded")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Loaded {len(self.frames)} frames")

        # Start
        self.publish_next_frame()

    # ========================================================
    # Data loading
    # ========================================================

    def load_frames(self):
        pkg = get_package_share_directory('sync_pkg')
        base = Path(pkg) / 'frame_data'

        poses_file = base / 'rover_poses.csv'

        with open(poses_file, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#') or 'frame_index' in line:
                    continue

                parts = line.strip().split(',')
                rgb_file = parts[1]
                frame_num = rgb_file.split('_')[-1].split('.')[0]

                frame = {
                    'frame_num': frame_num,
                    'pos': [float(parts[2]), float(parts[3]), float(parts[4])],
                    'quat': [float(parts[5]), float(parts[6]),
                             float(parts[7]), float(parts[8])],
                    'seg_path': base / 'labels' / f'rgb_{frame_num}.txt',
                    'normals_path': base / 'normals' / f'normals_{frame_num}.npy',
                    'depth_path': base / 'depth' / f'depth_{frame_num}.npy',
                    'w': 960,
                    'h': 540
                }

                if frame['seg_path'].exists() and \
                   frame['normals_path'].exists() and \
                   frame['depth_path'].exists():
                    self.frames.append(frame)

    # ========================================================
    # Frame publishing
    # ========================================================

    def publish_next_frame(self):
        if self.current_frame_idx >= len(self.frames):
            self.get_logger().info("All frames processed")
            return

        if self.waiting_for_costmap:
            return

        frame = self.frames[self.current_frame_idx]
        stamp = self.get_clock().now()

        self.get_logger().info(
            f"Publishing frame {self.current_frame_idx} (id {frame['frame_num']})"
        )

        # ---- Segmentation (GT)
        seg = self.yolo_to_mask(frame['seg_path'], frame['w'], frame['h'])
        seg_encoded = (seg.astype(np.uint16) << 8) | 255
        seg_msg = self.bridge.cv2_to_imgmsg(seg_encoded, encoding='16UC1')
        seg_msg.header.stamp = stamp.to_msg()
        seg_msg.header.frame_id = 'camera_depth_frame'
        self.seg_pub.publish(seg_msg)

        # ---- Normals (GT)
        normals = np.load(frame['normals_path'])[:, :, :3]
        normals_msg = self.bridge.cv2_to_imgmsg(normals.astype(np.float32), '32FC3')
        normals_msg.header.stamp = stamp.to_msg()
        normals_msg.header.frame_id = 'camera_depth_frame'
        self.normals_pub.publish(normals_msg)

        # ---- Pointcloud (from depth)
        depth = np.load(frame['depth_path'])
        pc_msg = self.depth_to_pointcloud(depth, stamp)
        self.pc_pub.publish(pc_msg)

        # ---- Pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp.to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = frame['pos']
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, \
        pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = frame['quat']
        self.pose_pub.publish(pose_msg)

        self.waiting_for_costmap = True

    # ========================================================
    # Costmap callback → stitch immediately
    # ========================================================

    def costmap_callback(self, msg: Costmap):
        if not self.waiting_for_costmap:
            return

        if not self.initialized:
            self.initialize_global_map(msg)

        frame = self.frames[self.current_frame_idx]

        yaw = yaw_from_quaternion(
            PoseStamped(
                pose=PoseStamped().pose
            ).pose.orientation
        )

        tx, ty = frame['pos'][0], frame['pos'][1]
        yaw = yaw_from_quaternion(type('q', (), {
            'x': frame['quat'][0],
            'y': frame['quat'][1],
            'z': frame['quat'][2],
            'w': frame['quat'][3]
        }))

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        res = msg.metadata.resolution
        sx = msg.metadata.size_x
        sy = msg.metadata.size_y
        ox = msg.metadata.origin.position.x
        oy = msg.metadata.origin.position.y

        local = np.array(msg.data, dtype=np.uint8).reshape(sx, sy).T

        for iy in range(sy):
            for ix in range(sx):
                cost = local[iy, ix]
                if cost == 255:
                    continue

                lx = ix * res + ox
                ly = iy * res + oy

                wx = tx + cos_y * lx - sin_y * ly
                wy = ty + sin_y * lx + cos_y * ly

                gx = int((wx - self.WORLD_MIN_X) / self.resolution)
                gy = int((wy - self.WORLD_MIN_Y) / self.resolution)

                if 0 <= gx < self.width and 0 <= gy < self.height:
                    if not self.observed[gy, gx] or self.fusion_policy == 'overwrite':
                        self.grid[gy, gx] = cost
                        self.observed[gy, gx] = True
                    elif self.fusion_policy == 'max':
                        self.grid[gy, gx] = max(self.grid[gy, gx], cost)

        self.update_plot()

        self.waiting_for_costmap = False
        self.current_frame_idx += 1
        self.publish_next_frame()

    # ========================================================
    # Global map init + visualization
    # ========================================================

    def initialize_global_map(self, msg):
        self.resolution = msg.metadata.resolution
        self.width = int((self.WORLD_MAX_X - self.WORLD_MIN_X) / self.resolution)
        self.height = int((self.WORLD_MAX_Y - self.WORLD_MIN_Y) / self.resolution)

        self.grid = np.full((self.height, self.width), 255, dtype=np.uint8)
        self.observed = np.zeros((self.height, self.width), dtype=bool)

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(
            np.zeros((self.height, self.width)),
            origin='lower',
            cmap='gray',
            vmin=0,
            vmax=255,
            extent=[
                self.WORLD_MIN_X, self.WORLD_MAX_X,
                self.WORLD_MIN_Y, self.WORLD_MAX_Y
            ]
        )
        self.fig.colorbar(self.im, ax=self.ax)
        self.initialized = True

    def update_plot(self):
        now = time.time()
        if now - self.last_plot < 0.1:
            return

        view = np.zeros_like(self.grid, dtype=float)
        view[self.observed] = 255 - self.grid[self.observed]
        self.im.set_data(view)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.last_plot = now

    # ========================================================
    # Utilities
    # ========================================================

    def yolo_to_mask(self, path, w, h):
        mask = np.zeros((h, w), dtype=np.uint8)
        with open(path, 'r') as f:
            for line in f:
                p = line.split()
                cid = int(p[0])
                pts = np.array([
                    [float(p[i]) * w, float(p[i+1]) * h]
                    for i in range(1, len(p), 2)
                ], np.int32)
                cv2.fillPoly(mask, [pts], cid)
        return mask

    def depth_to_pointcloud(self, depth, stamp):
        h, w = depth.shape
        msg = PointCloud2()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = 'camera_depth_frame'
        msg.height = h
        msg.width = w
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.point_step = 12
        msg.row_step = msg.point_step * w
        msg.is_dense = False
        msg.data = bytearray(msg.row_step * h)

        for v in range(h):
            for u in range(w):
                d = depth[v, u]
                off = (v * w + u) * msg.point_step
                if d <= 0 or not np.isfinite(d):
                    struct.pack_into('fff', msg.data, off, float('nan'), float('nan'), float('nan'))
                else:
                    x = (u - self.cx) * d / self.fx
                    y = (v - self.cy) * d / self.fy
                    struct.pack_into('fff', msg.data, off, x, y, d)

        return msg


def main():
    rclpy.init()
    node = GTReplayAndStitch()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
