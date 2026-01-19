#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import math
import threading
import time

from nav2_msgs.msg import Costmap
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


class StitchedGlobalCostmap(Node):

    def __init__(self):
        super().__init__('stitched_global_costmap')

        # =====================================================
        # FIXED WORLD DEFINITION (MATCHES ISAAC SIM)
        # =====================================================
        self.WORLD_MIN_X = -5.0
        self.WORLD_MAX_X =  5.0
        self.WORLD_MIN_Y = -5.0
        self.WORLD_MAX_Y =  5.0

        # =====================================================
        # COSTMAP FUSION POLICY
        # =====================================================
        # Options:
        #   'max'       -> conservative (keep worst cost ever seen)
        #   'overwrite' -> always trust latest observation
        self.fusion_policy = 'overwrite'

        # =====================================================
        # Resolution / grid (initialized lazily)
        # =====================================================
        self.resolution = None
        self.width = None
        self.height = None
        self.grid = None
        self.observed = None
        self.initialized = False

        self.lock = threading.Lock()

        # =====================================================
        # Rover pose state (world frame)
        # =====================================================
        self.pose_valid = False
        self.rover_x = 0.0
        self.rover_y = 0.0
        self.rover_yaw = 0.0

        # =====================================================
        # Topics
        # =====================================================
        self.pose_topic = '/tamt/gated/rover_pose'
        self.local_costmap_topic = '/tamt/gated/costmap/combined'

        # =====================================================
        # ROS interfaces
        # =====================================================
        self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10
        )

        self.create_subscription(
            Costmap,
            self.local_costmap_topic,
            self.costmap_callback,
            10
        )

        # =====================================================
        # Matplotlib (initialized after resolution known)
        # =====================================================
        plt.ion()
        self.fig = None
        self.ax = None
        self.im = None
        self.last_plot = time.time()

        self.get_logger().info(
            f"Stitched global costmap running (fusion_policy={self.fusion_policy})"
        )

    # =====================================================
    # Rover pose callback
    # =====================================================
    def pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.rover_x = msg.pose.position.x
            self.rover_y = msg.pose.position.y
            self.rover_yaw = yaw_from_quaternion(msg.pose.orientation)
            self.pose_valid = True

    # =====================================================
    # Initialize global map from first costmap
    # =====================================================
    def initialize_from_costmap(self, msg: Costmap):

        self.resolution = msg.metadata.resolution

        self.width = int((self.WORLD_MAX_X - self.WORLD_MIN_X) / self.resolution)
        self.height = int((self.WORLD_MAX_Y - self.WORLD_MIN_Y) / self.resolution)

        self.grid = np.full((self.height, self.width), 255, dtype=np.uint8)
        self.observed = np.zeros((self.height, self.width), dtype=bool)

        # Matplotlib setup
        self.fig, self.ax = plt.subplots()
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

        self.ax.set_title("Global Costmap (Isaac Sim World)")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.fig.colorbar(self.im, ax=self.ax)

        self.initialized = True

        self.get_logger().info(
            f"Initialized global costmap with resolution {self.resolution:.3f} m "
            f"({self.width} x {self.height} cells)"
        )

    # =====================================================
    # Costmap stitching
    # =====================================================
    def costmap_callback(self, msg: Costmap):

        if not self.pose_valid:
            self.get_logger().warn("No rover pose yet, skipping costmap")
            return

        if not self.initialized:
            self.initialize_from_costmap(msg)

        with self.lock:
            tx = self.rover_x
            ty = self.rover_y
            yaw = self.rover_yaw

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        res = msg.metadata.resolution
        sx = msg.metadata.size_x
        sy = msg.metadata.size_y
        ox = msg.metadata.origin.position.x
        oy = msg.metadata.origin.position.y

        # nav2 Costmap is column-major → transpose
        local = np.array(msg.data, dtype=np.uint8).reshape(sx, sy).T

        with self.lock:
            for iy in range(sy):
                for ix in range(sx):
                    cost = local[iy, ix]

                    # Unknown → skip entirely
                    if cost == 255:
                        continue

                    # Local costmap → rover frame
                    lx = ix * res + ox
                    ly = iy * res + oy

                    # Rover frame → world frame
                    wx = tx + cos_y * lx - sin_y * ly
                    wy = ty + sin_y * lx + cos_y * ly

                    # World → grid
                    gx = int((wx - self.WORLD_MIN_X) / self.resolution)
                    gy = int((wy - self.WORLD_MIN_Y) / self.resolution)

                    if 0 <= gx < self.width and 0 <= gy < self.height:

                        if not self.observed[gy, gx]:
                            self.grid[gy, gx] = cost
                            self.observed[gy, gx] = True
                        else:
                            if self.fusion_policy == 'max':
                                self.grid[gy, gx] = max(self.grid[gy, gx], cost)
                            elif self.fusion_policy == 'overwrite':
                                self.grid[gy, gx] = cost
                            else:
                                raise ValueError(
                                    f"Unknown fusion_policy: {self.fusion_policy}"
                                )

        self.update_plot()

    # =====================================================
    # Visualization
    # =====================================================
    def update_plot(self):
        now = time.time()
        if now - self.last_plot < 0.1:
            return

        with self.lock:
            view = np.zeros_like(self.grid, dtype=float)
            view[self.observed] = 255 - self.grid[self.observed]
            view[~self.observed] = 0
            self.im.set_data(view)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.last_plot = now


def main():
    rclpy.init()
    node = StitchedGlobalCostmap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
