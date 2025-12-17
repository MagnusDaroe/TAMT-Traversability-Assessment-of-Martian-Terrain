import csv
import math
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)


# ============================================================================
# CONFIG
# ============================================================================

CSV_PATH = (
    "/home/tamt/replicator_output/rover_data/"
    "session_20251216_132907/rover_poses.csv"
)

# Map settings (meters)
MAP_HALF_SIZE_XY = 8.0   # +/- in X and Y -> 8x8 m square
MAP_HALF_SIZE_Z  = 3.0   # +/- in Z


# ============================================================================
# MATH HELPERS
# ============================================================================

def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    """
    Quaternion [x,y,z,w] -> 3x3 rotation matrix (Hamilton).

    This matches what you write in the CSV: [qx,qy,qz,qw].
    """
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) + 1e-12
    x /= n
    y /= n
    z /= n
    w /= n

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=float)
    return R


# ============================================================================
# CSV READING
# ============================================================================

def read_rover_poses(csv_path: str) -> List[dict]:
    """
    Read rover_poses.csv written by the Isaac script.

    Current format:
        # rover_T_world (rover to world). Quaternion = [x,y,z,w] Hamilton
        frame_index,image,tx,ty,tz,qx,qy,qz,qw

    So tx,ty,tz and qx,qy,qz,qw describe the transform:
        p_W = R_WR * p_R + t_WR
    (rover frame -> world frame).
    """
    poses = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue  # comment line
            if row[0] == "frame_index":
                continue  # header

            frame_index = int(row[0])
            image_rel = row[1]
            tx = float(row[2])
            ty = float(row[3])
            tz = float(row[4])
            qx = float(row[5])
            qy = float(row[6])
            qz = float(row[7])
            qw = float(row[8])

            poses.append(
                {
                    "frame_index": frame_index,
                    "image": image_rel,
                    # This is t_WR: rover origin expressed in world coords
                    "t_wr": np.array([tx, ty, tz], dtype=float),
                    # This is q_WR: rover->world rotation
                    "q_wr": np.array([qx, qy, qz, qw], dtype=float),
                }
            )
    return poses


# ============================================================================
# MAIN VISUALIZATION (3D)
# ============================================================================

def main():
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH does not exist: {CSV_PATH}")

    poses = read_rover_poses(CSV_PATH)
    if not poses:
        raise RuntimeError("No poses found in the CSV file.")

    xs, ys, zs = [], [], []
    # Rover axes in world: X, Y, Z
    fx, fy, fz = [], [], []  # +X axis
    ux, uy, uz = [], [], []  # +Y axis
    vx, vy, vz = [], [], []  # +Z axis

    for p in poses:
        # rover -> world (R_WR, t_WR) from CSV
        R_WR = quat_xyzw_to_rot(p["q_wr"])
        t_WR = p["t_wr"]

        # Rover position in world coordinates:
        x_w, y_w, z_w = t_WR
        xs.append(x_w)
        ys.append(y_w)
        zs.append(z_w)

        # Rover axes in world (columns of R_WR)
        x_axis = R_WR[:, 0]  # +X
        y_axis = R_WR[:, 1]  # +Y
        z_axis = R_WR[:, 2]  # +Z

        fx.append(x_axis[0])
        fy.append(x_axis[1])
        fz.append(x_axis[2])

        ux.append(y_axis[0])
        uy.append(y_axis[1])
        uz.append(y_axis[2])

        vx.append(z_axis[0])
        vy.append(z_axis[1])
        vz.append(z_axis[2])

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    fx = np.array(fx); fy = np.array(fy); fz = np.array(fz)
    ux = np.array(ux); uy = np.array(uy); uz = np.array(uz)
    vx = np.array(vx); vy = np.array(vy); vz = np.array(vz)

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-MAP_HALF_SIZE_XY, MAP_HALF_SIZE_XY)
    ax.set_ylim(-MAP_HALF_SIZE_XY, MAP_HALF_SIZE_XY)
    ax.set_zlim(-MAP_HALF_SIZE_Z, MAP_HALF_SIZE_Z)

    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Y [m]")
    ax.set_zlabel("World Z [m]")
    ax.set_title("Rover poses (origin + X/Y/Z axes in world frame)")

    # Rover positions
    ax.scatter(xs, ys, zs, marker="o")

    arrow_len = 0.5  # meters

    # Orientation arrows for rover +X axis (red)
    qx = ax.quiver(
        xs, ys, zs,
        fx, fy, fz,
        length=arrow_len,
        normalize=True,
        color="red",
    )

    # Orientation arrows for rover +Y axis (green)
    qy = ax.quiver(
        xs, ys, zs,
        ux, uy, uz,
        length=arrow_len,
        normalize=True,
        color="green",
    )

    # Orientation arrows for rover +Z axis (blue)
    qz = ax.quiver(
        xs, ys, zs,
        vx, vy, vz,
        length=arrow_len,
        normalize=True,
        color="blue",
    )

    # Legend using one artist from each quiver
    ax.legend([qx, qy, qz], ["rover +X", "rover +Y", "rover +Z"])

    # World origin marker
    ax.scatter([0.0], [0.0], [0.0], marker="+", s=80, color="orange")
    ax.text(0.0, 0.0, 0.0, "World origin")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
