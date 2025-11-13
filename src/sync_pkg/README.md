# Sync Package (sync_pkg)

## Overview

The `sync_pkg` is a ROS 2 package designed for synchronizing multi-modal sensor data (RGB images, depth maps, and camera poses) in a robotics pipeline. This package is part of the **TAMT (Traversability Assessment of Martian Terrain)** project, which focuses on analyzing Martian terrain for rover navigation and scientific exploration.

### Key Features

- **Raw Data Publishing**: Loads and publishes unsynchronized RGB, depth, and pose data from file system "/data"
- **Time-Based Synchronization**: Uses ROS 2 message filters with approximate time policy to synchronize multi-modal sensor streams
- **Point Cloud Generation**: Converts depth images to 3D point clouds using camera intrinsics
- **Service-Based Triggering**: Provides a ROS 2 service to trigger synchronized data publication on-demand
- **Sanity Testing**: Includes verification node to validate synchronization quality and save synchronized frames in "/sanity_data"

---

## Repository Structure

```
sync_pkg/
├── CMakeLists.txt                          # Build configuration
├── package.xml                             # Package dependencies and metadata
├── README.md                               # This file
│
├── data/                                   # Input data directory
│   ├── data_1/
│   │   ├── rgb_img_00001.png              # RGB images
│   │   ├── depth_00001.npy                # Depth maps (numpy format)
│   │   └── campose_00001.csv              # Camera pose (CSV format)
│   ├── data_2/
│   └── data_N/
│
├── srv/                                    # Custom service definitions
│   └── TriggerSync.srv                    # Service to trigger synchronization
│
├── include/
│   └── depth_to_pointcloud.hpp            # Utility for depth-to-pointcloud conversion
│
└── src/
    ├── raw_data_publisher/
    │   └── raw_data_publisher.cpp         # Publishes unsynchronized raw data
    │
    ├── synchronised_data_publisher/
    │   ├── sync_service_node.cpp          # Synchronization service server
    │   └── sync_client_node.cpp           # Client to trigger sync service
    │
    └── sanity_tester/
        └── sanity_test.cpp                # Validates synchronization quality
```

---

## Data Flow Architecture

### 1. Data Publishing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAW DATA PUBLISHER NODE                         │
│                      (publish_raw_data executable)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Reads from filesystem:
                                    │  • data_*/rgb_img_*.png
                                    │  • data_*/depth_*.npy
                                    │  • data_*/campose_*.csv
                                    │
                        ┌───────────┼───────────┐
                        │           │           │
                        ▼           ▼           ▼
                  /left_image   /depth   /camera_pose
                (sensor_msgs)  (sensor_msgs)  (geometry_msgs)
                 [UNSYNCHRONIZED - Different timestamps]
```

### 2. Synchronization Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA SYNCHRONIZATION NODE                          │
│                    (data_synchroniser executable)                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │          Message Filters Synchronizer                          │    │
│  │    (ApproximateTime Policy with Queue Size = 50)               │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Subscribes to:                    Service:                             │
│    • /left_image                    • trigger_sync (TriggerSync)       │
│    • /depth                                                              │
│    • /camera_pose                  When triggered, publishes:           │
│                                      • /sync_rgb                         │
│  Buffers and matches timestamps     • /sync_depth                       │
│  within configured threshold        • /sync_pointcloud (generated)      │
│                                      • /sync_cam_2_glob_pose            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Service Call
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                        ▼                       ▼
            ┌──────────────────────┐   ┌──────────────────────┐
            │   SYNC CLIENT NODE   │   │  SANITY TEST NODE    │
            │ (data_synchroniser_  │   │   (sanity_test)      │
            │      client)         │   │                      │
            │                      │   │ Subscribes to synced │
            │ Triggers sync via    │   │ topics & validates   │
            │ service call         │   │ timing alignment     │
            └──────────────────────┘   └──────────────────────┘
```

### 3. Complete System Data Flow

```
┌──────────────────┐
│  File System     │
│  (data/*.png,    │
│   *.npy, *.csv)  │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Raw Data Publisher                           │
│  • Loads frames sequentially                                    │
│  • Publishes at configurable frequency (default: 1 Hz)          │
│  • Supports loop playback mode                                  │
└──────────────┬──────────────────────────────────────────────────┘
               │
               │ Publishes (unsynchronized)
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│            Message Filters Synchronizer                         │
│  • Buffers incoming messages                                    │
│  • Matches messages with similar timestamps                     │
│  • Allows slight time difference (approximate policy)           │
└──────────────┬──────────────────────────────────────────────────┘
               │
               │ Stores latest synchronized set
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│               Service Request (trigger_sync)                    │
│  • Client calls service                                         │
│  • Server publishes latest synchronized data                    │
│  • Generates point cloud from depth + intrinsics                │
└──────────────┬──────────────────────────────────────────────────┘
               │
               │ Publishes synchronized data
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│          Downstream Processing Nodes                            │
│  • Sanity Test (validation)                                     │
│  • Terrain Segmentation                                         │
│  • Surface Normal Estimation                                    │
│  • Traversability Analysis                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Node Descriptions

### 1. Raw Data Publisher (`publish_raw_data`)

**Purpose**: Reads sensor data from the file system and publishes it to ROS 2 topics.

**Published Topics**:
- `/left_image` (sensor_msgs/Image) - RGB images
- `/depth` (sensor_msgs/Image) - Depth maps
- `/camera_pose` (geometry_msgs/PoseStamped) - Camera pose in global frame

**Parameters**:
- `publish_frequency_hz` (double, default: 1.0) - Publishing rate in Hz
- `loop_playback` (bool, default: true) - Loop through dataset when finished
- `frame_id` (string, default: "camera_frame") - TF frame ID for messages

**Input Data Format**:
- RGB: PNG images (e.g., `rgb_img_00001.png`)
- Depth: NumPy files (e.g., `depth_00001.npy`)
- Pose: CSV files with format: `x,y,z,qx,qy,qz,qw`

### 2. Data Synchronization Node (`data_synchroniser`)

**Purpose**: Synchronizes multi-modal sensor streams using approximate time matching and provides synchronized data via service.

**Subscribed Topics**:
- `/left_image` (sensor_msgs/Image)
- `/depth` (sensor_msgs/Image)
- `/camera_pose` (geometry_msgs/PoseStamped)

**Published Topics** (after service trigger):
- `/sync_rgb` (sensor_msgs/Image) - Synchronized RGB
- `/sync_depth` (sensor_msgs/Image) - Synchronized depth
- `/sync_pointcloud` (sensor_msgs/PointCloud2) - Generated 3D point cloud
- `/sync_cam_2_glob_pose` (geometry_msgs/PoseStamped) - Synchronized camera pose

**Services**:
- `trigger_sync` (sync_pkg/srv/TriggerSync) - Triggers publication of synchronized data

**Parameters**:
- `fx` (double, default: 525.0) - Focal length in x (pixels)
- `fy` (double, default: 525.0) - Focal length in y (pixels)
- `cx` (double, default: 319.5) - Principal point x-coordinate (pixels)
- `cy` (double, default: 239.5) - Principal point y-coordinate (pixels)

**Synchronization Policy**:
- Uses `ApproximateTime` policy with queue size of 50
- Matches messages with nearest timestamps within threshold

### 3. Synchronization Client (`data_synchroniser_client`)

**Purpose**: Simple client to trigger the synchronization service.

**Service Client**:
- `trigger_sync` (sync_pkg/srv/TriggerSync)

### 4. Sanity Test Node (`sanity_test`)

**Purpose**: Validates synchronization by subscribing to all synchronized topics and checking timestamp alignment.

**Subscribed Topics**:
- `/sync_rgb`
- `/sync_depth`
- `/sync_pointcloud`
- `/sync_cam_2_glob_pose`

**Functionality**:
- Logs timestamp differences between synchronized messages
- Saves synchronized frames to disk for manual inspection
- Verifies data integrity and time alignment

---

## Commands Reference

### Prerequisites

```bash
# Source ROS 2 workspace
cd ~/TAMT-Traversability-Assessment-of-Martian-Terrain
source install/setup.bash

# For Python-based nodes (if using YOLO/terrain segmentation)
source setup_env.sh  # Activates venv_tamt virtual environment
```

### Building the Package

```bash
# Build all packages in workspace
colcon build

# Build only sync_pkg
colcon build --packages-select sync_pkg

### Running Individual Nodes

#### 1. Start Raw Data Publisher

```bash
# Basic usage (1 Hz, loop enabled)
ros2 run sync_pkg publish_raw_data

# With custom parameters
ros2 run sync_pkg publish_raw_data --ros-args \
  -p publish_frequency_hz:=0.5 \
  -p loop_playback:=true
```

#### 2. Start Data Synchronization Service

```bash
# With default camera intrinsics
ros2 run sync_pkg data_synchroniser

# With custom camera intrinsics
ros2 run sync_pkg data_synchroniser --ros-args \
  -p fx:=525.0 \
  -p fy:=525.0 \
  -p cx:=319.5 \
  -p cy:=239.5
```

#### 3. Trigger Synchronization (Client)

```bash
# Using the client node
ros2 run sync_pkg data_synchroniser_client

# Using command line service call
ros2 service call /trigger_sync sync_pkg/srv/TriggerSync
```

#### 4. Run Sanity Test

```bash
ros2 run sync_pkg sanity_test
```

### Complete Pipeline Example

Open 4 terminals and run the following commands:

**Terminal 1** - Raw Data Publisher:
```bash
cd ~/TAMT-Traversability-Assessment-of-Martian-Terrain
source install/setup.bash
ros2 run sync_pkg publish_raw_data --ros-args -p publish_frequency_hz:=1.0
```

**Terminal 2** - Synchronization Service:
```bash
cd ~/TAMT-Traversability-Assessment-of-Martian-Terrain
source install/setup.bash
ros2 run sync_pkg data_synchroniser --ros-args -p fx:=525.0 -p fy:=525.0
```

**Terminal 3** - Sanity Test (Optional):
```bash
cd ~/TAMT-Traversability-Assessment-of-Martian-Terrain
source install/setup.bash
ros2 run sync_pkg sanity_test
```

**Terminal 4** - Trigger Synchronization:
```bash
cd ~/TAMT-Traversability-Assessment-of-Martian-Terrain
source install/setup.bash
ros2 service call /trigger_sync sync_pkg/srv/TriggerSync
```

### Debugging and Monitoring

#### View Active Topics

```bash
# List all topics
ros2 topic list

# Monitor raw topics
ros2 topic echo /left_image
ros2 topic echo /depth
ros2 topic echo /camera_pose

# Monitor synchronized topics
ros2 topic echo /sync_rgb
ros2 topic echo /sync_depth
ros2 topic echo /sync_pointcloud
ros2 topic echo /sync_cam_2_glob_pose
```

#### Check Topic Rates

```bash
# Check publishing rate of raw data
ros2 topic hz /left_image
ros2 topic hz /depth

# Check synchronized data rate
ros2 topic hz /sync_rgb
```

#### Inspect Services

```bash
# List services
ros2 service list

# Get service type
ros2 service type /trigger_sync

# View service interface
ros2 interface show sync_pkg/srv/TriggerSync
```

#### Visualize in RViz2

```bash
# Launch RViz2
rviz2

# Add displays:
# - Image: /sync_rgb
# - Image: /sync_depth
# - PointCloud2: /sync_pointcloud
# - TF frames
```

---

## Integration with TAMT Pipeline

The `sync_pkg` serves as the data preprocessing layer for the TAMT system:

1. **sync_pkg** → Provides synchronized multi-modal data
2. **surface_normal_estimator** → Estimates surface normals from point clouds
3. **terrain_segmentation** → Segments terrain using YOLO-based models
4. **Traversability Analysis** → Combines geometric and semantic features for navigation

---

## Troubleshooting

### Build Errors

**Problem**: CMake cannot find source files
```
CMake Error: Cannot find source file: src/input_publisher_nodes/raw_data_publisher.cpp
```

**Solution**: Verify source paths in `CMakeLists.txt` match actual directory structure.

---

### No Data Published

**Problem**: Nodes start but no data appears on topics

**Solution**:
1. Check that data files exist in `sync_pkg/data/` directory
2. Verify file naming convention matches expected format
3. Check file permissions

```bash
# Verify data directory
ls -la install/sync_pkg/share/sync_pkg/data/
```

---

### Synchronization Issues

**Problem**: Service call returns "No synchronized data available"

**Solution**:
1. Ensure raw data publisher is running and publishing data
2. Check that all three topics (/left_image, /depth, /camera_pose) are active
3. Verify timestamps are within reasonable range for approximate matching
4. Increase queue size in sync policy if messages are arriving too fast

---

### Point Cloud Not Generated

**Problem**: `/sync_pointcloud` topic is empty or has invalid data

**Solution**:
1. Verify camera intrinsic parameters are correct
2. Check depth image format and encoding
3. Ensure depth values are in valid range (not NaN or infinity)

---

## Dependencies

### ROS 2 Packages
- `rclcpp` - ROS 2 C++ client library
- `sensor_msgs` - Standard sensor message types
- `geometry_msgs` - Geometric primitives
- `message_filters` - Message synchronization
- `cv_bridge` - OpenCV-ROS bridge
- `ament_index_cpp` - Package resource lookup

### External Libraries
- OpenCV 4.x - Image processing
- Eigen3 (optional) - Linear algebra operations

---

## License

Apache License 2.0

---

## Contributing

This package is part of an academic project. For questions or collaboration, please contact the maintainer.
