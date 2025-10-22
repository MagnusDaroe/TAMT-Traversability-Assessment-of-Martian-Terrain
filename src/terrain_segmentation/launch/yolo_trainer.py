#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='seg_train.yaml',
        description='Name of the configuration file'
    )
    
    # Get the package share directory
    package_share = get_package_share_directory('terrain_segmentation')
    config_path = os.path.join(package_share, 'config', 'seg_train.yaml')
    
    # Alternative: use absolute path if config is not in package
    # config_path = '/path/to/your/seg_train.yaml'
    
    # Create the node
    yolo_trainer_node = Node(
        package='terrain_segmentation',
        executable='train_model.py',
        name='yolo_trainer_node',
        output='screen',
        parameters=[{
            'config_file': config_path
        }],
        emulate_tty=True,
    )
    
    return LaunchDescription([
        config_file_arg,
        yolo_trainer_node
    ])