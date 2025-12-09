from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('cost_module')
    
    # Path to your config file
    config_file = os.path.join(pkg_share, 'config', 'initial_params.yaml')
    
    # 1. Segmentation node
    terrain_segmentation = Node(
        package='terrain_segmentation',
        executable='segment.py',
        name='terrain_segmentation_node',
        namespace='tamt',
        parameters=[config_file],
        arguments=['--ros-args', '--log-level', 'INFO'],
        output='screen'
    )
   
    # 2. Surface normal estimator (delayed 1 second, WARN level)
    surface_normal = Node(
        package='surface_normal_estimator',
        executable='surface_normal_estimator',
        name='surface_normal_estimator',
        namespace='tamt',
        parameters=[config_file],
        arguments=['--ros-args', '--log-level', 'WARN'],
        output='screen'
    )

    # 3. Costmap SNE (delayed 1 second, INFO level - keep all logs)
    costmap_sne = Node(
        package='cost_module',
        executable='costmap_sne',
        name='costmap_module',
        namespace='tamt',
        parameters=[config_file],
        output='screen'
    )
    delayed_costmap_sne = TimerAction(
        period=5.0,
        actions=[costmap_sne]
    )

    # 4. Data synchroniser
    data_synchroniser = Node(
        package='sync_pkg',
        executable='data_synchroniser',
        name='data_synchroniser',
        namespace='tamt',
        parameters=[config_file],
        arguments=['--ros-args', '--log-level', 'WARN'],
        output='screen'
    )
    delayed_synchronizer = TimerAction(
        period=7.0,
        actions=[data_synchroniser]
    )

    # 5. Publisher all data
    publisher_all = Node(
        package='sync_pkg',
        executable='publish_updated_raw_data',
        name='raw_data_publisher',
        parameters=[config_file],
        arguments=['--ros-args', '--log-level', 'WARN'],
        output='screen'
    )
    delayed_publisher_all = TimerAction(
        period=7.0,
        actions=[publisher_all]
    )

    return LaunchDescription([
        terrain_segmentation,
        surface_normal,
        delayed_costmap_sne,
        delayed_synchronizer,
        delayed_publisher_all
    ])
