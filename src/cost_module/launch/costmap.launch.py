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
    
    # 1. Static transform publisher (WARN level to suppress INFO)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=[
            '1.709320068', '-4.636882782', '0.800000012',
            '-0.174203563', '0.681054306', '0.70024741', '-0.124385352',
            'map', 'zed2i_left_camera_optical_frame',
            '--ros-args', '--log-level', 'WARN'
        ],
        output='screen'
    )

    # 2. Test publisher sync data (no delay, WARN level)
    test_publisher = Node(
        package='sync_pkg',
        executable='test_publisher_sync_data',
        name='test_publisher_sync_data',
        arguments=['--ros-args', '--log-level', 'WARN'],
        output='screen'
    )

    # 3. Publisher all data (delayed 1 second, WARN level)
    publisher_all = Node(
        package='sync_pkg',
        executable='publish_raw_data',
        name='raw_data_publisher',
        arguments=['--ros-args', '--log-level', 'WARN'],
        output='screen'
    )
    delayed_publisher_all = TimerAction(
        period=1.0,
        actions=[publisher_all]
    )

    # 4. Surface normal estimator (delayed 1 second, WARN level)
    surface_normal = Node(
        package='surface_normal_estimator',
        executable='surface_normal_estimator',
        name='surface_normal_estimator',
        arguments=['--ros-args', '--log-level', 'WARN'],
        output='screen'
    )
    delayed_surface_normal = TimerAction(
        period=1.0,
        actions=[surface_normal]
    )

    # 5. Costmap SNE (delayed 1 second, INFO level - keep all logs)
    costmap_sne = Node(
        package='cost_module',
        executable='costmap_sne',
        name='costmap_sne',
        parameters=[config_file],
        output='screen'
    )
    delayed_costmap_sne = TimerAction(
        period=1.0,
        actions=[costmap_sne]
    )

    return LaunchDescription([
        static_tf,
        test_publisher,
        delayed_publisher_all,
        delayed_surface_normal,
        delayed_costmap_sne
    ])
