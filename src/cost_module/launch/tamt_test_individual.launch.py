from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('cost_module')
    
    # Path to your config file
    config_file = os.path.join(pkg_share, 'config', 'params.yaml')


    
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
    cost_module = Node(
        package='cost_module',
        executable='cost_module',
        name='costmap_module',
        namespace='tamt',
        parameters=[config_file],
        output='screen'
    )
    delayed_cost_module = TimerAction(
        period=5.0,
        actions=[cost_module]
    )

    # 4. Cost module tester
    cost_module_tester = Node(
        package='cost_module',
        executable='test_costmodule_all.py',
        name='cost_module_tester',
        namespace='tamt',
        parameters=[config_file],
        arguments=['--ros-args', '--log-level', 'INFO'],
        output='screen'
    )
    delayed_cost_module_tester = TimerAction(
        period=10.0,
        actions=[cost_module_tester]
    )

    return LaunchDescription([
        terrain_segmentation,
        surface_normal,
        delayed_cost_module,
        delayed_cost_module_tester
    ])
