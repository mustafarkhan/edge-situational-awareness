from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model = LaunchConfiguration("model")
    source = LaunchConfiguration("source")
    return LaunchDescription([
        DeclareLaunchArgument("model", default_value="best.engine"),
        DeclareLaunchArgument("source", default_value="0"),
        Node(package="esa_ros", executable="detector_node", name="detector", output="screen",
             parameters=[{"model": model, "source": source, "imgsz": 640,
                          "alert_labels": ["swimmer"], "alt_m": 60.0, "hfov_deg": 70.0}]),
    ])
