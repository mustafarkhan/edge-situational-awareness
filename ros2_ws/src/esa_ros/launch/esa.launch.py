"""Detector node with live telemetry from MAVROS (PX4 / ArduPilot).

  ros2 launch esa_ros esa.launch.py model:=weights/sds_y11n_ep4.pt source:=0 use_telemetry:=true

MAVROS gives position, height above home and compass heading directly (remapped below). Gimbal pitch
has no single standard topic: publish it on /gimbal_pitch as std_msgs/Float64 in degrees BELOW the
horizon (90 = straight down), or leave it unpublished for a fixed camera and set pitch_deg.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arg = LaunchConfiguration
    return LaunchDescription(
        [
            DeclareLaunchArgument("model", default_value="best.engine"),
            DeclareLaunchArgument("source", default_value="0"),
            DeclareLaunchArgument("use_telemetry", default_value="false"),
            DeclareLaunchArgument("pitch_deg", default_value="90.0"),
            DeclareLaunchArgument("hfov_deg", default_value="70.0"),
            Node(
                package="esa_ros",
                executable="detector_node",
                name="detector",
                output="screen",
                parameters=[
                    {
                        "model": arg("model"),
                        "source": arg("source"),
                        "imgsz": 1280,
                        "conf": 0.25,
                        "alert_labels": ["swimmer"],
                        "use_telemetry": ParameterValue(arg("use_telemetry"), value_type=bool),
                        "pitch_deg": ParameterValue(arg("pitch_deg"), value_type=float),
                        "hfov_deg": ParameterValue(arg("hfov_deg"), value_type=float),
                    }
                ],
                remappings=[
                    ("fix", "/mavros/global_position/global"),
                    ("rel_alt", "/mavros/global_position/rel_alt"),
                    ("heading", "/mavros/global_position/compass_hdg"),
                    ("gimbal_pitch", "/gimbal_pitch"),
                ],
            ),
        ]
    )
