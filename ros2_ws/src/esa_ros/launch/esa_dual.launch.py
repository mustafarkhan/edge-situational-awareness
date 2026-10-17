"""RGB + thermal gimbal: one detector per camera, live MAVROS telemetry, optional flight recording.

  ros2 launch esa_ros esa_dual.launch.py \
      rgb_topic:=/camera/rgb/image_raw thermal_topic:=/camera/thermal/image_raw \
      rgb_model:=models/sds_y11n_ep4.engine thermal_model:=models/hit_y11n_ep5.engine record:=true

Each camera runs in its own namespace (/rgb/..., /thermal/...) with its own model, field of view and alert
labels; both share the drone telemetry and the operator's review decisions. The ground station takes both cameras:

  python web/backend/server.py --ros --camera-topics /rgb/tracks_geo /thermal/tracks_geo

and shows a person seen by both cameras as one alert (esa.fusion). Tested end to end in CI (ros2_ws/test_dual.sh).

record:=true writes one rosbag2 per sortie (bag_dir/sortie_<UTC time>) with the camera images, telemetry,
detections, alerts, status and operator decisions, so every number in a field-test report can be recomputed
later (docs/DEPLOYMENT.md).
"""

import time
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

MAVROS = [
    ("fix", "/mavros/global_position/global"),
    ("rel_alt", "/mavros/global_position/rel_alt"),
    ("heading", "/mavros/global_position/compass_hdg"),
    ("gimbal_pitch", "/gimbal_pitch"),
    ("alert_reviews", "/alert_reviews"),
]
RECORDED = [
    "/mavros/global_position/global",
    "/mavros/global_position/rel_alt",
    "/mavros/global_position/compass_hdg",
    "/mavros/state",
    "/gimbal_pitch",
    "/alert_reviews",
    *[f"/{cam}/{topic}" for cam in ("rgb", "thermal") for topic in ("detections", "tracks_geo", "alerts", "status")],
]


def detector(cam: str, labels: list[str], imgsz: int) -> Node:
    arg = LaunchConfiguration
    return Node(
        package="esa_ros",
        executable="detector_node",
        name="detector",
        namespace=cam,
        output="screen",
        parameters=[
            {
                "model": arg(f"{cam}_model"),
                "image_topic": arg(f"{cam}_topic"),
                "camera_name": cam,
                "imgsz": imgsz,
                "conf": 0.25,
                "alert_labels": labels,
                "use_telemetry": True,
                "hfov_deg": ParameterValue(arg(f"{cam}_hfov_deg"), value_type=float),
                "pitch_deg": ParameterValue(arg("pitch_deg"), value_type=float),
            }
        ],
        remappings=MAVROS,
    )


def recorder(context, *_):
    """ros2 bag record into bag_dir/sortie_<UTC time>; arguments are resolved when the launch starts."""
    bag_dir = Path(LaunchConfiguration("bag_dir").perform(context)).expanduser()
    bag_dir.mkdir(parents=True, exist_ok=True)
    out = bag_dir / time.strftime("sortie_%Y%m%dT%H%M%SZ", time.gmtime())
    cameras = [LaunchConfiguration(t).perform(context) for t in ("rgb_topic", "thermal_topic")]
    return [
        ExecuteProcess(
            cmd=["ros2", "bag", "record", "-o", str(out), *cameras, *RECORDED],
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("rgb_topic", default_value="/camera/rgb/image_raw"),
            DeclareLaunchArgument("thermal_topic", default_value="/camera/thermal/image_raw"),
            DeclareLaunchArgument("rgb_model", default_value="models/sds_y11n_ep4.pt"),
            DeclareLaunchArgument("thermal_model", default_value="models/hit_y11n_ep5.pt"),
            DeclareLaunchArgument("rgb_hfov_deg", default_value="65.5"),
            DeclareLaunchArgument("thermal_hfov_deg", default_value="25.0"),
            DeclareLaunchArgument("pitch_deg", default_value="90.0"),
            DeclareLaunchArgument("record", default_value="false"),
            DeclareLaunchArgument("bag_dir", default_value="~/esa_bags"),
            detector("rgb", ["swimmer", "person", "pedestrian"], 1280),
            detector("thermal", ["person", "swimmer"], 640),
            OpaqueFunction(function=recorder, condition=IfCondition(LaunchConfiguration("record"))),
        ]
    )
