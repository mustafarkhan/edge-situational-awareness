"""RGB + thermal gimbal: one detector per camera, live MAVROS telemetry, optional flight recording.

  ros2 launch esa_ros esa_dual.launch.py \
      rgb_topic:=/camera/rgb/image_raw thermal_topic:=/camera/thermal/image_raw \
      rgb_model:=models/sds_y11n_ep4.engine thermal_model:=models/hit_y11n_ep5.engine record:=true

Each camera runs in its own namespace (/rgb/..., /thermal/...) with its own model, field of view and alert
labels; both share the drone telemetry and the operator's review decisions. The web backend and TAK bridge
subscribe to /rgb/alerts and /thermal/alerts. On a Jetson Orin Nano the two models run at roughly half the
single-camera frame rate each; see docs/JETSON.md for how to measure that on your unit.

record:=true writes a rosbag2 of the camera images, telemetry, detections, alerts and status to ~/esa_bags.
Field-test sorties are recorded so every number in the test report can be recomputed later (docs/DEPLOYMENT.md).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
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


def detector(cam: str, labels: list[str], imgsz: int):
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


def generate_launch_description():
    record_topics = [
        LaunchConfiguration("rgb_topic"),
        LaunchConfiguration("thermal_topic"),
        "/mavros/global_position/global",
        "/mavros/global_position/rel_alt",
        "/mavros/global_position/compass_hdg",
        "/mavros/state",
        "/gimbal_pitch",
        "/alert_reviews",
        *[f"/{c}/{t}" for c in ("rgb", "thermal") for t in ("detections", "tracks_geo", "alerts", "status")],
    ]
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
            # one bag per sortie, named by start time (rosbag2's default name), in bag_dir
            ExecuteProcess(
                condition=IfCondition(LaunchConfiguration("record")),
                cmd=[
                    "mkdir -p",
                    LaunchConfiguration("bag_dir"),
                    "&& cd",
                    LaunchConfiguration("bag_dir"),
                    "&& ros2 bag record",
                    *[[" ", t] for t in record_topics],
                ],  # fmt: skip
                shell=True,
                output="screen",
            ),
        ]
    )
