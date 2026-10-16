"""Publish a folder of images as sensor_msgs/Image (bgr8), like a camera driver. For test_node.sh.

python3 pub_frames.py <folder> --topic /camera/image_raw --hz 10 --seconds 40
"""

import argparse
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--topic", default="/camera/image_raw")
ap.add_argument("--hz", type=float, default=10.0)
ap.add_argument("--seconds", type=float, default=40.0)
a = ap.parse_args()

frames = [cv2.imread(str(f)) for f in sorted(Path(a.folder).glob("*.jpg"))]
rclpy.init()
node = rclpy.create_node("frame_publisher")
pub = node.create_publisher(Image, a.topic, qos_profile_sensor_data)
t_end, i = time.time() + a.seconds, 0
while time.time() < t_end and rclpy.ok():
    f = frames[i % len(frames)]
    m = Image(height=f.shape[0], width=f.shape[1], encoding="bgr8", step=f.shape[1] * 3, data=f.tobytes())
    m.header.stamp = node.get_clock().now().to_msg()
    m.header.frame_id = "camera"
    pub.publish(m)
    i += 1
    time.sleep(1.0 / a.hz)
print(f"published {i} frames")
node.destroy_node()
rclpy.shutdown()
