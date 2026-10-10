"""ROS 2 node: camera/video -> TensorRT YOLO -> tracker -> topics.

Publishes
  /detections  vision_msgs/Detection2DArray  (standard message type: other ROS tools understand it)
  /tracks_geo  std_msgs/String (JSON)        (detections + lat/lon, consumed by the web bridge)
  /alerts      std_msgs/String (JSON)

Requires `pip install -e .` of the repo root (for the shared `esa` package) in the same environment.
"""

import json
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from esa.alerts import AlertManager
from esa.geo import CameraGeo, pixel_to_latlon


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector")
        p = self.declare_parameter
        p("model", "best.engine")
        p("source", "0")
        p("imgsz", 640)
        p("conf", 0.25)
        p("tracker", "bytetrack.yaml")
        p("lat", 60.15)
        p("lon", 24.95)
        p("alt_m", 60.0)
        p("hfov_deg", 70.0)
        p("heading_deg", 0.0)
        p("alert_labels", ["swimmer"])
        p("alert_min_frames", 5)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.model = YOLO(g("model"), task="detect")
        self.source = int(g("source")) if str(g("source")).isdigit() else g("source")
        self.imgsz, self.conf, self.tracker = g("imgsz"), g("conf"), g("tracker")
        self.cam = CameraGeo(g("lat"), g("lon"), g("alt_m"), g("hfov_deg"), g("heading_deg"))
        self.alerts = AlertManager(set(g("alert_labels")), g("alert_min_frames"))
        self.pub_det = self.create_publisher(Detection2DArray, "detections", 10)
        self.pub_geo = self.create_publisher(String, "tracks_geo", 10)
        self.pub_alert = self.create_publisher(String, "alerts", 10)
        self.get_logger().info(f"model={g('model')} source={self.source}")

    def run(self):
        t_prev = time.perf_counter()
        for i, r in enumerate(
            self.model.track(
                self.source,
                stream=True,
                persist=True,
                tracker=self.tracker,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
            )
        ):
            if not rclpy.ok():
                break
            now = time.perf_counter()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            stamp = self.get_clock().now().to_msg()
            h, w = r.orig_shape
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)

            arr = Detection2DArray()
            arr.header.stamp = stamp
            arr.header.frame_id = "camera"
            dets = []
            for (x1, y1, x2, y2), c, s, tid in zip(
                r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), ids
            ):
                d = Detection2D()
                d.header = arr.header
                d.id = "" if tid is None else str(tid)
                d.bbox.center.position.x = (x1 + x2) / 2
                d.bbox.center.position.y = (y1 + y2) / 2
                d.bbox.size_x, d.bbox.size_y = x2 - x1, y2 - y1
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = r.names[c]
                hyp.hypothesis.score = float(s)
                d.results.append(hyp)
                arr.detections.append(d)
                lat, lon = pixel_to_latlon((x1 + x2) / 2, (y1 + y2) / 2, w, h, self.cam)
                dets.append(
                    {
                        "track_id": tid,
                        "label": r.names[c],
                        "conf": round(s, 3),
                        "lat": round(lat, 7),
                        "lon": round(lon, 7),
                    }
                )

            self.pub_det.publish(arr)
            new_alerts = self.alerts.update(dets)
            self.pub_geo.publish(
                String(
                    data=json.dumps(
                        {
                            "frame": i,
                            "t": time.time(),
                            "fps": round(fps, 1),
                            "drone": {"lat": self.cam.lat, "lon": self.cam.lon},
                            "detections": dets,
                            "alerts": new_alerts,
                        }
                    )
                )
            )
            for a in new_alerts:
                self.pub_alert.publish(String(data=json.dumps(a)))
                self.get_logger().warn(a["message"])


def main():
    rclpy.init()
    node = DetectorNode()
    # Inference runs in the main thread; ROS callbacks (parameters, services) are serviced by an
    # executor in a background thread. Shut down in order -- stop the executor, join its thread,
    # then destroy the node -- or the C++ layer aborts at exit ("terminate called without an
    # active exception"), which CI caught in the first end-to-end run.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        spin.join(timeout=5)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
