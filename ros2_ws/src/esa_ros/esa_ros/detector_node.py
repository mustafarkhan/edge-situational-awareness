"""ROS 2 node: camera/video -> TensorRT YOLO -> tracker -> topics.

Subscribes (only with use_telemetry:=true; otherwise position and attitude are fixed parameters)
  fix           sensor_msgs/NavSatFix   drone latitude/longitude (e.g. remap from /mavros/global_position/global)
  rel_alt       std_msgs/Float64        height above ground/sea in metres (/mavros/global_position/rel_alt)
  heading       std_msgs/Float64        camera heading, degrees from north (/mavros/global_position/compass_hdg
                                        for a camera fixed to the airframe)
  gimbal_pitch  std_msgs/Float64        degrees BELOW the horizon, 90 = straight down (DJI/MAVLink report
                                        -90 for straight down: negate before publishing)
  heading and gimbal_pitch are optional: until a message arrives, the heading_deg / pitch_deg parameters
  are used. Any value older than telemetry_timeout_s means no coordinates are published (null), rather
  than coordinates computed from stale telemetry.

Also subscribes to
  alert_reviews std_msgs/String (JSON)     operator decisions from the ground station (web/backend/server.py);
                                           a "reject" suppresses new alerts near that spot (esa.alerts)

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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64, String
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from esa.alerts import AlertManager
from esa.geo import CameraGeo, pixel_to_latlon
from esa.telemetry import TelemetryState


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
        p("pitch_deg", 90.0)  # camera tilt below the horizon; 90 = straight down
        p("use_telemetry", False)
        p("telemetry_timeout_s", 1.0)
        p("alert_labels", ["swimmer"])
        p("alert_min_frames", 5)
        p("alert_dedupe_m", 3.0)  # same spot, new track ID -> not a new alert (benchmarks/alert_load.py)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.model = YOLO(g("model"), task="detect")
        self.source = int(g("source")) if str(g("source")).isdigit() else g("source")
        self.imgsz, self.conf, self.tracker = g("imgsz"), g("conf"), g("tracker")
        self.cam = CameraGeo(g("lat"), g("lon"), g("alt_m"), g("hfov_deg"), g("heading_deg"), g("pitch_deg"))
        self.telemetry = None
        if g("use_telemetry"):
            self.telemetry = TelemetryState(
                hfov_deg=g("hfov_deg"),
                timeout_s=g("telemetry_timeout_s"),
                defaults={"heading_deg": g("heading_deg"), "pitch_deg": g("pitch_deg")},
            )
            sub = self.create_subscription
            # best-effort "sensor data" QoS: MAVROS publishes these best-effort, and a reliable subscriber
            # would silently receive nothing from a best-effort publisher (this one receives from both)
            qos = qos_profile_sensor_data
            sub(NavSatFix, "fix", self._on_fix, qos)
            sub(Float64, "rel_alt", lambda m: self._tel(alt_m=m.data), qos)
            sub(Float64, "heading", lambda m: self._tel(heading_deg=m.data), qos)
            sub(Float64, "gimbal_pitch", lambda m: self._tel(pitch_deg=m.data), qos)
            self._last_warn = 0.0
        self.alerts = AlertManager(set(g("alert_labels")), g("alert_min_frames"), dedupe_radius_m=g("alert_dedupe_m"))
        self.create_subscription(String, "alert_reviews", self._on_review, 10)
        self.pub_det = self.create_publisher(Detection2DArray, "detections", 10)
        self.pub_geo = self.create_publisher(String, "tracks_geo", 10)
        # Alerts are latched (transient-local, last 10 kept): a ground station that connects or
        # reconnects late still receives recent alerts. Found when the end-to-end test missed alerts
        # fired before its subscriber had connected.
        alert_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub_alert = self.create_publisher(String, "alerts", alert_qos)
        self.get_logger().info(f"model={g('model')} source={self.source}")

    def _on_review(self, m: String):
        try:
            r = json.loads(m.data)
        except json.JSONDecodeError:
            return
        if r.get("decision") == "reject":
            self.alerts.reject(r.get("lat"), r.get("lon"), r.get("label", ""))
            self.get_logger().info(f"operator rejected track {r.get('track_id')}: suppressing alerts near it")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _tel(self, **values):
        self.telemetry.update(self._now(), **values)

    def _on_fix(self, m: NavSatFix):
        if m.status.status >= 0:  # -1 = no fix
            self._tel(lat=m.latitude, lon=m.longitude)

    def current_camera(self) -> CameraGeo | None:
        """Camera model for this frame: fixed parameters, or the latest fresh telemetry (None if stale)."""
        if self.telemetry is None:
            return self.cam
        now = self._now()
        cam = self.telemetry.camera(now)
        if cam is None and now - self._last_warn > 5.0:
            self._last_warn = now
            self.get_logger().warn("no coordinates: " + ", ".join(self.telemetry.problems(now)))
        return cam

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
            cam = self.current_camera()
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
                # None above the horizon, or when telemetry is missing or stale
                pos = pixel_to_latlon((x1 + x2) / 2, (y1 + y2) / 2, w, h, cam) if cam else None
                lat, lon = pos if pos else (None, None)
                dets.append(
                    {
                        "track_id": tid,
                        "label": r.names[c],
                        "conf": round(s, 3),
                        "lat": round(lat, 7) if lat is not None else None,
                        "lon": round(lon, 7) if lon is not None else None,
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
                            "drone": (
                                {
                                    "lat": cam.lat,
                                    "lon": cam.lon,
                                    "alt_m": cam.alt_m,
                                    "heading_deg": cam.heading_deg,
                                    "pitch_deg": cam.pitch_deg,
                                }
                                if cam
                                else None
                            ),
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
