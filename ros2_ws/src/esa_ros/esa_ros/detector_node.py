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

Camera input: `source` (device index, file, RTSP/UDP URL) or, when `image_topic` is set, a sensor_msgs/Image
topic (a ROS camera driver, a gimbal SDK bridge, a rosbag replay). From a topic only the NEWEST frame is processed:
frames that arrive while the model is busy are dropped and counted, so latency never builds up.

Publishes
  /status      std_msgs/String (JSON, 1 Hz)  health: OK / DEGRADED (no coordinates) / NO_CAMERA, fps, dropped
                                             frames, telemetry problems, alerts so far. For the ground station
                                             and the flight log; the autopilot never depends on it.
  /detections  vision_msgs/Detection2DArray  (standard message type: other ROS tools understand it)
  /tracks_geo  std_msgs/String (JSON)        (detections + lat/lon, consumed by the web bridge)
  /alerts      std_msgs/String (JSON)

Requires `pip install -e .` of the repo root (for the shared `esa` package) in the same environment.
"""

import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, NavSatFix
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
        p("image_topic", "")  # non-empty: read frames from this sensor_msgs/Image topic instead of `source`
        p("camera_name", "rgb")  # tag on every message, for rigs with more than one camera (rgb, thermal)
        p("camera_timeout_s", 2.0)  # no frame for this long -> status NO_CAMERA
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
        self.camera_name, self.camera_timeout_s = g("camera_name"), g("camera_timeout_s")
        self.image_topic = g("image_topic")
        self._frame, self._frame_lock, self._frame_event = None, threading.Lock(), threading.Event()
        self.stats = {"frames": 0, "dropped": 0, "alerts": 0, "fps": 0.0, "last_frame": None}
        if self.image_topic:
            self.create_subscription(Image, self.image_topic, self._on_image, qos_profile_sensor_data)
        self.pub_status = self.create_publisher(String, "status", 10)
        self.create_timer(1.0, self._publish_status)
        src = f"topic {self.image_topic}" if self.image_topic else self.source
        self.get_logger().info(f"camera={self.camera_name} model={g('model')} source={src}")

    def _on_review(self, m: String):
        try:
            r = json.loads(m.data)
        except json.JSONDecodeError:
            return
        if r.get("decision") == "reject":
            self.alerts.reject(r.get("lat"), r.get("lon"), r.get("label", ""))
            self.get_logger().info(f"operator rejected track {r.get('track_id')}: suppressing alerts near it")

    def _on_image(self, m: Image):
        frame = image_to_bgr(m)
        if frame is None:
            self.get_logger().warn(f"unsupported image encoding {m.encoding!r}", throttle_duration_sec=10)
            return
        with self._frame_lock:
            if self._frame is not None:
                self.stats["dropped"] += 1  # the model had not taken the previous frame yet
            self._frame = frame
        self._frame_event.set()

    def _frames_from_topic(self):
        while rclpy.ok():
            if not self._frame_event.wait(timeout=0.5):
                continue
            with self._frame_lock:
                frame, self._frame = self._frame, None
                self._frame_event.clear()
            if frame is not None:
                yield frame

    def status(self) -> dict:
        now = time.monotonic()
        last = self.stats["last_frame"]
        problems = []
        if last is None or now - last > self.camera_timeout_s:
            state = "NO_CAMERA"
            problems.append("no camera frames" if last is None else f"no frame for {now - last:.1f} s")
        else:
            state = "OK"
        if self.telemetry is not None:
            tel = self.telemetry.problems(self._now())
            if tel:
                problems += tel
                state = "DEGRADED" if state == "OK" else state
        return {
            "camera": self.camera_name,
            "state": state,
            "problems": problems,
            "fps": self.stats["fps"],
            "frames": self.stats["frames"],
            "dropped_frames": self.stats["dropped"],
            "alerts": self.stats["alerts"],
            "telemetry": "live" if self.telemetry is not None else "fixed parameters",
            "t": time.time(),
        }

    def _publish_status(self):
        self.pub_status.publish(String(data=json.dumps(self.status())))

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
        kw = dict(persist=True, tracker=self.tracker, imgsz=self.imgsz, conf=self.conf, verbose=False)
        if self.image_topic:
            results = (self.model.track(f, **kw)[0] for f in self._frames_from_topic())
        else:
            results = self.model.track(self.source, stream=True, **kw)
        t_prev = time.perf_counter()
        for i, r in enumerate(results):
            if not rclpy.ok():
                break
            now = time.perf_counter()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            self.stats.update(frames=i + 1, fps=round(fps, 1), last_frame=time.monotonic())
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
                            "camera": self.camera_name,
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
            self.stats["alerts"] += len(new_alerts)
            for a in new_alerts:
                a["camera"] = self.camera_name
                self.pub_alert.publish(String(data=json.dumps(a)))
                self.get_logger().warn(a["message"])


def image_to_bgr(m: Image):
    """sensor_msgs/Image -> HxWx3 uint8 BGR array without cv_bridge (one dependency fewer on the Jetson).

    Thermal cameras often publish 16-bit counts (mono16): those are stretched per frame to 8 bits, the same
    contrast stretch as the HIT-UAV images the thermal model was trained on.
    """
    enc = m.encoding.lower()
    buf = np.frombuffer(m.data, dtype=np.uint16 if enc in ("mono16", "16uc1") else np.uint8)
    if enc in ("bgr8", "rgb8"):
        a = buf.reshape(m.height, m.step)[:, : m.width * 3].reshape(m.height, m.width, 3)
        return a[:, :, ::-1].copy() if enc == "rgb8" else a.copy()
    if enc in ("mono8", "8uc1"):
        a = buf.reshape(m.height, m.step)[:, : m.width]
    elif enc in ("mono16", "16uc1"):
        a = buf.reshape(m.height, m.step // 2)[:, : m.width].astype(np.float32)
        lo, hi = np.percentile(a, (1, 99))
        a = np.clip((a - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
    else:
        return None
    return np.repeat(a[:, :, None], 3, axis=2)


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
