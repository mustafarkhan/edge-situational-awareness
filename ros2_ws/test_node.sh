#!/usr/bin/env bash
# End-to-end check of the ROS 2 detector node (used in CI, also runnable on the Jetson).
# Runs the node on Ultralytics' two sample images (people + a bus), repeated to 30 frames, with an
# alert on "person" after 1 frame, and asserts that each output topic received at least one message.
# Results are also emitted as GitHub Actions annotations, so they are readable without the raw log.
set -o pipefail
source /opt/ros/humble/setup.bash
source "$(dirname "$0")/install/setup.bash"
OUT=$(mktemp -d)
MODEL=${MODEL:-yolo11n.pt}
PITCH=${PITCH:-90.0}  # gimbal pitch below the horizon: 90 = straight down; e.g. 45 tests the tilt-aware geolocation
# TELEMETRY=1: position, altitude and gimbal come from live topics (as from MAVROS) instead of parameters.
# The fake drone is put over Tampere (61.4978 N) so coordinates can only come from the topics.
TELEMETRY=${TELEMETRY:-0}
if [ "$TELEMETRY" = 1 ]; then LAT_RE='"lat": 61\.49[0-9]{5}'; MODE="live telemetry"; else LAT_RE='"lat": 60\.1[45][0-9]{4}'; MODE="pitch $PITCH"; fi
mkdir -p "$OUT/frames"
for img in bus zidane; do  # Ultralytics' standard sample images (people + vehicles)
  curl -fsSL -o "$OUT/$img.jpg" "https://ultralytics.com/images/$img.jpg" ||
    curl -fsSL -o "$OUT/$img.jpg" "https://github.com/ultralytics/assets/releases/download/v0.0.0/$img.jpg"
done
for i in $(seq -w 1 15); do
  cp "$OUT/bus.jpg" "$OUT/frames/${i}a.jpg" && cp "$OUT/zidane.jpg" "$OUT/frames/${i}b.jpg" || exit 1
done
echo "frames: $(ls "$OUT/frames" | wc -l)  model: $MODEL  mode: $MODE"
[ "$(ls "$OUT/frames" | wc -l)" -eq 30 ] || { echo "::error title=setup::sample images missing"; exit 1; }

annotate() {  # level title file: one annotation with the file's first 60 lines
  local body
  body=$(head -n 60 "$3" | sed -e 's/%/%25/g' | sed -e ':a;N;$!ba;s/\n/%0A/g' | tr -d '\r')
  echo "::$1 title=$2::${body:-<empty>}"
}

# Message types are given explicitly: without them, `ros2 topic echo` exits at once if the topic
# does not exist yet, and these subscribers start before the node.
declare -A TYPES=([detections]=vision_msgs/msg/Detection2DArray [tracks_geo]=std_msgs/msg/String [alerts]=std_msgs/msg/String)
# --full-length: echo otherwise cuts strings at 128 characters, hiding the per-object coordinates.
# /tracks_geo is recorded for the whole run (not just the first message): with live telemetry the first
# frames may legitimately have no coordinates yet.
timeout 300 ros2 topic echo --once --full-length /detections "${TYPES[detections]}" > "$OUT/detections.txt" 2>&1 &
timeout 300 ros2 topic echo --full-length /tracks_geo "${TYPES[tracks_geo]}" > "$OUT/tracks_geo.txt" 2>&1 &
# alerts are latched (transient local): subscribe with matching QoS, so alerts published before
# this subscriber finished connecting are still delivered, as they would be to a late ground station
timeout 300 ros2 topic echo --once --qos-durability transient_local --qos-reliability reliable \
  /alerts std_msgs/msg/String > "$OUT/alerts.txt" 2>&1 &
EXTRA=()
if [ "$TELEMETRY" = 1 ]; then
  timeout 300 ros2 topic pub -r 10 /fix sensor_msgs/msg/NavSatFix \
    "{latitude: 61.4978, longitude: 23.761, altitude: 120.0, status: {status: 0}}" > /dev/null 2>&1 &
  timeout 300 ros2 topic pub -r 10 /rel_alt std_msgs/msg/Float64 "{data: 40.0}" > /dev/null 2>&1 &
  timeout 300 ros2 topic pub -r 10 /gimbal_pitch std_msgs/msg/Float64 "{data: 60.0}" > /dev/null 2>&1 &
  EXTRA=(-p use_telemetry:=true)
fi
sleep 3
timeout 300 ros2 run esa_ros detector_node --ros-args \
  -p model:="$MODEL" -p source:="$OUT/frames" -p imgsz:=640 -p conf:=0.25 \
  -p alert_labels:="['person']" -p alert_min_frames:=1 \
  -p lat:=60.15 -p lon:=24.95 -p alt_m:=60.0 -p hfov_deg:=70.0 -p pitch_deg:="$PITCH" "${EXTRA[@]}" > "$OUT/node.log" 2>&1
NODE_RC=$?
echo "node exit code: $NODE_RC"
sleep 2
for job in $(jobs -p); do kill "$job" 2>/dev/null; done
wait 2>/dev/null

tail -n 40 "$OUT/node.log" > "$OUT/node_tail.log"
fail=0
for t in detections tracks_geo alerts; do
  if grep -qE "detections:|data:" "$OUT/$t.txt"; then
    echo "OK   /$t"; annotate notice "/$t received" "$OUT/$t.txt"
  else
    echo "FAIL /$t received nothing"; annotate error "/$t received nothing" "$OUT/$t.txt"; fail=1
  fi
done
# geolocation must produce real coordinates near the expected drone position, not only null positions
# (an object's latitude has 7 decimals; the drone's own has fewer, so it cannot match)
if grep -qE "$LAT_RE" "$OUT/tracks_geo.txt"; then
  echo "OK   /tracks_geo has geolocated objects ($MODE)"
  grep -oE "$LAT_RE[^}]*" "$OUT/tracks_geo.txt" | head -n 3 > "$OUT/geo_sample.txt"
  annotate notice "geolocated objects ($MODE)" "$OUT/geo_sample.txt"
else
  echo "FAIL /tracks_geo has no geolocated objects ($MODE)"
  annotate error "no geolocated objects ($MODE)" "$OUT/tracks_geo.txt"; fail=1
fi
if [ "$NODE_RC" != 0 ] || grep -q "terminate called\|Aborted" "$OUT/node.log"; then
  echo "FAIL node did not exit cleanly (code $NODE_RC)"; fail=1
fi
annotate "$([ $fail = 0 ] && echo notice || echo error)" "node log (last 40 lines) - $MODE - exit code $NODE_RC" "$OUT/node_tail.log"
cat "$OUT/node.log"
exit $fail
