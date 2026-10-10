#!/usr/bin/env bash
# End-to-end check of the ROS 2 detector node (used in CI, also runnable on the Jetson).
# Runs the node on Ultralytics' two sample images (people + a bus), with an alert on "person"
# after 1 frame, and asserts that each output topic received at least one message.
set -eo pipefail
source /opt/ros/humble/setup.bash
source "$(dirname "$0")/install/setup.bash"
OUT=$(mktemp -d)
SAMPLES=$(python3 -c "import ultralytics, os; print(os.path.join(os.path.dirname(ultralytics.__file__), 'assets'))")
MODEL=${MODEL:-yolo11n.pt}
# 30 frames (the two samples, repeated) so publishing lasts long enough for subscribers to connect
mkdir -p "$OUT/frames"
for i in $(seq -w 1 15); do
  cp "$SAMPLES/bus.jpg" "$OUT/frames/${i}a.jpg"; cp "$SAMPLES/zidane.jpg" "$OUT/frames/${i}b.jpg"
done
echo "frames: $(ls "$OUT/frames" | wc -l)  model: $MODEL"

for t in detections tracks_geo alerts; do
  timeout 240 ros2 topic echo --once "/$t" > "$OUT/$t.txt" 2>&1 &
done
sleep 3
timeout 300 ros2 run esa_ros detector_node --ros-args \
  -p model:="$MODEL" -p source:="$OUT/frames" -p imgsz:=640 -p conf:=0.25 \
  -p alert_labels:="['person']" -p alert_min_frames:=1 \
  -p lat:=60.15 -p lon:=24.95 -p alt_m:=60.0 -p hfov_deg:=70.0 &
NODE=$!
wait $NODE || true
wait || true

fail=0
for t in detections tracks_geo alerts; do
  if grep -qE "detections|data|class_id" "$OUT/$t.txt"; then
    echo "OK   /$t"; head -c 400 "$OUT/$t.txt"; echo
  else
    echo "FAIL /$t received nothing"; cat "$OUT/$t.txt"; fail=1
  fi
done
exit $fail
