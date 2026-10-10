#!/usr/bin/env bash
# End-to-end check of the ROS 2 detector node (used in CI, also runnable on the Jetson).
# Runs the node on Ultralytics' two sample images (people + a bus), repeated to 30 frames, with an
# alert on "person" after 1 frame, and asserts that each output topic received at least one message.
# Results are also emitted as GitHub Actions annotations, so they are readable without the raw log.
set -o pipefail
source /opt/ros/humble/setup.bash
source "$(dirname "$0")/install/setup.bash"
OUT=$(mktemp -d)
SAMPLES=$(python3 -c "import ultralytics, os; print(os.path.join(os.path.dirname(ultralytics.__file__), 'assets'))")
MODEL=${MODEL:-yolo11n.pt}
mkdir -p "$OUT/frames"
for i in $(seq -w 1 15); do
  cp "$SAMPLES/bus.jpg" "$OUT/frames/${i}a.jpg"; cp "$SAMPLES/zidane.jpg" "$OUT/frames/${i}b.jpg"
done
echo "frames: $(ls "$OUT/frames" | wc -l)  model: $MODEL"

annotate() {  # level title file: one annotation with the file's first 60 lines
  local body
  body=$(head -n 60 "$3" | sed -e 's/%/%25/g' | sed -e ':a;N;$!ba;s/\n/%0A/g' | tr -d '\r')
  echo "::$1 title=$2::${body:-<empty>}"
}

# Message types are given explicitly: without them, `ros2 topic echo` exits at once if the topic
# does not exist yet, and these subscribers start before the node.
declare -A TYPES=([detections]=vision_msgs/msg/Detection2DArray [tracks_geo]=std_msgs/msg/String [alerts]=std_msgs/msg/String)
for t in "${!TYPES[@]}"; do
  timeout 300 ros2 topic echo --once "/$t" "${TYPES[$t]}" > "$OUT/$t.txt" 2>&1 &
done
sleep 3
timeout 300 ros2 run esa_ros detector_node --ros-args \
  -p model:="$MODEL" -p source:="$OUT/frames" -p imgsz:=640 -p conf:=0.25 \
  -p alert_labels:="['person']" -p alert_min_frames:=1 \
  -p lat:=60.15 -p lon:=24.95 -p alt_m:=60.0 -p hfov_deg:=70.0 > "$OUT/node.log" 2>&1
echo "node exit code: $?"
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
annotate "$([ $fail = 0 ] && echo notice || echo error)" "node log (last 40 lines)" "$OUT/node_tail.log"
cat "$OUT/node.log"
exit $fail
