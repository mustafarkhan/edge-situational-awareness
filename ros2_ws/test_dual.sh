#!/usr/bin/env bash
# End to end, the RGB + thermal setup as it runs on the drone and the ground station (used in CI):
#   two camera topics (the same sample images, as from a gimbal's two sensors) + MAVROS-style telemetry
#   -> esa_dual.launch.py (one detector per camera)
#   -> web/backend/server.py --ros --camera-topics /rgb/tracks_geo /thermal/tracks_geo
#   -> an operator client, which must see both cameras, camera-scoped IDs, an alert, and one person seen by both
#      cameras reported as one alert.
# Both detectors get the same field of view here, so the same person maps to the same spot from both cameras.
set -o pipefail
source /opt/ros/humble/setup.bash
source "$(dirname "$0")/install/setup.bash"
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mktemp -d)
MODEL=${MODEL:-yolo11n.pt}
TOKEN=$(python3 -c "from esa.access import new_token; print(new_token())")
mkdir -p "$OUT/frames"
for img in bus zidane; do
  curl -fsSL -o "$OUT/$img.jpg" "https://ultralytics.com/images/$img.jpg" ||
    curl -fsSL -o "$OUT/$img.jpg" "https://github.com/ultralytics/assets/releases/download/v0.0.0/$img.jpg"
done
for i in $(seq -w 1 15); do cp "$OUT/bus.jpg" "$OUT/frames/${i}a.jpg"; cp "$OUT/zidane.jpg" "$OUT/frames/${i}b.jpg"; done
ros2 daemon stop > /dev/null 2>&1 || true

pids=()
start() { "$@" & pids+=($!); }
start timeout 300 ros2 topic pub -r 10 /mavros/global_position/global sensor_msgs/msg/NavSatFix \
  "{latitude: 61.4978, longitude: 23.761, altitude: 120.0, status: {status: 0}}" > /dev/null 2>&1
start timeout 300 ros2 topic pub -r 10 /mavros/global_position/rel_alt std_msgs/msg/Float64 "{data: 40.0}" > /dev/null 2>&1
start timeout 300 ros2 topic pub -r 10 /mavros/global_position/compass_hdg std_msgs/msg/Float64 "{data: 0.0}" > /dev/null 2>&1
for cam in rgb thermal; do
  start timeout 300 python3 "$REPO/ros2_ws/pub_frames.py" "$OUT/frames" --topic "/camera/$cam/image_raw" --hz 5 \
    --seconds 300 > "$OUT/pub_$cam.log" 2>&1
done
start timeout -s INT 240 ros2 launch esa_ros esa_dual.launch.py rgb_model:="$MODEL" thermal_model:="$MODEL" \
  rgb_hfov_deg:=65.5 thermal_hfov_deg:=65.5 > "$OUT/launch.log" 2>&1
start timeout 240 python3 "$REPO/web/backend/server.py" --ros --camera-topics /rgb/tracks_geo /thermal/tracks_geo \
  --token "$TOKEN" --port 8000 --reviews "$OUT/reviews.jsonl" > "$OUT/server.log" 2>&1
sleep 8
python3 "$REPO/ros2_ws/check_ground_station.py" --token "$TOKEN" --seconds 150 --out "$OUT/result.json"
RC=$?
for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
wait 2>/dev/null
body=$(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "$OUT/result.json" 2>/dev/null)
if [ "$RC" = 0 ]; then echo "::notice title=RGB + thermal end to end::$body"; else
  echo "::error title=RGB + thermal end to end::$body"
  tail -n 40 "$OUT/launch.log"; tail -n 20 "$OUT/server.log"
fi
exit $RC
