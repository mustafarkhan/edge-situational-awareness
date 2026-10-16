#!/usr/bin/env bash
# Fly a search plan in ArduPilot SITL (software in the loop) and check the flown track.
#
#   ARDUCOPTER=path/to/arducopter bash scripts/sitl_fly.sh samples/search_example.plan out.json
#
# Starts the simulator at the plan's home position, uploads mission + geofence with tools/fly_mission.py,
# arms, flies the mission in AUTO and checks every waypoint was reached and the drone stayed inside the fence.
# ARDUPILOT is the ArduPilot source tree (for the default copter parameters); SPEEDUP runs the simulation faster.
set -euo pipefail
PLAN=${1:-samples/search_example.plan}
REPORT=${2:-sitl_report.json}
ARDUCOPTER=${ARDUCOPTER:-arducopter}
ARDUPILOT=${ARDUPILOT:-}
SPEEDUP=${SPEEDUP:-10}
if python3 -c "import socket; socket.create_connection(('127.0.0.1', 5760), 1)" 2>/dev/null; then
  echo "something is already listening on port 5760 (another simulator?): stop it first" >&2; exit 2
fi
WORK=$(mktemp -d)
HOME_POS=$(python3 -c "import json,sys; h=json.load(open(sys.argv[1]))['mission']['plannedHomePosition']; print(f'{h[0]},{h[1]},{h[2] or 0},0')" "$PLAN")
DEFAULTS=()
[ -n "$ARDUPILOT" ] && DEFAULTS=(--defaults "$ARDUPILOT/Tools/autotest/default_params/copter.parm")
(cd "$WORK" && exec "$ARDUCOPTER" --model + --speedup "$SPEEDUP" --home "$HOME_POS" "${DEFAULTS[@]}" -I0 > sitl.log 2>&1) &
SITL=$!
trap 'kill $SITL 2>/dev/null || true' EXIT
sleep 5
python3 -u tools/fly_mission.py --connect tcp:127.0.0.1:5760 --plan "$PLAN" --sitl --fly --timeout "${TIMEOUT:-1200}" \
  --report "$REPORT"
