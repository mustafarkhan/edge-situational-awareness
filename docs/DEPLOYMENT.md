# From the bench to real drones: deployment and field-test guide

This page is for taking the system onto a university drone and testing it properly: what to build, what the
software is and is not allowed to do in the air, the rules to check, and a staged test plan whose results can be
compared with the predictions in README §10. It is written for a small team (a pilot, an observer, someone at the
ground station) flying under a university's drone operations.

**Read first:** nothing in this repository has flown yet. Every number in the README comes from public datasets
and simulation. The field tests below are how that changes.

## 1. What the system does in the air, and what it never does

| | the autopilot (PX4 / ArduPilot) | this software (on the Jetson) |
|---|---|---|
| flies the drone, take-off, landing, failsafes | **yes** | no |
| keeps the drone inside the approved area | **yes**: geofence uploaded with every mission, breach → return to launch | no |
| reads position, height, heading, gimbal angle | | yes, read-only through MAVROS |
| finds people, raises alerts, puts them on the map / TAK | | yes, **advisory only**: a person at the ground station confirms or rejects every alert |
| arms the drone or starts a mission | no: the pilot does, from the ground station | `tools/fly_mission.py` refuses unless the autopilot is a simulator |

The design rule: **if the Jetson freezes, reboots or is unplugged mid-flight, the flight is unaffected.** The
detector only reads telemetry; it sends no commands to the autopilot. The ground station shows when the detector
is unhealthy (`/status`: `NO_CAMERA`, or `DEGRADED` when telemetry is stale so no coordinates are published) so
the crew knows when they are searching by eye alone.

Missions come from `benchmarks/search_planning.py` (sea) or `benchmarks/land_search.py` (land, night): a
QGroundControl `.plan` with take-off, the search pattern at the chosen altitude and speed, return to launch, and an
inclusion geofence 50 m around the search box. Open it in QGroundControl, check it, and upload it from there, or
upload it with `tools/fly_mission.py --connect ... --plan ... --rtl-alt <m>` (upload only on a real vehicle). The
tool refuses plans less than 5 m under the 120 m ceiling, sets the autopilot's fence ceiling to 120 m with
return-to-launch on breach, and sets the return altitude. Choose `--rtl-alt` from the site survey: above the
tallest obstacle between the search area and home, plus margin. If QGroundControl uploads the mission instead,
set the same parameters there (ArduCopter: `FENCE_ALT_MAX`, `FENCE_ACTION`, `RTL_ALT_M`; PX4: `GF_MAX_VER_DIST`,
`GF_ACTION`, `RTL_RETURN_ALT`).

## 2. Hardware

A configuration that fits the software as it is (examples; check current specifications and prices):

| part | what it needs to do | examples |
|---|---|---|
| airframe + autopilot | carries ~1.5 kg payload for 20+ min, runs PX4 or ArduPilot | Holybro X500 V2 with a Pixhawk 6C, or the university's existing PX4/ArduPilot platform |
| companion computer | runs the detector | Jetson Orin Nano 8 GB (see [JETSON.md](JETSON.md) to measure it first), on a small carrier board |
| camera | RGB + thermal on a stabilised gimbal, video over Ethernet (RTSP) or USB | dual-sensor gimbals with a 640×512 radiometric or non-radiometric thermal core |
| telemetry link | ground station ↔ autopilot | the autopilot's normal telemetry radio |
| data link to the ground | detections and alerts (small JSON, a few kB/s), optionally video | Wi-Fi / mesh radio; the alerts do not need video bandwidth |
| ground station | QGroundControl + the web map / TAK (`web/backend/server.py --tak ...`) | a laptop; ATAK on a tablet for the field team |

Set `rgb_hfov_deg` / `thermal_hfov_deg` from the gimbal's datasheet for the lens actually fitted: the search
planner and the geolocation both depend on it (README §10, §4). A thermal lens that is not the 25° HIT-UAV one needs
its own detection curve before the planner's altitudes can be trusted (test stage T2 below measures it).

Wiring on the drone:

```
gimbal ──RTSP/USB──▶ camera driver ──/camera/{rgb,thermal}/image_raw──▶ esa_dual.launch.py
Pixhawk ──serial/USB──▶ MAVROS ──/mavros/global_position/*──▶            │
                                                          /rgb/alerts, /thermal/alerts, /*/status
                                                                         ▼
                                   data link ──▶ ground station: web map + TAK (operator confirms/rejects)
```

```bash
ros2 launch mavros apm.launch fcu_url:=/dev/ttyTHS1:921600        # or px4.launch
ros2 launch esa_ros esa_dual.launch.py \
    rgb_topic:=/camera/rgb/image_raw thermal_topic:=/camera/thermal/image_raw \
    rgb_model:=models/sds_y11n_ep4.engine thermal_model:=models/hit_y11n_ep5.engine \
    thermal_hfov_deg:=<from datasheet> record:=true
```

Gimbal pitch has no standard ROS topic: publish it on `/gimbal_pitch` (degrees below the horizon, 90 = straight
down) from the gimbal's SDK, or fly with the camera locked straight down and leave `pitch_deg:=90`.

Ground station, for a field team on a shared network:

```bash
# one line per operator: '<name> <token>'; generate tokens with: python -c "from esa.access import new_token; print(new_token())"
python web/backend/server.py --ros --camera-topics /rgb/tracks_geo /thermal/tracks_geo --operators ops.txt --host 0.0.0.0 \
    --ssl-certfile gcs.crt --ssl-keyfile gcs.key --allow-origin https://gcs.local:5173 --tak udp://239.2.3.1:6969
```

Each operator opens the map with their own link (`https://gcs.local:5173/#token=<their token>`); their decisions are
logged under their name in `outputs/reviews.jsonl`. Check the log with
`python -c "from esa.access import AuditLog; print(AuditLog.verify('outputs/reviews.jsonl'))"`. Treat the token file
like a password list: keep it off shared drives and out of git. Known gaps (audit, fix plan steps 3 and 5): ROS 2
traffic and TAK output are not yet encrypted, so keep ROS on the drone (`ROS_LOCALHOST_ONLY=1`) and the TAK network
closed.

## 3. Rules (EU, Finland) — check before every campaign

This is a checklist, not legal advice; the university's drone operations lead and the national authority decide.

* **EU Regulation 2019/947** applies across the EU. Flights under 120 m within visual line of sight can usually be
  done in the **Open** category; anything beyond visual line of sight, over people, or with a heavier or custom
  aircraft usually needs the **Specific** category (a standard scenario or an operational authorisation based on a
  risk assessment such as SORA). A self-built drone with a Jetson and a gimbal has no class mark, which limits which
  Open subcategory it can use.
* **Operator registration and pilot competency**: the operator (the university or research group) is registered
  with the national authority (Traficom in Finland) and the remote pilots hold the competency the category needs.
* **Airspace**: check restricted zones and controlled airspace for the test site before each day, and NOTAMs.
* **Privacy**: test areas are chosen and announced so that people who did not agree to be filmed are not; the
  recordings are personal data when people are recognisable (section 6).
* **Exports**: drones and thermal cameras can be dual-use items. Taking the system abroad, or sharing it with a
  partner outside the EU, goes through the university's export-control office first.
* **Insurance** for the aircraft and third-party liability, per the university's rules.

The software helps with compliance but does not replace it: plans stay 5 m under the EU 120 m limit, the autopilot's
fence ceiling is set to 120 m, the planner writes a geofence, and the flight log (rosbag) records what was flown.

## 4. Where this system is meant to be used

Search and rescue: finding people who need help, on land, at sea, after disasters (floods, earthquakes,
wildfires), and in the humanitarian setting of armed conflict — locating casualties, missing people and people
trapped in damaged areas so that rescuers and medical teams can reach them.

**It is not for targeting, for tracking or surveilling combatants or civilians, or for any use of force.** Alerts
mark *people who may need help* for a human rescuer to check. That boundary is part of the project: the model
labels, the alert wording ("possible person"), the TAK marker types (unknown, not hostile) and the operator review
step are all built for rescue. Anyone extending the system for other purposes does so outside this project and its
licence obligations still apply (AGPL-3.0).

In conflict and disaster zones, coordinate with the agency in charge of the response (national rescue services,
humanitarian coordination) before flying: an unannounced drone can stop crewed rescue helicopters from flying.

## 5. Staged field tests

Each stage has a pass condition; the next stage starts only when it is met. The aim is to measure the same
quantities the README predicts, so the predictions can be confirmed or corrected in public.

| stage | where | what | measured | pass when |
|---|---|---|---|---|
| **T0 bench** | lab | ArduPilot SITL + MAVROS + detector node (as in CI); the real Jetson + camera on the desk | frame rate, latency, `/status` behaviour when the camera or telemetry is unplugged | `NO_CAMERA` / `DEGRADED` shown within 2 s; flight unaffected by killing the detector |
| **T1 static** | open field, ground | camera on a mast or the drone on the ground, volunteers and mannequins at surveyed distances | pixels-on-target vs detection, per class, RGB and thermal, day and night | the measured curve is recorded (it replaces or confirms the dataset curve) |
| **T2 hover ladder** | open field, VLOS | hover over surveyed targets at 20/40/60/80/100/120 m | detection probability vs altitude, geolocation error vs RTK-surveyed target positions | geolocation median within the README §4 budget for the measured attitude errors |
| **T3 blind search, land** | field / forest edge, VLOS | planned lawnmower search; a third party hides 5–10 targets the crew does not know about | POD (found / hidden), false alerts per km², time to first confirmed find, area per hour | measured POD compared with the planned POD (report both, whatever the result) |
| **T4 blind search, sea** | lake or sheltered sea with a safety boat and a water-rescue partner | as T3 with swimmers in dry suits and floating mannequins | as T3, plus sea state and sun glint conditions | as T3 |
| **T5 night, thermal** | as T3, after dark (where permitted) | thermal search with the land planner's altitude | as T3 | as T3 |
| **T6 disaster-like terrain** | rubble training ground of a rescue service, by agreement | partially hidden people and mannequins among debris | as T3, plus how often a person is visible at all | documented; no pass condition (this is where the method's limits are found) |

Safety for every sortie: a pilot in command with manual override, an observer, a briefing with go/no-go criteria
(wind, battery, link), a cleared area, and volunteers who signed up, wear high-visibility clothing when not playing a
casualty, and are never flown over at low altitude. At sea, a safety boat stays with the swimmers.

## 6. Data from the field

* **Record every sortie** (`record:=true`): images, telemetry, detections, alerts, status, operator decisions.
  Ground truth (where targets were, surveyed with RTK GNSS where possible) goes in a sortie sheet next to the bag.
* **Consent**: volunteers sign consent for recording and for the data's use in research and, if wanted, publication.
  Bystanders who are recognisable are blurred before anything is shared.
* **Storage**: on university storage under the group's data management plan; only consented, blurred subsets are
  ever published.
* **The sortie report** for each flight lists: area, altitude, speed, flight time, targets hidden / found, false
  alerts, time to first find, geolocation errors, frames dropped, detector health — so the README can report
  field results the same way it reports dataset results.

## 7. What is tested before every field campaign

| check | how | in CI |
|---|---|---|
| detector node runs on real images; topics published; coordinates correct straight down, tilted, from live telemetry and from an Image topic | `ros2_ws/test_node.sh` | yes |
| mission upload (mission + geofence) and a full simulated flight of the search plan | `tools/fly_mission.py` against ArduPilot SITL | yes |
| ground station ↔ TAK server | `tests/tak_e2e.py` | yes |
| map app in a real browser: token required, labels never run as script, no CSP violations, reviews attributed | `tests/browser_e2e.py` | yes |
| the README's numbers still match the result files | `benchmarks/check_readme.py` | yes |
| the Jetson's frame rate and accuracy after TensorRT conversion | `benchmarks/run_on_jetson.sh` | no (needs the board) |
