# Security policy

This project is research software for search and rescue. Some of it touches safety: the ground station decides what
operators see and what reaches rescuers' TAK maps, and the mission tool uploads flight plans and geofences to an
autopilot. Reports about those parts are especially welcome.

## Reporting a vulnerability

Please report privately through GitHub: **Security → Report a vulnerability** on this repository
(private vulnerability reporting). Do not open a public issue for a vulnerability.

Include what you found, how to reproduce it, and what an attacker could do with it. You should get a first reply
within 7 days. Fixes are made on a private branch where needed, released, and then described in the changelog with
credit to the reporter unless you prefer otherwise.

## Supported versions

Only the latest commit on `main` and the latest release are supported.

## Scope

In scope:
- `web/backend/server.py` and the map app (`web/frontend/`): authentication, origin checks, injection, denial of
  service, the review audit log
- `tools/fly_mission.py` and `src/esa/mission.py`: anything that lets a plan bypass the altitude, geofence or
  return-to-launch checks, or arms a real vehicle
- the ROS 2 nodes, the TAK (Cursor on Target) output, and the GitHub workflows
- the shipped model files (`models/`, checked against `models/SHA256SUMS`)

Out of scope: the third-party datasets and upstream projects (Ultralytics, ArduPilot, PX4, ROS 2, MapLibre); please
report those upstream.

## What the project already does

Operator tokens and origin checks on the ground station, loopback-only by default and TLS when given a certificate;
hash-chained audit log of operator decisions; rate- and size-limited messages; a Content-Security-Policy on the map
app; plan checks before every upload; read-only, SHA-pinned CI workflows checked by `zizmor`; Dependabot updates;
checksummed model files. Known gaps are listed in [docs/LIMITATIONS.md](docs/LIMITATIONS.md) and
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): TAK over plain TCP (use a VPN or TAK's TLS port), ROS 2 without SROS2,
MAVLink without signing.
