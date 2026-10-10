import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";

const WS_URL = import.meta.env.VITE_WS_URL ?? `ws://${location.hostname}:8000/ws`;
const TRAIL = 30; // positions kept per track

// OpenStreetMap raster tiles: fine for development and a demo video.
// For anything heavier, use your own tile server or a provider (OSM tile usage policy).
const STYLE = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};

// ?offline=1 : no tile server needed (plain background + 20 m grid). Used for demos recorded
// without internet access, and handy on a field laptop.
const PARAMS = new URLSearchParams(location.search);
const OFFLINE = PARAMS.has("offline");
// ?center=lat,lon&zoom=z pins the view (e.g. on a planned search area) instead of jumping to the drone
const FIXED_CENTER = PARAMS.get("center")?.split(",").map(Number);
const FIXED_ZOOM = PARAMS.has("zoom") ? Number(PARAMS.get("zoom")) : null;
const OFFLINE_STYLE = {
  version: 8, sources: {},
  layers: [{ id: "bg", type: "background", paint: { "background-color": "#1b2027" } }],
};

const COLORS = ["match", ["get", "label"],
  "swimmer", "#ff5252", "pedestrian", "#ff5252", "people", "#ff8a80",
  "boat", "#42a5f5", "jetski", "#ba68c8", "bicycle", "#ba68c8", "motor", "#ba68c8",
  "tricycle", "#ce93d8", "awning-tricycle", "#ce93d8",
  "car", "#42a5f5", "van", "#29b6f6", "truck", "#ffb74d", "bus", "#4db6ac", "#9ccc65"];

function gridAround(lat, lon, stepM = 20, spanM = 400) {
  const mLat = 111320, mLon = 111320 * Math.cos((lat * Math.PI) / 180);
  const lines = [];
  for (let d = -spanM; d <= spanM; d += stepM) {
    lines.push({ type: "Feature", properties: {}, geometry: { type: "LineString",
      coordinates: [[lon - spanM / mLon, lat + d / mLat], [lon + spanM / mLon, lat + d / mLat]] } });
    lines.push({ type: "Feature", properties: {}, geometry: { type: "LineString",
      coordinates: [[lon + d / mLon, lat - spanM / mLat], [lon + d / mLon, lat + spanM / mLat]] } });
  }
  return { type: "FeatureCollection", features: lines };
}

const fc = (features) => ({ type: "FeatureCollection", features });

export default function App() {
  const mapDiv = useRef(null);
  const map = useRef(null);
  const trails = useRef(new Map());
  const centred = useRef(false);
  const dronePath = useRef([]);
  const seenTracks = useRef(new Set());
  const [uniqueTracks, setUniqueTracks] = useState(0);
  const [status, setStatus] = useState("connecting…");
  const [frame, setFrame] = useState(null);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    const m = new maplibregl.Map({ container: mapDiv.current, style: OFFLINE ? OFFLINE_STYLE : STYLE,
      center: FIXED_CENTER ? [FIXED_CENTER[1], FIXED_CENTER[0]] : [24.95, 60.15],
      zoom: FIXED_ZOOM ?? (OFFLINE ? 18.3 : 16) });
    m.addControl(new maplibregl.NavigationControl());
    m.on("load", () => {
      m.addSource("grid", { type: "geojson", data: fc([]) });
      m.addLayer({ id: "grid", type: "line", source: "grid", paint: { "line-color": "#2c333d", "line-width": 1 } });
      m.addSource("dronepath", { type: "geojson", data: fc([]) });
      m.addLayer({ id: "dronepath", type: "line", source: "dronepath",
        paint: { "line-color": "#ffd600", "line-width": 2, "line-dasharray": [2, 2], "line-opacity": 0.8 } });
      m.addSource("trails", { type: "geojson", data: fc([]) });
      m.addSource("dets", { type: "geojson", data: fc([]) });
      m.addSource("drone", { type: "geojson", data: fc([]) });
      m.addLayer({ id: "trails", type: "line", source: "trails",
        paint: { "line-color": COLORS, "line-width": 2, "line-opacity": 0.6 } });
      m.addLayer({ id: "dets", type: "circle", source: "dets",
        paint: { "circle-radius": 7, "circle-color": COLORS, "circle-stroke-color": "#fff", "circle-stroke-width": 2 } });
      m.addLayer({ id: "drone", type: "circle", source: "drone",
        paint: { "circle-radius": 9, "circle-color": "#111", "circle-stroke-color": "#ffd600", "circle-stroke-width": 3 } });
      m.on("click", "dets", (e) => {
        const p = e.features[0].properties;
        new maplibregl.Popup().setLngLat(e.lngLat)
          .setHTML(`<b>${p.label}</b><br/>track ${p.track_id}<br/>conf ${p.conf}`).addTo(m);
      });
    });
    map.current = m;
    return () => m.remove();
  }, []);

  useEffect(() => {
    let ws, retry;
    const connect = () => {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => setStatus("live");
      ws.onclose = () => { setStatus("disconnected – retrying"); retry = setTimeout(connect, 2000); };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.frame === 0) {  // a replay restarted: start clean so nothing is counted twice
          setAlerts([]); seenTracks.current = new Set(); trails.current = new Map();
        }
        setFrame(msg);
        if (msg.alerts?.length) setAlerts((a) => [...msg.alerts.map((x) => ({ ...x, at: new Date() })), ...a].slice(0, 20));
        draw(msg);
      };
    };
    connect();
    return () => { clearTimeout(retry); ws && (ws.onclose = null, ws.close()); };
  }, []);

  function draw(msg) {
    const m = map.current;
    if (!m?.isStyleLoaded()) return;
    if (msg.frame === 0) { dronePath.current = []; }
    if (msg.drone && !centred.current) {
      if (!FIXED_CENTER) m.jumpTo({ center: [msg.drone.lon, msg.drone.lat] });
      if (OFFLINE) m.getSource("grid").setData(gridAround(msg.drone.lat, msg.drone.lon));
      centred.current = true;
    }
    if (msg.drone) {
      dronePath.current.push([msg.drone.lon, msg.drone.lat]);
      if (dronePath.current.length > 400) dronePath.current.shift();
      m.getSource("dronepath").setData(fc([{ type: "Feature", properties: {},
        geometry: { type: "LineString", coordinates: dronePath.current } }]));
    }
    // detections above the horizon (tilted camera) have no ground position: keep them off the map
    const dets = (msg.detections ?? []).filter((d) => d.lat != null && d.lon != null);
    for (const d of dets) if (d.track_id != null) seenTracks.current.add(d.track_id);
    setUniqueTracks(seenTracks.current.size);
    for (const d of dets) {
      if (d.track_id == null) continue;
      const t = trails.current.get(d.track_id) ?? { label: d.label, pts: [] };
      t.pts.push([d.lon, d.lat]); if (t.pts.length > TRAIL) t.pts.shift();
      trails.current.set(d.track_id, t);
    }
    const live = new Set(dets.map((d) => d.track_id));
    for (const id of trails.current.keys()) if (!live.has(id)) trails.current.delete(id); // simple: drop lost tracks
    m.getSource("dets").setData(fc(dets.map((d) => ({ type: "Feature",
      geometry: { type: "Point", coordinates: [d.lon, d.lat] }, properties: d }))));
    m.getSource("trails").setData(fc([...trails.current.values()].filter((t) => t.pts.length > 1)
      .map((t) => ({ type: "Feature", geometry: { type: "LineString", coordinates: t.pts }, properties: { label: t.label } }))));
    if (msg.drone) m.getSource("drone").setData(fc([{ type: "Feature",
      geometry: { type: "Point", coordinates: [msg.drone.lon, msg.drone.lat] }, properties: {} }]));
  }

  const counts = {};
  for (const d of frame?.detections ?? []) counts[d.label] = (counts[d.label] ?? 0) + 1;

  return (
    <div className="app">
      <div ref={mapDiv} className="map" />
      <aside className="panel">
        <h1>Situational awareness</h1>
        <p className={`status ${status === "live" ? "ok" : ""}`}>● {status}</p>
        {frame && <p>Frame {frame.frame} · {frame.fps} FPS · {uniqueTracks} track IDs so far</p>}
        <h2>In view</h2>
        {Object.keys(counts).length ? (
          <ul>{Object.entries(counts).map(([k, v]) => <li key={k}>{k}: {v}</li>)}</ul>
        ) : <p className="muted">nothing detected</p>}
        <h2>Alerts</h2>
        {alerts.length ? (
          <ul className="alerts">{alerts.map((a, i) => (
            <li key={i}><b>{a.message}</b><br />
              <span className="muted">{a.at.toLocaleTimeString()} · {a.lat?.toFixed(5)}, {a.lon?.toFixed(5)}</span></li>
          ))}</ul>
        ) : <p className="muted">none</p>}
        {OFFLINE && <p className="muted small">Offline map: background grid every 20 m, dashed yellow line = drone path.</p>}
        <p className="muted small">Positions are estimated from pixel location assuming a nadir camera over flat ground — see docs/LIMITATIONS.md.</p>
      </aside>
    </div>
  );
}
