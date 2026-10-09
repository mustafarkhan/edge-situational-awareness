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

const COLORS = ["match", ["get", "label"],
  "swimmer", "#e53935", "boat", "#1e88e5", "jetski", "#8e24aa",
  "car", "#fb8c00", "truck", "#6d4c41", "bus", "#00897b", "#43a047"];

const fc = (features) => ({ type: "FeatureCollection", features });

export default function App() {
  const mapDiv = useRef(null);
  const map = useRef(null);
  const trails = useRef(new Map());
  const centred = useRef(false);
  const [status, setStatus] = useState("connecting…");
  const [frame, setFrame] = useState(null);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    const m = new maplibregl.Map({ container: mapDiv.current, style: STYLE, center: [24.95, 60.15], zoom: 16 });
    m.addControl(new maplibregl.NavigationControl());
    m.on("load", () => {
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
    if (msg.drone && !centred.current) { m.jumpTo({ center: [msg.drone.lon, msg.drone.lat] }); centred.current = true; }
    const dets = msg.detections ?? [];
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
        {frame && <p>Frame {frame.frame} · {frame.fps} FPS</p>}
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
        <p className="muted small">Positions are estimated from pixel location assuming a nadir camera over flat ground — see docs/LIMITATIONS.md.</p>
      </aside>
    </div>
  );
}
