// One renderer per manifest panel type (ADR-0129 Interface A), ported from the
// vanilla dashboard. Everything the robot sends is rendered as text through
// React — never as HTML.
import { Component, useEffect, useRef, useState } from "react";
import { SAFETY_LIVENESS_S } from "./lib/safety.js";
import {
  b64Blob,
  jpegBlob,
  coverMap,
  eventMeta,
  fmtAge,
  fmtMs,
  fmtNum,
  fmtTime,
  footprintToPixels,
  labelSuffix,
  metricNamespace,
  MISSION_MARK,
  num,
  overlayColor,
  overlayFade,
  overlayInk,
  rewardBand,
  rewardValueBand,
  robotNow,
  sevBucket,
  shortId,
  sparkGeometry,
  str,
  stripMetricPrefix,
  worldToPixel,
} from "./lib/render.js";

const STALE_S = 10;
const WIDE = new Set(["metrics", "events", "reasoner", "map"]);
const DOT_LABEL = { wait: "waiting for data", live: "receiving data", paused: "data paused", error: "error" };
const finite = (v) => typeof v === "number" && Number.isFinite(v);
const clamp01 = (v) => Math.max(0, Math.min(1, Number(v) || 0));
const pct = (v) => `${Math.round(clamp01(v) * 100)}%`;
const s = (n) => (n === 1 ? "" : "s");

// `ts === false` hides the freshness dot and age (panels with no data stream);
// `right` takes the age's place at the top right.
function Card({ panel, ts, error, extra, right, children }) {
  const st = error ? "error" : ts ? (robotNow() - ts < STALE_S ? "live" : "paused") : "wait";
  const title = String(panel.title || panel.type);
  return (
    <section className={WIDE.has(panel.type) ? "card wide" : "card"} aria-label={title}>
      <header className="card-hd">
        {ts !== false && <span className={`dot st-${st}`} title={DOT_LABEL[st]} aria-hidden="true" />}
        <h2>{title}</h2>
        {ts !== false && <span className="sr-only">{DOT_LABEL[st]}</span>}
        {extra}
        {right ?? (ts !== false && <span className="age">{fmtAge(ts)}</span>)}
      </header>
      <div className="card-bd">{children}</div>
    </section>
  );
}

const Empty = ({ children }) => <p className="empty">{children}</p>;

function KV({ rows }) {
  return (
    <dl className="kv">
      {rows
        .filter((r) => r && r[1] != null && r[1] !== "")
        .map(([k, v, cls]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd className={cls}>{String(v)}</dd>
          </div>
        ))}
    </dl>
  );
}

// A panel that throws on unexpected robot data must not take the page (and
// the E-STOP above it) down. Retries on the next snapshot.
export class Boundary extends Component {
  state = { err: null };
  static getDerivedStateFromError(err) {
    return { err };
  }
  componentDidUpdate(prev) {
    if (this.state.err && prev.reset !== this.props.reset) this.setState({ err: null });
  }
  render() {
    if (!this.state.err) return this.props.children;
    return (
      <section className="card">
        <header className="card-hd">
          <h2>{String(this.props.title ?? "")}</h2>
        </header>
        <p className="empty">This panel couldn't render the robot's data ({String(this.state.err?.message || this.state.err)}).</p>
      </section>
    );
  }
}

// base64 PNG from the snapshot → blob: URL (the website's CSP allows blob:, not data:).
function useB64Url(b64, type) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    let u = null;
    try {
      if (typeof b64 === "string" && b64) u = URL.createObjectURL(b64Blob(b64, type));
    } catch {
      u = null; // not base64
    }
    setUrl(u);
    return () => u && URL.revokeObjectURL(u);
  }, [b64, type]);
  return url;
}

// ── identity ──────────────────────────────────────────────────────────────
function Identity({ panel, snap, manifest }) {
  const id = snap?.identity || {};
  const eng = id["inference.engine"];
  const dev = id["inference.device"];
  return (
    <Card panel={panel} ts={snap?.last_ingest_ts}>
      <KV
        rows={[
          ["robot", manifest?.name],
          ["robot id", manifest?.robot_id],
          ["unit serial", manifest?.unit_serial],
          ["embodiment", manifest?.embodiment_kind],
          ["service", snap?.service_name],
          ["run mode", snap?.run_mode],
          ["run id", str(snap?.run_id, 12)],
          ["git sha", str(snap?.git_sha, 8)],
          ["robot model", id["openral.hal.robot.model"]],
          ["hal adapter", id["openral.hal.adapter"]],
          ["control mode", id["openral.hal.control_mode"]],
          ["skill id", id["openral.rskill.id"] || id["rskill.id"]],
          ["skill role", id["openral.rskill.role"] || id["rskill.role"]],
          ["engine · device", eng || dev ? `${eng || "—"} · ${dev || "—"}` : null],
          ["chunk size", id["inference.chunk_size"]],
          ["safety kernel", id["safety.kernel"]],
          ["latest trace", snap?.topics?.trace?.latest_trace_id],
        ]}
      />
    </Card>
  );
}

// ── camera ────────────────────────────────────────────────────────────────
function useVisible() {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return setVisible(true);
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting));
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return [ref, visible];
}

// Poll a JPEG while `path` is set (tile on screen) and the tab is visible;
// each frame becomes a blob: URL and the previous one is revoked.
function usePolledImage(api, path, every) {
  const [img, setImg] = useState({ url: null, err: null });
  useEffect(() => {
    if (!path) return;
    let stop = false;
    let t;
    let cur = null;
    const tick = async () => {
      let ok = false;
      if (!document.hidden) {
        try {
          const r = await api.call(path);
          ok = r.ok;
          if (r.ok) {
            const u = URL.createObjectURL(jpegBlob(await r.arrayBuffer()));
            if (stop) return URL.revokeObjectURL(u);
            setImg({ url: u, err: null });
            if (cur) URL.revokeObjectURL(cur);
            cur = u;
          } else if (!stop) setImg((x) => ({ ...x, err: r.status === 404 ? "no frame yet" : `HTTP ${r.status}` }));
        } catch {
          if (!stop) setImg((x) => ({ ...x, err: "robot unreachable" }));
        }
      }
      // No frame (404) or an error: ask again every 2 s, not at frame rate.
      if (!stop) t = setTimeout(tick, ok || document.hidden ? every : Math.max(every, 2000));
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(t);
      if (cur) URL.revokeObjectURL(cur);
    };
  }, [api, path, every]);
  return img;
}

// Segmenter masks arrive as base64 LA PNGs whose alpha is the mask; decode
// each once per payload.
function maskImage(cache, key, i, m, onLoad) {
  if (!m || typeof m.png_b64 !== "string") return null;
  let slot = cache.masks[i];
  if (!slot || slot.key !== key) {
    const el = new Image();
    let url = null;
    try {
      url = URL.createObjectURL(b64Blob(m.png_b64, "image/png"));
    } catch {
      return null;
    }
    el.onload = () => {
      URL.revokeObjectURL(url);
      onLoad();
    };
    el.onerror = () => URL.revokeObjectURL(url);
    el.src = url;
    slot = cache.masks[i] = { key, el };
  }
  return slot.el;
}

// Detector boxes and segmenter masks over the camera frame (canvas).
function drawOverlay(canvas, img, ov, cam, cache, onMaskLoad) {
  if (!canvas) return;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!ov) return;
  const now = robotNow();
  const camTs = cam?.ts_unix;
  const nw = img?.naturalWidth || 0;
  const nh = img?.naturalHeight || 0;

  const mk = ov.masks;
  if (mk && Array.isArray(mk.masks) && mk.masks.length) {
    const fade = overlayFade(mk, now, camTs);
    if (fade > 0) {
      const first = mk.masks[0] || {};
      const map = coverMap(nw, nh, w, h, (first.width || 1) / (first.height || 1));
      mk.masks.forEach((m, i) => {
        const bm = maskImage(cache, `${mk.stamp_unix}#${i}`, i, m, onMaskLoad);
        if (!bm || !bm.complete || !bm.naturalWidth) return;
        // Tint: draw the mask, then flood under `source-in` so the colour
        // lands on exactly the set pixels.
        const off = (cache.scratch ||= document.createElement("canvas"));
        off.width = bm.naturalWidth;
        off.height = bm.naturalHeight;
        const o = off.getContext("2d");
        o.drawImage(bm, 0, 0);
        o.globalCompositeOperation = "source-in";
        o.fillStyle = overlayColor(`${mk.rskill_id}#${i}`);
        o.fillRect(0, 0, off.width, off.height);
        o.globalCompositeOperation = "source-over";
        ctx.save();
        ctx.globalAlpha = 0.38 * fade;
        if (mk.flip_180) {
          ctx.translate(map.ox + map.dw / 2, map.oy + map.dh / 2);
          ctx.rotate(Math.PI);
          ctx.drawImage(off, -map.dw / 2, -map.dh / 2, map.dw, map.dh);
        } else ctx.drawImage(off, map.ox, map.oy, map.dw, map.dh);
        ctx.restore();
      });
    }
  }

  const det = ov.detections;
  if (det && Array.isArray(det.boxes) && det.boxes.length && det.frame_width && det.frame_height) {
    const fade = overlayFade(det, now, camTs);
    if (fade > 0) {
      const map = coverMap(nw, nh, w, h, det.frame_width / det.frame_height);
      // OPENRAL_DASHBOARD_FLIP_180: the tile is rotated, perception ran on the raw frame.
      const f = (u) => (det.flip_180 ? 1 - u : u);
      ctx.save();
      ctx.globalAlpha = fade;
      ctx.font = "500 10px 'IBM Plex Mono', ui-monospace, monospace";
      ctx.textBaseline = "top";
      ctx.lineWidth = 1.5;
      for (const b of det.boxes) {
        const bb = b?.bbox_xyxy;
        if (!Array.isArray(bb) || bb.length < 4) continue;
        const [u0, v0, u1, v1] = [
          f(bb[0] / det.frame_width),
          f(bb[1] / det.frame_height),
          f(bb[2] / det.frame_width),
          f(bb[3] / det.frame_height),
        ];
        const x0 = map.ox + Math.min(u0, u1) * map.dw;
        const y0 = map.oy + Math.min(v0, v1) * map.dh;
        const x1 = map.ox + Math.max(u0, u1) * map.dw;
        const y1 = map.oy + Math.max(v0, v1) * map.dh;
        // Colour by label, not array position: reordered output keeps colours.
        const color = overlayColor(b.label);
        ctx.strokeStyle = color;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        const text = `${b.label} ${Number(b.confidence || 0).toFixed(2)}`;
        const tw = ctx.measureText(text).width;
        const cy = y0 - 13 >= 0 ? y0 - 13 : y0 + 1;
        ctx.fillStyle = color;
        ctx.fillRect(x0, cy, tw + 8, 13);
        ctx.fillStyle = overlayInk(color);
        ctx.fillText(text, x0 + 4, cy + 2);
      }
      ctx.restore();
    }
  }
}

function Camera({ panel, snap, api, cfg }) {
  const source = typeof panel.source === "string" ? panel.source : "";
  const per = snap?.topics?.perception || {};
  const cam = per.cameras?.[source] || null;
  const ov = per.overlays?.[source] || null;
  const [box, visible] = useVisible();
  const { url, err } = usePolledImage(
    api,
    visible && source ? `/api/camera/${encodeURIComponent(source)}/snapshot.jpg` : null,
    cfg.mode === "remote" ? 1000 : 250,
  );
  const img = useRef(null);
  const canvas = useRef(null);
  const cache = useRef({ masks: {}, scratch: null });
  const [, redraw] = useState(0);
  const bump = () => redraw((n) => n + 1);
  useEffect(() => {
    drawOverlay(canvas.current, img.current, ov, cam, cache.current, bump);
  });
  const now = robotNow();
  const producers = [
    ov?.masks?.rskill_id && overlayFade(ov.masks, now, cam?.ts_unix) > 0 && shortId(ov.masks.rskill_id),
    ov?.detections?.model_id && overlayFade(ov.detections, now, cam?.ts_unix) > 0 && shortId(ov.detections.model_id),
  ].filter(Boolean);
  const live = !!cam?.ts_unix && now - cam.ts_unix < 2;
  return (
    <Card
      panel={panel}
      ts={cam?.ts_unix}
      right={
        cam && (
          <span className="age" title={cam.encoding || cam.modality || undefined}>
            {`${cam.width ?? "?"} × ${cam.height ?? "?"}`}
          </span>
        )
      }
    >
      <div className="cam" ref={box}>
        {url ? (
          <img ref={img} src={url} alt={`Camera ${source}`} onLoad={bump} />
        ) : (
          <div className="cam-ph">
            {err || "waiting for a frame"}
            {cam?.modality ? ` · ${cam.modality}` : ""}
          </div>
        )}
        <canvas ref={canvas} className="cam-ov" aria-hidden="true" />
        <div className="cam-pills">
          {live && <span className="pill live">live</span>}
          <span className="pill">{String(cam?.role || cam?.modality || "camera")}</span>
        </div>
        {producers.length > 0 && <div className="cam-src">{producers.join(" · ")}</div>}
      </div>
      <p className="meta">
        <span>{cam?.age_ms != null ? `age ${num(cam.age_ms, 0)} ms` : "—"}</span>
        {cam?.fps != null && <span className="right">{num(cam.fps, 0)} fps</span>}
      </p>
    </Card>
  );
}

// ── joints ────────────────────────────────────────────────────────────────
function Joints({ panel, snap }) {
  const rs = snap?.topics?.robot_state;
  const next = snap?.topics?.commands?.next_row;
  const names = rs?.names;
  const pos = rs?.positions;
  return (
    <Card panel={panel} ts={rs?.ts_unix}>
      {!Array.isArray(names) || !Array.isArray(pos) ? (
        <Empty>waiting for hal.read_state</Empty>
      ) : (
        <>
          <div className="joints">
            {names.map((name, i) => {
              const lo = finite(rs.limits_lo?.[i]) && rs.limits_lo[i] > -1e5 ? rs.limits_lo[i] : -Math.PI;
              const hi = finite(rs.limits_hi?.[i]) && rs.limits_hi[i] < 1e5 ? rs.limits_hi[i] : Math.PI;
              const at = (v) => `${Math.max(0, Math.min(100, ((v - lo) / (hi - lo || 1)) * 100))}%`;
              const vel = rs.velocities?.[i];
              const cmd = Array.isArray(next) ? next[i] : null;
              return (
                <div className="joint" key={i}>
                  <span className="jname">{String(name)}</span>
                  <span
                    className="jbar"
                    role="img"
                    aria-label={`${name} at ${num(pos[i], 3)}, range ${num(lo, 2)} to ${num(hi, 2)}`}
                  >
                    {finite(cmd) && <i className="jcmd" style={{ left: at(cmd) }} />}
                    {finite(pos[i]) && <i className="jdot" style={{ left: at(pos[i]) }} />}
                  </span>
                  <span className="num">{num(pos[i], 3)}</span>
                  <span className="num muted">{finite(vel) ? `${num(vel, 2)} /s` : "—"}</span>
                </div>
              );
            })}
          </div>
          <p className="legend">
            <span>
              <i className="sw jdot" /> actual
            </span>
            <span>
              <i className="sw jcmd" /> commanded
            </span>
            <span>bar = joint range</span>
          </p>
        </>
      )}
    </Card>
  );
}

// ── SLAM map ──────────────────────────────────────────────────────────────
function RobotMarker({ m }) {
  const { width: W, height: H, resolution_m: res } = m;
  if (!(W && H && res) || ![m.robot_x, m.robot_y, m.robot_yaw].every(finite)) return null;
  const ox = m.origin_x || 0;
  const oy = m.origin_y || 0;
  const yaw = m.robot_yaw;
  const { px, py } = worldToPixel(m.robot_x, m.robot_y, ox, oy, res, H);
  const poly = m.footprint_polygon;
  if (Array.isArray(poly) && poly.length >= 3) {
    // Real base outline, oriented by yaw, plus a heading line to the front edge.
    const pts = footprintToPixels(poly, m.robot_x, m.robot_y, yaw, ox, oy, res, H);
    const front = Math.max(...poly.map((p) => p[0]));
    const f = worldToPixel(m.robot_x + front * Math.cos(yaw), m.robot_y + front * Math.sin(yaw), ox, oy, res, H);
    const sw = Math.max(0.4, (front / res) * 0.12);
    return (
      <g className="robot-mark" strokeWidth={sw}>
        <polygon points={pts.map((p) => `${p.px},${p.py}`).join(" ")} />
        <line x1={px} y1={py} x2={f.px} y2={f.py} />
      </g>
    );
  }
  // Fallback: footprint circle + heading wedge (image y is flipped).
  const r = finite(m.footprint_radius_m) ? m.footprint_radius_m / res : Math.max(4, 0.25 / res);
  const hx = Math.cos(yaw);
  const hy = -Math.sin(yaw);
  const bx = px + hx * r * 0.2;
  const by = py + hy * r * 0.2;
  const half = r * 0.55;
  const wedge = [
    [px + hx * r * 1.6, py + hy * r * 1.6],
    [bx - hy * half, by + hx * half],
    [bx + hy * half, by - hx * half],
  ];
  return (
    <g className="robot-mark" strokeWidth={Math.max(0.5, r * 0.08)}>
      <circle cx={px} cy={py} r={r} />
      <polygon className="head" points={wedge.map((p) => p.join(",")).join(" ")} />
    </g>
  );
}

function SceneObjectMarks({ m, objects }) {
  const { height: H, resolution_m: res } = m;
  const r = Math.max(2, 0.12 / res);
  const font = Math.max(6, 0.35 / res);
  return objects
    .filter((o) => finite(o.x) && finite(o.y) && !(o.frame_id && m.frame_id && o.frame_id !== m.frame_id))
    .map((o, i) => {
      const { px, py } = worldToPixel(o.x, o.y, m.origin_x || 0, m.origin_y || 0, res, H);
      return (
        <g key={i} className="obj-mark">
          <circle cx={px} cy={py} r={r} strokeWidth={Math.max(0.3, r * 0.18)} />
          <text x={px + r * 1.4} y={py - r * 0.6} fontSize={font} strokeWidth={font * 0.12}>
            {String(o.label || o.id || "?")}
          </text>
        </g>
      );
    });
}

function MapPanel({ panel, snap }) {
  const m = snap?.topics?.slam || {};
  const so = snap?.topics?.scene_objects;
  const url = useB64Url(m.png_b64, "image/png");
  const { width: W, height: H } = m;
  const objects = Array.isArray(so?.objects) ? so.objects.filter((o) => o && typeof o === "object") : [];
  if (!m.ts_unix) {
    return (
      <Card panel={panel}>
        <Empty>waiting for slam.occupancy_grid (the SLAM backend must publish /map)</Empty>
      </Card>
    );
  }
  const posed = [m.robot_x, m.robot_y, m.robot_yaw].every(finite);
  return (
    <Card panel={panel} ts={m.ts_unix}>
      {url && W > 0 && H > 0 ? (
        <div className="map-stage" style={{ aspectRatio: `${W} / ${H}`, width: `min(100%, ${Math.round((420 * W) / H)}px)` }}>
          <img src={url} alt="SLAM occupancy grid" />
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
            <RobotMarker m={m} />
            {m.resolution_m > 0 && <SceneObjectMarks m={m} objects={objects} />}
          </svg>
        </div>
      ) : (
        <Empty>map metadata received; no map image in this stream</Empty>
      )}
      <p className="meta">
        <span>
          {W ?? "?"} × {H ?? "?"} cells
        </span>
        {m.resolution_m != null && <span>resolution {num(m.resolution_m, 3)} m/cell</span>}
        <span>
          origin ({num(m.origin_x || 0, 2)}, {num(m.origin_y || 0, 2)})
        </span>
        <span>frame {String(m.frame_id || "?")}</span>
        <span>source {String(m.source_node || "?")}</span>
        {posed && (
          <span>
            robot ({num(m.robot_x, 2)}, {num(m.robot_y, 2)}) yaw {num((m.robot_yaw * 180) / Math.PI, 0)}°
          </span>
        )}
      </p>
    </Card>
  );
}

// ── pointcloud ────────────────────────────────────────────────────────────
function Pointcloud({ panel, snap }) {
  const pc = snap?.topics?.pointcloud || {};
  const url = useB64Url(pc.png_b64, "image/png");
  return (
    <Card panel={panel} ts={pc.ts_unix}>
      {!pc.ts_unix ? (
        <Empty>waiting for world.pointcloud (deploy with --enable-octomap)</Empty>
      ) : (
        <>
          {url ? <img className="framed" src={url} alt="Robot-perspective octomap pointcloud" /> : <Empty>no pointcloud image in this stream</Empty>}
          <p className="meta">
            <span>{pc.n_points ?? "?"} points</span>
            {pc.range_max_m != null && <span>range {num(pc.range_max_m, 1)} m</span>}
            <span>frame {String(pc.frame_id || "?")}</span>
            <span>source {String(pc.source_node || "?")}</span>
          </p>
        </>
      )}
    </Card>
  );
}

// ── reasoner · mission ────────────────────────────────────────────────────
function Reasoner({ panel, snap }) {
  const r = snap?.topics?.reasoner;
  if (!r?.ts_unix) {
    return (
      <Card panel={panel}>
        <Empty>waiting for reasoner.tick (the reasoner_node must be ACTIVE)</Empty>
      </Card>
    );
  }
  const tasks = Array.isArray(r.mission?.tasks) ? r.mission.tasks.filter((t) => t && typeof t === "object") : [];
  const max = r.mission?.max_attempts || 3;
  const active = tasks.findIndex((t) => t.status === "active" || t.status === "verifying");
  const done = tasks.filter((t) => t.status === "done").length;
  const failed = tasks.filter((t) => t.status === "abandoned").length;
  const finished = active < 0 && !tasks.some((t) => t.status === "pending");
  const foot = [
    `tool ${r.tool || (r.suppressed_reason ? `suppressed: ${r.suppressed_reason}` : "no tool")}`,
    r.tick_idx != null && `tick ${r.tick_idx}`,
    r.rskill_id && `rskill ${r.rskill_id}`,
    r.model && `model ${r.model}`,
    r.force && "forced",
    r.error_kind && `error ${r.error_kind}`,
  ].filter(Boolean);
  return (
    <Card panel={panel} ts={r.ts_unix} error={!!r.error_kind}>
      {tasks.length === 0 ? (
        <p className="muted">no active mission (bare operator goal)</p>
      ) : (
        <>
          {finished && failed > 0 && (
            <p className="banner err">
              ✗ mission failed · {failed} of {tasks.length} task{s(tasks.length)} abandoned
            </p>
          )}
          <p className="mission-head">
            {tasks.length} task{s(tasks.length)} · {done} done{failed ? ` · ${failed} failed` : ""}
            {active >= 0 ? ` · on ${active + 1}/${tasks.length}` : " · complete"}
          </p>
          <ol className="mission">
            {tasks.map((t, i) => {
              // Dot-path ids (t1.2.1) carry the subdivision depth.
              const depth = typeof t.id === "string" ? t.id.split(".").length - 1 : 0;
              const score = /(?:success|ambiguous)=([0-9.]+)/.exec(String(t.verdict || ""));
              return (
                <li key={i} className={`task st-${t.status}`} style={depth ? { paddingLeft: depth * 14 } : undefined}>
                  <span className="mk" title={String(t.status)}>
                    {MISSION_MARK[t.status] || "·"}
                  </span>
                  <span className="tx">
                    {depth > 0 && <span className="tid">{t.id}</span>}
                    {String(t.text ?? "")}
                  </span>
                  <span className="mt">
                    {score && (
                      <span className={`bar band-${rewardBand(t)}`} title={`reward ${t.verdict}`}>
                        <i style={{ width: pct(parseFloat(score[1])) }} />
                      </span>
                    )}
                    {t.attempts || t.status === "active" || t.status === "verifying" ? (
                      <span className="num">
                        {String(t.attempts ?? 0)}/{String(max)}
                      </span>
                    ) : null}
                    {t.verdict && (
                      <span className="verdict" title={String(t.verdict)}>
                        {String(t.verdict)}
                      </span>
                    )}
                  </span>
                </li>
              );
            })}
          </ol>
        </>
      )}
      <p className="meta">{foot.join(" · ")}</p>
    </Card>
  );
}

// ── rSkill strip ──────────────────────────────────────────────────────────
function Rskill({ panel, snap }) {
  const c = snap?.cards || {};
  const sk = c.rskill_execute || c.rskill_tick || c.rskill_activate || null;
  const inf = c.inference || null;
  if (!sk) {
    return (
      <Card panel={panel}>
        <Empty>waiting for rskill.execute</Empty>
      </Card>
    );
  }
  const a = sk.attrs || {};
  const kind = inf?.attrs?.["inference.kind"];
  // The full skill step and the model forward pass inside it, on one line.
  const lat = [
    sk.duration_ms != null && `step ${fmtMs(sk.duration_ms)}`,
    inf?.ts_unix != null && `forward ${fmtMs(inf.duration_ms)}${kind && kind !== "foreground" ? ` (${kind})` : ""}`,
  ].filter(Boolean);
  const failed = sk.status_code === 2;
  return (
    <Card panel={panel} ts={sk.ts_unix} error={failed}>
      <p className="primary">{String(a["rskill.id"] ?? a["openral.skill.id"] ?? sk.name)}</p>
      {lat.length > 0 && <p className="sub">{lat.join("  ·  ")}</p>}
      <KV
        rows={[
          ["role", a["rskill.role"]],
          ["action applied", a["skill.action_applied"]],
          ["tick", a["openral.tick.idx"]],
          failed && ["error", sk.status_message || "error", "err"],
        ]}
      />
    </Card>
  );
}

// ── reward ────────────────────────────────────────────────────────────────
function Reward({ panel, snap }) {
  const rw = snap?.cards?.reward_score;
  // Reward queries are reasoner-paced: allow a minute before calling it stale.
  if (!rw?.ts_unix || robotNow() - rw.ts_unix > 60) {
    return (
      <Card panel={panel} ts={rw?.ts_unix}>
        <Empty>no reward score in the last minute</Empty>
      </Card>
    );
  }
  const a = rw.attrs || {};
  return (
    <Card panel={panel} ts={rw.ts_unix}>
      {[
        ["progress", "reward.progress"],
        ["success", "reward.success"],
      ].map(([label, key]) => {
        const v = clamp01(a[key]);
        return (
          <div className="gauge" key={key}>
            <span className="glabel">{label}</span>
            <span className={`bar band-${rewardValueBand(v)}`} role="img" aria-label={`${label} ${pct(v)}`}>
              <i style={{ width: pct(v) }} />
            </span>
            <span className="num">{pct(v)}</span>
          </div>
        );
      })}
      <p className="meta">
        {[
          `scored ${fmtAge(rw.ts_unix)} ago`,
          a["reward.camera"] && `camera ${a["reward.camera"]}`,
          a["reward.stalled"] && "stalled",
          a["reward.succeeded"] && "succeeded",
        ]
          .filter(Boolean)
          .join(" · ")}
      </p>
    </Card>
  );
}

// ── safety ────────────────────────────────────────────────────────────────
// ADR-0096: the latched /openral/safety_status. Nothing here is inferred: no
// status means "unknown", never "clear". Publishers re-stamp at 1 Hz, so an
// age past 3 s means the latched value can't be trusted (HZ-0096-1).

function SafetyStatus({ panel, snap }) {
  const st = snap?.topics?.safety_status;
  if (!st || st.ts_unix == null) {
    return (
      <Card panel={panel}>
        <p className="primary">waiting for /openral/safety_status</p>
        <p className="muted">No safety node seen: the state is unknown, not clear.</p>
      </Card>
    );
  }
  const age = robotNow() - st.ts_unix;
  const stale = age > SAFETY_LIVENESS_S;
  const reason = st.drop_reason_label || "";
  return (
    <Card panel={panel} ts={st.ts_unix} error={!!st.latched}>
      <p className="primary">
        <span className={`pill ${st.latched ? "violation" : stale ? "stale" : "ok"}`}>
          {st.latched ? "latched" : stale ? "stale" : "clear"}
        </span>{" "}
        {/* A DROP_* reason without a latch is a fail-closed drop: actions are suppressed. */}
        {st.latched ? reason : reason && reason !== "drop_none" ? `dropping · ${reason}` : "no drop in effect"}
      </p>
      <KV
        rows={[
          ["detail", st.detail],
          ["rskill", st.rskill_id],
          ["drop reason", reason ? `${reason} (${st.drop_reason})` : null],
          ["last transition", `${fmtAge(st.ts_unix)} ago`],
          stale && [
            "liveness",
            `no refresh for ${age.toFixed(1)} s (> ${SAFETY_LIVENESS_S} s): treat as UNKNOWN, not safe`,
            "err",
          ],
          ["trace", st.trace_id],
        ]}
      />
    </Card>
  );
}

function Ledger({ panel, snap }) {
  const sf = snap?.topics?.safety || {};
  const checks = sf.checks && typeof sf.checks === "object" ? Object.entries(sf.checks) : [];
  checks.sort((a, b) => (b[1]?.ts_unix || 0) - (a[1]?.ts_unix || 0));
  const v = sf.last_violation;
  return (
    <Card panel={panel} ts={sf.latest_ts_unix}>
      {checks.length === 0 && !v ? (
        <Empty>no safety checks recorded yet</Empty>
      ) : (
        <dl className="kv pills">
          {checks.map(([name, c]) => {
            const sev = String(c?.severity || "info").toLowerCase();
            return (
              <div key={name}>
                <dt>{name}</dt>
                <dd>
                  <span className={`pill ${sev}`}>{sev}</span>
                </dd>
              </div>
            );
          })}
          {/* Kept until the next violation: the per-check pill resets on the next
              pass and the event row is evicted within seconds. */}
          {v && (
            <div>
              <dt>last violation</dt>
              <dd>
                <span className="pill violation" title={JSON.stringify(v)}>
                  {String(v.drop_reason || v.check_name || "violation")}
                  {typeof v.violation_value === "number" ? ` ${v.violation_value.toPrecision(3)}` : ""} · {fmtAge(v.ts_unix)}
                </span>
              </dd>
            </div>
          )}
        </dl>
      )}
    </Card>
  );
}

// ── system ────────────────────────────────────────────────────────────────
function System({ panel, snap }) {
  const sys = snap?.topics?.system;
  const rows = [];
  if (sys) {
    if (finite(sys.cpu_util_pct)) rows.push(["cpu", sys.cpu_util_pct, `${num(sys.cpu_util_pct, 0)} %`]);
    if (finite(sys.ram_used_mb) && sys.ram_total_mb) {
      rows.push(["ram", (sys.ram_used_mb / sys.ram_total_mb) * 100, `${num(sys.ram_used_mb / 1024, 1)} / ${num(sys.ram_total_mb / 1024, 1)} GiB`]);
    }
    for (const [i, g] of Object.entries(sys.gpus && typeof sys.gpus === "object" ? sys.gpus : {})) {
      if (finite(g?.util_pct)) rows.push([`gpu${i}`, g.util_pct, `${num(g.util_pct, 0)} %`]);
      if (finite(g?.memory_used_mb) && g.memory_total_mb) {
        rows.push([
          `gpu${i} mem`,
          (g.memory_used_mb / g.memory_total_mb) * 100,
          `${num(g.memory_used_mb / 1024, 1)} / ${num(g.memory_total_mb / 1024, 1)} GiB`,
        ]);
      }
    }
  }
  return (
    <Card panel={panel} ts={sys?.ts_unix}>
      {rows.length === 0 ? (
        <Empty>waiting for system metrics</Empty>
      ) : (
        rows.map(([label, p, val]) => (
          <div className="gauge" key={label}>
            <span className="glabel">{label}</span>
            <span className={`bar${p >= 90 ? " crit" : p >= 75 ? " warn" : ""}`} role="img" aria-label={`${label} ${Math.round(p)}%`}>
              <i style={{ width: pct(p / 100) }} />
            </span>
            <span className="num">{val}</span>
          </div>
        ))
      )}
    </Card>
  );
}

// ── metrics ───────────────────────────────────────────────────────────────
const GRID = [0, 0.25, 0.5, 0.75, 1];

function MetricRow({ m, win, events, hover, setHover }) {
  const key = `${m.name}${labelSuffix(m.labels)}`;
  const stripped = stripMetricPrefix(m.name);
  const dot = stripped.lastIndexOf(".");
  const g = sparkGeometry(m.samples, win, m.threshold, m.threshold_dir);
  const p = (v) => (finite(v) ? v.toFixed(1) : "");
  let latest = m.latest != null ? num(m.latest, 2) : "—";
  if (m.kind === "histogram") latest = `n=${Array.isArray(m.samples) ? m.samples.length : 0}`;
  else if (m.kind === "sum") latest = num(m.cumulative ?? 0, 1);
  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const t = g.tMin + Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * (g.tMax - g.tMin);
    let best = m.samples[0];
    for (const smp of m.samples) if (Math.abs(smp[0] - t) < Math.abs(best[0] - t)) best = smp;
    setHover({ key, t: best[0], v: best[1], x: g.x(best[0]) / g.w, y: g.y(best[1]) / g.h });
  };
  const thr = typeof m.threshold === "number" ? m.threshold : null;
  const over = hover && thr != null && (m.threshold_dir === "lower" ? hover.v < thr : hover.v > thr);
  return (
    <div className="metric">
      <span className="mname" title={`${key}${m.unit ? ` (${m.unit})` : ""}`}>
        {dot >= 0 && <span className="ns">{stripped.slice(0, dot + 1)}</span>}
        {stripped.slice(dot + 1)}
        <span className="ns">{labelSuffix(m.labels)}</span>
      </span>
      <span className="unit">{String(m.unit || m.kind || "")}</span>
      <span className="num">{m.kind === "histogram" && p(m.p50) ? `p50 ${p(m.p50)}` : ""}</span>
      <span className="num">{m.kind === "histogram" && p(m.p95) ? `p95 ${p(m.p95)}` : ""}</span>
      <span className="num latest">{latest}</span>
      <span className="spark" onMouseMove={g ? onMove : undefined} onMouseLeave={() => setHover(null)}>
        {g && (
          <>
            <svg viewBox={`0 0 ${g.w} ${g.h}`} preserveAspectRatio="none" aria-hidden="true">
              {GRID.map((f) => (
                <line key={f} className="grid" x1={f * g.w} y1="0" x2={f * g.w} y2={g.h} />
              ))}
              {events
                .filter((ev) => ev.ts >= g.tMin && ev.ts <= g.tMax)
                .map((ev, i) => (
                  <line key={i} className={`evm sev-${ev.sev}`} x1={g.x(ev.ts)} y1="0" x2={g.x(ev.ts)} y2={g.h} />
                ))}
              {g.thrY != null && <line className="thr" x1="0" y1={g.thrY} x2={g.w} y2={g.thrY} />}
              <polyline className={g.breach ? "breach" : undefined} points={g.points} />
            </svg>
            <span className="yax ymax">{fmtNum(g.hi)}</span>
            <span className="yax ymin">{fmtNum(g.lo)}</span>
            {hover && (
              <>
                <span className="spark-dot" style={{ left: `${hover.x * 100}%`, top: `${hover.y * 100}%` }} />
                <span className="spark-tip">
                  <b>{fmtNum(hover.v)}</b> {fmtTime(hover.t)}
                  {over && <em>{m.threshold_dir === "lower" ? " under floor" : " over budget"}</em>}
                </span>
              </>
            )}
          </>
        )}
      </span>
    </div>
  );
}

function Metrics({ panel, snap }) {
  const [group, setGroup] = useState("all");
  const [frozen, setFrozen] = useState(null);
  const [hover, setHover] = useState(null);
  const live = Array.isArray(snap?.metrics) ? snap.metrics.filter((m) => m && typeof m.name === "string") : [];
  const all = frozen || live;
  const counts = { all: all.length };
  for (const m of all) counts[metricNamespace(m.name)] = (counts[metricNamespace(m.name)] || 0) + 1;
  const g = counts[group] ? group : "all";
  const shown = (g === "all" ? all : all.filter((m) => metricNamespace(m.name) === g))
    .slice()
    .sort((a, b) => metricNamespace(a.name).localeCompare(metricNamespace(b.name)) || a.name.localeCompare(b.name));
  // One time window for every row, so gridlines and the axis line up.
  let tMin = Infinity;
  let tMax = -Infinity;
  for (const m of shown) {
    for (const smp of Array.isArray(m.samples) ? m.samples : []) {
      tMin = Math.min(tMin, smp[0]);
      tMax = Math.max(tMax, smp[0]);
    }
  }
  const win = tMax > tMin ? { tMin, tMax } : null;
  const events = (Array.isArray(snap?.events) ? snap.events : [])
    .map((ev) => ({ ts: ev?.ts_unix, sev: sevBucket(ev?.severity) }))
    .filter((ev) => finite(ev.ts) && (ev.sev === "warn" || ev.sev === "error"));
  let lastNs = null;
  const chips = (
    <span className="chips" role="group" aria-label="Metric group">
      {Object.keys(counts)
        .sort((a, b) => (a === "all" ? -1 : b === "all" ? 1 : a.localeCompare(b)))
        .map((ns) => (
          <button type="button" key={ns} className="chip" aria-pressed={ns === g} onClick={() => setGroup(ns)}>
            {ns} <span className="cnt">{counts[ns]}</span>
          </button>
        ))}
      <button type="button" className="chip" aria-pressed={!!frozen} onClick={() => setFrozen(frozen ? null : live)}>
        {frozen ? "resume" : "freeze"}
      </button>
    </span>
  );
  return (
    <Card panel={panel} ts={snap?.last_ingest_ts} extra={chips}>
      {all.length === 0 ? (
        <Empty>No metrics ingested yet. Point a workload at this robot with OTEL_EXPORTER_OTLP_ENDPOINT.</Empty>
      ) : (
        <div className="metrics">
          {shown.map((m) => {
            const ns = metricNamespace(m.name);
            const head = g === "all" && ns !== lastNs;
            lastNs = ns;
            const key = `${m.name}${labelSuffix(m.labels)}`;
            return (
              <div key={key}>
                {head && (
                  <p className="group-hd">
                    {ns} · {counts[ns]}
                  </p>
                )}
                <MetricRow m={m} win={win} events={events} hover={hover?.key === key ? hover : null} setHover={setHover} />
              </div>
            );
          })}
          {win && (
            <div className="metric axis">
              <span className="mname">time →</span>
              <span className="ticks">
                {GRID.map((f) => (
                  <span key={f}>{fmtTime(win.tMin + f * (win.tMax - win.tMin))}</span>
                ))}
              </span>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

// ── events ────────────────────────────────────────────────────────────────
const COUNTERS = [
  ["safety violations", "openral.event.safety_violation", "danger"],
  ["e-stops", "openral.event.estop_requested", "danger"],
  ["deadline misses", "openral.event.deadline_missed", "warn"],
  ["sensor stale", "openral.event.sensor_stale", "warn"],
  ["skill failures", "openral.event.skill_failure", "danger"],
];
const BUCKETS = ["debug", "info", "warn", "error"];

function Events({ panel, snap }) {
  // DEBUG is off by default: the bridged structlog stream would flood the view.
  const [on, setOn] = useState({ debug: false, info: true, warn: true, error: true });
  const events = (Array.isArray(snap?.events) ? snap.events : []).filter((e) => e && typeof e === "object");
  const counters = snap?.counters || {};
  const counts = { debug: 0, info: 0, warn: 0, error: 0 };
  for (const ev of events) counts[sevBucket(ev.severity)]++;
  const shown = events.filter((ev) => on[sevBucket(ev.severity)]).slice(0, 60);
  const failState = events.find((e) => e.kind === "openral.event.skill_failure")?.attrs?.["openral.event.skill_failure.state"];
  const chips = (
    <span className="chips" role="group" aria-label="Severity filter">
      {BUCKETS.map((b) => (
        <button type="button" key={b} className={`chip sev-${b}`} aria-pressed={on[b]} onClick={() => setOn({ ...on, [b]: !on[b] })}>
          {b} <span className="cnt">{counts[b]}</span>
        </button>
      ))}
    </span>
  );
  return (
    <Card panel={panel} ts={events[0]?.ts_unix} extra={chips}>
      <div className="counters">
        {COUNTERS.map(([label, key, tone]) => {
          const v = Number(counters[key]) || 0;
          return (
            <div key={key} className={v ? `counter ${tone}` : "counter"}>
              <span className="clabel">
                {label}
                {key === "openral.event.skill_failure" && failState ? ` · ${failState}` : ""}
              </span>
              <span className="cval">{v}</span>
            </div>
          );
        })}
      </div>
      {events.length === 0 ? (
        <Empty>No events yet.</Empty>
      ) : shown.length === 0 ? (
        <Empty>No events match the active filters.</Empty>
      ) : (
        <ol className="events">
          {shown.map((ev, i) => (
            <li key={i} className={`ev sev-${sevBucket(ev.severity)}`}>
              <span className="num muted">{fmtTime(ev.ts_unix)}</span>
              <span className="lvl">{String(ev.severity || "info").toLowerCase()}</span>
              <span className="body">{String(ev.title || ev.kind || "")}</span>
              <span className="num muted">{eventMeta(ev)}</span>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}

// ── world state ───────────────────────────────────────────────────────────
function WorldState({ panel, snap }) {
  const ws = snap?.topics?.world_state;
  const so = snap?.topics?.scene_objects;
  const objects = (Array.isArray(so?.objects) ? so.objects : [])
    .filter((o) => o && typeof o === "object")
    .sort((a, b) => (b.last_seen_ns || 0) - (a.last_seen_ns || 0));
  const ee = ws?.ee_poses && typeof ws.ee_poses === "object" ? Object.entries(ws.ee_poses).slice(0, 3) : [];
  return (
    <Card panel={panel} ts={ws?.ts_unix}>
      {ws?.ts_unix == null ? (
        <Empty>waiting for world_state.snapshot</Empty>
      ) : (
        <>
          <KV
            rows={[
              ["components stale", ws.components_stale ?? "—"],
              ["latched error", ws.has_latched_error ? "YES" : "no", ws.has_latched_error ? "err" : undefined],
              ["battery", ws.battery_pct != null ? `${num(ws.battery_pct, 1)} %` : "—"],
              ...ee
                .filter(([, p]) => Array.isArray(p) && p.length >= 3)
                .map(([n, p]) => [`ee ${n}`, `[${num(p[0], 2)}, ${num(p[1], 2)}, ${num(p[2], 2)}]`]),
            ]}
          />
          {ws.diagnostics && typeof ws.diagnostics === "object" && (
            <dl className="kv pills">
              {Object.entries(ws.diagnostics).map(([k, v]) => {
                const sev = String(v || "").toLowerCase();
                return (
                  <div key={k}>
                    <dt>{k}</dt>
                    <dd>
                      <span className={`pill ${["ok", "stale", "warn"].includes(sev) ? sev : "error"}`}>{sev || "?"}</span>
                    </dd>
                  </div>
                );
              })}
            </dl>
          )}
        </>
      )}
      <p className="subhead">spatial memory · scene objects</p>
      {!so?.ts_unix ? (
        <Empty>waiting for world.scene_objects (preload a scene graph via spatial_memory_path, or run the perception object-lift producer)</Empty>
      ) : objects.length === 0 ? (
        <Empty>spatial memory is empty (0 objects remembered)</Empty>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">object</th>
                <th scope="col">position (m)</th>
                <th scope="col">conf</th>
                <th scope="col">seen</th>
                <th scope="col">obs</th>
              </tr>
            </thead>
            <tbody>
              {objects.map((o, i) => (
                <tr key={i}>
                  <td>
                    {String(o.label || o.id || "?")}
                    {o.is_container ? " (container)" : ""}
                  </td>
                  <td className="num">
                    ({num(o.x || 0, 2)}, {num(o.y || 0, 2)}, {num(o.z || 0, 2)})
                  </td>
                  <td className="num">{o.confidence != null ? num(o.confidence, 2) : "—"}</td>
                  <td className="num">{o.last_seen_ns ? fmtAge(o.last_seen_ns / 1e9) : "—"}</td>
                  <td className="num">{o.observation_count != null ? String(o.observation_count) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="meta">
            <span>
              {objects.length} object{s(objects.length)}
            </span>
            <span>frame {String(so.frame_id || "?")}</span>
            <span>source {String(so.source_node || "?")}</span>
          </p>
        </div>
      )}
    </Card>
  );
}

export const PANELS = {
  identity: Identity,
  camera: Camera,
  joints: Joints,
  map: MapPanel,
  pointcloud: Pointcloud,
  reasoner: Reasoner,
  rskill: Rskill,
  reward: Reward,
  safety_status: SafetyStatus,
  safety_ledger: Ledger,
  system: System,
  metrics: Metrics,
  events: Events,
  world_state: WorldState,
};
