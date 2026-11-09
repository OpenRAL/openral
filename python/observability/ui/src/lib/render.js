// Pure render math ported from the vanilla dashboard (dashboard.js): formats,
// map/overlay geometry, sparklines, severity buckets. No DOM.

// Ages are measured on the robot's clock (snapshot.now_unix), so a remote
// viewer whose clock is off still reads true ages.
let skew = 0;
export function setRobotNow(nowUnix) {
  if (Number.isFinite(nowUnix)) skew = nowUnix - Date.now() / 1000;
}
export const robotNow = () => Date.now() / 1000 + skew;

export const num = (v, d = 2) => (v == null || !Number.isFinite(Number(v)) ? "—" : Number(v).toFixed(d));
export const fmtMs = (ms) => (ms == null || !Number.isFinite(Number(ms)) ? "—" : `${Number(ms).toFixed(1)} ms`);
export const fmtTime = (ts) => (ts ? new Date(ts * 1000).toTimeString().slice(0, 8) : "");
export const shortId = (s) => String(s).split("/").pop();
export const str = (v, n) => (v == null || v === "" ? null : n ? String(v).slice(0, n) : String(v));

export function fmtAge(ts) {
  if (!ts) return "—";
  const dt = robotNow() - ts;
  if (dt < 1) return "now";
  if (dt < 60) return `${dt.toFixed(1)}s`;
  if (dt < 3600) return `${Math.floor(dt / 60)}m`;
  return `${Math.floor(dt / 3600)}h`;
}

export function fmtNum(v) {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 1e5)) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  return v.toFixed(3);
}

export function b64Blob(b64, type) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Blob([u8], { type });
}

// ── SLAM map ──────────────────────────────────────────────────────────────
// The bridge rasterises the OccupancyGrid with a vertical flip (PIL top-left
// vs grid bottom-left), so pixel-y is mirrored.
export function worldToPixel(wx, wy, originX, originY, resolution, height) {
  return { px: (wx - originX) / resolution, py: height - 1 - (wy - originY) / resolution };
}

// Base-frame footprint vertices → map pixels: rotate by yaw, translate to the
// robot pose, then worldToPixel.
export function footprintToPixels(polygon, rx, ry, yaw, originX, originY, resolution, height) {
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return polygon.map(([bx, by]) =>
    worldToPixel(rx + bx * c - by * s, ry + bx * s + by * c, originX, originY, resolution, height),
  );
}

// ── camera overlays ───────────────────────────────────────────────────────
// Instance colours for detector boxes / segmenter masks: the one place the UI
// uses chroma outside status, because identity over a photograph can't be
// carried by a grey ramp. Every mark is also labelled, so colour is never
// the only cue.
export const OVERLAY_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const OVERLAY_FADE_S = 1.5; // past this the overlay dims — it may be lying
const OVERLAY_DROP_S = 4.0; // past this it is cleared
const OVERLAY_SKEW_S = 2.0; // cleared when the tile's frame is this much newer

// FNV-1a: a label keeps its colour across frames and reloads.
export function overlayColor(key) {
  let h = 0x811c9dc5;
  const s = String(key);
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return OVERLAY_COLORS[h % OVERLAY_COLORS.length];
}

// Readable ink for a label chip filled with `hex`.
export function overlayInk(hex) {
  const c = [1, 3, 5].map((i) => {
    const v = parseInt(hex.slice(i, i + 2), 16) / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] > 0.18 ? "#0e0f13" : "#ffffff";
}

// Opacity for an overlay lane given its age and how far the camera frame has
// moved past it; 0 = don't draw (a stale overlay is cleared, never left lying).
export function overlayFade(lane, now, camTs) {
  const ts = lane && lane.ts_unix;
  if (!ts) return 0;
  const age = now - ts;
  if (age > OVERLAY_DROP_S) return 0;
  if (camTs && camTs - ts > OVERLAY_SKEW_S) return 0;
  if (age <= OVERLAY_FADE_S) return 1;
  return 1 - (age - OVERLAY_FADE_S) / (OVERLAY_DROP_S - OVERLAY_FADE_S);
}

// Map normalised source coords onto a tile drawn with `object-fit: cover`
// (which crops): without this boxes sit at a constant, believable offset.
export function coverMap(nw, nh, w, h, fallbackAspect) {
  const aspect = nw && nh ? nw / nh : fallbackAspect || 1;
  const dw = Math.max(w, h * aspect);
  const dh = Math.max(h, w / aspect);
  return { ox: (w - dw) / 2, oy: (h - dh) / 2, dw, dh };
}

// ── reasoner / reward ─────────────────────────────────────────────────────
export const MISSION_MARK = { pending: "·", active: "▶", verifying: "?", done: "✓", abandoned: "✗" };

// The verdict text the reasoner stamps is the source of truth.
export function rewardBand(task) {
  if (task.status === "abandoned") return "fail";
  const v = String(task.verdict || "");
  if (v.startsWith("success=")) return "ok";
  if (v.startsWith("ambiguous=")) return "ambiguous";
  return "fail";
}
export const rewardValueBand = (v) => (v >= 0.7 ? "ok" : v >= 0.4 ? "ambiguous" : "fail");

// ── events ────────────────────────────────────────────────────────────────
export const sevBucket = (sev) => {
  const s = String(sev || "info").toLowerCase();
  return s === "debug" || s === "info" || s === "warn" ? s : "error";
};

export function eventMeta(ev) {
  const a = ev && ev.attrs;
  if (!a) return "";
  const dur = a.duration_ms ?? a["openral.duration_ms"];
  if (dur != null && Number.isFinite(Number(dur))) return `${Number(dur).toFixed(1)} ms`;
  const tick = a["openral.tick.idx"] ?? a["tick.idx"];
  return tick != null ? `tick ${tick}` : "";
}

// ── metrics ───────────────────────────────────────────────────────────────
const KNOWN_NS = ["system", "world_state", "hal", "rskill", "reasoner", "safety", "perception"];
export function metricNamespace(name) {
  const parts = String(name || "").split(".");
  if (parts[0] === "openral" && parts.length > 1) return KNOWN_NS.includes(parts[1]) ? parts[1] : "sdk";
  return parts[0] || "other";
}

const PREFIXES = ["openral.world_state.", "openral.system.", "otel.sdk.", "openral."];
export function stripMetricPrefix(n) {
  const p = PREFIXES.find((x) => n.startsWith(x));
  return p ? n.slice(p.length) : n;
}

// "{component=hal}" so per-label series don't read as duplicates.
export function labelSuffix(labels) {
  if (!labels || typeof labels !== "object") return "";
  const e = Object.entries(labels)
    .map(([k, v]) => [k.split(".").pop(), v])
    .filter(([k]) => k !== "gpu");
  return e.length ? ` {${e.map(([k, v]) => `${k}=${v}`).join(", ")}}` : "";
}

// Sparkline geometry. X rides the shared window `win` so every row lines up;
// Y autoscales per series (widened to include the threshold). `breach` is the
// latest visible sample past the threshold (`dir` "lower" = a floor).
export function sparkGeometry(samples, win, threshold, dir, w = 240, h = 26) {
  if (!Array.isArray(samples) || samples.length < 2) return null;
  const tMin = win ? win.tMin : samples[0][0];
  const tMax = win ? win.tMax : samples[samples.length - 1][0];
  const visible = samples.filter((s) => s[0] >= tMin && s[0] <= tMax);
  const values = (visible.length ? visible : samples).map((s) => s[1]);
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  const thr = typeof threshold === "number" ? threshold : null;
  if (thr != null) {
    lo = Math.min(lo, thr);
    hi = Math.max(hi, thr);
  }
  const span = hi - lo || 1;
  const tSpan = tMax - tMin || 1;
  const x = (t) => ((t - tMin) / tSpan) * w;
  const y = (v) => h - 2 - ((v - lo) / span) * (h - 4);
  const last = values[values.length - 1];
  return {
    points: samples.map((s) => `${x(s[0]).toFixed(1)},${y(s[1]).toFixed(1)}`).join(" "),
    x,
    y,
    lo,
    hi,
    w,
    h,
    tMin,
    tMax,
    thrY: thr == null ? null : y(thr),
    breach: thr != null && (dir === "lower" ? last < thr : last > thr),
  };
}
