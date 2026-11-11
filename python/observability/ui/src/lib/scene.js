// Decoded ROS messages → plain data the 3D scene draws. Pure: no three.js, no
// DOM, so every conversion here runs under `node --test`.
//
// Message shapes are what @foxglove/rosmsg2-serialization decodes from CDR:
// field names as in the .msg files, arrays as typed arrays where the type allows.

// ── rigid transforms: { t: [x, y, z], q: [x, y, z, w] } ──────────────────────

export const IDENTITY = Object.freeze({ t: [0, 0, 0], q: [0, 0, 0, 1] });

function qmul(a, b) {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

const qconj = ([x, y, z, w]) => [-x, -y, -z, w];

export function rotate(q, v) {
  const r = qmul(qmul(q, [v[0], v[1], v[2], 0]), qconj(q));
  return [r[0], r[1], r[2]];
}

// a ∘ b: apply b, then a.
export function compose(a, b) {
  const t = rotate(a.q, b.t);
  return { t: [a.t[0] + t[0], a.t[1] + t[1], a.t[2] + t[2]], q: qmul(a.q, b.q) };
}

export function invert(a) {
  const q = qconj(a.q);
  const t = rotate(q, a.t);
  return { t: [-t[0], -t[1], -t[2]], q };
}

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : 0);

export function fromPose(pose) {
  const p = pose?.position ?? pose?.translation ?? {};
  const o = pose?.orientation ?? pose?.rotation ?? {};
  const q = [num(o.x), num(o.y), num(o.z), o.w == null ? 1 : num(o.w)];
  const n = Math.hypot(...q) || 1;
  return { t: [num(p.x), num(p.y), num(p.z)], q: q.map((c) => c / n) };
}

const strip = (f) => String(f ?? "").replace(/^\/+/, "");

// The TF tree from /tf and /tf_static. `lookup(target, source)` is the pose of
// `source` in `target`, or null when the two frames are not connected.
export class TfBuffer {
  constructor() {
    this.parent = new Map(); // child -> { parent, tf }
  }

  addMessage(msg) {
    for (const tr of Array.isArray(msg?.transforms) ? msg.transforms : []) {
      const child = strip(tr?.child_frame_id);
      const parent = strip(tr?.header?.frame_id);
      if (!child || !parent || child === parent) continue;
      this.parent.set(child, { parent, tf: fromPose(tr.transform) });
    }
  }

  frames() {
    const all = new Set(this.parent.keys());
    for (const { parent } of this.parent.values()) all.add(parent);
    return [...all].sort();
  }

  // Pose of `frame` in its tree's root, and the root's name.
  toRoot(frame) {
    let f = strip(frame);
    let tf = IDENTITY;
    const seen = new Set();
    while (this.parent.has(f) && !seen.has(f)) {
      seen.add(f);
      const e = this.parent.get(f);
      tf = compose(e.tf, tf);
      f = e.parent;
    }
    return { root: f, tf };
  }

  lookup(target, source) {
    const a = this.toRoot(target);
    const b = this.toRoot(source);
    if (!a.root || a.root !== b.root) return null;
    return compose(invert(a.tf), b.tf);
  }
}

// ── nav_msgs/OccupancyGrid → RGBA texture ─────────────────────────────────────

// colors: { free: [r,g,b,a], occupied: [r,g,b,a], unknown: [r,g,b,a] }, 0–255.
// Cell (0,0) is the grid's first cell; the texture keeps that order (row = y).
export function occupancyToRGBA(grid, colors) {
  const w = grid?.info?.width | 0;
  const h = grid?.info?.height | 0;
  const data = grid?.data;
  if (w <= 0 || h <= 0 || !data || data.length < w * h) return null;
  const out = new Uint8ClampedArray(w * h * 4);
  const { unknown, free, occupied } = colors;
  for (let i = 0, o = 0; i < w * h; i++, o += 4) {
    const v = data[i];
    if (v < 0) {
      out[o] = unknown[0];
      out[o + 1] = unknown[1];
      out[o + 2] = unknown[2];
      out[o + 3] = unknown[3];
      continue;
    }
    const k = Math.min(100, v) / 100;
    for (let j = 0; j < 4; j++) out[o + j] = free[j] + (occupied[j] - free[j]) * k;
  }
  return {
    width: w,
    height: h,
    rgba: out,
    resolution: num(grid.info.resolution) || 0.05,
    origin: fromPose(grid.info.origin),
    frame: strip(grid.header?.frame_id),
  };
}

// ── sensor_msgs/PointCloud2 → positions (+ colors), at most `budget` points ───

const FLOAT32 = 7;
const FLOAT64 = 8;
const UINT32 = 6;

export function cloudToPoints(msg, budget = 60000) {
  const fields = new Map((Array.isArray(msg?.fields) ? msg.fields : []).map((f) => [f.name, f]));
  const fx = fields.get("x");
  const fy = fields.get("y");
  const fz = fields.get("z");
  const step = msg?.point_step | 0;
  const data = msg?.data;
  if (!fx || !fy || !fz || step <= 0 || !data) return null;
  const n = Math.min((msg.width | 0) * Math.max(1, msg.height | 0), Math.floor(data.byteLength / step));
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const little = !msg.is_bigendian;
  const read = (f, off) =>
    f.datatype === FLOAT64 ? view.getFloat64(off + f.offset, little) : view.getFloat32(off + f.offset, little);
  const rgb = fields.get("rgb") ?? fields.get("rgba");
  const stride = Math.max(1, Math.ceil(n / budget));
  const cap = Math.ceil(n / stride);
  const positions = new Float32Array(cap * 3);
  const colors = rgb ? new Float32Array(cap * 3) : null;
  let k = 0;
  for (let i = 0; i < n; i += stride) {
    const off = i * step;
    const x = read(fx, off);
    const y = read(fy, off);
    const z = read(fz, off);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    positions.set([x, y, z], k * 3);
    if (colors) {
      // rgb is a packed 0x00RRGGBB, stored as float32 (PCL) or uint32.
      const packed =
        rgb.datatype === UINT32 || rgb.datatype === FLOAT32
          ? view.getUint32(off + rgb.offset, little)
          : 0xffffff;
      colors.set([((packed >> 16) & 255) / 255, ((packed >> 8) & 255) / 255, (packed & 255) / 255], k * 3);
    }
    k++;
  }
  return {
    count: k,
    positions: positions.subarray(0, k * 3),
    colors: colors ? colors.subarray(0, k * 3) : null,
    frame: strip(msg.header?.frame_id),
    decimated: stride > 1,
  };
}

// ── visualization_msgs/MarkerArray → primitives ───────────────────────────────

// visualization_msgs/Marker.type → what the scene draws. TRIANGLE_LIST (11) is how
// declared mesh fixtures arrive (points in threes, already in metres × scale).
const KIND = { 0: "arrow", 1: "cube", 2: "sphere", 3: "cylinder", 4: "line", 5: "lines", 6: "cubes", 7: "spheres", 8: "points", 9: "text", 10: "mesh", 11: "triangles" };
const DELETE = 2;
const DELETE_ALL = 3;

// Applies one MarkerArray to `store` (a Map keyed "ns/id") and returns it. Keep one
// store per topic: DELETEALL clears the whole store, and each producer opens its
// array with one. Marker lifetimes are not tracked: a producer re-publishes or deletes.
export function applyMarkers(store, msg) {
  for (const m of Array.isArray(msg?.markers) ? msg.markers : []) {
    const key = `${m?.ns ?? ""}/${m?.id ?? 0}`;
    if (m?.action === DELETE_ALL) {
      store.clear();
      continue;
    }
    if (m?.action === DELETE) {
      store.delete(key);
      continue;
    }
    const kind = KIND[m?.type];
    if (!kind) continue;
    const c = m.color ?? {};
    store.set(key, {
      key,
      kind,
      frame: strip(m.header?.frame_id),
      pose: fromPose(m.pose),
      scale: [num(m.scale?.x), num(m.scale?.y), num(m.scale?.z)],
      color: [num(c.r), num(c.g), num(c.b), c.a == null ? 1 : num(c.a)],
      points: Array.isArray(m.points) ? m.points.map((p) => [num(p.x), num(p.y), num(p.z)]) : [],
      text: typeof m.text === "string" ? m.text : "",
      mesh: typeof m.mesh_resource === "string" ? m.mesh_resource : "",
    });
  }
  return store;
}

// ── world-state objects (dashboard snapshot) → labelled points ────────────────

export function sceneObjects(so, nowNanos) {
  const list = Array.isArray(so?.objects) ? so.objects : [];
  return list
    .filter((o) => o && [o.x, o.y, o.z].every((v) => typeof v === "number" && Number.isFinite(v)))
    .map((o, i) => ({
      key: String(o.id ?? `${o.label ?? "object"}#${i}`),
      label: String(o.label ?? o.id ?? "object"),
      frame: strip(o.frame_id),
      t: [o.x, o.y, o.z],
      confidence: typeof o.confidence === "number" ? o.confidence : null,
      ageS: nowNanos && o.last_seen_ns ? Math.max(0, (nowNanos - o.last_seen_ns) / 1e9) : null,
      container: !!o.is_container,
    }));
}

// ── channel choice ────────────────────────────────────────────────────────────

// The topics each 3D layer reads, in preference order.
export const LAYER_TOPICS = {
  robot: ["/openral/robot_description", "/robot_description"],
  joints: ["/joint_states"],
  tf: ["/tf", "/tf_static"],
  grid: ["/map"],
  cloud: ["/openral/world_cloud/self_filtered_rgb", "/octomap_point_cloud_centers", "/openral/world_voxels_cloud"],
  collision: ["/openral/viz/static_geometry", "/openral/viz/collisions"],
  payload: ["/openral/viz/attachments"],
};

// Layers that take every advertised topic; the others take the first one.
const ALL_OF = new Set(["tf", "collision"]);

// Which advertised topics to subscribe to, given the layers that are on.
// TF and joint states are always needed to place anything.
export function wantedTopics(advertised, layersOn) {
  const have = new Set(advertised);
  const want = new Set();
  const take = (layer) => {
    const topics = LAYER_TOPICS[layer].filter((t) => have.has(t));
    for (const t of ALL_OF.has(layer) ? topics : topics.slice(0, 1)) want.add(t);
  };
  take("tf");
  if (layersOn.robot) take("robot"), take("joints");
  for (const layer of ["grid", "cloud", "collision", "payload"]) if (layersOn[layer]) take(layer);
  return want;
}
