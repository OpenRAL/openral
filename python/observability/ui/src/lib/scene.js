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

// The 3D view's fixed frame. A mobile robot has a world frame (map, else odom)
// connected to its base: the view is pinned to it unless `follow` is set, so the
// robot drives through a still world. A fixed-base arm has none: its base frame.
// `world` says whether a world frame exists (the Follow button is shown only then).
export function viewFrame(tf, base, follow) {
  const world = ["map", "odom"].find((f) => f !== base && tf.lookup(f, base)) ?? null;
  return { fixed: follow || !world ? base : world, world };
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

// ── what the cameras have seen, kept in the world frame ──────────────────────

// Depth points placed in the world (map) frame as they arrive, one per `cell`-sized
// voxel (the newest point and colour win), oldest voxel dropped past `capacity`. The
// robot drives on and what its camera saw stays where it was: the warehouse builds up
// as a coloured cloud. Matrix math only; the caller decides when to draw.
export class WorldCloud {
  constructor({ cell = 0.05, capacity = 300000 } = {}) {
    this.cell = cell;
    this.capacity = capacity;
    this.cells = new Map(); // voxel key -> [x, y, z, r, g, b]
    this.version = 0;
  }

  // `pts` from cloudToPoints (sensor frame); `tf` the sensor frame in the world frame.
  add(pts, tf) {
    if (!pts || !tf || !pts.count) return 0;
    const [qx, qy, qz, qw] = tf.q;
    const [tx, ty, tz] = tf.t;
    const inv = 1 / this.cell;
    const { positions: p, colors: c } = pts;
    for (let i = 0; i < pts.count; i++) {
      const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
      // v' = q v q* + t (expanded: no allocation per point)
      const ix = qw * x + qy * z - qz * y;
      const iy = qw * y + qz * x - qx * z;
      const iz = qw * z + qx * y - qy * x;
      const iw = -qx * x - qy * y - qz * z;
      const wx = ix * qw + iw * -qx + iy * -qz - iz * -qy + tx;
      const wy = iy * qw + iw * -qy + iz * -qx - ix * -qz + ty;
      const wz = iz * qw + iw * -qz + ix * -qy - iy * -qx + tz;
      const key = `${Math.floor(wx * inv)},${Math.floor(wy * inv)},${Math.floor(wz * inv)}`;
      this.cells.delete(key); // re-insert: a voxel seen again is the newest
      this.cells.set(key, [wx, wy, wz, c ? c[i * 3] : -1, c ? c[i * 3 + 1] : -1, c ? c[i * 3 + 2] : -1]);
    }
    for (const key of this.cells.keys()) {
      if (this.cells.size <= this.capacity) break;
      this.cells.delete(key);
    }
    this.version += 1;
    return this.cells.size;
  }

  clear() {
    this.cells.clear();
    this.version += 1;
  }

  // { positions, colors } (colors null when no point carried one).
  arrays() {
    const n = this.cells.size;
    const positions = new Float32Array(n * 3);
    let colors = null;
    let k = 0;
    for (const v of this.cells.values()) {
      positions[k * 3] = v[0];
      positions[k * 3 + 1] = v[1];
      positions[k * 3 + 2] = v[2];
      if (v[3] >= 0) {
        colors ??= new Float32Array(n * 3).fill(0.6);
        colors[k * 3] = v[3];
        colors[k * 3 + 1] = v[4];
        colors[k * 3 + 2] = v[5];
      }
      k++;
    }
    return { count: n, positions, colors };
  }
}

// ── sensor_msgs/LaserScan → points in the scan frame ──────────────────────────

// Returns only real returns: inf/NaN (no return, REP-117) and readings outside
// [range_min, range_max] are dropped.
export function scanToPoints(msg) {
  const ranges = msg?.ranges;
  const a0 = Number(msg?.angle_min);
  const da = Number(msg?.angle_increment);
  if (!ranges || !Number.isFinite(a0) || !Number.isFinite(da)) return null;
  const lo = Number.isFinite(msg.range_min) ? msg.range_min : 0;
  const hi = Number.isFinite(msg.range_max) ? msg.range_max : Infinity;
  const positions = new Float32Array(ranges.length * 3);
  let k = 0;
  for (let i = 0; i < ranges.length; i++) {
    const r = ranges[i];
    if (!Number.isFinite(r) || r < lo || r > hi) continue;
    const a = a0 + i * da;
    positions[k * 3] = r * Math.cos(a);
    positions[k * 3 + 1] = r * Math.sin(a);
    k++;
  }
  return { count: k, positions: positions.subarray(0, k * 3), colors: null, frame: strip(msg.header?.frame_id) };
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
  // In priority order: the self-filtered RGB world cloud, octomap's voxel centres, the
  // voxel grid re-published as a cloud.
  cloud: ["/openral/world_cloud/self_filtered_rgb", "/octomap_point_cloud_centers", "/openral/world_voxels_cloud"],
  depth: [], // every depth camera's thinned cloud (CAMERA_POINTS), see layerTopics
  scan: ["/scan", "/openral/nav2/scan"],
  collision: ["/openral/viz/static_geometry", "/openral/viz/collisions"],
  payload: ["/openral/viz/attachments"],
};

// A depth camera's cloud as viewers get it: the robot's viz converter re-publishes
// `/openral/cameras/<name>/points` thinned to ~40k points on this topic. The raw cloud
// (2 M points, 33 MB a frame from a 1080p ZED) is never subscribed.
export const CAMERA_POINTS = /^\/openral\/viz\/cameras\/[^/]+\/points$/;

// The advertised topics a layer can draw, in its priority order.
export function layerTopics(layer, advertised) {
  if (layer === "depth") return advertised.filter((t) => CAMERA_POINTS.test(t)).sort();
  const have = new Set(advertised);
  return (LAYER_TOPICS[layer] || []).filter((t) => have.has(t));
}

// Layers that take every advertised topic.
const ALL_OF = new Set(["tf", "depth", "collision"]);

// Which advertised topics to subscribe to, given the layers that are on and the
// topics that have delivered a message (`live`). TF and joint states are always
// needed to place anything. `tf`, `depth` and `collision` take every topic; `cloud`
// takes its candidates down to the first live one (an advertised topic can be
// silent: the voxel re-publisher runs with octomap off), so a silent topic never
// hides one with data and two copies of the same world are not streamed; the others
// take their first advertised topic.
export function wantedTopics(advertised, layersOn, live = new Set()) {
  const want = new Set();
  const take = (layer) => {
    const topics = layerTopics(layer, advertised);
    if (ALL_OF.has(layer)) {
      for (const t of topics) want.add(t);
    } else if (layer === "cloud") {
      const first = topics.findIndex((t) => live.has(t));
      for (const t of first < 0 ? topics : topics.slice(0, first + 1)) want.add(t);
    } else if (topics.length) want.add(topics[0]);
  };
  take("tf");
  if (layersOn.robot) take("robot"), take("joints");
  for (const layer of ["grid", "cloud", "depth", "scan", "collision", "payload"]) if (layersOn[layer]) take(layer);
  return want;
}

// The cloud-layer topic to draw: the highest-priority one that has delivered.
export function cloudTopic(subscribed, live) {
  return LAYER_TOPICS.cloud.find((t) => subscribed.has(t) && live.has(t)) ?? null;
}
