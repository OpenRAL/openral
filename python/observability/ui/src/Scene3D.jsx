// The console's 3D scene, drawn in the page with three.js from the robot's
// foxglove_bridge (no Lichtblick). Layers subscribe only while they are on; the
// world-state objects come from the dashboard stream the page already reads.
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ColladaLoader } from "three/examples/jsm/loaders/ColladaLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import URDFLoader from "urdf-loader";
import { renewDelayS } from "./lib/auth.js";
import { Bridge } from "./lib/bridge.js";
import { LAYERS } from "./lib/layout.js";
import { robotNow } from "./lib/render.js";
import {
  applyMarkers,
  cloudToPoints,
  compose,
  LAYER_TOPICS,
  occupancyToRGBA,
  sceneObjects,
  TfBuffer,
  wantedTopics,
} from "./lib/scene.js";
import { backoffMs } from "./lib/sse.js";
import { localBridgeUrl, remoteBridgeUrl } from "./lib/viz.js";

const COLLISION = new Set(LAYER_TOPICS.collision);
const PAYLOAD = new Set(LAYER_TOPICS.payload);
const ROBOT = new Set(LAYER_TOPICS.robot);

// Theme colours for the scene, read from CSS custom properties on <html>.
function sceneColors() {
  const css = getComputedStyle(document.documentElement);
  const get = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
  const rgba = (hex, a) => {
    const c = new THREE.Color(hex);
    return [c.r * 255, c.g * 255, c.b * 255, a];
  };
  return {
    bg: get("--scene-bg", "#121419"),
    grid: get("--scene-grid", "#2c313b"),
    accent: get("--scene-accent", "#8fb3ff"),
    label: get("--text", "#e8e9ed"),
    occupancy: {
      unknown: rgba(get("--scene-bg", "#121419"), 0),
      free: rgba(get("--map-free", "#1c1f26"), 200),
      occupied: rgba(get("--map-occupied", "#e8e9ed"), 235),
    },
  };
}

const placeAt = (obj, tf) => {
  obj.position.set(tf.t[0], tf.t[1], tf.t[2]);
  obj.quaternion.set(tf.q[0], tf.q[1], tf.q[2], tf.q[3]);
};

// One marker primitive → a three.js object (or null for kinds not drawn).
function markerObject(m) {
  const color = new THREE.Color(m.color[0], m.color[1], m.color[2]);
  const opacity = Math.max(0.15, Math.min(1, m.color[3]));
  const material = new THREE.MeshStandardMaterial({ color, transparent: opacity < 1, opacity, roughness: 0.8 });
  const [sx, sy, sz] = m.scale.map((v) => Math.max(v, 1e-3));
  let obj = null;
  // A MESH_RESOURCE marker is not drawn: a box of its scale would misstate its shape.
  if (m.kind === "cube") obj = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), material);
  else if (m.kind === "triangles" && m.points.length >= 3) {
    // TRIANGLE_LIST: a declared mesh fixture, points in threes.
    const g = new THREE.BufferGeometry().setFromPoints(m.points.map((p) => new THREE.Vector3(p[0] * sx, p[1] * sy, p[2] * sz)));
    g.computeVertexNormals();
    material.side = THREE.DoubleSide;
    obj = new THREE.Mesh(g, material);
  }
  else if (m.kind === "sphere") obj = new THREE.Mesh(new THREE.SphereGeometry(0.5, 20, 14), material), obj.scale.set(sx, sy, sz);
  else if (m.kind === "cylinder") {
    obj = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 1, 24), material);
    obj.geometry.rotateX(Math.PI / 2); // ROS cylinders run along z
    obj.scale.set(sx, sy, sz);
  } else if ((m.kind === "line" || m.kind === "lines") && m.points.length > 1) {
    const g = new THREE.BufferGeometry().setFromPoints(m.points.map((p) => new THREE.Vector3(...p)));
    const mat = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
    obj = m.kind === "line" ? new THREE.Line(g, mat) : new THREE.LineSegments(g, mat);
  } else if (["points", "cubes", "spheres"].includes(m.kind) && m.points.length) {
    const g = new THREE.BufferGeometry().setFromPoints(m.points.map((p) => new THREE.Vector3(...p)));
    obj = new THREE.Points(g, new THREE.PointsMaterial({ color, size: Math.max(sx, 0.01) }));
  }
  if (obj && m.kind === "cube") {
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(obj.geometry),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: Math.min(1, opacity + 0.4) }),
    );
    obj.add(edges);
  }
  return obj;
}

function disposeTree(obj) {
  obj.traverse((o) => {
    o.geometry?.dispose?.();
    const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
    for (const m of mats) {
      m.map?.dispose?.();
      m.dispose?.();
    }
  });
}

// Robot-supplied URDF and Collada files can name texture images by URL. The page
// never fetches them: every URL a loader asks for becomes this 1×1 transparent GIF.
const NO_TEXTURE = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
function noFetchManager() {
  const manager = new THREE.LoadingManager();
  manager.setURLModifier(() => NO_TEXTURE);
  return manager;
}

// Mesh loading for urdf-loader: package:// through the bridge's assets capability
// (`getBytes`, cached per URI so a mesh used by several links downloads once);
// anything else cannot be reached from the browser. `track` counts meshes in
// flight: track(+1) when one starts, track(-1, ok) when it ends.
function meshLoader(getBytes, track) {
  return (path, _manager, material, done) => {
    track(1);
    const finish = (obj, err) => {
      done(obj, err);
      track(-1, !err);
    };
    if (!path.startsWith("package://")) return finish(null, new Error(`mesh not reachable: ${path}`));
    getBytes(path)
      .then((bytes) => {
        const ext = path.split(".").pop().toLowerCase();
        if (ext === "stl") {
          const g = new STLLoader().parse(bytes.slice().buffer);
          return finish(new THREE.Mesh(g, material));
        }
        const text = new TextDecoder().decode(bytes);
        if (ext === "dae") return finish(new ColladaLoader(noFetchManager()).parse(text, "").scene);
        if (ext === "obj") return finish(new OBJLoader().parse(text));
        throw new Error(`mesh type .${ext} not supported`);
      })
      .catch((e) => finish(null, e));
  };
}

// Pose every URDF link that has a TF frame from /tf (what robot_state_publisher
// computed, in URDF names and units). /joint_states cannot do this in general: a HAL
// may publish manifest joint names and units that differ from the URDF's (a 0–1
// gripper fraction, other joint names). Returns how many links were posed.
const _m = new THREE.Matrix4();
const _inv = new THREE.Matrix4();
const _p = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _one = new THREE.Vector3(1, 1, 1);
function poseLinksFromTf(robot, at) {
  let posed = 0;
  robot.traverse((o) => {
    if (!o.isURDFLink || o === robot || !o.parent) return;
    const tf = at(o.urdfName);
    if (!tf) return;
    _m.compose(_p.set(tf.t[0], tf.t[1], tf.t[2]), _q.set(tf.q[0], tf.q[1], tf.q[2], tf.q[3]), _one);
    o.parent.updateWorldMatrix(true, false); // ancestors were posed first (pre-order)
    _m.premultiply(_inv.copy(o.parent.matrixWorld).invert());
    _m.decompose(o.position, o.quaternion, o.scale);
    posed++;
  });
  return posed;
}

// The robot as plain shapes when its meshes are unreachable: a sphere per link
// origin and a rod from each joint to the next.
function stickFigure(robot, color) {
  const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.5 });
  for (const joint of Object.values(robot.joints)) {
    const child = joint.children.find((c) => c.isURDFLink);
    if (!child) continue;
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12), mat);
    child.add(ball);
    const length = joint.position.length();
    if (length > 1e-4 && joint.parent) {
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, length, 12), mat);
      rod.position.copy(joint.position).multiplyScalar(0.5);
      rod.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), joint.position.clone().normalize());
      joint.parent.add(rod);
    }
  }
}

// Connect to the bridge. On the robot itself (loopback) /ws needs nothing; from the
// LAN and through the tunnel the robot wants a viz ticket (`needTicket`).
// Reconnects with backoff, and before each ticket expires.
function useBridge(api, needTicket, handlers) {
  const ref = useRef(null);
  const [status, setStatus] = useState({ state: "connecting", detail: null });
  useEffect(() => {
    if (!api) return undefined;
    let stop = false;
    let timer = null;
    let attempt = 0;
    const open = async () => {
      if (stop) return;
      let url;
      let renewMs = null;
      try {
        if (!needTicket) url = localBridgeUrl(window.location);
        else {
          const r = await api.call("/api/viz/ticket", { method: "POST" });
          if (!r.ok) throw new Error(`the robot refused a 3D ticket (HTTP ${r.status})`);
          const j = await r.json();
          if (typeof j?.ticket !== "string" || !j.ticket) throw new Error("the robot sent no 3D ticket");
          url = api.remote ? remoteBridgeUrl(api.base, j.ticket) : localBridgeUrl(window.location, j.ticket);
          renewMs = renewDelayS(j.expires_in) * 1000;
        }
      } catch (e) {
        setStatus({ state: "closed", detail: e.message });
        timer = setTimeout(open, backoffMs(attempt++));
        return;
      }
      if (stop) return;
      const prev = ref.current;
      const b = new Bridge({
        url,
        onEvent: (t, m) => handlers.current.onEvent(t, m),
        onTopics: (t) => handlers.current.onTopics(t),
        onStatus: (state, detail) => {
          if (ref.current !== b) return;
          setStatus({ state, detail: detail ?? null });
          if (state === "open") {
            attempt = 0;
            handlers.current.onOpen(b);
            if (renewMs) {
              clearTimeout(timer);
              timer = setTimeout(open, renewMs); // fresh ticket before this one expires
            }
          }
          if (state === "closed" && !stop) {
            clearTimeout(timer);
            timer = setTimeout(open, backoffMs(attempt++));
          }
        },
      });
      ref.current = b;
      if (!prev) setStatus({ state: "connecting", detail: null });
      prev?.close();
    };
    open();
    return () => {
      stop = true;
      clearTimeout(timer);
      ref.current?.close();
      ref.current = null;
    };
  }, [api, needTicket]);
  return [ref, status];
}

export function Scene3D({ api, mode, manifest, snap, layers, setLayer, theme, bridgeOn }) {
  const host = useRef(null);
  const labels = useRef(null);
  const world = useRef(null); // three.js state, created once
  const [topics, setTopics] = useState([]);
  const [info, setInfo] = useState({ robot: "waiting", meshesMissing: false });
  const fixed = String(manifest?.base_frame || "base");
  const fixedRef = useRef(fixed);
  if (fixedRef.current !== fixed) {
    fixedRef.current = fixed;
    if (world.current) world.current.replace = true;
  }
  const layersRef = useRef(layers);
  layersRef.current = layers;
  const snapRef = useRef(snap);
  snapRef.current = snap;

  const handlers = useRef(null);
  handlers.current = {
    onOpen: (b) => {
      // A new connection redelivers the latched URDF: let it reload the meshes that
      // failed last time (a dropped socket or a slow link mid-download).
      const w = world.current;
      if (w?.meshesMissing) w.urdf = null;
      b.want(wantedTopics(b.topics(), layersRef.current));
    },
    onTopics: (t) => {
      setTopics(t);
      bridgeRef.current?.want(wantedTopics(t, layersRef.current));
    },
    onEvent: (topic, msg) => {
      const w = world.current;
      if (!w) return;
      w.dirty = true;
      if (topic === "/tf" || topic === "/tf_static") w.tf.addMessage(msg), (w.tfVersion += 1);
      else if (COLLISION.has(topic)) markersFrom(w.collision, topic, msg);
      else if (PAYLOAD.has(topic)) markersFrom(w.payload, topic, msg);
      else if (ROBOT.has(topic) && typeof msg?.data === "string") loadRobot(msg.data);
    },
  };
  const [bridgeRef, status] = useBridge(bridgeOn ? api : null, api.remote || mode !== "local", handlers);

  // One marker store per topic: each producer opens its array with DELETEALL, which
  // must not wipe another topic's markers in the same layer.
  const markersFrom = (layer, topic, msg) => {
    if (!layer.stores.has(topic)) layer.stores.set(topic, new Map());
    applyMarkers(layer.stores.get(topic), msg);
    layer.dirty = true;
  };

  // Mesh bytes by URI, shared by every link that uses the mesh and kept across
  // reloads; a failed download is forgotten so the next load tries it again.
  const meshBytes = (uri) => {
    const w = world.current;
    const bridge = bridgeRef.current;
    if (!w || !bridge) return Promise.reject(new Error("no 3D stream"));
    if (!w.meshCache.has(uri)) {
      const p = bridge.fetchAsset(uri).then((view) => new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
      p.catch(() => w.meshCache.get(uri) === p && w.meshCache.delete(uri));
      w.meshCache.set(uri, p);
    }
    return w.meshCache.get(uri);
  };

  const loadRobot = (urdf) => {
    const w = world.current;
    if (!w || w.urdf === urdf) return;
    w.urdf = urdf;
    let missing = false;
    let inFlight = 0;
    let parsed = false;
    let robot;
    const settle = () => {
      if (!parsed || inFlight > 0 || w.robot !== robot) return;
      if (missing && !robot.userData.stickAdded) {
        robot.userData.stickAdded = true;
        stickFigure(robot, w.colors.accent);
      }
      w.meshesMissing = missing;
      w.fitPending = true; // frame the robot once its links are posed
      setInfo({ robot: "ok", meshesMissing: missing });
    };
    const loader = new URDFLoader(noFetchManager());
    loader.packages = (pkg) => `package://${pkg}`;
    loader.loadMeshCb = meshLoader(meshBytes, (step, ok) => {
      inFlight += step;
      w.dirty = true;
      if (step < 0 && !ok) missing = true;
      if (step < 0) settle();
    });
    try {
      robot = loader.parse(urdf);
    } catch (e) {
      setInfo({ robot: `URDF did not parse: ${e.message}`, meshesMissing: false });
      return;
    }
    parsed = true;
    if (w.robot) {
      w.layers.robot.remove(w.robot);
      disposeTree(w.robot);
    }
    w.robot = robot;
    w.meshesMissing = false;
    w.replace = true;
    w.layers.robot.add(robot);
    setInfo({ robot: "loading", meshesMissing: false });
    settle();
  };

  // ── three.js setup, once ──
  useEffect(() => {
    const el = host.current;
    const colors = sceneColors();
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(colors.bg);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
    camera.up.set(0, 0, 1); // ROS: z up
    camera.position.set(0.55, -0.55, 0.45);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.12);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x888888, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(1, -1, 2);
    scene.add(sun);
    const grid = new THREE.GridHelper(2, 40, colors.grid, colors.grid);
    grid.rotation.x = Math.PI / 2;
    grid.material.transparent = true;
    grid.material.opacity = 0.5;
    scene.add(grid);
    const layerGroups = Object.fromEntries(LAYERS.map((l) => [l.id, new THREE.Group()]));
    for (const g of Object.values(layerGroups)) scene.add(g);
    const w = {
      renderer,
      scene,
      camera,
      controls,
      grid,
      colors,
      layers: layerGroups,
      tf: new TfBuffer(),
      robot: null,
      urdf: null,
      collision: { stores: new Map(), dirty: false },
      payload: { stores: new Map(), dirty: false },
      meshCache: new Map(),
      meshesMissing: false,
      tfVersion: 0,
      placedTf: -1,
      view: { c: [0, 0, 0.12], r: 0.3 },
      fitPending: false,
      replace: true, // something was (re)drawn: place it even if TF did not move
      map: null,
      cloud: null,
      axes: new Map(),
      objects: new Map(),
      dirty: true,
    };
    world.current = w;

    const resize = () => {
      const { clientWidth: cw, clientHeight: ch } = el;
      if (!cw || !ch) return;
      renderer.setSize(cw, ch, false);
      camera.aspect = cw / ch;
      camera.updateProjectionMatrix();
      w.dirty = true;
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    let raf = 0;
    // Draw only when something changed: new data, a camera move, a resize.
    controls.addEventListener("change", () => (w.dirty = true));
    const frame = () => {
      raf = requestAnimationFrame(frame);
      if (document.hidden) return;
      update(w);
      controls.update();
      if (!w.dirty) return;
      w.dirty = false;
      renderer.render(scene, camera);
      projectLabels(w);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      disposeTree(scene);
      renderer.dispose();
      renderer.domElement.remove();
      world.current = null;
    };
  }, []);

  // ── theme ──
  useEffect(() => {
    const w = world.current;
    if (!w) return;
    w.colors = sceneColors();
    w.scene.background = new THREE.Color(w.colors.bg);
    w.grid.material.color = new THREE.Color(w.colors.grid);
    w.mapStale = true;
    w.replace = true;
    w.dirty = true;
  }, [theme]);

  // ── layers on/off: visibility + subscriptions ──
  useEffect(() => {
    const w = world.current;
    if (w) {
      for (const [id, g] of Object.entries(w.layers)) g.visible = !!layers[id];
      w.replace = true;
      w.dirty = true;
    }
    bridgeRef.current?.want(wantedTopics(bridgeRef.current.topics(), layers));
  }, [layers, bridgeRef]);

  // Per frame: decode the newest snapshot messages; re-place things in the fixed
  // frame only when TF moved or something was redrawn.
  function update(w) {
    const b = bridgeRef.current;
    const L = layersRef.current;
    const at = (frame) => (frame ? w.tf.lookup(fixedRef.current, frame) : null);
    const js = b && L.robot ? b.take(LAYER_TOPICS.joints[0]) : null;
    if (b && L.grid) {
      const topic = LAYER_TOPICS.grid.find((t) => b.subs.has(t));
      const msg = topic && b.take(topic);
      if (msg) (w.mapMsg = msg), (w.mapStale = true);
      if (w.mapStale && w.mapMsg) drawMap(w), (w.replace = true);
    }
    if (b && L.cloud) {
      const topic = LAYER_TOPICS.cloud.find((t) => b.subs.has(t));
      const msg = topic && b.take(topic);
      if (msg) drawCloud(w, msg), (w.replace = true);
    }
    for (const id of ["collision", "payload"]) if (w[id].dirty) drawMarkers(w, id), (w.replace = true);

    const place = w.replace || w.placedTf !== w.tfVersion;
    if (place) {
      w.replace = false;
      w.placedTf = w.tfVersion;
      w.dirty = true;
      if (w.robot) {
        const root = at(w.robot.urdfName); // the URDF's root link
        if (root) placeAt(w.robot, root);
        w.robotFromTf = poseLinksFromTf(w.robot, at) > 0;
      }
      if (w.map) {
        const tf = at(w.map.userData.frame);
        w.map.visible = !!tf;
        if (tf) placeAt(w.map, compose(tf, w.map.userData.origin));
      }
      if (w.cloud) {
        const tf = at(w.cloud.userData.frame);
        w.cloud.visible = !!tf;
        if (tf) placeAt(w.cloud, tf);
      }
      for (const id of ["collision", "payload"]) {
        for (const obj of w.layers[id].children) {
          const tf = at(obj.userData.frame);
          obj.visible = !!tf;
          if (tf) placeAt(obj, compose(tf, obj.userData.pose));
        }
      }
      if (L.tf) drawAxes(w, at);
    }
    if (w.fitPending && w.robot) {
      w.fitPending = false;
      fitView(w);
    }
    // No TF for the links (no robot_state_publisher): fall back to joint states,
    // which is right only where the HAL publishes URDF joint names and units.
    if (js && w.robot && !w.robotFromTf && Array.isArray(js.name)) {
      js.name.forEach((n, i) => {
        const v = js.position?.[i];
        if (typeof v === "number" && Object.hasOwn(w.robot.joints, n)) w.robot.setJointValue(n, v);
      });
      w.dirty = true;
    }
    if (L.objects) drawObjects(w, at);
  }

  // Frame the robot: a desk arm and a bimanual torso differ by several times in size,
  // so the view presets are scaled to the robot's bounding sphere.
  function fitView(w) {
    w.robot.updateWorldMatrix(true, true);
    const sphere = new THREE.Box3().setFromObject(w.robot).getBoundingSphere(new THREE.Sphere());
    if (!Number.isFinite(sphere.radius) || sphere.radius <= 0) return;
    w.view = { c: sphere.center.toArray(), r: Math.max(0.15, sphere.radius) };
    setView(w, "perspective");
  }

  function setView(w, name) {
    const { c, r } = w.view;
    const d = r * 2.4;
    const eye = { perspective: [0.62 * d, -0.62 * d, 0.48 * d], top: [0, -0.001, 1.2 * d], side: [0, -d, 0.15 * d] }[name];
    w.camera.position.set(c[0] + eye[0], c[1] + eye[1], c[2] + eye[2]);
    w.controls.target.set(c[0], c[1], c[2]);
    w.dirty = true;
  }

  function drawMap(w) {
    w.mapStale = false;
    const tex = occupancyToRGBA(w.mapMsg, w.colors.occupancy);
    if (!tex) return;
    if (w.map) w.layers.grid.remove(w.map), disposeTree(w.map);
    const texture = new THREE.DataTexture(tex.rgba, tex.width, tex.height, THREE.RGBAFormat);
    texture.magFilter = THREE.NearestFilter;
    texture.needsUpdate = true;
    const sx = tex.width * tex.resolution;
    const sy = tex.height * tex.resolution;
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(sx, sy).translate(sx / 2, sy / 2, -0.001),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false }),
    );
    const g = new THREE.Group();
    g.add(plane);
    g.userData = { frame: tex.frame, origin: tex.origin };
    w.map = g;
    w.layers.grid.add(g);
  }

  function drawCloud(w, msg) {
    const pts = cloudToPoints(msg);
    if (!pts) return;
    if (w.cloud) w.layers.cloud.remove(w.cloud), disposeTree(w.cloud);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pts.positions, 3));
    if (pts.colors) g.setAttribute("color", new THREE.BufferAttribute(pts.colors, 3));
    const material = new THREE.PointsMaterial({
      size: 0.01,
      vertexColors: !!pts.colors,
      color: pts.colors ? 0xffffff : new THREE.Color(w.colors.accent),
    });
    const points = new THREE.Points(g, material);
    points.userData.frame = pts.frame;
    w.cloud = points;
    w.layers.cloud.add(points);
  }

  function drawMarkers(w, id) {
    const st = w[id];
    st.dirty = false;
    const group = w.layers[id];
    for (const c of [...group.children]) group.remove(c), disposeTree(c);
    for (const store of st.stores.values()) {
      for (const m of store.values()) {
        const obj = markerObject(m);
        if (!obj) continue;
        obj.userData = { frame: m.frame, pose: m.pose };
        group.add(obj);
      }
    }
  }

  function drawAxes(w, at) {
    for (const f of w.tf.frames()) {
      let a = w.axes.get(f);
      if (!a) {
        a = new THREE.AxesHelper(0.05);
        w.axes.set(f, a);
        w.layers.tf.add(a);
      }
      const tf = at(f);
      a.visible = !!tf;
      if (tf) placeAt(a, tf);
    }
  }

  function drawObjects(w, at) {
    const so = snapRef.current?.topics?.scene_objects;
    if (so === w.lastObjects && w.tfVersion === w.lastTf) return;
    w.lastObjects = so;
    w.lastTf = w.tfVersion;
    w.dirty = true;
    const list = sceneObjects(so, robotNow() * 1e9);
    const seen = new Set();
    for (const o of list) {
      seen.add(o.key);
      let entry = w.objects.get(o.key);
      if (!entry) {
        const mesh = new THREE.Mesh(
          new THREE.SphereGeometry(0.012, 16, 12),
          new THREE.MeshStandardMaterial({ color: new THREE.Color(w.colors.accent), transparent: true }),
        );
        w.layers.objects.add(mesh);
        entry = { mesh, label: null };
        w.objects.set(o.key, entry);
      }
      const base = o.frame ? at(o.frame) : { t: [0, 0, 0], q: [0, 0, 0, 1] };
      entry.mesh.visible = !!base;
      if (base) placeAt(entry.mesh, compose(base, { t: o.t, q: [0, 0, 0, 1] }));
      entry.mesh.material.opacity = o.ageS != null && o.ageS > 3 ? 0.35 : 1;
      entry.text = `${o.label}${o.confidence != null ? ` ${o.confidence.toFixed(2)}` : ""}${
        o.ageS != null && o.ageS > 3 ? ` · ${Math.round(o.ageS)} s old` : ""
      }`;
      entry.stale = o.ageS != null && o.ageS > 3;
    }
    for (const [k, e] of w.objects) {
      if (seen.has(k)) continue;
      w.layers.objects.remove(e.mesh);
      disposeTree(e.mesh);
      e.label?.remove();
      w.objects.delete(k);
    }
  }

  // HTML labels for objects, positioned over the canvas each frame (text only).
  function projectLabels(w) {
    const box = labels.current;
    if (!box) return;
    const show = layersRef.current.objects;
    const { clientWidth: cw, clientHeight: ch } = box;
    const v = new THREE.Vector3();
    for (const e of w.objects.values()) {
      if (!e.label) {
        e.label = document.createElement("span");
        e.label.className = "scene-label";
        box.appendChild(e.label);
      }
      e.mesh.getWorldPosition(v);
      v.project(w.camera);
      const visible = show && e.mesh.visible && v.z < 1 && Math.abs(v.x) < 1 && Math.abs(v.y) < 1;
      e.label.hidden = !visible;
      if (!visible) continue;
      if (e.label.textContent !== e.text) e.label.textContent = e.text;
      e.label.classList.toggle("stale", !!e.stale);
      e.label.style.transform = `translate(${((v.x + 1) / 2) * cw + 10}px, ${((1 - v.y) / 2) * ch - 10}px)`;
    }
  }

  const available = (id) => {
    if (id === "objects") return !!snap?.topics?.scene_objects;
    const list = id === "tf" ? LAYER_TOPICS.tf : LAYER_TOPICS[id] || [];
    return list.some((t) => topics.includes(t));
  };
  const view = (name) => () => world.current && setView(world.current, name);

  return (
    <section className="scene" aria-label="3D scene">
      <div className="scene-canvas" ref={host} />
      <div className="scene-labels" ref={labels} aria-hidden="true" />
      <div className="scene-layers glass" role="group" aria-label="Layers">
        <span className="glass-hd">Layers</span>
        {LAYERS.map((l) => (
          <label key={l.id} className={available(l.id) ? "layer" : "layer off"}>
            <input type="checkbox" checked={!!layers[l.id]} onChange={() => setLayer(l.id)} />
            <span>{l.label}</span>
            {!available(l.id) && <span className="layer-na">no data</span>}
          </label>
        ))}
      </div>
      <div className="scene-views">
        <button type="button" className="glass-btn" onClick={view("perspective")}>
          Perspective
        </button>
        <button type="button" className="glass-btn" onClick={view("top")}>
          Top
        </button>
        <button type="button" className="glass-btn" onClick={view("side")}>
          Side
        </button>
      </div>
      <p className="scene-status" role="status">
        {!bridgeOn
          ? "3D needs foxglove_bridge on the robot (deploy without --no-foxglove)."
          : status.state === "open"
            ? `fixed frame ${fixed} · ${topics.length} topics${info.meshesMissing ? " · meshes unavailable, drawing links" : ""}${
                info.robot !== "ok" && info.robot !== "waiting" && info.robot !== "loading" ? ` · ${info.robot}` : ""
              }`
            : status.state === "connecting"
              ? "connecting to the robot's 3D stream…"
              : `3D stream ${status.detail || "closed"} · retrying`}
      </p>
    </section>
  );
}
