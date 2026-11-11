// Console layout: which manifest panel goes where, and the operator's layout
// settings (visible tabs, their order, the default tab, 3D layers, theme).
// Pure: no DOM or React. Storage is passed in.

const SLOT = { left: 0, center: 1, right: 2 };

// Camera panels, left → center → right. A panel without a position (an older
// robot) sits in the middle; manifest order is kept inside a slot.
export function orderCameras(panels) {
  return (Array.isArray(panels) ? panels : [])
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => p && p.type === "camera" && typeof p.source === "string" && p.source)
    .sort((a, b) => (SLOT[a.p.position] ?? 1) - (SLOT[b.p.position] ?? 1) || a.i - b.i)
    .map(({ p }) => p);
}

// Cameras to put in the row: the ordered camera panels, minus the ones the operator
// hid, minus any that is not an operator view: a camera with no frame yet (a depth
// sensor whose image is never streamed, a camera that is not plugged in) or whose
// frames are depth. `cameras` is `snap.topics.perception.cameras`.
export function rowCameras(panels, cameras, hidden = []) {
  const live = cameras && typeof cameras === "object" ? cameras : {};
  return orderCameras(panels).filter((p) => {
    const cam = live[p.source];
    return !hidden.includes(p.source) && cam && cam.ts_unix != null && cam.modality !== "depth";
  });
}

// The right pane's width: draggable, kept within what leaves room for the scene.
export const PANE_WIDTH = { min: 320, max: 900, initial: 440 };
export const clampPaneWidth = (w) =>
  Number.isFinite(w) ? Math.round(Math.max(PANE_WIDTH.min, Math.min(PANE_WIDTH.max, w))) : PANE_WIDTH.initial;

// The right pane's tabs, in default order. Each lists the manifest panel types
// it draws; a tab shows only when the robot declares at least one of them.
export const TABS = [
  { id: "reasoner", label: "Reasoner", types: ["reasoner"] },
  { id: "events", label: "Events", types: ["events"] },
  { id: "world", label: "World", types: ["world_state"] },
  { id: "safety", label: "Safety", types: ["safety_status", "safety_ledger"] },
  { id: "skill", label: "Skill", types: ["rskill", "reward"] },
  { id: "robot", label: "Robot", types: ["joints", "pointcloud"] },
  { id: "system", label: "System", types: ["identity", "system", "metrics"] },
];

// Tabs the robot can fill: [{ ...tab, panels: [manifest panels in tab-type order] }].
export function availableTabs(panels) {
  const list = Array.isArray(panels) ? panels : [];
  return TABS.map((t) => ({
    ...t,
    panels: t.types.flatMap((type) => list.filter((p) => p && p.type === type)),
  })).filter((t) => t.panels.length > 0);
}

// 3D layers: id, label, and the topics each subscribes to (first one advertised wins
// for single-source layers; all advertised ones for multi-source layers).
export const LAYERS = [
  { id: "robot", label: "Robot model", on: true },
  { id: "grid", label: "Occupancy grid", on: true },
  { id: "objects", label: "World-state objects", on: true },
  { id: "cloud", label: "Point cloud", on: true },
  { id: "depth", label: "Camera depth", on: true },
  { id: "scan", label: "Lidar", on: true },
  { id: "collision", label: "Collision boxes", on: true },
  { id: "payload", label: "Payload", on: true },
  { id: "tf", label: "TF frames", on: false },
];

export const THEMES = ["system", "light", "dark"];

export const DEFAULT_PREFS = Object.freeze({
  order: TABS.map((t) => t.id),
  hidden: [],
  tab: "reasoner",
  layers: Object.fromEntries(LAYERS.map((l) => [l.id, l.on])),
  hiddenCameras: [],
  theme: "system",
  paneWidth: PANE_WIDTH.initial,
  // The 3D view's frame: false = the world (map, else odom) so the robot moves
  // through it; true = the robot's base, so the world moves around the robot.
  follow: false,
});

const ids = (v, known) => (Array.isArray(v) ? [...new Set(v.filter((x) => known.has(x)))] : []);

// Anything stored, from any version, becomes a valid prefs object.
export function normalizePrefs(raw) {
  const r = raw && typeof raw === "object" ? raw : {};
  const known = new Set(TABS.map((t) => t.id));
  const order = ids(r.order, known);
  for (const t of TABS) if (!order.includes(t.id)) order.push(t.id);
  const layers = { ...DEFAULT_PREFS.layers };
  if (r.layers && typeof r.layers === "object") {
    for (const id of Object.keys(layers)) if (typeof r.layers[id] === "boolean") layers[id] = r.layers[id];
  }
  return {
    order,
    hidden: ids(r.hidden, known),
    tab: known.has(r.tab) ? r.tab : DEFAULT_PREFS.tab,
    layers,
    hiddenCameras: Array.isArray(r.hiddenCameras) ? r.hiddenCameras.filter((x) => typeof x === "string") : [],
    theme: THEMES.includes(r.theme) ? r.theme : "system",
    paneWidth: clampPaneWidth(r.paneWidth),
    follow: r.follow === true,
  };
}

// Tabs to draw: available, not hidden, in the operator's order.
export function visibleTabs(prefs, panels) {
  const avail = new Map(availableTabs(panels).map((t) => [t.id, t]));
  return prefs.order.filter((id) => avail.has(id) && !prefs.hidden.includes(id)).map((id) => avail.get(id));
}

// The tab to show: the chosen one when visible, else the first visible one.
export function activeTab(prefs, tabs) {
  return tabs.find((t) => t.id === prefs.tab)?.id ?? tabs[0]?.id ?? null;
}

// Move a tab one step up (-1) or down (+1) among the tabs in `among` (the ones the
// operator can see in the list), skipping over any others in the stored order.
export function moveTab(prefs, id, step, among = prefs.order) {
  const shown = prefs.order.filter((x) => among.includes(x));
  const other = shown[shown.indexOf(id) + step];
  if (!shown.includes(id) || other === undefined) return prefs;
  const order = [...prefs.order];
  const i = order.indexOf(id);
  const j = order.indexOf(other);
  [order[i], order[j]] = [order[j], order[i]];
  return { ...prefs, order };
}

export const toggleIn = (list, id) => (list.includes(id) ? list.filter((x) => x !== id) : [...list, id]);

const key = (robotId) => `openral.console.${robotId || "robot"}`;

export function loadPrefs(storage, robotId) {
  try {
    return normalizePrefs(JSON.parse(storage?.getItem(key(robotId)) || "null"));
  } catch {
    return normalizePrefs(null);
  }
}

export function savePrefs(storage, robotId, prefs) {
  try {
    storage?.setItem(key(robotId), JSON.stringify(prefs));
  } catch {
    // private window or blocked storage: the layout just isn't remembered
  }
}

// "system" follows the OS; the page sets data-theme on <html>.
export function resolveTheme(theme, prefersDark) {
  if (theme === "light" || theme === "dark") return theme;
  return prefersDark ? "dark" : "light";
}
