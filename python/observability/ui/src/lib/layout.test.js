import assert from "node:assert/strict";
import test from "node:test";
import {
  activeTab,
  availableTabs,
  DEFAULT_PREFS,
  loadPrefs,
  moveTab,
  normalizePrefs,
  orderCameras,
  resolveTheme,
  savePrefs,
  toggleIn,
  visibleTabs,
} from "./layout.js";

// The so101_follower manifest's panels, as build_robot_manifest emits them.
const SO101 = [
  { id: "identity", type: "identity", title: "Robot" },
  { id: "cam:top", type: "camera", title: "top", source: "top", position: "center" },
  { id: "cam:wrist", type: "camera", title: "wrist", source: "wrist", position: "left" },
  { id: "joints", type: "joints", title: "Joints" },
  { id: "reasoner", type: "reasoner", title: "Reasoner" },
  { id: "rskill", type: "rskill", title: "rSkill" },
  { id: "safety_status", type: "safety_status", title: "Safety status" },
  { id: "safety_ledger", type: "safety_ledger", title: "Safety ledger" },
  { id: "system", type: "system", title: "System" },
  { id: "metrics", type: "metrics", title: "Metrics" },
  { id: "events", type: "events", title: "Events" },
  { id: "world_state", type: "world_state", title: "World state" },
  { id: "viz3d", type: "viz3d", title: "3D" },
];

test("cameras go left → center → right, manifest order inside a slot", () => {
  const panels = [
    ...SO101,
    { id: "cam:r", type: "camera", source: "r", position: "right" },
    { id: "cam:old", type: "camera", source: "old" }, // older robot: no position
  ];
  assert.deepEqual(
    orderCameras(panels).map((p) => p.source),
    ["wrist", "top", "old", "r"],
  );
  assert.deepEqual(orderCameras(null), []);
});

test("tabs appear only for panel types the robot declares", () => {
  const ids = availableTabs(SO101).map((t) => t.id);
  assert.deepEqual(ids, ["reasoner", "events", "world", "safety", "skill", "robot", "system"]);
  const noReasoner = availableTabs(SO101.filter((p) => p.type !== "reasoner")).map((t) => t.id);
  assert.ok(!noReasoner.includes("reasoner"));
  const safety = availableTabs(SO101).find((t) => t.id === "safety");
  assert.deepEqual(
    safety.panels.map((p) => p.id),
    ["safety_status", "safety_ledger"],
  );
});

test("stored prefs from any version normalize to a valid object", () => {
  assert.deepEqual(normalizePrefs(null), { ...DEFAULT_PREFS, layers: { ...DEFAULT_PREFS.layers } });
  const p = normalizePrefs({ order: ["system", "bogus", "system"], hidden: ["events", 7], tab: "nope", theme: "pink", layers: { cloud: false, x: true } });
  assert.equal(p.order[0], "system");
  assert.equal(p.order.length, DEFAULT_PREFS.order.length);
  assert.deepEqual(p.hidden, ["events"]);
  assert.equal(p.tab, "reasoner");
  assert.equal(p.theme, "system");
  assert.equal(p.layers.cloud, false);
  assert.ok(!("x" in p.layers));
});

test("visible tabs follow order and hidden; the active one falls back", () => {
  let prefs = normalizePrefs({ hidden: ["reasoner"], tab: "reasoner" });
  prefs = moveTab(prefs, "system", -1);
  const tabs = visibleTabs(prefs, SO101);
  assert.equal(tabs[0].id, "events");
  assert.ok(tabs.findIndex((t) => t.id === "system") < tabs.findIndex((t) => t.id === "robot"));
  assert.equal(activeTab(prefs, tabs), "events");
  assert.equal(activeTab(prefs, []), null);
  assert.equal(moveTab(prefs, "reasoner", -1), prefs); // already first: unchanged
});

test("moving a tab steps over tabs the robot doesn't have", () => {
  const prefs = normalizePrefs(null); // reasoner, events, world, safety, …
  const shown = ["reasoner", "world", "safety"]; // this robot has no events tab
  const moved = moveTab(prefs, "world", -1, shown);
  assert.deepEqual(
    moved.order.filter((id) => shown.includes(id)),
    ["world", "reasoner", "safety"],
  );
  assert.equal(moveTab(prefs, "reasoner", -1, shown), prefs);
  assert.equal(moveTab(prefs, "safety", 1, shown), prefs);
  assert.equal(moveTab(prefs, "events", 1, shown), prefs); // not in the list
});

test("toggleIn adds and removes", () => {
  assert.deepEqual(toggleIn(["a"], "b"), ["a", "b"]);
  assert.deepEqual(toggleIn(["a", "b"], "a"), ["b"]);
});

test("prefs survive a round trip and broken storage", () => {
  const mem = new Map();
  const storage = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v) };
  const prefs = normalizePrefs({ tab: "events", theme: "dark" });
  savePrefs(storage, "so101_follower", prefs);
  assert.deepEqual(loadPrefs(storage, "so101_follower"), prefs);
  assert.deepEqual(loadPrefs(storage, "other").tab, "reasoner");
  const broken = {
    getItem: () => {
      throw new Error("blocked");
    },
    setItem: () => {
      throw new Error("blocked");
    },
  };
  assert.equal(loadPrefs(broken, "x").tab, "reasoner");
  savePrefs(broken, "x", prefs); // does not throw
});

test("system theme follows the OS", () => {
  assert.equal(resolveTheme("system", true), "dark");
  assert.equal(resolveTheme("system", false), "light");
  assert.equal(resolveTheme("light", true), "light");
});
