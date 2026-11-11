// The operator console: cameras (left → center → right) over the 3D scene, and a
// right pane that shows one tab at a time. The operator chooses which tabs show,
// their order, the default tab, the 3D layers and the theme (saved per robot in
// this browser). E-STOP lives in the header, outside anything they can hide.
import { Component, lazy, Suspense, useEffect, useLayoutEffect, useMemo, useState } from "react";
import { ActionBar } from "./ActionBar.jsx";
import {
  activeTab,
  loadPrefs,
  moveTab,
  orderCameras,
  resolveTheme,
  savePrefs,
  TABS,
  THEMES,
  toggleIn,
  availableTabs,
  visibleTabs,
} from "./lib/layout.js";
import { fmtTime, shortId } from "./lib/render.js";
import { traceAppend } from "./lib/trace.js";
import { localVizUrl } from "./lib/viz.js";
import { Boundary, PANELS } from "./panels.jsx";

// three.js is most of the bundle: load it after the header (and E-STOP) are up.
const Scene3D = lazy(() => import("./Scene3D.jsx").then((m) => ({ default: m.Scene3D })));

const storage = () => {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
};

export function usePrefs(robotId) {
  const [prefs, setPrefs] = useState(() => loadPrefs(storage(), robotId));
  useEffect(() => setPrefs(loadPrefs(storage(), robotId)), [robotId]);
  const update = (fn) =>
    setPrefs((cur) => {
      const next = fn(cur);
      savePrefs(storage(), robotId, next);
      return next;
    });
  return [prefs, update];
}

// Sets data-theme on <html>; "system" follows the OS and its changes.
export function useTheme(theme) {
  const [dark, setDark] = useState(() => window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return undefined;
    const on = (e) => setDark(e.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  const resolved = resolveTheme(theme, dark);
  // A layout effect: it must be on <html> before the 3D scene's own effect reads the
  // theme's CSS variables (a child's effects run before its parent's).
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = resolved;
  }, [resolved]);
  return resolved;
}

const KIND_LABEL = {
  act: "Act",
  plan: "Plan",
  perceive: "Perceive",
  prompt: "Prompt",
  lifecycle: "Lifecycle",
  wait: "Wait",
  think: "Think",
  error: "Error",
};

// The stream carries only the latest tick: keep the ones seen this session. Lives in
// Console, which stays mounted, so the history survives a tab change or the Layout sheet.
function useTrace(snap) {
  const [trace, setTrace] = useState([]);
  const tick = snap?.topics?.reasoner;
  useEffect(() => {
    if (tick?.ts_unix) setTrace((h) => traceAppend(h, tick));
  }, [tick]);
  return trace;
}

// Robot data is rendered as text only; a non-string (a span attribute can be a list
// or a map) must not reach React as a child.
const text = (v) => (v == null ? "" : typeof v === "string" ? v : JSON.stringify(v));

function ReasonerTrace({ trace }) {
  if (trace.length === 0) return null;
  return (
    <section className="trace" aria-label="Reasoner trace">
      <h3 className="pane-sub">Trace · this session</h3>
      <ol>
        {trace.map((e) => (
          <li key={`${text(e.tick_idx)}-${e.ts_unix}`} className={`trace-row k-${e.kind}`}>
            <span className="trace-t">{fmtTime(e.ts_unix)}</span>
            <span className="trace-dot" aria-hidden="true" />
            <span className="trace-body">
              <span className="trace-kind">{KIND_LABEL[e.kind] || e.kind}</span>
              <span className="trace-text">
                {e.error_kind
                  ? `${e.tool ? `${text(e.tool)} → ` : ""}${text(e.error_kind)}`
                  : e.rskill_id
                    ? `${text(e.tool) || "ExecuteRskill"} · ${shortId(text(e.rskill_id))}`
                    : text(e.tool) || text(e.suppressed_reason) || "no tool this tick"}
              </span>
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

function CameraStrip({ cams, snap, api, cfg }) {
  if (cams.length === 0) return null;
  const Camera = PANELS.camera;
  return (
    <section className="cams" aria-label="Cameras" data-count={cams.length}>
      {cams.map((p) => (
        <div className="cam-slot" key={p.id}>
          <Boundary title={p.title} reset={snap}>
            <Camera panel={p} snap={snap} api={api} cfg={cfg} />
          </Boundary>
        </div>
      ))}
    </section>
  );
}

function LayoutSheet({ prefs, update, tabsAvail, cams, onClose }) {
  const avail = new Set(tabsAvail.map((t) => t.id));
  return (
    <section className="sheet" aria-label="Layout settings">
      <header className="sheet-hd">
        <h2>Layout</h2>
        <span className="muted">saved in this browser for this robot</span>
        <button type="button" className="chip" onClick={onClose}>
          done
        </button>
      </header>
      <h3 className="pane-sub">Right pane</h3>
      <ul className="sheet-list">
        {prefs.order
          .filter((id) => avail.has(id))
          .map((id, i, list) => {
            const tab = TABS.find((t) => t.id === id);
            const shown = !prefs.hidden.includes(id);
            return (
              <li key={id}>
                <label>
                  <input
                    type="checkbox"
                    checked={shown}
                    onChange={() => update((p) => ({ ...p, hidden: toggleIn(p.hidden, id) }))}
                  />
                  {tab.label}
                </label>
                <label className="sheet-default">
                  <input
                    type="radio"
                    name="default-tab"
                    checked={prefs.tab === id}
                    disabled={!shown}
                    onChange={() => update((p) => ({ ...p, tab: id }))}
                  />
                  default
                </label>
                <button
                  type="button"
                  className="icon-btn"
                  aria-label={`Move ${tab.label} up`}
                  disabled={i === 0}
                  onClick={() => update((p) => moveTab(p, id, -1, [...avail]))}
                >
                  ↑
                </button>
                <button
                  type="button"
                  className="icon-btn"
                  aria-label={`Move ${tab.label} down`}
                  disabled={i === list.length - 1}
                  onClick={() => update((p) => moveTab(p, id, 1, [...avail]))}
                >
                  ↓
                </button>
              </li>
            );
          })}
      </ul>
      {cams.length > 0 && (
        <>
          <h3 className="pane-sub">Cameras</h3>
          <ul className="sheet-list">
            {cams.map((c) => (
              <li key={c.id}>
                <label>
                  <input
                    type="checkbox"
                    checked={!prefs.hiddenCameras.includes(c.source)}
                    onChange={() => update((p) => ({ ...p, hiddenCameras: toggleIn(p.hiddenCameras, c.source) }))}
                  />
                  {String(c.title || c.source)}
                </label>
                <span className="muted">{String(c.position || "center")}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      <h3 className="pane-sub">Theme</h3>
      <div className="seg" role="radiogroup" aria-label="Theme">
        {THEMES.map((t) => (
          <label key={t} className={prefs.theme === t ? "on" : undefined}>
            <input type="radio" name="theme" checked={prefs.theme === t} onChange={() => update((p) => ({ ...p, theme: t }))} />
            {t}
          </label>
        ))}
      </div>
    </section>
  );
}

function RightPane({ tabs, active, setActive, snap, api, cfg, manifest, bar, bridgeOn, trace }) {
  const tab = tabs.find((t) => t.id === active);
  return (
    <aside className="pane" aria-label="Details">
      <nav className="tabs" role="tablist" aria-label="Panes">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={t.id === active}
            aria-controls="pane-body"
            className={t.id === active ? "tab on" : "tab"}
            onClick={() => setActive(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <div className="pane-body" id="pane-body" role="tabpanel" aria-labelledby={tab ? `tab-${tab.id}` : undefined}>
        {!tab && <p className="empty">Every pane is hidden. Open Layout to show one.</p>}
        {tab?.id === "reasoner" && (
          <Boundary title="Reasoner trace" reset={trace}>
            <ReasonerTrace trace={trace} />
          </Boundary>
        )}
        {tab?.panels.map((p) => {
          const P = PANELS[p.type];
          return (
            <Boundary key={p.id} title={p.title} reset={snap}>
              <P panel={p} snap={snap} api={api} cfg={cfg} manifest={manifest} />
            </Boundary>
          );
        })}
        {tab?.id === "system" && bridgeOn && cfg.mode === "local" && (
          <p className="meta">
            <a href={localVizUrl(window.location)} target="_blank" rel="noopener noreferrer">
              Open the raw topics in Foxglove (Lichtblick)
            </a>
          </p>
        )}
      </div>
      <footer className="pane-actions">
        <Boundary title="Actions" reset={manifest}>
          <ActionBar part="rest" {...bar} />
        </Boundary>
      </footer>
    </aside>
  );
}

const SceneFallback = ({ children }) => (
  <section className="scene" aria-label="3D scene">
    <p className="scene-status" role="status">
      {children}
    </p>
  </section>
);

// The 3D scene must never take the page (and the E-STOP above it) down: no WebGL, a
// chunk that fails to load, or a message it can't draw all end here.
class SceneBoundary extends Component {
  state = { err: null };
  static getDerivedStateFromError(err) {
    return { err };
  }
  render() {
    if (!this.state.err) return this.props.children;
    return (
      <SceneFallback>
        The 3D view is unavailable in this browser ({String(this.state.err?.message || this.state.err)}). Everything else
        keeps working; reload to try again.
      </SceneFallback>
    );
  }
}

export function Console({ api, cfg, manifest, snap, bar, robotId, prefs, update, theme, layoutOpen, setLayoutOpen }) {
  const declared = useMemo(() => (Array.isArray(manifest?.panels) ? manifest.panels : []).filter((p) => p && typeof p.id === "string"), [manifest]);
  const panels = useMemo(() => declared.filter((p) => Object.hasOwn(PANELS, p.type)), [declared]);
  const allCams = orderCameras(panels);
  const cams = allCams.filter((c) => !prefs.hiddenCameras.includes(c.source));
  const tabsAvail = availableTabs(panels);
  const tabs = visibleTabs(prefs, panels);
  const [chosen, setChosen] = useState(null);
  const active = tabs.some((t) => t.id === chosen) ? chosen : activeTab(prefs, tabs);
  // `viz3d` in the manifest means foxglove_bridge runs: the 3D scene has data.
  const bridgeOn = declared.some((p) => p.type === "viz3d");
  const trace = useTrace(snap);
  const setLayer = (id) => update((p) => ({ ...p, layers: { ...p.layers, [id]: !p.layers[id] } }));
  return (
    <div className="console">
      <div className="stage">
        <CameraStrip cams={cams} snap={snap} api={api} cfg={cfg} />
        <SceneBoundary key={robotId}>
          <Suspense fallback={<SceneFallback>loading the 3D view…</SceneFallback>}>
            <Scene3D
              api={api}
              mode={cfg.mode}
              manifest={manifest}
              snap={snap}
              layers={prefs.layers}
              setLayer={setLayer}
              theme={theme}
              bridgeOn={bridgeOn}
            />
          </Suspense>
        </SceneBoundary>
      </div>
      {layoutOpen ? (
        <aside className="pane">
          <LayoutSheet prefs={prefs} update={update} tabsAvail={tabsAvail} cams={allCams} onClose={() => setLayoutOpen(false)} />
        </aside>
      ) : (
        <RightPane
          tabs={tabs}
          active={active}
          setActive={setChosen}
          snap={snap}
          api={api}
          cfg={cfg}
          manifest={manifest}
          bar={bar}
          bridgeOn={bridgeOn}
          trace={trace}
        />
      )}
    </div>
  );
}
