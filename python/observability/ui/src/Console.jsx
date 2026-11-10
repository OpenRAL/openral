// The operator console: cameras (left → center → right) over the 3D scene, and a
// right pane that shows one tab at a time. The operator chooses which tabs show,
// their order, the default tab, the 3D layers and the theme (saved per robot in
// this browser). E-STOP lives in the header, outside anything they can hide.
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
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
  useEffect(() => {
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

function ReasonerTrace({ snap }) {
  const [trace, setTrace] = useState([]);
  const tick = snap?.topics?.reasoner;
  useEffect(() => {
    if (tick?.ts_unix) setTrace((h) => traceAppend(h, tick));
  }, [tick]);
  if (trace.length === 0) return null;
  return (
    <section className="trace" aria-label="Reasoner trace">
      <h3 className="pane-sub">Trace · this session</h3>
      <ol>
        {trace.map((e) => (
          <li key={`${e.tick_idx ?? ""}-${e.ts_unix}`} className={`trace-row k-${e.kind}`}>
            <span className="trace-t">{fmtTime(e.ts_unix)}</span>
            <span className="trace-dot" aria-hidden="true" />
            <span className="trace-body">
              <span className="trace-kind">{KIND_LABEL[e.kind] || e.kind}</span>
              <span className="trace-text">
                {e.error_kind
                  ? `${e.tool ? `${e.tool} → ` : ""}${e.error_kind}`
                  : e.rskill_id
                    ? `${e.tool || "ExecuteRskill"} · ${shortId(e.rskill_id)}`
                    : e.tool || e.suppressed_reason || "no tool this tick"}
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
                  onClick={() => update((p) => moveTab(p, id, -1))}
                >
                  ↑
                </button>
                <button
                  type="button"
                  className="icon-btn"
                  aria-label={`Move ${tab.label} down`}
                  disabled={i === list.length - 1}
                  onClick={() => update((p) => moveTab(p, id, 1))}
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

function RightPane({ tabs, active, setActive, snap, api, cfg, manifest, bar, bridgeOn }) {
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
        {tab?.id === "reasoner" && <ReasonerTrace snap={snap} />}
        {tab?.panels.map((p) => {
          const P = PANELS[p.type];
          return (
            <Boundary key={p.id} title={p.title} reset={snap}>
              <P panel={p} snap={snap} api={api} cfg={cfg} manifest={manifest} />
            </Boundary>
          );
        })}
        {tab?.id === "system" && bridgeOn && !api.remote && (
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

export function Console({ api, cfg, manifest, snap, bar, robotId, prefs, update, theme, layoutOpen, setLayoutOpen }) {
  const panels = useMemo(
    () => (Array.isArray(manifest?.panels) ? manifest.panels : []).filter((p) => p && typeof p.id === "string" && Object.hasOwn(PANELS, p.type)),
    [manifest],
  );
  const allCams = orderCameras(panels);
  const cams = allCams.filter((c) => !prefs.hiddenCameras.includes(c.source));
  const tabsAvail = availableTabs(panels);
  const tabs = visibleTabs(prefs, panels);
  const [chosen, setChosen] = useState(null);
  const active = tabs.some((t) => t.id === chosen) ? chosen : activeTab(prefs, tabs);
  const bridgeOn = panels.some((p) => p.type === "viz3d");
  const setLayer = (id) => update((p) => ({ ...p, layers: { ...p.layers, [id]: !p.layers[id] } }));
  return (
    <div className="console">
      <div className="stage">
        <CameraStrip cams={cams} snap={snap} api={api} cfg={cfg} />
        <Suspense fallback={<section className="scene" aria-label="3D scene"><p className="scene-status">loading the 3D view…</p></section>}>
          <Scene3D
          key={robotId}
          api={api}
          manifest={manifest}
          snap={snap}
          layers={prefs.layers}
          setLayer={setLayer}
          theme={theme}
          bridgeOn={bridgeOn}
          />
        </Suspense>
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
        />
      )}
    </div>
  );
}
