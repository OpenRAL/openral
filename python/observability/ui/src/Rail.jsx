import { useEffect, useRef, useState } from "react";
import { viewerLabel, workspaceLabel } from "./lib/viewer.js";
import { SettingsPane } from "./SettingsPane.jsx";

// The left pane: whose workspace this is at the top (the org's logo or first letter,
// and its name), then on the website the robots this account reaches (a click opens
// that one on the dashboard) and, at the bottom, the viewer with a ⋯ menu
// that opens the settings pane (and, on the website, the account page).
// It starts closed (an icon rail) and remembers open/closed in this browser.
const KEY = "openral.rail.open";

function useOpen() {
  const [open, setOpen] = useState(() => {
    try {
      return localStorage.getItem(KEY) === "1";
    } catch {
      return false;
    }
  });
  const set = (v) => {
    setOpen(v);
    try {
      localStorage.setItem(KEY, v ? "1" : "0");
    } catch {
      // blocked storage: the pane just isn't remembered
    }
  };
  return [open, set];
}

// On the website the console is served same-origin under /robot-ui/, so the site's
// own /api/dashboard says who is signed in and /api/robots which robots they reach.
// On the robot there is no account.
function useSite(onWebsite, path) {
  const [body, setBody] = useState(null);
  useEffect(() => {
    if (!onWebsite) return;
    let live = true;
    fetch(path, { credentials: "same-origin" })
      .then((r) => (r.ok ? r.json() : null))
      .then((b) => live && setBody(b), () => {});
    return () => {
      live = false;
    };
  }, [onWebsite, path]);
  return body;
}

export function Rail({ cfg, onWebsite, robotId, robotName, reasonerModel }) {
  const [open, setOpen] = useOpen();
  const [menu, setMenu] = useState(false);
  const [settings, setSettings] = useState(false);
  const me = useSite(onWebsite, "/api/dashboard");
  const robots = useSite(onWebsite, "/api/robots")?.robots || [];
  const v = viewerLabel(me, cfg);
  const ws = workspaceLabel(me, cfg, robotName);
  const box = useRef(null);

  useEffect(() => {
    if (!menu) return;
    const close = (e) => {
      if (e.type === "keydown" ? e.key === "Escape" : !box.current?.contains(e.target)) setMenu(false);
    };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", close);
    };
  }, [menu]);

  return (
    <>
      <aside className={open ? "rail open" : "rail"} aria-label={ws.name}>
        <div className="rail-top">
          {ws.logo ? (
            <img className="rail-org" src={ws.logo} alt="" width="28" height="28" />
          ) : (
            <span className="rail-org" aria-hidden="true">
              {ws.initial}
            </span>
          )}
          {open ? <span className="rail-brand">{ws.name}</span> : <span className="sr-only">{ws.name}</span>}
          <button
            type="button"
            className="rail-toggle"
            aria-expanded={open}
            aria-label={open ? "Close the side pane" : "Open the side pane"}
            title={open ? "Close" : "Open"}
            onClick={() => setOpen(!open)}
          >
            {open ? "‹" : "›"}
          </button>
        </div>
        <nav className="rail-nav" aria-label="Robots">
          {robots.map((r) => (
            // The whole dashboard page (target _top) switches, so its heading names the same robot.
            <a
              key={r.id}
              className="rail-robot"
              href={`/dashboard?robot=${encodeURIComponent(r.id)}#robots`}
              target="_top"
              aria-current={r.id === robotId ? "page" : undefined}
              title={r.name}
            >
              <span className="rail-robot-icon" aria-hidden="true">
                {(r.name || "?")[0].toUpperCase()}
              </span>
              {open ? <span className="rail-robot-name">{r.name}</span> : <span className="sr-only">{r.name}</span>}
            </a>
          ))}
        </nav>
        <div className="rail-profile" ref={box}>
          {menu && (
            <div className="rail-menu" role="menu" aria-label={v.name}>
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setMenu(false);
                  setSettings(true);
                }}
              >
                Settings
              </button>
              {onWebsite && (
                <a role="menuitem" href="/account" target="_top">
                  Account
                </a>
              )}
            </div>
          )}
          <div className="rail-me" title={`${v.name} · ${v.detail}`}>
            <button
              type="button"
              className="rail-avatar"
              aria-label={`${v.name}: menu`}
              aria-haspopup="menu"
              aria-expanded={menu}
              onClick={() => setMenu(!menu)}
            >
              {v.initials}
            </button>
            {open && (
              <span className="rail-who">
                <span className="rail-name">{v.name}</span>
                <span className="rail-detail">{v.detail}</span>
              </span>
            )}
            {open && (
              <button
                type="button"
                className="rail-more"
                aria-label="More"
                aria-haspopup="menu"
                aria-expanded={menu}
                onClick={() => setMenu(!menu)}
              >
                ⋯
              </button>
            )}
          </div>
        </div>
      </aside>
      {settings && (
        <SettingsPane onWebsite={onWebsite} reasonerModel={reasonerModel} onClose={() => setSettings(false)} />
      )}
    </>
  );
}
