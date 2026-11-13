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
// own /api/dashboard says who is signed in. On the robot there is no account.
function useMe(onWebsite) {
  const [me, setMe] = useState(null);
  useEffect(() => {
    if (!onWebsite) return;
    let live = true;
    fetch("/api/dashboard", { credentials: "same-origin" })
      .then((r) => (r.ok ? r.json() : null))
      .then((b) => live && setMe(b), () => {});
    return () => {
      live = false;
    };
  }, [onWebsite]);
  return me;
}

// The menu's Theme switch shows the mode in use: a moon in dark mode, a sun in light.
function ThemeIcon({ dark }) {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      {dark ? (
        <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      ) : (
        <g fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4" />
        </g>
      )}
    </svg>
  );
}

// robots: robotLinks() of the list the session already loaded (none on the robot).
export function Rail({ cfg, onWebsite, robotId, robots = [], robotName, reasonerModel, theme, toggleTheme }) {
  const [open, setOpen] = useOpen();
  const [menu, setMenu] = useState(false);
  const [settings, setSettings] = useState(false);
  const me = useMe(onWebsite);
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
              href={r.href}
              target="_top"
              aria-current={r.id === robotId ? "page" : undefined}
              title={r.name}
            >
              <span className="rail-robot-icon" aria-hidden="true">
                {r.initial}
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
              <button
                type="button"
                role="menuitem"
                className="rail-theme"
                aria-label={theme === "dark" ? "Theme: dark. Switch to light" : "Theme: light. Switch to dark"}
                onClick={toggleTheme}
              >
                Theme
                <ThemeIcon dark={theme === "dark"} />
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
