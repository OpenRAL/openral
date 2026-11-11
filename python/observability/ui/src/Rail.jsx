import { useEffect, useRef, useState } from "react";
import { viewerLabel } from "./lib/viewer.js";

// The left pane: the OpenRAL logo at the top, the sections (none yet) and, at the
// bottom, the viewer's profile, which opens Settings (and the website account).
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

export function Rail({ cfg, onWebsite, onSettings }) {
  const [open, setOpen] = useOpen();
  const [menu, setMenu] = useState(false);
  const me = useMe(onWebsite);
  const v = viewerLabel(me, cfg);
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
    <aside className={open ? "rail open" : "rail"} aria-label="OpenRAL">
      <div className="rail-top">
        <img className="rail-logo" src="openral-logo.svg" alt="OpenRAL" width="24" height="27" />
        {open && <span className="rail-brand">OpenRAL</span>}
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
      <nav className="rail-nav" aria-label="Sections" />
      <div className="rail-profile" ref={box}>
        {menu && (
          <div className="rail-menu" role="menu">
            <button
              type="button"
              role="menuitem"
              disabled={!onSettings}
              title={onSettings ? undefined : "Settings open once the robot has announced itself."}
              onClick={() => {
                setMenu(false);
                onSettings?.();
              }}
            >
              Settings
            </button>
            {onWebsite && (
              <a role="menuitem" href="/account" target="_top">
                Account settings
              </a>
            )}
          </div>
        )}
        <button
          type="button"
          className="rail-me"
          aria-haspopup="menu"
          aria-expanded={menu}
          title={`${v.name} · ${v.detail}`}
          onClick={() => setMenu(!menu)}
        >
          <span className="rail-avatar" aria-hidden="true">
            {v.initials}
          </span>
          {open && (
            <span className="rail-who">
              <span className="rail-name">{v.name}</span>
              <span className="rail-detail">{v.detail}</span>
            </span>
          )}
          {!open && <span className="sr-only">{v.name}</span>}
        </button>
      </div>
    </aside>
  );
}
