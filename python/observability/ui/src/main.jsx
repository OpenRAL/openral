import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles/tokens.css";
import "./styles/app.css";
import "./styles/console.css";
import { createApi, createTokenClient, findRobot, websiteMinter } from "./lib/auth.js";
import { connectSSE } from "./lib/sse.js";
import { fmtAge, robotNow, setRobotNow } from "./lib/render.js";
import { ActionBar } from "./ActionBar.jsx";
import { Console, usePrefs, useTheme } from "./Console.jsx";

// Mode comes from the location: `?robot=<id>` means we are www.openral.com's
// /robot-ui/ (remote, through the robot's tunnel); otherwise the robot itself
// served this page (local or LAN, same origin, no token).
function useSession() {
  const [s, setS] = useState({ phase: "boot" });
  useEffect(() => {
    const id = new URLSearchParams(location.search).get("robot");
    (async () => {
      let api = createApi();
      let robot = null;
      if (id) {
        const r = await fetch("/api/robots", { credentials: "same-origin" });
        if (r.status === 401) throw new Error("Sign in to openral.com to see this robot.");
        if (!r.ok) throw new Error(`Couldn't load your robots from openral.com (HTTP ${r.status}).`);
        robot = findRobot(await r.json(), id);
        if (!robot) throw new Error(`"${id}" isn't one of your robots, or it has no valid https address.`);
        api = createApi({ base: robot.base, auth: createTokenClient({ mint: websiteMinter(robot.url) }) });
      }
      setS({ phase: "ready", api, robot });
    })().catch((e) => setS({ phase: "error", error: e.message }));
  }, []);
  return s;
}

// Poll a JSON route: every 5 s until it answers, then every 30 s. Keeps the
// last good body through later failures. `state`: loading | ok | waiting (404) | error.
function usePolledJson(api, path) {
  const [m, setM] = useState({ state: "loading", body: null });
  useEffect(() => {
    let stop = false;
    let t;
    const keep = (next) => setM((cur) => (cur.body ? cur : next));
    const load = async () => {
      let again = 5000;
      try {
        const r = await api.call(path);
        if (r.ok) {
          const body = await r.json();
          if (!stop) setM({ state: "ok", body });
          again = 30000;
        } else keep({ state: r.status === 404 ? "waiting" : "error", error: `HTTP ${r.status}`, body: null });
      } catch {
        keep({ state: "error", error: "robot unreachable", body: null });
      }
      if (!stop) t = setTimeout(load, again);
    };
    load();
    return () => {
      stop = true;
      clearTimeout(t);
    };
  }, [api, path]);
  return m;
}

// /api/state for first paint, then the SSE stream. Each message is a full
// snapshot; only the latest is parsed and rendered, at most 4×/s.
function useSnapshot(api, remote) {
  const [snap, setSnap] = useState(null);
  const [conn, setConn] = useState("connecting");
  useEffect(() => {
    let raw = null;
    let timer = null;
    const commit = (s) => {
      setRobotNow(s?.now_unix);
      setSnap(s);
    };
    api
      .call("/api/state")
      .then((r) => (r.ok ? r.json() : null))
      .then((s) => s && raw === null && commit(s))
      .catch(() => {});
    const stop = connectSSE({
      open: (signal) =>
        api.call(remote ? "/api/stream?images=0" : "/api/stream", { signal, headers: { Accept: "text/event-stream" } }),
      onStatus: setConn,
      onData: (d) => {
        raw = d;
        timer ??= setTimeout(() => {
          timer = null;
          try {
            commit(JSON.parse(raw));
          } catch {
            // malformed frame: keep the last good snapshot
          }
        }, 250);
      },
    });
    return () => {
      stop();
      clearTimeout(timer);
    };
  }, [api, remote]);
  return { snap, conn };
}

function useTick(ms) {
  const [, set] = useState(0);
  useEffect(() => {
    const t = setInterval(() => set((n) => n + 1), ms);
    return () => clearInterval(t);
  }, [ms]);
}

function Conn({ conn, snap }) {
  let st = "wait";
  let label = "connecting…";
  if (conn === "reconnecting") label = "reconnecting…";
  else if (conn === "live") {
    const ts = snap?.last_ingest_ts;
    const dt = ts ? robotNow() - ts : null;
    if (dt == null) label = "no telemetry yet";
    else if (dt < 10) [st, label] = ["live", "LIVE"];
    else [st, label] = ["paused", "telemetry stale"];
  }
  return (
    <span className="conn" role="status" title={snap?.last_ingest_ts ? `last telemetry ${fmtAge(snap.last_ingest_ts)} ago` : undefined}>
      <span className={`dot st-${st}`} aria-hidden="true" />
      {label}
    </span>
  );
}

function Header({ children }) {
  return (
    <div className="brand">
      <img src="openral-logo.svg" alt="" width="22" height="25" />
      <span className="brand-name">OpenRAL</span>
      {children}
    </div>
  );
}

// Always in the header: the safety kernel's latched state, from the stream.
function SafetyPill({ topics }) {
  const st = topics?.safety_status;
  const latched = !!(st?.latched || topics?.safety?.estopped);
  const known = st?.ts_unix != null || topics?.safety?.ts_unix != null;
  const label = !known ? "safety: waiting" : latched ? "E-STOP latched" : "safety nominal";
  return (
    <span className={`pill safety ${!known ? "" : latched ? "bad" : "ok"}`} role="status">
      {label}
    </span>
  );
}

const MODE_LABEL = { local: "local", lan: "LAN", remote: "remote" };

function Robot({ api, robot }) {
  // GET /api/config decides mode and role; until it answers, least privilege.
  const c = usePolledJson(api, "/api/config").body;
  const text = (...vs) => vs.find((v) => typeof v === "string" && v);
  const cfg = {
    ...c,
    mode: api.remote ? "remote" : text(c?.mode) || "local",
    role: text(c?.role, robot?.role) || "viewer",
    reachable: !!c,
  };
  // GET /api/manifest is 404 until the robot announces itself, and is re-read
  // so new rSkills (skill_registry_changed) show up.
  const m = usePolledJson(api, "/api/manifest");
  const { snap, conn } = useSnapshot(api, api.remote);
  useTick(1000);
  const manifest = m.body;
  const name = text(manifest?.name, manifest?.robot_id, robot?.name, c?.robot_id) || "robot";
  useEffect(() => {
    document.title = `${name} · OpenRAL`;
  }, [name]);
  const robotId = text(manifest?.robot_id, robot?.id, c?.robot_id) || "robot";
  const [prefs, update] = usePrefs(robotId);
  const theme = useTheme(prefs.theme);
  const [layoutOpen, setLayoutOpen] = useState(false);
  const t = snap?.topics || {};
  const bar = {
    api,
    cfg,
    actions: manifest?.actions,
    latched: !!(t.safety_status?.latched || t.safety?.estopped),
  };
  return (
    <>
      <header className="top">
        <Header>
          <span className="robot-name">{name}</span>
          <span className="badge">
            {MODE_LABEL[cfg.mode] || cfg.mode} · {cfg.role}
          </span>
          <SafetyPill topics={t} />
          <span className="spacer" />
          <Conn conn={conn} snap={snap} />
          <button
            type="button"
            className="chip layout-btn"
            aria-pressed={layoutOpen}
            onClick={() => setLayoutOpen((o) => !o)}
          >
            Layout
          </button>
        </Header>
        <ActionBar part="estop" {...bar} />
      </header>
      <main>
        {!cfg.reachable && (
          <p className="banner err" role="alert">
            Can't reach the robot's dashboard API. Retrying in the background.
          </p>
        )}
        {!manifest && (
          <div className="waiting">
            {m.state === "waiting" ? (
              <>
                <h1>Waiting for the robot to announce itself</h1>
                <p>
                  The robot publishes its manifest (panels and actions) when its runtime starts. Start a deploy, or check that
                  the runtime exports telemetry to this dashboard.
                </p>
              </>
            ) : m.state === "error" ? (
              <>
                <h1>Can't read the robot's manifest</h1>
                <p>{m.error}. Retrying every few seconds.</p>
              </>
            ) : (
              <h1>Loading the robot's manifest…</h1>
            )}
          </div>
        )}
        {manifest && (
          <Console
            api={api}
            cfg={cfg}
            manifest={manifest}
            snap={snap}
            bar={bar}
            robotId={robotId}
            prefs={prefs}
            update={update}
            theme={theme}
            layoutOpen={layoutOpen}
            setLayoutOpen={setLayoutOpen}
          />
        )}
      </main>
    </>
  );
}

function App() {
  const s = useSession();
  if (s.phase === "ready") return <Robot {...s} />;
  return (
    <header className="top">
      <Header />
      {s.phase === "error" ? (
        <p className="banner err" role="alert">
          {s.error}
        </p>
      ) : (
        <p className="banner">Connecting to the robot…</p>
      )}
    </header>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
