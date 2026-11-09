# OpenRAL robot UI

The web view of one robot: cameras, joints, map, reasoner, safety, metrics,
events, the 3D scene, and the robot's actions (E-STOP first). It renders only
what the robot declares in its manifest (`GET /api/manifest`), so the same build
works for any embodiment. Decided in ADR-0129 (private management repo); the
robot API it talks to is that ADR's Interface A, and the remote token flow is
Interface B.

It is a Vite + React 18 app with plain CSS. The design tokens in
`src/styles/tokens.css` are copied verbatim from the OpenRAL website, and the
fonts (SIL OFL 1.1, see `public/fonts/OFL.txt`) are bundled, so the UI works
fully offline.

## Where it runs

The same `dist/` is served in two places:

| Where | URL | Mode | Auth |
|---|---|---|---|
| The robot (`openral dashboard`) | `http://<robot>:4318/` | `local` on loopback (full control), `lan` from other hosts (watch + E-STOP) | none, same origin |
| www.openral.com | `/robot-ui/?robot=<id>` | `remote`, through the robot's Cloudflare Tunnel | a 5-minute robot-scoped JWT |

- **On the robot**, the dashboard serves the pinned release, or a directory you
  point it at with `OPENRAL_DASHBOARD_UI_DIR=<path to dist>`.
- **On openral.com**, the website pins a release by version and sha256 and
  serves it at `/robot-ui/`, framed from the Robots tab.

Mode and role come from the page location and `GET /api/config`. In remote mode
the UI reads `GET /api/robots` from openral.com, asks `POST /api/robot-token`
for a token scoped to that robot, keeps it in memory only, and renews it at about
240 s or on a 401. It never puts the token in a URL. The 3D view uses a separate
robot-issued viz ticket instead.

## Develop

Needs Node 22 or newer.

```bash
cd python/observability/ui
npm ci
npm run dev            # http://localhost:5173, proxies /api, /viz, /ws to http://127.0.0.1:4318
ROBOT=http://my-robot.local:4318 npm run dev   # proxy to another robot
```

Start a robot dashboard (`openral dashboard`) first; without one the page shows
"can't reach the robot" and keeps retrying.

```bash
npm test               # node --test over the pure helpers in src/lib/
npm run build          # → dist/
npm run preview        # serve dist/ with the same proxy
```

## Layout

- `src/main.jsx`: mode detection, manifest/config polling, the SSE stream, and the page.
- `src/ActionBar.jsx`: manifest actions filtered by role and remote policy, with a minimal JSON-schema form.
- `src/panels.jsx`: one renderer per panel type. Unknown types are ignored.
- `src/lib/`: pure modules with no DOM or React, each with a `*.test.js`:
  - `sse.js`: the SSE parser and reconnect loop;
  - `auth.js`: the token cache and API client;
  - `actions.js`: action filtering, the schema form, and the action call (the website's fleet view can reuse it);
  - `viz.js`: the Lichtblick URLs;
  - `render.js`: formats and the map, overlay and sparkline math.

## Rules

- Robot data is rendered as text through React only: no `dangerouslySetInnerHTML`, no robot-supplied HTML.
- No service workers and no `crypto.randomUUID`: a LAN page is not a secure context.
- Images arrive as `blob:` URLs (camera snapshots, the map and pointcloud PNGs), never `data:`, to fit the website's CSP.

## Release

`.github/workflows/robot-ui.yml` runs `npm ci && npm test && npm run build` on
pull requests that touch this directory. On a published release it attaches
`openral-robot-ui-<tag>.tar.gz` (the contents of `dist/`) and its `.sha256` to
the release.
