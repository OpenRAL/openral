// Remote mode (ADR-0129 Interface B): www.openral.com mints a 5-minute,
// robot-scoped JWT; it lives in memory only and is never put in a URL.

// When to renew a credential that lives `expiresIn` seconds: a minute early,
// never later than `cap`, and never before half its life (short-lived ones).
export function renewDelayS(expiresIn, cap = Infinity) {
  const e = Number(expiresIn) > 0 ? Number(expiresIn) : 300;
  return Math.max(e / 2, Math.min(cap, e - 60));
}

// Lazy token cache: a call after ~240 s (or a forced one, after a 401) mints a
// fresh token; concurrent callers share one request.
export function createTokenClient({ mint, now = Date.now }) {
  let token = null;
  let renewAt = 0;
  let inflight = null;
  return {
    get(force = false) {
      if (!force && token && now() < renewAt) return Promise.resolve(token);
      inflight ??= (async () => {
        try {
          const r = await mint();
          if (!r || typeof r.access_token !== "string") throw new Error("token response has no access_token");
          token = r.access_token;
          renewAt = now() + renewDelayS(r.expires_in, 240) * 1000;
          return token;
        } finally {
          inflight = null;
        }
      })();
      return inflight;
    },
  };
}

// POST /api/robot-token on the website (same-origin, session cookie).
export function websiteMinter(resource, f = globalThis.fetch) {
  return async () => {
    const r = await f("/api/robot-token", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resource }),
    });
    if (!r.ok) throw new Error(`openral.com refused a token for this robot (HTTP ${r.status})`);
    return r.json();
  };
}

// Robot API client. Local mode: base "" and no auth (same origin). Remote:
// base is the robot's origin and every call carries the Bearer token; a 401 is
// retried once with a freshly minted token.
export function createApi({ base = "", auth = null, fetch: f = (...a) => globalThis.fetch(...a) } = {}) {
  async function call(path, init = {}) {
    for (let attempt = 0; ; attempt++) {
      const headers = new Headers(init.headers);
      if (auth) headers.set("Authorization", `Bearer ${await auth.get(attempt > 0)}`);
      const res = await f(base + path, { ...init, headers });
      if (res.status !== 401 || !auth || attempt > 0) return res;
    }
  }
  return { base, remote: !!auth, call };
}

// The robot entry from GET /api/robots, or null. A token is only ever sent to
// an https origin (plain http allowed for loopback development only).
export function findRobot(list, id) {
  const r = (list?.robots || []).find((x) => x && x.id === id);
  if (!r || typeof r.url !== "string") return null;
  let u;
  try {
    u = new URL(r.url);
  } catch {
    return null;
  }
  const loopback = u.hostname === "127.0.0.1" || u.hostname === "localhost";
  if (u.username || u.password || !(u.protocol === "https:" || (loopback && u.protocol === "http:"))) return null;
  return { ...r, base: u.origin };
}
