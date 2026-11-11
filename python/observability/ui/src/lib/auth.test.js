import { test } from "node:test";
import assert from "node:assert/strict";
import { createApi, createTokenClient, findRobot, renewDelayS } from "./auth.js";

test("renew timing: a 300 s token at 240 s, a 900 s ticket at 840 s, short ones at half-life", () => {
  assert.equal(renewDelayS(300, 240), 240);
  assert.equal(renewDelayS(900), 840);
  assert.equal(renewDelayS(60), 30);
  assert.equal(renewDelayS(undefined, 240), 240);
});

test("token client caches until ~240 s, then re-mints; concurrent gets share one mint", async () => {
  let t = 0;
  let mints = 0;
  const c = createTokenClient({ now: () => t, mint: async () => ({ access_token: `tok${++mints}`, expires_in: 300 }) });
  const [a, b] = await Promise.all([c.get(), c.get()]);
  assert.deepEqual([a, b, mints], ["tok1", "tok1", 1]);
  t = 239_000;
  assert.equal(await c.get(), "tok1");
  t = 240_000;
  assert.equal(await c.get(), "tok2");
  assert.equal(await c.get(true), "tok3");
});

test("token client rejects a response without a token", async () => {
  const c = createTokenClient({ mint: async () => ({}) });
  await assert.rejects(c.get(), /access_token/);
});

test("api: Bearer on every call, one forced re-mint and retry on 401", async () => {
  let mints = 0;
  const auth = createTokenClient({ mint: async () => ({ access_token: `t${++mints}`, expires_in: 300 }) });
  const seen = [];
  const statuses = [401, 200];
  const api = createApi({
    base: "https://r-1.openral.com",
    auth,
    fetch: async (url, init) => {
      seen.push([url, init.headers.get("Authorization")]);
      return { status: statuses.shift() };
    },
  });
  assert.equal((await api.call("/api/state")).status, 200);
  assert.deepEqual(seen, [
    ["https://r-1.openral.com/api/state", "Bearer t1"],
    ["https://r-1.openral.com/api/state", "Bearer t2"],
  ]);
});

test("api: every write but E-STOP gets a fresh token; reads reuse the cached one", async () => {
  let n = 0;
  const auth = createTokenClient({ mint: async () => ({ access_token: `t${++n}`, expires_in: 300 }) });
  const seen = [];
  const api = createApi({
    base: "https://r-1.openral.com",
    auth,
    fetch: async (url, init) => (seen.push([init.method || "GET", url.slice(23), init.headers.get("Authorization")]), { status: 200 }),
  });
  await api.call("/api/state");
  await api.call("/api/actions/prompt", { method: "POST" });
  await api.call("/api/viz/ticket", { method: "POST" });
  await api.call("/api/actions/estop", { method: "POST" });
  await api.call("/api/state");
  assert.deepEqual(seen, [
    ["GET", "/api/state", "Bearer t1"],
    ["POST", "/api/actions/prompt", "Bearer t2"],
    ["POST", "/api/viz/ticket", "Bearer t3"],
    ["POST", "/api/actions/estop", "Bearer t3"],
    ["GET", "/api/state", "Bearer t3"],
  ]);
});

test("api: local mode sends no Authorization and does not retry", async () => {
  const seen = [];
  const api = createApi({ fetch: async (url, init) => (seen.push([url, init.headers.has("Authorization")]), { status: 401 }) });
  assert.equal((await api.call("/api/config")).status, 401);
  assert.deepEqual(seen, [["/api/config", false]]);
});

test("findRobot only accepts https origins (http for loopback dev)", () => {
  const list = (url) => ({ robots: [{ id: "a", url }] });
  assert.equal(findRobot(list("https://r-a.openral.com"), "a").base, "https://r-a.openral.com");
  assert.equal(findRobot(list("http://127.0.0.1:4319"), "a").base, "http://127.0.0.1:4319");
  for (const bad of ["http://r-a.openral.com", "javascript:alert(1)", "https://u:p@r-a.openral.com", "nope"]) {
    assert.equal(findRobot(list(bad), "a"), null, bad);
  }
  assert.equal(findRobot(list("https://r-a.openral.com"), "b"), null);
  assert.equal(findRobot(null, "a"), null);
});

test("concurrent writes each get their own token", async () => {
  let n = 0;
  const auth = createTokenClient({
    mint: async () => {
      await new Promise((r) => setTimeout(r, 5));
      return { access_token: `t${++n}`, expires_in: 300 };
    },
  });
  const seen = [];
  const api = createApi({
    base: "https://r-1.openral.com",
    auth,
    fetch: async (url, init) => (seen.push(init.headers.get("Authorization")), { status: 200 }),
  });
  await Promise.all([1, 2, 3].map(() => api.call("/api/actions/prompt", { method: "POST" })));
  assert.equal(new Set(seen).size, 3);
});

test("E-STOP sends a still-valid token at once and renews on the side; an expired one is never sent", async () => {
  let t = 0;
  let n = 0;
  let release;
  const auth = createTokenClient({
    now: () => t,
    mint: async () => {
      if (n >= 1) await new Promise((r) => (release = r)); // the website stalls
      return { access_token: `t${++n}`, expires_in: 300 };
    },
  });
  const seen = [];
  const api = createApi({
    base: "https://r-1.openral.com",
    auth,
    fetch: async (url, init) => (seen.push(init.headers.get("Authorization")), { status: 200 }),
  });
  await api.call("/api/state"); // t1, renew due at 240 s, expires at 300 s
  t = 250_000;
  await api.call("/api/actions/estop", { method: "POST" }); // does not wait on the stalled mint
  assert.deepEqual(seen, ["Bearer t1", "Bearer t1"]);
  release();
  await new Promise((r) => setTimeout(r, 0));
  t = 600_000; // t2 (minted at 250 s) expired too
  const stop = api.call("/api/actions/estop", { method: "POST" });
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(seen.length, 2); // waits for a fresh token instead of sending an expired one
  release();
  await stop;
  assert.equal(seen[2], "Bearer t3");
});
