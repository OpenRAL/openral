import { test } from "node:test";
import assert from "node:assert/strict";
import { localVizUrl, remoteVizUrl } from "./viz.js";

test("local viz: same-origin /viz/ with ws(s) bridge and layout", () => {
  const u = new URL(localVizUrl(new URL("http://robot.lan:4318/")));
  assert.equal(u.origin + u.pathname, "http://robot.lan:4318/viz/");
  assert.equal(u.searchParams.get("ds"), "foxglove-websocket");
  assert.equal(u.searchParams.get("ds.url"), "ws://robot.lan:4318/ws");
  assert.equal(u.searchParams.get("layoutUrl"), "http://robot.lan:4318/api/viz/layout.json");
  assert.equal(new URL(localVizUrl(new URL("https://x.test/"))).searchParams.get("ds.url"), "wss://x.test/ws");
});

test("remote viz: viz.openral.com, ticket encoded inside the nested URLs", () => {
  const t = "a+b/c=&d";
  const u = new URL(remoteVizUrl("https://r-7.openral.com", t));
  assert.equal(u.origin, "https://viz.openral.com");
  const ws = new URL(u.searchParams.get("ds.url"));
  const layout = new URL(u.searchParams.get("layoutUrl"));
  assert.equal(ws.href.split("?")[0], "wss://r-7.openral.com/ws");
  assert.equal(ws.searchParams.get("ticket"), t);
  assert.equal(layout.href.split("?")[0], "https://r-7.openral.com/api/viz/layout.json");
  assert.equal(layout.searchParams.get("ticket"), t);
  assert.equal([...u.searchParams.keys()].join(), "ds,ds.url,layoutUrl", "no stray params from the ticket");
});
