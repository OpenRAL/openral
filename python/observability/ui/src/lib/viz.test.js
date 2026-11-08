import { test } from "node:test";
import assert from "node:assert/strict";
import { localVizUrl, remoteVizUrl } from "./viz.js";

test("local viz: same-origin /viz/ with a ws(s) bridge and no layoutUrl", () => {
  const u = new URL(localVizUrl(new URL("http://robot.lan:4318/")));
  assert.equal(u.origin + u.pathname, "http://robot.lan:4318/viz/");
  assert.equal(u.searchParams.get("ds"), "foxglove-websocket");
  assert.equal(u.searchParams.get("ds.url"), "ws://robot.lan:4318/ws");
  assert.equal([...u.searchParams.keys()].join(), "ds,ds.url", "the layout rides in /viz/ itself");
  assert.equal(new URL(localVizUrl(new URL("https://x.test/"))).searchParams.get("ds.url"), "wss://x.test/ws");
});

test("remote viz: viz.openral.com, ticket encoded inside the bridge URL", () => {
  const t = "a+b/c=&d";
  const u = new URL(remoteVizUrl("https://r-7.openral.com", t));
  assert.equal(u.origin, "https://viz.openral.com");
  const ws = new URL(u.searchParams.get("ds.url"));
  assert.equal(ws.href.split("?")[0], "wss://r-7.openral.com/ws");
  assert.equal(ws.searchParams.get("ticket"), t);
  assert.equal([...u.searchParams.keys()].join(), "ds,ds.url", "no layoutUrl, no stray params from the ticket");
});
