// The bridge client against a real Foxglove WebSocket server (@foxglove/ws-protocol's
// FoxgloveServer over `ws`), speaking real CDR: the same wire the robot's bridge speaks.
import assert from "node:assert/strict";
import { once } from "node:events";
import test from "node:test";
import { parse } from "@foxglove/rosmsg";
import { MessageWriter } from "@foxglove/rosmsg2-serialization";
import wsProtocol from "@foxglove/ws-protocol";
import { WebSocketServer } from "ws";
import { Bridge } from "./bridge.js";

const { FoxgloveServer } = wsProtocol;
const STRING = "string data";
// A snapshot-type topic for the client (anything that is not TF / markers / String).
const SCAN = "float32[] ranges";
const write = (schema, msg) => new MessageWriter(parse(schema, { ros2: true })).writeMessage(msg);
const until = async (cond, ms = 3000) => {
  const end = Date.now() + ms;
  while (!cond()) {
    if (Date.now() > end) throw new Error("timed out waiting");
    await new Promise((r) => setTimeout(r, 10));
  }
};

async function serve() {
  const server = new FoxgloveServer({ name: "test", capabilities: ["assets"] });
  const wss = new WebSocketServer({ port: 0, handleProtocols: (p) => server.handleProtocols(p) });
  await once(wss, "listening");
  wss.on("connection", (conn, req) => server.handleConnection(conn, `${req.socket.remoteAddress}:${req.socket.remotePort}`));
  const add = (topic, schemaName, schema) => server.addChannel({ topic, encoding: "cdr", schemaName, schema, schemaEncoding: "ros2msg" });
  const ch = {
    urdf: add("/robot_description", "std_msgs/msg/String", STRING),
    scan: add("/scan", "sensor_msgs/msg/LaserScan", SCAN),
    other: add("/other", "std_msgs/msg/String", STRING),
  };
  const subscribed = [];
  server.on("subscribe", (id) => subscribed.push(id));
  const close = () => new Promise((r) => (wss.clients.forEach((c) => c.terminate()), wss.close(r)));
  return { server, wss, ch, subscribed, url: `ws://127.0.0.1:${wss.address().port}`, close };
}

test("subscribes only to wanted topics; ordered events and newest-only snapshots", async () => {
  const s = await serve();
  const events = [];
  const statuses = [];
  const b = new Bridge({ url: s.url, onEvent: (t, m) => events.push([t, m.data]), onStatus: (st) => statuses.push(st) });
  try {
    await until(() => b.topics().length === 3);
    b.want(["/robot_description", "/scan"]);
    await until(() => s.subscribed.length === 2);
    assert.deepEqual(s.subscribed.sort(), [s.ch.urdf, s.ch.scan].sort()); // never /other
    for (const text of ["a", "b", "c"]) s.server.sendMessage(s.ch.urdf, 1n, write(STRING, { data: text }));
    for (const r of [1, 2, 3]) s.server.sendMessage(s.ch.scan, 1n, write(SCAN, { ranges: new Float32Array([r]) }));
    await until(() => events.length === 3 && b.latest.has("/scan"));
    assert.deepEqual(events, [["/robot_description", "a"], ["/robot_description", "b"], ["/robot_description", "c"]]);
    await new Promise((r) => setTimeout(r, 50)); // let all three scans land
    assert.deepEqual([...b.take("/scan").ranges], [3]); // the newest
    assert.equal(b.take("/scan"), null); // taken once
    b.want(["/scan"]);
    assert.ok(!b.subs.has("/robot_description"));
  } finally {
    b.close();
    await until(() => statuses.includes("closed"));
    await s.close();
  }
  assert.deepEqual(statuses, ["open", "closed"]); // closed exactly once
});

test("assets: a few at a time, errors reported, the rest rejected when the socket drops", async () => {
  const s = await serve();
  const statuses = [];
  let inFlight = 0;
  let peak = 0;
  const held = [];
  s.server.on("fetchAsset", ({ uri, requestId }, conn) => {
    inFlight++;
    peak = Math.max(peak, inFlight);
    const reply = () => {
      inFlight--;
      const ok = !uri.endsWith("missing.stl");
      s.server.sendFetchAssetResponse(
        ok ? { op: 4, requestId, status: 0, data: new Uint8Array([1, 2, 3]) } : { op: 4, requestId, status: 1, error: "Asset URI not allowed" },
        conn,
      );
    };
    if (uri.includes("never")) held.push(reply);
    else setTimeout(reply, 20);
  });
  const b = new Bridge({ url: s.url, onStatus: (st) => statuses.push(st) });
  try {
  await until(() => statuses.includes("open"));
  const many = Array.from({ length: 8 }, (_, i) => b.fetchAsset(`package://so101/assets/m${i}.stl`));
  const bytes = await Promise.all(many);
  assert.equal(bytes.length, 8);
  assert.deepEqual([...new Uint8Array(bytes[0].buffer, bytes[0].byteOffset, bytes[0].byteLength)], [1, 2, 3]);
  assert.ok(peak <= 3, `at most 3 in flight, saw ${peak}`);
  await assert.rejects(b.fetchAsset("package://so101/assets/missing.stl"), /not allowed/);
  const stuck = b.fetchAsset("package://so101/assets/never.stl");
  const queued = [1, 2, 3].map((i) => b.fetchAsset(`package://so101/assets/never.stl?${i}`));
  await until(() => held.length === 3); // three sent and unanswered, one still queued
  assert.equal(b.assetQueue.length, 1);
  await s.close(); // the robot goes away mid-download
  await assert.rejects(stuck, /bridge closed/);
  for (const q of queued) await assert.rejects(q, /bridge closed/);
  await assert.rejects(b.fetchAsset("package://so101/assets/late.stl"), /bridge closed/);
  } finally {
    b.close();
    await s.close();
  }
  assert.deepEqual(statuses, ["open", "closed"]);
});
