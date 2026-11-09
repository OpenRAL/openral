import { test } from "node:test";
import assert from "node:assert/strict";
import { coerceArgs, describeActionResult, initialValues, runAction, schemaFields, visibleActions } from "./actions.js";

const A = [
  { id: "prompt", kind: "prompt", min_role: "operator", remote: false },
  { id: "estop_reset", kind: "estop_reset", min_role: "operator", remote: true },
  { id: "estop", kind: "estop", min_role: "viewer", remote: true },
  { id: "skill:x", kind: "skill", min_role: "admin", remote: true },
  { id: "weird", kind: "skill", min_role: "superuser", remote: true },
];
const ids = (r) => r.shown.map((a) => a.id);

test("viewer (LAN) sees only E-STOP", () => {
  const r = visibleActions(A, "viewer", false);
  assert.deepEqual(ids(r), ["estop"]);
  assert.deepEqual(r.hidden, { role: 4, remote: 0 });
});

test("operator locally sees operator actions, E-STOP first; admin ranks above operator", () => {
  assert.deepEqual(ids(visibleActions(A, "operator", false)), ["estop", "prompt", "estop_reset"]);
  assert.deepEqual(ids(visibleActions(A, "admin", false)), ["estop", "prompt", "estop_reset", "skill:x"]);
});

test("remote drops actions the robot doesn't allow over the tunnel", () => {
  const r = visibleActions(A, "admin", true);
  assert.deepEqual(ids(r), ["estop", "estop_reset", "skill:x"]);
  assert.deepEqual(r.hidden, { role: 1, remote: 1 });
  assert.equal(r.blocked.length, 1); // shown disabled, with the reason
  assert.ok(r.blocked.every((a) => a.remote !== true && a.kind !== "estop"));
});

test("E-STOP survives an unknown role, a malformed list, and a manifest without one", () => {
  assert.deepEqual(ids(visibleActions([null, { kind: "estop" }, ...A], "nobody", true)), ["estop"]);
  assert.deepEqual(ids(visibleActions(undefined, "admin", false)), ["estop"]);
  assert.deepEqual(ids(visibleActions([A[0]], "operator", false)), ["estop", "prompt"]);
});

const schema = {
  type: "object",
  required: ["text", "n"],
  properties: {
    text: { type: "string" },
    n: { type: "integer", default: 3 },
    speed: { type: "number" },
    fast: { type: "boolean", default: true },
    side: { enum: ["left", "right", 2] },
    xyz: { type: "array" },
    odd: { type: ["string", "null"], title: "Odd one" },
  },
};

test("schema → fields with types, labels, required", () => {
  const f = schemaFields(schema);
  assert.deepEqual(f.map((x) => [x.name, x.type, x.required]), [
    ["text", "string", true], ["n", "integer", true], ["speed", "number", false], ["fast", "boolean", false],
    ["side", "enum", false], ["xyz", "array", false], ["odd", "string", false],
  ]);
  assert.equal(f[6].label, "Odd one");
  assert.deepEqual(schemaFields(null), []);
  assert.deepEqual(schemaFields({ type: "object", properties: {} }), []);
});

test("coercion: types, enum values keep their type, optional empties omitted", () => {
  const f = schemaFields(schema);
  const v = { ...initialValues(f), text: " pick the cube ", speed: "0.5", side: "2", xyz: "[1, 2, 3]" };
  assert.deepEqual(coerceArgs(f, v), {
    args: { text: " pick the cube ", n: 3, speed: 0.5, fast: true, side: 2, xyz: [1, 2, 3] },
    errors: null,
  });
});

test("validation: required, numbers, integers, enum range, JSON shape", () => {
  const f = schemaFields(schema);
  const { errors } = coerceArgs(f, { text: "   ", n: "2.5", speed: "fast", side: "9", xyz: "{}" });
  assert.deepEqual(errors, {
    text: "required", n: "must be a whole number", speed: "must be a number",
    side: "pick one of the options", xyz: "must be a JSON array",
  });
  assert.deepEqual(coerceArgs(f, {}).errors, { text: "required", n: "required" });
});

test("action responses read as plain text", () => {
  assert.deepEqual(describeActionResult(202, { ok: true, goal_id: "g-1" }), { ok: true, text: "accepted · goal g-1" });
  assert.deepEqual(describeActionResult(200, { ok: true }), { ok: true, text: "done" });
  assert.equal(describeActionResult(403, null).text, "not allowed for your role or from this connection");
  assert.equal(describeActionResult(409, { error: "cooldown 2 s" }).text, "rejected: cooldown or invalid robot state: cooldown 2 s");
  assert.equal(describeActionResult(503, { detail: [{ msg: "x" }] }).text, "the robot's ROS stack is unavailable");
  assert.equal(describeActionResult(418, {}).text, "HTTP 418");
});

test("runAction posts {args} to an encoded action id", async () => {
  let seen;
  const r = await runAction(async (path, init) => {
    seen = [path, init.method, init.body];
    return { status: 202, json: async () => ({ ok: true, goal_id: "g" }) };
  }, "skill:pick/cube", { a: 1 });
  assert.deepEqual(seen, ["/api/actions/skill%3Apick%2Fcube", "POST", '{"args":{"a":1}}']);
  assert.equal(r.ok, true);
  assert.equal((await runAction(async () => { throw new Error("offline"); }, "estop")).text, "robot unreachable (offline)");
});
