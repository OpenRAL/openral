import assert from "node:assert/strict";
import test from "node:test";
import { viewerLabel } from "./viewer.js";

test("on the website the profile names the signed-in account", () => {
  const v = viewerLabel({ name: "Ada Lovelace", email: "ada@example.com", org: "acme" }, { mode: "remote", role: "operator" });
  assert.deepEqual(v, { name: "Ada Lovelace", initials: "AL", detail: "operator · acme" });
  assert.equal(viewerLabel({ email: "grace.hopper@example.com" }, { mode: "remote", role: "viewer" }).initials, "GH");
  assert.equal(viewerLabel({ email: "x@y.z" }, { mode: "remote", role: "viewer" }).detail, "viewer · openral.com");
});

test("on the robot there is no account: the local operator, by role", () => {
  assert.deepEqual(viewerLabel(null, { mode: "local", role: "operator" }), {
    name: "Local operator",
    initials: "LO",
    detail: "operator · on the robot",
  });
  assert.equal(viewerLabel(undefined, {}).name, "Local viewer");
});
