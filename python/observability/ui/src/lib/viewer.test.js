import assert from "node:assert/strict";
import test from "node:test";
import { robotLinks, viewerLabel } from "./viewer.js";

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

test("the left pane lists the robots the console can open, each linking to its dashboard page", () => {
  const list = {
    robots: [
      { id: "ab12", name: "arm 1", url: "https://r-ab12.openral.com" },
      { id: "cd34", name: "🤖 Rover", url: "https://r-cd34.openral.com" },
      { id: "ef56", name: "", url: "https://r-ef56.openral.com" },
      { id: "bad", name: "No address", url: "javascript:alert(1)" },
    ],
  };
  assert.deepEqual(robotLinks(list), [
    { id: "ab12", name: "arm 1", initial: "A", href: "/dashboard?robot=ab12#robots" },
    { id: "cd34", name: "🤖 Rover", initial: "🤖", href: "/dashboard?robot=cd34#robots" },
    { id: "ef56", name: "ef56", initial: "E", href: "/dashboard?robot=ef56#robots" },
  ]);
  assert.deepEqual(robotLinks(null), []);
});
