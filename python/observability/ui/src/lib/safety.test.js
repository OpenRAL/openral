import assert from "node:assert/strict";
import test from "node:test";
import { SAFETY_LIVENESS_S, safetyVerdict } from "./safety.js";

// Topic shapes as the dashboard store emits them (store.set_safety_status, safety.check spans).
const NOW = 1_000;
const status = (extra = {}) => ({ ts_unix: NOW - 1, latched: false, drop_reason_label: "drop_none", ...extra });

test("nominal needs a fresh, unlatched status that drops nothing", () => {
  assert.deepEqual(safetyVerdict({ safety_status: status() }, NOW), { state: "ok", label: "safety nominal" });
});

test("a status older than the liveness window is stale, never nominal", () => {
  const v = safetyVerdict({ safety_status: status({ ts_unix: NOW - SAFETY_LIVENESS_S - 0.5 }) }, NOW);
  assert.equal(v.state, "warn");
  assert.match(v.label, /stale/);
  // The kernel died an hour ago with latched=false: still not green.
  assert.equal(safetyVerdict({ safety_status: status({ ts_unix: NOW - 3600 }) }, NOW).state, "warn");
});

test("no status at all is unknown, not clear", () => {
  assert.equal(safetyVerdict({}, NOW).state, "unknown");
  assert.equal(safetyVerdict({ safety_status: {} }, NOW).state, "unknown");
  assert.equal(safetyVerdict(undefined, NOW).state, "unknown");
});

test("a latch always shows, with or without a status feed, fresh or stale", () => {
  assert.equal(safetyVerdict({ safety_status: status({ latched: true }) }, NOW).label, "E-STOP latched");
  assert.equal(safetyVerdict({ safety_status: status({ latched: true, ts_unix: NOW - 999 }) }, NOW).state, "bad");
  // OTLP-only dashboard: no /openral/safety_status, but an operator E-STOP was seen.
  assert.equal(safetyVerdict({ safety_status: {}, safety: { checks: {}, estopped: true } }, NOW).state, "bad");
});

test("a fail-closed drop without a latch is a warning", () => {
  const v = safetyVerdict({ safety_status: status({ drop_reason_label: "drop_stale_world" }) }, NOW);
  assert.deepEqual(v, { state: "warn", label: "safety: dropping actions" });
});
