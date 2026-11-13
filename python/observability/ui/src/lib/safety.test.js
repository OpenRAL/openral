import assert from "node:assert/strict";
import test from "node:test";
import { SAFETY_LIVENESS_S, releaseInPlaceOfStop, safetyVerdict } from "./safety.js";

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

test("E-STOP gives way to Release only on a fresh latched status this session can release", () => {
  const reset = { id: "estop_reset", kind: "estop_reset" };
  const latched = { safety_status: status({ latched: true }) };
  assert.equal(releaseInPlaceOfStop(latched, NOW, reset), true);
  // No release for this session (viewer, or remote with remote control off): E-STOP stays.
  assert.equal(releaseInPlaceOfStop(latched, NOW, undefined), false);
  // Not latched, or latched long ago (kernel gone): E-STOP stays.
  assert.equal(releaseInPlaceOfStop({ safety_status: status() }, NOW, reset), false);
  assert.equal(releaseInPlaceOfStop({ safety_status: status({ latched: true, ts_unix: NOW - SAFETY_LIVENESS_S - 0.5 }) }, NOW, reset), false);
  // A status from the future (clock skew) is not trusted either.
  assert.equal(releaseInPlaceOfStop({ safety_status: status({ latched: true, ts_unix: NOW + 5 }) }, NOW, reset), false);
  // A latch learned only from an event, with no status feed: E-STOP stays.
  assert.equal(releaseInPlaceOfStop({ safety: { estopped: true } }, NOW, reset), false);
  assert.equal(releaseInPlaceOfStop({ safety_status: { latched: true } }, NOW, reset), false);
  assert.equal(releaseInPlaceOfStop(undefined, NOW, reset), false);
});
