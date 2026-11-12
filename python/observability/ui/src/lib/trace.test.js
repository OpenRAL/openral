import assert from "node:assert/strict";
import test from "node:test";
import { tickKind, traceAppend } from "./trace.js";

// reasoner.tick attributes as the dashboard store relays them (store._record_reasoner_tick).
const tick = (idx, extra = {}) => ({ ts_unix: 100 + idx, tick_idx: idx, model: "claude-opus-5-5", ...extra });

test("ticks are labelled by what they did", () => {
  assert.equal(tickKind(tick(1, { tool: "ExecuteRskillTool", rskill_id: "OpenRAL/rskill-smolvla-so101-eraser_place-bf16" })), "act");
  assert.equal(tickKind(tick(2, { tool: "EmitPromptTool" })), "prompt");
  assert.equal(tickKind(tick(3, { tool: "LifecycleTransitionTool" })), "lifecycle");
  assert.equal(tickKind(tick(4, { tool: "ReloadGstPipelineTool" })), "perceive");
  assert.equal(tickKind(tick(5, { suppressed_reason: "skill running" })), "wait");
  assert.equal(tickKind(tick(6, { tool: "ExecuteRskillTool", error_kind: "ROSReasonerInvalidPlan" })), "error");
  assert.equal(tickKind(tick(7)), "think");
});

test("the trace keeps each tick once, newest first, bounded", () => {
  let h = [];
  h = traceAppend(h, tick(1, { tool: "ExecuteRskillTool" }));
  h = traceAppend(h, tick(1, { tool: "ExecuteRskillTool" })); // same snapshot again
  h = traceAppend(h, tick(2));
  assert.deepEqual(
    h.map((e) => [e.tick_idx, e.kind]),
    [
      [2, "think"],
      [1, "act"],
    ],
  );
  assert.equal(traceAppend(h, null), h);
  assert.equal(traceAppend(h, {}), h);
  for (let i = 3; i < 20; i++) h = traceAppend(h, tick(i), 5);
  assert.equal(h.length, 5);
  assert.equal(h[0].tick_idx, 19);
});
