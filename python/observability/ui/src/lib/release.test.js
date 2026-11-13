import assert from "node:assert/strict";
import test from "node:test";
import { RELEASE_IDLE, releaseStep, releaseView } from "./release.js";

// Feed events in order; collect how many requests were sent.
function run(events, state = RELEASE_IDLE) {
  let sends = 0;
  for (const e of events) {
    const [next, send] = releaseStep(state, e);
    state = next;
    if (send) sends += 1;
  }
  return { state, sends };
}

test("disabled until armed: presses before RELEASE_ARM_MS do nothing", () => {
  const { state, sends } = run([{ type: "press" }, { type: "press" }, { type: "press" }]);
  assert.equal(sends, 0);
  assert.equal(state.asking, false);
  assert.equal(releaseView(state).disabled, true);
  assert.equal(releaseView(run([{ type: "arm" }]).state).disabled, false);
});

test("one press asks; a second sends; letting the confirm lapse sends nothing", () => {
  const asked = run([{ type: "arm" }, { type: "press" }]);
  assert.equal(asked.sends, 0);
  assert.equal(releaseView(asked.state).label, "Confirm release");
  assert.equal(run([{ type: "arm" }, { type: "press" }, { type: "press" }]).sends, 1);
  // Cancelled (the window closed): the next press asks again rather than sending.
  const lapsed = run([{ type: "arm" }, { type: "press" }, { type: "timeout" }, { type: "press" }]);
  assert.equal(lapsed.sends, 0);
  assert.equal(lapsed.state.asking, true);
});

test("one request in flight: further presses send nothing until it answers", () => {
  const { state, sends } = run([
    { type: "arm" },
    { type: "press" },
    { type: "press" }, // sends
    { type: "press" },
    { type: "press" },
    { type: "press" },
  ]);
  assert.equal(sends, 1);
  assert.equal(state.busy, true);
  assert.equal(releaseView(state).disabled, true);
});

test("a failed reset shows its error and re-enables the control; success clears it", () => {
  const sent = run([{ type: "arm" }, { type: "press" }, { type: "press" }]).state;
  const failed = run([{ type: "result", ok: false, text: "rejected: cooldown" }], sent).state;
  assert.equal(releaseView(failed).error, "rejected: cooldown");
  assert.equal(releaseView(failed).disabled, false);
  assert.equal(run([{ type: "press" }], failed).state.error, ""); // a new attempt clears it
  const ok = run([{ type: "result", ok: true, text: "done" }], sent).state;
  assert.equal(ok.error, "");
  assert.equal(ok.busy, false);
});
