import assert from "node:assert/strict";
import test from "node:test";
import { reasonerDown } from "./reasoner.js";

// Entries as dashboard/store.py records them from deploy.bringup spans.
const at = (transition, result) => ({
    openral_reasoner: { transition, result, ts_unix: 1 },
});

test("unknown or active reasoner leaves the prompt bar usable", () => {
    assert.equal(reasonerDown(undefined), null);
    assert.equal(
        reasonerDown({
            other_node: { transition: "on_configure", result: "FAILURE" },
        }),
        null,
    );
    assert.equal(reasonerDown(at("on_configure", "SUCCESS")), null);
    assert.equal(reasonerDown(at("on_activate", "SUCCESS")), null);
});

test("a failed or stopped reasoner disables it with a reason", () => {
    assert.match(
        reasonerDown(at("on_configure", "FAILURE")),
        /on_configure failed/,
    );
    assert.match(reasonerDown(at("on_activate", "ERROR")), /failed to start/);
    assert.match(reasonerDown(at("on_deactivate", "SUCCESS")), /stopped/);
});
