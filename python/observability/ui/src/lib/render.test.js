import { test } from "node:test";
import assert from "node:assert/strict";
import { camAge, coverMap, jpegBlob, footprintToPixels, overlayColor, overlayFade, sparkGeometry, sevBucket, worldToPixel } from "./render.js";

test("map: world → flipped pixel, footprint rotates with yaw", () => {
  assert.deepEqual(worldToPixel(1, 0, 0, 0, 0.5, 10), { px: 2, py: 9 });
  const [p] = footprintToPixels([[1, 0]], 0, 0, Math.PI / 2, 0, 0, 1, 10);
  assert.ok(Math.abs(p.px) < 1e-9 && Math.abs(p.py - 8) < 1e-9);
});

test("overlays: fresh draws, fading dims, stale or out-run by the frame clears", () => {
  assert.equal(overlayFade({ ts_unix: 100 }, 101, 101), 1);
  assert.ok(overlayFade({ ts_unix: 100 }, 102.75, 102) > 0.4);
  assert.equal(overlayFade({ ts_unix: 100 }, 105, 100), 0);
  assert.equal(overlayFade({ ts_unix: 100 }, 101, 103), 0);
  assert.equal(overlayFade(null, 1, 1), 0);
  assert.equal(overlayColor("cup"), overlayColor("cup"));
  assert.deepEqual(coverMap(200, 100, 100, 100), { ox: -50, oy: 0, dw: 200, dh: 100 });
});

test("sparkline: threshold widens the domain and flags a breach", () => {
  const s = [[0, 1], [1, 2], [2, 5]];
  const g = sparkGeometry(s, null, 4, "upper");
  assert.equal(g.breach, true);
  assert.deepEqual([g.lo, g.hi], [1, 5]);
  assert.equal(sparkGeometry(s, null, 6, "lower").breach, true);
  assert.equal(sparkGeometry(s, null, 6, "upper").hi, 6);
  assert.equal(sparkGeometry([[0, 1]], null), null);
});

test("severity buckets", () => {
  assert.deepEqual(["debug", "INFO", "warn", "fatal", undefined].map(sevBucket), ["debug", "info", "warn", "error", "info"]);
});

test("a polled camera frame is always image/jpeg, whatever the robot claimed", () => {
  const b = jpegBlob(new Uint8Array([0xff, 0xd8]).buffer);
  assert.equal(b.type, "image/jpeg");
  assert.equal(b.size, 2);
});

test("camera age: the frame's age_ms while live, time since the last frame once stalled", () => {
  assert.equal(camAge({ ts_unix: 100, age_ms: 33.4 }, 100.5), "age 33 ms");
  assert.equal(camAge({ ts_unix: 100, age_ms: 33.4 }, 104.25), "age 4.3s");
  assert.equal(camAge({ ts_unix: 100, age_ms: 33.4 }, 400), "age 5m");
  assert.equal(camAge(null, 100), "—");
});
