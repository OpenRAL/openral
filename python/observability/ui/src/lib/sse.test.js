import { test } from "node:test";
import assert from "node:assert/strict";
import { backoffMs, createSSEParser } from "./sse.js";

const feed = (chunks) => {
  const out = [];
  const push = createSSEParser((d) => out.push(d));
  for (const c of chunks) push(c);
  return out;
};

test("dispatches on a blank line, ignores comments and other fields", () => {
  assert.deepEqual(feed([': keepalive\n\nevent: x\nid: 7\ndata: {"a":1}\n\n']), ['{"a":1}']);
});

test("survives any chunk boundary, including a split CRLF", () => {
  const s = 'data: {"a":1}\r\ndata: {"b":2}\r\n\r\n: keepalive\r\n\r\ndata:no-space\n\n';
  const want = ['{"a":1}\n{"b":2}', "no-space"];
  for (let i = 0; i <= s.length; i++) assert.deepEqual(feed([s.slice(0, i), s.slice(i)]), want, `split at ${i}`);
  assert.deepEqual(feed([...s]), want, "one char at a time");
});

test("multi-line data joins with LF; empty data and unterminated events don't dispatch", () => {
  assert.deepEqual(feed(["data: a\ndata:\ndata: b\n\n"]), ["a\n\nb"]);
  assert.deepEqual(feed(["data:\n\n", "data: pending"]), []);
  // A lone trailing CR waits for the next byte (it may be half a CRLF).
  assert.deepEqual(feed(["data: x\rdata: y\r\r", ": keepalive"]), ["x\ny"]);
});

test("backoff doubles from 1 s and caps at 15 s", () => {
  assert.deepEqual([0, 1, 2, 3, 4, 9].map(backoffMs), [1000, 2000, 4000, 8000, 15000, 15000]);
});
