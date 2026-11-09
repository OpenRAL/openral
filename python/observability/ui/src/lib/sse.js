// Server-Sent Events over fetch + ReadableStream. EventSource can't send an
// Authorization header, and the remote robot needs a Bearer token, so one
// code path serves both modes.

// Incremental text/event-stream parser (WHATWG HTML §9.2.6, the parts the
// robot uses): `data:` lines join with "\n", a blank line dispatches, `:` lines
// are comments (the robot's `: keepalive`), other fields are ignored.
export function createSSEParser(onData) {
  let buf = "";
  let data = [];
  return function push(chunk) {
    buf += chunk;
    // A trailing CR may be the first half of a CRLF split across chunks:
    // hold it back so the LF doesn't read as an extra blank line.
    let hold = "";
    if (buf.endsWith("\r")) {
      hold = "\r";
      buf = buf.slice(0, -1);
    }
    const lines = buf.split(/\r\n|\r|\n/);
    buf = lines.pop() + hold;
    for (const line of lines) {
      if (line === "") {
        if (data.length) {
          const s = data.join("\n");
          data = [];
          if (s !== "") onData(s);
        }
        continue;
      }
      if (line[0] === ":") continue;
      const i = line.indexOf(":");
      const field = i < 0 ? line : line.slice(0, i);
      let value = i < 0 ? "" : line.slice(i + 1);
      if (value[0] === " ") value = value.slice(1);
      if (field === "data") data.push(value);
    }
  };
}

export const backoffMs = (attempt) => Math.min(15000, 1000 * 2 ** Math.max(0, attempt));

// Keep one stream open: reconnect with backoff, and treat 45 s without a byte
// (three missed 15 s keepalives) as a dead connection — a tunnel can leave a
// half-open socket that never errors. `open(signal)` returns a fetch Response.
export function connectSSE({ open, onData, onStatus = () => {}, idleMs = 45000 }) {
  let stopped = false;
  let ctrl = null;
  let wait = null;
  let attempt = 0;
  (async () => {
    while (!stopped) {
      ctrl = new AbortController();
      let idle = null;
      const kick = () => {
        clearTimeout(idle);
        idle = setTimeout(() => ctrl.abort(), idleMs);
      };
      try {
        kick();
        const res = await open(ctrl.signal);
        if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
        onStatus("live");
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        const push = createSSEParser((d) => {
          attempt = 0;
          onData(d);
        });
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          kick();
          push(dec.decode(value, { stream: true }));
        }
      } catch {
        // fall through to the reconnect below
      } finally {
        clearTimeout(idle);
      }
      if (stopped) return;
      onStatus("reconnecting");
      await new Promise((r) => (wait = setTimeout(r, backoffMs(attempt++))));
    }
  })();
  return () => {
    stopped = true;
    ctrl?.abort();
    clearTimeout(wait);
  };
}
