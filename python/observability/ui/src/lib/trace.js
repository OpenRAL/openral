// The reasoner trace: the stream carries only the latest `reasoner.tick`, so the
// page keeps the ticks it has seen, newest first. Pure.

export const TRACE_MAX = 60;

// What a tick did, as a short label for the trace.
export function tickKind(t) {
  if (t?.error_kind) return "error";
  if (t?.suppressed_reason && !t?.tool) return "wait";
  const tool = String(t?.tool || "").toLowerCase();
  if (!tool) return "think";
  if (tool.includes("rskill")) return "act";
  if (tool.includes("prompt")) return "prompt";
  if (tool.includes("lifecycle")) return "lifecycle";
  if (tool.includes("pipeline") || tool.includes("gst")) return "perceive";
  if (tool.includes("memory") || tool.includes("locate") || tool.includes("recall")) return "perceive";
  if (tool.includes("wait")) return "wait";
  return "plan";
}

const same = (a, b) =>
  a.tick_idx != null && b.tick_idx != null ? a.tick_idx === b.tick_idx : a.ts_unix === b.ts_unix;

// Adds `tick` to the history (newest first) unless it is already there.
export function traceAppend(history, tick, max = TRACE_MAX) {
  if (!tick || typeof tick !== "object" || tick.ts_unix == null) return history;
  if (history.length && same(history[0], tick)) return history;
  const entry = {
    ts_unix: tick.ts_unix,
    tick_idx: tick.tick_idx ?? null,
    kind: tickKind(tick),
    tool: tick.tool ?? null,
    rskill_id: tick.rskill_id ?? null,
    error_kind: tick.error_kind ?? null,
    suppressed_reason: tick.suppressed_reason ?? null,
    model: tick.model ?? null,
  };
  return [entry, ...history].slice(0, max);
}
