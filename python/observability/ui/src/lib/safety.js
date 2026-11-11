// The header's safety pill: one verdict from the safety topics in the stream.
// Pure. Mirrors the Safety status panel's rules, because the pill is what an
// operator sees when that panel's tab is closed.

// HZ-0096-1: a /openral/safety_status older than this is UNKNOWN, not safe.
export const SAFETY_LIVENESS_S = 3;

// → { state: "bad" | "warn" | "ok" | "unknown", label }
// A latch always wins, however old or however it was learned; "nominal" needs a
// fresh status that is not latched and is dropping nothing.
export function safetyVerdict(topics, nowUnix) {
  const st = topics?.safety_status;
  if (st?.latched || topics?.safety?.estopped) return { state: "bad", label: "E-STOP latched" };
  if (!st || st.ts_unix == null) return { state: "unknown", label: "safety: unknown" };
  const age = nowUnix - st.ts_unix;
  if (!(age <= SAFETY_LIVENESS_S)) return { state: "warn", label: `safety: stale ${Math.max(0, Math.round(age))} s` };
  const reason = typeof st.drop_reason_label === "string" ? st.drop_reason_label : "";
  if (reason && reason !== "drop_none") return { state: "warn", label: "safety: dropping actions" };
  return { state: "ok", label: "safety nominal" };
}
