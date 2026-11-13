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
  // The kernel released, but a HAL kept its own latch (a restart-only HAL, a failed re-arm).
  if (halLatched(topics, nowUnix)) return { state: "bad", label: "E-STOP latched (robot)" };
  if (!st || st.ts_unix == null) return { state: "unknown", label: "safety: unknown" };
  const age = nowUnix - st.ts_unix;
  if (!(age <= SAFETY_LIVENESS_S)) return { state: "warn", label: `safety: stale ${Math.max(0, Math.round(age))} s` };
  const reason = typeof st.drop_reason_label === "string" ? st.drop_reason_label : "";
  if (reason && reason !== "drop_none") return { state: "warn", label: "safety: dropping actions" };
  return { state: "ok", label: "safety nominal" };
}

// Whether the header shows "Release e-stop" in E-STOP's place. Only while a fresh
// /openral/safety_status says the kernel is latched (stopped now, not an hour ago), and
// only when this session may release it (`release`: the estop_reset action it can run).
// Anything else (stale or missing status, a latch learned only from an event, no release
// for this role or connection) keeps E-STOP on screen.
// `maxAgeS` is the freshness the swap needs: SAFETY_LIVENESS_S to show Release, and the
// longer RELEASE_HOLD_S to keep it once shown, so a short stall in the stream does not
// flip it back to E-STOP (and reset its arming and confirm) mid-press.
export function releaseInPlaceOfStop(topics, nowUnix, release, maxAgeS = SAFETY_LIVENESS_S) {
  const st = topics?.safety_status;
  if (!release || st?.latched !== true || typeof st.ts_unix !== "number") return false;
  const age = nowUnix - st.ts_unix;
  return age >= -1 && age <= maxAgeS;
}

export const RELEASE_HOLD_S = 10;

// A HAL whose own latch is set in a fresh heartbeat (``/diagnostics``, 1 Hz). The kernel's
// release does not release it: a restart-only HAL keeps it, so does a failed re-arm.
export function halLatched(topics, nowUnix) {
  return Object.values(topics?.hal_estop || {}).some(
    (h) => h?.estopped === true && typeof h.ts_unix === "number" && nowUnix - h.ts_unix <= SAFETY_LIVENESS_S,
  );
}

// A release can only be pressed this long after it replaced E-STOP, so a double-click on
// E-STOP (or a second press "to be sure") never releases the stop it just set.
export const RELEASE_ARM_MS = 1500;
