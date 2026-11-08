// URLs for the embedded Lichtblick 3D view (ADR-0129 §5).

export const REMOTE_VIZ = "https://viz.openral.com/";

const query = (wsUrl, layoutUrl) =>
  new URLSearchParams({ ds: "foxglove-websocket", "ds.url": wsUrl, layoutUrl }).toString();

// Local / LAN: the robot serves Lichtblick at /viz/ and proxies the bridge at /ws.
export function localVizUrl(loc) {
  const ws = `${loc.protocol === "https:" ? "wss:" : "ws:"}//${loc.host}/ws`;
  return `${loc.origin}/viz/?${query(ws, `${loc.origin}/api/viz/layout.json`)}`;
}

// Remote: viz.openral.com, reaching the robot's /ws and layout with a
// robot-issued viz ticket (never the JWT).
export function remoteVizUrl(robotBase, ticket) {
  const ws = new URL("/ws", robotBase);
  ws.protocol = ws.protocol === "http:" ? "ws:" : "wss:";
  ws.searchParams.set("ticket", String(ticket));
  const layout = new URL("/api/viz/layout.json", robotBase);
  layout.searchParams.set("ticket", String(ticket));
  return `${REMOTE_VIZ}?${query(ws.href, layout.href)}`;
}
