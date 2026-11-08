// URLs for the embedded Lichtblick 3D view (ADR-0129 §5). The layout is not a URL
// parameter: the robot's /viz/ and viz.openral.com each carry one in their index.html.

export const REMOTE_VIZ = "https://viz.openral.com/";

const query = (wsUrl) => new URLSearchParams({ ds: "foxglove-websocket", "ds.url": wsUrl }).toString();

// Local / LAN: the robot serves Lichtblick at /viz/ and proxies the bridge at /ws.
export function localVizUrl(loc) {
  const ws = `${loc.protocol === "https:" ? "wss:" : "ws:"}//${loc.host}/ws`;
  return `${loc.origin}/viz/?${query(ws)}`;
}

// Remote: viz.openral.com, reaching the robot's /ws with a robot-issued viz ticket
// (never the JWT).
export function remoteVizUrl(robotBase, ticket) {
  const ws = new URL("/ws", robotBase);
  ws.protocol = ws.protocol === "http:" ? "ws:" : "wss:";
  ws.searchParams.set("ticket", String(ticket));
  return `${REMOTE_VIZ}?${query(ws.href)}`;
}
