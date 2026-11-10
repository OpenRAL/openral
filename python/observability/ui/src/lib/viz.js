// URLs for the robot's foxglove_bridge (ADR-0129 §5): the console's own 3D scene
// opens the bridge WebSocket directly; the Lichtblick URLs remain for opening the
// full Foxglove view in a new tab. The layout is not a URL parameter: the robot's
// /viz/ and viz.openral.com each carry one in their index.html.

export const REMOTE_VIZ = "https://viz.openral.com/";

const query = (wsUrl) => new URLSearchParams({ ds: "foxglove-websocket", "ds.url": wsUrl }).toString();

// Local / LAN: the robot proxies the bridge at /ws on the page's own origin.
export function localBridgeUrl(loc) {
  return `${loc.protocol === "https:" ? "wss:" : "ws:"}//${loc.host}/ws`;
}

// Remote: the robot's /ws through its tunnel, with a robot-issued viz ticket
// (never the JWT).
export function remoteBridgeUrl(robotBase, ticket) {
  const ws = new URL("/ws", robotBase);
  ws.protocol = ws.protocol === "http:" ? "ws:" : "wss:";
  ws.searchParams.set("ticket", String(ticket));
  return ws.href;
}

// Lichtblick, served by the robot at /viz/.
export function localVizUrl(loc) {
  return `${loc.origin}/viz/?${query(localBridgeUrl(loc))}`;
}

// Lichtblick on viz.openral.com, reaching the robot's /ws with a viz ticket.
export function remoteVizUrl(robotBase, ticket) {
  return `${REMOTE_VIZ}?${query(remoteBridgeUrl(robotBase, ticket))}`;
}
