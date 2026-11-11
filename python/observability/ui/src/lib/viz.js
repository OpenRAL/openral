// URLs for the robot's foxglove_bridge (ADR-0129 §5): the console's own 3D scene
// opens the bridge WebSocket directly. Lichtblick stays available on the robot at
// /viz/ for raw topic inspection, opened in its own tab.

const query = (wsUrl) => new URLSearchParams({ ds: "foxglove-websocket", "ds.url": wsUrl }).toString();

// Local / LAN: the robot proxies the bridge at /ws on the page's own origin.
// Loopback needs no ticket; a LAN client must pass one (POST /api/viz/ticket).
export function localBridgeUrl(loc, ticket = null) {
  const ws = new URL(`${loc.protocol === "https:" ? "wss:" : "ws:"}//${loc.host}/ws`);
  if (ticket) ws.searchParams.set("ticket", String(ticket));
  return ws.href;
}

// Remote: the robot's /ws through its tunnel, with a robot-issued viz ticket
// (never the JWT).
export function remoteBridgeUrl(robotBase, ticket) {
  const ws = new URL("/ws", robotBase);
  ws.protocol = ws.protocol === "http:" ? "ws:" : "wss:";
  ws.searchParams.set("ticket", String(ticket));
  return ws.href;
}

// Lichtblick, served by the robot at /viz/ (loopback: no ticket needed).
export function localVizUrl(loc) {
  return `${loc.origin}/viz/?${query(localBridgeUrl(loc))}`;
}
