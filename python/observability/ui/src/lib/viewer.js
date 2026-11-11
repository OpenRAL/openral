// Who the profile button at the bottom of the left pane names.
// On the website (`remote`), `me` is GET /api/dashboard (the signed-in account);
// on the robot there is no account: the viewer is the local operator, by role.
export function viewerLabel(me, cfg) {
  const role = typeof cfg?.role === "string" ? cfg.role : "viewer";
  const text = (v) => (typeof v === "string" && v.trim() ? v.trim() : "");
  const name = text(me?.name) || text(me?.email) || (cfg?.mode === "remote" ? "Signed in" : `Local ${role}`);
  const words = name.replace(/@.*/, "").split(/[\s._-]+/).filter(Boolean);
  const initials = (words.length > 1 ? words[0][0] + words[1][0] : (words[0] || "?").slice(0, 2)).toUpperCase();
  const where = cfg?.mode === "remote" ? text(me?.org) || "openral.com" : cfg?.mode === "lan" ? "LAN" : "on the robot";
  return { name, initials, detail: `${role} · ${where}` };
}
