import { findRobot } from "./auth.js";

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

// The top of the left pane: whose workspace this is. The org (its logo when it has
// one, else its first letter) on the website and on an org's robot; a free account
// is the person's own; a robot with no org is just itself.
export function workspaceLabel(me, cfg, robotName) {
  const text = (v) => (typeof v === "string" && v.trim() ? v.trim() : "");
  const name =
    text(me?.orgName) || text(me?.org) || text(cfg?.org) || text(me?.name) || text(me?.email) || text(robotName) || "Robot";
  const logo = text(me?.orgLogo);
  return { name, initial: name[0].toUpperCase(), logo: /^(https:\/\/|\/(?![/\\]))/.test(logo) ? logo : null };
}

// The left pane's robot list on the website: the robots in GET /api/robots that the
// console can open (findRobot), each linking to its dashboard page. The initial is
// the name's first character, not its first UTF-16 unit, so an emoji stays whole.
export function robotLinks(list) {
  return (list?.robots || [])
    .filter((r) => findRobot(list, r.id))
    .map((r) => {
      const name = typeof r.name === "string" && r.name.trim() ? r.name.trim() : r.id;
      return {
        id: r.id,
        name,
        initial: Array.from(name)[0].toUpperCase(),
        href: `/dashboard?robot=${encodeURIComponent(r.id)}#robots`,
      };
    });
}
