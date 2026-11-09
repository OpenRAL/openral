// Manifest actions (ADR-0129 Interface A): who may see which, a minimal
// JSON-schema form, and POST /api/actions/{id}. No DOM, no React — the website's
// fleet view can reuse this module as-is.

export const ROLE_RANK = { viewer: 0, operator: 1, admin: 2 };

// Offered when the manifest is missing or declares no E-STOP: POST
// /api/actions/estop is the robot's E-STOP whatever the manifest says.
export const BUILTIN_ESTOP = { id: "estop", kind: "estop", label: "E-STOP", min_role: "viewer", remote: true };

// Actions this session may use, E-STOP first. E-STOP is always offered, to
// every role and over every connection (ADR-0129 §6). `hidden` counts what
// was filtered so the UI can say why controls are missing; `blocked` lists the
// ones the role allows but the robot refuses over the network, shown disabled.
export function visibleActions(actions, role, remote) {
  const have = ROLE_RANK[role] ?? 0;
  const shown = [];
  const blocked = [];
  const hidden = { role: 0, remote: 0 };
  for (const a of Array.isArray(actions) ? actions : []) {
    if (!a || typeof a.id !== "string") continue;
    if (a.kind === "estop") shown.push(a);
    else if (have < (ROLE_RANK[a.min_role] ?? Infinity)) hidden.role++;
    else if (remote && a.remote !== true) {
      hidden.remote++;
      blocked.push(a);
    }
    else shown.push(a);
  }
  if (!shown.some((a) => a.kind === "estop")) shown.push(BUILTIN_ESTOP);
  shown.sort((a, b) => (b.kind === "estop") - (a.kind === "estop"));
  return { shown, hidden, blocked };
}

const TYPES = ["string", "number", "integer", "boolean", "object", "array"];

// The top-level properties of an object schema as form fields.
export function schemaFields(schema) {
  const props = schema && typeof schema === "object" ? schema.properties : null;
  if (!props || typeof props !== "object") return [];
  const required = new Set(Array.isArray(schema.required) ? schema.required : []);
  return Object.entries(props).map(([name, p]) => {
    const s = p && typeof p === "object" ? p : {};
    const isEnum = Array.isArray(s.enum) && s.enum.length > 0;
    return {
      name,
      type: isEnum ? "enum" : TYPES.includes(s.type) ? s.type : "string",
      label: typeof s.title === "string" ? s.title : name,
      description: typeof s.description === "string" ? s.description : "",
      required: required.has(name),
      options: isEnum ? s.enum : null,
      default: s.default,
    };
  });
}

// Initial form state: strings for inputs, the option index for enums.
export function initialValues(fields) {
  const v = {};
  for (const f of fields) {
    if (f.type === "boolean") v[f.name] = f.default === true;
    else if (f.type === "enum") {
      const i = f.options.indexOf(f.default);
      v[f.name] = i >= 0 ? String(i) : "";
    } else if (f.type === "object" || f.type === "array") {
      v[f.name] = f.default === undefined ? "" : JSON.stringify(f.default);
    } else v[f.name] = f.default == null ? "" : String(f.default);
  }
  return v;
}

// Form state → `args`, or per-field errors. Empty optional fields are left out.
export function coerceArgs(fields, values) {
  const args = {};
  const errors = {};
  for (const f of fields) {
    const raw = values?.[f.name];
    if (f.type === "boolean") {
      args[f.name] = raw === true;
      continue;
    }
    if (raw === undefined || raw === null || String(raw).trim() === "") {
      if (f.required) errors[f.name] = "required";
      continue;
    }
    if (f.type === "enum") {
      const i = Number(raw);
      if (Number.isInteger(i) && i >= 0 && i < f.options.length) args[f.name] = f.options[i];
      else errors[f.name] = "pick one of the options";
    } else if (f.type === "number" || f.type === "integer") {
      const n = Number(raw);
      if (!Number.isFinite(n)) errors[f.name] = "must be a number";
      else if (f.type === "integer" && !Number.isInteger(n)) errors[f.name] = "must be a whole number";
      else args[f.name] = n;
    } else if (f.type === "object" || f.type === "array") {
      let v;
      try {
        v = JSON.parse(raw);
      } catch {
        v = undefined;
      }
      const ok = f.type === "array" ? Array.isArray(v) : v !== null && typeof v === "object" && !Array.isArray(v);
      if (ok) args[f.name] = v;
      else errors[f.name] = `must be a JSON ${f.type}`;
    } else args[f.name] = String(raw);
  }
  return { args, errors: Object.keys(errors).length ? errors : null };
}

const WHY = {
  400: "invalid arguments",
  401: "not signed in",
  403: "not allowed for your role or from this connection",
  404: "unknown action",
  409: "rejected: cooldown or invalid robot state",
  503: "the robot's ROS stack is unavailable",
};

// An action response as one line of plain text.
export function describeActionResult(status, body) {
  const raw = body && (typeof body.error === "string" ? body.error : typeof body.detail === "string" ? body.detail : "");
  const detail = raw ? raw.slice(0, 200) : "";
  if (status === 202) {
    const goal = body && typeof body.goal_id === "string" ? body.goal_id : "";
    return { ok: true, text: goal ? `accepted · goal ${goal}` : "accepted" };
  }
  if (status >= 200 && status < 300) return { ok: true, text: "done" };
  const why = WHY[status] || `HTTP ${status}`;
  return { ok: false, text: detail ? `${why}: ${detail}` : why };
}

// `call(path, init)` is the robot API client (auth.js createApi().call).
export async function runAction(call, id, args) {
  try {
    const r = await call(`/api/actions/${encodeURIComponent(id)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ args: args || {} }),
    });
    const body = await r.json().catch(() => null);
    return describeActionResult(r.status, body);
  } catch (e) {
    return { ok: false, text: `robot unreachable (${e?.message || e})` };
  }
}
