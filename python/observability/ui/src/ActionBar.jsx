import { useEffect, useState } from "react";
import { coerceArgs, initialValues, runAction, schemaFields, visibleActions } from "./lib/actions.js";
import { RELEASE_ARM_MS, releaseInPlaceOfStop } from "./lib/safety.js";

// One form per manifest action. Fields come from `args_schema`; an action
// without fields is just its button.
function ActionForm({ api, action }) {
  const fields = schemaFields(action.args_schema);
  const [values, setValues] = useState(() => initialValues(fields));
  const [errors, setErrors] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const estop = action.kind === "estop";
  const label = typeof action.label === "string" && action.label ? action.label : action.id;
  const fid = (f) => `act-${action.id}-${f.name}`;

  async function submit(e) {
    e.preventDefault();
    const { args, errors: errs } = coerceArgs(fields, values);
    setErrors(errs);
    if (errs) return;
    // E-STOP never waits on a dialog.
    if (!estop && typeof action.confirm === "string" && action.confirm && !window.confirm(action.confirm)) return;
    setBusy(true);
    const res = await runAction(api.call, action.id, args);
    setResult(res);
    setBusy(false);
    if (res.ok && !estop) setValues(initialValues(fields));
  }

  return (
    <form className={`action${estop ? " action-estop" : ""}`} onSubmit={submit} noValidate>
      {fields.map((f) => (
        <label key={f.name} className={f.type === "boolean" ? "field check" : "field"} htmlFor={fid(f)}>
          <span className="field-label">
            {f.label}
            {f.required && <span aria-hidden="true"> *</span>}
          </span>
          {f.type === "boolean" ? (
            <input
              id={fid(f)}
              type="checkbox"
              checked={values[f.name] === true}
              onChange={(e) => setValues({ ...values, [f.name]: e.target.checked })}
            />
          ) : f.type === "enum" ? (
            <select
              id={fid(f)}
              value={values[f.name]}
              required={f.required}
              aria-invalid={!!errors?.[f.name]}
              onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
            >
              <option value="">—</option>
              {f.options.map((o, i) => (
                <option key={i} value={String(i)}>
                  {typeof o === "string" ? o : JSON.stringify(o)}
                </option>
              ))}
            </select>
          ) : (
            <input
              id={fid(f)}
              type={f.type === "number" || f.type === "integer" ? "number" : "text"}
              step={f.type === "integer" ? 1 : "any"}
              placeholder={f.type === "object" || f.type === "array" ? `JSON ${f.type}` : f.description}
              title={f.description || undefined}
              value={values[f.name]}
              required={f.required}
              aria-invalid={!!errors?.[f.name]}
              autoComplete="off"
              spellCheck={false}
              onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
            />
          )}
          {errors?.[f.name] && (
            <span className="field-err" role="alert">
              {errors[f.name]}
            </span>
          )}
        </label>
      ))}
      <button type="submit" className={estop ? "btn estop" : "btn"} disabled={busy}>
        {label}
      </button>
      <span className={`action-result${result && !result.ok ? " err" : ""}`} role="status" aria-live="polite">
        {result?.text}
      </span>
    </form>
  );
}

// E-STOP's place while the kernel is latched: a green "Release e-stop" (the estop_reset
// action). Pressable only RELEASE_ARM_MS after it appears, and only after a confirm: the
// servos take torque back and may move toward their last goal.
function ReleaseButton({ api, action }) {
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  useEffect(() => {
    const t = setTimeout(() => setArmed(true), RELEASE_ARM_MS);
    return () => clearTimeout(t);
  }, []);
  async function release() {
    if (!window.confirm("Release the e-stop? Torque comes back on, and the arm may move toward its last goal. Make sure the area around the robot is clear.")) return;
    setBusy(true);
    setResult(await runAction(api.call, action.id, {}));
    setBusy(false);
  }
  return (
    <div className="action action-release">
      <button type="button" className="btn release" disabled={!armed || busy} onClick={release}>
        Release e-stop
      </button>
      <span className={`action-result${result && !result.ok ? " err" : ""}`} role="status" aria-live="polite">
        {result?.text}
      </span>
    </div>
  );
}

// The action bar, in two parts: `estop` (E-STOP, always present, in the
// sticky header so it never scrolls away) and `rest` (whatever the manifest
// allows this role over this connection).
// `skipKinds`: action kinds drawn elsewhere on the page (the prompt bar).
export function ActionBar({ api, cfg, actions, latched, topics, now, part, skipKinds = [] }) {
  const all = visibleActions(actions, cfg.role, cfg.mode === "remote");
  const { hidden } = all;
  const blocked = all.blocked.filter((a) => !skipKinds.includes(a.kind));
  const mine = all.shown.filter((a) => (a.kind === "estop") === (part === "estop") && !skipKinds.includes(a.kind));
  if (part === "estop") {
    const release = all.shown.find((a) => a.kind === "estop_reset");
    if (releaseInPlaceOfStop(topics, now, release)) {
      return (
        <section className="actions" aria-label="Emergency stop">
          <ReleaseButton api={api} action={release} />
          <span className="pill violation">e-stop latched</span>
        </section>
      );
    }
    return (
      <section className="actions" aria-label="Emergency stop">
        {mine.map((a) => (
          <ActionForm key={a.id} api={api} action={a} />
        ))}
        {latched && <span className="pill violation">e-stop latched</span>}
        {latched && !release && cfg.mode === "remote" && (
          <span className="actions-note">Release it at the robot: http://&lt;robot&gt;:4318.</span>
        )}
      </section>
    );
  }
  const notes = [];
  if (hidden.role) notes.push(`Your role (${cfg.role}) can only watch and E-STOP.`);
  if (hidden.remote)
    notes.push(
      "Off over the network on this robot: its site turns them on with OPENRAL_DASHBOARD_REMOTE_CONTROL=1. E-STOP always works.",
    );
  if (cfg.mode === "lan") notes.push("LAN view: full control is on the robot's own screen.");
  if (!mine.length && !notes.length) return null;
  return (
    <section className="actions" aria-label="Robot actions">
      {mine.map((a) => (
        <ActionForm key={a.id} api={api} action={a} />
      ))}
      {blocked.map((a) => (
        <button key={a.id} type="button" className="btn" disabled title="Off over the network on this robot">
          {a.label || a.id}
        </button>
      ))}
      {notes.length > 0 && <p className="actions-note">{notes.join(" ")}</p>}
    </section>
  );
}
