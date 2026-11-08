import { useState } from "react";
import { coerceArgs, initialValues, runAction, schemaFields, visibleActions } from "./lib/actions.js";

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

// The action bar, in two parts: `estop` (E-STOP, always present, in the
// sticky header so it never scrolls away) and `rest` (whatever the manifest
// allows this role over this connection).
export function ActionBar({ api, cfg, actions, latched, part }) {
  const { shown, hidden } = visibleActions(actions, cfg.role, cfg.mode === "remote");
  const mine = shown.filter((a) => (a.kind === "estop") === (part === "estop"));
  if (part === "estop") {
    return (
      <section className="actions" aria-label="Emergency stop">
        {mine.map((a) => (
          <ActionForm key={a.id} api={api} action={a} />
        ))}
        {latched && <span className="pill violation">e-stop latched</span>}
      </section>
    );
  }
  const notes = [];
  if (hidden.role) notes.push(`Your role (${cfg.role}) can only watch and E-STOP.`);
  if (hidden.remote) notes.push("This robot allows only E-STOP remotely (OPENRAL_DASHBOARD_REMOTE_CONTROL is off).");
  if (cfg.mode === "lan") notes.push("LAN view: full control is on the robot's own screen.");
  if (!mine.length && !notes.length) return null;
  return (
    <section className="actions" aria-label="Robot actions">
      {mine.map((a) => (
        <ActionForm key={a.id} api={api} action={a} />
      ))}
      {notes.length > 0 && <p className="actions-note">{notes.join(" ")}</p>}
    </section>
  );
}
