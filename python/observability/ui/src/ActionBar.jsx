import { useEffect, useRef, useState } from "react";
import { coerceArgs, initialValues, runAction, schemaFields, visibleActions } from "./lib/actions.js";
import { RELEASE_IDLE, releaseStep, releaseView } from "./lib/release.js";
import { RELEASE_ARM_MS, RELEASE_HOLD_S, halLatched, releaseInPlaceOfStop } from "./lib/safety.js";

// A press that must be confirmed, on the page: the first press asks (the caller shows
// `asking`), a second within CONFIRM_MS goes. Never window.confirm: an embedded browser and
// a cross-origin frame (openral.com's robot view) suppress it, silently answering "no".
const CONFIRM_MS = 5000;
function useConfirm() {
  const [asking, setAsking] = useState(false);
  useEffect(() => {
    if (!asking) return undefined;
    const t = setTimeout(() => setAsking(false), CONFIRM_MS);
    return () => clearTimeout(t);
  }, [asking]);
  // true when this press should go ahead
  const confirm = () => {
    if (asking) {
      setAsking(false);
      return true;
    }
    setAsking(true);
    return false;
  };
  return [asking, confirm];
}

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
  const needsConfirm = !estop && typeof action.confirm === "string" && !!action.confirm; // E-STOP never waits
  const [asking, confirm] = useConfirm();

  async function submit(e) {
    e.preventDefault();
    const { args, errors: errs } = coerceArgs(fields, values);
    setErrors(errs);
    if (errs) return;
    if (needsConfirm && !confirm()) return;
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
      {asking && <span className="confirm-ask">{action.confirm}</span>}
      <button type="submit" className={estop ? "btn estop" : "btn"} disabled={busy}>
        {asking ? `Confirm ${label}` : label}
      </button>
      <span className={`action-result${result && !result.ok ? " err" : ""}`} role="status" aria-live="polite">
        {/* A stop speaks through the header (the latched pill, Release); only its failures need words. */}
        {estop && result?.ok ? "" : result?.text}
      </span>
    </form>
  );
}

// E-STOP's place while the kernel is latched: a green "Release e-stop" (the estop_reset
// action). Pressable only RELEASE_ARM_MS after it appears, then confirmed on the page
// (`useConfirm`): the first press asks, a second releases.
function ReleaseButton({ api, action }) {
  // The logic lives in lib/release.js (releaseStep), tested without a DOM.
  const [state, setState] = useState(RELEASE_IDLE);
  const stateRef = useRef(state);
  const dispatch = (event) => {
    const [next, send] = releaseStep(stateRef.current, event);
    stateRef.current = next;
    setState(next);
    if (send) runAction(api.call, action.id, {}).then((r) => dispatch({ type: "result", ok: r.ok, text: r.text }));
  };
  useEffect(() => {
    const t = setTimeout(() => dispatch({ type: "arm" }), RELEASE_ARM_MS);
    return () => clearTimeout(t);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!state.asking) return undefined;
    const t = setTimeout(() => dispatch({ type: "timeout" }), CONFIRM_MS);
    return () => clearTimeout(t);
  }, [state.asking]); // eslint-disable-line react-hooks/exhaustive-deps
  const view = releaseView(state);
  return (
    <div className="action action-release">
      <button
        type="button"
        className={view.asking ? "btn release asking" : "btn release"}
        disabled={view.disabled}
        onClick={() => dispatch({ type: "press" })}
      >
        {view.label}
      </button>
      <span className="action-result err" role="status" aria-live="polite">
        {view.error}
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
  const release = all.shown.find((a) => a.kind === "estop_reset");
  // Shown from a fresh latch; once shown, held through a short stall in the stream.
  const swapped = useRef(false);
  swapped.current = releaseInPlaceOfStop(topics, now, release, swapped.current ? RELEASE_HOLD_S : undefined);
  if (part === "estop") {
    if (swapped.current) {
      return (
        <section className="actions" aria-label="Emergency stop">
          <ReleaseButton api={api} action={release} />
        </section>
      );
    }
    return (
      <section className="actions" aria-label="Emergency stop">
        {mine.map((a) => (
          <ActionForm key={a.id} api={api} action={a} />
        ))}
        {latched && !release && cfg.mode === "remote" && (
          <span className="actions-note">Release it at the robot: http://&lt;robot&gt;:4318.</span>
        )}
        {!topics?.safety_status?.latched && halLatched(topics, now) && (
          <span className="actions-note">The arm is still stopped: restart the deploy to resume.</span>
        )}
      </section>
    );
  }
  // Reset lives in the header ("Release e-stop" in E-STOP's place). Here only as the
  // fallback for a latch the header can't confirm (stale or missing safety status).
  const rest = mine.filter((a) => a.kind !== "estop_reset" || (latched && !swapped.current));
  const notes = [];
  if (hidden.role) notes.push(`Your role (${cfg.role}) can only watch and E-STOP.`);
  if (hidden.remote)
    notes.push(
      "Off over the network on this robot: its site turns them on with OPENRAL_DASHBOARD_REMOTE_CONTROL=1. E-STOP always works.",
    );
  if (cfg.mode === "lan") notes.push("LAN view: full control is on the robot's own screen.");
  if (!rest.length && !blocked.length && !notes.length) return null;
  return (
    <section className="actions" aria-label="Robot actions">
      {rest.map((a) => (
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
