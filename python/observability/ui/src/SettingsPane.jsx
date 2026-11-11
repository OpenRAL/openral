import { useEffect, useRef, useState } from "react";
import { modelProblem, shownValue } from "./lib/settings.js";

// Settings, opened from the left pane's ⋯ menu: the reasoner's provider API keys
// (Anthropic, OpenAI) and its model, edited in place. On the website these are the
// account's and the org's settings (the site's GET/PUT /api/settings: only an org admin
// writes the org's); on the robot they are the robot's own (the robot's /api/settings,
// written only by the operator at the robot). Either way a secret only ever comes back
// as "set …abcd", and robots use the values from their next deploy start.
// On the website the pane holds two scopes and names them; on the robot only one.
const TITLES = { org: "Organization", me: "Your own robot" };

function Scope({ which, url = `/api/settings?scope=${which}` }) {
  const [data, setData] = useState(null);
  const [draft, setDraft] = useState({});
  const [msg, setMsg] = useState({ text: "", kind: "" });
  useEffect(() => {
    let live = true;
    fetch(url, { credentials: "same-origin" })
      .then((r) => (r.ok ? r.json() : null))
      .then((b) => live && setData(b), () => {});
    return () => {
      live = false;
    };
  }, [url]);
  const save = async (name, value) => {
    try {
      const r = await fetch(url, {
        method: "PUT",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, value }),
      });
      const out = await r.json().catch(() => ({}));
      if (!r.ok) return setMsg({ text: out.error || "Could not save.", kind: "err" });
      setData(out);
      setDraft((d) => ({ ...d, [name]: "" }));
      setMsg({ text: value ? "Saved. Robots use it from their next deploy start." : "Cleared.", kind: "ok" });
    } catch {
      setMsg({ text: "Could not save: check your connection and try again.", kind: "err" });
    }
  };
  if (!data || data.scope !== which) return null; // no org: only your own
  const problem = modelProblem(data);
  return (
    <section className="set-scope" aria-label={TITLES[which] || "Reasoner"}>
      {TITLES[which] && <h3>{TITLES[which]}</h3>}
      {Object.entries(data.settings).map(([name, s]) => {
        const v = data.values[name];
        if (Array.isArray(s.options)) {
          const isSet = (n) => Boolean(data.values[n]?.set);
          const chosen = s.options.find((o) => o.id === v?.value);
          return (
            <div key={name} className="set-row">
              <span className="set-label">{s.label}</span>
              <select
                value={chosen ? chosen.id : ""}
                disabled={!data.canWrite}
                onChange={(e) => save(name, e.target.value)}
                aria-label={s.label}
              >
                <option value="" disabled>
                  Choose a model
                </option>
                {s.options.map((o) => (
                  <option key={o.id} value={o.id} disabled={Boolean(o.needs) && !isSet(o.needs)}>
                    {o.label}
                    {o.needs && !isSet(o.needs) ? ` (needs the ${data.settings[o.needs]?.label || o.needs})` : ""}
                  </option>
                ))}
              </select>
              {problem && (
                <small className="set-err" role="alert">
                  {problem}
                </small>
              )}
            </div>
          );
        }
        return (
          <div key={name} className="set-row">
            <span className="set-label">
              {s.label} <span className="set-value">{shownValue(s, v)}</span>
            </span>
            {data.canWrite && (
              <form
                className="set-edit"
                onSubmit={(e) => {
                  e.preventDefault();
                  save(name, draft[name] || "");
                }}
              >
                <input
                  type={s.secret ? "password" : "text"}
                  autoComplete="off"
                  maxLength={4096}
                  placeholder={v?.set ? "Replace" : "Paste the key"}
                  value={draft[name] || ""}
                  onChange={(e) => setDraft({ ...draft, [name]: e.target.value })}
                  aria-label={s.label}
                />
                <button className="btn" type="submit" disabled={!draft[name]}>
                  Save
                </button>
                {v?.set && (
                  <button className="btn" type="button" onClick={() => window.confirm(`Clear the ${s.label}?`) && save(name, "")}>
                    Clear
                  </button>
                )}
              </form>
            )}
          </div>
        );
      })}
      {msg.text && (
        <p className={`set-msg ${msg.kind}`} role="status">
          {msg.text}
        </p>
      )}
    </section>
  );
}

export function SettingsPane({ onWebsite, reasonerModel, onClose }) {
  const ref = useRef(null);
  useEffect(() => {
    ref.current?.focus();
    const esc = (e) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", esc);
    return () => document.removeEventListener("keydown", esc);
  }, [onClose]);
  return (
    <section className="settings-pane" role="dialog" aria-modal="false" aria-labelledby="settings-title" tabIndex={-1} ref={ref}>
      <header className="settings-hd">
        <h2 id="settings-title">Settings</h2>
        <button type="button" className="rail-toggle" aria-label="Close settings" onClick={onClose}>
          ×
        </button>
      </header>
      <h3 className="settings-sub">Reasoner</h3>
      {onWebsite ? (
        <>
          <Scope which="org" />
          <Scope which="me" />
        </>
      ) : (
        <>
          <Scope which="robot" url="/api/settings" />
          <div className="set-row">
            <span className="set-label">
              Model this deploy runs <span className="set-value">{reasonerModel || "no reasoner tick yet"}</span>
            </span>
          </div>
        </>
      )}
    </section>
  );
}
