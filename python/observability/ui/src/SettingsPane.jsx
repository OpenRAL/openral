import { useEffect, useRef, useState } from "react";
import { modelProblem, shownValue } from "./lib/settings.js";

// Settings, opened from the left pane's ⋯ menu: the reasoner's provider API keys
// (Anthropic, OpenAI) and its model. On the website these are the account's and the
// org's settings (GET/PUT /api/settings, the same rules as the account page: only an
// org admin writes the org's, a secret only ever comes back as "set …abcd"); robots
// read them when a deploy starts. On the robot itself there is no account: the pane
// says where they are set and which model this deploy runs.
function Scope({ which }) {
  const url = `/api/settings?scope=${which}`;
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
    <section className="set-scope" aria-label={which === "org" ? "Organization" : "Your own robot"}>
      <h3>{which === "org" ? "Organization" : "Your own robot"}</h3>
      <p className="set-note">
        {which === "org" ? "Your organization's robots" : "The robot on your own account"} read these when a deploy starts; a value
        set on the robot itself (an environment variable) wins.{data.canWrite ? "" : " Only an org admin can change them."}
      </p>
      {Object.entries(data.settings).map(([name, s]) => {
        const v = data.values[name];
        if (Array.isArray(s.options)) {
          const isSet = (n) => Boolean(data.values[n]?.set);
          const chosen = s.options.find((o) => o.id === v?.value);
          return (
            <label key={name} className="set-row">
              <span className="set-label">{s.label}</span>
              <select value={chosen ? chosen.id : ""} disabled={!data.canWrite} onChange={(e) => save(name, e.target.value)}>
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
            </label>
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
        <div className="set-scope">
          <p className="set-note">
            This robot reads its reasoner's API key and model when a deploy starts: from your account's or organization's settings
            on openral.com (Account → Settings) once the robot is linked, or from environment variables on the robot
            (<code>OPENRAL_REASONER_MODEL</code>, <code>OPENRAL_REASONER_API_KEY</code>), which win.
          </p>
          <div className="set-row">
            <span className="set-label">
              Model in use <span className="set-value">{reasonerModel || "no reasoner tick yet"}</span>
            </span>
          </div>
        </div>
      )}
    </section>
  );
}
