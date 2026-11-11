// The reasoner settings the website keeps for a robot (GET/PUT /api/settings,
// website server/settings.js): provider API keys (secret: only "set …abcd" ever comes
// back) and the model, one of OpenRAL's curated models, each usable once its
// provider's key is set. Pure helpers for the settings pane.

// Why the model setting can't be used as it is, or "".
export function modelProblem(data) {
  const entry = Object.entries(data?.settings || {}).find(([, s]) => Array.isArray(s.options));
  if (!entry) return "";
  const [name, s] = entry;
  const v = data.values?.[name];
  const isSet = (n) => Boolean(data.values?.[n]?.set);
  const label = (n) => data.settings[n]?.label || n;
  const chosen = s.options.find((o) => o.id === v?.value);
  // A model that needs no key (one that runs on the robot) is always usable.
  if (!s.options.some((o) => !o.needs || isSet(o.needs))) return "Set an Anthropic or OpenAI API key to choose a model.";
  if (v?.set && !chosen) return `"${v.value}" isn't a model robots run: pick one.`;
  if (chosen?.needs && !isSet(chosen.needs)) return `${chosen.label} needs the ${label(chosen.needs)}.`;
  return "";
}

// What a setting shows: a secret only as set/not set with its last four characters.
export function shownValue(setting, value) {
  if (!value?.set) return "not set";
  return setting?.secret ? `set ${value.hint || ""}`.trim() : String(value.value ?? "");
}
