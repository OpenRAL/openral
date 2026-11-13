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
  // On the robot no model is a choice (a deploy uses the account's); elsewhere a robot
  // has no reasoner until one is chosen (no library default, ADR-0088).
  if (!v?.set && data.scope !== "robot")
    return data.canWrite
      ? "Choose a model: robots have no reasoner until one is chosen."
      : "No model is chosen, so robots have no reasoner: an admin must choose one.";
  if (chosen?.needs && !isSet(chosen.needs)) return `${chosen.label} needs the ${label(chosen.needs)}.`;
  return "";
}

// What a setting shows: a secret only as set/not set with its last four characters.
export function shownValue(setting, value) {
  if (!value?.set) return "not set";
  return setting?.secret ? `set ${value.hint || ""}`.trim() : String(value.value ?? "");
}

// Open the left pane's Settings at a section ("reasoner"), from anywhere on the page:
// the Rail listens for this event. The section is highlighted when the pane opens.
export const OPEN_SETTINGS_EVENT = "openral:open-settings";
export function openSettings(section) {
  globalThis.dispatchEvent?.(new CustomEvent(OPEN_SETTINGS_EVENT, { detail: { section } }));
}

// Why the reasoner is not running for this robot, or "": its settings' problem (no
// model, a missing key), else whether its lifecycle start failed (`reasonerDown`).
export function reasonerNotConfigured(settingsData, lifecycleProblem) {
  const problem = settingsData ? modelProblem(settingsData) : "";
  if (problem) return `Reasoner is not configured. ${problem}`;
  if (lifecycleProblem) return "Reasoner is not configured, or it failed to start (see Events).";
  return "";
}
