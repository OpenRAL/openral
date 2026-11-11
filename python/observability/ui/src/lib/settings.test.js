import assert from "node:assert/strict";
import test from "node:test";
import { modelProblem, shownValue } from "./settings.js";
import { workspaceLabel } from "./viewer.js";

// The shape GET /api/settings returns (website functions/api/settings.js).
const SETTINGS = {
  ANTHROPIC_API_KEY: { secret: true, label: "Anthropic API key" },
  OPENAI_API_KEY: { secret: true, label: "OpenAI API key" },
  OPENRAL_REASONER_MODEL: {
    secret: false,
    label: "Reasoner model",
    options: [
      { id: "claude-sonnet-5-5", label: "Claude Sonnet 5.5", needs: "ANTHROPIC_API_KEY" },
      { id: "gpt-5.5", label: "GPT-5.5", needs: "OPENAI_API_KEY" },
      { id: "cosmos3-edge", label: "Cosmos 3 Edge", needs: null },
    ],
  },
};
const data = (values) => ({ scope: "org", canWrite: true, settings: SETTINGS, values });

test("the model needs a key, a curated id, and its own provider's key", () => {
  assert.match(modelProblem(data({})), /Set an Anthropic or OpenAI API key/);
  const anth = { ANTHROPIC_API_KEY: { set: true, hint: "…abcd" } };
  assert.equal(modelProblem(data({ ...anth, OPENRAL_REASONER_MODEL: { set: true, value: "claude-sonnet-5-5" } })), "");
  assert.match(modelProblem(data({ ...anth, OPENRAL_REASONER_MODEL: { set: true, value: "gpt-5.5" } })), /GPT-5.5 needs the OpenAI API key/);
  assert.match(modelProblem(data({ ...anth, OPENRAL_REASONER_MODEL: { set: true, value: "llama" } })), /isn't a model robots run/);
  // An on-device model needs no cloud key.
  assert.equal(modelProblem(data({ OPENRAL_REASONER_MODEL: { set: true, value: "cosmos3-edge" } })), "");
});

test("a model that runs on the robot needs no key", () => {
  const onRobot = { id: "cosmos3-edge", label: "NVIDIA Cosmos 3 Edge", needs: null };
  const withLocal = { ...SETTINGS, OPENRAL_REASONER_MODEL: { ...SETTINGS.OPENRAL_REASONER_MODEL, options: [...SETTINGS.OPENRAL_REASONER_MODEL.options, onRobot] } };
  const d = (values) => ({ scope: "org", canWrite: true, settings: withLocal, values });
  assert.equal(modelProblem(d({})), "");
  assert.equal(modelProblem(d({ OPENRAL_REASONER_MODEL: { set: true, value: "cosmos3-edge" } })), "");
  assert.match(modelProblem(d({ OPENRAL_REASONER_MODEL: { set: true, value: "gpt-5.5" } })), /GPT-5.5 needs the OpenAI API key/);
});

test("a secret only ever shows as set with its last four characters", () => {
  assert.equal(shownValue(SETTINGS.ANTHROPIC_API_KEY, { set: true, hint: "…abcd" }), "set …abcd");
  assert.equal(shownValue(SETTINGS.ANTHROPIC_API_KEY, null), "not set");
  assert.equal(shownValue(SETTINGS.OPENRAL_REASONER_MODEL, { set: true, value: "gpt-5.5" }), "gpt-5.5");
});

test("the pane's top names the org, else the person, else the robot", () => {
  assert.deepEqual(workspaceLabel({ org: "acme", orgName: "Acme Robotics" }, {}, "amr"), { name: "Acme Robotics", initial: "A", logo: null });
  assert.equal(workspaceLabel({ org: "acme", orgName: "Acme", orgLogo: "/org-logo/acme.png" }, {}).logo, "/org-logo/acme.png");
  assert.equal(workspaceLabel({ orgName: "Acme", orgLogo: "javascript:alert(1)" }, {}).logo, null);
  assert.equal(workspaceLabel({ orgName: "Acme", orgLogo: "//evil.example/x.png" }, {}).logo, null);
  assert.equal(workspaceLabel({ orgName: "Acme", orgLogo: "/\\evil.example/x.png" }, {}).logo, null);
  assert.equal(workspaceLabel({ orgName: "Acme", orgLogo: "https://cdn.openral.com/a.png" }, {}).logo, "https://cdn.openral.com/a.png");
  assert.equal(workspaceLabel({ name: "Ada Lovelace", org: null }, { mode: "remote" }).name, "Ada Lovelace");
  assert.equal(workspaceLabel(null, { org: "acme" }, "amr").name, "acme");
  assert.deepEqual(workspaceLabel(null, { org: "" }, "openamrobot_v2"), { name: "openamrobot_v2", initial: "O", logo: null });
});
