# Goal: No failures or hidden skips in provisioned tests

## User Request

Ensure there are no more failures and that no tests skip given the correct provisioning. Use parallel agents if necessary.

## Refined Goal

Make the OpenRAL test workflow reliable by eliminating current test failures and ensuring tests do not silently skip once their required environment is provisioned. Locally provisionable dependency lanes must run with zero skips. Hardware and proprietary/external sidecar lanes must remain explicit preflight-gated blockers when missing; do not fake devices, sidecars, assets, or proprietary simulators.

## Acceptance Criteria

- [ ] The default locally runnable pytest suite has no failures after sourcing the available ROS overlay when needed.
- [ ] `tools/verify_test_envs.py` can verify every locally provisionable lane it owns with passing tests and zero skips.
- [ ] Tests that still require hardware, proprietary sidecars, remote-code opt-ins, or unavailable physical assets are either covered by explicit verifier preflight blockers or documented as intentional non-default gates.
- [ ] Any skip that can be removed by normal local provisioning is fixed, moved into the correct strict dependency lane, or excluded from strict lanes only when it is an intentional negative/deferred test.
- [ ] Targeted quality gates for the changed tooling and selectors pass.

## Scope Boundaries

**In scope:**
- Test failures surfaced by current full local pytest runs.
- Skip classification and routing for optional dependency groups, ROS/live rclpy gates, lowering/assets, sidecars, and HIL tests.
- `tools/verify_test_envs.py`, selective-test mapping, CI lane wiring, docs, and minimal test fixes needed for the acceptance criteria.
- Running locally provisionable environments on this host.

**Out of scope:**
- Faking or auto-provisioning proprietary simulators, model sidecars, hardware devices, lab robots, or unavailable physical assets.
- Disabling safety checks, weakening HIL gates, or pretending hardware tests passed without the device.
- Large unrelated refactors, new dependency abstractions, or broad test-suite rewrites.

## Applicable Project Conventions

**Quality gate command:**
- `just lint`
- `just test`
- `just sim` where applicable
- Targeted alternatives are acceptable for this goal when full gates require unavailable hardware/proprietary provisioning, but the exact commands run must be reported.

**Commit convention:**
- Conventional Commits, e.g. `feat(scope): ...`, `fix(scope): ...`, `chore(scope): ...`
- Builder commits must use `type(scope): [B] description`
- Inspector commits must use `chore(scope): [I] description`
- Assisted-by trailer required: `Assisted-by: Claude:Sonnet-4.6` for Builder and `Assisted-by: Claude:Haiku-4.5` for Inspector

**Guidelines:**
- `AGENTS.md` redirects to `CLAUDE.md`
- `CLAUDE.md`
- `docs/contributing/selective-testing.md`

**Rules:**
- Safety beats helpfulness; never bypass E-stop, deadman, or safety checks.
- Tests are part of the change; no mocks/stubs/smoke substitutes for real schemas/manifests/simulators where required.
- Use `just sync`, not bare `uv sync`.
- Keep changes minimal and root-cause oriented; reuse existing helpers/patterns before adding code.
- Update docs with directly related tooling behavior.
