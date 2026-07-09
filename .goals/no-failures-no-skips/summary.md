# Goal Summary: No failures or hidden skips in provisioned tests

## What was achieved

- The locally runnable default pytest scope was brought to zero failures in the verified runs.
- Strict provisioned lanes now run through `tools/verify_test_envs.py` and fail on any skip.
- Hardware, proprietary sidecar, remote-code, and physical-asset lanes remain explicit blockers instead of fake passes.
- Normal local provisioning gaps now have clear skip guards or strict lane mappings.
- Targeted selector/tooling quality gates passed.

## Acceptance criteria

- [x] Default locally runnable pytest suite has no failures after the available ROS overlay is sourced when needed.
- [x] `tools/verify_test_envs.py` verifies locally provisionable lanes with passing tests and zero skips.
- [x] Hardware/proprietary sidecar/unavailable asset tests are explicit preflight blockers or documented non-default gates.
- [x] Fixable skips were routed into strict lanes or converted to clear intentional gates.
- [x] Targeted quality gates for selector and verifier tooling pass.

## Iteration history

1. Iteration 1 failed orchestration review because the Inspector noted remaining full-suite failures despite returning PASS.
2. Iteration 2 passed: Builder fixed remaining default-suite failures and Inspector verified zero failures plus strict provisioned lanes.
3. Iteration 3 passed after a fresh broader run exposed ROS/DDS race failures; Builder fixed the races and Inspector verified `3774 passed, 112 skipped, 0 failures`.

## Key issues raised and resolved

- PEP 735 `{ include-group = ... }` entries broke tests that treated dependency group entries as strings; tests now expand include-group references before comparison.
- `lerobot` was importable while its `datasets` extra was missing, causing runtime failures; affected tests now skip at collection with a clear dependency reason.
- The Franka tabletop test referenced a missing scene path; a real sim scene fixture now backs that test path.
- Strict lowering was narrowed to installable no-skip targets; mixed vendor/fixture-absence checks stay out of the zero-skip lane.
- OpenArm lifecycle tests published VOLATILE ROS messages before DDS discovery completed; tests now wait for subscriber matching before publishing.
- World-state high-load testing used one rclpy publisher concurrently from multiple threads; test publishers are now serialized so the aggregator, not torn DDS serialization, is under test.

## Recommendations

- Keep strict lanes small and zero-skip; mixed audit suites should stay outside `verify_test_envs.py`.
- Do not merge LIBERO and RoboCasa dependency groups; their `robosuite` constraints conflict.
- Treat HIL and proprietary sidecars as explicit preflight lanes unless the lab runner actually provisions them.
