# `galaxea_a1` — Hardware bring-up runbook

Real-only Galaxea A1 manifest (`robot.yaml`) and operator runbook. OpenRAL
stays ROS 2 / Python 3.12; the operator's official ROS 1 Noetic SDK runs out
of process behind a literal IPv4-loopback JSON-lines sidecar
(`python/hal/src/openral_hal/galaxea_a1.py::GalaxeaA1HAL`,
`tools/galaxea_a1_ros1_sidecar.py`, `tools/run_galaxea_a1_sidecar.sh`). No
vendor source, binary, or message package is distributed. See
[`docs/methods/01-hal.md`](../../docs/methods/01-hal.md) for the HAL's public
symbol inventory.

## Galaxea A1 hardware bring-up

The first session is observation-only until the HAL graph is healthy. Ensure no
other process/container owns the serial device, the arm workspace is clear, and
the physical e-stop is reachable.

```bash
# One-time: build OpenRAL's vendor-free Noetic runtime image. The official SDK
# is mounted at run time and is never copied into the image.
docker build \
  -t openral/galaxea-a1-sidecar:noetic \
  docker/galaxea_a1_sidecar

# One-time: build OpenRAL's standard public x86 deploy image (Jazzy/Python 3.12).
just docker-build-x86

# Read-only gate — checks the image, SDK, serial ownership, port, and lock.
tools/run_galaxea_a1_sidecar.sh \
  --image openral/galaxea-a1-sidecar:noetic \
  --sdk-root /absolute/path/to/A1_SDK \
  --serial /dev/a1 \
  --check-only

# Terminal 1 — isolated ROS 1 bridge network; only TCP 46011 reaches loopback.
tools/run_galaxea_a1_sidecar.sh \
  --image openral/galaxea-a1-sidecar:noetic \
  --sdk-root /absolute/path/to/A1_SDK \
  --serial /dev/a1

# Terminal 2 — OpenRAL's standard real-hardware path. The OpenRAL container uses
# host networking only for ROS 2 DDS and the sidecar's loopback TCP port; it owns
# no Galaxea serial device and cannot see the ROS 1 master inside the sidecar.
docker run --rm --name openral-galaxea-a1 --network host \
  --volume "$(pwd)/robots:/workspace/robots:ro" \
  --volume "$(pwd)/scenes:/workspace/scenes:ro" \
  --volume "$(pwd)/tests:/workspace/tests:ro" \
  openral:x86 \
  --config scenes/deploy/galaxea_a1_bench.yaml

# Terminal 3 — observation gate: six named joints update; diagnostics are clean.
docker exec openral-galaxea-a1 bash -lc \
  'source /opt/ros/jazzy/setup.bash && source /workspace/install/setup.bash && \
   ros2 topic echo /joint_states --once && ros2 topic echo /diagnostics --once'
```

An optional HAL-level HIL gate can run between Terminal 1 and the full deploy.
It opens one sidecar session, validates three fresh finite named-joint samples
plus cached motor health, and ends by verifying that downstream e-stop stops the
owned ROS 1 stack. Restart Terminal 1 afterwards:

```bash
GALAXEA_A1_HIL=1 just hil galaxea_a1
```

Only after that observation-only run passes, opt into a measured-current-pose
hold. Feedback within the tracked 0.01 rad endpoint tolerance is projected to
the exact command limit; any larger projection fails before publication. The
test also waits for the sidecar relay to report `ACTIVE`, proving the official
tracker has converged from its compiled `task.info` initial pose before any
host motor command is forwarded:

```bash
GALAXEA_A1_HIL=1 GALAXEA_A1_ALLOW_HOLD=1 just hil galaxea_a1
```

After the hold passes, a separate lab opt-in moves `arm_joint1` by +0.01 rad,
requires it to settle within 0.008 rad (covering the measured 0.007 rad
small-command residual), continuously bounds all six joint excursions, returns
to the measured start, and then performs the same downstream e-stop:

```bash
GALAXEA_A1_HIL=1 GALAXEA_A1_ALLOW_NUDGE=1 just hil galaxea_a1
```

The G2 gripper has its own opt-in. It uses the vendor example's 10 mm step,
mapped through the normalized `0..1` contract over the configured 104 mm
stroke, chooses the direction away from the nearest endpoint, verifies feedback
within the measured 2.5 mm steady-state tolerance, and returns to the measured
opening even when the outbound-leg assertion fails:

```bash
GALAXEA_A1_HIL=1 GALAXEA_A1_ALLOW_GRIPPER=1 just hil galaxea_a1
```

After the HAL-level gates pass, the full-graph HIL runs inside the deploy
container. It captures the current named-joint feedback itself, requires the
C++ kernel and real HAL to be active while the relay is still `LOCKED`, then
publishes only that measured hold through `candidate_action`. It verifies the
matching `safe_action`, exact staged/forwarded targets, zero kernel drops, and
less than one degree of drift. Its `finally` path publishes `/openral/estop`
three times and requires the HAL diagnostics to confirm the latch:

```bash
docker exec \
  --env GALAXEA_A1_DEPLOY_HIL=1 \
  --env GALAXEA_A1_ALLOW_HOLD=1 \
  openral-galaxea-a1 \
  bash -lc 'source /opt/ros/jazzy/setup.bash && \
    source /workspace/install/setup.bash && \
    pytest -q /workspace/tests/hil/test_galaxea_a1_deploy.py'
```

After the current-pose full-graph gate passes, the same fixture has a separate
motion opt-in. It moves `arm_joint1` by +0.01 rad through
`candidate_action -> C++ safety kernel -> safe_action`, bounds all six joint
excursions, and returns to the measured start before the downstream e-stop:

```bash
docker exec \
  --env GALAXEA_A1_DEPLOY_HIL=1 \
  --env GALAXEA_A1_ALLOW_HOLD=1 \
  --env GALAXEA_A1_ALLOW_NUDGE=1 \
  openral-galaxea-a1 \
  bash -lc 'source /opt/ros/jazzy/setup.bash && \
    source /workspace/install/setup.bash && \
    pytest -q /workspace/tests/hil/test_galaxea_a1_deploy.py'
```

This test intentionally ends the hardware session. Restart both the sidecar
and deploy container before any later motion test.

Do not start a policy on the first pass. Stop both commands and investigate if
feedback/status becomes stale, a motor code other than the manifest's explicit
idle/gripper masks appears, joint order differs, the sidecar exits, or the arm
moves before an approved safe action. Motion validation then proceeds with a
current-pose hold and a single <=0.01 rad joint increment through OpenRAL's
standard candidate-action → C++ kernel → safe-action path, then return-to-start;
only afterwards run an A1-specific rSkill.

## LingBot-VA rSkill through the complete OpenRAL path

`rskills/lingbot-va-galaxea-a1-fruit-placement/rskill.yaml` is the first
checkpoint-specific A1 rSkill. The dependency direction is deliberate:

```text
A1 Camera Bridge -> OpenRAL WorldState -> LingBot-VA rSkill
  -> A1 Runtime policy gateway (model contract + EEF/cache + IK)
  -> OpenRAL candidate_action -> C++ safety kernel -> safe_action
  -> GalaxeaA1HAL -> isolated ROS 1 sidecar -> official A1 driver
```

The A1 Runtime is a public capability provider, not a second controller:
start only its persistent camera owner, LingBot policy server, and OpenRAL
policy gateway. The gateway has no ROS imports or command publisher. Do not
start its LingBot ROS execution bridge or A1 joint runtime while OpenRAL owns
the deployment. The rSkill owns its `policy_extras.max_joint_substep_rad` replay
setting and reads the independent `max_target_step_rad` ceiling from the same
`RobotDescription` used to construct the HAL. Startup rejects a policy bound
that exceeds either the HAL's live target-step ceiling or locked-relay
alignment tolerance. The policy bound is 0.045 rad, below the 0.05 rad
locked-relay alignment threshold; the independent HAL/sidecar live limit is
0.08 rad. The gateway constructs Runtime's IK implementation with the active
OpenRAL `RobotDescription`'s ordered command limits after verifying they are no
wider than Runtime's envelope. Runtime calibration margins therefore cannot
widen the typed OpenRAL, safety-kernel, or official sidecar command envelope.

The gateway emits one bounded target per 30 Hz control tick. When its IK
solution is farther than 0.045 rad from fresh feedback, it keeps advancing
toward that same solved target on subsequent ticks and only consumes the next
model action after the solved target has been dispatched. The FK of the actual
dispatched target is written into the KV cache, so the policy state reflects
what OpenRAL commanded rather than an unreachable ideal. The official tracker's
steady-state error cannot widen the command envelope or bypass the bounded
step. The A1 Runtime's 1.70 rad IK-solution validation remains an upstream
reachability check, not a motor-command step limit.

```bash
# Terminal A — A1 Runtime capability providers only (no ROS command publisher).
cd /absolute/path/to/A1-Research
just cameras start
scripts/apps/lingbot/a1_lingbot_runtime.sh server
uv run galaxea-a1-openral-policy \
  --config configs/deployments/lingbot/fruit_placement_eef.toml \
  --repo-root .

# Terminal B — official ROS 1 sidecar, as in the bring-up section above.
cd /absolute/path/to/OpenRAL
tools/run_galaxea_a1_sidecar.sh \
  --image openral/galaxea-a1-sidecar:noetic \
  --sdk-root /absolute/path/to/A1_SDK \
  --serial /dev/a1

# Terminal C — the complete OpenRAL real deployment.
cd /absolute/path/to/OpenRAL
uv run --group lingbot openral deploy run \
  --config scenes/deploy/galaxea_a1_bench.yaml
```

Submit the exact trained prompt (for example, `put the red mango into the blue
plate`) through the dashboard. Before allowing a task motion, first repeat the
observation, hold, joint-nudge, gripper, and full-graph gates above. Stop the
LingBot server afterwards with
`scripts/apps/lingbot/a1_lingbot_runtime.sh server-stop`.

The A1 opts into hardware-downstream e-stop: `/openral/estop` stops the
sidecar-owned tracker and driver immediately. The generic
`/openral/estop_cleared` broadcast cannot re-arm this HAL; restart the lifecycle
and sidecar, re-read motor health, and repeat initial alignment instead.
