# `openral_hal_galaxea_a1`

ROS 2 lifecycle host for the real-only Galaxea A1 HAL. The Python HAL talks to
`tools/galaxea_a1_ros1_sidecar.py`; the sidecar runs in an operator-provisioned
ROS Noetic environment and owns the official SDK driver and joint tracker.
The HAL accepts only literal IPv4 loopback sidecar addresses and strict integer
TCP ports; remote, IPv6, and DNS-resolved endpoints are rejected before any
connection attempt.

The vendor SDK is not bundled. See `docs/methods/01-hal.md` for the deployment
and hardware bring-up sequence.

Before opening the serial device, run the launcher's read-only preflight:

```bash
tools/run_galaxea_a1_sidecar.sh \
  --image openral/galaxea-a1-sidecar:noetic \
  --sdk-root /absolute/path/to/A1_SDK \
  --serial /dev/a1 \
  --check-only
```

The SDK mount stays read-only. The official joint tracker writes its generated
CppAD artifacts only to the launcher's dedicated XDG cache under the operator's
home directory.

The lab-gated HIL test uses one sidecar session and ends by stopping the owned
ROS 1 stack:

```bash
GALAXEA_A1_HIL=1 just hil galaxea_a1
```

Set `GALAXEA_A1_ALLOW_HOLD=1` only after the observation-only run passes. It
replays the measured current joint pose and verifies less than one degree of
drift. A feedback value within the tracked endpoint tolerance is projected to
the exact command limit; a larger projection is rejected before publication.

After the hold gate passes, `GALAXEA_A1_ALLOW_NUDGE=1` performs a bounded
`arm_joint1` +0.01 rad round trip, while `GALAXEA_A1_ALLOW_GRIPPER=1` performs
the official G2 example's 10 mm gripper step and returns to the measured
opening. Each gate implies the hold gate and ends through the downstream
e-stop path.

The final deployment gate is `tests/hil/test_galaxea_a1_deploy.py`. Run it
inside an active `openral deploy run` container with both
`GALAXEA_A1_DEPLOY_HIL=1` and `GALAXEA_A1_ALLOW_HOLD=1`. It verifies the
measured hold through the C++ Safety Kernel and ROS 1 relay, then intentionally
ends the session through `/openral/estop`.
