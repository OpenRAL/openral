# Tests · HIL bridges

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

Lab-runner-only `rclpy` bridges that wire the real-HW HAL adapters
(`UR5eRealHAL` / `UR10eRealHAL` / `FrankaPandaRealHAL` / `SawyerRealHAL` /
`AlohaHAL`) onto a live `ros2_control` controller stack via the HALs'
injected `publish_fn` / `state_fn` callables. Off-lab they are guarded
behind `importlib.util.find_spec("rclpy") is None` plus a per-robot env
probe; the unit-lane conformance tests use `SimTransport` instead.

The Galaxea A1 uses an isolated ROS 1 sidecar rather than a
`ros2_control` bridge. Its HAL-level gate lives in
`tests/hil/test_galaxea_a1.py`; the separate
`tests/hil/test_galaxea_a1_deploy.py` gate runs inside an active deploy
container and proves a measured hold across the real C++ kernel, ROS 2 HAL,
and staged ROS 1 command relay. Both are environment-gated, time-bounded, and
end by stopping the sidecar through the downstream e-stop path.

The OpenArm has no `ros2_control` HIL bridge yet — its gate
(`tests/hil/test_openarm_can_live.py`) runs one layer lower, on the CAN
transport itself, and is gated on the two udev-pinned interfaces
(`openarm_left` / `openarm_right`) being up. It splits in two: transport
checks (CAN FD at 1 Mbit/s nominal / 5 Mbit/s data, `openral detect`
identifying the arm from the bus alone, `OpenArmRealHAL`'s preflight
passing) run whenever the links are up, while the motor round-trip
additionally requires a bus in `ERROR-ACTIVE` — an unpowered arm leaves
the links up but `ERROR-PASSIVE`, which is a skip, not a failure, because
it is a fact about the bench and not about the code. The round-trip is
read-only by construction: it queries motor state and never calls
`enable_all()`, so it cannot energise or move the arm.

`tests/hil/test_openarm_restock_deploy_preflight.py` is the cell-level gate
for the restocking deploy: the committed scene / robot manifest / rSkill, the
CAN links, the three physical cameras, and — since ADR-0102 — the
slot-dispatched 16-DoF vector. That last tier drives the **production**
dispatcher (`rskill_runner_node._dispatch_slots`) over the committed
manifest's `slots:` block into a real `OpenArmRealHAL` that has passed its bus
preflight, then checks the four controller messages reconstruct the policy's
own vector, that arrival order does not change them, and that a standalone
gripper action raises instead of vanishing. A policy value equal to its joint
index makes a misroute or side swap read as a mismatch rather than as a
plausible pose. It cannot move the arm whether or not the arm is powered:
`openarm_bringup`'s `ros2_control` stack is the only path from those topics to
`openarm_can`, and these tests never publish to ROS at all — the messages are
collected in-process through the adapter's own `publish_fn` seam. What it
therefore does **not** cover is command→motion: that the composed vector moves
the right joints by the right amount, and that the gripper physically
actuates, still needs a powered run.

### `tests/hil/_ros_control_transport.py`
_Single-controller bridge. Used by UR5e, UR10e, Franka Panda, Sawyer._

- `_make_trajectory_publisher(node, command_topic) -> Publisher` — Module-private helper that creates a `trajectory_msgs/JointTrajectory` publisher with the shared HIL QoS profile. Reused by `_aloha_ros_transport.py`. (L52)
- `RosControlHILTransport(node, joint_names, *, command_topic, joint_state_topic)` — Subscribes to `joint_state_topic`, exposes `publish(_topic, msg)` for the HAL's outgoing trajectories and `state()` for the cached joint state. Helpers: `spin_once`, `wait_for_first_state`, `last_stamp`. (L62)
- `make_hil_transport(node_name, joint_names, *, command_topic, joint_state_topic) -> tuple[Node, RosControlHILTransport, Callable[[], None]]` — Factory that initialises rclpy, creates the node, and returns a teardown callable. (L173)

### `tests/hil/_openarm_ros_transport.py`
_4-way fan-out bridge for the bimanual OpenArm v2 HAL — the only thing between `OpenArmRealHAL` and a physical arm._

- `OpenArmHILTransport(node, joint_names, *, command_topics, joint_state_topic, time_from_start_s=0.8)` — Four `JointTrajectory` publishers plus one aggregated `JointState` subscriber. Simpler than the ALOHA bridge because ADR-0102 puts `joint_names` **in the message**, so the transport forwards them rather than keeping a second copy of the slice table that could drift. Build it from `OpenArmRealHAL.ros2_control_joint_names()` — the URDF namespace (`openarm_left_joint1`) that `/joint_states` is keyed by, **not** the manifest's (`left_joint1`). (L50)
- `time_from_start_s` is a constructor argument, unlike the 100 ms the production transports hardcode. A `JointTrajectoryController` given an absolute target and a 100 ms deadline moves at `(target - current) / 0.1s`, so the rate is set by how wrong the command is. A longer window bounds it by construction; a test that cares about production timing passes 0.1.
- `publish(topic, msg)` dispatches by topic match and raises on an unknown topic or a name/value width mismatch — silently dropping a command is the ADR-0102 failure mode itself. `state()` zero-fills joints it has never heard from; `missing_joints()` / `wait_for_every_joint()` are what let a caller tell that apart from a real pose.
- Its own tests (`tests/hil/test_openarm_ros_transport.py`) need only a ROS install, not the cell: real publishers and real messages over DDS on an isolated domain with LOCALHOST discovery (#227 — a stray graph once reached a live OpenArm on another host, and this suite's whole subject is arm command topics).

### `tests/hil/test_openarm_bringup_agreement.py`
_`OpenArmRealHAL`'s controller/joint table vs `openarm_bringup`'s own YAML — no hardware, not even the CAN links._

- The HAL's `_command_groups` is a hand-copied transcription of names that live in a **different repo** on a different cadence, and nothing compared the two. A `JointTrajectoryController` handed joints it does not own **rejects the whole message**: no publisher-side exception, nothing odd on `/joint_states`, the arm just does not move — indistinguishable from a policy holding still. Same failure shape ADR-0102 ended, one layer further out.
- The gripper is the trap and the reason the file exists: bringup calls it `openarm_left_finger_joint1`, **not** `openarm_left_gripper`. Every plausible guess is wrong, so the config is the only safe source.
- Four checks: every controller the HAL publishes to is declared; each is a `JointTrajectoryController` (a `ForwardCommandController` takes `Float64MultiArray` on `/commands`, so the HAL's topic and message type would both miss); the joint names match per controller; and the four lists concatenate to exactly the HAL's 16-DoF order, which is what the slice spans mean.
- Gated on `openarm_bringup` being on the ament prefix path, so it skips cleanly off-rig. It answers **statically** what would otherwise need a powered cell: bringing the controllers up to look at them runs `OpenArmHW::on_activate` → `openarm_->enable_all()`, energising all sixteen motors. Reading the config energises nothing.

### `tests/hil/test_openarm_slot_group_motion.py`
_The one test in the tree that commands a real OpenArm to move._

- Closes the last ADR-0102 gate: `test_openarm_restock_deploy_preflight.py` proves the composed 16-DoF vector reaches the four controllers correctly named and sliced, but never publishes, so it cannot see a sign flip, a scale error, or a joint the controller ignores. This one commands measured-pose + `0.02 rad` on **one** joint, holds the other fifteen at their measured values, and reads the physical result back off `/joint_states` — asserting the addressed joint arrived and every other joint stayed put.
- Two independent gates, both explicit: `OPENRAL_OPENARM_ALLOW_MOTION=1` (this bench can move) and `OPENRAL_OPENARM_ATTENDED=1` (someone is at the E-stop right now). Separate on purpose, so a rig that leaves the first exported does not thereby become a rig that moves unattended.
- Why a small number is not by itself a small motion: the wire format is an **absolute** position with a deadline, so travel is `(target - measured) / time_from_start` — set by how wrong the command is, which is the quantity under test. The bounds that do work are the measured-pose baseline, the 0.8 s window, and the refusal to start until all 16 joints have appeared on `/joint_states` (a partial state zero-fills into a plausible pose, and "measured + delta" on top of that is a command to slew the cell home).
- Cases run gripper-first (1-DoF, lowest inertia, and the actuator ADR-0102 exists for) then the most distal arm joint, and each restores the measured pose before returning, so the suite is idempotent.

### `tests/hil/_aloha_ros_transport.py`
_4-way fan-out bridge for the bimanual ALOHA HAL._

- `AlohaHILTransport(node, joint_names, *, left_arm_command_topic, right_arm_command_topic, left_gripper_command_topic, right_gripper_command_topic, joint_state_topic)` — Owns four `JointTrajectory` publishers (two arms 6-DOF + two grippers 1-DOF — Trossen gripper modeled as a 1-DOF `JointTrajectoryController`) and one aggregated `JointState` subscriber. `publish(topic, msg)` dispatches by topic match; unknown topics raise `ValueError`. (L62)
- `make_aloha_hil_transport(node_name, *, left/right arm + gripper command topics, joint_state_topic) -> tuple[Node, AlohaHILTransport, Callable[[], None]]` — Factory; derives the canonical 14 joint names from the public `ALOHA_REAL_DESCRIPTION.joints` so the bridge stays in lock-step with the HAL manifest. (L223)
