# Glossary

Terms as used in OpenRAL, alphabetical.

- **ADR**: Architecture Decision Record. The log is private; docs cite bare `ADR-NNNN` ids. See [Design decisions](../decisions.md).
- **Cerebellum / S0**: The C++ realtime layer below S1 (500-1000 Hz, inside `ros2_control`), for humanoids. Never an rSkill.
- **Deadman watchdog**: Independent process that fires E-stop when `/openral/safe_action` goes silent past a deadline; the deploy graph will not start without it.
- **Deploy sim / digital twin**: `openral deploy sim`: the real deploy graph (HAL, safety kernel, reasoner) with a MuJoCo or Isaac twin in place of hardware.
- **DeployScene / SimScene / BenchmarkScene**: The three scene YAML schemas under `scenes/deploy`, `scenes/sim` and `scenes/benchmark`.
- **Embodiment tag**: A short string (`g1`, `so100_follower`, `aloha`, `widowx`) that ties a skill to a robot's action layout. Must match the robot's `embodiment_tags`.
- **HAL**: Hardware Abstraction Layer. The `openral_hal.HAL` Protocol; adapters live in `python/hal/`; every robot runs the one manifest-driven node in `packages/openral_hal_node/` (ROS node name `openral_hal_<robot_id>`).
- **OpenRAL Pro**: The private commercial monorepo: weight formats of 8 bits or fewer, TensorRT/NVMM fast path, WAM, fleet/cloud dispatch, relaxing contact exemptions. Plugs in through entry points.
- **Reasoner / S2**: The slow planning loop: an LLM emitting typed `ReasonerToolCall`s. No actuation authority.
- **Replanning ladder**: Bounded sequence: retry, param-tweak, substitute skill, goal-replan, human handoff.
- **RobotDescription**: The Pydantic manifest (`robots/<id>/robot.yaml`) describing a robot: joints, links, sensors, embodiment tags, HAL classes.
- **RobotUnit overlay**: Per-host file `robots/<id>/units/<unit>.yaml` that holds a physical unit's calibrated sensor extrinsics and serial.
- **rSkill**: A packaged, capability-tagged unit of robot behavior: one Hugging Face Hub repo with `rskill.yaml`, weights and `eval/`. Loaded by `rSkill.from_pretrained(...)`. Provenance signing is planned, not implemented.
- **rSkill catalog**: The `OpenRAL` org on the Hub; find skills with `openral rskill search`. `rskills/` in this repo is only the worked-examples subset.
- **rSkill kind**: What an rSkill is: `vla`, `detector`, `vlm`, `reward`, `depth_estimator`, `ros_action`, `segmenter` or `playbook`. The kind decides its slot (S1 or S2) and whether the reasoner may dispatch it.
- **Safety kernel**: The separate, deny-by-default C++ process that validates every action chunk before it reaches the HAL. Python proposes; C++ disposes.
- **Skill / S1**: The fast policy loop (a VLA or scripted skill, action-chunked, 30-200 Hz).
- **Trace**: An OTel span tree plus LeRobotDataset row capturing one execution end to end.
- **Tracking mode**: A scene's `object_tracking` setting: `off`, `tracking` (open; the tracker only adds checked geometry and contact rules come from scene declarations) or `on` (Pro).
- **VLA**: Vision-Language-Action model: `(images, language, state)` to an action chunk. Examples: pi0, GR00T N1.x, SmolVLA, OpenVLA-OFT.
- **WAM**: World Action Model: a generative simulator for mental rollouts and failure anticipation (Cosmos Predict, Genie 3, UnifoLM-WMA-0). Implementations are Pro-only.
- **WorldState**: A typed snapshot of tf2 and sensor topics consumed by skills and the reasoner.
