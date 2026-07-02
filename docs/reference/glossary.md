# Glossary

Terms as used in OpenRAL.

- **VLA** — Vision-Language-Action model. Maps `(images, language, state) → action[chunk]`. Examples: π0, GR00T N1.x, SmolVLA, OpenVLA-OFT.
- **WAM** — World Action Model. Generative simulator used for mental rollouts and failure anticipation. Examples: Cosmos Predict, Genie 3, IRASim, UnifoLM-WMA-0.
- **rSkill** — A packaged, capability-tagged unit of robot behavior (sigstore signing planned, not yet implemented — ADR-0006). One HF Hub repo. Loaded by `rSkill.from_pretrained(...)`.
- **HAL** — Hardware Abstraction Layer. The `openral_hal.HAL` Protocol; Python adapters live in `python/hal/`, per-robot ROS lifecycle nodes in `packages/openral_hal_<robot>/`.
- **WorldState** — A typed snapshot consumed by Skills and the Reasoner. Backed by tf2 + sensor topics.
- **Reasoner / S2** — The slow planning loop (LLM → typed tool calls).
- **Skill / S1** — The fast policy loop (VLA or scripted, action-chunked).
- **Cerebellum / S0** — The C++ realtime layer below S1, where applicable (humanoids).
- **Embodiment tag** — A short string (e.g., `g1`, `so100_follower`, `aloha`, `oxe_droid`) that maps a skill to a robot's heads / action layouts.
- **Trace** — An OTel span tree + LeRobotDataset row capturing one execution end-to-end.
- **Replanning ladder** — The bounded sequence: retry → param-tweak → substitute → goal-replan → human-handoff.
