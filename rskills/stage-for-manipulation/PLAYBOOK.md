# stage-for-manipulation

> **Hand-authored decision procedure (SOP).** Unlike the generated `SKILL.md`
> discovery view, this file is the *content the S2 Reasoner reads and follows*.
> It is injected into the reasoner's system prompt when this playbook is
> installed. The `rskill.yaml` `playbook.body_uri` points here. ADR-0071.

## Trigger
A chosen manipulation rSkill declares a `starting_pose` (pre-grasp) that the
robot has **not** yet reached (e.g. a pick policy expects the gripper hovering
above the black bowl, but the arm is parked at home). Dispatching the
manipulation from a bad initial pose is a common, avoidable grasp failure.

## Preconditions
- The target manipulation skill and its `starting_pose` are known (read from the
  skill's manifest / contract).
- An arm-motion **approach** skill is installed — the collision-aware MoveGroup
  plan-arm rSkill (`openral-moveit-plan-arm`, ADR-0051) — and/or a navigate skill
  for mobile bases.

## Steps
1. **Read the pre-grasp.** Read the target manipulation skill's `starting_pose`
   (the declared pre-grasp the policy expects to begin from).
2. **Base staging (mobile only).** If the robot has a mobile base, `resolve_place`
   a stand pose that puts the target inside the arm's workspace, then
   `execute_rskill(NAVIGATE, goal=place)`. Skip on a fixed-base arm.
3. **Approach to pre-grasp.** `execute_rskill` the collision-aware **approach**
   skill (ADR-0051 — the MoveGroup plan-arm rSkill) retargeted at `starting_pose`,
   so the arm moves to the pre-grasp under MoveIt. **Never** a hand-rolled IK.
4. **Verify.** `query_scene` to confirm the pre-grasp ("is the gripper positioned
   above the black bowl?"). Treat an unconfirmed pose as not staged.
5. **Hand back.** Only once verified, return control so the manipulation policy
   runs from a good initial pose.

## Verify (done predicate)
The gripper (and base, for mobile embodiments) are at the declared `starting_pose`
pre-grasp, **confirmed by `query_scene`**. An unverified pose is a failure, not a
success — do not hand control to the manipulation policy.

## Fallbacks
- The approach skill cannot plan a collision-free path to `starting_pose`
  (obstructed / unreachable pre-grasp) → `emit_prompt` that staging failed, so the
  reasoner replans or hands off rather than dispatching a manipulation from a bad
  pose. This is the terminal human-handoff rung of the replanning ladder.
- Pairs with the **preflight-reach** playbook (which checks reachability before a
  skill is even chosen); this one stages and verifies the chosen skill's pose.
- **Never** loop past `max_steps`. Every approach attempt and its outcome are on
  the OTel trace, so staging is replayable.

## Safety
This playbook only *decides* and *sequences*. Every motion it triggers is an
`execute_rskill` → Action chunk that still crosses the C++ safety kernel
(ADR-0030); a bad `starting_pose` yields a plan the kernel still vetoes, never a
relaxed check (CLAUDE.md §1.1).
