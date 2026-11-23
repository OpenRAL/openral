# Galaxea R1 Pro (BEHAVIOR-1K simulation)

This manifest describes the simulated R1 Pro used by the 2026 BEHAVIOR-1K
challenge. Joint position, velocity, and effort limits come from the upstream
`behavior-1k/omnigibson-robot-assets` `r1pro.urdf`.

`assets.urdf` is `behavior:r1pro`: that same `r1pro.urdf` at the pinned dataset
revision, downloaded on first use, with absolute mesh paths and without the
collision meshes BEHAVIOR does not publish — visual geometry only
(`openral_hal._behavior_r1pro_assets`).

The robot is **simulation-only** in OpenRAL. `hal.real` remains null; no physical
R1 Pro safety or transport claim is made.

```bash
openral deploy sim --config scenes/deploy/behavior_r1pro.yaml \
  --initial-task "turn on the radio"
```

The scene sidecar requires the official BEHAVIOR environment via
`OPENRAL_BEHAVIOR_SIDECAR_PYTHON`. The GR00T policy sidecar requires the pinned
Isaac-GR00T environment and organizer checkpoint documented in
`rskills/gr00t-n17-b1k-turning-on-radio/README.md`.
