#!/usr/bin/env bash
# The RoboCasa sim suite — the one list that defines what it is.
#
# `just test-robocasa-sim` execs this file. No CI lane exists; runs on a
# developer host only. Needs, at once: a colcon-built overlay (spawns the
# real `safety_kernel_node`), MuJoCo, and a provisioned RoboCasa kitchen.
#
# Why no CI lane:
# - Disk: RoboCasa assets are 23 GB (aigen_objs 13 GB, objaverse 6.2 GB,
#   lightwheel 1.5 GB, fixtures 1.4 GB, generative_textures 1.2 GB, textures
#   521 MB; per-bundle from utexas.box.com, no sub-bundle granularity). The
#   only hosted image with a colcon overlay (`docker-build`) is already
#   25.2 GB on a runner that reclaims ~14 GB by `rm -rf`-ing dotnet/android/
#   CodeQL; even a trimmed ~8 GB set (fixtures+textures+one bundle) doesn't
#   fit. No GPU needed — disk is the only constraint.
# - Security: a self-hosted runner was built, registered, then REMOVED. This
#   is a PUBLIC repo; a runner label is a routing request, not access
#   control — any workflow naming the label can claim the runner, and for
#   `pull_request` GitHub executes the workflow definition from the fork's
#   ref. Three fork-reachable `pull_request` workflows exist (`dco.yml`,
#   `quality.yml`, `test-selective.yml`); a fork PR could retarget one to
#   `runs-on: [self-hosted, <label>]` for arbitrary code with the runner's
#   SSH keys, `gh` credentials, and LAN access to the lab robots. Fixing
#   this needs an org runner group, unavailable on this org's GitHub Free
#   plan.
#
# `tests/unit/test_robocasa_sim_targets.py` keeps this list honest (asserts
# every `importorskip("robocasa")`-gated test appears below); the rest of
# `tests/sim/` is manual by declared policy
# (`.github/workflows/test-selective.yml`).
#
# Run:
#   source /opt/ros/jazzy/setup.bash && just ros2-build \
#     && source install/setup.bash && just test-robocasa-sim
# Before merging anything touching the kernel, the panda_mobile manifest,
# the HAL sim bridge, or a `scenes/deploy/robocasa_*.yaml` pin.
#
# A RoboCasa kitchen compose is heavy — the whole suite OOM-killed a 15 GB
# host running an editor+browser. Narrow with `-k` (extra args append to
# TARGETS, they don't select from it): `just test-robocasa-sim -k geom_distance`.
# NOT in TARGETS: anything needing HF weights or a GPU — geometry + kernel
# behaviour only.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TARGETS=(
    # Issue #102's third acceptance item, both halves. The shipped
    # `layout_ids: [47]` pin (#224) and the genuinely colliding pose at zero
    # margin (#232) — the only place the kernel is shown refusing a real
    # kitchen at its deployed standoff for a certified real reason.
    tests/sim/safety/test_kernel_fridge_layout_pin_start_state.py
    # The instrument the census and every adjudicated round depend on.
    # `mj_geomDistance` is wrong on exactly this pair class under mujoco 3.8.0
    # (+0.000 mm against a certified +0.148512 mm; -352 mm through a 48 mm
    # panel), so this pins `openral_hal.convex_distance` on the census's own
    # layout-9 state. If it regresses, every distance in the ledger is suspect.
    tests/sim/safety/test_geom_distance_instrument_robocasa.py
    # The support-contact witness against a real kitchen — the ADR-0092 D6
    # attestation whose failure direction is a MISSING exemption.
    tests/sim/safety/test_support_probe_instrument_robocasa.py
    # The depth synth's multi-ray path against a real kitchen, which is what
    # turns camera returns into the occupancy the kernel gates on.
    tests/sim/safety/test_depth_multiray_equivalence_robocasa.py
    # The HAL's side of the same kitchen: camera streams, BODY_TWIST on a
    # translating base, and the layout pins the scenes ship.
    tests/sim/test_panda_mobile_hal_robocasa_cameras.py
    tests/sim/test_panda_mobile_hal_robocasa_body_twist.py
    tests/sim/test_panda_mobile_hal_robocasa_layout_pin.py
)
# tests/unit/test_robocasa_sim_targets.py asserts every file gated on
# `importorskip("robocasa")` appears above — a gated test missing from this
# list runs on NO CI surface.

exec python -m pytest "${TARGETS[@]}" "$@"
