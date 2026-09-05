#!/usr/bin/env bash
# The RoboCasa sim suite — the one list that defines what it is.
#
# `just test-robocasa-sim` execs this file. There is no second caller, because
# there is no CI lane: this suite runs on a developer host or it does not run.
# That is a measured conclusion, not an omission, and it is written down here
# so nobody re-derives it.
#
# WHY THERE IS NO CI LANE
# -----------------------
# These tests need three things at once: a colcon-built overlay (they spawn the
# real `safety_kernel_node`), MuJoCo, and a provisioned RoboCasa kitchen
# backend.
#
# The third rules out every GitHub-hosted runner. RoboCasa's assets are 23 GB
# on disk — `objects/aigen_objs` 13 GB, `objects/objaverse` 6.2 GB,
# `objects/lightwheel` 1.5 GB, `fixtures` 1.4 GB, `generative_textures` 1.2 GB,
# `textures` 521 MB — downloaded per-bundle from utexas.box.com with no
# sub-bundle granularity. The only hosted surface with a colcon overlay is the
# `docker-build` image, already 25.2 GB, building on a runner that has to
# `rm -rf` dotnet, android and CodeQL to reclaim its ~14 GB. Even a trimmed set
# (fixtures + textures + one object bundle, ~8 GB) is past that runner's whole
# disk. No GPU is needed, so the constraint is disk and nothing else.
#
# A self-hosted runner was built, registered and then REMOVED on security
# grounds. `OpenRAL/openral` is a PUBLIC repository, and a runner label is a
# routing request made by a workflow, not an access control enforced by the
# runner: a repo-scoped runner accepts jobs from ANY workflow in the repo that
# names its labels, and for a `pull_request` event GitHub executes the workflow
# definition from the FORK's ref. This repo has three fork-reachable
# `pull_request` workflows (`dco.yml`, `quality.yml`, `test-selective.yml`), so
# a fork PR can simply edit one to `runs-on: [self-hosted, <label>]` and run
# arbitrary code on the runner host — with that user's SSH keys, `gh`
# credentials and LAN access to the lab robots. Restricting a runner to
# selected workflows requires an organisation runner group, which this org's
# GitHub Free plan does not provide, so there is no native control that makes
# the label mean anything. The trigger config of the workflow being protected
# is irrelevant: the exposed asset is the runner, not the workflow.
#
# So the suite is manual, `tests/unit/test_robocasa_sim_targets.py` is what
# keeps this list honest, and the rest of `tests/sim/` is manual by declared
# policy anyway (`.github/workflows/test-selective.yml`).
#
#   source /opt/ros/jazzy/setup.bash && just ros2-build \
#     && source install/setup.bash && just test-robocasa-sim
#
# Run it before merging anything that touches the kernel, the panda_mobile
# manifest, the HAL sim bridge, or a `scenes/deploy/robocasa_*.yaml` pin.
#
# MEMORY: a RoboCasa kitchen compose is heavy. The whole suite in one pytest
# process OOM-killed a 15 GB host that was also running an editor and a
# browser. Narrow it with `-k` when the machine is busy — extra args are
# appended to the pytest invocation, so naming a path ADDS it to TARGETS
# rather than selecting it:
#   just test-robocasa-sim -k geom_distance
#
# NOT in TARGETS: anything needing HF weights or a GPU. This list is geometry
# and kernel behaviour against a real kitchen.

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
