# The validation matrix

"Run the four sims and see how they perform" is one versioned command with a machine-readable verdict. It needs a GPU host with RoboCasa and the colcon overlay; it is not a CI lane.

| Question | Entry point |
| --- | --- |
| Run a round | `just validation-matrix --round-id <id> --expect-sha $(git rev-parse HEAD)` |
| Re-derive a round's verdicts offline | `just validation-matrix-verdicts <round-dir>` |
| What changed since another round | `just validation-matrix-diff <round-dir> <baseline-dir>` |
| Make a pre-harness round queryable | `just validation-matrix-import <round-dir> --round-id <id> --executed-sha <sha> --stem seed1` |

All four are [`tools/validation_matrix.py`](https://github.com/OpenRAL/openral/blob/master/tools/validation_matrix.py). Every round feeds the evidence ledger, [`collision-validation-evidence.md`](../reference/collision-validation-evidence.md), which also records which rule closed which lost round.

Exit codes: **0** clean; **2** usage; **3** a guardrail refused and nothing was written, not even the round directory; **4** the round ran but at least one scene bucketed `harness-error`.

## 1. What a round is

Four RoboCasa scenes, one seed, one policy, one stack:

| Scene key | Config | Task | Kitchen |
| --- | --- | --- | --- |
| `baguette` | `scenes/deploy/robocasa_baguette.yaml` | counter to cabinet | drawn |
| `sink_cup` | `scenes/deploy/robocasa_sink_cup.yaml` | counter to sink | drawn |
| `fridge` | `scenes/deploy/robocasa_fridge_drawer.yaml` | fridge shelf to fridge drawer | `layout_ids: [47]` |
| `utensil` | `scenes/deploy/robocasa_drawer_utensil.yaml` | counter to drawer | `layout_ids: [3]` |

**Why two scenes pin their kitchen.** `seed: 1` does not identify a kitchen: RoboCasa draws layout, style, fixtures and robot spawn from one shared `env.rng` on every reset, so any upstream change to the draw order reshuffles the scene at the same seed. For `fridge` most unpinned kitchens spawn a robot link inside the kernel's world model and the run E-stops before any action; for `utensil` the pin is for reproducibility only. Each scene file records the measurement behind its pin and is the source of truth; the per-layout census is [`robocasa-start-state-census.md`](../reference/robocasa-start-state-census.md). Do not restore a layout a scene file records as refuted. Every reset records the composed kitchen as `robocasa_scene_composition` in the run artifacts ([telemetry](../reference/telemetry.md)), and a pin the env cannot honour is a `ROSConfigError`.

**Control surfaces.** Tracked scenes are never modified; each round writes a resolved copy beside its artifacts. Seven stack knobs are CLI flags (explicit flag beats scene `runtime:` beats default):

```
--enable-slam --enable-nav2 --enable-octomap --enable-octomap-kernel-check
--no-object-detector --no-enable-scene-vlm --no-dashboard
--hal viewer_enabled=false
```

`enable_reasoner` has no flag. `openral deploy sim` reads it from the scene's `runtime:` block and defaults it to `true`, so the harness splices `runtime: {enable_reasoner: false}` (and the seed, when it differs) into the resolved copy, re-parses it, and diffs it against the tracked scene so a safety key cannot ride along. `verdicts.json` records both halves as `stack_argv` and `scene_pins`.

With the reasoner off nothing issues a goal, so [`tools/_validation_matrix_dispatch.py`](https://github.com/OpenRAL/openral/blob/master/tools/_validation_matrix_dispatch.py) sends one `ExecuteRskill` goal directly, and [`tools/_validation_matrix_monitor.py`](https://github.com/OpenRAL/openral/blob/master/tools/_validation_matrix_monitor.py) records the attachment, voxel and witness stream alongside it.

## 2. Guardrails

`run` refuses (exit 3, no partial round) rather than warning. Each rule closed a round that was lost to it.

| Guardrail | Refuses when | Why |
| --- | --- | --- |
| `assert_worktree_clean` | `git status --porcelain` is non-empty | A recorded SHA must describe code someone can check out. |
| `assert_sha` (`--expect-sha`) | `HEAD` is not the requested checkout | The round must validate the commit it claims. |
| `assert_overlay_fresh` | `install/` is older than any tracked `.cpp/.hpp/.h/.msg/.idl` under `cpp/` or `packages/` | A stale overlay validates the previous commit's C++. Rebuild: `rm -rf build install log && just ros2-build`. |
| `resolve_launcher` | this checkout's `.venv/bin/openral` is missing | The `~/.local/bin/openral` wrapper execs the baked checkout's venv, overlay and `robots/`. The harness calls the venv binary by absolute path and exports `OPENRAL_REPO_ROOT`. |
| `assert_sidecar_wire` | `pyzmq` is not importable | The XR-1 adapter is dead without it. Sync: `just sync --group robocasa --group sidecar-wire`. |
| `assert_no_safety_overrides` | an argv token matches a safety-knob pattern | The matrix observes the kernel, never moves it (CLAUDE.md §1.1). Tokens are lowercased with `-` folded to `_`. |
| `assert_scene_safety_unmoved` | the resolved scene moves a safety-relevant key of the tracked scene | Argv inspection cannot see a margin moved in the scene copy. |
| `gpu_status` | other compute processes hold the GPU (override: `--force-shared-gpu`) | The host is shared; announce a round against what is resident. |

| Refused (safety) | Pinnable (composition) |
| --- | --- |
| `safety:`, `hal:` (margins, tolerances), `extra_allowed_collision_pairs:`, `place_declaration:` (grants a contact exemption), `runtime.enable_octomap_kernel_check` | `runtime.enable_reasoner`, `enable_slam`, `enable_nav2`, `enable_octomap`, the detector, the scene VLM, `seed:` |

Any key whose name reads as a margin, tolerance, allowance, limit, watchdog, deadman or E-stop is safety-relevant by name. The comparison is tracked versus resolved: composition may differ, a safety key may not.

Each round's `metadata.json` records the executed SHA, worktree cleanliness, overlay build time, launcher path, `OPENRAL_REPO_ROOT`, robot manifest, sync groups, stack argv, and the DDS scope (`ros_domain_id`, `ros_automatic_discovery_range`; an empty string means the variable was unset, i.e. domain 0 with subnet-wide discovery). `deploy sim` confines itself to `LOCALHOST` plus a private domain, and `deploy sim` / `deploy run` refuse to start onto a graph that already has a `/joint_states` publisher.

## 3. The verdict contract

Each round writes `verdicts.json`, a [`ValidationRoundVerdicts`](../reference/schemas/ValidationRoundVerdicts.json) (in `openral_core.schemas`), and a human-readable `NOTES.md`. Per scene, one `outcome`:

| Outcome | Meaning |
| --- | --- |
| `completed` | `sim.task_success_final` reported success. The only success. |
| `estop-collision-real` | The kernel stopped the run and the simulator's distance probe found solid geometry at or below 0 m. The stop was correct. |
| `estop-collision-false-positive` | The nearest true geometry is further from the tripping party than the admissible kernel-versus-probe gap can explain. |
| `estop-collision-within-quantization` | The kernel was conservative by an amount that gap accounts for. Correct behaviour. |
| `estop-collision-unadjudicated` | The probe was truncated or absent, no budget was known, or the probe does not attest that both sides were collidability-filtered. Not a synonym for "fine"; `ground_truth.unadjudicated_reason` says why. |
| `estop-initial-configuration` | The stop landed before any action reached the HAL: a scene-config defect no margin change can clear. Outranks ground-truth adjudication. |
| `deadline-after-grasp` / `deadline-no-grasp` | No stop and no success. Read `deadline-no-grasp` as an instrument symptom first: it is defined by absence, which is what a silently dead graph also produces. Check the delivered chunk count before reading it as a policy property. |
| `harness-error` | No usable artifact set. Never read as a clean deadline. |

A scene is a `harness-error` (reason in `harness_error_reason` and `NOTES.md`; the round exits 4) when:

1. the runner wrote `<stem>_launch_failed.txt` (`/openral/execute_rskill` never appeared);
2. the deploy log has `[ERROR] [launch]: Caught exception in launch` (no marker file, no usage banner, and the log still looks like a run);
3. the deploy log's first line is a `Usage: openral ...` banner;
4. `<stem>_goal.log` has output but no JSON status line (the dispatcher raised; a real deadline still prints `{"status": -1, ...}`);
5. there is no deploy log;
6. Nav2's `lifecycle_manager_navigation` lost a bond heartbeat early in the run and tore the navigation stack down, leaving a healthy-looking, inert log (the test is *when*: the threshold sits in a measured 99 s gap and is declined on `_deploy_excerpt.log`, which begins mid-run; `BOND_TIMEOUT_S` in `nav2.launch.py` raises Nav2's 4 s default to 30 s); or
7. a lifecycle node never completed a transition (e.g. `transition 'configure' on '/openral_hal_panda_mobile' did not advance the FSM within 300.0s`).

### Adjudicating a collision stop

The kernel measures OBB-to-voxel; the ground-truth probe measures mesh-to-mesh. Their difference is only meaningful against the gap that difference of representation can produce. The sim HAL publishes it per run as `adjudication_budget.admissible_gap_m` (the collision model's corner slop plus the voxel half-diagonal; 88.2 mm on the panda at the 25 mm grid). A snapshot without a budget falls back to `quantization_budget_m`, the voxel term alone (21.7 mm), recorded as `budget_source`.

The fallback is asymmetric because the voxel term is a strict lower bound of the true gap:

| Discrepancy versus voxel term | Sound? | Verdict |
|---|---|---|
| at or below | yes | `within-quantization` |
| above | no, it proves nothing about the true gap | `unadjudicated` |

In order:

1. Any probed pair at or below 0 m gives `real-contact`. It is deliberately not keyed to the body the kernel named: the probe finding another link at 0.000 m means the configuration is unsafe.
2. Otherwise the tripping party's clearance is compared with the kernel's reported depth: beyond the gap is `false-positive`, within it `within-quantization` (beyond a `grid-quantization` gap is `unadjudicated`).
3. A truncated probe, a missing snapshot, an unknown budget or an unattested probe gives `unadjudicated`.

Rules the harness honours:

- **A 0 m pair is evidence only when both geoms are solid.** A geom with neither `contype` nor `conaffinity` cannot collide and the kernel never checks it. The HAL's probe filters to solid geoms and reports `noncollidable_{world,side,other}_geoms_excluded`; snapshots without that attestation are `unadjudicated`.
- **A zero MuJoCo contact count is not an emptiness test.** Contact exclusions can hide real interpenetration; adjudicate from the distance probes.
- **An untruncated probe that returns no pair** proves the nearest solid geometry is beyond `distmax_m`, a strict lower bound.
- **A very early stop can be unadjudicable, and so can a deaf monitor.** The grid resolution comes from the monitor's first `world_voxels` record. `monitor_records` counts what it received, and `NOTES.md` lists "Monitor received nothing" (a harness fault) apart from "Stopped before the monitor saw a voxel grid" (a fact about the run). The monitor is gated on the CLI's `dds_transport_ready:` line, recorded in `<stem>_monitor_gate.txt`.
- **Exemptions.** `ValidationStopEvidence.exemption_active` means `sweep_min_distance_m` is strictly deeper than `min_distance_m`: a support-contact witness exempted a deeper cell. A disagreement with the kernel's arm/separate transitions is itself a finding. `place_allowance_active` is transcribed from the trip line.

**Payload-versus-link self stop** (`a=attached:sim:obj_main b=panda_link1`, `kind=self`) is not a world stop and is not scored against world geometry. The rule branches on who the other side is:

| Other side | Pair set | Coverage block | Budget |
| --- | --- | --- | --- |
| `voxel_*` or `place:*` | `nearest_payload_world_pairs` | `nearest_payload_world_coverage` | `admissible_gap_m` |
| a robot link | `nearest_payload_robot_pairs` | `nearest_payload_robot_coverage` | `self_collision.admissible_gap_m` (corner slop plus the payload's; 124.6 mm on the panda carrying `obj_main`) |

A missing pair set is a missing instrument, so the stop is `unadjudicated`.

**Link-versus-link self stop** is never scored against world geometry, and the kernel states which budget applies:

| `depth_is_box_bound` | Measured | Budget | Outcome |
| --- | --- | --- | --- |
| `1` | the OBB bound | `2 x corner_slop` (`adjudication_budget.link_link.admissible_gap_box_m`) | adjudicable |
| `0` | the exact hulls | `hull_overhang_m(link_a) + hull_overhang_m(link_b)`, from `collision_model_slop.links` | adjudicable once both links have a measured overhang, else `unadjudicated` |

The hull term is per link, never maxed or doubled. A link with no hull, or a hull with no source mesh, ships no `hull_overhang_m` (never a silent `0`). The voxel fallback is deliberately unreachable here: neither side is a voxel.

## 4. Diffing rounds

```console
$ just validation-matrix-diff outputs/validation-matrix/<new> outputs/validation-matrix/<old>
2026-08-16-master-1 -> 2026-08-22-master-1  (reproducibility)
  CHANGED baguette   estop-collision-real -> estop-collision-false-positive
           stop.party_a: attached:sim:obj_main -> panda_link5
           stop.min_distance_m: -0.00418456 -> -0.0209178
```

Equal `executed_sha` and equal `seed` makes it a reproducibility comparison; anything else is before/after (a different seed is a different scene). The policy is stochastic even at a pinned seed, so compare failure classes and tripping pairs, never step counts.

## 5. Artifacts

Per round, under `outputs/validation-matrix/<round-id>/`:

```
metadata.json              what ran, on what, from what
verdicts.json              ValidationRoundVerdicts
NOTES.md                   the human-readable summary
<scene>/
  <scene>_seed<N>.yaml     the resolved scene copy that was launched
  run_deploy.log           the full deploy log
  run_monitor.jsonl        the monitor stream
  run_goal.log             the dispatcher's one JSON line
  run_monitor_gate.txt     whether the monitor started after the DDS clean
  run_launch_failed.txt    written only when the graph never came up
  run_snapshots/           .npz at every E-stop and while carrying
  run_cinecam/             frames
  run_kernel_evidence.txt, run_task_success.txt,
  run_allowance_active.txt, run_gt_snapshot.json   derived excerpts for reading
```

The verdict derives from `run_deploy.log`, `run_monitor.jsonl`, `run_goal.log` and `run_launch_failed.txt` only; the excerpts are for reviewers.

## 6. Importing pre-harness rounds

Older rounds use scene directories `bag1` / `sink1` / `fridge1` / `utensil1`, artifact stem `seed1` and no metadata. `validation-matrix-import` makes one queryable in place:

```console
$ just validation-matrix-import <round-dir> --round-id 2026-08-22-master-1 \
    --executed-sha <sha> --stem seed1 --host <host> --gpu-name "<gpu>" \
    --sync-group robocasa --sync-group sidecar-wire
```

It writes `metadata.json`, then derives `verdicts.json` and `NOTES.md`; `verdicts` and `diff` then treat the round like any other. Unrecognised layouts take `--scene-alias <scene>=<dir>`. It derives `stack_argv` (from the deploy log's resolved `argv: ... launch ...` echo; scenes that disagree are refused), `started_at`, `repo_root`, `robot_id`, `robot_manifest_path` and `scene_configs` from the artifacts. It stores the SHA, hostname, GPU and sync groups as given, and leaves `launcher_path` and `worktree_clean` empty rather than guess.

## 7. Running a round on a shared GPU host

```bash
cd <checkout>                       # clean and committed
just sync --group robocasa --group sidecar-wire
rm -rf build install log && just ros2-build     # when kernel, msgs or bridge changed
source /opt/ros/jazzy/setup.bash && source install/setup.bash
source .venv/bin/activate
nvidia-smi                          # who else is on the GPU?
just validation-matrix --round-id $(date +%F)-<name> --expect-sha $(git rev-parse HEAD)
```

- **Check `nvidia-smi` first.** Use `--force-shared-gpu` only for processes that are yours; never evict someone else's run.
- **Append `~/.local/bin` to `PATH`, never prepend it,** if `just` lives only there. The `~/.local/bin/openral` wrapper execs the baked checkout; the venv's `openral` must win. The harness uses an absolute launcher path, but ordering still decides what everything else you type runs.
- **Sync deliberately.** Every `just validation-matrix*` recipe uses `uv run --no-sync`, because a mid-round re-resolve can uninstall `pyzmq`.
- **One round at a time:** four scenes at about 7 minutes each, roughly half an hour of exclusive GPU.
- **Copy the round off the host** and re-derive verdicts anywhere with `just validation-matrix-verdicts`.

## 8. Testing the harness

`tests/unit/test_validation_matrix.py` runs verdict derivation, diff, import and every guardrail against recorded artifacts from three real rounds (`tests/unit/fixtures/validation_matrix/`, provenance in `SOURCE.txt`), pinned to what the evidence ledger concluded. Two rounds are the same code, scene and seed with different outcomes (a real round-over-round diff); one is the harness's first live round, four scenes of click usage error, the fixture for "a launch failure is not a deadline". The flags in `STACK_ARGV` are checked against the live `openral deploy sim` parser. The runner half needs a GPU host with RoboCasa and is exercised there, not in CI.
